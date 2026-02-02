"""Milestone C: Training script for text-based emotion classifier using GoEmotions."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.emotion_classifier import EMOTION_LABELS
from src.core.text_encoder import TextEmotionEncoder, create_text_encoder
from src.milestone_b.goemotions_loader import GoEmotionsLoader
from src.milestone_b.audio_features import DatasetLoader


class TextEmotionDataset(Dataset):
    """Dataset for text-based emotion classification."""
    
    def __init__(self, texts: List[str], labels: List[str], text_encoder, label_encoder: LabelEncoder):
        self.texts = texts
        self.labels = label_encoder.transform(labels)
        self.text_encoder = text_encoder
        self.label_encoder = label_encoder
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Encode text
        text_embedding = self.text_encoder.encode(self.texts[idx])
        
        # Convert to tensor if needed
        if isinstance(text_embedding, np.ndarray):
            text_embedding = torch.FloatTensor(text_embedding)
        
        return text_embedding, torch.LongTensor([self.labels[idx]])[0]


def train_text_classifier(
    texts: List[str],
    labels: List[str],
    model_save_path: str,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    train_split: float = 0.8,
    device: str = "cpu",
    use_pretrained: bool = True
) -> Tuple[object, dict]:
    """
    Train text-based emotion classifier using DistilBERT.
    
    Args:
        texts: List of input texts
        labels: List of emotion labels
        model_save_path: Path to save trained model
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        train_split: Train/validation split ratio
        device: Device to train on ('cpu' or 'cuda')
        use_pretrained: Use pretrained DistilBERT encoder
        
    Returns:
        Trained model and training history
    """
    device = torch.device(device)
    
    # Initialize text encoder
    if use_pretrained:
        try:
            text_encoder = create_text_encoder()
            print("Using pretrained DistilBERT encoder")
        except Exception as e:
            print(f"Warning: Could not load pretrained encoder: {e}")
            print("Falling back to basic text encoder")
            text_encoder = TextEmotionEncoder()
    else:
        text_encoder = TextEmotionEncoder()
    
    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(EMOTION_LABELS)
    num_classes = len(EMOTION_LABELS)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=1-train_split, random_state=42, stratify=labels
    )
    
    # Create datasets
    train_dataset = TextEmotionDataset(X_train, y_train, text_encoder, label_encoder)
    val_dataset = TextEmotionDataset(X_val, y_val, text_encoder, label_encoder)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Get embedding dimension from encoder
    sample_embedding = text_encoder.encode(X_train[0])
    if isinstance(sample_embedding, np.ndarray):
        embedding_dim = sample_embedding.shape[0] if len(sample_embedding.shape) == 1 else sample_embedding.shape[-1]
    else:
        embedding_dim = 768  # Default DistilBERT dimension
    
    # Simple classifier head
    class TextEmotionClassifier(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 256)
            self.dropout1 = nn.Dropout(0.3)
            self.fc2 = nn.Linear(256, 128)
            self.dropout2 = nn.Dropout(0.3)
            self.fc3 = nn.Linear(128, num_classes)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout1(x)
            x = self.relu(self.fc2(x))
            x = self.dropout2(x)
            x = self.fc3(x)
            return x
    
    model = TextEmotionClassifier(embedding_dim, num_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    print(f"Training on {device}")
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    print(f"Number of classes: {num_classes}")
    print(f"Embedding dimension: {embedding_dim}")
    print()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_embeddings, batch_labels in train_loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_embeddings)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_embeddings, batch_labels in val_loader:
                batch_embeddings = batch_embeddings.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_embeddings)
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'text_encoder': text_encoder,
                'label_encoder': label_encoder,
                'num_classes': num_classes,
                'embedding_dim': embedding_dim,
                'epoch': epoch,
                'val_acc': val_acc,
                'history': history
            }, model_save_path)
        
        # Print progress
        if (epoch + 1) % 1 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print()
    
    print(f"Training complete! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to {model_save_path}")
    
    # Load best model
    checkpoint = torch.load(model_save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, history


def main():
    """Example training script for text-based emotion classification."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train text emotion classifier on GoEmotions")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/goemotions",
        help="Path to GoEmotions dataset directory"
    )
    parser.add_argument(
        "--from_huggingface",
        action="store_true",
        help="Load directly from Hugging Face (requires internet)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split to use"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models",
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (cpu/cuda)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to use (for testing)"
    )
    parser.add_argument(
        "--simplified",
        action="store_true",
        default=True,
        help="Use simplified GoEmotions version"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Text Emotion Classifier Training (GoEmotions)")
    print("=" * 60)
    print()
    
    # Load dataset
    loader = DatasetLoader(args.data_dir)
    
    if args.from_huggingface:
        print("Loading GoEmotions from Hugging Face...")
        df = loader.load_goemotions_dataset(
            from_huggingface=True,
            split=args.split,
            simplified=args.simplified,
            max_samples=args.max_samples
        )
    else:
        # Load from local files
        data_path = Path(args.data_dir)
        if (data_path / f"goemotions_{args.split}.csv").exists():
            print(f"Loading GoEmotions from local file...")
            df = loader.load_goemotions_dataset(
                dataset_path=str(data_path / f"goemotions_{args.split}.csv"),
                from_huggingface=False,
                max_samples=args.max_samples
            )
        else:
            raise FileNotFoundError(
                f"GoEmotions file not found. Run download script first:\n"
                f"  python scripts/download_goemotions.py --output_dir {args.data_dir}"
            )
    
    # Preprocess text dataset
    print("\nPreprocessing text data...")
    preprocessed_texts, labels = loader.preprocess_text_dataset(df, max_samples=args.max_samples)
    
    # Extract just the text strings from preprocessed dicts
    texts = [item.get('original_text', '') if isinstance(item, dict) else str(item) 
             for item in preprocessed_texts]
    
    if len(texts) == 0:
        raise ValueError("No text samples processed. Check dataset format.")
    
    print(f"Loaded {len(texts)} text samples")
    print(f"Emotion distribution: {pd.Series(labels).value_counts().to_dict()}")
    print()
    
    # Train model
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "text_emotion_classifier.pt"
    
    model, history = train_text_classifier(
        texts=texts,
        labels=labels,
        model_save_path=str(model_path),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device
    )
    
    # Save training history
    history_path = output_path / "text_training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining history saved to {history_path}")
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
