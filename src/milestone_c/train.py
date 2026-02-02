"""Milestone C: Training script for CNN-LSTM emotion classifier."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Tuple, List
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import json

from src.core.emotion_classifier import CNNLSTMEmotionClassifier, EMOTION_LABELS


class EmotionDataset(Dataset):
    """Dataset for emotion classification."""
    
    def __init__(self, features: List[np.ndarray], labels: List[str], label_encoder: LabelEncoder):
        self.features = [torch.FloatTensor(f) for f in features]
        self.labels = label_encoder.transform(labels)
        self.label_encoder = label_encoder
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], torch.LongTensor([self.labels[idx]])[0]


def train_model(
    features: List[np.ndarray],
    labels: List[str],
    model_save_path: str,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    train_split: float = 0.8,
    device: str = "cpu"
) -> Tuple[CNNLSTMEmotionClassifier, dict]:
    """
    Train CNN-LSTM emotion classifier.
    
    Args:
        features: List of feature arrays
        labels: List of emotion labels
        model_save_path: Path to save trained model
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        train_split: Train/validation split ratio
        device: Device to train on ('cpu' or 'cuda')
        
    Returns:
        Trained model and training history
    """
    device = torch.device(device)
    
    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(EMOTION_LABELS)
    num_classes = len(EMOTION_LABELS)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=1-train_split, random_state=42, stratify=labels
    )
    
    # Create datasets
    train_dataset = EmotionDataset(X_train, y_train, label_encoder)
    val_dataset = EmotionDataset(X_val, y_val, label_encoder)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_dim = features[0].shape[1] if len(features[0].shape) > 1 else 13
    model = CNNLSTMEmotionClassifier(
        input_dim=input_dim,
        num_classes=num_classes
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
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
    print()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
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
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_features)
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
                'label_encoder': label_encoder,
                'num_classes': num_classes,
                'input_dim': input_dim,
                'epoch': epoch,
                'val_acc': val_acc,
                'history': history
            }, model_save_path)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
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
    """Example training script."""
    import argparse
    from src.milestone_b.audio_features import DatasetLoader
    
    parser = argparse.ArgumentParser(description="Train emotion classifier")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="./models", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--max_files", type=int, default=None, help="Max files to process")
    
    args = parser.parse_args()
    
    # Load dataset
    loader = DatasetLoader(args.data_dir)
    
    # Try to load RAVDESS or custom dataset
    try:
        df = loader.load_ravdess_dataset(args.data_dir)
    except:
        # Try custom dataset
        metadata_path = Path(args.data_dir) / "metadata.csv"
        if metadata_path.exists():
            df = loader.load_custom_dataset(str(metadata_path))
        else:
            raise ValueError(f"Could not load dataset from {args.data_dir}")
    
    # Preprocess
    features, labels = loader.preprocess_dataset(df, max_files=args.max_files)
    
    if len(features) == 0:
        raise ValueError("No features extracted. Check dataset path and format.")
    
    # Train model
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "emotion_classifier.pt"
    
    model, history = train_model(
        features=features,
        labels=labels,
        model_save_path=str(model_path),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device
    )
    
    # Save training history
    history_path = output_path / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining history saved to {history_path}")


if __name__ == "__main__":
    main()

