"""Emotion classification module using CNN-LSTM architecture."""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Emotion labels (can be customized)
EMOTION_LABELS = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgusted",
    "surprised",
    "calm",
    "anxious",
    "stressed"
]


class CNNLSTMEmotionClassifier(nn.Module):
    """CNN-LSTM model for emotion classification from audio features."""
    
    def __init__(
        self,
        input_dim: int = 13,  # MFCC features
        num_classes: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super(CNNLSTMEmotionClassifier, self).__init__()
        
        self.num_classes = num_classes
        
        # CNN layers for feature extraction (with BatchNorm - Stage 3 enhancement)
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        self.dropout_conv = nn.Dropout(dropout)
        
        # LSTM layers for temporal modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, 64)  # *2 for bidirectional
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout_fc = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Emotion probabilities
        """
        # Reshape for CNN: (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        # CNN layers with BatchNorm
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        # Reshape for LSTM: (batch, seq_len, features)
        x = x.transpose(1, 2)
        
        # LSTM layers
        lstm_out, _ = self.lstm(x)
        
        # Use last output
        x = lstm_out[:, -1, :]
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return self.softmax(x)


class TextEmotionClassifier:
    """Simple rule-based emotion classifier for text (Milestone A)."""
    
    def __init__(self):
        self.emotion_keywords = {
            "happy": ["happy", "joy", "excited", "great", "wonderful", "amazing", "love"],
            "sad": ["sad", "depressed", "down", "unhappy", "miserable", "lonely"],
            "angry": ["angry", "mad", "furious", "annoyed", "irritated", "frustrated"],
            "anxious": ["anxious", "worried", "nervous", "stressed", "panic", "afraid"],
            "calm": ["calm", "peaceful", "relaxed", "serene", "tranquil", "zen"],
            "neutral": []
        }
    
    def predict(self, text: str, preprocessed: Optional[Dict] = None) -> Dict[str, float]:
        """
        Predict emotion probabilities from text.
        
        Args:
            text: Input text
            preprocessed: Optional preprocessed text dict
            
        Returns:
            Dictionary mapping emotion labels to probabilities
        """
        text_lower = text.lower()
        scores = {}
        
        # Count keyword matches
        for emotion, keywords in self.emotion_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            scores[emotion] = count
        
        # Normalize to probabilities
        total = sum(scores.values())
        if total == 0:
            # Default to neutral
            return {label: 1.0 if label == "neutral" else 0.0 for label in EMOTION_LABELS}
        
        # Convert to probabilities
        probs = {label: scores.get(label, 0) / total for label in EMOTION_LABELS}
        
        # Add some smoothing
        smoothing = 0.1
        for label in probs:
            probs[label] = (1 - smoothing) * probs[label] + smoothing / len(EMOTION_LABELS)
        
        return probs


class EmotionClassifier:
    """Main emotion classifier interface."""
    
    def __init__(self, model_path: Optional[str] = None, use_text_fallback: bool = True):
        self.model_path = model_path
        self.use_text_fallback = use_text_fallback
        self.model = None
        self.text_classifier = TextEmotionClassifier() if use_text_fallback else None
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load trained model from file."""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model = CNNLSTMEmotionClassifier()
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
            else:
                self.model = checkpoint
                if isinstance(self.model, nn.Module):
                    self.model.eval()
        except Exception as e:
            print(f"Warning: Could not load model from {model_path}: {e}")
            self.model = None
    
    def predict_from_text(self, text: str, preprocessed: Optional[Dict] = None) -> Dict[str, float]:
        """
        Predict emotion from text.
        
        Args:
            text: Input text
            preprocessed: Optional preprocessed text dict
            
        Returns:
            Emotion probabilities
        """
        if self.text_classifier:
            return self.text_classifier.predict(text, preprocessed)
        else:
            # Default neutral
            return {label: 1.0 if label == "neutral" else 0.0 for label in EMOTION_LABELS}
    
    def predict_from_audio(self, audio_features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Predict emotion from audio features.
        
        Args:
            audio_features: Dictionary with 'mfcc' and other features
            
        Returns:
            Emotion probabilities
        """
        if self.model is None:
            # Fallback to text if available, else neutral
            return {label: 1.0 if label == "neutral" else 0.0 for label in EMOTION_LABELS}
        
        # Prepare input tensor
        mfcc = audio_features.get('mfcc', np.zeros(13))
        if len(mfcc.shape) == 1:
            # Add sequence dimension: (1, seq_len, features)
            mfcc = mfcc.reshape(1, -1, 1)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(mfcc)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = outputs[0].numpy()
        
        # Map to emotion labels
        return {EMOTION_LABELS[i]: float(probs[i]) for i in range(len(EMOTION_LABELS))}
    
    def get_top_emotion(self, emotion_probs: Dict[str, float]) -> Tuple[str, float]:
        """Get the emotion with highest probability."""
        top_emotion = max(emotion_probs.items(), key=lambda x: x[1])
        return top_emotion

