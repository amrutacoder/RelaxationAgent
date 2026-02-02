"""Acoustic Emotion Encoder - CNN + BiLSTM for audio features (Stage 3)."""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class AcousticEmotionEncoder(nn.Module):
    """
    CNN + BiLSTM acoustic emotion encoder.
    
    Stage 3 of architecture:
    - 1D CNN (local spectral patterns)
    - BatchNorm + ReLU
    - MaxPooling
    - BiLSTM (temporal emotion dynamics)
    - Returns acoustic emotion embedding
    """
    
    def __init__(
        self,
        input_dim: int = 40,  # MFCC + other features
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        output_embedding: bool = True
    ):
        """
        Initialize acoustic emotion encoder.
        
        Args:
            input_dim: Input feature dimension (e.g., MFCC)
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            output_embedding: If True, return embedding; if False, return raw LSTM output
        """
        super(AcousticEmotionEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_embedding = output_embedding
        
        # 1D CNN layers (local spectral patterns)
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.dropout_conv = nn.Dropout(dropout)
        
        # Bidirectional LSTM (temporal emotion dynamics)
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Output dimension: hidden_dim * 2 (bidirectional)
        self.output_dim = hidden_dim * 2 if output_embedding else hidden_dim * 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            Acoustic emotion embedding of shape (batch, output_dim) or (output_dim,)
        """
        # Reshape for CNN: (batch, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # CNN layers with BatchNorm
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout_conv(x)
        
        # Reshape for LSTM: (batch, seq_len, features)
        x = x.transpose(1, 2)
        
        # Bidirectional LSTM
        lstm_out, _ = self.bilstm(x)  # (batch, seq_len, hidden_dim * 2)
        
        # Use last timestep as embedding
        embedding = lstm_out[:, -1, :]  # (batch, hidden_dim * 2)
        
        return embedding
    
    def encode(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Encode audio features into emotion embedding.
        
        Args:
            audio_features: Audio feature tensor
            
        Returns:
            Emotion embedding
        """
        return self.forward(audio_features)


def create_acoustic_encoder(
    input_dim: int = 40,
    hidden_dim: int = 128,
    **kwargs
) -> AcousticEmotionEncoder:
    """
    Factory function to create acoustic encoder.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: LSTM hidden dimension
        **kwargs: Additional arguments
        
    Returns:
        AcousticEmotionEncoder instance
    """
    return AcousticEmotionEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        **kwargs
    )

