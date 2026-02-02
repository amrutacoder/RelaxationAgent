"""Multimodal Fusion - Combines text and acoustic embeddings (Stage 4)."""

import torch
import torch.nn as nn
from typing import Literal, Optional, Tuple

FusionMethod = Literal["concatenation", "weighted_sum", "attention"]


class MultimodalFusion:
    """
    Fuses text and acoustic emotion embeddings.
    
    Stage 4 of architecture: Simple concatenation-based fusion
    (can be extended with attention-based fusion).
    """
    
    def __init__(self, method: FusionMethod = "concatenation"):
        """
        Initialize multimodal fusion.
        
        Args:
            method: Fusion method ('concatenation', 'weighted_sum', 'attention')
        """
        self.method = method
        self.text_dim = None
        self.acoustic_dim = None
    
    def fuse(
        self, 
        text_embedding: torch.Tensor, 
        acoustic_embedding: torch.Tensor,
        text_weight: float = 0.5,
        acoustic_weight: float = 0.5
    ) -> torch.Tensor:
        """
        Fuse text and acoustic embeddings.
        
        Args:
            text_embedding: Text emotion embedding (from DistilBERT)
            acoustic_embedding: Acoustic emotion embedding (from CNN+BiLSTM)
            text_weight: Weight for text embedding (for weighted_sum method)
            acoustic_weight: Weight for acoustic embedding (for weighted_sum method)
            
        Returns:
            Fused embedding tensor
        """
        # Store dimensions if not set
        if self.text_dim is None:
            self.text_dim = text_embedding.shape[-1]
        if self.acoustic_dim is None:
            self.acoustic_dim = acoustic_embedding.shape[-1]
        
        # Ensure same batch dimension
        if len(text_embedding.shape) == 1:
            text_embedding = text_embedding.unsqueeze(0)
        if len(acoustic_embedding.shape) == 1:
            acoustic_embedding = acoustic_embedding.unsqueeze(0)
        
        if self.method == "concatenation":
            # Simple concatenation: [text_emb || acoustic_emb]
            fused = torch.cat([text_embedding, acoustic_embedding], dim=-1)
        
        elif self.method == "weighted_sum":
            # Weighted sum (requires same dimension)
            if self.text_dim != self.acoustic_dim:
                raise ValueError(
                    f"Text and acoustic embeddings must have same dimension "
                    f"for weighted_sum fusion. Got {self.text_dim} and {self.acoustic_dim}"
                )
            fused = text_weight * text_embedding + acoustic_weight * acoustic_embedding
        
        elif self.method == "attention":
            # Attention-based fusion (requires initialization)
            if not hasattr(self, 'attention_weights'):
                raise ValueError("Attention fusion requires initialization. Use AttentionFusion class.")
            fused = self._attention_fuse(text_embedding, acoustic_embedding)
        
        else:
            raise ValueError(f"Unknown fusion method: {self.method}")
        
        # Remove batch dimension if input was single sample
        if fused.shape[0] == 1 and len(fused.shape) > 1:
            fused = fused.squeeze(0)
        
        return fused
    
    def fuse_batch(
        self, 
        text_embeddings: torch.Tensor,
        acoustic_embeddings: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Fuse batch of embeddings.
        
        Args:
            text_embeddings: Batch of text embeddings (batch, text_dim)
            acoustic_embeddings: Batch of acoustic embeddings (batch, acoustic_dim)
            **kwargs: Additional arguments for fuse()
            
        Returns:
            Batch of fused embeddings (batch, fused_dim)
        """
        return self.fuse(text_embeddings, acoustic_embeddings, **kwargs)
    
    def get_fused_dim(self) -> Optional[int]:
        """Get dimension of fused embedding."""
        if self.text_dim is None or self.acoustic_dim is None:
            return None
        
        if self.method == "concatenation":
            return self.text_dim + self.acoustic_dim
        elif self.method == "weighted_sum":
            return self.text_dim  # Same as text/acoustic dim
        else:
            return None


class AttentionFusion(nn.Module):
    """
    Attention-based multimodal fusion (advanced option).
    
    Uses learned attention weights to combine text and acoustic embeddings.
    """
    
    def __init__(self, text_dim: int, acoustic_dim: int, fused_dim: int):
        """
        Initialize attention fusion.
        
        Args:
            text_dim: Dimension of text embeddings
            acoustic_dim: Dimension of acoustic embeddings
            fused_dim: Dimension of output fused embedding
        """
        super().__init__()
        
        self.text_dim = text_dim
        self.acoustic_dim = acoustic_dim
        self.fused_dim = fused_dim
        
        # Project embeddings to same dimension for attention
        self.text_proj = nn.Linear(text_dim, fused_dim)
        self.acoustic_proj = nn.Linear(acoustic_dim, fused_dim)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=fused_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Final projection
        self.final_proj = nn.Linear(fused_dim, fused_dim)
    
    def forward(
        self, 
        text_embedding: torch.Tensor, 
        acoustic_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with attention fusion.
        
        Args:
            text_embedding: Text embedding (batch, text_dim) or (text_dim,)
            acoustic_embedding: Acoustic embedding (batch, acoustic_dim) or (acoustic_dim,)
            
        Returns:
            Fused embedding (batch, fused_dim) or (fused_dim,)
        """
        # Ensure batch dimension
        single_sample = len(text_embedding.shape) == 1
        if single_sample:
            text_embedding = text_embedding.unsqueeze(0)
            acoustic_embedding = acoustic_embedding.unsqueeze(0)
        
        # Project to same dimension
        text_proj = self.text_proj(text_embedding).unsqueeze(1)  # (batch, 1, fused_dim)
        acoustic_proj = self.acoustic_proj(acoustic_embedding).unsqueeze(1)  # (batch, 1, fused_dim)
        
        # Stack for attention: (batch, 2, fused_dim)
        combined = torch.cat([text_proj, acoustic_proj], dim=1)
        
        # Self-attention
        attn_out, _ = self.attention(combined, combined, combined)
        
        # Pool (mean of two modalities)
        fused = attn_out.mean(dim=1)  # (batch, fused_dim)
        
        # Final projection
        fused = self.final_proj(fused)
        
        # Remove batch dimension if single sample
        if single_sample:
            fused = fused.squeeze(0)
        
        return fused


def create_fusion(
    method: FusionMethod = "concatenation",
    text_dim: Optional[int] = None,
    acoustic_dim: Optional[int] = None,
    fused_dim: Optional[int] = None
):
    """
    Factory function to create fusion component.
    
    Args:
        method: Fusion method
        text_dim: Text embedding dimension (required for attention)
        acoustic_dim: Acoustic embedding dimension (required for attention)
        fused_dim: Fused embedding dimension (required for attention)
        
    Returns:
        Fusion component
    """
    if method == "attention":
        if text_dim is None or acoustic_dim is None or fused_dim is None:
            raise ValueError("text_dim, acoustic_dim, and fused_dim required for attention fusion")
        return AttentionFusion(text_dim, acoustic_dim, fused_dim)
    else:
        return MultimodalFusion(method=method)

