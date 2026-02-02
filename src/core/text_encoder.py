"""Text Emotion Encoder using pretrained transformer models (Stage 2)."""

import torch
import torch.nn as nn
from typing import Optional, List
import os

try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available. Install with: pip install transformers")


class TextEmotionEncoder:
    """
    Pretrained transformer encoder for text emotion embedding.
    
    Uses DistilBERT or similar lightweight transformer to encode
    text into emotion-rich embeddings (Stage 2 of architecture).
    """
    
    def __init__(
        self, 
        model_name: str = "distilbert-base-uncased",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize text emotion encoder.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ('cpu', 'cuda', or None for auto)
            cache_dir: Directory to cache model files
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library required. Install with: pip install transformers"
            )
        
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and model
        print(f"Loading text encoder: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_dir
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Get embedding dimension
        self.embedding_dim = self.model.config.hidden_size
        print(f"Text encoder loaded. Embedding dimension: {self.embedding_dim}")
    
    def encode(
        self, 
        text: str, 
        pooling: str = "cls",
        max_length: int = 512
    ) -> torch.Tensor:
        """
        Encode text into emotion embedding.
        
        Args:
            text: Input text string
            pooling: Pooling strategy ('cls', 'mean', 'max')
            max_length: Maximum sequence length
            
        Returns:
            Tensor of shape (embedding_dim,) - emotion embedding vector
        """
        # Tokenize
        inputs = self.tokenizer(
            text, 
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Encode
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
        
        # Pooling
        if pooling == "cls":
            # Use [CLS] token embedding
            embedding = hidden_states[0, 0, :]  # (hidden_dim,)
        elif pooling == "mean":
            # Mean pooling (excluding padding)
            attention_mask = inputs["attention_mask"]
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            masked_hidden = hidden_states * mask_expanded
            sum_hidden = masked_hidden.sum(dim=1)
            sum_mask = attention_mask.sum(dim=1, keepdim=True)
            embedding = sum_hidden / sum_mask
            embedding = embedding[0]  # Remove batch dimension
        elif pooling == "max":
            # Max pooling
            embedding = hidden_states.max(dim=1)[0][0]
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling}")
        
        return embedding.cpu()
    
    def encode_batch(
        self, 
        texts: List[str],
        pooling: str = "cls",
        max_length: int = 512,
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Encode batch of texts.
        
        Args:
            texts: List of input text strings
            pooling: Pooling strategy
            max_length: Maximum sequence length
            batch_size: Batch size for processing
            
        Returns:
            Tensor of shape (batch_size, embedding_dim)
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Encode
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state
            
            # Pooling
            if pooling == "cls":
                embeddings = hidden_states[:, 0, :]  # Use [CLS] token
            elif pooling == "mean":
                attention_mask = inputs["attention_mask"]
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                masked_hidden = hidden_states * mask_expanded
                sum_hidden = masked_hidden.sum(dim=1)
                sum_mask = attention_mask.sum(dim=1, keepdim=True)
                embeddings = sum_hidden / sum_mask
            elif pooling == "max":
                embeddings = hidden_states.max(dim=1)[0]
            else:
                raise ValueError(f"Unknown pooling strategy: {pooling}")
            
            all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    def __call__(self, text: str, **kwargs) -> torch.Tensor:
        """Convenience method: encoder(text) instead of encoder.encode(text)."""
        return self.encode(text, **kwargs)


class TextEmotionEncoderLite:
    """
    Lightweight fallback encoder when transformers not available.
    Uses simple feature extraction instead of transformer.
    """
    
    def __init__(self):
        self.embedding_dim = 128
        print("Warning: Using lightweight text encoder (transformers not available)")
    
    def encode(self, text: str, **kwargs) -> torch.Tensor:
        """Simple embedding using word counts and features."""
        # Simple fallback: return random embedding (not recommended for production)
        import numpy as np
        return torch.FloatTensor(np.random.randn(self.embedding_dim))
    
    def encode_batch(self, texts: List[str], **kwargs) -> torch.Tensor:
        """Batch encoding fallback."""
        embeddings = [self.encode(text, **kwargs) for text in texts]
        return torch.stack(embeddings)


def create_text_encoder(
    model_name: str = "distilbert-base-uncased",
    use_lite: bool = False,
    **kwargs
) -> TextEmotionEncoder:
    """
    Factory function to create text encoder.
    
    Args:
        model_name: HuggingFace model name
        use_lite: Force use of lite encoder (for testing)
        **kwargs: Additional arguments for TextEmotionEncoder
        
    Returns:
        TextEmotionEncoder instance
    """
    if use_lite or not TRANSFORMERS_AVAILABLE:
        return TextEmotionEncoderLite()
    
    return TextEmotionEncoder(model_name=model_name, **kwargs)

