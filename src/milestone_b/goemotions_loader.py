"""GoEmotions dataset loader for text-based emotion classification."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# GoEmotions has 27 emotion categories + neutral
# We need to map these to our 10 emotion labels
GOEMOTIONS_TO_PROJECT_EMOTIONS = {
    # Positive emotions -> happy
    "admiration": "happy",
    "amusement": "happy",
    "approval": "happy",
    "caring": "happy",
    "curiosity": "happy",
    "desire": "happy",
    "excitement": "happy",
    "gratitude": "happy",
    "joy": "happy",
    "love": "happy",
    "optimism": "happy",
    "pride": "happy",
    "relief": "happy",
    
    # Negative emotions
    "anger": "angry",
    "annoyance": "angry",
    "disappointment": "sad",
    "disapproval": "angry",
    "disgust": "disgusted",
    "embarrassment": "sad",
    "fear": "fearful",
    "grief": "sad",
    "nervousness": "anxious",
    "remorse": "sad",
    "sadness": "sad",
    
    # Neutral/calm
    "neutral": "neutral",
    "surprise": "surprised",
    
    # Additional mappings
    "confusion": "neutral",
    "realization": "surprised",
}

# GoEmotions emotion labels (27 categories)
GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]


class GoEmotionsLoader:
    """Loader for GoEmotions dataset from Hugging Face or local files."""
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize GoEmotions loader.
        
        Args:
            data_dir: Optional directory containing GoEmotions data files
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self.emotion_mapping = GOEMOTIONS_TO_PROJECT_EMOTIONS
    
    def load_from_huggingface(
        self,
        split: str = "train",
        simplified: bool = True,
        max_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load GoEmotions dataset from Hugging Face.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            simplified: Use simplified version (54k vs 211k rows)
            max_samples: Optional limit on number of samples
            
        Returns:
            DataFrame with text and emotion columns
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "Hugging Face datasets library required. Install with: pip install datasets"
            )
        
        # Load dataset
        dataset_name = "google-research-datasets/go_emotions"
        config = "simplified" if simplified else "raw"
        
        print(f"Loading GoEmotions dataset ({config}, {split})...")
        dataset = load_dataset(dataset_name, config, split=split)
        
        # Convert to pandas
        df = dataset.to_pandas()
        
        # Limit samples if specified
        if max_samples:
            df = df.head(max_samples)
        
        # Process labels (GoEmotions is multi-label, we'll take the first/primary label)
        df['emotion'] = df['labels'].apply(self._map_goemotions_labels)
        
        # Filter out rows where we couldn't map to our emotion set
        df = df[df['emotion'].notna()]
        
        # Select relevant columns
        result_df = pd.DataFrame({
            'text': df['text'],
            'emotion': df['emotion'],
            'original_labels': df['labels']
        })
        
        print(f"Loaded {len(result_df)} samples")
        print(f"Emotion distribution:\n{result_df['emotion'].value_counts()}")
        
        return result_df
    
    def load_from_local(
        self,
        file_path: str,
        max_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load GoEmotions dataset from local CSV/TSV file.
        
        Expected format: text, labels (comma-separated emotion labels)
        
        Args:
            file_path: Path to local data file
            max_samples: Optional limit on number of samples
            
        Returns:
            DataFrame with text and emotion columns
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"GoEmotions file not found: {file_path}")
        
        # Try to detect format
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix == '.tsv':
            df = pd.read_csv(file_path, sep='\t')
        elif file_path.suffix == '.json':
            df = pd.read_json(file_path)
        else:
            # Try CSV first
            try:
                df = pd.read_csv(file_path)
            except:
                df = pd.read_csv(file_path, sep='\t')
        
        # Process labels
        if 'labels' in df.columns:
            df['emotion'] = df['labels'].apply(self._map_goemotions_labels)
        elif 'emotion' in df.columns:
            # Already mapped
            pass
        else:
            raise ValueError("Dataset must have 'labels' or 'emotion' column")
        
        # Filter out unmapped emotions
        df = df[df['emotion'].notna()]
        
        # Limit samples
        if max_samples:
            df = df.head(max_samples)
        
        # Select relevant columns
        result_df = pd.DataFrame({
            'text': df['text'],
            'emotion': df['emotion']
        })
        
        print(f"Loaded {len(result_df)} samples from {file_path}")
        print(f"Emotion distribution:\n{result_df['emotion'].value_counts()}")
        
        return result_df
    
    def _map_goemotions_labels(self, labels) -> Optional[str]:
        """
        Map GoEmotions labels to project emotion labels.

        Args:
            labels: List of GoEmotions label indices or label names

        Returns:
            Mapped emotion label or None
        """

        # ---- 1. Handle None explicitly
        if labels is None:
            return None

        # ---- 2. Handle empty list / array
        if isinstance(labels, (list, tuple, np.ndarray)) and len(labels) == 0:
            return None

        # ---- 3. Handle scalar NaN (ONLY for non-list types)
        if not isinstance(labels, (list, tuple, np.ndarray)):
            if pd.isna(labels):
                return None

        # ---- 4. Convert label indices → label names if needed
        if isinstance(labels, (list, tuple, np.ndarray)):
            # GoEmotions HF gives label *indices*
            labels = [GOEMOTIONS_LABELS[int(l)] for l in labels]
        elif isinstance(labels, str):
            try:
                labels = json.loads(labels)
            except:
                labels = [l.strip() for l in labels.split(',')]
        else:
            labels = [str(labels)]

        # ---- 5. Map to project emotion (pick first valid)
        for label in labels:
            label_lower = label.lower().strip()

            if label_lower in self.emotion_mapping:
                return self.emotion_mapping[label_lower]

        # ---- 6. Fallback
        return "neutral"
    
    def download_and_save(
        self,
        output_dir: str,
        splits: List[str] = ["train", "validation", "test"],
        simplified: bool = True,
        max_samples_per_split: Optional[int] = None
    ):
        """
        Download GoEmotions dataset and save to local files.
        
        Args:
            output_dir: Directory to save downloaded data
            splits: Which splits to download
            simplified: Use simplified version
            max_samples_per_split: Optional limit per split
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for split in splits:
            print(f"\nDownloading {split} split...")
            df = self.load_from_huggingface(
                split=split,
                simplified=simplified,
                max_samples=max_samples_per_split
            )
            
            # Save to CSV
            output_file = output_path / f"goemotions_{split}.csv"
            df.to_csv(output_file, index=False)
            print(f"Saved {split} split to {output_file}")
        
        print(f"\nDownload complete! Files saved to {output_dir}")
    
    def preprocess_for_text_training(
        self,
        df: pd.DataFrame,
        output_dir: Optional[str] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Preprocess GoEmotions dataset for text-based training.
        
        Args:
            df: DataFrame with text and emotion columns
            output_dir: Optional directory to save preprocessed data
            
        Returns:
            Tuple of (texts_list, labels_list)
        """
        texts = df['text'].tolist()
        labels = df['emotion'].tolist()
        
        # Filter out any None labels
        valid_indices = [i for i, label in enumerate(labels) if label is not None]
        texts = [texts[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]
        
        print(f"Preprocessed {len(texts)} text samples")
        
        # Save if output_dir provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save as CSV
            preprocessed_df = pd.DataFrame({'text': texts, 'emotion': labels})
            preprocessed_df.to_csv(output_path / "goemotions_preprocessed.csv", index=False)
            
            # Save as JSON for easy loading
            with open(output_path / "goemotions_preprocessed.json", 'w') as f:
                json.dump({'texts': texts, 'labels': labels}, f, indent=2)
            
            print(f"Saved preprocessed data to {output_dir}")
        
        return texts, labels
