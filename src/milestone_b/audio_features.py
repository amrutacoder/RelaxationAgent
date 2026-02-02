"""Milestone B: Audio feature extraction and dataset handling."""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from src.core.preprocessor import AudioPreprocessor, TextPreprocessor


class AudioFeatureExtractor:
    """Extracts audio features for emotion classification."""
    
    def __init__(self, sample_rate: int = 22050, n_mfcc: int = 13):
        self.preprocessor = AudioPreprocessor(sample_rate=sample_rate, n_mfcc=n_mfcc)
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
    
    def extract_from_file(self, audio_path: str) -> Dict[str, np.ndarray]:
        """Extract features from audio file."""
        return self.preprocessor.extract_features(audio_path)
    
    def extract_from_array(self, audio_array: np.ndarray, 
                          sample_rate: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Extract features from audio array."""
        return self.preprocessor.extract_features_from_array(
            audio_array, 
            sample_rate or self.sample_rate
        )
    
    def prepare_for_model(self, features: Dict[str, np.ndarray], 
                         sequence_length: int = 100) -> np.ndarray:
        """
        Prepare features for CNN-LSTM model input.
        
        Args:
            features: Dictionary of extracted features
            sequence_length: Desired sequence length for model
            
        Returns:
            Array of shape (sequence_length, n_mfcc) ready for model
        """
        mfcc = features.get('mfcc', np.zeros(self.n_mfcc))
        
        # If MFCC is 1D, we need to create a sequence
        # For now, we'll pad/repeat to create a sequence
        if len(mfcc.shape) == 1:
            # Repeat MFCC to create sequence
            mfcc_sequence = np.tile(mfcc, (sequence_length, 1))
        else:
            # Already a sequence, pad or truncate
            current_length = mfcc.shape[0]
            if current_length < sequence_length:
                # Pad with last frame
                padding = np.tile(mfcc[-1:], (sequence_length - current_length, 1))
                mfcc_sequence = np.vstack([mfcc, padding])
            else:
                # Truncate
                mfcc_sequence = mfcc[:sequence_length]
        
        return mfcc_sequence.astype(np.float32)


class DatasetLoader:
    """Loads and preprocesses emotion speech and text datasets."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.extractor = AudioFeatureExtractor()
        self.text_preprocessor = TextPreprocessor()
    
    def load_ravdess_dataset(self, dataset_path: str) -> pd.DataFrame:
        """
        Load RAVDESS (Ryerson Audio-Visual Database) dataset.
        
        RAVDESS files are named: [Modality]-[VocalChannel]-[Emotion]-[Intensity]-[Statement]-[Repetition]-[Actor].wav
        
        Args:
            dataset_path: Path to RAVDESS dataset directory
            
        Returns:
            DataFrame with file paths and labels
        """
        dataset_path = Path(dataset_path)
        files = list(dataset_path.rglob("*.wav"))
        
        data = []
        for file_path in files:
            filename = file_path.stem
            parts = filename.split('-')
            
            if len(parts) >= 3:
                emotion_code = int(parts[2])
                # RAVDESS emotion mapping
                emotion_map = {
                    1: "neutral", 2: "calm", 3: "happy", 4: "sad",
                    5: "angry", 6: "fearful", 7: "disgusted", 8: "surprised"
                }
                emotion = emotion_map.get(emotion_code, "neutral")
                
                data.append({
                    "file_path": str(file_path),
                    "emotion": emotion,
                    "emotion_code": emotion_code
                })
        
        return pd.DataFrame(data)
    
    def load_custom_dataset(self, metadata_path: str) -> pd.DataFrame:
        """
        Load custom dataset from metadata CSV.
        
        CSV should have columns: file_path, emotion
        
        Args:
            metadata_path: Path to metadata CSV file
            
        Returns:
            DataFrame with file paths and labels
        """
        return pd.read_csv(metadata_path)
    
    def preprocess_dataset(
        self,
        df: pd.DataFrame,
        output_dir: Optional[str] = None,
        max_files: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        Preprocess dataset and extract features.
        
        Args:
            df: DataFrame with file_path and emotion columns
            output_dir: Optional directory to save preprocessed features
            max_files: Optional limit on number of files to process
            
        Returns:
            Tuple of (features_list, labels_list)
        """
        if max_files:
            df = df.head(max_files)
        
        features_list = []
        labels_list = []
        
        print(f"Processing {len(df)} audio files...")
        
        for idx, row in df.iterrows():
            try:
                features = self.extractor.extract_from_file(row['file_path'])
                model_input = self.extractor.prepare_for_model(features)
                
                features_list.append(model_input)
                labels_list.append(row['emotion'])
                
                if (idx + 1) % 10 == 0:
                    print(f"Processed {idx + 1}/{len(df)} files...")
            except Exception as e:
                print(f"Error processing {row['file_path']}: {e}")
                continue
        
        print(f"Successfully processed {len(features_list)} files.")
        
        # Save if output_dir provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            np.save(output_path / "features.npy", np.array(features_list))
            np.save(output_path / "labels.npy", np.array(labels_list))
            
            print(f"Saved preprocessed data to {output_dir}")
        
        return features_list, labels_list
    
    def load_goemotions_dataset(
        self,
        dataset_path: Optional[str] = None,
        from_huggingface: bool = True,
        split: str = "train",
        simplified: bool = True,
        max_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load GoEmotions text emotion dataset.
        
        Args:
            dataset_path: Path to local GoEmotions files (if from_huggingface=False)
            from_huggingface: Load from Hugging Face (requires datasets library)
            split: Dataset split ('train', 'validation', 'test')
            simplified: Use simplified version
            max_samples: Optional limit on number of samples
            
        Returns:
            DataFrame with text and emotion columns
        """
        from src.milestone_b.goemotions_loader import GoEmotionsLoader
        
        loader = GoEmotionsLoader(data_dir=dataset_path)
        
        if from_huggingface:
            return loader.load_from_huggingface(
                split=split,
                simplified=simplified,
                max_samples=max_samples
            )
        else:
            if dataset_path is None:
                raise ValueError("dataset_path required when from_huggingface=False")
            return loader.load_from_local(dataset_path, max_samples=max_samples)
    
    def preprocess_text_dataset(
        self,
        df: pd.DataFrame,
        output_dir: Optional[str] = None,
        max_samples: Optional[int] = None
    ) -> Tuple[List[Dict], List[str]]:
        """
        Preprocess text dataset for emotion classification.
        
        Args:
            df: DataFrame with text and emotion columns
            output_dir: Optional directory to save preprocessed features
            max_samples: Optional limit on number of samples
            
        Returns:
            Tuple of (preprocessed_texts_list, labels_list)
        """
        if max_samples:
            df = df.head(max_samples)
        
        preprocessed_list = []
        labels_list = []
        
        print(f"Processing {len(df)} text samples...")
        
        for idx, row in df.iterrows():
            try:
                # Preprocess text
                preprocessed = self.text_preprocessor.preprocess(row['text'])
                preprocessed_list.append(preprocessed)
                labels_list.append(row['emotion'])
                
                if (idx + 1) % 100 == 0:
                    print(f"Processed {idx + 1}/{len(df)} samples...")
            except Exception as e:
                print(f"Error processing text at index {idx}: {e}")
                continue
        
        print(f"Successfully processed {len(preprocessed_list)} text samples.")
        
        # Save if output_dir provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save preprocessed data
            import json
            data_to_save = {
                'preprocessed_texts': preprocessed_list,
                'labels': labels_list
            }
            with open(output_path / "text_features.json", 'w') as f:
                json.dump(data_to_save, f, indent=2, default=str)
            
            print(f"Saved preprocessed text data to {output_dir}")
        
        return preprocessed_list, labels_list

