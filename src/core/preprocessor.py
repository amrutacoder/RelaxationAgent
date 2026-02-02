"""Text and audio preprocessing module."""

import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (will be done on first import)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


class TextPreprocessor:
    """Preprocesses text input for emotion classification."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def preprocess(self, text: str) -> Dict[str, Union[List[str], np.ndarray]]:
        """
        Preprocess text for emotion classification.
        
        Args:
            text: Raw input text
            
        Returns:
            Dictionary with processed tokens and features
        """
        # Clean text
        text = self._clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and lemmatize
        processed_tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token.isalnum() and token not in self.stop_words
        ]
        
        # Extract features
        features = self._extract_text_features(text, processed_tokens)
        
        return {
            "tokens": processed_tokens,
            "features": features,
            "original_text": text
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove special characters but keep punctuation for emotion cues
        text = re.sub(r'[^\w\s.,!?]', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    def _extract_text_features(self, text: str, tokens: List[str]) -> np.ndarray:
        """
        Extract numerical features from text.
        
        Returns:
            Feature vector: [word_count, avg_word_length, exclamation_count, 
                            question_count, negative_words, positive_words]
        """
        word_count = len(tokens)
        avg_word_length = np.mean([len(t) for t in tokens]) if tokens else 0
        
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        # Simple sentiment word lists (can be expanded)
        negative_words = ['sad', 'angry', 'frustrated', 'stressed', 'worried', 
                         'anxious', 'tired', 'upset', 'hurt', 'scared']
        positive_words = ['happy', 'joy', 'excited', 'calm', 'relaxed', 
                         'peaceful', 'content', 'grateful', 'hopeful']
        
        negative_count = sum(1 for token in tokens if token in negative_words)
        positive_count = sum(1 for token in tokens if token in positive_words)
        
        return np.array([
            word_count,
            avg_word_length,
            exclamation_count,
            question_count,
            negative_count,
            positive_count
        ], dtype=np.float32)


class AudioPreprocessor:
    """Preprocesses audio for emotion classification."""
    
    def __init__(self, sample_rate: int = 22050, n_mfcc: int = 13):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
    
    def extract_features(self, audio_path: str) -> Dict[str, np.ndarray]:
        """
        Extract audio features from file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with MFCC, pitch, and other features
        """
        try:
            import librosa
        except ImportError:
            raise ImportError("librosa is required for audio processing. Install with: pip install librosa")
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        mfccs_mean = np.mean(mfccs, axis=1)
        
        # Extract pitch (fundamental frequency)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0.0
        
        # Extract spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        
        # Extract tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        return {
            "mfcc": mfccs_mean,
            "pitch": np.array([pitch_mean], dtype=np.float32),
            "spectral_centroid": np.mean(spectral_centroids),
            "spectral_rolloff": np.mean(spectral_rolloff),
            "zero_crossing_rate": np.mean(zero_crossing_rate),
            "tempo": np.array([tempo], dtype=np.float32),
            "duration": len(y) / sr
        }
    
    def extract_features_from_array(self, audio_array: np.ndarray, 
                                   sample_rate: int = None) -> Dict[str, np.ndarray]:
        """
        Extract features from audio array.
        
        Args:
            audio_array: Audio signal as numpy array
            sample_rate: Sample rate (uses self.sample_rate if None)
            
        Returns:
            Dictionary with extracted features
        """
        try:
            import librosa
        except ImportError:
            raise ImportError("librosa is required for audio processing.")
        
        sr = sample_rate or self.sample_rate
        
        # Extract features (similar to extract_features)
        mfccs = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=self.n_mfcc)
        mfccs_mean = np.mean(mfccs, axis=1)
        
        pitches, magnitudes = librosa.piptrack(y=audio_array, sr=sr)
        pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0.0
        
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_array, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_array, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_array)[0]
        
        tempo, _ = librosa.beat.beat_track(y=audio_array, sr=sr)
        
        return {
            "mfcc": mfccs_mean,
            "pitch": np.array([pitch_mean], dtype=np.float32),
            "spectral_centroid": np.mean(spectral_centroids),
            "spectral_rolloff": np.mean(spectral_rolloff),
            "zero_crossing_rate": np.mean(zero_crossing_rate),
            "tempo": np.array([tempo], dtype=np.float32),
            "duration": len(audio_array) / sr
        }

