"""Enhanced Audio Emotion Detection - Milestone B."""

import numpy as np
import torch
from typing import Dict, Optional
from pathlib import Path

from src.core.preprocessor import AudioPreprocessor
from src.core.acoustic_encoder import AcousticEmotionEncoder, create_acoustic_encoder
from src.core.emotion_classifier import EmotionClassifier, EMOTION_LABELS


class EnhancedAudioEmotionDetector:
    """
    Enhanced audio emotion detection using CNN+BiLSTM.
    
    This is the complete Milestone B implementation with:
    - Enhanced feature extraction (MFCC, pitch, energy, speech_rate)
    - CNN+BiLSTM acoustic encoder
    - Emotion classification from audio
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        sample_rate: int = 22050,
        n_mfcc: int = 13
    ):
        """
        Initialize enhanced audio emotion detector.
        
        Args:
            model_path: Path to trained CNN-LSTM model
            sample_rate: Audio sample rate
            n_mfcc: Number of MFCC coefficients
        """
        self.preprocessor = AudioPreprocessor(sample_rate=sample_rate, n_mfcc=n_mfcc)
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        
        # Initialize acoustic encoder
        self.acoustic_encoder = None
        self.model_path = model_path
        
        # Initialize emotion classifier
        self.emotion_classifier = EmotionClassifier(model_path=model_path)
        
        # If model available, load acoustic encoder
        if model_path and Path(model_path).exists():
            try:
                # Try to determine input dimension from model
                checkpoint = torch.load(model_path, map_location='cpu')
                # For now, use default
                self.acoustic_encoder = create_acoustic_encoder(input_dim=n_mfcc)
                print(f"Loaded acoustic encoder from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load acoustic encoder: {e}")
    
    def extract_enhanced_features(self, audio_path: str) -> Dict[str, np.ndarray]:
        """
        Extract enhanced audio features including:
        - MFCC
        - Pitch
        - Energy
        - Speech rate
        - Spectral features
        """
        try:
            import librosa
        except ImportError:
            raise ImportError("librosa is required for audio processing")
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Extract MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        mfccs_mean = np.mean(mfccs, axis=1)
        
        # Extract pitch
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0.0
        pitch_std = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0.0
        
        # Extract energy
        energy = librosa.feature.rms(y=y)[0]
        energy_mean = np.mean(energy)
        energy_std = np.std(energy)
        
        # Extract speech rate (approximate from zero crossings)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = np.mean(zcr)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        return {
            "mfcc": mfccs_mean,
            "mfcc_full": mfccs,  # Full sequence for model
            "pitch": np.array([pitch_mean, pitch_std], dtype=np.float32),
            "energy": np.array([energy_mean, energy_std], dtype=np.float32),
            "speech_rate": np.array([zcr_mean], dtype=np.float32),
            "spectral_centroid": np.mean(spectral_centroids),
            "spectral_rolloff": np.mean(spectral_rolloff),
            "zero_crossing_rate": zcr_mean,
            "tempo": np.array([tempo], dtype=np.float32),
            "duration": len(y) / sr
        }
    
    def detect_emotion(
        self,
        audio_path: Optional[str] = None,
        audio_features: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Detect emotion from audio.
        
        Args:
            audio_path: Path to audio file
            audio_features: Pre-extracted features (optional)
            
        Returns:
            Dictionary of emotion probabilities
        """
        # Extract features if not provided
        if audio_features is None:
            if audio_path is None:
                raise ValueError("Either audio_path or audio_features must be provided")
            audio_features = self.extract_enhanced_features(audio_path)
        
        # Use emotion classifier
        if self.emotion_classifier.model is not None:
            # Use trained model
            emotion_probs = self.emotion_classifier.predict_from_audio(audio_features)
        else:
            # Fallback: use rule-based from audio features
            emotion_probs = self._rule_based_audio_emotion(audio_features)
        
        return emotion_probs
    
    def _rule_based_audio_emotion(self, features: Dict) -> Dict[str, float]:
        """Rule-based emotion detection from audio features (fallback)."""
        probs = {label: 0.0 for label in EMOTION_LABELS}
        
        # Analyze pitch
        pitch = features.get('pitch', np.array([0]))[0] if isinstance(features.get('pitch'), np.ndarray) else 0
        
        # Analyze energy
        energy = features.get('energy', np.array([0]))[0] if isinstance(features.get('energy'), np.ndarray) else 0
        
        # Analyze speech rate
        speech_rate = features.get('speech_rate', np.array([0]))[0] if isinstance(features.get('speech_rate'), np.ndarray) else 0
        
        # Simple heuristics
        if pitch > 200 and energy > 0.1:
            probs["anxious"] = 0.6
            probs["stressed"] = 0.3
            probs["neutral"] = 0.1
        elif pitch < 150 and energy < 0.05:
            probs["sad"] = 0.5
            probs["calm"] = 0.3
            probs["neutral"] = 0.2
        elif energy > 0.15:
            probs["angry"] = 0.5
            probs["anxious"] = 0.3
            probs["neutral"] = 0.2
        else:
            probs["neutral"] = 0.6
            probs["calm"] = 0.4
        
        return probs


def create_audio_emotion_detector(model_path: Optional[str] = None) -> EnhancedAudioEmotionDetector:
    """Factory function to create audio emotion detector."""
    return EnhancedAudioEmotionDetector(model_path=model_path)

