"""Stress scoring module."""

from typing import Dict, Optional
import numpy as np
from .config import Config


class StressScorer:
    """Computes stress scores from emotion predictions and context."""
    
    def __init__(self):
        # Emotion to stress mapping (0-1 scale, where 1 is highest stress)
        self.emotion_stress_map = {
            "neutral": 0.3,
            "happy": 0.1,
            "calm": 0.2,
            "sad": 0.6,
            "angry": 0.8,
            "fearful": 0.7,
            "anxious": 0.85,
            "stressed": 0.9,
            "disgusted": 0.5,
            "surprised": 0.4
        }
        
        self.high_threshold = Config.STRESS_THRESHOLD_HIGH
        self.medium_threshold = Config.STRESS_THRESHOLD_MEDIUM
    
    def compute_stress_score(
        self,
        emotion_probs: Dict[str, float],
        text_features: Optional[Dict] = None,
        audio_features: Optional[Dict] = None,
        use_architecture_formula: bool = True
    ) -> Dict[str, float]:
        """
        Compute stress score from emotion probabilities and features.
        
        Args:
            emotion_probs: Dictionary of emotion probabilities
            text_features: Optional text features from preprocessor
            audio_features: Optional audio features
            use_architecture_formula: If True, use formula: 100 × (0.6·A + 0.3·G + 0.1·D)
            
        Returns:
            Dictionary with stress_score, stress_level, and breakdown
        """
        if use_architecture_formula:
            # Architecture formula: stress_score = 100 × (0.6·A + 0.3·G + 0.1·D)
            # Where: A = anxious, G = angry, D = distracted
            A = emotion_probs.get("anxious", 0.0)
            G = emotion_probs.get("angry", 0.0)
            D = emotion_probs.get("distracted", 0.0)
            
            # Map other emotions to these categories if needed
            # If "stressed" exists, add to anxious
            if "stressed" in emotion_probs:
                A = max(A, emotion_probs["stressed"])
            
            base_score_100 = 100.0 * (0.6 * A + 0.3 * G + 0.1 * D)
            base_score = base_score_100 / 100.0  # Normalize to 0-1
        else:
            # Legacy formula: weighted sum of all emotions
            base_score = sum(
                prob * self.emotion_stress_map.get(emotion, 0.5)
                for emotion, prob in emotion_probs.items()
            )
        
        # Adjust based on text features
        if text_features:
            features = text_features.get('features', np.array([]))
            if len(features) >= 6:
                # Negative words increase stress
                negative_words = features[4] if len(features) > 4 else 0
                # Positive words decrease stress
                positive_words = features[5] if len(features) > 5 else 0
                
                # Normalize adjustments
                text_adjustment = (negative_words * 0.1) - (positive_words * 0.1)
                base_score = np.clip(base_score + text_adjustment, 0.0, 1.0)
        
        # Adjust based on audio features
        if audio_features:
            # High pitch variation can indicate stress
            pitch = audio_features.get('pitch', np.array([0]))[0] if isinstance(audio_features.get('pitch'), np.ndarray) else 0
            if pitch > 200:  # High pitch threshold
                base_score = min(base_score + 0.1, 1.0)
            
            # High zero crossing rate can indicate stress
            zcr = audio_features.get('zero_crossing_rate', 0)
            if zcr > 0.1:
                base_score = min(base_score + 0.05, 1.0)
        
        # Determine stress level
        if base_score >= self.high_threshold:
            stress_level = "high"
        elif base_score >= self.medium_threshold:
            stress_level = "medium"
        else:
            stress_level = "low"
        
        return {
            "stress_score": float(base_score),
            "stress_level": stress_level,
            "emotion_breakdown": emotion_probs,
            "thresholds": {
                "high": self.high_threshold,
                "medium": self.medium_threshold
            }
        }

