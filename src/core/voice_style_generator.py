"""Voice Style Generator - Maps emotions/stress to TTS voice styles."""

from typing import Literal, Dict, Optional
from .stress_scorer import StressScorer

VoiceStyle = Literal[
    "calm_slow",
    "calm_normal",
    "supportive_gentle",
    "supportive_warm",
    "urgent_calm",
    "reassuring",
    "neutral"
]


class VoiceStyleGenerator:
    """
    Generates voice style recommendations for TTS based on:
    - Detected emotion
    - Stress level
    - Coping strategy
    """
    
    def __init__(self):
        """Initialize voice style generator."""
        # Voice style mappings
        self.style_mappings = {
            # High stress + anxious → calm_slow
            ("high", "anxious", "breathing"): "calm_slow",
            ("high", "anxious", None): "calm_slow",
            
            # High stress + angry → supportive_gentle
            ("high", "angry", "grounding"): "supportive_gentle",
            ("high", "angry", None): "supportive_gentle",
            
            # High stress + distracted → urgent_calm
            ("high", "distracted", "focus_reset"): "urgent_calm",
            ("high", "distracted", None): "urgent_calm",
            
            # Medium stress → supportive_warm
            ("medium", None, None): "supportive_warm",
            
            # Low stress → calm_normal
            ("low", None, None): "calm_normal",
            
            # Default → neutral
            (None, None, None): "neutral"
        }
        
        # Emotion-based style preferences
        self.emotion_styles = {
            "anxious": "calm_slow",
            "stressed": "calm_slow",
            "angry": "supportive_gentle",
            "fearful": "reassuring",
            "sad": "supportive_warm",
            "distracted": "urgent_calm",
            "calm": "calm_normal",
            "happy": "calm_normal",
            "neutral": "neutral"
        }
    
    def generate_voice_style(
        self,
        stress_level: str,
        top_emotion: str,
        strategy: Optional[str] = None,
        stress_score: Optional[float] = None
    ) -> VoiceStyle:
        """
        Generate voice style based on stress, emotion, and strategy.
        
        Args:
            stress_level: "high", "medium", or "low"
            top_emotion: Primary emotion detected
            strategy: Coping strategy (optional)
            stress_score: Stress score (optional, for fine-tuning)
            
        Returns:
            Voice style string
        """
        # Try exact match first
        key = (stress_level, top_emotion, strategy)
        if key in self.style_mappings:
            return self.style_mappings[key]
        
        # Try without strategy
        key = (stress_level, top_emotion, None)
        if key in self.style_mappings:
            return self.style_mappings[key]
        
        # Try with just stress level
        key = (stress_level, None, None)
        if key in self.style_mappings:
            return self.style_mappings[key]
        
        # Fall back to emotion-based style
        if top_emotion in self.emotion_styles:
            return self.emotion_styles[top_emotion]
        
        # Default
        return "neutral"
    
    def get_voice_style_properties(self, style: VoiceStyle) -> Dict[str, any]:
        """
        Get detailed properties for voice style (for TTS systems).
        
        Args:
            style: Voice style identifier
            
        Returns:
            Dictionary with TTS parameters
        """
        style_properties = {
            "calm_slow": {
                "pitch": "low",
                "speed": "slow",  # 0.7x normal
                "energy": "soft",
                "prosody": "gentle",
                "pause_duration": "long"
            },
            "calm_normal": {
                "pitch": "medium",
                "speed": "normal",
                "energy": "moderate",
                "prosody": "smooth",
                "pause_duration": "normal"
            },
            "supportive_gentle": {
                "pitch": "medium_low",
                "speed": "slow",  # 0.8x normal
                "energy": "soft",
                "prosody": "warm",
                "pause_duration": "medium"
            },
            "supportive_warm": {
                "pitch": "medium",
                "speed": "normal",
                "energy": "warm",
                "prosody": "friendly",
                "pause_duration": "normal"
            },
            "urgent_calm": {
                "pitch": "medium",
                "speed": "slightly_slow",  # 0.9x normal
                "energy": "calm",
                "prosody": "clear",
                "pause_duration": "short"
            },
            "reassuring": {
                "pitch": "medium_low",
                "speed": "normal",
                "energy": "gentle",
                "prosody": "reassuring",
                "pause_duration": "medium"
            },
            "neutral": {
                "pitch": "medium",
                "speed": "normal",
                "energy": "neutral",
                "prosody": "neutral",
                "pause_duration": "normal"
            }
        }
        
        return style_properties.get(style, style_properties["neutral"])


def create_voice_style_generator() -> VoiceStyleGenerator:
    """Factory function to create voice style generator."""
    return VoiceStyleGenerator()

