"""Milestone A: Text-based emotion detection prototype.

This is the first milestone - a working text-only prototype that:
- Takes text input (simulating STT output)
- Predicts emotion label
- Computes stress score
- Generates coping prompt
- Returns results via console or REST
"""

from typing import Dict, Optional
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.preprocessor import TextPreprocessor
from src.core.emotion_classifier import EmotionClassifier
from src.core.stress_scorer import StressScorer
from src.core.prompt_generator import CopingPromptGenerator
from src.core.voice_style_generator import create_voice_style_generator
from src.core.logger import Logger


class RelaxationAgentPipeline:
    """Main pipeline for relaxation agent processing."""
    
    def __init__(self, enable_logging: bool = True):
        self.text_preprocessor = TextPreprocessor()
        self.emotion_classifier = EmotionClassifier(use_text_fallback=True)
        self.stress_scorer = StressScorer()
        self.prompt_generator = CopingPromptGenerator()
        self.voice_style_generator = create_voice_style_generator()
        self.logger = Logger() if enable_logging else None
        self.communicator = None  # Will be added in Milestone D
    
    def process_text(
        self,
        text: str,
        user_id: Optional[str] = None,
        publish_alerts: bool = False
    ) -> Dict:
        """
        Process text input and return emotion, stress, and prompt.
        
        Args:
            text: Input text (simulated STT output)
            user_id: Optional user identifier
            publish_alerts: Whether to publish alerts (requires communicator)
            
        Returns:
            Dictionary with analysis results
        """
        # Preprocess text
        preprocessed = self.text_preprocessor.preprocess(text)
        
        # Classify emotion
        emotion_probs = self.emotion_classifier.predict_from_text(text, preprocessed)
        top_emotion, top_prob = self.emotion_classifier.get_top_emotion(emotion_probs)
        
        # Compute stress score
        stress_result = self.stress_scorer.compute_stress_score(
            emotion_probs,
            text_features=preprocessed
        )
        
        # Generate coping prompt
        prompt_result = self.prompt_generator.generate(
            stress_level=stress_result["stress_level"],
            top_emotion=top_emotion,
            stress_score=stress_result["stress_score"],
            emotion_probs=emotion_probs,
            use_strategy_selection=True
        )
        
        # Get strategy for voice style
        strategy = self.prompt_generator.select_strategy(emotion_probs, stress_result["stress_score"])
        
        # Generate voice style
        voice_style = self.voice_style_generator.generate_voice_style(
            stress_level=stress_result["stress_level"],
            top_emotion=top_emotion,
            strategy=strategy,
            stress_score=stress_result["stress_score"]
        )
        
        # Log result
        if self.logger:
            self.logger.log_emotion_analysis(
                text_input=text,
                emotion_probs=emotion_probs,
                top_emotion=top_emotion,
                stress_score=stress_result["stress_score"],
                stress_level=stress_result["stress_level"],
                prompt=prompt_result["prompt"],
                user_id=user_id
            )
        
        # Publish alerts if high stress
        if publish_alerts and stress_result["stress_level"] in ["high", "medium"]:
            if self.communicator:
                self.communicator.publish_stress_alert(
                    stress_score=stress_result["stress_score"],
                    stress_level=stress_result["stress_level"],
                    emotion=top_emotion,
                    user_id=user_id
                )
        
        return {
            "input_text": text,
            "emotion": {
                "top_emotion": top_emotion,
                "probability": top_prob,
                "all_emotions": emotion_probs
            },
            "stress": stress_result,
            "coping_prompt": prompt_result,
            "voice_output": {
                "text": prompt_result["prompt"],
                "voice_style": voice_style
            },
            "user_id": user_id
        }


def main():
    """Test the text prototype."""
    print("=" * 60)
    print("Relaxation Agent - Milestone A: Text Prototype")
    print("=" * 60)
    print()
    
    pipeline = RelaxationAgentPipeline(enable_logging=True)
    
    # Test cases
    test_cases = [
        "I'm feeling really stressed and anxious about my upcoming exam.",
        "I'm so happy and excited about the weekend!",
        "I'm angry and frustrated with this situation.",
        "I feel calm and peaceful right now.",
        "I'm worried and nervous about the presentation tomorrow."
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}: {text}")
        print('='*60)
        
        result = pipeline.process_text(text, user_id=f"test_user_{i}")
        
        print(f"\n📊 Emotion Analysis:")
        print(f"   Top Emotion: {result['emotion']['top_emotion']} ({result['emotion']['probability']:.2%})")
        
        print(f"\n😰 Stress Analysis:")
        print(f"   Stress Score: {result['stress']['stress_score']:.2f}")
        print(f"   Stress Level: {result['stress']['stress_level'].upper()}")
        
        print(f"\n💡 Coping Prompt:")
        print(f"   {result['coping_prompt']['prompt']}")
        print()


if __name__ == "__main__":
    main()

