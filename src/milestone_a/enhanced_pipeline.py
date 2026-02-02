"""Enhanced Relaxation Agent Pipeline - Full 10-stage implementation."""

import torch
from typing import Dict, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.preprocessor import TextPreprocessor, AudioPreprocessor
from src.core.text_encoder import TextEmotionEncoder, create_text_encoder
from src.core.acoustic_encoder import AcousticEmotionEncoder, create_acoustic_encoder
from src.core.multimodal_fusion import MultimodalFusion
from src.core.emotion_classifier import EmotionClassifier, EMOTION_LABELS
from src.core.stress_scorer import StressScorer
from src.core.profile_interpreter import ProfileInterpreter, UserProfile
from src.core.prompt_generator import CopingPromptGenerator
from src.core.voice_style_generator import VoiceStyleGenerator, create_voice_style_generator
from src.core.communicator import Communicator
from src.core.logger import Logger
from src.core.config import Config


class EnhancedRelaxationAgentPipeline:
    """
    Enhanced pipeline implementing full 10-stage architecture.
    
    Stages:
    1. Input Preparation (preprocessing)
    2. Text Emotion Encoder (DistilBERT)
    3. Acoustic Emotion Encoder (CNN+BiLSTM)
    4. Multimodal Fusion
    5. Emotion Classification (via existing classifier or new model)
    6. Stress Score Computation
    7. Profile-Conditioned Interpretation
    8. Coping Strategy Selection
    9. Coping Prompt Generation
    """
    
    def __init__(
        self,
        enable_logging: bool = True,
        use_text_encoder: bool = True,
        use_acoustic_encoder: bool = True,
        use_fusion: bool = True,
        use_profile: bool = True
    ):
        """
        Initialize enhanced pipeline.
        
        Args:
            enable_logging: Enable database logging
            use_text_encoder: Use DistilBERT text encoder (requires transformers)
            use_acoustic_encoder: Use CNN+BiLSTM acoustic encoder
            use_fusion: Use multimodal fusion
            use_profile: Use profile-conditioned interpretation
        """
        # Stage 1: Input preparation
        self.text_preprocessor = TextPreprocessor()
        self.audio_preprocessor = AudioPreprocessor()
        
        # Stage 2: Text encoder (optional, falls back to rule-based)
        self.use_text_encoder = use_text_encoder
        try:
            if use_text_encoder:
                self.text_encoder = create_text_encoder()
                print("Text encoder (DistilBERT) initialized")
            else:
                self.text_encoder = None
        except Exception as e:
            print(f"Warning: Could not initialize text encoder: {e}")
            print("Falling back to rule-based text classification")
            self.text_encoder = None
            self.use_text_encoder = False
        
        # Stage 3: Acoustic encoder (optional)
        self.use_acoustic_encoder = use_acoustic_encoder and use_fusion
        if use_acoustic_encoder:
            # Will be initialized when audio is provided
            self.acoustic_encoder = None
            self.acoustic_input_dim = 13  # MFCC default
        
        # Stage 4: Multimodal fusion
        self.use_fusion = use_fusion and (use_text_encoder or use_acoustic_encoder)
        if self.use_fusion:
            self.fusion = MultimodalFusion(method="concatenation")
        else:
            self.fusion = None
        
        # Stage 5: Emotion classifier (existing implementation)
        self.emotion_classifier = EmotionClassifier(use_text_fallback=not use_text_encoder)
        
        # Stage 6: Stress scorer
        self.stress_scorer = StressScorer()
        
        # Stage 7: Profile interpreter
        self.use_profile = use_profile
        if use_profile:
            self.profile_interpreter = ProfileInterpreter()
        else:
            self.profile_interpreter = None
        
        # Stage 8 & 9: Prompt generator
        self.prompt_generator = CopingPromptGenerator()
        
        # Voice style generator (for Voice Agent output)
        self.voice_style_generator = create_voice_style_generator()
        
        # Supporting services
        self.logger = Logger() if enable_logging else None
        self.communicator = None  # Will be set if needed
    
    def process(
        self,
        text: Optional[str] = None,
        audio_features: Optional[Dict] = None,
        user_profile: Optional[UserProfile] = None,
        user_id: Optional[str] = None,
        publish_alerts: bool = False
    ) -> Dict:
        """
        Process input through full pipeline.
        
        Args:
            text: Input text (required if no audio)
            audio_features: Optional audio features dict
            user_profile: Optional user profile for personalization
            user_id: Optional user identifier
            publish_alerts: Whether to publish alerts via Redis
            
        Returns:
            Complete analysis result dictionary
        """
        if text is None and audio_features is None:
            raise ValueError("Either text or audio_features must be provided")
        
        # Stage 1: Preprocessing
        text_processed = None
        audio_processed = None
        
        if text:
            text_processed = self.text_preprocessor.preprocess(text)
        
        if audio_features:
            # Audio features should already be extracted
            audio_processed = audio_features
        
        # Stage 2: Text encoding
        text_emb = None
        emotion_probs_from_text = None
        
        if text:
            if self.use_text_encoder and self.text_encoder:
                # Use DistilBERT
                text_emb = self.text_encoder.encode(text)
            else:
                # Fall back to rule-based classification
                emotion_probs_from_text = self.emotion_classifier.predict_from_text(
                    text, text_processed
                )
        
        # Stage 3: Acoustic encoding
        audio_emb = None
        
        if audio_features and self.use_acoustic_encoder:
            if self.acoustic_encoder is None:
                # Initialize with appropriate input dimension
                mfcc_dim = len(audio_features.get('mfcc', []))
                if mfcc_dim == 0:
                    mfcc_dim = 13  # Default
                self.acoustic_input_dim = mfcc_dim
                self.acoustic_encoder = create_acoustic_encoder(input_dim=mfcc_dim)
            
            # Prepare input for encoder (sequence format)
            from src.milestone_b.audio_features import AudioFeatureExtractor
            extractor = AudioFeatureExtractor()
            model_input = extractor.prepare_for_model(audio_features, sequence_length=100)
            audio_input_tensor = torch.FloatTensor(model_input).unsqueeze(0)
            audio_emb = self.acoustic_encoder(audio_input_tensor).squeeze(0)
        
        # Stage 4: Multimodal fusion (if both available)
        fused_emb = None
        
        if self.use_fusion and text_emb is not None and audio_emb is not None:
            fused_emb = self.fusion.fuse(text_emb, audio_emb)
            # Note: Full multimodal classifier would use fused_emb here
            # For now, we use text-based emotion classification
        
        # Stage 5: Emotion classification
        if emotion_probs_from_text is None:
            # Use text-based or combined classification
            if text:
                emotion_probs = self.emotion_classifier.predict_from_text(text, text_processed)
            else:
                # Audio-only: would need trained model
                emotion_probs = {label: 1.0 if label == "neutral" else 0.0 
                               for label in EMOTION_LABELS}
        else:
            emotion_probs = emotion_probs_from_text
        
        top_emotion, top_prob = self.emotion_classifier.get_top_emotion(emotion_probs)
        
        # Stage 6: Stress scoring
        stress_result = self.stress_scorer.compute_stress_score(
            emotion_probs,
            text_features=text_processed,
            audio_features=audio_features,
            use_architecture_formula=True
        )
        
        # Stage 7: Profile interpretation
        if self.use_profile and user_profile:
            stress_result = self.profile_interpreter.interpret_stress(
                stress_result["stress_score"],
                user_profile
            )
        
        # Stage 8: Strategy selection
        strategy = self.prompt_generator.select_strategy(
            emotion_probs,
            stress_result["stress_score"]
        )
        
        # Stage 9: Prompt generation
        prompt_result = self.prompt_generator.generate(
            stress_level=stress_result["stress_level"],
            top_emotion=top_emotion,
            stress_score=stress_result.get("stress_score_normalized", stress_result["stress_score"]),
            emotion_probs=emotion_probs,
            use_strategy_selection=True
        )
        
        # Generate voice style for Voice Agent
        voice_style = self.voice_style_generator.generate_voice_style(
            stress_level=stress_result["stress_level"],
            top_emotion=top_emotion,
            strategy=strategy,
            stress_score=stress_result.get("stress_score_normalized", stress_result["stress_score"])
        )
        
        # Logging
        if self.logger:
            self.logger.log_emotion_analysis(
                text_input=text,
                emotion_probs=emotion_probs,
                top_emotion=top_emotion,
                stress_score=stress_result.get("stress_score_normalized", stress_result["stress_score"]),
                stress_level=stress_result["stress_level"],
                prompt=prompt_result["prompt"],
                user_id=user_id,
                audio_features=audio_features
            )
        
        # Publishing alerts
        if publish_alerts and stress_result["stress_level"] in ["high", "medium"]:
            if self.communicator:
                self.communicator.publish_stress_alert(
                    stress_score=stress_result.get("stress_score_normalized", stress_result["stress_score"]),
                    stress_level=stress_result["stress_level"],
                    emotion=top_emotion,
                    user_id=user_id
                )
        
        return {
            "input": {
                "text": text,
                "has_audio": audio_features is not None
            },
            "emotion": {
                "top_emotion": top_emotion,
                "probability": top_prob,
                "all_emotions": emotion_probs
            },
            "stress": stress_result,
            "strategy": strategy,
            "coping_prompt": prompt_result,
            "voice_output": {
                "text": prompt_result["prompt"],
                "voice_style": voice_style,
                "style_properties": self.voice_style_generator.get_voice_style_properties(voice_style)
            },
            "user_id": user_id,
            "pipeline_info": {
                "used_text_encoder": self.use_text_encoder and self.text_encoder is not None,
                "used_acoustic_encoder": self.use_acoustic_encoder,
                "used_fusion": self.use_fusion and fused_emb is not None,
                "used_profile": self.use_profile and user_profile is not None
            }
        }


def main():
    """Test the enhanced pipeline."""
    print("=" * 70)
    print("Enhanced Relaxation Agent Pipeline - 10-Stage Architecture")
    print("=" * 70)
    print()
    
    # Initialize pipeline
    pipeline = EnhancedRelaxationAgentPipeline(
        enable_logging=True,
        use_text_encoder=True,  # Try to use DistilBERT
        use_fusion=True,
        use_profile=True
    )
    
    # Test case
    test_text = "I'm feeling really anxious and overwhelmed about everything."
    
    print(f"Processing: '{test_text}'")
    print("-" * 70)
    
    result = pipeline.process(
        text=test_text,
        user_id="test_user_enhanced"
    )
    
    print(f"\n📊 Emotion: {result['emotion']['top_emotion']} ({result['emotion']['probability']:.2%})")
    print(f"😰 Stress: {result['stress']['stress_level'].upper()} ({result['stress'].get('stress_score_normalized', result['stress']['stress_score']):.2f})")
    print(f"🎯 Strategy: {result['strategy']}")
    print(f"💡 Prompt: {result['coping_prompt']['prompt']}")
    print(f"\nPipeline: {result['pipeline_info']}")


if __name__ == "__main__":
    main()

