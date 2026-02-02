"""Tests for Milestone A: Text prototype."""

import pytest
from src.milestone_a.text_prototype import RelaxationAgentPipeline
from src.core.preprocessor import TextPreprocessor
from src.core.emotion_classifier import EmotionClassifier, TextEmotionClassifier
from src.core.stress_scorer import StressScorer
from src.core.prompt_generator import CopingPromptGenerator


def test_text_preprocessor():
    """Test text preprocessing."""
    preprocessor = TextPreprocessor()
    result = preprocessor.preprocess("I'm feeling really stressed and anxious!")
    
    assert "tokens" in result
    assert "features" in result
    assert len(result["tokens"]) > 0
    assert len(result["features"]) == 6


def test_emotion_classifier():
    """Test emotion classification."""
    classifier = TextEmotionClassifier()
    
    # Test anxious text
    probs = classifier.predict("I'm anxious and worried about the exam")
    assert "anxious" in probs
    assert probs["anxious"] > 0
    
    # Test happy text
    probs = classifier.predict("I'm so happy and excited!")
    assert "happy" in probs
    assert probs["happy"] > 0


def test_stress_scorer():
    """Test stress scoring."""
    scorer = StressScorer()
    
    # High stress emotion
    emotion_probs = {"anxious": 0.8, "stressed": 0.2}
    result = scorer.compute_stress_score(emotion_probs)
    
    assert "stress_score" in result
    assert "stress_level" in result
    assert result["stress_score"] > 0.5
    assert result["stress_level"] in ["low", "medium", "high"]


def test_prompt_generator():
    """Test prompt generation."""
    generator = CopingPromptGenerator()
    
    result = generator.generate(
        stress_level="high",
        top_emotion="anxious",
        stress_score=0.8
    )
    
    assert "prompt" in result
    assert len(result["prompt"]) > 0
    assert result["stress_level"] == "high"


def test_pipeline():
    """Test full pipeline."""
    pipeline = RelaxationAgentPipeline(enable_logging=False)
    
    result = pipeline.process_text(
        "I'm feeling really stressed and anxious about my upcoming exam."
    )
    
    assert "emotion" in result
    assert "stress" in result
    assert "coping_prompt" in result
    assert result["emotion"]["top_emotion"] is not None
    assert result["stress"]["stress_score"] >= 0.0
    assert result["stress"]["stress_score"] <= 1.0


def test_pipeline_stress_levels():
    """Test pipeline with different stress levels."""
    pipeline = RelaxationAgentPipeline(enable_logging=False)
    
    # High stress
    result = pipeline.process_text("I'm extremely anxious and stressed!")
    assert result["stress"]["stress_level"] in ["high", "medium"]
    
    # Low stress
    result = pipeline.process_text("I'm feeling calm and peaceful.")
    assert result["stress"]["stress_level"] in ["low", "medium"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

