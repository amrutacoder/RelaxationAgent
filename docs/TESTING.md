# Testing and Evaluation Plan

## Test Plan Overview

This document outlines the testing strategy, dataset requirements, and evaluation metrics for the Relaxation Agent.

## Unit Tests

### Core Modules

#### Preprocessor Tests
- Text cleaning and normalization
- Tokenization and lemmatization
- Feature extraction
- Audio feature extraction (MFCC, pitch, etc.)

#### Emotion Classifier Tests
- Text-based emotion prediction
- Audio-based emotion prediction (when model available)
- Probability normalization
- Edge cases (empty input, very long input)

#### Stress Scorer Tests
- Stress score computation
- Threshold classification (low/medium/high)
- Feature weighting
- Boundary conditions

#### Prompt Generator Tests
- Rule-based prompt selection
- LLM prompt generation (if enabled)
- Stress level matching
- Emotion-specific prompts

#### Communicator Tests
- Redis connection
- Message publishing
- REST callback delivery
- Error handling

#### Logger Tests
- Database logging
- History retrieval
- Query filtering

## Integration Tests

### API Endpoints

Test all FastAPI endpoints:
- `/` - Root endpoint
- `/health` - Health check
- `/api/analyze/text` - Text analysis
- `/api/analyze/audio` - Audio analysis
- `/api/analyze/combined` - Combined analysis
- `/api/history` - History retrieval
- `/api/stress-alerts` - Alert history

### Redis Integration

- Publish/subscribe functionality
- Message format validation
- Channel subscription
- Error recovery

### Agent Communication

- Route Agent callback
- UI Agent callback
- Error handling for unavailable agents

## Dataset Plan

### Training Dataset

#### RAVDESS (Ryerson Audio-Visual Database)

**Description**: Professional actors expressing 8 emotions
- Emotions: neutral, calm, happy, sad, angry, fearful, disgusted, surprised
- Format: Audio files (WAV)
- Size: ~1,440 files
- Download: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

**Usage**:
```bash
# Download and extract RAVDESS
# Place in data/ravdess/

# Preprocess
python -m src.milestone_b.audio_features DatasetLoader
loader = DatasetLoader("data/ravdess")
df = loader.load_ravdess_dataset("data/ravdess")
features, labels = loader.preprocess_dataset(df)
```

#### CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)

**Description**: Crowd-sourced emotional speech
- Emotions: happy, sad, angry, fearful, neutral, disgusted
- Format: Audio files
- Size: ~7,442 files
- Download: https://www.kaggle.com/datasets/ejlok1/cremad

#### Custom Dataset

Create custom dataset with CSV metadata:

```csv
file_path,emotion
data/audio/happy_001.wav,happy
data/audio/sad_001.wav,sad
data/audio/anxious_001.wav,anxious
```

### Evaluation Dataset

Reserve 20% of training data for validation.

Create separate test set with:
- Diverse speakers
- Various audio qualities
- Real-world scenarios

### Text Dataset (for Milestone A)

Use public sentiment/emotion text datasets:
- **EmoBank**: Emotion-annotated texts
- **GoEmotions**: Reddit comments with emotions
- **ISEAR**: International Survey on Emotion Antecedents and Reactions

## Evaluation Metrics

### Emotion Classification

1. **Accuracy**: Overall classification accuracy
2. **Precision**: Per-emotion precision
3. **Recall**: Per-emotion recall
4. **F1-Score**: Per-emotion F1-score
5. **Confusion Matrix**: Visualize classification errors

### Stress Scoring

1. **Correlation**: Correlation with ground truth stress labels
2. **Threshold Accuracy**: Accuracy of stress level classification
3. **Calibration**: Stress score distribution vs. actual stress

### Coping Prompts

1. **Relevance**: Manual evaluation of prompt relevance
2. **Helpfulness**: User feedback on prompt effectiveness
3. **Diversity**: Variety of prompts generated

### System Performance

1. **Latency**: Response time for analysis
2. **Throughput**: Requests per second
3. **Resource Usage**: CPU, memory, GPU utilization

## Evaluation Scripts

### Model Evaluation

```python
# tests/evaluate_model.py
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.core.emotion_classifier import EmotionClassifier

def evaluate_model(model_path, test_features, test_labels):
    """Evaluate trained model."""
    classifier = EmotionClassifier(model_path)
    
    predictions = []
    for features in test_features:
        probs = classifier.predict_from_audio({'mfcc': features})
        top_emotion = max(probs.items(), key=lambda x: x[1])[0]
        predictions.append(top_emotion)
    
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions)
    cm = confusion_matrix(test_labels, predictions)
    
    print(f"Accuracy: {accuracy:.2%}")
    print(report)
    print("Confusion Matrix:")
    print(cm)
    
    return accuracy, report, cm
```

### Stress Score Evaluation

```python
# tests/evaluate_stress.py
from src.core.stress_scorer import StressScorer
import numpy as np

def evaluate_stress_scorer(emotion_probs_list, ground_truth_stress):
    """Evaluate stress scoring against ground truth."""
    scorer = StressScorer()
    
    predicted_scores = []
    for probs in emotion_probs_list:
        result = scorer.compute_stress_score(probs)
        predicted_scores.append(result['stress_score'])
    
    correlation = np.corrcoef(ground_truth_stress, predicted_scores)[0, 1]
    
    print(f"Correlation with ground truth: {correlation:.3f}")
    
    return correlation
```

### End-to-End Evaluation

```python
# tests/evaluate_e2e.py
from src.milestone_a.text_prototype import RelaxationAgentPipeline

def evaluate_pipeline(test_cases):
    """Evaluate full pipeline on test cases."""
    pipeline = RelaxationAgentPipeline()
    
    results = []
    for text, expected_emotion, expected_stress_level in test_cases:
        result = pipeline.process_text(text)
        
        predicted_emotion = result['emotion']['top_emotion']
        predicted_stress = result['stress']['stress_level']
        
        results.append({
            'text': text,
            'expected_emotion': expected_emotion,
            'predicted_emotion': predicted_emotion,
            'expected_stress': expected_stress_level,
            'predicted_stress': predicted_stress,
            'match_emotion': predicted_emotion == expected_emotion,
            'match_stress': predicted_stress == expected_stress_level
        })
    
    emotion_accuracy = sum(r['match_emotion'] for r in results) / len(results)
    stress_accuracy = sum(r['match_stress'] for r in results) / len(results)
    
    print(f"Emotion Accuracy: {emotion_accuracy:.2%}")
    print(f"Stress Level Accuracy: {stress_accuracy:.2%}")
    
    return results
```

## Test Cases

### Text Analysis Test Cases

```python
test_cases = [
    ("I'm feeling really stressed and anxious about my exam", "anxious", "high"),
    ("I'm so happy and excited about the weekend!", "happy", "low"),
    ("I'm angry and frustrated with this situation", "angry", "high"),
    ("I feel calm and peaceful right now", "calm", "low"),
    ("I'm worried and nervous about the presentation", "anxious", "medium"),
    ("", "neutral", "low"),  # Empty input
    ("I " * 1000, "neutral", "low"),  # Very long input
]
```

### Audio Analysis Test Cases

- Various audio qualities (clean, noisy)
- Different sample rates
- Different durations (short, long)
- Multiple speakers
- Background noise

### Edge Cases

- Empty input
- Very long input
- Special characters
- Multiple languages (if supported)
- Invalid audio format
- Missing Redis connection
- Agent callback failures

## Continuous Testing

### Pre-commit Hooks

```bash
# .git/hooks/pre-commit
#!/bin/bash
pytest tests/ -v
pylint src/
black --check src/
```

### CI/CD Pipeline

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v --cov=src
```

## Performance Benchmarks

### Target Metrics

- Text analysis: < 100ms
- Audio analysis: < 500ms
- Combined analysis: < 600ms
- Model inference: < 200ms (on CPU)

### Load Testing

Use tools like `locust` or `k6`:

```python
# tests/load_test.py
from locust import HttpUser, task, between

class RelaxationAgentUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def analyze_text(self):
        self.client.post(
            "/api/analyze/text",
            json={"text": "I'm feeling stressed", "publish_alerts": False}
        )
```

## Dataset Preparation Checklist

- [ ] Download RAVDESS dataset
- [ ] Download CREMA-D dataset (optional)
- [ ] Create custom dataset metadata CSV
- [ ] Preprocess all audio files
- [ ] Split into train/validation/test sets
- [ ] Augment dataset (if needed)
- [ ] Verify label distribution
- [ ] Create evaluation test set

## Evaluation Checklist

- [ ] Train model on training set
- [ ] Evaluate on validation set
- [ ] Tune hyperparameters
- [ ] Evaluate on test set
- [ ] Measure latency
- [ ] Test edge cases
- [ ] User acceptance testing
- [ ] Document results

