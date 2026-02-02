# Input/Output Guide - Relaxation Agent

This document explains the inputs, outputs, and data flow for the Relaxation Agent system.

## Table of Contents

1. [Overview](#overview)
2. [Milestone A: Text Prototype](#milestone-a-text-prototype)
3. [Milestone B: Audio Features](#milestone-b-audio-features)
4. [Milestone C: Model Inference](#milestone-c-model-inference)
5. [Milestone D: API Endpoints](#milestone-d-api-endpoints)
6. [Complete Workflow Examples](#complete-workflow-examples)

---

## Overview

The Relaxation Agent processes user input (text and/or audio) to:
1. **Detect emotions** from speech/text
2. **Calculate stress scores** based on detected emotions
3. **Generate coping prompts** personalized to the user's stress level
4. **Publish alerts** to other agents via Redis
5. **Log results** to database

### High-Level Flow

```
Input (Text/Audio) 
    ↓
Preprocessing (Tokenization, Feature Extraction)
    ↓
Emotion Classification (CNN-LSTM or Rule-based)
    ↓
Stress Scoring (0.0 - 1.0)
    ↓
Prompt Generation (Rule-based or LLM)
    ↓
Output (JSON Response + Redis Pub/Sub + Database Log)
```

---

## Milestone A: Text Prototype

### Input

**Type:** Plain text string (simulated STT output)

**Example Input:**
```python
text = "I'm feeling really stressed and anxious about my upcoming exam tomorrow."
```

### Processing Steps

1. **Text Preprocessing**
   - Clean text (remove URLs, special chars)
   - Tokenize and lemmatize
   - Extract features (word count, negative/positive words, punctuation)

2. **Emotion Classification**
   - Rule-based keyword matching
   - Returns probability distribution over emotions

3. **Stress Scoring**
   - Maps emotions to stress values
   - Considers text features (negative words increase stress)
   - Outputs: stress_score (0.0-1.0) and stress_level ("low"/"medium"/"high")

4. **Prompt Generation**
   - Selects appropriate coping prompt based on stress level and emotion

### Output

**Type:** Python dictionary / JSON

**Example Output:**
```json
{
  "input_text": "I'm feeling really stressed and anxious about my upcoming exam tomorrow.",
  "emotion": {
    "top_emotion": "anxious",
    "probability": 0.75,
    "all_emotions": {
      "neutral": 0.05,
      "happy": 0.02,
      "sad": 0.03,
      "angry": 0.05,
      "fearful": 0.05,
      "disgusted": 0.02,
      "surprised": 0.03,
      "calm": 0.00,
      "anxious": 0.75,
      "stressed": 0.00
    }
  },
  "stress": {
    "stress_score": 0.82,
    "stress_level": "high",
    "emotion_breakdown": {
      "anxious": 0.75,
      "stressed": 0.00,
      ...
    },
    "thresholds": {
      "high": 0.7,
      "medium": 0.4
    }
  },
  "coping_prompt": {
    "prompt": "Take a deep breath. Inhale for 4 counts, hold for 4, exhale for 4.",
    "type": "rule_based",
    "stress_level": "high",
    "emotion": "anxious",
    "stress_score": 0.82
  },
  "user_id": null
}
```

### Code Example

```python
from src.milestone_a.text_prototype import RelaxationAgentPipeline

# Initialize pipeline
pipeline = RelaxationAgentPipeline(enable_logging=True)

# Process text
result = pipeline.process_text(
    text="I'm feeling really stressed and anxious about my exam.",
    user_id="user_123",
    publish_alerts=True
)

# Access results
print(f"Emotion: {result['emotion']['top_emotion']}")
print(f"Stress Score: {result['stress']['stress_score']}")
print(f"Stress Level: {result['stress']['stress_level']}")
print(f"Prompt: {result['coping_prompt']['prompt']}")
```

---

## Milestone B: Audio Features

### Input

**Type:** Audio file path or audio array

**Supported Formats:** WAV, MP3, FLAC (via librosa)

**Example Input:**
```python
audio_path = "data/audio/stressed_speech.wav"
# OR
audio_array = np.array([...])  # Raw audio samples
sample_rate = 22050
```

### Processing Steps

1. **Audio Loading**
   - Load audio file using librosa
   - Resample to 22050 Hz if needed

2. **Feature Extraction**
   - **MFCC** (Mel-frequency cepstral coefficients): 13 features
   - **Pitch** (Fundamental frequency)
   - **Spectral Centroid**: Brightness of sound
   - **Spectral Rolloff**: Frequency below which 85% of energy is contained
   - **Zero Crossing Rate**: Rate of sign changes
   - **Tempo**: Beats per minute
   - **Duration**: Length of audio

### Output

**Type:** Dictionary with numpy arrays

**Example Output:**
```python
{
  "mfcc": array([-5.23, -4.12, -3.45, ..., -2.11], dtype=float32),  # 13 values
  "pitch": array([185.5], dtype=float32),  # Mean pitch
  "spectral_centroid": 2345.67,
  "spectral_rolloff": 4567.89,
  "zero_crossing_rate": 0.123,
  "tempo": array([120.5], dtype=float32),
  "duration": 3.45  # seconds
}
```

### Code Example

```python
from src.milestone_b.audio_features import AudioFeatureExtractor

# Initialize extractor
extractor = AudioFeatureExtractor(sample_rate=22050, n_mfcc=13)

# Extract features from file
features = extractor.extract_from_file("audio.wav")

# Or from array
features = extractor.extract_from_array(audio_array, sample_rate=22050)

# Prepare for model input
model_input = extractor.prepare_for_model(features, sequence_length=100)
# Returns: numpy array of shape (100, 13) ready for CNN-LSTM
```

---

## Milestone C: Model Inference

### Input

**Type:** Preprocessed audio features (MFCC sequence)

**Shape:** `(sequence_length, n_mfcc)` - typically `(100, 13)`

**Example Input:**
```python
audio_features = {
    "mfcc": np.array([[...], [...], ...]),  # Shape: (100, 13)
    "pitch": np.array([185.5]),
    "zero_crossing_rate": 0.123,
    ...
}
```

### Processing Steps

1. **Model Loading**
   - Load trained CNN-LSTM model from file
   - Set to evaluation mode

2. **Inference**
   - Pass features through CNN layers (feature extraction)
   - Pass through LSTM layers (temporal modeling)
   - Get emotion probabilities from final fully connected layer

### Output

**Type:** Dictionary mapping emotions to probabilities

**Example Output:**
```python
{
  "neutral": 0.05,
  "happy": 0.02,
  "sad": 0.08,
  "angry": 0.15,
  "fearful": 0.10,
  "disgusted": 0.03,
  "surprised": 0.02,
  "calm": 0.05,
  "anxious": 0.40,
  "stressed": 0.20
}
```

### Code Example

```python
from src.core.emotion_classifier import EmotionClassifier

# Load trained model
classifier = EmotionClassifier(model_path="./models/emotion_classifier.pt")

# Predict from audio features
emotion_probs = classifier.predict_from_audio(audio_features)

# Get top emotion
top_emotion, top_prob = classifier.get_top_emotion(emotion_probs)
print(f"Top emotion: {top_emotion} ({top_prob:.2%})")
```

---

## Milestone D: API Endpoints

### Endpoint 1: Text Analysis

**URL:** `POST /api/analyze/text`

**Input (Request Body):**
```json
{
  "text": "I'm feeling really stressed and anxious about my exam.",
  "user_id": "user_123",
  "publish_alerts": true
}
```

**Output (Response):**
```json
{
  "input_text": "I'm feeling really stressed and anxious about my exam.",
  "emotion": {
    "top_emotion": "anxious",
    "probability": 0.75,
    "all_emotions": {
      "neutral": 0.05,
      "happy": 0.02,
      "sad": 0.03,
      "angry": 0.05,
      "fearful": 0.05,
      "disgusted": 0.02,
      "surprised": 0.03,
      "calm": 0.00,
      "anxious": 0.75,
      "stressed": 0.00
    }
  },
  "stress": {
    "stress_score": 0.82,
    "stress_level": "high",
    "emotion_breakdown": {...},
    "thresholds": {
      "high": 0.7,
      "medium": 0.4
    }
  },
  "coping_prompt": {
    "prompt": "Take a deep breath. Inhale for 4 counts, hold for 4, exhale for 4.",
    "type": "rule_based",
    "stress_level": "high",
    "emotion": "anxious",
    "stress_score": 0.82
  },
  "user_id": "user_123"
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/analyze/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I am feeling stressed",
    "user_id": "user_123",
    "publish_alerts": true
  }'
```

---

### Endpoint 2: Audio Analysis

**URL:** `POST /api/analyze/audio`

**Input (Multipart Form Data):**
```
audio_file: <binary audio data>
user_id: "user_123" (optional)
publish_alerts: true (optional)
```

**Output:** Same structure as text analysis endpoint

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/analyze/audio \
  -F "audio_file=@audio.wav" \
  -F "user_id=user_123" \
  -F "publish_alerts=true"
```

---

### Endpoint 3: Combined Analysis

**URL:** `POST /api/analyze/combined`

**Input (Multipart Form Data):**
```
text: "I'm feeling stressed"
audio_file: <binary audio data> (optional)
user_id: "user_123" (optional)
publish_alerts: true (optional)
```

**Output:** Same structure, but combines text and audio emotion predictions (weighted: 60% audio, 40% text)

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/analyze/combined \
  -F "text=I'm feeling stressed" \
  -F "audio_file=@audio.wav" \
  -F "user_id=user_123"
```

---

### Endpoint 4: Health Check

**URL:** `GET /health`

**Input:** None

**Output:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "redis_connected": true
}
```

---

### Endpoint 5: Emotion History

**URL:** `GET /api/history?user_id=user_123&limit=100`

**Input:** Query parameters
- `user_id` (optional): Filter by user
- `limit` (optional): Number of records (default: 100)

**Output:**
```json
{
  "history": [
    {
      "id": 1,
      "timestamp": "2024-01-15T10:30:00Z",
      "user_id": "user_123",
      "text_input": "I'm feeling stressed",
      "emotion_probs": "{\"anxious\": 0.75, ...}",
      "top_emotion": "anxious",
      "stress_score": 0.82,
      "stress_level": "high",
      "prompt": "Take a deep breath...",
      ...
    },
    ...
  ],
  "count": 25
}
```

---

## Complete Workflow Examples

### Example 1: High Stress Detection

**Input:**
```
Text: "I'm extremely anxious and can't stop worrying about everything!"
```

**Processing:**
1. Preprocessor detects: negative words (anxious, worrying), high word count
2. Emotion classifier: 85% anxious, 10% stressed
3. Stress scorer: 0.88 (high stress)
4. Prompt generator: Selects high-stress anxious prompt

**Output:**
```json
{
  "emotion": {"top_emotion": "anxious", "probability": 0.85},
  "stress": {"stress_score": 0.88, "stress_level": "high"},
  "coping_prompt": {
    "prompt": "Try the 5-4-3-2-1 grounding technique: Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste."
  }
}
```

**Redis Pub/Sub:**
```json
{
  "channel": "stress:alerts",
  "message": {
    "stress_score": 0.88,
    "stress_level": "high",
    "emotion": "anxious",
    "user_id": "user_123",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

**Database Log:**
- Entry saved to `emotion_logs` table
- Entry saved to `stress_alerts` table (because stress_level = "high")

---

### Example 2: Low Stress / Calm State

**Input:**
```
Text: "I'm feeling calm and peaceful today. Everything is going well."
```

**Processing:**
1. Preprocessor detects: positive words (calm, peaceful, well)
2. Emotion classifier: 70% calm, 20% happy, 10% neutral
3. Stress scorer: 0.25 (low stress)
4. Prompt generator: Selects low-stress encouragement prompt

**Output:**
```json
{
  "emotion": {"top_emotion": "calm", "probability": 0.70},
  "stress": {"stress_score": 0.25, "stress_level": "low"},
  "coping_prompt": {
    "prompt": "Great job managing your stress! Keep up the good work."
  }
}
```

**Redis Pub/Sub:**
- No stress alert published (stress_level = "low")
- Emotion update published to `emotion:updates` channel

**Database Log:**
- Entry saved to `emotion_logs` table only

---

### Example 3: Combined Text + Audio

**Input:**
```
Text: "I'm okay" (but voice sounds stressed)
Audio: High pitch, fast speech, high zero-crossing rate
```

**Processing:**
1. Text analysis: 60% neutral (text says "okay")
2. Audio analysis: 80% anxious (voice features indicate stress)
3. Combined: 60% * 0.80 (audio) + 40% * 0.60 (text) = 0.72 anxious
4. Stress scorer: 0.75 (high stress - audio reveals true state)
5. Prompt generator: High-stress prompt

**Output:**
```json
{
  "emotion": {"top_emotion": "anxious", "probability": 0.72},
  "stress": {"stress_score": 0.75, "stress_level": "high"},
  "coping_prompt": {
    "prompt": "Take a deep breath. Inhale for 4 counts, hold for 4, exhale for 4."
  }
}
```

**Key Insight:** Audio features can reveal stress even when text doesn't express it!

---

## Data Types Reference

### Emotion Labels
```python
EMOTION_LABELS = [
    "neutral",    # No strong emotion
    "happy",      # Positive, joyful
    "sad",        # Negative, down
    "angry",      # Negative, frustrated
    "fearful",    # Negative, scared
    "disgusted",  # Negative, repulsed
    "surprised",  # Neutral/positive, unexpected
    "calm",       # Positive, relaxed
    "anxious",    # Negative, worried
    "stressed"    # Negative, overwhelmed
]
```

### Stress Levels
- **low**: stress_score < 0.4
- **medium**: 0.4 <= stress_score < 0.7
- **high**: stress_score >= 0.7

### Stress Score Range
- **0.0 - 0.3**: Very low stress (calm, happy states)
- **0.4 - 0.6**: Moderate stress (normal daily stress)
- **0.7 - 1.0**: High stress (needs intervention)

---

## Error Handling

### Invalid Input

**Input:**
```json
{
  "text": ""  // Empty string
}
```

**Output:**
```json
{
  "emotion": {
    "top_emotion": "neutral",
    "probability": 1.0
  },
  "stress": {
    "stress_score": 0.3,
    "stress_level": "low"
  }
}
```
*System defaults to neutral emotion and low stress for empty input*

### Missing Audio File

**Input:**
```bash
curl -X POST /api/analyze/audio
# No audio_file provided
```

**Output:**
```json
{
  "detail": "Missing required field: audio_file"
}
```
*HTTP 422 Unprocessable Entity*

### Redis Connection Failure

**Behavior:**
- System continues processing
- Results still returned to API
- Database logging still works
- Redis pub/sub skipped (no error thrown)
- Warning logged: "Could not connect to Redis"

---

## Integration Points

### Voice Agent → Relaxation Agent

**Input Format:**
```json
{
  "text": "Transcribed speech text",
  "audio_file": "<base64 or binary>",
  "user_id": "user_123",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Relaxation Agent → Route Agent

**REST Callback:**
```json
POST http://localhost:8002/api/stress-update
{
  "stress_level": "high",
  "emotion": "anxious",
  "user_id": "user_123",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Relaxation Agent → UI Agent

**REST Callback:**
```json
POST http://localhost:8003/api/relaxation-update
{
  "stress_score": 0.82,
  "stress_level": "high",
  "emotion": "anxious",
  "prompt": "Take a deep breath...",
  "user_id": "user_123",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## Summary

| Component | Input | Output |
|-----------|-------|--------|
| **Text Preprocessor** | Raw text string | Tokens + features array |
| **Audio Preprocessor** | Audio file/array | MFCC + pitch + spectral features |
| **Emotion Classifier** | Text or audio features | Emotion probabilities (dict) |
| **Stress Scorer** | Emotion probabilities | Stress score (0-1) + level (low/med/high) |
| **Prompt Generator** | Stress level + emotion | Coping prompt text |
| **API Endpoint** | HTTP request (JSON/form) | HTTP response (JSON) |
| **Redis Pub/Sub** | Stress alerts | Published messages |
| **Database Logger** | Analysis results | SQLite records |

---

For more details, see:
- [Development Guide](DEVELOPMENT.md)
- [Integration Guide](INTEGRATION.md)
- [Testing Plan](TESTING.md)

