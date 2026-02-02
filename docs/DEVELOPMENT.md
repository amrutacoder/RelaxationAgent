# Development Guide

## Setup

### Prerequisites
- Python 3.9+
- Redis server
- (Optional) CUDA-capable GPU for training

### Installation

```bash
# Clone repository
git clone <repo-url>
cd relaxation-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (if not auto-downloaded)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

### Running Redis

```bash
# On Linux/Mac
redis-server

# On Windows (if installed)
redis-server

# Or use Docker
docker run -d -p 6379:6379 redis:latest
```

## Development Workflow

### Milestone A: Text Prototype

Test the text-only prototype:

```bash
python -m src.milestone_a.text_prototype
```

This will run test cases and demonstrate the pipeline.

### Milestone B: Audio Features

Extract audio features from a file:

```python
from src.milestone_b.audio_features import AudioFeatureExtractor

extractor = AudioFeatureExtractor()
features = extractor.extract_from_file("path/to/audio.wav")
print(features)
```

### Milestone C: Model Training

Train the CNN-LSTM model:

```bash
# Using RAVDESS dataset
python -m src.milestone_c.train --data_dir ./data/ravdess --output_dir ./models --epochs 50 --device cpu

# Using custom dataset (CSV with file_path and emotion columns)
python -m src.milestone_c.train --data_dir ./data/custom --output_dir ./models --epochs 50
```

### Milestone D: API Server

Start the FastAPI server:

```bash
python -m src.api.main
```

Or use uvicorn directly:

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Test the API:

```bash
# Health check
curl http://localhost:8000/health

# Analyze text
curl -X POST http://localhost:8000/api/analyze/text \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling stressed", "publish_alerts": false}'
```

Open the UI mock:

```bash
# Open src/milestone_d/ui_mock.html in a browser
# Make sure the API is running on localhost:8000
```

## Testing

Run all tests:

```bash
pytest tests/ -v
```

Run specific test file:

```bash
pytest tests/test_milestone_a.py -v
```

Run with coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

## Code Structure

```
src/
├── core/              # Core modules (shared across milestones)
│   ├── config.py     # Configuration management
│   ├── preprocessor.py
│   ├── emotion_classifier.py
│   ├── stress_scorer.py
│   ├── prompt_generator.py
│   ├── communicator.py
│   └── logger.py
├── milestone_a/       # Text prototype
├── milestone_b/       # Audio features
├── milestone_c/       # Model training
├── milestone_d/       # API + Redis
└── api/              # FastAPI application
```

## Adding New Features

### Adding a New Emotion

1. Update `EMOTION_LABELS` in `src/core/emotion_classifier.py`
2. Add emotion keywords in `TextEmotionClassifier.emotion_keywords`
3. Add stress mapping in `StressScorer.emotion_stress_map`
4. Retrain model if using audio features

### Adding New Coping Prompts

Edit `src/core/prompt_generator.py` and add prompts to the `self.prompts` dictionary.

### Customizing Stress Thresholds

Edit `.env` file:
```
STRESS_THRESHOLD_HIGH=0.7
STRESS_THRESHOLD_MEDIUM=0.4
```

## Debugging

### Check Redis Connection

```python
from src.core.communicator import Communicator
comm = Communicator()
print(comm.redis_client.ping())  # Should return True
```

### View Database Logs

```python
from src.core.logger import Logger
logger = Logger()
history = logger.get_emotion_history(limit=10)
print(history)
```

### Enable Debug Logging

Set in `.env`:
```
LOG_LEVEL=DEBUG
```

## Performance Optimization

### For Production

1. Use GPU for model inference (if available)
2. Enable Redis connection pooling
3. Use async database operations
4. Add caching for frequent queries
5. Use production WSGI server (gunicorn)

### Model Optimization

- Quantize model for faster inference
- Use ONNX for cross-platform deployment
- Batch process multiple requests

## Troubleshooting

### Redis Connection Errors

- Ensure Redis is running: `redis-cli ping`
- Check host/port in `.env`
- Verify firewall settings

### Model Loading Errors

- Ensure model file exists at path in `.env`
- Check model file format (should be PyTorch .pt)
- Verify model architecture matches code

### Audio Processing Errors

- Install librosa: `pip install librosa soundfile`
- Check audio file format (WAV, MP3, etc.)
- Verify sample rate compatibility

