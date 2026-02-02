# Quick Start Guide

## Prerequisites

- Python 3.9+
- Redis server
- (Optional) CUDA for GPU training

## Installation

### Windows

```cmd
scripts\setup.bat
```

### Linux/Mac

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### Manual Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Create directories
mkdir -p data models logs

# Copy environment file
cp .env.example .env
# Edit .env with your settings
```

## Start Redis

### Windows
```cmd
redis-server
```

### Linux/Mac
```bash
redis-server
```

### Docker
```bash
docker run -d -p 6379:6379 redis:latest
```

## Test Milestone A (Text Prototype)

```bash
python -m src.milestone_a.text_prototype
```

This will run test cases and show:
- Emotion detection
- Stress scoring
- Coping prompt generation

## Start API Server

```bash
python -m src.api.main
```

Or with uvicorn:

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: http://localhost:8000

## Test API

### Health Check
```bash
curl http://localhost:8000/health
```

### Analyze Text
```bash
curl -X POST http://localhost:8000/api/analyze/text \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling stressed", "publish_alerts": false}'
```

### Open UI Mock

Open `src/milestone_d/ui_mock.html` in your browser (make sure API is running).

## Run Tests

```bash
pytest tests/ -v
```

## Next Steps

1. **Milestone B**: Add audio feature extraction
2. **Milestone C**: Train CNN-LSTM model on emotion dataset
3. **Milestone D**: Test Redis pub/sub and agent callbacks
4. **Milestone E**: Integrate with Voice Agent and Flutter UI
5. **Milestone F**: Add logging, history, and polish

## Troubleshooting

### Redis Connection Error

- Ensure Redis is running: `redis-cli ping`
- Check `.env` file for correct host/port
- Verify firewall settings

### Import Errors

- Ensure virtual environment is activated
- Run `pip install -r requirements.txt` again
- Check Python version (3.9+)

### NLTK Data Missing

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Documentation

- [Development Guide](docs/DEVELOPMENT.md)
- [Integration Guide](docs/INTEGRATION.md)
- [Testing Plan](docs/TESTING.md)

