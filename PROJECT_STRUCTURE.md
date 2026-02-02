# Project Structure

## Directory Layout

```
relaxation-agent/
├── src/                          # Source code
│   ├── __init__.py
│   ├── core/                     # Core modules (shared)
│   │   ├── __init__.py
│   │   ├── config.py            # Configuration management
│   │   ├── preprocessor.py      # Text & audio preprocessing
│   │   ├── emotion_classifier.py # Emotion classification (CNN-LSTM + text fallback)
│   │   ├── stress_scorer.py     # Stress score computation
│   │   ├── prompt_generator.py  # Coping prompt generation
│   │   ├── communicator.py      # Redis pub/sub & REST callbacks
│   │   └── logger.py            # Database logging
│   │
│   ├── milestone_a/             # Milestone A: Text prototype
│   │   ├── __init__.py
│   │   └── text_prototype.py    # Text-only pipeline demo
│   │
│   ├── milestone_b/             # Milestone B: Audio features
│   │   ├── __init__.py
│   │   └── audio_features.py    # Audio feature extraction & dataset loading
│   │
│   ├── milestone_c/             # Milestone C: Model training
│   │   ├── __init__.py
│   │   └── train.py             # CNN-LSTM training script
│   │
│   ├── milestone_d/             # Milestone D: API & comms
│   │   ├── __init__.py
│   │   └── ui_mock.html         # Simple UI mock for testing
│   │
│   └── api/                     # FastAPI application
│       ├── __init__.py
│       └── main.py              # API endpoints
│
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── test_milestone_a.py     # Tests for text prototype
│   └── test_api.py              # API endpoint tests
│
├── docs/                        # Documentation
│   ├── DEVELOPMENT.md           # Development guide
│   ├── INTEGRATION.md           # Integration guide
│   └── TESTING.md               # Testing & evaluation plan
│
├── scripts/                     # Utility scripts
│   ├── setup.sh                # Setup script (Linux/Mac)
│   └── setup.bat               # Setup script (Windows)
│
├── data/                        # Data directory
│   ├── raw/                     # Raw datasets
│   └── processed/               # Processed features
│
├── models/                      # Trained models
│   └── emotion_classifier.pt   # Saved model (after training)
│
├── logs/                        # Log files
│   └── relaxation_agent.log
│
├── requirements.txt             # Python dependencies
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules
├── README.md                   # Main README
├── QUICKSTART.md               # Quick start guide
└── PROJECT_STRUCTURE.md        # This file
```

## Component Overview

### Core Modules (`src/core/`)

**config.py**
- Loads environment variables
- Provides configuration constants
- Centralized settings management

**preprocessor.py**
- `TextPreprocessor`: Cleans, tokenizes, and extracts text features
- `AudioPreprocessor`: Extracts MFCC, pitch, spectral features from audio

**emotion_classifier.py**
- `CNNLSTMEmotionClassifier`: PyTorch model for audio emotion classification
- `TextEmotionClassifier`: Rule-based text emotion classification
- `EmotionClassifier`: Main interface supporting both text and audio

**stress_scorer.py**
- `StressScorer`: Computes stress scores from emotion probabilities
- Maps emotions to stress levels (low/medium/high)

**prompt_generator.py**
- `CopingPromptGenerator`: Generates personalized coping prompts
- Rule-based prompts (default)
- Optional LLM integration (async)

**communicator.py**
- `Communicator`: Handles Redis pub/sub and REST callbacks
- Publishes stress alerts and emotion updates
- Notifies Route Agent and UI Agent

**logger.py**
- `Logger`: Database logging and history
- SQLite for persistent storage
- File-based logging

### Milestone Modules

**milestone_a/text_prototype.py**
- `RelaxationAgentPipeline`: Main processing pipeline
- Text-only emotion detection and stress scoring
- Demonstrates full workflow

**milestone_b/audio_features.py**
- `AudioFeatureExtractor`: Extracts audio features
- `DatasetLoader`: Loads and preprocesses emotion datasets
- Supports RAVDESS and custom datasets

**milestone_c/train.py**
- Training script for CNN-LSTM model
- Handles data loading, training loop, validation
- Saves trained model and training history

**milestone_d/ui_mock.html**
- Simple HTML/JavaScript UI for testing
- Connects to API and displays results
- Shows Redis connection status

### API (`src/api/main.py`)

FastAPI application with endpoints:
- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /api/analyze/text` - Text analysis
- `POST /api/analyze/audio` - Audio analysis
- `POST /api/analyze/combined` - Combined text + audio
- `GET /api/history` - Emotion history
- `GET /api/stress-alerts` - Stress alert history

## Data Flow

```
Input (Text/Audio)
    ↓
Preprocessor
    ↓
Emotion Classifier
    ↓
Stress Scorer
    ↓
Prompt Generator
    ↓
Logger + Communicator
    ↓
Output (JSON Response + Redis Pub/Sub)
```

## File Naming Conventions

- **Modules**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- `fastapi`, `uvicorn` - API framework
- `redis` - Redis client
- `torch` - PyTorch for ML models
- `librosa` - Audio processing
- `nltk` - Text processing
- `sqlalchemy`, `aiosqlite` - Database

## Configuration

All configuration via environment variables (`.env`):
- Redis settings
- API settings
- Model paths
- Stress thresholds
- Agent URLs

## Testing

- Unit tests in `tests/`
- Integration tests for API endpoints
- Test data in `tests/fixtures/` (if needed)

## Documentation

- `README.md` - Overview and architecture
- `QUICKSTART.md` - Quick setup guide
- `docs/DEVELOPMENT.md` - Development workflow
- `docs/INTEGRATION.md` - Integration guide
- `docs/TESTING.md` - Testing plan

