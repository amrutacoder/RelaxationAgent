# Relaxation Agent - System Implementation

## Overview

The Relaxation Agent is a microservice that analyzes user speech (text + audio features) to detect emotions, compute stress scores, and generate personalized coping prompts. It communicates with other agents (Voice, Route, UI) via Redis pub/sub and REST APIs.

## Architecture

```
┌─────────────┐
│ Voice Agent │ (STT → speech_text + audio_features)
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│     Relaxation Agent (This)         │
│  ┌──────────────┐  ┌──────────────┐ │
│  │ Preprocessor │  │ Emotion      │ │
│  │ (text+audio) │→ │ Classifier   │ │
│  └──────────────┘  │ (CNN-LSTM)   │ │
│                    └──────┬───────┘ │
│                           │         │
│  ┌──────────────┐  ┌──────▼───────┐ │
│  │ Stress       │← │ Coping Prompt│ │
│  │ Scorer       │  │ Generator    │ │
│  └──────┬───────┘  └──────────────┘ │
│         │                            │
│  ┌──────▼───────┐  ┌──────────────┐ │
│  │ Communicator │  │ Logger/DB    │ │
│  │ (Redis+REST) │  │ (SQLite)     │ │
│  └──────┬───────┘  └──────────────┘ │
└─────────┼────────────────────────────┘
          │
          ▼
    ┌─────────────┐
    │ Redis       │ (pub/sub channels)
    │ SQLite      │ (persistent logs)
    └─────────────┘
          │
          ▼
    ┌─────────────┐
    │ Route Agent │
    │ UI Agent    │
    └─────────────┘
```

## Components

### Core Services
- **Preprocessor**: Extracts text features and audio features (MFCC, pitch)
- **Emotion Classifier**: CNN-LSTM model for emotion detection
- **Stress Scorer**: Computes stress score from emotion + context
- **Coping Prompt Generator**: Rule-based + optional LLM for personalized prompts
- **Communicator**: Publishes stress alerts to Redis, handles REST callbacks
- **Logger/DB**: SQLite for logs, Redis for shared state

### Storage
- **Redis**: Pub/sub channels + short-term key/value
- **SQLite**: Persistent logs and emotion history

### API
- **FastAPI**: REST endpoints for inference
- **Redis Pub/Sub**: Real-time stress signal broadcasting

## Milestones

- [x] **Milestone A**: Text prototype
- [ ] **Milestone B**: Audio feature integration
- [ ] **Milestone C**: Model training/inference pipeline
- [ ] **Milestone D**: Agent API + comms
- [ ] **Milestone E**: Voice + Route + UI integration
- [ ] **Milestone F**: Polishing & logging

## Quick Start

### Prerequisites
- Python 3.9+
- Redis server
- (Optional) CUDA for GPU training

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your Redis and other configs

# Run Redis (if not running)
redis-server

# Start the Relaxation Agent API
python -m src.api.main
```

### Testing Milestone A (Text Prototype)

```bash
# Test text-based emotion detection
python -m src.milestone_a.text_prototype
```

## Project Structure

```
relaxation-agent/
├── src/
│   ├── milestone_a/          # Text prototype
│   ├── milestone_b/          # Audio features
│   ├── milestone_c/          # Model training
│   ├── milestone_d/          # API + Redis
│   ├── core/                 # Shared core modules
│   │   ├── preprocessor.py
│   │   ├── emotion_classifier.py
│   │   ├── stress_scorer.py
│   │   ├── prompt_generator.py
│   │   ├── communicator.py
│   │   └── logger.py
│   └── api/                  # FastAPI application
├── models/                   # Trained model files
├── data/                     # Datasets and processed data
├── tests/                    # Test suites
├── docs/                     # Documentation
├── requirements.txt
├── .env.example
└── README.md
```

## Documentation

- **[Architecture Alignment](docs/ARCHITECTURE_ALIGNMENT.md)** - Mapping current implementation to 10-stage architecture design
- **[Input/Output Guide](docs/INPUT_OUTPUT_GUIDE.md)** - Complete guide to system inputs, outputs, and data flow
- **[Development Guide](docs/DEVELOPMENT.md)** - Detailed development notes and workflow
- **[Integration Guide](docs/INTEGRATION.md)** - Integration with Voice Agent, Route Agent, and Flutter UI
- **[Testing Plan](docs/TESTING.md)** - Testing strategy and evaluation metrics

