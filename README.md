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


