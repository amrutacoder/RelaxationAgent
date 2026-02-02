# Integration Guide

## Overview

This document describes how to integrate the Relaxation Agent with other services:
- Voice Agent (STT)
- Route Agent
- UI Agent (Flutter)

## Architecture

```
┌─────────────┐
│ Voice Agent │
│  (STT)      │
└──────┬──────┘
       │ HTTP POST /api/analyze/combined
       │ (text + audio)
       ▼
┌─────────────────────┐
│ Relaxation Agent    │
│  (This Service)     │
└──────┬──────────────┘
       │
       ├──► Redis Pub/Sub
       │    (stress:alerts, emotion:updates)
       │
       ├──► HTTP POST (Route Agent)
       │    /api/stress-update
       │
       └──► HTTP POST (UI Agent)
            /api/relaxation-update
```

## Integration Points

### 1. Voice Agent Integration

The Voice Agent should send both transcribed text and audio features to the Relaxation Agent.

#### Endpoint: `/api/analyze/combined`

**Request:**
```http
POST /api/analyze/combined
Content-Type: multipart/form-data

text: "I'm feeling really stressed"
audio_file: <binary audio data>
user_id: "user_123"
publish_alerts: true
```

**Response:**
```json
{
  "input_text": "I'm feeling really stressed",
  "emotion": {
    "top_emotion": "anxious",
    "probability": 0.85,
    "all_emotions": {
      "anxious": 0.85,
      "stressed": 0.10,
      "neutral": 0.05
    }
  },
  "stress": {
    "stress_score": 0.82,
    "stress_level": "high",
    "thresholds": {
      "high": 0.7,
      "medium": 0.4
    }
  },
  "coping_prompt": {
    "prompt": "Take a deep breath. Inhale for 4 counts, hold for 4, exhale for 4.",
    "type": "rule_based",
    "stress_level": "high",
    "emotion": "anxious"
  },
  "user_id": "user_123"
}
```

#### Example Voice Agent Code (Python)

```python
import httpx
import requests

async def send_to_relaxation_agent(text: str, audio_path: str, user_id: str):
    """Send STT result to Relaxation Agent."""
    
    url = "http://localhost:8000/api/analyze/combined"
    
    with open(audio_path, 'rb') as audio_file:
        files = {'audio_file': audio_file}
        data = {
            'text': text,
            'user_id': user_id,
            'publish_alerts': True
        }
        
        response = requests.post(url, files=files, data=data)
        return response.json()
```

### 2. Redis Pub/Sub Integration

The Relaxation Agent publishes stress alerts and emotion updates to Redis channels.

#### Channel: `stress:alerts`

**Message Format:**
```json
{
  "stress_score": 0.82,
  "stress_level": "high",
  "emotion": "anxious",
  "user_id": "user_123",
  "timestamp": "2024-01-15T10:30:00Z",
  "metadata": {}
}
```

#### Channel: `emotion:updates`

**Message Format:**
```json
{
  "emotions": {
    "anxious": 0.85,
    "stressed": 0.10,
    "neutral": 0.05
  },
  "top_emotion": "anxious",
  "user_id": "user_123",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Example Subscriber (Python)

```python
import redis
import json

def subscribe_to_stress_alerts():
    """Subscribe to stress alerts from Relaxation Agent."""
    
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    pubsub = r.pubsub()
    pubsub.subscribe('stress:alerts')
    
    for message in pubsub.listen():
        if message['type'] == 'message':
            data = json.loads(message['data'])
            print(f"Stress Alert: {data['stress_level']} ({data['stress_score']})")
            # Handle alert (e.g., notify Route Agent, update UI)
```

### 3. Route Agent Integration

The Relaxation Agent can send REST callbacks to the Route Agent when stress is detected.

#### Endpoint: `POST /api/stress-update`

**Request (from Relaxation Agent):**
```json
{
  "stress_level": "high",
  "emotion": "anxious",
  "user_id": "user_123",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Expected Response:**
```json
{
  "status": "received",
  "action_taken": "route_adjusted"
}
```

#### Route Agent Implementation Example

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class StressUpdate(BaseModel):
    stress_level: str
    emotion: str
    user_id: str
    timestamp: str

@app.post("/api/stress-update")
async def handle_stress_update(update: StressUpdate):
    """Handle stress update from Relaxation Agent."""
    
    if update.stress_level == "high":
        # Adjust route to avoid stressful areas
        # or suggest alternative routes
        pass
    
    return {"status": "received", "action_taken": "route_adjusted"}
```

### 4. UI Agent (Flutter) Integration

The Relaxation Agent sends updates to the UI Agent for display.

#### Endpoint: `POST /api/relaxation-update`

**Request (from Relaxation Agent):**
```json
{
  "stress_score": 0.82,
  "stress_level": "high",
  "emotion": "anxious",
  "prompt": "Take a deep breath...",
  "user_id": "user_123",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Flutter Integration Example

```dart
import 'package:http/http.dart' as http;
import 'dart:convert';

class RelaxationAgentService {
  final String baseUrl = 'http://localhost:8000';
  
  Future<Map<String, dynamic>> analyzeText(String text) async {
    final response = await http.post(
      Uri.parse('$baseUrl/api/analyze/text'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode({
        'text': text,
        'publish_alerts': true,
      }),
    );
    
    if (response.statusCode == 200) {
      return json.decode(response.body);
    } else {
      throw Exception('Failed to analyze text');
    }
  }
  
  Stream<Map<String, dynamic>> subscribeToStressAlerts() async* {
    // Connect to Redis via WebSocket or polling
    // For now, use polling as example
    while (true) {
      await Future.delayed(Duration(seconds: 2));
      // Poll for updates or use WebSocket
    }
  }
}
```

#### Flutter UI Example

```dart
import 'package:flutter/material.dart';

class RelaxationWidget extends StatefulWidget {
  @override
  _RelaxationWidgetState createState() => _RelaxationWidgetState();
}

class _RelaxationWidgetState extends State<RelaxationWidget> {
  final RelaxationAgentService _service = RelaxationAgentService();
  Map<String, dynamic>? _currentState;
  
  @override
  Widget build(BuildContext context) {
    return Card(
      child: Column(
        children: [
          if (_currentState != null) ...[
            Text('Emotion: ${_currentState!['emotion']['top_emotion']}'),
            Text('Stress: ${_currentState!['stress']['stress_level']}'),
            Text('${_currentState!['coping_prompt']['prompt']}'),
          ],
          ElevatedButton(
            onPressed: () async {
              final result = await _service.analyzeText('I feel stressed');
              setState(() => _currentState = result);
            },
            child: Text('Analyze'),
          ),
        ],
      ),
    );
  }
}
```

## Configuration

### Environment Variables

Set in `.env`:

```env
# Agent URLs
VOICE_AGENT_URL=http://localhost:8001
ROUTE_AGENT_URL=http://localhost:8002
UI_AGENT_URL=http://localhost:8003

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_CHANNEL_STRESS_ALERT=stress:alerts
REDIS_CHANNEL_EMOTION_UPDATE=emotion:updates
```

## Testing Integration

### Test Voice Agent Integration

```bash
# Start Relaxation Agent
python -m src.api.main

# In another terminal, test the endpoint
curl -X POST http://localhost:8000/api/analyze/combined \
  -F "text=I'm feeling stressed" \
  -F "audio_file=@test_audio.wav" \
  -F "user_id=test_user"
```

### Test Redis Pub/Sub

```bash
# Start Redis CLI
redis-cli

# Subscribe to stress alerts
SUBSCRIBE stress:alerts

# In another terminal, trigger an alert via API
curl -X POST http://localhost:8000/api/analyze/text \
  -H "Content-Type: application/json" \
  -d '{"text": "I am extremely anxious", "publish_alerts": true}'
```

### Test Route Agent Callback

```python
# Set ROUTE_AGENT_URL in .env
# Start a mock Route Agent server
# Trigger high stress via API
# Check Route Agent receives callback
```

## Error Handling

### Retry Logic

Implement retry logic for REST callbacks:

```python
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def notify_route_agent(stress_level, emotion, user_id):
    # Implementation with automatic retry
    pass
```

### Fallback Behavior

- If Redis is unavailable, log alerts to database only
- If Route Agent is unavailable, continue processing (non-blocking)
- If UI Agent is unavailable, alerts still published to Redis

## Security Considerations

1. **Authentication**: Add API keys or OAuth for production
2. **Rate Limiting**: Implement rate limiting on endpoints
3. **Input Validation**: Validate all inputs
4. **HTTPS**: Use HTTPS in production
5. **Redis Security**: Set Redis password in production

## Performance

- Use connection pooling for Redis
- Batch process multiple requests when possible
- Cache frequent queries
- Use async/await for I/O operations

## Monitoring

Monitor:
- API response times
- Redis connection status
- Error rates
- Stress alert frequency
- Agent callback success rates

## Troubleshooting

### Redis Connection Issues

```python
# Test Redis connection
from src.core.communicator import Communicator
comm = Communicator()
print(comm.redis_client.ping())
```

### Agent Callback Failures

Check logs for HTTP errors. Ensure:
- Agent URLs are correct
- Agents are running
- Network connectivity
- Firewall rules

### Message Not Received

- Verify Redis channel names match
- Check subscriber is actually listening
- Verify message format matches expected schema

