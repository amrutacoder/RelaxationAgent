# Multi-Agent Integration Guide

Complete guide for integrating Relaxation Agent with Voice Agent, Route Agent, and UI Agent.

## Architecture Overview

```
┌─────────────┐
│ Voice Agent │
│  (STT+TTS)  │
└──────┬──────┘
       │ POST /api/agents/voice/analyze
       │ {text, acoustic_features}
       ▼
┌─────────────────────┐
│ Relaxation Agent    │
│  (This Service)      │
└──────┬──────────────┘
       │
       ├──► POST /api/agents/route/calculate
       │    Route Agent
       │
       └──► POST /api/agents/ui/update
            UI Agent
```

---

## A. Voice Agent Integration

### Input from Voice Agent

**Endpoint:** `POST /api/agents/voice/analyze`

**Request:**
```json
{
  "text": "I'm feeling really stressed",
  "acoustic_features": {
    "mfcc": [T × 13],
    "pitch": [T],
    "energy": [T],
    "speech_rate": [T]
  },
  "user_id": "user123"
}
```

### Output to Voice Agent

**Response:**
```json
{
  "text": "Let's slow down together. Take a gentle breath in.",
  "voice_style": "calm_slow",
  "stress_score": 0.82,
  "emotion": "anxious"
}
```

**Voice Styles:**
- `calm_slow` - High stress, anxious (slow, low pitch)
- `calm_normal` - Low stress (normal pace)
- `supportive_gentle` - High stress, angry (gentle, warm)
- `supportive_warm` - Medium stress (warm, friendly)
- `urgent_calm` - High stress, distracted (clear, slightly slow)
- `reassuring` - Fearful emotions (reassuring tone)
- `neutral` - Default

**Example cURL:**
```bash
curl -X POST http://localhost:8000/api/agents/voice/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I am feeling overwhelmed",
    "acoustic_features": {
      "mfcc": [[...], [...]],
      "pitch": [185.5, 190.2, ...],
      "energy": [0.12, 0.15, ...],
      "speech_rate": [0.08, 0.09, ...]
    },
    "user_id": "user123"
  }'
```

**Voice Agent Implementation (Python):**
```python
import requests

def send_to_relaxation_agent(text, acoustic_features, user_id):
    """Send speech data to Relaxation Agent."""
    response = requests.post(
        "http://localhost:8000/api/agents/voice/analyze",
        json={
            "text": text,
            "acoustic_features": acoustic_features,
            "user_id": user_id
        }
    )
    result = response.json()
    
    # Use result for TTS
    tts_text = result["text"]
    voice_style = result["voice_style"]
    stress_score = result["stress_score"]
    
    # Generate TTS with voice_style
    audio_output = generate_tts(tts_text, voice_style)
    
    return audio_output, stress_score
```

---

## B. Relaxation Agent Output

### Standard Output Format

**All analysis endpoints now include `voice_output`:**

```json
{
  "emotion": {...},
  "stress": {...},
  "coping_prompt": {...},
  "voice_output": {
    "text": "Let's slow down together. Take a gentle breath in.",
    "voice_style": "calm_slow",
    "style_properties": {
      "pitch": "low",
      "speed": "slow",
      "energy": "soft",
      "prosody": "gentle",
      "pause_duration": "long"
    }
  }
}
```

### Endpoints with Voice Output

1. `POST /api/analyze/text` - Original endpoint (updated)
2. `POST /api/v2/analyze/enhanced` - Enhanced endpoint
3. `POST /api/agents/voice/analyze` - Voice Agent endpoint

---

## C. Route Agent Integration

### Input from Route Agent

**Endpoint:** `POST /api/agents/route/calculate`

**Request:**
```json
{
  "stress_score": 0.82,
  "user_id": "user123",
  "source": {"lat": 40.7128, "lng": -74.0060},
  "destination": {"lat": 40.7580, "lng": -73.9855}
}
```

### Output to Route Agent

**Response:**
```json
{
  "route_coordinates": [
    {"lat": 40.7128, "lng": -74.0060},
    {"lat": 40.7354, "lng": -73.9958},
    {"lat": 40.7580, "lng": -73.9855}
  ],
  "route_id": "route_user123_82",
  "stress_consideration": "High stress detected - selecting calmer route with less traffic"
}
```

**Route Selection Logic:**
- **High stress (≥0.7)**: Calmer routes (less traffic, scenic)
- **Medium stress (0.4-0.7)**: Balanced routes
- **Low stress (<0.4)**: Optimal routes (fastest)

**Example cURL:**
```bash
curl -X POST http://localhost:8000/api/agents/route/calculate \
  -H "Content-Type: application/json" \
  -d '{
    "stress_score": 0.82,
    "user_id": "user123",
    "source": {"lat": 40.7128, "lng": -74.0060},
    "destination": {"lat": 40.7580, "lng": -73.9855}
  }'
```

**Route Agent Implementation:**
```python
def get_route_from_relaxation_agent(stress_score, user_id, source, destination):
    """Get route recommendation from Relaxation Agent."""
    response = requests.post(
        "http://localhost:8000/api/agents/route/calculate",
        json={
            "stress_score": stress_score,
            "user_id": user_id,
            "source": source,
            "destination": destination
        }
    )
    return response.json()
```

---

## D. UI Agent Integration

### Input from UI Agent

**Endpoint:** `POST /api/agents/ui/update`

**Request:**
```json
{
  "stress_score": 0.82,
  "user_id": "user123",
  "route_coordinates": [
    {"lat": 40.7128, "lng": -74.0060},
    {"lat": 40.7580, "lng": -73.9855}
  ]
}
```

### Output to UI Agent

**Response:**
```json
{
  "display_data": {
    "stress_score": 0.82,
    "stress_level": "high",
    "color": "red",
    "route_coordinates": [...],
    "user_profile": {
      "profile_type": "ADHD",
      "threshold": 55.0
    }
  },
  "recommendations": [
    "Take deep breaths",
    "Consider taking a break",
    "Use coping strategies"
  ]
}
```

**Example cURL:**
```bash
curl -X POST http://localhost:8000/api/agents/ui/update \
  -H "Content-Type: application/json" \
  -d '{
    "stress_score": 0.82,
    "user_id": "user123",
    "route_coordinates": [...]
  }'
```

**UI Agent Implementation (Flutter/Dart):**
```dart
Future<Map<String, dynamic>> updateUI(
  double stressScore,
  String userId,
  List<Map<String, double>> routeCoordinates
) async {
  final response = await http.post(
    Uri.parse('http://localhost:8000/api/agents/ui/update'),
    headers: {'Content-Type': 'application/json'},
    body: json.encode({
      'stress_score': stressScore,
      'user_id': userId,
      'route_coordinates': routeCoordinates,
    }),
  );
  
  return json.decode(response.body);
}
```

---

## Complete Workflow Example

### End-to-End Flow

1. **User speaks** → Voice Agent captures audio
2. **Voice Agent** → STT + Feature extraction
3. **Voice Agent** → `POST /api/agents/voice/analyze`
4. **Relaxation Agent** → Processes, returns `{text, voice_style, stress_score}`
5. **Voice Agent** → TTS with voice_style → Audio to user
6. **Relaxation Agent** → Publishes stress alert to Redis
7. **Route Agent** → Subscribes to Redis, gets stress score
8. **Route Agent** → `POST /api/agents/route/calculate`
9. **Relaxation Agent** → Returns route coordinates
10. **Route Agent** → Sends route to UI Agent
11. **UI Agent** → `POST /api/agents/ui/update`
12. **Relaxation Agent** → Returns display data
13. **UI Agent** → Renders map with route

### Code Example (Python)

```python
# Voice Agent
voice_result = requests.post(
    "http://localhost:8000/api/agents/voice/analyze",
    json={"text": "I'm stressed", "user_id": "user123"}
).json()

stress_score = voice_result["stress_score"]
voice_style = voice_result["voice_style"]
tts_text = voice_result["text"]

# Route Agent (after receiving stress from Redis)
route_result = requests.post(
    "http://localhost:8000/api/agents/route/calculate",
    json={
        "stress_score": stress_score,
        "user_id": "user123",
        "source": {"lat": 40.7128, "lng": -74.0060},
        "destination": {"lat": 40.7580, "lng": -73.9855}
    }
).json()

route_coords = route_result["route_coordinates"]

# UI Agent
ui_result = requests.post(
    "http://localhost:8000/api/agents/ui/update",
    json={
        "stress_score": stress_score,
        "user_id": "user123",
        "route_coordinates": route_coords
    }
).json()

display_data = ui_result["display_data"]
recommendations = ui_result["recommendations"]
```

---

## Voice Style Properties

Each voice style has detailed TTS properties:

| Style | Pitch | Speed | Energy | Prosody | Pause |
|-------|-------|-------|--------|---------|-------|
| calm_slow | low | slow (0.7x) | soft | gentle | long |
| calm_normal | medium | normal | moderate | smooth | normal |
| supportive_gentle | medium_low | slow (0.8x) | soft | warm | medium |
| supportive_warm | medium | normal | warm | friendly | normal |
| urgent_calm | medium | slightly_slow (0.9x) | calm | clear | short |
| reassuring | medium_low | normal | gentle | reassuring | medium |
| neutral | medium | normal | neutral | neutral | normal |

---

## Error Handling

All endpoints return standard HTTP status codes:

- `200 OK` - Success
- `400 Bad Request` - Invalid input
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error

**Error Response Format:**
```json
{
  "detail": "Error message here"
}
```

---

## Testing

### Test Voice Agent Integration

```bash
# Test with text only
curl -X POST http://localhost:8000/api/agents/voice/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I am anxious", "user_id": "test"}'

# Test with acoustic features
curl -X POST http://localhost:8000/api/agents/voice/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I am anxious",
    "acoustic_features": {"mfcc": [[1,2,3]], "pitch": [180]},
    "user_id": "test"
  }'
```

### Test Route Agent Integration

```bash
curl -X POST http://localhost:8000/api/agents/route/calculate \
  -H "Content-Type: application/json" \
  -d '{
    "stress_score": 0.75,
    "user_id": "test",
    "source": {"lat": 40.7128, "lng": -74.0060},
    "destination": {"lat": 40.7580, "lng": -73.9855}
  }'
```

### Test UI Agent Integration

```bash
curl -X POST http://localhost:8000/api/agents/ui/update \
  -H "Content-Type: application/json" \
  -d '{
    "stress_score": 0.75,
    "user_id": "test",
    "route_coordinates": [{"lat": 40.7128, "lng": -74.0060}]
  }'
```

---

## Configuration

Update `.env` for agent URLs:

```env
# Voice Agent
VOICE_AGENT_URL=http://localhost:8001

# Route Agent
ROUTE_AGENT_URL=http://localhost:8002

# UI Agent
UI_AGENT_URL=http://localhost:8003
```

---

## Summary

| Agent | Input | Output |
|-------|-------|--------|
| **Voice Agent** | text + acoustic_features | text + voice_style + stress_score |
| **Route Agent** | stress_score + coords | route_coordinates + route_id |
| **UI Agent** | stress_score + route_coords | display_data + recommendations |

All integrations are RESTful and can be tested independently.

