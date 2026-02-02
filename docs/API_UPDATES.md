# API Updates - Enhanced Features

This document describes the new API endpoints added while maintaining backward compatibility with existing endpoints.

## ✅ Backward Compatibility

**All existing endpoints remain unchanged and functional:**
- ✅ `POST /api/analyze/text` - Original text analysis
- ✅ `POST /api/analyze/audio` - Original audio analysis
- ✅ `POST /api/analyze/combined` - Original combined analysis
- ✅ `GET /api/history` - Emotion history
- ✅ `GET /api/stress-alerts` - Stress alerts
- ✅ `GET /health` - Health check

## 🆕 New Endpoints

### 1. Enhanced Analysis Endpoint

**Endpoint:** `POST /api/v2/analyze/enhanced`

**Description:** Uses the full 10-stage enhanced pipeline with optional features:
- DistilBERT text encoder
- Multimodal fusion
- Profile-conditioned interpretation
- Strategy-based coping prompts

**Request Body:**
```json
{
  "text": "I'm feeling anxious",
  "user_id": "user123",
  "profile_type": "ADHD",  // Optional: "ADHD", "Autism", or "baseline"
  "publish_alerts": true,
  "use_text_encoder": true,  // Use DistilBERT
  "use_fusion": true,        // Use multimodal fusion
  "use_profile": true        // Use profile interpretation
}
```

**Response:**
```json
{
  "input": {
    "text": "I'm feeling anxious",
    "has_audio": false
  },
  "emotion": {
    "top_emotion": "anxious",
    "probability": 0.85,
    "all_emotions": {...}
  },
  "stress": {
    "stress_score": 75.0,
    "stress_score_normalized": 0.75,
    "threshold": 55.0,
    "stress_level": "high",
    "personalized": true,
    "profile_type": "ADHD"
  },
  "strategy": "breathing",
  "coping_prompt": {
    "prompt": "Let's practice deep breathing...",
    "type": "strategy_based",
    "strategy": "breathing"
  },
  "pipeline_info": {
    "used_text_encoder": true,
    "used_acoustic_encoder": false,
    "used_fusion": false,
    "used_profile": true
  }
}
```

**Example cURL:**
```bash
curl -X POST http://localhost:8000/api/v2/analyze/enhanced \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I am feeling overwhelmed",
    "user_id": "user123",
    "profile_type": "ADHD",
    "use_text_encoder": true,
    "use_profile": true
  }'
```

---

### 2. Profile Management Endpoints

#### Create/Update Profile

**Endpoint:** `POST /api/profiles`

**Description:** Create or update a user profile for personalization.

**Request Body:**
```json
{
  "user_id": "user123",
  "profile_type": "ADHD",  // "ADHD", "Autism", or "baseline"
  "stress_tolerance": 55.0,  // Optional
  "custom_threshold": 50.0   // Optional custom threshold
}
```

**Response:**
```json
{
  "user_id": "user123",
  "profile_type": "ADHD",
  "stress_tolerance": 55.0,
  "custom_threshold": 50.0,
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z"
}
```

#### Get Profile

**Endpoint:** `GET /api/profiles/{user_id}`

**Description:** Retrieve user profile.

**Response:** Same as create endpoint

**Example:**
```bash
curl http://localhost:8000/api/profiles/user123
```

#### Update Profile

**Endpoint:** `PUT /api/profiles/{user_id}`

**Description:** Update existing user profile.

**Request Body:** Same as create endpoint

**Example:**
```bash
curl -X PUT http://localhost:8000/api/profiles/user123 \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "profile_type": "Autism",
    "custom_threshold": 45.0
  }'
```

#### Delete Profile

**Endpoint:** `DELETE /api/profiles/{user_id}`

**Description:** Delete user profile.

**Response:**
```json
{
  "message": "Profile deleted successfully"
}
```

**Example:**
```bash
curl -X DELETE http://localhost:8000/api/profiles/user123
```

---

## 🔄 Migration Guide

### Using Enhanced Features

**Option 1: Use Enhanced Endpoint Directly**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v2/analyze/enhanced",
    json={
        "text": "I'm stressed",
        "user_id": "user123",
        "profile_type": "ADHD",
        "use_text_encoder": True,
        "use_profile": True
    }
)
result = response.json()
```

**Option 2: Create Profile First, Then Use Enhanced Endpoint**
```python
# Create profile
requests.post(
    "http://localhost:8000/api/profiles",
    json={
        "user_id": "user123",
        "profile_type": "ADHD"
    }
)

# Use enhanced analysis (profile will be auto-loaded)
response = requests.post(
    "http://localhost:8000/api/v2/analyze/enhanced",
    json={
        "text": "I'm stressed",
        "user_id": "user123",
        "use_profile": True
    }
)
```

### Backward Compatibility

**Existing code continues to work:**
```python
# Original endpoint - still works!
response = requests.post(
    "http://localhost:8000/api/analyze/text",
    json={
        "text": "I'm stressed",
        "user_id": "user123"
    }
)
```

---

## 📊 Feature Comparison

| Feature | Original Endpoint | Enhanced Endpoint |
|---------|------------------|-------------------|
| Text Analysis | ✅ Rule-based | ✅ DistilBERT (optional) |
| Stress Scoring | ✅ Basic formula | ✅ Architecture formula + Profile |
| Coping Prompts | ✅ Template-based | ✅ Strategy-based |
| Profile Support | ❌ | ✅ Full support |
| Multimodal Fusion | ⚠️ Simple weighted | ✅ Advanced fusion |
| Strategy Selection | ❌ | ✅ Explicit strategies |

---

## 🧪 Testing

### Test Enhanced Endpoint

```bash
# Test with profile
curl -X POST http://localhost:8000/api/v2/analyze/enhanced \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I am extremely anxious about my presentation",
    "user_id": "test_user",
    "profile_type": "ADHD",
    "use_text_encoder": true,
    "use_profile": true
  }'
```

### Test Profile Management

```bash
# Create profile
curl -X POST http://localhost:8000/api/profiles \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "profile_type": "ADHD"
  }'

# Get profile
curl http://localhost:8000/api/profiles/test_user

# Update profile
curl -X PUT http://localhost:8000/api/profiles/test_user \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "profile_type": "Autism",
    "custom_threshold": 45.0
  }'
```

---

## 🔧 Configuration

The enhanced pipeline features can be controlled via request parameters:

- `use_text_encoder`: Enable DistilBERT (requires transformers library)
- `use_fusion`: Enable multimodal fusion (requires both text and audio)
- `use_profile`: Enable profile-conditioned interpretation

All features are **optional** and fall back gracefully if unavailable.

---

## 📝 Notes

1. **Database**: Profile table is automatically created on first API startup
2. **Lazy Loading**: Enhanced pipeline is loaded on first use (not at startup)
3. **Fallbacks**: All features have fallbacks if components unavailable
4. **Backward Compatible**: Original endpoints unchanged

---

## 🔗 Related Documentation

- [Implementation Summary](IMPLEMENTATION_SUMMARY.md) - Component details
- [Architecture Alignment](ARCHITECTURE_ALIGNMENT.md) - Architecture design
- [Input/Output Guide](INPUT_OUTPUT_GUIDE.md) - API documentation

