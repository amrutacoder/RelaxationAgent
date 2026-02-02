"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Relaxation Agent"


def test_health():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_analyze_text():
    """Test text analysis endpoint."""
    response = client.post(
        "/api/analyze/text",
        json={
            "text": "I'm feeling really stressed and anxious",
            "publish_alerts": False
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "emotion" in data
    assert "stress" in data
    assert "coping_prompt" in data
    assert data["emotion"]["top_emotion"] is not None
    assert 0.0 <= data["stress"]["stress_score"] <= 1.0


def test_analyze_text_empty():
    """Test text analysis with empty text."""
    response = client.post(
        "/api/analyze/text",
        json={"text": "", "publish_alerts": False}
    )
    # Should still work, just return neutral
    assert response.status_code == 200


def test_get_history():
    """Test history endpoint."""
    response = client.get("/api/history?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "history" in data
    assert "count" in data


def test_get_stress_alerts():
    """Test stress alerts endpoint."""
    response = client.get("/api/stress-alerts?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "alerts" in data
    assert "count" in data

