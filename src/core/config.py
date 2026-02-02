"""Configuration management for Relaxation Agent."""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration."""
    
    # Redis
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")
    
    # Redis Channels
    REDIS_CHANNEL_STRESS_ALERT: str = os.getenv("REDIS_CHANNEL_STRESS_ALERT", "stress:alerts")
    REDIS_CHANNEL_EMOTION_UPDATE: str = os.getenv("REDIS_CHANNEL_EMOTION_UPDATE", "emotion:updates")
    
    # API
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    
    # Database
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", "./data/relaxation_agent.db")
    
    # Model
    MODEL_PATH: str = os.getenv("MODEL_PATH", "./models/emotion_classifier.pt")
    MODEL_TYPE: str = os.getenv("MODEL_TYPE", "cnn_lstm")
    
    # Stress Thresholds
    STRESS_THRESHOLD_HIGH: float = float(os.getenv("STRESS_THRESHOLD_HIGH", "0.7"))
    STRESS_THRESHOLD_MEDIUM: float = float(os.getenv("STRESS_THRESHOLD_MEDIUM", "0.4"))
    
    # Coping Prompts
    USE_LLM_PROMPTS: bool = os.getenv("USE_LLM_PROMPTS", "false").lower() == "true"
    LLM_API_URL: Optional[str] = os.getenv("LLM_API_URL")
    LLM_API_KEY: Optional[str] = os.getenv("LLM_API_KEY")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "./logs/relaxation_agent.log")
    
    # Agent URLs
    VOICE_AGENT_URL: str = os.getenv("VOICE_AGENT_URL", "http://localhost:8001")
    ROUTE_AGENT_URL: str = os.getenv("ROUTE_AGENT_URL", "http://localhost:8002")
    UI_AGENT_URL: str = os.getenv("UI_AGENT_URL", "http://localhost:8003")

