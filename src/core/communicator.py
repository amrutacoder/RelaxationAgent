"""Communication module for Redis pub/sub and REST callbacks."""

import json
import redis
from typing import Dict, Optional, List
import httpx
from .config import Config


class Communicator:
    """Handles communication with other agents via Redis and REST."""
    
    def __init__(self):
        self.redis_client = None
        self.pubsub = None
        self._connect_redis()
    
    def _connect_redis(self):
        """Connect to Redis server."""
        try:
            self.redis_client = redis.Redis(
                host=Config.REDIS_HOST,
                port=Config.REDIS_PORT,
                db=Config.REDIS_DB,
                password=Config.REDIS_PASSWORD,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            print(f"Connected to Redis at {Config.REDIS_HOST}:{Config.REDIS_PORT}")
        except Exception as e:
            print(f"Warning: Could not connect to Redis: {e}")
            self.redis_client = None
    
    def publish_stress_alert(
        self,
        stress_score: float,
        stress_level: str,
        emotion: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Publish stress alert to Redis channel.
        
        Args:
            stress_score: Stress score (0-1)
            stress_level: "high", "medium", or "low"
            emotion: Primary emotion detected
            user_id: Optional user identifier
            metadata: Optional additional metadata
            
        Returns:
            True if published successfully
        """
        if not self.redis_client:
            return False
        
        message = {
            "stress_score": stress_score,
            "stress_level": stress_level,
            "emotion": emotion,
            "user_id": user_id,
            "timestamp": self._get_timestamp(),
            "metadata": metadata or {}
        }
        
        try:
            self.redis_client.publish(
                Config.REDIS_CHANNEL_STRESS_ALERT,
                json.dumps(message)
            )
            return True
        except Exception as e:
            print(f"Error publishing stress alert: {e}")
            return False
    
    def publish_emotion_update(
        self,
        emotion_probs: Dict[str, float],
        top_emotion: str,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Publish emotion update to Redis channel.
        
        Args:
            emotion_probs: Dictionary of emotion probabilities
            top_emotion: Primary emotion
            user_id: Optional user identifier
            
        Returns:
            True if published successfully
        """
        if not self.redis_client:
            return False
        
        message = {
            "emotions": emotion_probs,
            "top_emotion": top_emotion,
            "user_id": user_id,
            "timestamp": self._get_timestamp()
        }
        
        try:
            self.redis_client.publish(
                Config.REDIS_CHANNEL_EMOTION_UPDATE,
                json.dumps(message)
            )
            return True
        except Exception as e:
            print(f"Error publishing emotion update: {e}")
            return False
    
    async def notify_route_agent(
        self,
        stress_level: str,
        emotion: str,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Send REST callback to Route Agent.
        
        Args:
            stress_level: Stress level
            emotion: Primary emotion
            user_id: Optional user identifier
            
        Returns:
            True if notification sent successfully
        """
        if not Config.ROUTE_AGENT_URL:
            return False
        
        payload = {
            "stress_level": stress_level,
            "emotion": emotion,
            "user_id": user_id,
            "timestamp": self._get_timestamp()
        }
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{Config.ROUTE_AGENT_URL}/api/stress-update",
                    json=payload
                )
                return response.status_code == 200
        except Exception as e:
            print(f"Error notifying Route Agent: {e}")
            return False
    
    async def notify_ui_agent(
        self,
        stress_score: float,
        stress_level: str,
        emotion: str,
        prompt: str,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Send REST callback to UI Agent.
        
        Args:
            stress_score: Stress score
            stress_level: Stress level
            emotion: Primary emotion
            prompt: Coping prompt
            user_id: Optional user identifier
            
        Returns:
            True if notification sent successfully
        """
        if not Config.UI_AGENT_URL:
            return False
        
        payload = {
            "stress_score": stress_score,
            "stress_level": stress_level,
            "emotion": emotion,
            "prompt": prompt,
            "user_id": user_id,
            "timestamp": self._get_timestamp()
        }
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{Config.UI_AGENT_URL}/api/relaxation-update",
                    json=payload
                )
                return response.status_code == 200
        except Exception as e:
            print(f"Error notifying UI Agent: {e}")
            return False
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as ISO string."""
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"

