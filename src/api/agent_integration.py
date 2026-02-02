"""Agent Integration Endpoints - Voice, Route, and UI Agent communication."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.milestone_a.enhanced_pipeline import EnhancedRelaxationAgentPipeline
from src.core.profile_interpreter import UserProfile
from src.core.communicator import Communicator
from src.core.config import Config

router = APIRouter(prefix="/api/agents", tags=["agent-integration"])


# ============================================================================
# Voice Agent Integration
# ============================================================================

class VoiceAgentRequest(BaseModel):
    """Request from Voice Agent."""
    text: str
    acoustic_features: Optional[Dict] = None
    user_id: Optional[str] = None


class VoiceAgentResponse(BaseModel):
    """Response to Voice Agent."""
    text: str
    voice_style: str
    stress_score: float
    emotion: str


@router.post("/voice/analyze", response_model=VoiceAgentResponse)
async def voice_agent_analyze(request: VoiceAgentRequest):
    """
    Endpoint for Voice Agent to send speech text + acoustic features.
    
    Input from Voice Agent:
    - text: Transcribed speech
    - acoustic_features: Extracted audio features (optional)
    
    Output to Voice Agent:
    - text: Coping prompt text
    - voice_style: Recommended TTS style
    - stress_score: Computed stress score
    - emotion: Detected emotion
    """
    try:
        # Initialize enhanced pipeline if needed
        global enhanced_pipeline
        if enhanced_pipeline is None:
            enhanced_pipeline = EnhancedRelaxationAgentPipeline(
                enable_logging=True,
                use_text_encoder=True,
                use_fusion=request.acoustic_features is not None,
                use_profile=True
            )
            enhanced_pipeline.communicator = Communicator()
        
        # Get user profile if available
        user_profile = None
        if request.user_id:
            user_profile = get_user_profile(request.user_id)
        
        # Process input
        result = enhanced_pipeline.process(
            text=request.text,
            audio_features=request.acoustic_features,
            user_profile=user_profile,
            user_id=request.user_id,
            publish_alerts=True
        )
        
        # Return format expected by Voice Agent
        return VoiceAgentResponse(
            text=result["voice_output"]["text"],
            voice_style=result["voice_output"]["voice_style"],
            stress_score=result["stress"].get("stress_score_normalized", result["stress"]["stress_score"]),
            emotion=result["emotion"]["top_emotion"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Route Agent Integration
# ============================================================================

class RouteAgentRequest(BaseModel):
    """Request from Route Agent."""
    stress_score: float
    user_id: str
    source: Optional[Dict] = None  # {"lat": float, "lng": float}
    destination: Optional[Dict] = None  # {"lat": float, "lng": float}


class RouteAgentResponse(BaseModel):
    """Response to Route Agent (coordinates)."""
    route_coordinates: List[Dict]  # [{"lat": float, "lng": float}, ...]
    route_id: str
    stress_consideration: str


@router.post("/route/calculate", response_model=RouteAgentResponse)
async def route_agent_calculate(request: RouteAgentRequest):
    """
    Endpoint for Route Agent to request route calculation.
    
    Input from Route Agent:
    - stress_score: Current stress score
    - user_id: User identifier
    - source: Source coordinates (optional)
    - destination: Destination coordinates (optional)
    
    Output to Route Agent:
    - route_coordinates: List of route waypoints
    - route_id: Route identifier
    - stress_consideration: How stress affected route selection
    """
    try:
        # Get user profile
        user_profile = get_user_profile(request.user_id)
        
        # Determine route strategy based on stress
        if request.stress_score >= 0.7:  # High stress
            # Prefer calmer routes (less traffic, scenic)
            route_strategy = "calm_route"
            stress_consideration = "High stress detected - selecting calmer route with less traffic"
        elif request.stress_score >= 0.4:  # Medium stress
            route_strategy = "balanced_route"
            stress_consideration = "Moderate stress - balancing speed and calmness"
        else:  # Low stress
            route_strategy = "optimal_route"
            stress_consideration = "Low stress - using optimal route"
        
        # Mock route coordinates (in real implementation, this would call routing service)
        # For now, return example coordinates
        if request.source and request.destination:
            # Simple linear interpolation (mock)
            route_coordinates = [
                request.source,
                {
                    "lat": (request.source["lat"] + request.destination["lat"]) / 2,
                    "lng": (request.source["lng"] + request.destination["lng"]) / 2
                },
                request.destination
            ]
        else:
            # Default mock route
            route_coordinates = [
                {"lat": 40.7128, "lng": -74.0060},  # NYC
                {"lat": 40.7580, "lng": -73.9855},  # Midpoint
                {"lat": 40.7831, "lng": -73.9712}   # Destination
            ]
        
        return RouteAgentResponse(
            route_coordinates=route_coordinates,
            route_id=f"route_{request.user_id}_{int(request.stress_score * 100)}",
            stress_consideration=stress_consideration
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# UI Agent Integration
# ============================================================================

class UIAgentRequest(BaseModel):
    """Request from UI Agent."""
    stress_score: float
    user_id: str
    route_coordinates: Optional[List[Dict]] = None


class UIAgentResponse(BaseModel):
    """Response to UI Agent."""
    display_data: Dict
    recommendations: List[str]


@router.post("/ui/update", response_model=UIAgentResponse)
async def ui_agent_update(request: UIAgentRequest):
    """
    Endpoint for UI Agent to get display updates.
    
    Input from UI Agent:
    - stress_score: Current stress score
    - user_id: User identifier
    - route_coordinates: Route from Route Agent (optional)
    
    Output to UI Agent:
    - display_data: Data for UI rendering
    - recommendations: List of recommendations
    """
    try:
        # Get user profile
        user_profile = get_user_profile(request.user_id)
        
        # Determine stress level
        if request.stress_score >= 0.7:
            stress_level = "high"
            color = "red"
            recommendations = [
                "Take deep breaths",
                "Consider taking a break",
                "Use coping strategies"
            ]
        elif request.stress_score >= 0.4:
            stress_level = "medium"
            color = "orange"
            recommendations = [
                "Monitor your stress",
                "Practice relaxation techniques"
            ]
        else:
            stress_level = "low"
            color = "green"
            recommendations = [
                "Maintain current state",
                "Continue stress management practices"
            ]
        
        # Prepare display data
        display_data = {
            "stress_score": request.stress_score,
            "stress_level": stress_level,
            "color": color,
            "route_coordinates": request.route_coordinates or [],
            "user_profile": {
                "profile_type": user_profile.profile_type if user_profile else "baseline",
                "threshold": user_profile.custom_threshold if user_profile and user_profile.custom_threshold else None
            } if user_profile else None
        }
        
        return UIAgentResponse(
            display_data=display_data,
            recommendations=recommendations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Helper function (imported from main.py or defined here)
import sqlite3

def get_user_profile(user_id: str):
    """Get user profile from database."""
    try:
        conn = sqlite3.connect(Config.DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM user_profiles WHERE user_id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return UserProfile(
                user_id=row["user_id"],
                profile_type=row["profile_type"],
                stress_tolerance=row["stress_tolerance"],
                custom_threshold=row["custom_threshold"],
                created_at=row["created_at"],
                updated_at=row["updated_at"]
            )
        return None
    except Exception:
        return None

# Global enhanced pipeline (lazy-loaded)
enhanced_pipeline = None

