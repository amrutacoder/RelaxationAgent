"""FastAPI main application for Relaxation Agent."""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import uvicorn
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.config import Config
from src.milestone_a.text_prototype import RelaxationAgentPipeline
from src.milestone_a.enhanced_pipeline import EnhancedRelaxationAgentPipeline
from src.core.communicator import Communicator
from src.milestone_b.audio_features import AudioFeatureExtractor
from src.core.profile_interpreter import ProfileInterpreter, UserProfile, ProfileType
from src.api.agent_integration import router as agent_router
from datetime import datetime
import sqlite3

app = FastAPI(
    title="Relaxation Agent API",
    description="Emotion detection and stress scoring API with multi-agent integration",
    version="0.2.0"
)

# Include agent integration router
app.include_router(agent_router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipelines (both original and enhanced)
pipeline = RelaxationAgentPipeline(enable_logging=True)
pipeline.communicator = Communicator()

# Initialize enhanced pipeline (optional, lazy-loaded)
enhanced_pipeline = None
profile_interpreter = ProfileInterpreter()
audio_extractor = AudioFeatureExtractor()

# Initialize profile database
def init_profile_db():
    """Initialize user profiles database table."""
    db_path = Config.DATABASE_PATH
    db_dir = Path(db_path).parent
    db_dir.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            profile_type TEXT NOT NULL CHECK(profile_type IN ('ADHD', 'Autism', 'baseline')),
            stress_tolerance REAL DEFAULT 60.0,
            custom_threshold REAL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

# Initialize on startup
init_profile_db()


# Request/Response models
class TextAnalysisRequest(BaseModel):
    text: str
    user_id: Optional[str] = None
    publish_alerts: bool = True


class TextAnalysisResponse(BaseModel):
    input_text: str
    emotion: Dict
    stress: Dict
    coping_prompt: Dict
    user_id: Optional[str] = None


class AudioAnalysisRequest(BaseModel):
    user_id: Optional[str] = None
    publish_alerts: bool = True


class HealthResponse(BaseModel):
    status: str
    version: str
    redis_connected: bool


class EmotionHistoryResponse(BaseModel):
    history: List[Dict]
    count: int


# Enhanced pipeline request/response models
class EnhancedAnalysisRequest(BaseModel):
    text: Optional[str] = None
    user_id: Optional[str] = None
    profile_type: Optional[ProfileType] = None
    publish_alerts: bool = True
    use_text_encoder: bool = True
    use_fusion: bool = True
    use_profile: bool = True


class UserProfileRequest(BaseModel):
    user_id: str
    profile_type: ProfileType = "baseline"
    stress_tolerance: Optional[float] = None
    custom_threshold: Optional[float] = None


class UserProfileResponse(BaseModel):
    user_id: str
    profile_type: ProfileType
    stress_tolerance: float
    custom_threshold: Optional[float]
    created_at: str
    updated_at: str


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint."""
    return {
        "service": "Relaxation Agent",
        "version": "0.2.0",
        "status": "running",
        "features": {
            "enhanced_pipeline": True,
            "agent_integration": True,
            "profile_support": True,
            "voice_style_generation": True
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    redis_connected = pipeline.communicator.redis_client is not None
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        redis_connected=redis_connected
    )


@app.post("/api/analyze/text", response_model=TextAnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    """
    Analyze text input for emotion and stress.
    
    This is the main endpoint for Milestone A (text prototype).
    """
    try:
        result = pipeline.process_text(
            text=request.text,
            user_id=request.user_id,
            publish_alerts=request.publish_alerts
        )
        
        return TextAnalysisResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze/audio")
async def analyze_audio(
    audio_file: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
    publish_alerts: bool = Form(True)
):
    """
    Analyze audio file for emotion and stress.
    
    This endpoint requires Milestone B (audio features) to be fully functional.
    """
    try:
        # Save uploaded file temporarily
        temp_path = Path(f"./temp_{audio_file.filename}")
        with open(temp_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)
        
        # Extract audio features
        audio_features = audio_extractor.extract_from_file(str(temp_path))
        
        # Preprocess text (if available from STT, otherwise use empty)
        # In real scenario, Voice Agent would send both text and audio
        text_preprocessed = pipeline.text_preprocessor.preprocess("")
        
        # Classify emotion from audio
        emotion_probs = pipeline.emotion_classifier.predict_from_audio(audio_features)
        top_emotion, top_prob = pipeline.emotion_classifier.get_top_emotion(emotion_probs)
        
        # Compute stress
        stress_result = pipeline.stress_scorer.compute_stress_score(
            emotion_probs,
            text_features=text_preprocessed,
            audio_features=audio_features
        )
        
        # Generate prompt
        prompt_result = pipeline.prompt_generator.generate(
            stress_level=stress_result["stress_level"],
            top_emotion=top_emotion,
            stress_score=stress_result["stress_score"]
        )
        
        # Log
        if pipeline.logger:
            pipeline.logger.log_emotion_analysis(
                text_input=None,
                emotion_probs=emotion_probs,
                top_emotion=top_emotion,
                stress_score=stress_result["stress_score"],
                stress_level=stress_result["stress_level"],
                prompt=prompt_result["prompt"],
                user_id=user_id,
                audio_features=audio_features
            )
        
        # Publish alerts
        if publish_alerts and stress_result["stress_level"] in ["high", "medium"]:
            if pipeline.communicator:
                pipeline.communicator.publish_stress_alert(
                    stress_score=stress_result["stress_score"],
                    stress_level=stress_result["stress_level"],
                    emotion=top_emotion,
                    user_id=user_id
                )
        
        # Cleanup
        temp_path.unlink()
        
        return {
            "emotion": {
                "top_emotion": top_emotion,
                "probability": top_prob,
                "all_emotions": emotion_probs
            },
            "stress": stress_result,
            "coping_prompt": prompt_result,
            "user_id": user_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze/combined")
async def analyze_combined(
    text: str = Form(...),
    audio_file: Optional[UploadFile] = File(None),
    user_id: Optional[str] = Form(None),
    publish_alerts: bool = Form(True)
):
    """
    Analyze combined text and audio input.
    
    This is the ideal endpoint when Voice Agent sends both STT text and audio features.
    """
    try:
        # Process text
        text_preprocessed = pipeline.text_preprocessor.preprocess(text)
        emotion_probs_text = pipeline.emotion_classifier.predict_from_text(text, text_preprocessed)
        
        # Process audio if provided
        audio_features = None
        emotion_probs_audio = None
        
        if audio_file:
            temp_path = Path(f"./temp_{audio_file.filename}")
            with open(temp_path, "wb") as f:
                content = await audio_file.read()
                f.write(content)
            
            audio_features = audio_extractor.extract_from_file(str(temp_path))
            emotion_probs_audio = pipeline.emotion_classifier.predict_from_audio(audio_features)
            temp_path.unlink()
        
        # Combine emotion probabilities (weighted average)
        if emotion_probs_audio:
            # Weight: 60% audio, 40% text (adjustable)
            emotion_probs = {
                emotion: 0.6 * emotion_probs_audio.get(emotion, 0) + 
                        0.4 * emotion_probs_text.get(emotion, 0)
                for emotion in set(list(emotion_probs_text.keys()) + list(emotion_probs_audio.keys()))
            }
        else:
            emotion_probs = emotion_probs_text
        
        # Normalize
        total = sum(emotion_probs.values())
        if total > 0:
            emotion_probs = {k: v / total for k, v in emotion_probs.items()}
        
        top_emotion, top_prob = pipeline.emotion_classifier.get_top_emotion(emotion_probs)
        
        # Compute stress
        stress_result = pipeline.stress_scorer.compute_stress_score(
            emotion_probs,
            text_features=text_preprocessed,
            audio_features=audio_features
        )
        
        # Generate prompt
        prompt_result = pipeline.prompt_generator.generate(
            stress_level=stress_result["stress_level"],
            top_emotion=top_emotion,
            stress_score=stress_result["stress_score"]
        )
        
        # Log
        if pipeline.logger:
            pipeline.logger.log_emotion_analysis(
                text_input=text,
                emotion_probs=emotion_probs,
                top_emotion=top_emotion,
                stress_score=stress_result["stress_score"],
                stress_level=stress_result["stress_level"],
                prompt=prompt_result["prompt"],
                user_id=user_id,
                audio_features=audio_features
            )
        
        # Publish alerts
        if publish_alerts and stress_result["stress_level"] in ["high", "medium"]:
            if pipeline.communicator:
                pipeline.communicator.publish_stress_alert(
                    stress_score=stress_result["stress_score"],
                    stress_level=stress_result["stress_level"],
                    emotion=top_emotion,
                    user_id=user_id
                )
        
        return {
            "input_text": text,
            "emotion": {
                "top_emotion": top_emotion,
                "probability": top_prob,
                "all_emotions": emotion_probs
            },
            "stress": stress_result,
            "coping_prompt": prompt_result,
            "user_id": user_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history", response_model=EmotionHistoryResponse)
async def get_emotion_history(
    user_id: Optional[str] = None,
    limit: int = 100
):
    """Get emotion analysis history."""
    if not pipeline.logger:
        raise HTTPException(status_code=503, detail="Logging not enabled")
    
    history = pipeline.logger.get_emotion_history(user_id=user_id, limit=limit)
    
    return EmotionHistoryResponse(
        history=history,
        count=len(history)
    )


@app.get("/api/stress-alerts")
async def get_stress_alerts(
    user_id: Optional[str] = None,
    stress_level: Optional[str] = None,
    limit: int = 100
):
    """Get stress alert history."""
    if not pipeline.logger:
        raise HTTPException(status_code=503, detail="Logging not enabled")
    
    alerts = pipeline.logger.get_stress_alerts(
        user_id=user_id,
        stress_level=stress_level,
        limit=limit
    )
    
    return {
        "alerts": alerts,
        "count": len(alerts)
    }


# ============================================================================
# Enhanced Pipeline Endpoints (New Features)
# ============================================================================

@app.post("/api/v2/analyze/enhanced")
async def analyze_enhanced(request: EnhancedAnalysisRequest):
    """
    Enhanced analysis using full 10-stage architecture.
    
    Features:
    - DistilBERT text encoder (optional)
    - Multimodal fusion (optional)
    - Profile-conditioned interpretation (optional)
    - Strategy-based coping prompts
    """
    global enhanced_pipeline
    
    try:
        # Lazy-load enhanced pipeline
        if enhanced_pipeline is None:
            enhanced_pipeline = EnhancedRelaxationAgentPipeline(
                enable_logging=True,
                use_text_encoder=request.use_text_encoder,
                use_fusion=request.use_fusion,
                use_profile=request.use_profile
            )
            enhanced_pipeline.communicator = Communicator()
        
        # Get or create user profile
        user_profile = None
        if request.user_id and request.use_profile:
            user_profile = get_user_profile(request.user_id)
            if user_profile is None and request.profile_type:
                # Create default profile
                user_profile = profile_interpreter.create_default_profile(
                    request.user_id,
                    request.profile_type
                )
                save_user_profile(user_profile)
        
        # Process with enhanced pipeline
        result = enhanced_pipeline.process(
            text=request.text,
            user_profile=user_profile,
            user_id=request.user_id,
            publish_alerts=request.publish_alerts
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Profile Management Endpoints
# ============================================================================

@app.post("/api/profiles", response_model=UserProfileResponse)
async def create_profile(profile: UserProfileRequest):
    """Create or update user profile."""
    try:
        user_profile = UserProfile(
            user_id=profile.user_id,
            profile_type=profile.profile_type,
            stress_tolerance=profile.stress_tolerance or 60.0,
            custom_threshold=profile.custom_threshold
        )
        
        save_user_profile(user_profile)
        
        return UserProfileResponse(
            user_id=user_profile.user_id,
            profile_type=user_profile.profile_type,
            stress_tolerance=user_profile.stress_tolerance,
            custom_threshold=user_profile.custom_threshold,
            created_at=user_profile.created_at,
            updated_at=user_profile.updated_at
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/profiles/{user_id}", response_model=UserProfileResponse)
async def get_profile(user_id: str):
    """Get user profile."""
    profile = get_user_profile(user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    return UserProfileResponse(
        user_id=profile.user_id,
        profile_type=profile.profile_type,
        stress_tolerance=profile.stress_tolerance,
        custom_threshold=profile.custom_threshold,
        created_at=profile.created_at,
        updated_at=profile.updated_at
    )


@app.put("/api/profiles/{user_id}", response_model=UserProfileResponse)
async def update_profile(user_id: str, profile: UserProfileRequest):
    """Update user profile."""
    existing = get_user_profile(user_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    # Update fields
    existing.profile_type = profile.profile_type
    if profile.stress_tolerance is not None:
        existing.stress_tolerance = profile.stress_tolerance
    if profile.custom_threshold is not None:
        existing.custom_threshold = profile.custom_threshold
    existing.updated_at = datetime.utcnow().isoformat() + "Z"
    
    save_user_profile(existing)
    
    return UserProfileResponse(
        user_id=existing.user_id,
        profile_type=existing.profile_type,
        stress_tolerance=existing.stress_tolerance,
        custom_threshold=existing.custom_threshold,
        created_at=existing.created_at,
        updated_at=existing.updated_at
    )


@app.delete("/api/profiles/{user_id}")
async def delete_profile(user_id: str):
    """Delete user profile."""
    try:
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))
        conn.commit()
        deleted = cursor.rowcount > 0
        conn.close()
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        return {"message": "Profile deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions for profile management
def get_user_profile(user_id: str) -> Optional[UserProfile]:
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


def save_user_profile(profile: UserProfile):
    """Save user profile to database."""
    conn = sqlite3.connect(Config.DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO user_profiles 
        (user_id, profile_type, stress_tolerance, custom_threshold, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        profile.user_id,
        profile.profile_type,
        profile.stress_tolerance,
        profile.custom_threshold,
        profile.created_at,
        profile.updated_at
    ))
    conn.commit()
    conn.close()


# Import datetime for profile updates
from datetime import datetime


def main():
    """Run the API server."""
    uvicorn.run(
        "src.api.main:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=True
    )


if __name__ == "__main__":
    main()

