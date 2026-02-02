"""Profile-Conditioned Stress Interpretation (Stage 7)."""

from typing import Literal, Optional, Dict
from dataclasses import dataclass
from datetime import datetime
from .config import Config

ProfileType = Literal["ADHD", "Autism", "baseline"]


@dataclass
class UserProfile:
    """User profile for personalization."""
    
    user_id: str
    profile_type: ProfileType = "baseline"
    stress_tolerance: float = 60.0  # Historical baseline stress score
    custom_threshold: Optional[float] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    def __post_init__(self):
        """Initialize timestamps if not provided."""
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat() + "Z"
        if self.updated_at is None:
            self.updated_at = datetime.utcnow().isoformat() + "Z"


class ProfileInterpreter:
    """
    Profile-conditioned stress interpretation.
    
    Adjusts stress thresholds and interpretations based on user profile
    (ADHD, Autism, or baseline). This provides personalization without
    modifying the core ML models.
    """
    
    DEFAULT_THRESHOLDS = {
        "ADHD": 55.0,    # Lower threshold (more sensitive)
        "Autism": 50.0,  # Even lower threshold (very sensitive)
        "baseline": 60.0 # Standard threshold
    }
    
    # Sensitivity multipliers for each profile type
    SENSITIVITY_MULTIPLIERS = {
        "ADHD": 1.15,    # 15% more sensitive
        "Autism": 1.25,  # 25% more sensitive
        "baseline": 1.0  # No adjustment
    }
    
    def __init__(self):
        """Initialize profile interpreter with default thresholds."""
        self.thresholds = self.DEFAULT_THRESHOLDS.copy()
    
    def get_threshold(self, profile: UserProfile) -> float:
        """
        Get stress threshold for user profile.
        
        Args:
            profile: User profile
            
        Returns:
            Stress threshold (0-100 scale)
        """
        if profile.custom_threshold is not None:
            return profile.custom_threshold
        
        return self.thresholds.get(
            profile.profile_type,
            self.thresholds["baseline"]
        )
    
    def interpret_stress(
        self, 
        stress_score: float, 
        profile: UserProfile
    ) -> Dict:
        """
        Interpret stress score based on user profile.
        
        Args:
            stress_score: Stress score (0.0-1.0 or 0-100 scale)
            profile: User profile
            
        Returns:
            Dictionary with interpreted stress information
        """
        # Normalize stress score to 0-100 scale if needed
        if stress_score <= 1.0:
            stress_score_100 = stress_score * 100.0
        else:
            stress_score_100 = stress_score
        
        # Get profile-specific threshold
        threshold = self.get_threshold(profile)
        
        # Apply sensitivity multiplier
        sensitivity = self.SENSITIVITY_MULTIPLIERS.get(
            profile.profile_type,
            self.SENSITIVITY_MULTIPLIERS["baseline"]
        )
        
        adjusted_threshold = threshold / sensitivity
        
        # Determine stress level based on adjusted threshold
        if stress_score_100 >= adjusted_threshold * 1.2:  # 20% above threshold
            level = "high"
        elif stress_score_100 >= adjusted_threshold:
            level = "medium"
        else:
            level = "low"
        
        # Calculate relative stress (how far above/below threshold)
        relative_stress = (stress_score_100 - adjusted_threshold) / adjusted_threshold
        
        return {
            "stress_score": stress_score_100,
            "stress_score_normalized": stress_score_100 / 100.0,  # 0-1 range
            "threshold": adjusted_threshold,
            "threshold_raw": threshold,
            "stress_level": level,
            "relative_stress": relative_stress,
            "personalized": True,
            "profile_type": profile.profile_type,
            "sensitivity_multiplier": sensitivity,
            "interpretation": self._get_interpretation(level, relative_stress)
        }
    
    def _get_interpretation(self, level: str, relative_stress: float) -> str:
        """Get human-readable interpretation."""
        if level == "high":
            if relative_stress > 0.5:
                return "Significantly elevated stress - immediate intervention recommended"
            else:
                return "Elevated stress - coping strategies recommended"
        elif level == "medium":
            return "Moderate stress - preventive measures advised"
        else:
            return "Low stress - maintaining current state"
    
    def update_profile_threshold(
        self, 
        profile: UserProfile, 
        new_threshold: float
    ) -> UserProfile:
        """
        Update user profile with new threshold.
        
        Args:
            profile: User profile to update
            new_threshold: New custom threshold
            
        Returns:
            Updated profile
        """
        profile.custom_threshold = new_threshold
        profile.updated_at = datetime.utcnow().isoformat() + "Z"
        return profile
    
    def create_default_profile(
        self, 
        user_id: str, 
        profile_type: ProfileType = "baseline"
    ) -> UserProfile:
        """
        Create default profile for user.
        
        Args:
            user_id: User identifier
            profile_type: Profile type
            
        Returns:
            Default user profile
        """
        return UserProfile(
            user_id=user_id,
            profile_type=profile_type,
            stress_tolerance=self.thresholds.get(profile_type, self.thresholds["baseline"])
        )


# Database schema for user profiles (SQLite)
PROFILE_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id TEXT PRIMARY KEY,
    profile_type TEXT NOT NULL CHECK(profile_type IN ('ADHD', 'Autism', 'baseline')),
    stress_tolerance REAL DEFAULT 60.0,
    custom_threshold REAL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

