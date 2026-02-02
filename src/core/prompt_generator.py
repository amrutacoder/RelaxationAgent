"""Coping prompt generator module."""

from typing import Dict, List, Optional, Literal
import random
import httpx
from .config import Config

CopingStrategy = Literal["breathing", "grounding", "focus_reset", "affirmation"]


class CopingPromptGenerator:
    """Generates personalized coping prompts based on stress and emotion."""
    
    def __init__(self):
        self.use_llm = Config.USE_LLM_PROMPTS
        self.llm_api_url = Config.LLM_API_URL
        self.llm_api_key = Config.LLM_API_KEY
        
        # Strategy-based prompts database (Stage 8)
        self.strategy_prompts = {
            "breathing": [
                "Let's practice deep breathing. Inhale for 4 counts, hold for 4, exhale for 4.",
                "Take a moment to breathe. Focus on slow, deep breaths. In through your nose, out through your mouth.",
                "Try box breathing: Breathe in for 4, hold for 4, out for 4, hold for 4. Repeat.",
            ],
            "grounding": [
                "Let's use the 5-4-3-2-1 technique. Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste.",
                "Ground yourself by focusing on your surroundings. What do you notice right now?",
                "Notice your feet on the ground. Feel the texture, the support. You're here, in this moment.",
            ],
            "focus_reset": [
                "Let's reset your focus. Take a short break and come back refreshed.",
                "It's okay to step away. We can continue when you're ready.",
                "Give yourself permission to pause. A brief reset can help you regain clarity.",
            ],
            "affirmation": [
                "You're doing your best. Be kind to yourself.",
                "Remember, you've handled difficult moments before. You've got this.",
                "You are capable and resilient. Trust in your ability to navigate this.",
            ]
        }
        
        # Legacy rule-based prompts database (for backward compatibility)
        self.prompts = {
            "high_stress": {
                "anxious": [
                    "Take a deep breath. Inhale for 4 counts, hold for 4, exhale for 4.",
                    "Try the 5-4-3-2-1 grounding technique: Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste.",
                    "Remember: This feeling will pass. You've handled difficult moments before."
                ],
                "angry": [
                    "Step away for a moment. Take 10 deep breaths before responding.",
                    "Try progressive muscle relaxation: Tense and release each muscle group.",
                    "Express your feelings in a journal or to a trusted person."
                ],
                "stressed": [
                    "Break your task into smaller, manageable steps.",
                    "Practice mindfulness: Focus on the present moment, not future worries.",
                    "Take a short walk or do light stretching."
                ],
                "default": [
                    "You're doing your best. Be kind to yourself.",
                    "Take a moment to pause and breathe deeply.",
                    "Remember: It's okay to ask for help when you need it."
                ]
            },
            "medium_stress": {
                "default": [
                    "Take a few deep breaths to center yourself.",
                    "Try a quick mindfulness exercise: Focus on your breathing for 2 minutes.",
                    "Consider what's within your control and what isn't."
                ]
            },
            "low_stress": {
                "default": [
                    "Great job managing your stress! Keep up the good work.",
                    "Continue practicing your relaxation techniques.",
                    "You're in a good place. Maintain your balance."
                ]
            }
        }
    
    def select_strategy(
        self, 
        emotion_probs: Dict[str, float], 
        stress_score: float
    ) -> CopingStrategy:
        """
        Select coping strategy based on emotion probabilities (Stage 8).
        
        Logic:
        - anxious > 0.4 → breathing
        - angry > 0.4 → grounding
        - distracted > 0.4 → focus_reset
        - else → affirmation
        
        Args:
            emotion_probs: Dictionary of emotion probabilities
            stress_score: Stress score (0-1)
            
        Returns:
            Selected coping strategy
        """
        anxious = emotion_probs.get("anxious", 0.0)
        angry = emotion_probs.get("angry", 0.0)
        distracted = emotion_probs.get("distracted", 0.0)
        
        # Map "stressed" to anxious if anxious not present
        if anxious == 0.0 and "stressed" in emotion_probs:
            anxious = emotion_probs["stressed"]
        
        # Select strategy based on dominant emotion
        if anxious > 0.4:
            return "breathing"
        elif angry > 0.4:
            return "grounding"
        elif distracted > 0.4:
            return "focus_reset"
        else:
            return "affirmation"
    
    def generate(
        self,
        stress_level: str,
        top_emotion: str,
        stress_score: float,
        context: Optional[Dict] = None,
        emotion_probs: Optional[Dict[str, float]] = None,
        use_strategy_selection: bool = True
    ) -> Dict[str, str]:
        """
        Generate coping prompt (synchronous).
        
        Args:
            stress_level: "high", "medium", or "low"
            top_emotion: Primary emotion detected
            stress_score: Numeric stress score
            context: Optional additional context
            emotion_probs: Optional emotion probabilities for strategy selection
            use_strategy_selection: If True, use strategy-based selection (Stage 8)
            
        Returns:
            Dictionary with prompt text and metadata
        """
        # Use strategy-based selection if enabled and emotion_probs provided
        if use_strategy_selection and emotion_probs:
            strategy = self.select_strategy(emotion_probs, stress_score)
            return self._generate_strategy_based_prompt(strategy, top_emotion, stress_level, stress_score)
        else:
            # Fall back to legacy rule-based
            return self._generate_rule_based_prompt(stress_level, top_emotion, stress_score)
    
    def _generate_strategy_based_prompt(
        self,
        strategy: CopingStrategy,
        emotion: str,
        stress_level: str,
        stress_score: float
    ) -> Dict[str, str]:
        """Generate prompt using strategy-based selection."""
        prompts = self.strategy_prompts.get(strategy, self.strategy_prompts["affirmation"])
        prompt_text = random.choice(prompts)
        
        return {
            "prompt": prompt_text,
            "type": "strategy_based",
            "strategy": strategy,
            "stress_level": stress_level,
            "emotion": emotion,
            "stress_score": stress_score
        }
    
    async def generate_async(
        self,
        stress_level: str,
        top_emotion: str,
        stress_score: float,
        context: Optional[Dict] = None
    ) -> Dict[str, str]:
        """
        Generate coping prompt (asynchronous, supports LLM).
        
        Args:
            stress_level: "high", "medium", or "low"
            top_emotion: Primary emotion detected
            stress_score: Numeric stress score
            context: Optional additional context
            
        Returns:
            Dictionary with prompt text and metadata
        """
        if self.use_llm and self.llm_api_url:
            return await self._generate_llm_prompt(stress_level, top_emotion, stress_score, context)
        else:
            return self._generate_rule_based_prompt(stress_level, top_emotion, stress_score)
    
    def _generate_rule_based_prompt(
        self,
        stress_level: str,
        top_emotion: str,
        stress_score: float
    ) -> Dict[str, str]:
        """Generate prompt using rule-based system."""
        import random
        
        # Get prompts for stress level
        level_prompts = self.prompts.get(stress_level, self.prompts["low_stress"])
        
        # Get emotion-specific prompts or default
        emotion_prompts = level_prompts.get(top_emotion, level_prompts.get("default", []))
        
        if not emotion_prompts:
            emotion_prompts = ["Take a moment to breathe and center yourself."]
        
        # Select random prompt
        prompt_text = random.choice(emotion_prompts)
        
        return {
            "prompt": prompt_text,
            "type": "rule_based",
            "stress_level": stress_level,
            "emotion": top_emotion,
            "stress_score": stress_score
        }
    
    async def _generate_llm_prompt(
        self,
        stress_level: str,
        top_emotion: str,
        stress_score: float,
        context: Optional[Dict] = None
    ) -> Dict[str, str]:
        """Generate prompt using LLM API."""
        if not self.llm_api_url:
            return self._generate_rule_based_prompt(stress_level, top_emotion, stress_score)
        
        # Construct prompt for LLM
        system_prompt = (
            f"You are a helpful relaxation and stress management assistant. "
            f"The user is experiencing {stress_level} stress ({stress_score:.2f}) "
            f"with primary emotion: {top_emotion}. "
            f"Generate a brief, personalized, and empathetic coping prompt (1-2 sentences)."
        )
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    self.llm_api_url,
                    json={
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": "Generate a coping prompt."}
                        ]
                    },
                    headers={"Authorization": f"Bearer {self.llm_api_key}"} if self.llm_api_key else {}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    prompt_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    
                    if prompt_text:
                        return {
                            "prompt": prompt_text,
                            "type": "llm",
                            "stress_level": stress_level,
                            "emotion": top_emotion,
                            "stress_score": stress_score
                        }
        except Exception as e:
            print(f"LLM prompt generation failed: {e}")
        
        # Fallback to rule-based
        return self._generate_rule_based_prompt(stress_level, top_emotion, stress_score)

