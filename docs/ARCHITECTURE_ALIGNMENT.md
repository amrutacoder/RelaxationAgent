# Architecture Alignment Guide

This document maps the current Relaxation Agent implementation to the detailed 10-stage architecture design and provides guidance for implementing the missing components.

## Current vs. Target Architecture

### Overview Comparison

| Stage | Target Architecture | Current Implementation | Status |
|-------|-------------------|----------------------|--------|
| 1 | Input Preparation (Signal Processing) | ✅ Basic text/audio preprocessing | **Partial** |
| 2 | Text Emotion Encoder (DistilBERT) | ❌ Rule-based keyword matching | **Missing** |
| 3 | Acoustic Emotion Encoder (CNN+BiLSTM) | ⚠️ CNN-LSTM skeleton exists | **Partial** |
| 4 | Multimodal Fusion | ❌ Not implemented | **Missing** |
| 5 | Emotion Classification (BiLSTM+Attention) | ⚠️ Basic emotion classifier | **Partial** |
| 6 | Stress Score Computation | ✅ Rule-based formula | **✅ Complete** |
| 7 | Profile-Conditioned Interpretation | ❌ Basic thresholds only | **Missing** |
| 8 | Coping Strategy Selection | ⚠️ Basic prompt selection | **Partial** |
| 9 | Coping Prompt Generation | ✅ Template-based | **✅ Complete** |
| 10 | Voice Agent (TTS) | ❌ Not in Relaxation Agent scope | **N/A** |

---

## Stage-by-Stage Implementation Guide

### STAGE 1: INPUT PREPARATION ✅ (Enhanced)

**Current Status:** Basic preprocessing exists

**Enhancement Needed:**

```python
# src/core/preprocessor.py - Enhanced version

class EnhancedTextPreprocessor:
    """Enhanced preprocessing with speaker normalization."""
    
    def preprocess_text(self, text: str):
        # Current implementation: ✅ Tokenization, cleaning
        # Enhancement: Add semantic preprocessing for emotion
        pass

class EnhancedAudioPreprocessor:
    """Enhanced acoustic feature extraction."""
    
    def extract_features(self, audio_path: str, sample_rate: int = 22050):
        features = {
            "mfcc": self.extract_mfcc(audio_path),  # ✅ Current
            "pitch": self.extract_pitch(audio_path),  # ✅ Current
            "energy": self.extract_energy(audio_path),  # ❌ Missing
            "speech_rate": self.extract_speech_rate(audio_path),  # ❌ Missing
        }
        
        # Speaker normalization
        features["pitch"] = self.normalize_pitch(features["pitch"])  # ❌ Missing
        features["energy"] = self.normalize_energy(features["energy"])  # ❌ Missing
        
        return features
```

**Implementation Tasks:**
- [ ] Add energy extraction
- [ ] Add speech rate calculation
- [ ] Implement speaker normalization (baseline pitch/energy)

---

### STAGE 2: TEXT EMOTION ENCODER ❌ (New Component)

**Current Status:** Rule-based keyword matching

**Target:** DistilBERT / MiniLM transformer encoder

**Implementation Plan:**

```python
# src/core/text_encoder.py - NEW FILE

from transformers import AutoModel, AutoTokenizer
import torch

class TextEmotionEncoder:
    """Pretrained transformer encoder for text emotion embedding."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
    
    def encode(self, text: str) -> torch.Tensor:
        """
        Encode text into emotion embedding.
        
        Returns:
            Tensor of shape (768,) - emotion embedding vector
        """
        inputs = self.tokenizer(text, return_tensors="pt", 
                               truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[0, 0, :]
        
        return embedding
    
    def encode_batch(self, texts: list[str]) -> torch.Tensor:
        """Encode batch of texts."""
        inputs = self.tokenizer(texts, return_tensors="pt", 
                               truncation=True, max_length=512, 
                               padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Average pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings

# Optional: Fine-tuning script
# src/milestone_c/finetune_text_encoder.py
```

**Fine-tuning Datasets:**
- GoEmotions: https://github.com/google-research/google-research/tree/master/goemotions
- DailyDialog: http://yanran.li/dailydialog
- ISEAR: International Survey on Emotion Antecedents and Reactions

**Implementation Tasks:**
- [ ] Create `TextEmotionEncoder` class
- [ ] Integrate with emotion classifier
- [ ] Add optional fine-tuning script
- [ ] Download pretrained model on first use

---

### STAGE 3: ACOUSTIC EMOTION ENCODER ⚠️ (Enhancement)

**Current Status:** CNN-LSTM skeleton exists in `src/core/emotion_classifier.py`

**Target:** CNN + BatchNorm + BiLSTM architecture

**Current vs. Target:**

```python
# Current: src/core/emotion_classifier.py
class CNNLSTMEmotionClassifier(nn.Module):
    # Has: Conv1d → Pool → Conv1d → Pool → LSTM
    # Missing: BatchNorm, Bidirectional LSTM
    
# Target Architecture:
class AcousticEmotionEncoder(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128):
        super().__init__()
        
        # 1D CNN layers (local spectral patterns)
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)  # ❌ Missing
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)  # ❌ Missing
        self.pool2 = nn.MaxPool1d(2)
        
        # Bidirectional LSTM (temporal emotion dynamics)
        self.bilstm = nn.LSTM(
            128, hidden_dim, 
            num_layers=2, 
            batch_first=True,
            bidirectional=True  # ⚠️ Currently unidirectional
        )
        
        # Output embedding dimension: hidden_dim * 2 (bidirectional)
    
    def forward(self, x):
        # x shape: (batch, T, F) -> (batch, F, T) for CNN
        x = x.transpose(1, 2)
        
        # CNN layers
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        
        # Back to (batch, T, features) for LSTM
        x = x.transpose(1, 2)
        
        # BiLSTM
        lstm_out, _ = self.bilstm(x)
        
        # Use last timestep
        embedding = lstm_out[:, -1, :]  # (batch, hidden_dim * 2)
        
        return embedding
```

**Implementation Tasks:**
- [ ] Update `CNNLSTMEmotionClassifier` to match target architecture
- [ ] Add BatchNorm layers
- [ ] Make LSTM bidirectional
- [ ] Update training script for new architecture

---

### STAGE 4: MULTIMODAL FUSION ❌ (New Component)

**Implementation:**

```python
# src/core/multimodal_fusion.py - NEW FILE

import torch

class MultimodalFusion:
    """Simple concatenation-based fusion of text and acoustic embeddings."""
    
    def fuse(self, text_embedding: torch.Tensor, 
             acoustic_embedding: torch.Tensor) -> torch.Tensor:
        """
        Concatenate text and acoustic embeddings.
        
        Args:
            text_embedding: Shape (768,) from DistilBERT
            acoustic_embedding: Shape (hidden_dim * 2,) from CNN+BiLSTM
            
        Returns:
            Fused embedding: Shape (768 + hidden_dim * 2,)
        """
        # Simple concatenation
        fused = torch.cat([text_embedding, acoustic_embedding], dim=-1)
        
        return fused
    
    def fuse_batch(self, text_embeddings: torch.Tensor,
                   acoustic_embeddings: torch.Tensor) -> torch.Tensor:
        """Fuse batch of embeddings."""
        return torch.cat([text_embeddings, acoustic_embeddings], dim=-1)

# Integration point:
# In emotion classifier pipeline:
text_emb = text_encoder.encode(text)  # (768,)
audio_emb = acoustic_encoder(audio_features)  # (256,) if hidden_dim=128
fused = fusion.fuse(text_emb, audio_emb)  # (1024,)
```

**Implementation Tasks:**
- [ ] Create `MultimodalFusion` class
- [ ] Integrate into emotion classification pipeline
- [ ] Test fusion output dimensions

---

### STAGE 5: EMOTION CLASSIFICATION MODEL ⚠️ (Enhancement)

**Current Status:** Basic emotion classifier

**Target:** BiLSTM + Attention + Softmax

**Implementation:**

```python
# src/core/emotion_classifier.py - Enhanced version

class MultimodalEmotionClassifier(nn.Module):
    """BiLSTM + Attention emotion classifier."""
    
    def __init__(self, fused_dim: int, num_classes: int = 4, hidden_dim: int = 128):
        super().__init__()
        
        # BiLSTM (emotion evolution)
        self.bilstm = nn.LSTM(
            fused_dim, hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # bidirectional
            num_heads=8,
            batch_first=True
        )
        
        # Classification head
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, fused_embedding: torch.Tensor):
        # Add sequence dimension if needed: (batch, features) -> (batch, 1, features)
        if len(fused_embedding.shape) == 2:
            fused_embedding = fused_embedding.unsqueeze(1)
        
        # BiLSTM
        lstm_out, _ = self.bilstm(fused_embedding)  # (batch, seq_len, hidden*2)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling (or use last timestep)
        pooled = attn_out.mean(dim=1)  # (batch, hidden*2)
        
        # Classification
        x = torch.relu(self.fc1(pooled))
        x = self.dropout(x)
        logits = self.fc2(x)
        probs = self.softmax(logits)
        
        return probs  # (batch, num_classes)

# Emotion labels mapping to 4 classes:
EMOTION_MAPPING = {
    "calm": 0,
    "anxious": 1,
    "angry": 2,
    "distracted": 3
}
```

**Training Dataset Construction:**
- Combine text emotion datasets (GoEmotions, DailyDialog, ISEAR)
- Combine speech emotion datasets (IEMOCAP, RAVDESS, CREMA-D)
- Map labels to 4-class space: {calm, anxious, angry, distracted}

**Implementation Tasks:**
- [ ] Create `MultimodalEmotionClassifier` class
- [ ] Add attention mechanism
- [ ] Update label mapping to 4 classes
- [ ] Create dataset alignment script
- [ ] Update training pipeline

---

### STAGE 6: STRESS SCORE COMPUTATION ✅ (Complete)

**Current Implementation:** Rule-based formula in `src/core/stress_scorer.py`

**Alignment Check:**

```python
# Current formula (adjustable):
stress_score = sum(prob * emotion_stress_map[emotion] for emotion, prob in emotions)

# Target formula:
stress_score = 100 × (0.6·A + 0.3·G + 0.1·D)
# Where: A = anxious, G = angry, D = distracted

# Update needed in StressScorer:
def compute_stress_score(self, emotion_probs: Dict[str, float]) -> float:
    A = emotion_probs.get("anxious", 0.0)
    G = emotion_probs.get("angry", 0.0)
    D = emotion_probs.get("distracted", 0.0)
    
    stress_score = 100 * (0.6 * A + 0.3 * G + 0.1 * D)
    
    return min(stress_score / 100.0, 1.0)  # Normalize to 0-1 range
```

**Implementation Tasks:**
- [x] Rule-based computation ✅
- [ ] Update formula to match target: `0.6·A + 0.3·G + 0.1·D`

---

### STAGE 7: PROFILE-CONDITIONED INTERPRETATION ❌ (New Component)

**Current Status:** Basic thresholds only

**Implementation:**

```python
# src/core/profile_interpreter.py - NEW FILE

from typing import Literal, Optional
from dataclasses import dataclass

ProfileType = Literal["ADHD", "Autism", "baseline"]

@dataclass
class UserProfile:
    profile_type: ProfileType
    stress_tolerance: float  # Historical baseline
    custom_threshold: Optional[float] = None

class ProfileInterpreter:
    """Profile-conditioned stress interpretation."""
    
    DEFAULT_THRESHOLDS = {
        "ADHD": 55.0,
        "Autism": 50.0,
        "baseline": 60.0
    }
    
    def interpret_stress(self, stress_score: float, 
                        profile: UserProfile) -> dict:
        """
        Interpret stress score based on user profile.
        
        Returns:
            {
                "adjusted_score": float,
                "threshold": float,
                "stress_level": "low" | "medium" | "high",
                "personalized": True
            }
        """
        # Get profile-specific threshold
        if profile.custom_threshold:
            threshold = profile.custom_threshold
        else:
            threshold = self.DEFAULT_THRESHOLDS.get(
                profile.profile_type, 
                self.DEFAULT_THRESHOLDS["baseline"]
            )
        
        # Normalize threshold to 0-1 range (if stress_score is 0-1)
        threshold_norm = threshold / 100.0
        
        # Determine stress level
        if stress_score >= threshold_norm * 1.2:  # 20% above threshold
            level = "high"
        elif stress_score >= threshold_norm:
            level = "medium"
        else:
            level = "low"
        
        return {
            "adjusted_score": stress_score,
            "threshold": threshold_norm,
            "stress_level": level,
            "personalized": True,
            "profile_type": profile.profile_type
        }
```

**Database Schema Addition:**

```sql
CREATE TABLE user_profiles (
    user_id TEXT PRIMARY KEY,
    profile_type TEXT,  -- 'ADHD', 'Autism', 'baseline'
    stress_tolerance REAL,
    custom_threshold REAL,
    created_at TEXT,
    updated_at TEXT
);
```

**Implementation Tasks:**
- [ ] Create `ProfileInterpreter` class
- [ ] Add user profile database table
- [ ] Integrate into stress scoring pipeline
- [ ] Add API endpoints for profile management

---

### STAGE 8: COPING STRATEGY SELECTION ⚠️ (Enhancement)

**Current Status:** Basic prompt selection based on stress level

**Target:** Explicit strategy selection: {breathing, grounding, focus_reset, affirmation}

**Implementation:**

```python
# src/core/prompt_generator.py - Enhanced

from typing import Literal

CopingStrategy = Literal["breathing", "grounding", "focus_reset", "affirmation"]

class EnhancedCopingPromptGenerator:
    """Strategy-based coping prompt generator."""
    
    def select_strategy(self, emotion_probs: Dict[str, float], 
                       stress_score: float) -> CopingStrategy:
        """
        Select coping strategy based on emotion probabilities.
        
        Logic:
        - anxious > 0.4 → breathing
        - angry > 0.4 → grounding
        - distracted > 0.4 → focus_reset
        - else → affirmation
        """
        anxious = emotion_probs.get("anxious", 0.0)
        angry = emotion_probs.get("angry", 0.0)
        distracted = emotion_probs.get("distracted", 0.0)
        
        if anxious > 0.4:
            return "breathing"
        elif angry > 0.4:
            return "grounding"
        elif distracted > 0.4:
            return "focus_reset"
        else:
            return "affirmation"
    
    def generate(self, strategy: CopingStrategy, emotion: str, 
                stress_level: str) -> str:
        """Generate prompt for selected strategy."""
        strategy_templates = {
            "breathing": [
                "Let's practice deep breathing. Inhale for 4 counts, hold for 4, exhale for 4.",
                "Take a moment to breathe. Focus on slow, deep breaths.",
            ],
            "grounding": [
                "Let's use the 5-4-3-2-1 technique. Name 5 things you see, 4 you can touch...",
                "Ground yourself by focusing on your surroundings. What do you notice?",
            ],
            "focus_reset": [
                "Let's reset your focus. Take a short break and come back refreshed.",
                "It's okay to step away. We can continue when you're ready.",
            ],
            "affirmation": [
                "You're doing your best. Be kind to yourself.",
                "Remember, you've handled difficult moments before. You've got this.",
            ]
        }
        
        templates = strategy_templates.get(strategy, strategy_templates["affirmation"])
        return random.choice(templates)
```

**Implementation Tasks:**
- [ ] Add `select_strategy()` method
- [ ] Update prompt generation to use strategies
- [ ] Add strategy-specific templates

---

### STAGE 9: COPING PROMPT GENERATION ✅ (Complete)

**Current Status:** Template-based generation exists

**Enhancement:** Optional LLM for phrasing (T5 / DistilGPT)

**Current Implementation:** ✅ Template-based (safe, default)

**Optional Enhancement:**

```python
# src/core/prompt_generator.py - Add LLM option

class LLMCopingPromptGenerator:
    """LLM-based prompt generation (advanced, optional)."""
    
    def __init__(self, model_name: str = "google/flan-t5-small"):
        # Load small LLM (T5 or DistilGPT)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    def generate(self, strategy: CopingStrategy, emotion: str, 
                stress_level: str) -> str:
        """Generate prompt using LLM."""
        prompt = f"""
        Generate a brief, empathetic coping prompt for someone who is:
        - Emotion: {emotion}
        - Stress level: {stress_level}
        - Strategy: {strategy}
        
        Keep it under 2 sentences, warm and supportive.
        """
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=100)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
```

**Implementation Tasks:**
- [x] Template-based generation ✅
- [ ] Add optional LLM generator (if needed)

---

### STAGE 10: VOICE AGENT (TTS) ❌ (Not in Relaxation Agent scope)

**Note:** This stage is handled by the Voice Agent service, not the Relaxation Agent.

**Integration Point:**
- Relaxation Agent sends: `{"text": "...", "prosody_targets": {...}}`
- Voice Agent receives and generates TTS with emotion-conditioned prosody

---

## Complete Pipeline Integration

### Updated Pipeline Flow

```python
# src/milestone_a/enhanced_pipeline.py - NEW FILE

class EnhancedRelaxationAgentPipeline:
    """Full 10-stage pipeline implementation."""
    
    def __init__(self):
        # Stage 1: Input preparation
        self.text_preprocessor = EnhancedTextPreprocessor()
        self.audio_preprocessor = EnhancedAudioPreprocessor()
        
        # Stage 2: Text encoder
        self.text_encoder = TextEmotionEncoder()
        
        # Stage 3: Acoustic encoder
        self.acoustic_encoder = AcousticEmotionEncoder()
        
        # Stage 4: Fusion
        self.fusion = MultimodalFusion()
        
        # Stage 5: Emotion classifier
        self.emotion_classifier = MultimodalEmotionClassifier()
        
        # Stage 6: Stress scorer
        self.stress_scorer = StressScorer()
        
        # Stage 7: Profile interpreter
        self.profile_interpreter = ProfileInterpreter()
        
        # Stage 8 & 9: Prompt generator
        self.prompt_generator = EnhancedCopingPromptGenerator()
        
        # Supporting services
        self.communicator = Communicator()
        self.logger = Logger()
    
    def process(self, text: str, audio_features: Optional[Dict] = None,
                user_profile: Optional[UserProfile] = None) -> Dict:
        """Complete pipeline processing."""
        
        # Stage 1: Preprocessing
        text_processed = self.text_preprocessor.preprocess(text)
        if audio_features:
            audio_processed = self.audio_preprocessor.extract_features(audio_features)
        
        # Stage 2: Text encoding
        text_emb = self.text_encoder.encode(text)
        
        # Stage 3: Acoustic encoding (if available)
        if audio_features:
            audio_emb = self.acoustic_encoder(audio_processed)
        else:
            audio_emb = torch.zeros(self.acoustic_encoder.output_dim)
        
        # Stage 4: Fusion
        fused_emb = self.fusion.fuse(text_emb, audio_emb)
        
        # Stage 5: Emotion classification
        emotion_probs = self.emotion_classifier(fused_emb)
        
        # Stage 6: Stress scoring
        stress_score = self.stress_scorer.compute_stress_score(emotion_probs)
        
        # Stage 7: Profile interpretation
        if user_profile:
            stress_result = self.profile_interpreter.interpret_stress(
                stress_score, user_profile
            )
        else:
            stress_result = {"stress_score": stress_score, "stress_level": "medium"}
        
        # Stage 8: Strategy selection
        strategy = self.prompt_generator.select_strategy(
            emotion_probs, stress_result["stress_score"]
        )
        
        # Stage 9: Prompt generation
        prompt = self.prompt_generator.generate(
            strategy, 
            emotion_probs["top_emotion"],
            stress_result["stress_level"]
        )
        
        return {
            "emotion_probs": emotion_probs,
            "stress": stress_result,
            "strategy": strategy,
            "prompt": prompt,
            "fused_embedding": fused_emb  # For debugging
        }
```

---

## Implementation Priority

### Phase 1: Core Enhancements (Immediate)
1. ✅ Stage 6: Update stress score formula
2. ⚠️ Stage 3: Add BatchNorm + BiLSTM to acoustic encoder
3. ❌ Stage 2: Add DistilBERT text encoder

### Phase 2: Multimodal Integration (Next)
4. ❌ Stage 4: Implement multimodal fusion
5. ⚠️ Stage 5: Enhance emotion classifier with attention

### Phase 3: Personalization (Advanced)
6. ❌ Stage 7: Profile-conditioned interpretation
7. ⚠️ Stage 8: Explicit strategy selection

### Phase 4: Optional Enhancements
8. ⚠️ Stage 1: Enhanced audio features (energy, speech_rate, normalization)
9. Optional: LLM-based prompt generation

---

## Migration Strategy

1. **Keep Current Implementation Working**
   - Current text prototype remains functional
   - API endpoints continue to work

2. **Add New Components Side-by-Side**
   - Create new classes (e.g., `TextEmotionEncoder`)
   - Keep old classes for backward compatibility
   - Add feature flags in config

3. **Gradual Migration**
   - Update API endpoints to use new pipeline
   - Test thoroughly before deprecating old code
   - Maintain compatibility layer

4. **Training Pipeline Updates**
   - Update model training scripts for new architectures
   - Prepare aligned datasets (4-class emotion space)
   - Fine-tune text encoder on emotion datasets

---

## Dataset Requirements

### For Stage 2 (Text Encoder Fine-tuning)
- GoEmotions (28 emotions → map to 4 classes)
- DailyDialog (emotion labels)
- ISEAR (emotion-antecedent pairs)

### For Stage 3 (Acoustic Encoder Training)
- IEMOCAP (existing)
- RAVDESS (existing)
- CREMA-D (existing)
- Map labels to: {calm, anxious, angry, distracted}

### For Stage 5 (Multimodal Classifier)
- Paired text + audio datasets
- Aligned emotion labels
- 4-class mapping

---

## Configuration Updates

```python
# src/core/config.py - Add new options

class Config:
    # ... existing config ...
    
    # Text encoder
    TEXT_ENCODER_MODEL: str = "distilbert-base-uncased"
    TEXT_ENCODER_FINETUNED: bool = False
    TEXT_ENCODER_PATH: Optional[str] = None
    
    # Acoustic encoder
    ACOUSTIC_ENCODER_BIDIRECTIONAL: bool = True
    ACOUSTIC_ENCODER_HIDDEN_DIM: int = 128
    
    # Fusion
    FUSION_METHOD: str = "concatenation"  # or "attention", "gated"
    
    # Emotion classes
    EMOTION_CLASSES: List[str] = ["calm", "anxious", "angry", "distracted"]
    
    # Profile defaults
    DEFAULT_PROFILE_TYPE: str = "baseline"
    PROFILE_THRESHOLDS: Dict[str, float] = {
        "ADHD": 55.0,
        "Autism": 50.0,
        "baseline": 60.0
    }
```

---

## Testing Strategy

1. **Unit Tests**: Each stage component separately
2. **Integration Tests**: Full pipeline end-to-end
3. **Regression Tests**: Ensure backward compatibility
4. **Performance Tests**: Latency for each stage
5. **Ablation Studies**: Test with/without each component

---

## Summary

The current implementation provides a **solid foundation** with:
- ✅ Basic text and audio preprocessing
- ✅ Stress scoring (rule-based)
- ✅ Template-based prompt generation
- ✅ API endpoints and communication

**To align with the target architecture**, implement:
- ❌ DistilBERT text encoder (Stage 2)
- ⚠️ Enhanced CNN+BiLSTM acoustic encoder (Stage 3)
- ❌ Multimodal fusion (Stage 4)
- ⚠️ BiLSTM+Attention classifier (Stage 5)
- ❌ Profile-conditioned interpretation (Stage 7)
- ⚠️ Strategy-based prompt selection (Stage 8)

This can be done incrementally while maintaining the current working system.

