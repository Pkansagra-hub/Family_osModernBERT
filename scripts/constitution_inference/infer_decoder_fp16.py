"""
FP16 Inference Script for UltraBERT-Gen Decoder (13th Head).

This is the production-ready inference engine for FamilyOS.
It implements a hybrid UltraBERT (Encoder) + GPT-2 (Decoder) architecture
optimized for generating counterfactual parenting advice.

KEY FEATURES:
1. Low-VRAM Optimization:
   - Loads only Encoder + Decoder (skips classification heads).
   - Runs in FP16 mode (~2-4GB VRAM usage).

2. Embedding Stabilization:
   - Implements `normalize_embeddings` to handle UltraBERT's high-variance latent space.
   - Supports `clamp_tight` ([-2, 2]) for high-emotion stability.

3. Family Constitution Awareness:
   - Dynamic context injection via `--constitution`.
   - Supports "Split Encoding" (`--split-encoding`) to process rules independently
     from the scenario, preventing signal dilution.
   - Automatically parses `data/family_constitutions.json`.

4. Advanced Sampling:
   - Parallel Sampling (`--num-candidates`) for beam-search-like diversity.
   - Structured Prompting (`[Role | Emotion | Sentiment]`).
   - Post-processing to ensure clean sentence endings.

USAGE EXAMPLES:

1. Basic Inference:
   python scripts/infer_decoder_fp16.py --text "My child won't eat"

2. High-Emotion Scenario (with Clamping):
   python scripts/infer_decoder_fp16.py \
     --text "I hate you!" \
     --emotion "angry" \
     --normalize-embeddings clamp_tight

3. Constitution-Aware (Split Encoding):
   python scripts/infer_decoder_fp16.py \
     --text "Teenager skipping school" \
     --constitution "traditional_strict" \
     --split-encoding \
     --normalize-embeddings clamp_tight

4. Interactive Mode:
   python scripts/infer_decoder_fp16.py --interactive
"""

from __future__ import annotations

import argparse
import gc
import logging
import re
import sys
import time
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Get device, preferring CUDA if available."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def clear_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model_fp16(
    checkpoint_path: str,
    device: torch.device,
):
    """
    Load encoder + decoder in FP16 mode.

    Windows-safe loading sequence:
    1. Load GPT-2 and ModernBERT from HuggingFace first
    2. Load checkpoint weights (as numpy to avoid safetensors/torch crash on Windows)
    3. Apply weights to models
    """
    from transformers import AutoTokenizer
    from modeling_studio.models.decoder_gpt2 import GPT2DecoderHead
    from modeling_studio.models.decoder_gpt2_config import GPT2DecoderConfig

    try:
        from transformers import ModernBertModel
    except ImportError:
        from transformers import AutoModel
        ModernBertModel = AutoModel

    logger.info(f"Loading from: {checkpoint_path}")
    checkpoint_path = Path(checkpoint_path)

    # Load tokenizer first
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)

    # Check for trainer state
    trainer_state_path = checkpoint_path / "trainer_state.json"
    if trainer_state_path.exists():
        import json
        try:
            with open(trainer_state_path, "r") as f:
                ts = json.load(f)
                logger.info(f"Checkpoint Info: Step={ts.get('global_step')}, Epoch={ts.get('epoch', 0):.2f}")
                if 'best_metric' in ts:
                    logger.info(f"Best Metric (Loss): {ts['best_metric']:.4f}")
        except Exception as e:
            logger.warning(f"Could not read trainer_state.json: {e}")

    # =========================================================================
    # STEP 1: Create models from HuggingFace FIRST (before loading safetensors)
    # =========================================================================
    logger.info("Loading encoder from HuggingFace...")
    encoder = ModernBertModel.from_pretrained(
        "answerdotai/ModernBERT-base",
        torch_dtype=torch.float16,
    )
    logger.info("Encoder created")

    decoder_config = GPT2DecoderConfig(
        gpt2_model_name="gpt2-medium",
        encoder_hidden_size=768,
        projection_hidden_size=1024,
        num_prefix_tokens=16,
        freeze_layers=12,
    )

    logger.info("Creating GPT-2 decoder from HuggingFace...")
    decoder = GPT2DecoderHead(
        config=decoder_config,
        encoder_hidden_size=768,
    )
    logger.info("Decoder created")

    # =========================================================================
    # STEP 2: Load checkpoint weights (using numpy workaround for Windows)
    # =========================================================================
    safetensors_path = checkpoint_path / "model.safetensors"
    pytorch_path = checkpoint_path / "pytorch_model.bin"

    logger.info("Loading checkpoint weights...")
    if safetensors_path.exists():
        # Windows-safe: load as numpy then convert to torch
        from safetensors import safe_open
        import numpy as np

        state_dict = {}
        with safe_open(str(safetensors_path), framework="numpy") as f:
            for key in f.keys():
                state_dict[key] = torch.from_numpy(f.get_tensor(key).copy())
    elif pytorch_path.exists():
        state_dict = torch.load(pytorch_path, map_location="cpu", weights_only=False)
    else:
        raise FileNotFoundError(f"No model weights found in {checkpoint_path}")

    logger.info(f"Loaded {len(state_dict)} total keys")

    # =========================================================================
    # STEP 3: Apply weights to encoder
    # =========================================================================
    logger.info("Applying encoder weights...")
    encoder_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("encoder."):
            encoder_state_dict[key.replace("encoder.", "", 1)] = value
        elif key.startswith("backbone."):
            encoder_state_dict[key.replace("backbone.", "", 1)] = value

    if encoder_state_dict:
        try:
            missing, unexpected = encoder.load_state_dict(encoder_state_dict, strict=False)
            logger.info(f"Encoder: loaded {len(encoder_state_dict)} keys, missing: {len(missing)}")
        except Exception as e:
            logger.warning(f"Could not load encoder weights: {e}, using pretrained")

    encoder = encoder.to(device).half().eval()
    del encoder_state_dict
    clear_memory()

    # =========================================================================
    # STEP 4: Apply weights to decoder
    # =========================================================================
    logger.info("Applying decoder weights...")
    decoder_state_dict = {}
    for key, value in state_dict.items():
        if "counterfactual" in key:
            if key.startswith("heads.counterfactual."):
                new_key = key.replace("heads.counterfactual.", "")
            elif key.startswith("head.counterfactual."):
                new_key = key.replace("head.counterfactual.", "")
            else:
                new_key = key
            decoder_state_dict[new_key] = value

    if decoder_state_dict:
        try:
            missing, unexpected = decoder.load_state_dict(decoder_state_dict, strict=False)
            logger.info(f"Decoder: loaded {len(decoder_state_dict)} keys, missing: {len(missing)}, unexpected: {len(unexpected)}")
        except Exception as e:
            logger.error(f"Failed to load decoder weights: {e}")
            raise

    decoder = decoder.to(device).half().eval()

    # Free all state dicts
    del decoder_state_dict
    del state_dict
    clear_memory()

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"GPU Memory: {allocated:.2f}GB allocated")

    return encoder, decoder, tokenizer


def normalize_embeddings(embeddings, method="unit_norm"):
    """
    Normalize embeddings for better GPT-2 interpretation
    """
    if method == "unit_norm":
        # Normalize each token vector to unit length
        # This preserves direction (emotional pattern) but normalizes intensity
        norm = torch.norm(embeddings, dim=-1, keepdim=True)
        return embeddings / (norm + 1e-8)

    elif method == "standardize":
        # Standardize across batch and sequence
        mean = embeddings.mean(dim=(0, 1), keepdim=True)
        std = embeddings.std(dim=(0, 1), keepdim=True)
        return (embeddings - mean) / (std + 1e-8)

    elif method == "clamp":
        # Clamp extreme values
        return torch.clamp(embeddings, -3, 3)

    elif method == "clamp_tight":
        # Tighter clamping for extreme emotions
        return torch.clamp(embeddings, -2, 2)

    elif method == "clamp_extreme":
        # Even tighter clamping for very extreme scenarios
        return torch.clamp(embeddings, -1.5, 1.5)

    elif method == "layer_norm":
        # LayerNorm normalizes across features
        return torch.nn.functional.layer_norm(embeddings, [embeddings.size(-1)])

    elif method == "smooth_clamp":
        # Gradual clamping
        threshold = 3.0
        scale = threshold / (torch.abs(embeddings) + threshold)
        return embeddings * scale

    return embeddings


class ConstitutionController:
    """
    Constitution Controller for steering text generation.

    Implements 3-Layer Constitution Architecture:
    - Layer 1: Family Values (static per tenant)
    - Layer 2: Individual Preferences (per actor)
    - Layer 3: Situational Rules (dynamic from P02 signals)

    Loads schemas from: data/constitutions/constitution_schemas_v2.json
    """

    def __init__(self, tokenizer, schema_path: str = None):
        self.tokenizer = tokenizer
        self.schema_path = schema_path
        self.schemas = self.load_schemas()
        # Cached resolved params for current request
        self._current_params = None
        self._current_layers = []

    def load_schemas(self):
        """
        Load 3-layer constitution schemas from external file.
        Returns dict with: family_values, individual_prefs, situational_rules
        """
        import json

        # Try to find schema file (prefer v2 format)
        project_root = Path(__file__).parent.parent.parent
        schema_locations = []
        if self.schema_path:
            schema_locations.append(Path(self.schema_path))

        schema_locations.extend([
            project_root / "data" / "constitutions" / "constitution_schemas_v2.json",
            project_root / "data" / "constitution_schemas_v2.json",
            project_root / "data" / "constitution_schemas.json",
        ])

        for schema_path in schema_locations:
            if schema_path.exists():
                logger.info(f"Loading constitution schemas from: {schema_path}")
                try:
                    with open(schema_path, "r", encoding="utf-8") as f:
                        raw_schemas = json.load(f)

                    # Check if v2 format (has family_values key)
                    if "family_values" in raw_schemas:
                        logger.info(f"Loaded v2 schema with {len(raw_schemas.get('family_values', {}))} family values")
                        self._schema_version = "v2"
                        return raw_schemas
                    else:
                        # Legacy format - wrap in family_values
                        raw_schemas.pop("_meta", None)
                        logger.info(f"Loaded legacy schema with {len(raw_schemas)} constitutions")
                        self._schema_version = "v1"
                        return {"family_values": raw_schemas, "situational_rules": {}, "individual_prefs": {}}

                except Exception as e:
                    logger.warning(f"Failed to load {schema_path}: {e}")
                    continue

        # Fallback to built-in defaults
        logger.info("Using built-in default constitution schemas")
        return self._get_default_schemas()

    def _get_default_schemas(self):
        """Built-in default schemas in v2 3-layer format."""
        return {
            "family_values": {
                "gentle_parenting": {
                    "description": "Empathetic, connection-focused parenting",
                    "positive_tokens": {
                        "feel": 0.4, "understand": 0.4, "connect": 0.5,
                        "together": 0.3, "help": 0.4, "support": 0.4,
                        "calm": 0.3, "listen": 0.4, "validate": 0.5,
                    },
                    "negative_tokens": {
                        "punish": -0.4, "timeout": -0.4, "obey": -0.3,
                        "in this situation": -0.8, "you might consider": -0.7,
                    },
                    "generation_params": {
                        "temperature": 0.55,
                        "repetition_penalty": 1.03,
                        "logits_strength": 0.5,
                    },
                    "prefix_pattern": [0.1, 0.4, -0.05, 0.2, 0.3, 0.2, -0.2, 0.1],
                    "prefix_injection": False,
                },
                "traditional_strict": {
                    "description": "Clear rules and consistent consequences",
                    "positive_tokens": {
                        "rule": 0.5, "clear": 0.4, "expectation": 0.4,
                        "respect": 0.5, "consequence": 0.4, "boundary": 0.4,
                    },
                    "negative_tokens": {
                        "maybe": -0.3, "perhaps": -0.3,
                        "in this situation": -0.6,
                    },
                    "generation_params": {
                        "temperature": 0.45,
                        "repetition_penalty": 1.1,
                        "logits_strength": 0.8,
                    },
                    "prefix_pattern": [0.8, 0.1, 0.9, -0.3, -0.4, 0.7, 0.2, 0.5],
                    "prefix_injection": True,
                },
                "indian_joint_family": {
                    "description": "Joint family harmony with elder respect",
                    "positive_tokens": {
                        "family": 0.6, "respect": 0.6, "elder": 0.5,
                        "together": 0.5, "home": 0.4, "values": 0.5,
                    },
                    "negative_tokens": {
                        "alone": -0.3, "ignore": -0.4, "disrespect": -0.5,
                        "in this situation": -0.6,
                    },
                    "generation_params": {
                        "temperature": 0.5,
                        "repetition_penalty": 1.08,
                        "logits_strength": 1.0,
                    },
                    "prefix_pattern": [0.3, 0.6, 0.4, 0.2, 0.5, -0.1, 0.3, 0.7],
                    "prefix_injection": True,
                }
            },
            "individual_prefs": {
                "default": {
                    "response_length": "moderate",
                    "formality": "neutral",
                    "needs_validation_first": True
                }
            },
            "situational_rules": {
                "high_arousal": {
                    "condition": "affect_arousal > 0.7",
                    "actions": {
                        "temperature_adjustment": -0.1,
                        "inject_tokens": {"calm": 0.3, "breathe": 0.2}
                    }
                },
                "crisis_band": {
                    "condition": "affect_band == CRISIS",
                    "actions": {
                        "force_deescalation": True,
                        "steering_weight_multiplier": 1.5,
                        "max_tokens": 60
                    }
                }
            }
        }

    def reload_schemas(self):
        """Reload schemas from external file (useful for hot-reloading during tuning)."""
        self.schemas = self.load_schemas()
        return self.schemas

    def set_affect_signals(self, arousal: float = 0.5, valence: float = 0.0, band: str = "GREEN",
                           social_context: str = "private", novelty_score: float = 0.5):
        """
        Set P02 affect signals for the next resolution.
        These will be used by resolve_constitution() to apply situational rules.
        """
        self._affect_arousal = arousal
        self._affect_valence = valence
        self._affect_band = band
        self._social_context = social_context
        self._novelty_score = novelty_score

    def get_affect_signals(self) -> dict:
        """Get currently set affect signals (P02 integration)."""
        return {
            "arousal": getattr(self, "_affect_arousal", 0.5),
            "valence": getattr(self, "_affect_valence", 0.0),
            "band": getattr(self, "_affect_band", "GREEN"),
            "social_context": getattr(self, "_social_context", "private"),
            "novelty_score": getattr(self, "_novelty_score", 0.5),
        }

    def get_family_values(self, key: str) -> dict:
        """Get family values schema by key."""
        return self.schemas.get("family_values", {}).get(key, {})

    def build_constitution_text(self, key: str) -> str:
        """
        Build semantic constitution text from schema content.
        This is what gets encoded - NOT just the key name.
        """
        fv = self.get_family_values(key)
        if not fv:
            return f"Parenting values: {key}"

        parts = []

        # Add description
        if fv.get("description"):
            parts.append(fv["description"])

        # Add core principles from positive tokens
        pos_tokens = fv.get("positive_tokens", {})
        if pos_tokens:
            # Sort by weight, take top 6
            top_values = sorted(pos_tokens.items(), key=lambda x: -x[1])[:6]
            value_words = [w for w, _ in top_values]
            parts.append(f"Core values: {', '.join(value_words)}")

        # Add what to avoid from negative tokens
        neg_tokens = fv.get("negative_tokens", {})
        if neg_tokens:
            avoid_words = [w for w, _ in sorted(neg_tokens.items(), key=lambda x: x[1])[:3]]
            parts.append(f"Avoid: {', '.join(avoid_words)}")

        return ". ".join(parts) if parts else f"Parenting style: {key}"

    def apply_situational_rules(self, affect_arousal: float = 0.5, affect_valence: float = 0.0,
                                  affect_band: str = "GREEN", social_context: str = "private",
                                  novelty_score: float = 0.5) -> dict:
        """Apply situational rules based on P02 signals. Returns action adjustments."""
        rules = self.schemas.get("situational_rules", {})
        applied = {}
        # DO NOT append family_values here - it's already in resolve_constitution()

        if affect_arousal > 0.7 and "high_arousal" in rules:
            applied.update(rules["high_arousal"].get("actions", {}))
            self._current_layers.append("situational:high_arousal")

        if affect_valence < -0.3 and "negative_valence" in rules:
            applied.update(rules["negative_valence"].get("actions", {}))
            self._current_layers.append("situational:negative_valence")

        if affect_band == "CRISIS" and "crisis_band" in rules:
            applied.update(rules["crisis_band"].get("actions", {}))
            self._current_layers.append("situational:crisis_band")

        if social_context == "public" and "public_context" in rules:
            applied.update(rules["public_context"].get("actions", {}))
            self._current_layers.append("situational:public_context")

        if novelty_score > 0.8 and "high_novelty" in rules:
            applied.update(rules["high_novelty"].get("actions", {}))
            self._current_layers.append("situational:high_novelty")

        return applied

    def resolve_constitution(self, family_values_key: str, individual_id: str = None,
                             affect_arousal: float = None, affect_valence: float = None,
                             affect_band: str = None, social_context: str = None,
                             novelty_score: float = None, steering_weight: float = 1.0) -> dict:
        """
        Resolve 3 layers into final generation parameters.
        This is the core method for the new contract-based system.

        Affect signals can be passed directly OR pre-set via set_affect_signals().
        Direct parameters take precedence.
        """
        self._current_layers = []

        # Use stored affect signals if not provided directly
        stored = self.get_affect_signals()
        affect_arousal = affect_arousal if affect_arousal is not None else stored["arousal"]
        affect_valence = affect_valence if affect_valence is not None else stored["valence"]
        affect_band = affect_band if affect_band is not None else stored["band"]
        social_context = social_context if social_context is not None else stored["social_context"]
        novelty_score = novelty_score if novelty_score is not None else stored["novelty_score"]

        # Layer 1: Family Values (base)
        fv = self.get_family_values(family_values_key)
        if not fv:
            logger.warning(f"Family values '{family_values_key}' not found, using gentle_parenting")
            fv = self.get_family_values("gentle_parenting") or {}

        gen_params = fv.get("generation_params", {})
        params = {
            "temperature": gen_params.get("temperature", 0.7),
            "repetition_penalty": gen_params.get("repetition_penalty", 1.1),
            "logits_strength": gen_params.get("logits_strength", 0.5),
            "positive_tokens": dict(fv.get("positive_tokens", {})),
            "negative_tokens": dict(fv.get("negative_tokens", {})),
            "prefix_injection": fv.get("prefix_injection", False),
            "prefix_pattern": fv.get("prefix_pattern", []),
            "steering_weight": steering_weight,
            "max_new_tokens": 96
        }
        self._current_layers.append("family_values")

        # Layer 2: Individual Preferences
        if individual_id:
            ind_prefs = self.schemas.get("individual_prefs", {}).get(individual_id, {})
            if ind_prefs:
                self._current_layers.append("individual")
                if ind_prefs.get("custom_positive_tokens"):
                    params["positive_tokens"].update(ind_prefs["custom_positive_tokens"])
                if ind_prefs.get("custom_negative_tokens"):
                    params["negative_tokens"].update(ind_prefs["custom_negative_tokens"])
                if ind_prefs.get("response_length") == "concise":
                    params["max_new_tokens"] = 60
                elif ind_prefs.get("response_length") == "detailed":
                    params["max_new_tokens"] = 128

        # Layer 3: Situational Rules
        sit_rules = self.apply_situational_rules(
            affect_arousal=affect_arousal,
            affect_valence=affect_valence,
            affect_band=affect_band,
            social_context=social_context,
            novelty_score=novelty_score
        )

        if "temperature_adjustment" in sit_rules:
            params["temperature"] = max(0.1, params["temperature"] + sit_rules["temperature_adjustment"])
        if "steering_weight_multiplier" in sit_rules:
            params["steering_weight"] *= sit_rules["steering_weight_multiplier"]
        if "inject_tokens" in sit_rules:
            params["positive_tokens"].update(sit_rules["inject_tokens"])
        if "max_tokens" in sit_rules:
            params["max_new_tokens"] = min(params["max_new_tokens"], sit_rules["max_tokens"])
        if sit_rules.get("force_deescalation"):
            params["positive_tokens"].update({"calm": 0.4, "safe": 0.3, "okay": 0.2})

        self._current_params = params
        logger.info(f"Resolved constitution: layers={self._current_layers}, temp={params['temperature']:.2f}")
        return params

    def get_token_ids(self, constitution: str):
        """
        Convert schema tokens to token IDs.
        Updated to handle v2 format (nested under family_values).
        """
        # v2 format: schemas.family_values.{key}
        fv = self.schemas.get("family_values", {})
        schema = fv.get(constitution, {}) if fv else self.schemas.get(constitution, {})

        pos_tokens = schema.get("positive_tokens", {})
        neg_tokens = schema.get("negative_tokens", {})

        pos_ids = {}
        neg_ids = {}

        for token_str, weight in pos_tokens.items():
            token_ids = self.tokenizer.encode(token_str, add_special_tokens=False)
            # For multi-token phrases, apply weight to EACH token
            for tid in token_ids:
                if tid not in pos_ids:
                    pos_ids[tid] = weight
                else:
                    # Take max weight if token appears in multiple phrases
                    pos_ids[tid] = max(pos_ids[tid], weight)

        for token_str, weight in neg_tokens.items():
            token_ids = self.tokenizer.encode(token_str, add_special_tokens=False)
            # For multi-token phrases, apply penalty to EACH token
            for tid in token_ids:
                if tid not in neg_ids:
                    neg_ids[tid] = weight
                else:
                    # Take min (most negative) weight if token appears in multiple phrases
                    neg_ids[tid] = min(neg_ids[tid], weight)

        return {"positive": pos_ids, "negative": neg_ids}

    def create_logit_processor(self, constitution: str, strength: float = 1.0, resolved_params: dict = None):
        """
        Create a logits processor for this constitution.
        If resolved_params provided, uses those tokens instead of just family_values.
        """
        if resolved_params:
            # Use merged tokens from 3-layer resolution
            pos_tokens = resolved_params.get("positive_tokens", {})
            neg_tokens = resolved_params.get("negative_tokens", {})
            token_ids = {"positive": {}, "negative": {}}

            for token_str, weight in pos_tokens.items():
                ids = self.tokenizer.encode(token_str, add_special_tokens=False)
                for tid in ids:
                    token_ids["positive"][tid] = max(token_ids["positive"].get(tid, 0), weight)

            for token_str, weight in neg_tokens.items():
                ids = self.tokenizer.encode(token_str, add_special_tokens=False)
                for tid in ids:
                    token_ids["negative"][tid] = min(token_ids["negative"].get(tid, 0), weight)
        else:
            token_ids = self.get_token_ids(constitution)

        def process_logits(input_ids, scores):
            # Apply constitution token biases (always)
            for token_id, weight in token_ids["positive"].items():
                scores[:, token_id] += (weight * strength)
            for token_id, weight in token_ids["negative"].items():
                scores[:, token_id] += (weight * strength)
            return scores

        return process_logits

    def get_layers_applied(self) -> list:
        """Return list of layers that were applied in the last resolution."""
        return self._current_layers.copy() if self._current_layers else []


def create_prefix_from_pattern(pattern: list, device: torch.device, num_tokens: int = 8, hidden_size: int = 768):
    """
    Create prefix tensor from schema-defined pattern.
    This is the canonical way to create prefix - uses pattern from resolved constitution.
    """
    if not pattern:
        pattern = [0.0] * 8

    # Create tensor [num_tokens, hidden_size]
    pattern_tensor = torch.tensor(pattern, device=device, dtype=torch.float16)

    # Repeat pattern to fill hidden_size (768)
    repeats = (hidden_size // len(pattern)) + 1
    full_vector = pattern_tensor.repeat(repeats)[:hidden_size]

    # Create num_tokens copies
    prefix = full_vector.unsqueeze(0).repeat(num_tokens, 1)  # [num_tokens, 768]

    # Add batch dim
    prefix = prefix.unsqueeze(0)  # [1, num_tokens, 768]

    return prefix


def detect_emotional_intensity(text: str) -> dict:
    """
    Detect emotional intensity from input text.
    Returns intensity score (0.0-1.0) and detected emotions.
    """
    text_lower = text.lower()

    # Emotion patterns with intensity weights
    emotion_patterns = {
        # High intensity emotions (weight: 1.0)
        "furious": 1.0, "enraged": 1.0, "hate": 1.0, "despise": 1.0,
        "terrified": 1.0, "panic": 1.0, "hysterical": 1.0,
        "devastated": 1.0, "heartbroken": 1.0, "suicidal": 1.0,

        # High-medium intensity (weight: 0.8)
        "angry": 0.8, "furious": 0.8, "screaming": 0.8, "yelling": 0.8,
        "crying": 0.8, "sobbing": 0.8, "tantrum": 0.8, "meltdown": 0.8,
        "frustrated": 0.8, "overwhelmed": 0.8, "exhausted": 0.8,

        # Medium intensity (weight: 0.6)
        "upset": 0.6, "annoyed": 0.6, "irritated": 0.6, "stressed": 0.6,
        "worried": 0.6, "anxious": 0.6, "scared": 0.6, "afraid": 0.6,
        "sad": 0.6, "hurt": 0.6, "disappointed": 0.6,

        # Low-medium intensity (weight: 0.4)
        "concerned": 0.4, "confused": 0.4, "uncertain": 0.4,
        "uncomfortable": 0.4, "uneasy": 0.4,

        # Low intensity (weight: 0.2)
        "curious": 0.2, "wondering": 0.2, "thinking": 0.2,
    }

    # Intensifier words that multiply the score
    intensifiers = {
        "very": 1.3, "extremely": 1.5, "so": 1.2, "really": 1.3,
        "incredibly": 1.5, "absolutely": 1.4, "completely": 1.4,
        "totally": 1.3, "super": 1.2,
    }

    # Detect emotions and calculate intensity
    detected_emotions = []
    max_intensity = 0.0

    for emotion, weight in emotion_patterns.items():
        # Use word boundary regex for accurate matching
        pattern = r'\b' + re.escape(emotion) + r'\b'
        if re.search(pattern, text_lower):
            detected_emotions.append(emotion)
            max_intensity = max(max_intensity, weight)

    # Check for intensifiers
    intensifier_multiplier = 1.0
    for intensifier, mult in intensifiers.items():
        pattern = r'\b' + re.escape(intensifier) + r'\b'
        if re.search(pattern, text_lower):
            intensifier_multiplier = max(intensifier_multiplier, mult)

    # Apply intensifier and cap at 1.0
    final_intensity = min(1.0, max_intensity * intensifier_multiplier)

    # Check for exclamation marks (adds 0.1 per mark, capped)
    exclamation_count = text.count('!')
    if exclamation_count > 0:
        final_intensity = min(1.0, final_intensity + (exclamation_count * 0.05))

    # Check for ALL CAPS words (adds intensity)
    caps_words = len([w for w in text.split() if w.isupper() and len(w) > 2])
    if caps_words > 0:
        final_intensity = min(1.0, final_intensity + (caps_words * 0.1))

    return {
        "intensity": final_intensity,
        "detected_emotions": detected_emotions,
        "intensifier_multiplier": intensifier_multiplier,
    }


def calculate_adaptive_weight(text: str, base_weight: float = 1.0) -> float:
    """
    Calculate adaptive constitution weight based on emotional intensity.

    Higher emotional intensity -> Higher constitution weight
    This ensures the constitution has more influence during emotional scenarios.

    Weight range: base_weight to base_weight + 0.5
    """
    emotion_info = detect_emotional_intensity(text)
    intensity = emotion_info["intensity"]

    # Scale: base_weight + (intensity * 0.5)
    # e.g., base=1.0, intensity=0.8 -> weight=1.4
    adaptive_weight = base_weight + (intensity * 0.5)

    return adaptive_weight, emotion_info


@torch.inference_mode()
def generate_counterfactual(
    text: str,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    tokenizer,
    device: torch.device,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    no_repeat_ngram_size: int = 2,
    num_candidates: int = 1,
    normalization_method: str = "none",
    constitution_text: str = None,
    split_encoding: bool = False,
    constitution_weight: float = 1.0,
    constitution_controller: ConstitutionController = None,
    schema_strength: float = 1.5,
    constitution_key: str = None,
    # Affect signals from input (P02/P03)
    affect_arousal: float = 0.5,
    affect_valence: float = 0.0,
    affect_band: str = "GREEN",
    social_context: str = "private",
) -> list[str]:
    """
    Generate counterfactual response for input text.

    Args:
        text: Input scenario text
        encoder: UltraBERT encoder model
        decoder: GPT-2 decoder head
        tokenizer: Tokenizer instance
        device: torch device
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Nucleus sampling
        repetition_penalty: Penalty for repeated tokens
        no_repeat_ngram_size: N-gram repetition prevention
        num_candidates: Number of parallel candidates
        normalization_method: Embedding normalization method
        constitution_text: Raw constitution text (legacy)
        split_encoding: Encode constitution separately
        constitution_weight: Base weight for constitution steering
        constitution_controller: Controller instance
        schema_strength: Logits bias strength
        constitution_key: Constitution family values key
        affect_arousal: P02 arousal signal (0.0-1.0)
        affect_valence: P02 valence signal (-1.0 to 1.0)
        affect_band: Affect band (GREEN/AMBER/RED/CRISIS)
        social_context: Context (private/family/public)

    Returns:
        List of generated counterfactual texts
    """

    # Helper to encode text
    def encode_text(txt):
        inputs = tokenizer(
            txt,
            return_tensors="pt",
            max_length=256,
            truncation=True,
            padding=True,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16):
            outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state

        return hidden, attention_mask

    # Determine controller key
    controller_key = constitution_key if constitution_key else constitution_text

    # ==========================================================================
    # SINGLE RESOLUTION POINT: Resolve constitution once, use everywhere
    # ==========================================================================
    resolved_params = None
    logits_processor = None

    if controller_key and constitution_controller:
        # Check if key exists in family_values (correct v2 path)
        family_values = constitution_controller.schemas.get("family_values", {})
        if controller_key in family_values:
            logger.debug(f"Resolving 3-Layer Constitution for: {controller_key}")

            # Resolve once - this merges family_values + individual + situational
            resolved_params = constitution_controller.resolve_constitution(
                family_values_key=controller_key,
                affect_arousal=affect_arousal,
                affect_valence=affect_valence,
                affect_band=affect_band,
                social_context=social_context,
                steering_weight=constitution_weight
            )

            # Override generation params from resolved constitution
            temperature = resolved_params["temperature"]
            repetition_penalty = resolved_params["repetition_penalty"]
            max_new_tokens = resolved_params.get("max_new_tokens", max_new_tokens)

            layers = constitution_controller.get_layers_applied()
            logger.debug(f"Layers Applied: {layers}")
            logger.debug(f"Temperature: {temperature:.2f}, Rep Penalty: {repetition_penalty:.2f}")

            # Create logits processor with RESOLVED params (includes situational tokens)
            _logits_strength = resolved_params.get("logits_strength", 0.0)
            if _logits_strength and _logits_strength > 0:
                from transformers import LogitsProcessorList, LogitsProcessor

                class ConstitutionLogitsProcessor(LogitsProcessor):
                    def __init__(self, processor_fn):
                        self.processor_fn = processor_fn

                    def __call__(self, input_ids, scores):
                        return self.processor_fn(input_ids, scores)

                proc_fn = constitution_controller.create_logit_processor(
                    controller_key,
                    strength=_logits_strength,
                    resolved_params=resolved_params,
                )
                logits_processor = LogitsProcessorList([ConstitutionLogitsProcessor(proc_fn)])
                logger.debug(f"Logits Processor Active (Strength: {_logits_strength})")

    # ==========================================================================
    # BUILD SEMANTIC CONSTITUTION TEXT (not just key name)
    # ==========================================================================
    constitution_semantic_text = constitution_text  # Default to passed value
    if resolved_params and controller_key and constitution_controller:
        # Build rich semantic text from schema content
        constitution_semantic_text = constitution_controller.build_constitution_text(controller_key)
        logger.debug(f"Constitution Text: {constitution_semantic_text[:80]}...")

    # Encode input
    with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16):

        if split_encoding and (constitution_text or constitution_semantic_text):
            # 1. Encode Constitution Separately (using SEMANTIC text, not just key)
            actual_const_text = constitution_semantic_text or constitution_text
            logger.debug(f"Encoding Constitution: {actual_const_text[:60]}...")
            const_hidden, const_mask = encode_text(f"[CONSTITUTION: {actual_const_text}]")

            # 2. Encode Scenario Separately
            logger.debug("Encoding Scenario separately...")
            scen_hidden, scen_mask = encode_text(text)

            # 3. Normalize Independently (Crucial for signal preservation)
            # NOTE: unit_norm breaks split encoding - use clamp methods instead
            effective_norm = normalization_method
            if normalization_method == "unit_norm":
                effective_norm = "clamp_tight"  # Auto-fix for split encoding
                logger.debug("Split encoding: auto-switching unit_norm -> clamp_tight")

            if effective_norm != "none":
                const_hidden = normalize_embeddings(const_hidden, method=effective_norm)
                scen_hidden = normalize_embeddings(scen_hidden, method=effective_norm)
                logger.debug(f"Applied independent {effective_norm} normalization")

            # 4. Calculate Adaptive Weight based on emotional intensity
            adaptive_weight, emotion_info = calculate_adaptive_weight(text, base_weight=constitution_weight)
            if emotion_info["detected_emotions"]:
                logger.debug(f"Detected emotions: {emotion_info['detected_emotions']} (intensity: {emotion_info['intensity']:.2f})")
                logger.debug(f"Adaptive constitution weight: {constitution_weight:.2f} -> {adaptive_weight:.2f}")

            # 5. Apply Weighting
            if adaptive_weight != 1.0:
                const_hidden = const_hidden * adaptive_weight

            # 5. Concatenate
            encoder_hidden = torch.cat([const_hidden, scen_hidden], dim=1)
            attention_mask = torch.cat([const_mask, scen_mask], dim=1)

            # 6. Inject Prefix if resolved_params has prefix_injection enabled
            if resolved_params and resolved_params.get("prefix_injection"):
                prefix_pattern = resolved_params.get("prefix_pattern", [])

                if prefix_pattern:
                    # Use pattern from resolved constitution (not hardcoded)
                    prefix = create_prefix_from_pattern(prefix_pattern, device)
                    # Expand prefix to match batch size
                    prefix = prefix.repeat(encoder_hidden.shape[0], 1, 1)

                    # Concatenate: [PREFIX] + [ENCODER_OUTPUT]
                    encoder_hidden = torch.cat([prefix, encoder_hidden], dim=1)

                    # Adjust attention mask
                    prefix_mask = torch.ones((attention_mask.shape[0], prefix.shape[1]), device=device, dtype=attention_mask.dtype)
                    attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
                    logger.debug(f"Injected Schema Prefix: {prefix.shape}")
            elif resolved_params:
                logger.debug(f"Prefix Injection DISABLED for {controller_key}")

            logger.debug(f"Fused Embeddings Shape: {encoder_hidden.shape}")

        else:
            # Standard Joint Encoding
            full_text = text
            if constitution_text and not split_encoding:
                # If not split, it should have been prepended by caller, but let's ensure
                if "CONSTITUTION" not in text:
                    full_text = f"[CONSTITUTION: {constitution_text}] {text}"

            inputs = tokenizer(
                full_text,
                return_tensors="pt",
                max_length=256,
                truncation=True,
                padding=True,
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            encoder_outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            encoder_hidden = encoder_outputs.last_hidden_state

            # Debug: Log embedding stats
            logger.debug(f"UltraBERT embedding stats - Shape: {encoder_hidden.shape}, Mean: {encoder_hidden.mean().item():.4f}, Std: {encoder_hidden.std().item():.4f}")

            # Apply normalization
            if normalization_method != "none":
                encoder_hidden = normalize_embeddings(encoder_hidden, method=normalization_method)
                logger.debug(f"Applied {normalization_method} normalization - Mean: {encoder_hidden.mean().item():.4f}, Std: {encoder_hidden.std().item():.4f}")

        # Expand for multiple candidates (Parallel Sampling)
        if num_candidates > 1:
            encoder_hidden = encoder_hidden.repeat_interleave(num_candidates, dim=0)
            attention_mask = attention_mask.repeat_interleave(num_candidates, dim=0)

        # Resolve EOS and PAD token IDs with proper fallback
        eos_token_id = tokenizer.sep_token_id or tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id
        if eos_token_id is None:
            eos_token_id = 50256  # GPT-2 default EOS
            logger.warning(f"Tokenizer missing eos_token_id, using GPT-2 default: {eos_token_id}")
        if pad_token_id is None:
            pad_token_id = eos_token_id  # Common practice: use EOS as PAD
            logger.warning(f"Tokenizer missing pad_token_id, using eos_token_id: {pad_token_id}")

        # Generate with decoder
        generated_ids = decoder.generate(
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            logits_processor=logits_processor,
        )

    # Decode output
    generated_texts = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    # Post-processing: Clean endings
    cleaned_texts = []
    irrelevant_phrases = [
        "changed the subject",
        "partner",
        "chore chart",
        "being forgotten",
        "household chores",
        "in summary",
        "ultimately",
        "the key takeaway",
    ]

    for text in generated_texts:
        # 1. Truncate at double newline (often indicates end of thought)
        if "\n\n" in text:
            text = text.split("\n\n")[0]

        # 2. Remove irrelevant topic shifts
        text_lower = text.lower()
        for phrase in irrelevant_phrases:
            if phrase in text_lower:
                idx = text_lower.find(phrase)
                # Keep only up to the previous sentence
                # Find the last sentence boundary before the irrelevant phrase
                pre_text = text[:idx]
                last_punct = max(pre_text.rfind("."), pre_text.rfind("!"), pre_text.rfind("?"))
                if last_punct != -1:
                    text = text[:last_punct + 1]
                else:
                    # If no punctuation found, just cut it off (fallback)
                    text = pre_text.strip()
                break

        # 3. Ensure it ends with punctuation
        if text and text[-1] not in ".!?":
            last_punct = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
            if last_punct != -1:
                text = text[:last_punct + 1]

        cleaned_texts.append(text)

    return cleaned_texts


def generate_from_contract(
    contract_input: dict,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    tokenizer,
    device: torch.device,
    constitution_controller: ConstitutionController = None,
    normalization_method: str = "unit_norm",
    model_version: str = "ultrabert-gen-decoder-v4",
) -> dict:
    """
    Contract-compliant API for R5 pipeline integration.

    Accepts input per contracts/input_schema.json
    Returns output per contracts/output_schema.json

    Args:
        contract_input: Dict matching input_schema.json
        encoder: UltraBERT encoder model
        decoder: GPT-2 decoder head
        tokenizer: Tokenizer instance
        device: torch device
        constitution_controller: Optional controller (will create if None)
        normalization_method: Embedding normalization method
        model_version: Model version string for metadata

    Returns:
        Dict matching output_schema.json
    """
    start_time = time.perf_counter()

    # Validate required fields
    event_id = contract_input.get("event_id")
    text = contract_input.get("text")
    constitution = contract_input.get("constitution", {})

    if not event_id or not text:
        return {
            "event_id": event_id or "unknown",
            "counterfactual": "",
            "generation_meta": {
                "constitution_applied": "",
                "latency_ms": 0,
            },
            "error": {
                "code": "INVALID_INPUT",
                "message": "Missing required fields: event_id and text are required"
            }
        }

    # Extract 3 layers from constitution
    family_values = constitution.get("family_values", {})
    individual = constitution.get("individual", {})
    situational = constitution.get("situational", {})

    # Get family values key
    constitution_key = family_values.get("key", "gentle_parenting")

    # Extract generation params from family_values (already resolved by P03)
    temperature = family_values.get("temperature", 0.7)
    repetition_penalty = family_values.get("repetition_penalty", 1.1)
    logits_strength = family_values.get("logits_strength", 0.5)
    prefix_injection = family_values.get("prefix_injection", False)
    prefix_pattern = family_values.get("prefix_pattern", [])
    positive_tokens = family_values.get("positive_tokens", {})
    negative_tokens = family_values.get("negative_tokens", {})

    # Extract situational signals
    affect_arousal = situational.get("affect_arousal", 0.5)
    affect_valence = situational.get("affect_valence", 0.0)
    affect_band = situational.get("affect_band", "GREEN")
    social_context = situational.get("social_context", "private")
    steering_weight = situational.get("steering_weight", 1.0)
    force_deescalation = situational.get("force_deescalation", False)
    inject_empathy_prefix = situational.get("inject_empathy_prefix", False)
    temp_adjustment = situational.get("temperature_adjustment", 0.0)

    # Apply temperature adjustment from situational layer
    temperature = max(0.1, temperature + temp_adjustment)

    # Extract individual prefs
    response_length = individual.get("response_length", "moderate")
    max_new_tokens = {"concise": 60, "moderate": 96, "detailed": 128}.get(response_length, 96)

    # Apply situational max_tokens override if present
    if "max_tokens" in situational:
        max_new_tokens = min(max_new_tokens, situational["max_tokens"])

    # Merge tokens from all layers
    if force_deescalation:
        positive_tokens = {**positive_tokens, "calm": 0.4, "safe": 0.3, "okay": 0.2}

    if individual.get("custom_positive_tokens"):
        positive_tokens = {**positive_tokens, **individual["custom_positive_tokens"]}
    if individual.get("custom_negative_tokens"):
        negative_tokens = {**negative_tokens, **individual["custom_negative_tokens"]}

    # Track layers applied
    layers_applied = ["family_values"]
    if individual:
        layers_applied.append("individual")
    if situational:
        layers_applied.append("situational")

    # Build logits processor from merged tokens
    logits_processor = None
    logits_bias_applied = []

    if logits_strength > 0 and (positive_tokens or negative_tokens):
        from transformers import LogitsProcessorList, LogitsProcessor

        token_ids = {"positive": {}, "negative": {}}

        for token_str, weight in positive_tokens.items():
            ids = tokenizer.encode(token_str, add_special_tokens=False)
            for tid in ids:
                token_ids["positive"][tid] = max(token_ids["positive"].get(tid, 0), weight)
            logits_bias_applied.append(token_str)

        for token_str, weight in negative_tokens.items():
            ids = tokenizer.encode(token_str, add_special_tokens=False)
            for tid in ids:
                token_ids["negative"][tid] = min(token_ids["negative"].get(tid, 0), weight)

        class ContractLogitsProcessor(LogitsProcessor):
            def __init__(self, pos_ids, neg_ids, strength):
                self.pos_ids = pos_ids
                self.neg_ids = neg_ids
                self.strength = strength

            def __call__(self, input_ids, scores):
                for token_id, weight in self.pos_ids.items():
                    scores[:, token_id] += (weight * self.strength)
                for token_id, weight in self.neg_ids.items():
                    scores[:, token_id] += (weight * self.strength)
                return scores

        logits_processor = LogitsProcessorList([
            ContractLogitsProcessor(token_ids["positive"], token_ids["negative"], logits_strength)
        ])

    try:
        # Encode text
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=256,
            truncation=True,
            padding=True,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.inference_mode():
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16):
                encoder_outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
                encoder_hidden = encoder_outputs.last_hidden_state

                # Normalize embeddings
                if normalization_method != "none":
                    encoder_hidden = normalize_embeddings(encoder_hidden, method=normalization_method)

                # Inject prefix if enabled
                if prefix_injection and prefix_pattern:
                    prefix = create_prefix_from_pattern(prefix_pattern, device)
                    encoder_hidden = torch.cat([prefix, encoder_hidden], dim=1)
                    prefix_mask = torch.ones((attention_mask.shape[0], prefix.shape[1]), device=device, dtype=attention_mask.dtype)
                    attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

                # Resolve token IDs
                eos_token_id = tokenizer.sep_token_id or tokenizer.eos_token_id or 50256
                pad_token_id = tokenizer.pad_token_id or eos_token_id

                # Generate
                generated_ids = decoder.generate(
                    encoder_hidden_states=encoder_hidden,
                    encoder_attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=50,
                    top_p=0.9,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=2,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    logits_processor=logits_processor,
                )

        # Decode
        counterfactual = tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # Clean output
        if "\n\n" in counterfactual:
            counterfactual = counterfactual.split("\n\n")[0]
        if counterfactual and counterfactual[-1] not in ".!?":
            last_punct = max(counterfactual.rfind("."), counterfactual.rfind("!"), counterfactual.rfind("?"))
            if last_punct != -1:
                counterfactual = counterfactual[:last_punct + 1]

        tokens_generated = len(generated_ids[0])

    except Exception as e:
        logger.error(f"Generation failed for event {event_id}: {e}")
        return {
            "event_id": event_id,
            "counterfactual": "",
            "generation_meta": {
                "constitution_applied": constitution_key,
                "latency_ms": int((time.perf_counter() - start_time) * 1000),
            },
            "error": {
                "code": "MODEL_ERROR",
                "message": str(e)
            }
        }

    latency_ms = int((time.perf_counter() - start_time) * 1000)

    return {
        "event_id": event_id,
        "counterfactual": counterfactual,
        "generation_meta": {
            "constitution_applied": constitution_key,
            "steering_weight_used": steering_weight,
            "temperature_final": temperature,
            "repetition_penalty_final": repetition_penalty,
            "tokens_generated": tokens_generated,
            "latency_ms": latency_ms,
            "model_version": model_version,
        },
        "trace": {
            "layers_applied": layers_applied,
            "deescalation_triggered": force_deescalation,
            "empathy_prefix_injected": inject_empathy_prefix,
            "logits_bias_applied": logits_bias_applied,
            "normalization_method": normalization_method,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="FP16 Decoder Inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="D:\\Modeling_studio\\outputs\\ultrabert-gen-decoder-v4",
        help="Path to decoder checkpoint",
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Input text to generate counterfactual for",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=96,
        help="Maximum tokens to generate (default: 96)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (default: 0.3)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling (default: 50)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling (default: 0.9)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.2,
        help="Repetition penalty (default: 1.2)",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=2,
        help="No repeat n-gram size (default: 2)",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=1,
        help="Number of candidates to generate (parallel sampling)",
    )
    parser.add_argument(
        "--normalize-embeddings",
        choices=["none", "unit_norm", "standardize", "clamp", "clamp_tight", "clamp_extreme", "layer_norm", "smooth_clamp"],
        default="unit_norm",
        help="Normalize UltraBERT embeddings before GPT-2",
    )
    parser.add_argument(
        "--constitution",
        type=str,
        default=None,
        help="Family constitution key (from data/family_constitutions.json) or raw text",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode",
    )
    parser.add_argument(
        "--split-encoding",
        action="store_true",
        help="Encode constitution and scenario separately (prevents dilution)",
    )
    parser.add_argument(
        "--constitution-weight",
        type=float,
        default=0.9,
        help="Weight for constitution embeddings in split encoding (default: 0.9)",
    )
    parser.add_argument(
        "--schema-strength",
        type=float,
        default=1.5,
        help="Strength of constitution schema bias (default: 1.5)",
    )
    args = parser.parse_args()

    # Get device
    device = torch.device("cpu") if args.cpu else get_device()
    logger.info(f"Using device: {device}")

    # Load model
    encoder, decoder, tokenizer = load_model_fp16(args.checkpoint, device)

    # Initialize Controller
    controller = ConstitutionController(tokenizer)

    if args.interactive:
        # Interactive mode
        logger.info("Starting UltraBERT-Gen Decoder - Interactive Mode")
        logger.info("Type 'quit' or 'exit' to stop")

        while True:
            try:
                text = input("Input: ").strip()
                if text.lower() in ["quit", "exit", "q"]:
                    break
                if not text:
                    continue

                outputs = generate_counterfactual(
                    text=text,
                    encoder=encoder,
                    decoder=decoder,
                    tokenizer=tokenizer,
                    device=device,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    num_candidates=args.num_candidates,
                    normalization_method=args.normalize_embeddings,
                    constitution_text=args.constitution,
                    split_encoding=args.split_encoding,
                    constitution_weight=args.constitution_weight,
                    constitution_controller=controller,
                    schema_strength=args.schema_strength,
                )

                for i, out in enumerate(outputs):
                    logger.info(f"Output {i+1}: {out}")

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error: {e}")
    else:
        if not args.text:
            logger.error("--text is required for non-interactive mode")
            return

        outputs = generate_counterfactual(
            text=args.text,
            encoder=encoder,
            decoder=decoder,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            num_candidates=args.num_candidates,
            normalization_method=args.normalize_embeddings,
            constitution_text=args.constitution,
            split_encoding=args.split_encoding,
            constitution_weight=args.constitution_weight,
            constitution_controller=controller,
            schema_strength=args.schema_strength,
        )

        for i, out in enumerate(outputs):
            logger.info(f"Output {i+1}: {out}")


if __name__ == "__main__":
    main()
