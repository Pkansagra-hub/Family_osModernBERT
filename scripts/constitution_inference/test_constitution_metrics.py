
import sys
import os
import re
import torch
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Add src to path (3 levels up from scripts/constitution_inference/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from infer_decoder_fp16 import load_model_fp16, generate_counterfactual, get_device, ConstitutionController

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Path to v2 constitution schemas
CONSTITUTION_SCHEMAS_V2_PATH = Path(__file__).parent.parent.parent / "data" / "constitutions" / "constitution_schemas_v2.json"

# Try to import TextBlob for sentiment analysis
try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False
    logger.warning("TextBlob not found. Using simple sentiment fallback.")

class ConstitutionMetrics:
    """Metrics calculator for 3-layer constitution alignment."""

    def __init__(self):
        # Load keywords from v2 schemas if available
        self.keywords = self._load_keywords_from_schema()

        # Expected sentiment polarity ranges (approximate)
        self.sentiment_targets = {
            "traditional_strict": (-0.1, 0.3),
            "gentle_parenting": (0.2, 0.6),
            "indian_joint_family": (0.1, 0.5),
            "balanced_approach": (0.1, 0.5),
            "authoritative": (0.0, 0.4),
            "default": (0.0, 0.4)
        }

    def _load_keywords_from_schema(self) -> dict:
        """Load keywords from v2 schema file."""
        if CONSTITUTION_SCHEMAS_V2_PATH.exists():
            try:
                with open(CONSTITUTION_SCHEMAS_V2_PATH, "r", encoding="utf-8") as f:
                    schemas = json.load(f)

                keywords = {}
                family_values = schemas.get("family_values", {})

                for key, config in family_values.items():
                    pos_tokens = config.get("positive_tokens", {})
                    # Convert token weights to keyword weights (higher = more important)
                    kw = {token: max(1, int(weight * 3)) for token, weight in pos_tokens.items()}
                    keywords[key] = kw

                logger.info(f"Loaded keywords for {len(keywords)} constitutions from v2 schema")
                return keywords
            except Exception as e:
                logger.warning(f"Failed to load v2 schema: {e}")

        # Fallback to hardcoded
        return {
            "traditional_strict": {
                "must": 2, "rule": 2, "respect": 2, "consequence": 2, "obey": 2,
                "clear": 1, "firm": 1, "limit": 1, "structure": 1, "boundary": 1
            },
            "gentle_parenting": {
                "feel": 2, "understand": 2, "connect": 2, "together": 2, "emotion": 2,
                "help": 1, "listen": 1, "support": 1, "calm": 1, "validate": 1
            },
            "indian_joint_family": {
                "elder": 3, "family": 2, "respect": 2, "harmony": 2, "community": 2,
                "adjust": 1, "blessing": 1, "together": 1, "home": 1, "values": 1
            },
            "balanced_approach": {
                "both": 2, "balance": 2, "clear": 1, "feel": 1, "understand": 1,
                "consequence": 1, "boundary": 1, "connect": 1
            },
            "authoritative": {
                "expect": 2, "explain": 2, "discuss": 1, "reason": 1, "understand": 1,
                "rule": 1, "respect": 1, "together": 1
            },
            "default": {}
        }

    def get_sentiment(self, text: str) -> float:
        if HAS_TEXTBLOB:
            return TextBlob(text).sentiment.polarity
        else:
            # Simple fallback
            positive = ["good", "great", "love", "happy", "understand", "help", "support", "calm", "respect"]
            negative = ["bad", "wrong", "stop", "no", "consequence", "limit", "firm", "strict"]
            words = text.lower().split()
            score = 0
            for w in words:
                if w in positive: score += 1
                if w in negative: score -= 0.5 # Strict words aren't necessarily negative sentiment, just firm
            return max(-1.0, min(1.0, score / (len(words) + 1) * 5))

    def calculate_keyword_score(self, text: str, constitution: str) -> float:
        """
        Calculate keyword score using regex word-boundary matching.
        This prevents false positives like 'helpful' matching 'help'.
        """
        if constitution not in self.keywords:
            return 0.0, []

        text_lower = text.lower()
        score = 0
        max_possible = 10  # Cap at 10 for normalization

        target_words = self.keywords[constitution]
        found_words = []

        for word, weight in target_words.items():
            # Use word boundary regex for accurate matching
            # \b ensures we match whole words only
            pattern = r'\b' + re.escape(word) + r'\b'
            matches = re.findall(pattern, text_lower)
            if matches:
                # Count occurrences (reward multiple uses)
                occurrence_bonus = min(len(matches), 3)  # Cap at 3x
                score += weight * occurrence_bonus
                found_words.append(f"{word}({len(matches)}x)" if len(matches) > 1 else word)

        normalized_score = min(100, (score / max_possible) * 100)
        return normalized_score, found_words

    def calculate_cas(self, text: str, constitution: str) -> Dict[str, Any]:
        """
        Calculate Constitution Alignment Score (CAS) with length normalization.
        """
        # 1. Keyword Match (35%) - reduced from 40%
        kw_score, found_kws = self.calculate_keyword_score(text, constitution)

        # 2. Sentiment Alignment (25%) - reduced from 30%
        sentiment = self.get_sentiment(text)
        target_range = self.sentiment_targets.get(constitution, (-1, 1))

        if target_range[0] <= sentiment <= target_range[1]:
            sent_score = 100
        else:
            dist = min(abs(sentiment - target_range[0]), abs(sentiment - target_range[1]))
            sent_score = max(0, 100 - (dist * 100))

        # 3. Length/Complexity with better normalization (25%)
        length = len(text.split())

        # Optimal length ranges per constitution
        optimal_ranges = {
            "traditional_strict": (15, 50),   # Direct but complete
            "gentle_parenting": (25, 70),     # Explanatory
            "indian_joint_family": (20, 60),  # Balanced
        }
        opt_min, opt_max = optimal_ranges.get(constitution, (15, 60))

        if opt_min <= length <= opt_max:
            len_score = 100
        elif length < opt_min:
            # Penalize short outputs less harshly
            len_score = max(50, 100 - (opt_min - length) * 3)
        else:
            # Penalize long outputs gently
            len_score = max(60, 100 - (length - opt_max) * 2)

        # 4. Fluency Bonus (15%) - NEW: rewards complete sentences
        fluency_score = 100
        # Check for proper ending
        if not text or text[-1] not in '.!?':
            fluency_score -= 20
        # Check for mid-sentence cutoff indicators
        if text.endswith(('the', 'a', 'an', 'to', 'and', 'or', 'but', 'with')):
            fluency_score -= 30
        # Reward longer, complete outputs
        if length >= 30 and text[-1] in '.!?':
            fluency_score = min(100, fluency_score + 10)

        # Weighted Total (35 + 25 + 25 + 15 = 100)
        total_cas = (kw_score * 0.35) + (sent_score * 0.25) + (len_score * 0.25) + (fluency_score * 0.15)

        return {
            "cas": total_cas,
            "keyword_score": kw_score,
            "sentiment_score": sent_score,
            "length_score": len_score,
            "fluency_score": fluency_score,
            "sentiment_val": sentiment,
            "found_keywords": found_kws,
            "word_count": length
        }

def load_split_model(encoder_path: str, decoder_path: str, device: torch.device):
    """
    Load encoder from one path and decoder from another.
    """
    from transformers import AutoTokenizer, AutoConfig, ModernBertModel
    from modeling_studio.models.decoder_gpt2 import GPT2DecoderHead
    from modeling_studio.models.decoder_gpt2_config import GPT2DecoderConfig
    from safetensors.torch import load_file
    from safetensors import safe_open
    import gc

    encoder_path = Path(encoder_path)
    decoder_path = Path(decoder_path)

    # -------------------------------------------------------------------------
    # 1. Load Decoder Weights FIRST (to minimize peak RAM usage)
    # -------------------------------------------------------------------------
    logger.info(f"Loading decoder weights from {decoder_path}...")
    decoder_state_dict = {}

    try:
        if (decoder_path / "model.safetensors").exists():
            logger.info("Found model.safetensors, using safe_open")
            with safe_open(decoder_path / "model.safetensors", framework="pt", device="cpu") as f:
                for key in f.keys():
                    if "counterfactual" in key:
                        if key.startswith("heads.counterfactual."):
                            new_key = key.replace("heads.counterfactual.", "")
                        elif key.startswith("head.counterfactual."):
                            new_key = key.replace("head.counterfactual.", "")
                        else:
                            new_key = key
                        decoder_state_dict[new_key] = f.get_tensor(key)

        elif (decoder_path / "pytorch_model.bin").exists():
            logger.info("Found pytorch_model.bin")
            state_dict = torch.load(decoder_path / "pytorch_model.bin", map_location="cpu")
            for key, value in state_dict.items():
                if "counterfactual" in key:
                    if key.startswith("heads.counterfactual."):
                        new_key = key.replace("heads.counterfactual.", "")
                    elif key.startswith("head.counterfactual."):
                        new_key = key.replace("head.counterfactual.", "")
                    else:
                        new_key = key
                    decoder_state_dict[new_key] = value
            del state_dict
        else:
            raise FileNotFoundError(f"No weights found in {decoder_path}")
    except Exception as e:
        logger.error(f"Failed to load decoder weights: {e}")
        raise e

    logger.info(f"Loaded {len(decoder_state_dict)} decoder keys")
    gc.collect()

    # -------------------------------------------------------------------------
    # 2. Initialize Decoder
    # -------------------------------------------------------------------------
    logger.info("Using default GPT2DecoderConfig...")
    decoder_config = GPT2DecoderConfig(
        gpt2_model_name="gpt2-medium",
        encoder_hidden_size=768,
        projection_hidden_size=1024,
        num_prefix_tokens=16,
        freeze_layers=12
    )

    logger.info("Initializing GPT2DecoderHead...")
    decoder = GPT2DecoderHead(config=decoder_config, encoder_hidden_size=768)

    logger.info("Loading keys into decoder...")
    missing, unexpected = decoder.load_state_dict(decoder_state_dict, strict=False)
    if missing:
        logger.info(f"Missing decoder keys: {len(missing)}")
    if unexpected:
        logger.info(f"Unexpected decoder keys: {len(unexpected)}")

    del decoder_state_dict
    gc.collect()

    logger.info("Moving decoder to device...")
    decoder = decoder.to(device).half().eval()

    # -------------------------------------------------------------------------
    # 3. Load Encoder
    # -------------------------------------------------------------------------
    logger.info(f"Loading Encoder from: {encoder_path}")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(encoder_path, trust_remote_code=True)

    # Load Encoder Weights
    logger.info("Loading encoder state dict...")
    if (encoder_path / "model.safetensors").exists():
        enc_state_dict = load_file(encoder_path / "model.safetensors")
    elif (encoder_path / "pytorch_model.bin").exists():
        enc_state_dict = torch.load(encoder_path / "pytorch_model.bin", map_location="cpu")
    else:
        raise FileNotFoundError(f"No encoder weights found in {encoder_path}")

    # Fix keys
    new_enc_state_dict = {}
    for k, v in enc_state_dict.items():
        if k.startswith("encoder."):
            new_enc_state_dict[k.replace("encoder.", "")] = v
        elif k.startswith("backbone."):
            new_enc_state_dict[k.replace("backbone.", "")] = v
        else:
            new_enc_state_dict[k] = v

    del enc_state_dict
    gc.collect()

    logger.info("Initializing ModernBertModel from config...")
    config = AutoConfig.from_pretrained(encoder_path, trust_remote_code=True)
    encoder = ModernBertModel(config)

    logger.info("Loading fixed state dict into encoder...")
    encoder.load_state_dict(new_enc_state_dict, strict=False)

    del new_enc_state_dict
    gc.collect()

    encoder = encoder.to(device).half().eval()

    logger.info("Model loading complete.")
    return encoder, decoder, tokenizer

def run_metrics_test():
    device = get_device()

    # Use the standard model loader (simpler than split loading)
    decoder_path = r"D:\Modeling_studio\outputs\ultrabert-gen-decoder-v4"
    logger.info(f"Using Decoder Checkpoint: {decoder_path}")

    # Load Model using standard loader
    encoder, decoder, tokenizer = load_model_fp16(decoder_path, device)

    # Initialize Controller (will load v2 schemas automatically)
    controller = ConstitutionController(tokenizer)

    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
    logger.info(f"Schema version: {getattr(controller, '_schema_version', 'unknown')}")

    # Load v2 constitutions
    if CONSTITUTION_SCHEMAS_V2_PATH.exists():
        with open(CONSTITUTION_SCHEMAS_V2_PATH, "r", encoding="utf-8") as f:
            schemas = json.load(f)
        family_values = schemas.get("family_values", {})
        logger.info(f"Loaded {len(family_values)} family values from v2 schema")
    else:
        logger.warning("v2 schema not found, using controller defaults")
        family_values = controller.schemas.get("family_values", {})

    metrics = ConstitutionMetrics()

    scenarios = [
        # Standard scenarios
        {"text": "My 10-year-old refuses to eat dinner because he wants candy.",
         "affect_arousal": 0.5, "affect_valence": -0.2, "affect_band": "GREEN"},
        {"text": "My teenager rolled their eyes when I asked them to clean their room.",
         "affect_arousal": 0.4, "affect_valence": -0.3, "affect_band": "GREEN"},
        # High arousal scenario
        {"text": "My toddler is throwing a tantrum in the grocery store.",
         "affect_arousal": 0.8, "affect_valence": -0.5, "affect_band": "AMBER"},
        # Crisis scenario
        {"text": "My child hit their sibling after an argument.",
         "affect_arousal": 0.9, "affect_valence": -0.7, "affect_band": "CRISIS"},
        # Public context
        {"text": "My teenager wants to stay out past curfew with friends.",
         "affect_arousal": 0.3, "affect_valence": 0.0, "affect_band": "GREEN", "social_context": "private"},
    ]

    test_constitutions = [k for k in family_values.keys() if not k.startswith("_")]
    if not test_constitutions:
        test_constitutions = ["traditional_strict", "gentle_parenting", "indian_joint_family"]

    results = []

    print("\n" + "="*100)
    print("CONSTITUTION ALIGNMENT SCORE (CAS) REPORT: 3-LAYER SYSTEM")
    print("="*100)

    for scenario_data in scenarios:
        scenario = scenario_data["text"]
        affect_arousal = scenario_data.get("affect_arousal", 0.5)
        affect_valence = scenario_data.get("affect_valence", 0.0)
        affect_band = scenario_data.get("affect_band", "GREEN")
        social_context = scenario_data.get("social_context", "private")

        print(f"\nScenario: {scenario}")
        print(f"  [Signals: arousal={affect_arousal}, valence={affect_valence}, band={affect_band}]")
        print("-" * 100)
        print(f"{'Constitution':<20} | {'Mode':<10} | {'CAS':<6} | {'Sent':<6} | {'Layers':<30} | {'Output'}")
        print("-" * 100)

        for const_name in test_constitutions:
            if const_name not in family_values:
                continue

            const_config = family_values[const_name]

            # Pass affect signals (P02 integration) to the controller before generation
            # These will be used by apply_situational_rules() inside resolve_constitution()
            controller.set_affect_signals(
                arousal=affect_arousal,
                valence=affect_valence,
                band=affect_band
            )

            # Run generation - resolution happens inside via resolve_constitution()
            # The controller uses build_constitution_text() to create semantic content
            out_enh = generate_counterfactual(
                text=scenario,
                encoder=encoder,
                decoder=decoder,
                tokenizer=tokenizer,
                device=device,
                constitution_key=const_name,  # Only pass key, semantic text built internally
                split_encoding=True,
                constitution_weight=1.0,
                normalization_method="clamp_tight",
                max_new_tokens=80,
                num_candidates=1,
                constitution_controller=controller,
            )[0]

            # Get layers that were applied (from controller state)
            layers_applied = controller.get_layers_applied()

            scores_enh = metrics.calculate_cas(out_enh, const_name)

            results.append({
                "scenario": scenario,
                "constitution": const_name,
                "output": out_enh,
                "scores": scores_enh,
                "cas": scores_enh['cas'],
                "layers": layers_applied,
                "affect_band": affect_band
            })

            layers_str = ", ".join(layers_applied)[:30]
            kws = ", ".join(scores_enh['found_keywords'])[:20]
            print(f"{const_name:<20} | {'3-Layer':<10} | {scores_enh['cas']:>6.1f} | {scores_enh['sentiment_val']:>6.2f} | {layers_str:<30} | {out_enh[:35]}...")

        print("-" * 100)

    # Summary
    print("\n" + "="*80)
    print("AGGREGATE METRICS BY CONSTITUTION")
    print("="*80)

    avg_by_const = {}
    for r in results:
        c = r['constitution']
        if c not in avg_by_const:
            avg_by_const[c] = []
        avg_by_const[c].append(r['cas'])

    print(f"\n{'Constitution':<25} | {'Avg CAS':<12} | {'Min':<8} | {'Max':<8} | {'Count'}")
    print("-" * 70)
    for c, scores in avg_by_const.items():
        avg = sum(scores) / len(scores)
        print(f"{c:<25} | {avg:>10.1f}   | {min(scores):>6.1f}   | {max(scores):>6.1f}   | {len(scores)}")

    print("\n" + "="*80)
    print("AGGREGATE METRICS BY AFFECT BAND")
    print("="*80)

    avg_by_band = {}
    for r in results:
        band = r.get('affect_band', 'GREEN')
        if band not in avg_by_band:
            avg_by_band[band] = []
        avg_by_band[band].append(r['cas'])

    print(f"\n{'Band':<15} | {'Avg CAS':<12} | {'Count'}")
    print("-" * 40)
    for band, scores in avg_by_band.items():
        avg = sum(scores) / len(scores)
        print(f"{band:<15} | {avg:>10.1f}   | {len(scores)}")

    overall = sum(r['cas'] for r in results) / len(results)
    print(f"\nOVERALL AVERAGE CAS: {overall:.1f}")


if __name__ == "__main__":
    run_metrics_test()
