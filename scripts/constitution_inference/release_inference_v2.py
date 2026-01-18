"""
FamilyOS Counterfactual Decoder - Contract-Based Release Inference

This script implements the Standard I/O Contract for the 13th Head (GPT-2 Decoder).
It accepts structured input from P03 and returns structured output.

Architecture:
    P03 (R5 Dream Phase) --> [This Script] --> Counterfactual Output

Contract Files:
    - contracts/input_schema.json  (What we accept)
    - contracts/output_schema.json (What we return)

Usage:
    # CLI Mode (for testing)
    python release_inference.py --input event.json --output result.json

    # API Mode (for P03 integration)
    from release_inference import DecoderAPI
    api = DecoderAPI(checkpoint_path)
    result = api.generate(input_dict)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from transformers import AutoTokenizer, LogitsProcessorList, LogitsProcessor

try:
    from transformers import ModernBertModel
except ImportError:
    from transformers import AutoModel as ModernBertModel

try:
    from modeling_studio.models.decoder_gpt2 import GPT2DecoderHead
    from modeling_studio.models.decoder_gpt2_config import GPT2DecoderConfig
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
    from modeling_studio.models.decoder_gpt2 import GPT2DecoderHead
    from modeling_studio.models.decoder_gpt2_config import GPT2DecoderConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("FamilyOS.Decoder")

# =============================================================================
# Constants
# =============================================================================

MODEL_VERSION = "v2.0.0"
DEFAULT_CHECKPOINT = Path(__file__).parent.parent.parent / "outputs" / "ultrabert-gen-decoder-v4"
CONSTITUTION_SCHEMA_PATH = Path(__file__).parent.parent.parent / "data" / "constitutions" / "constitution_schemas_v2.json"


# =============================================================================
# Data Classes for Type Safety
# =============================================================================

@dataclass
class FamilyValuesLayer:
    """Layer 1: Static family values"""
    key: str
    description: str = ""
    positive_tokens: Dict[str, float] = None
    negative_tokens: Dict[str, float] = None
    temperature: float = 0.7
    repetition_penalty: float = 1.1
    logits_strength: float = 0.5
    prefix_injection: bool = False
    prefix_pattern: List[float] = None


@dataclass
class IndividualLayer:
    """Layer 2: Per-actor preferences"""
    actor_id: str = None
    response_length: str = "moderate"
    formality: str = "neutral"
    needs_validation_first: bool = True
    custom_positive_tokens: Dict[str, float] = None
    custom_negative_tokens: Dict[str, float] = None


@dataclass
class SituationalLayer:
    """Layer 3: Dynamic context from event signals"""
    affect_arousal: float = 0.5
    affect_valence: float = 0.0
    affect_band: str = "GREEN"
    social_context: str = "private"
    salience_score: float = 0.5
    novelty_score: float = 0.5
    steering_weight: float = 1.0
    temperature_adjustment: float = 0.0
    force_deescalation: bool = False
    inject_empathy_prefix: bool = False


@dataclass
class Constitution:
    """Complete 3-layer constitution"""
    family_values: FamilyValuesLayer
    individual: IndividualLayer = None
    situational: SituationalLayer = None


@dataclass
class DecoderInput:
    """Standard Input Contract"""
    event_id: str
    text: str
    constitution: Constitution
    context: Dict[str, Any] = None


@dataclass
class GenerationMeta:
    """Metadata about generation"""
    constitution_applied: str
    steering_weight_used: float
    temperature_final: float
    repetition_penalty_final: float
    tokens_generated: int
    latency_ms: int
    model_version: str = MODEL_VERSION


@dataclass
class Trace:
    """Debug/Explainability trace"""
    layers_applied: List[str]
    deescalation_triggered: bool
    empathy_prefix_injected: bool
    logits_bias_applied: List[str]
    normalization_method: str


@dataclass
class DecoderOutput:
    """Standard Output Contract"""
    event_id: str
    counterfactual: str
    generation_meta: GenerationMeta
    trace: Trace
    error: Dict[str, str] = None


# =============================================================================
# Constitution Schema Loader
# =============================================================================

class ConstitutionSchemaLoader:
    """
    Loads and manages the 3-layer constitution schemas.
    Reads from constitution_schemas_v2.json
    """

    def __init__(self, schema_path: Path = CONSTITUTION_SCHEMA_PATH):
        self.schema_path = schema_path
        self.schemas = self._load_schemas()

    def _load_schemas(self) -> Dict:
        if not self.schema_path.exists():
            logger.warning(f"Schema file not found: {self.schema_path}")
            return self._get_defaults()

        try:
            with open(self.schema_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Loaded constitution schemas from {self.schema_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load schemas: {e}")
            return self._get_defaults()

    def _get_defaults(self) -> Dict:
        return {
            "family_values": {
                "gentle_parenting": {
                    "description": "Default gentle parenting",
                    "positive_tokens": {"understand": 0.5, "feel": 0.5},
                    "negative_tokens": {"punish": -0.5},
                    "generation_params": {"temperature": 0.7, "repetition_penalty": 1.0, "logits_strength": 0.0}
                }
            },
            "individual_prefs": {"default": {}},
            "situational_rules": {}
        }

    def get_family_values(self, key: str) -> Dict:
        """Get family values layer by key"""
        return self.schemas.get("family_values", {}).get(key, {})

    def get_individual_prefs(self, actor_id: str) -> Dict:
        """Get individual preferences by actor_id"""
        prefs = self.schemas.get("individual_prefs", {})
        return prefs.get(actor_id, prefs.get("default", {}))

    def apply_situational_rules(self, situational: SituationalLayer) -> Dict:
        """Apply situational rules based on signals"""
        rules = self.schemas.get("situational_rules", {})
        applied = {}

        # High arousal rule
        if situational.affect_arousal > 0.7 and "high_arousal" in rules:
            applied.update(rules["high_arousal"].get("actions", {}))

        # Negative valence rule
        if situational.affect_valence < -0.3 and "negative_valence" in rules:
            applied.update(rules["negative_valence"].get("actions", {}))

        # Crisis band rule
        if situational.affect_band == "CRISIS" and "crisis_band" in rules:
            applied.update(rules["crisis_band"].get("actions", {}))

        # Public context rule
        if situational.social_context == "public" and "public_context" in rules:
            applied.update(rules["public_context"].get("actions", {}))

        # High novelty rule
        if situational.novelty_score > 0.8 and "high_novelty" in rules:
            applied.update(rules["high_novelty"].get("actions", {}))

        return applied


# =============================================================================
# Decoder API (Main Class)
# =============================================================================

class DecoderAPI:
    """
    The main API class for the Counterfactual Decoder (13th Head).
    Implements the Standard I/O Contract.
    """

    def __init__(self, checkpoint_path: str = None, device: str = None):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
        self.device = torch.device(device) if device else self._get_device()

        # Load components
        self.schema_loader = ConstitutionSchemaLoader()
        self.encoder, self.decoder, self.tokenizer = self._load_model()

        logger.info(f"DecoderAPI initialized on {self.device}")

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_model(self):
        """Load encoder + decoder with Windows-safe loading"""
        logger.info(f"Loading model from {self.checkpoint_path}")

        # Tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path, trust_remote_code=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Encoder (try local cache first)
        try:
            encoder = ModernBertModel.from_pretrained(
                "answerdotai/ModernBERT-base",
                torch_dtype=torch.float16,
                local_files_only=True
            )
        except Exception as e:
            logger.warning(f"Could not load from cache: {e}, trying network...")
            encoder = ModernBertModel.from_pretrained(
                "answerdotai/ModernBERT-base",
                torch_dtype=torch.float16
            )

        # Decoder
        decoder_config = GPT2DecoderConfig(
            gpt2_model_name="gpt2-medium",
            encoder_hidden_size=768,
            projection_hidden_size=1024,
            num_prefix_tokens=16,
            freeze_layers=12,
        )
        decoder = GPT2DecoderHead(config=decoder_config, encoder_hidden_size=768)

        # Load weights (Windows-safe)
        state_dict = self._load_weights()

        # Apply encoder weights
        encoder_sd = {k.replace("encoder.", "", 1): v for k, v in state_dict.items()
                     if k.startswith("encoder.") or k.startswith("backbone.")}
        if encoder_sd:
            encoder.load_state_dict(encoder_sd, strict=False)

        # Apply decoder weights
        decoder_sd = {k.replace("decoder.", "", 1): v for k, v in state_dict.items()
                     if k.startswith("decoder.")}
        if decoder_sd:
            decoder.load_state_dict(decoder_sd, strict=False)

        encoder = encoder.to(self.device).half().eval()
        decoder = decoder.to(self.device).half().eval()

        return encoder, decoder, tokenizer

    def _load_weights(self) -> Dict:
        """Windows-safe weight loading"""
        safetensors_path = self.checkpoint_path / "model.safetensors"
        pytorch_path = self.checkpoint_path / "pytorch_model.bin"

        if safetensors_path.exists():
            from safetensors import safe_open
            state_dict = {}
            with safe_open(str(safetensors_path), framework="numpy") as f:
                for key in f.keys():
                    state_dict[key] = torch.from_numpy(f.get_tensor(key).copy())
            return state_dict
        elif pytorch_path.exists():
            return torch.load(pytorch_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"No model weights in {self.checkpoint_path}")

    def generate(self, input_data: Dict) -> Dict:
        """
        Main generation method. Accepts contract-compliant input, returns contract-compliant output.

        Args:
            input_data: Dict matching input_schema.json

        Returns:
            Dict matching output_schema.json
        """
        start_time = time.time()

        try:
            # Parse input
            decoder_input = self._parse_input(input_data)

            # Resolve constitution parameters
            params, trace_info = self._resolve_constitution(decoder_input.constitution)

            # Generate
            counterfactual, tokens_generated = self._generate_text(
                decoder_input.text,
                decoder_input.constitution,
                params
            )

            latency_ms = int((time.time() - start_time) * 1000)

            # Build output
            output = DecoderOutput(
                event_id=decoder_input.event_id,
                counterfactual=counterfactual,
                generation_meta=GenerationMeta(
                    constitution_applied=decoder_input.constitution.family_values.key,
                    steering_weight_used=params["steering_weight"],
                    temperature_final=params["temperature"],
                    repetition_penalty_final=params["repetition_penalty"],
                    tokens_generated=tokens_generated,
                    latency_ms=latency_ms
                ),
                trace=Trace(
                    layers_applied=trace_info["layers_applied"],
                    deescalation_triggered=trace_info["deescalation_triggered"],
                    empathy_prefix_injected=trace_info["empathy_prefix_injected"],
                    logits_bias_applied=trace_info["logits_bias_applied"],
                    normalization_method="clamp_tight"
                )
            )

            return self._to_dict(output)

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                "event_id": input_data.get("event_id", "unknown"),
                "counterfactual": "",
                "generation_meta": {"latency_ms": int((time.time() - start_time) * 1000)},
                "error": {"code": "MODEL_ERROR", "message": str(e)}
            }

    def _parse_input(self, data: Dict) -> DecoderInput:
        """Parse raw dict into typed DecoderInput"""
        const_data = data.get("constitution", {})
        fv_data = const_data.get("family_values", {})
        ind_data = const_data.get("individual", {})
        sit_data = const_data.get("situational", {})

        family_values = FamilyValuesLayer(
            key=fv_data.get("key", "gentle_parenting"),
            description=fv_data.get("description", ""),
            positive_tokens=fv_data.get("positive_tokens", {}),
            negative_tokens=fv_data.get("negative_tokens", {}),
            temperature=fv_data.get("temperature", 0.7),
            repetition_penalty=fv_data.get("repetition_penalty", 1.1),
            logits_strength=fv_data.get("logits_strength", 0.5),
            prefix_injection=fv_data.get("prefix_injection", False),
            prefix_pattern=fv_data.get("prefix_pattern", [])
        )

        individual = IndividualLayer(
            actor_id=ind_data.get("actor_id"),
            response_length=ind_data.get("response_length", "moderate"),
            formality=ind_data.get("formality", "neutral"),
            needs_validation_first=ind_data.get("needs_validation_first", True),
            custom_positive_tokens=ind_data.get("custom_positive_tokens", {}),
            custom_negative_tokens=ind_data.get("custom_negative_tokens", {})
        )

        situational = SituationalLayer(
            affect_arousal=sit_data.get("affect_arousal", 0.5),
            affect_valence=sit_data.get("affect_valence", 0.0),
            affect_band=sit_data.get("affect_band", "GREEN"),
            social_context=sit_data.get("social_context", "private"),
            salience_score=sit_data.get("salience_score", 0.5),
            novelty_score=sit_data.get("novelty_score", 0.5),
            steering_weight=sit_data.get("steering_weight", 1.0),
            temperature_adjustment=sit_data.get("temperature_adjustment", 0.0),
            force_deescalation=sit_data.get("force_deescalation", False),
            inject_empathy_prefix=sit_data.get("inject_empathy_prefix", False)
        )

        return DecoderInput(
            event_id=data.get("event_id", ""),
            text=data.get("text", ""),
            constitution=Constitution(
                family_values=family_values,
                individual=individual,
                situational=situational
            ),
            context=data.get("context", {})
        )

    def _resolve_constitution(self, constitution: Constitution) -> tuple:
        """Merge the 3 layers into final generation parameters"""
        fv = constitution.family_values
        ind = constitution.individual or IndividualLayer()
        sit = constitution.situational or SituationalLayer()

        layers_applied = ["family_values"]

        # Start with family values
        params = {
            "temperature": fv.temperature,
            "repetition_penalty": fv.repetition_penalty,
            "logits_strength": fv.logits_strength,
            "positive_tokens": fv.positive_tokens or {},
            "negative_tokens": fv.negative_tokens or {},
            "prefix_injection": fv.prefix_injection,
            "prefix_pattern": fv.prefix_pattern or [],
            "steering_weight": 1.0,
            "max_new_tokens": 96
        }

        # Apply individual layer overrides
        if ind.actor_id:
            layers_applied.append("individual")
            # Merge custom tokens
            if ind.custom_positive_tokens:
                params["positive_tokens"].update(ind.custom_positive_tokens)
            if ind.custom_negative_tokens:
                params["negative_tokens"].update(ind.custom_negative_tokens)
            # Adjust max tokens based on response length
            if ind.response_length == "concise":
                params["max_new_tokens"] = 60
            elif ind.response_length == "detailed":
                params["max_new_tokens"] = 128

        # Apply situational layer adjustments
        if sit.affect_arousal > 0.5 or sit.affect_valence < 0:
            layers_applied.append("situational")

        # Situational temperature adjustment
        params["temperature"] = max(0.1, params["temperature"] + sit.temperature_adjustment)

        # Situational steering weight
        params["steering_weight"] = sit.steering_weight

        # Apply dynamic rules from schema
        situational_rules = self.schema_loader.apply_situational_rules(sit)

        if "temperature_adjustment" in situational_rules:
            params["temperature"] = max(0.1, params["temperature"] + situational_rules["temperature_adjustment"])

        if "steering_weight_multiplier" in situational_rules:
            params["steering_weight"] *= situational_rules["steering_weight_multiplier"]

        if "inject_tokens" in situational_rules:
            params["positive_tokens"].update(situational_rules["inject_tokens"])

        if "max_tokens" in situational_rules:
            params["max_new_tokens"] = min(params["max_new_tokens"], situational_rules["max_tokens"])

        # Build trace info
        trace_info = {
            "layers_applied": layers_applied,
            "deescalation_triggered": sit.force_deescalation or situational_rules.get("force_deescalation", False),
            "empathy_prefix_injected": sit.inject_empathy_prefix or situational_rules.get("inject_empathy_prefix", False),
            "logits_bias_applied": list(params["positive_tokens"].keys()) + list(params["negative_tokens"].keys())
        }

        return params, trace_info

    @torch.inference_mode()
    def _generate_text(self, text: str, constitution: Constitution, params: Dict) -> tuple:
        """Core generation logic"""
        fv = constitution.family_values

        # Create logits processor
        logits_processor = None
        if params["logits_strength"] > 0:
            logits_processor = self._create_logits_processor(
                params["positive_tokens"],
                params["negative_tokens"],
                params["logits_strength"] * params["steering_weight"]
            )

        # Encode
        def encode(txt):
            inputs = self.tokenizer(txt, return_tensors="pt", max_length=256, truncation=True, padding=True)
            with torch.autocast(device_type="cuda" if self.device.type == "cuda" else "cpu", dtype=torch.float16):
                out = self.encoder(
                    input_ids=inputs["input_ids"].to(self.device),
                    attention_mask=inputs["attention_mask"].to(self.device)
                )
            return out.last_hidden_state, inputs["attention_mask"].to(self.device)

        with torch.autocast(device_type="cuda" if self.device.type == "cuda" else "cpu", dtype=torch.float16):
            # Split encoding
            const_h, const_m = encode(f"[CONSTITUTION: {fv.key}]")
            scen_h, scen_m = encode(text)

            # Normalize embeddings
            const_h = torch.clamp(const_h, -2, 2)
            scen_h = torch.clamp(scen_h, -2, 2)

            # Apply steering weight
            const_h = const_h * params["steering_weight"]

            encoder_hidden = torch.cat([const_h, scen_h], dim=1)
            attention_mask = torch.cat([const_m, scen_m], dim=1)

            # Prefix injection
            if params["prefix_injection"] and params["prefix_pattern"]:
                prefix = self._create_prefix(params["prefix_pattern"])
                encoder_hidden = torch.cat([prefix, encoder_hidden], dim=1)
                prefix_mask = torch.ones((attention_mask.shape[0], prefix.shape[1]), device=self.device).long()
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

            # Generate
            generated_ids = self.decoder.generate(
                encoder_hidden_states=encoder_hidden,
                encoder_attention_mask=attention_mask,
                max_new_tokens=params["max_new_tokens"],
                temperature=params["temperature"],
                repetition_penalty=params["repetition_penalty"],
                top_k=50,
                top_p=0.9,
                do_sample=True,
                logits_processor=logits_processor,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        output_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Cleanup
        output_text = self._clean_output(output_text)

        tokens_generated = len(self.tokenizer.encode(output_text))

        return output_text, tokens_generated

    def _create_logits_processor(self, positive: Dict, negative: Dict, strength: float):
        """Create logits processor for token biasing"""
        pos_ids = {}
        neg_ids = {}

        for token_str, weight in (positive or {}).items():
            token_ids = self.tokenizer.encode(token_str, add_special_tokens=False)
            for tid in token_ids:
                pos_ids[tid] = max(pos_ids.get(tid, 0), weight)

        for token_str, weight in (negative or {}).items():
            token_ids = self.tokenizer.encode(token_str, add_special_tokens=False)
            for tid in token_ids:
                neg_ids[tid] = min(neg_ids.get(tid, 0), weight)

        class ConstitutionLogitsProcessor(LogitsProcessor):
            def __call__(self, input_ids, scores):
                for tid, w in pos_ids.items():
                    scores[:, tid] += (w * strength)
                for tid, w in neg_ids.items():
                    scores[:, tid] += (w * strength)
                return scores

        return LogitsProcessorList([ConstitutionLogitsProcessor()])

    def _create_prefix(self, pattern: List[float], num_tokens: int = 8, hidden_size: int = 768) -> torch.Tensor:
        """Create steering prefix tensor"""
        pattern_tensor = torch.tensor(pattern, device=self.device, dtype=torch.float16)
        repeats = (hidden_size // len(pattern)) + 1
        full_vector = pattern_tensor.repeat(repeats)[:hidden_size]
        prefix = full_vector.unsqueeze(0).repeat(num_tokens, 1)
        return prefix.unsqueeze(0)

    def _clean_output(self, text: str) -> str:
        """Clean up generated text"""
        # Remove double newlines
        if "\n\n" in text:
            text = text.split("\n\n")[0]

        # Ensure proper ending
        if text and text[-1] not in ".!?":
            last_punct = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
            if last_punct != -1:
                text = text[:last_punct + 1]

        return text.strip()

    def _to_dict(self, obj) -> Dict:
        """Convert dataclass to dict, handling nested dataclasses"""
        if hasattr(obj, "__dataclass_fields__"):
            return {k: self._to_dict(v) for k, v in asdict(obj).items() if v is not None}
        return obj


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="FamilyOS Counterfactual Decoder (Contract-Based)")
    parser.add_argument("--input", type=str, help="Path to input JSON file (contract format)")
    parser.add_argument("--output", type=str, help="Path to write output JSON file")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Model checkpoint path")
    parser.add_argument("--text", type=str, help="Direct text input (for quick testing)")
    parser.add_argument("--constitution", type=str, default="gentle_parenting", help="Constitution key")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    # Initialize API
    api = DecoderAPI(checkpoint_path=args.checkpoint)

    if args.input:
        # Contract mode: read from file
        with open(args.input, "r", encoding="utf-8") as f:
            input_data = json.load(f)

        result = api.generate(input_data)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Output written to {args.output}")
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.text:
        # Quick test mode
        input_data = {
            "event_id": "test_001",
            "text": args.text,
            "constitution": {
                "family_values": {
                    "key": args.constitution
                },
                "situational": {
                    "affect_arousal": 0.5,
                    "affect_valence": 0.0
                }
            }
        }

        result = api.generate(input_data)
        print(f"\nCounterfactual:\n{result['counterfactual']}\n")
        print(f"Meta: {json.dumps(result['generation_meta'], indent=2)}")

    elif args.interactive:
        # Interactive mode
        print("\n=== FamilyOS Decoder (Contract Mode) ===")
        print("Type 'quit' to exit\n")

        while True:
            text = input("Scenario: ").strip()
            if text.lower() in ["quit", "exit"]:
                break

            const = input(f"Constitution [{args.constitution}]: ").strip() or args.constitution
            arousal = input("Affect Arousal [0.5]: ").strip()
            arousal = float(arousal) if arousal else 0.5

            input_data = {
                "event_id": f"interactive_{int(time.time())}",
                "text": text,
                "constitution": {
                    "family_values": {"key": const},
                    "situational": {"affect_arousal": arousal, "affect_valence": -0.3}
                }
            }

            result = api.generate(input_data)
            print(f"\nCounterfactual: {result['counterfactual']}")
            print(f"Latency: {result['generation_meta']['latency_ms']}ms")
            print(f"Layers: {result['trace']['layers_applied']}\n")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
