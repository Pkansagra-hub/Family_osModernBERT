#!/usr/bin/env python
"""
Inference Script

This script provides a simple interface for running inference
with trained multi-task models.

Supported Tasks:
    - ner_general: Named entity recognition
    - ner_family: FamilyOS family NER
    - sentiment: Sentiment classification
    - emotions: Emotion detection
    - safety_generic: Safety/toxicity classification
    - safety_familyos: FamilyOS policy bands
    - ingress: Domain classification
    - embedding: Text embeddings
    - nli: Natural language inference
    - temporal: Temporal expression extraction
    - relation: Family relationship extraction
    - intent: User intent classification

Usage:
    # Interactive mode
    python scripts/infer.py \
        --model outputs/modernbert-multitask-v0 \
        --interactive

    # Single text, multiple tasks
    python scripts/infer.py \
        --model outputs/modernbert-multitask-v0 \
        --text "I'm feeling really anxious about the meeting" \
        --tasks sentiment emotions safety_familyos

    # Batch inference from file
    python scripts/infer.py \
        --model outputs/modernbert-multitask-v0 \
        --input data/test_samples.jsonl \
        --output predictions.jsonl \
        --tasks all

    # NLI inference
    python scripts/infer.py \
        --model outputs/modernbert-multitask-v0 \
        --premise "The restaurant was crowded" \
        --hypothesis "There were many people" \
        --tasks nli

    # Get embeddings
    python scripts/infer.py \
        --model outputs/modernbert-multitask-v0 \
        --text "Sample text for embedding" \
        --tasks embedding \
        --output-format numpy

Output Formats:
    - json: Structured JSON output
    - pretty: Human-readable colored output
    - numpy: NumPy arrays (for embeddings)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modeling_studio.data.labels import (
    CAPABILITY_TO_LABELS,
    Capability,
)
from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Post-Processing Functions
# =============================================================================


def extract_entities_from_bio(
    tokens: list[str],
    labels: list[str],
    input_ids: list[int],
    tokenizer: Any,
) -> list[dict]:
    """
    Convert BIO tags to entity spans.
    
    Args:
        tokens: List of tokens
        labels: List of BIO labels
        input_ids: Original input IDs for offset mapping
        tokenizer: Tokenizer for decoding
        
    Returns:
        List of entity dictionaries with text, label, start, end
    """
    entities = []
    current_entity = None
    
    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label.startswith("B-"):
            # Save previous entity
            if current_entity:
                entities.append(current_entity)
            # Start new entity
            entity_type = label[2:]
            current_entity = {
                "text": token.replace("##", "").replace("Ġ", " ").strip(),
                "label": entity_type,
                "start_token": i,
                "end_token": i,
            }
        elif label.startswith("I-") and current_entity:
            # Continue entity
            entity_type = label[2:]
            if entity_type == current_entity["label"]:
                current_entity["text"] += token.replace("##", "").replace("Ġ", " ")
                current_entity["end_token"] = i
            else:
                # Entity type mismatch, save and start new
                entities.append(current_entity)
                current_entity = {
                    "text": token.replace("##", "").replace("Ġ", " ").strip(),
                    "label": entity_type,
                    "start_token": i,
                    "end_token": i,
                }
        else:
            # O tag - save current entity
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    # Don't forget last entity
    if current_entity:
        entities.append(current_entity)
    
    # Clean up entity texts
    for entity in entities:
        entity["text"] = entity["text"].strip()
    
    return entities


def get_top_emotions(
    logits: torch.Tensor,
    labels_schema: Any,
    threshold: float = 0.3,
    top_k: int = 5,
) -> list[dict]:
    """
    Get top emotions from multi-label logits.
    
    Args:
        logits: Raw logits (num_labels,)
        labels_schema: LabelSchema with id2label
        threshold: Confidence threshold for positive
        top_k: Maximum emotions to return
        
    Returns:
        List of {emotion, confidence} dicts
    """
    probs = torch.sigmoid(logits).cpu().numpy()
    
    # Get indices sorted by probability
    sorted_indices = np.argsort(probs)[::-1]
    
    emotions = []
    for idx in sorted_indices[:top_k]:
        prob = float(probs[idx])
        if prob >= threshold or len(emotions) == 0:
            emotions.append({
                "emotion": labels_schema.id2label[int(idx)],
                "confidence": round(prob, 4),
            })
    
    return emotions


def get_safety_band(
    logits: torch.Tensor,
    labels_schema: Any,
) -> dict:
    """
    Get safety band prediction with confidence.
    
    Args:
        logits: Raw logits (4,) for GREEN/AMBER/RED/CRISIS
        labels_schema: LabelSchema with id2label
        
    Returns:
        Dict with band, confidence, and all probabilities
    """
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    pred_idx = int(np.argmax(probs))
    
    return {
        "band": labels_schema.id2label[pred_idx],
        "confidence": round(float(probs[pred_idx]), 4),
        "probabilities": {
            labels_schema.id2label[i]: round(float(p), 4)
            for i, p in enumerate(probs)
        },
    }


def format_single_label(
    logits: torch.Tensor,
    labels_schema: Any,
) -> dict:
    """Format single-label classification output."""
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    pred_idx = int(np.argmax(probs))
    
    return {
        "prediction": labels_schema.id2label[pred_idx],
        "confidence": round(float(probs[pred_idx]), 4),
        "all_scores": {
            labels_schema.id2label[i]: round(float(p), 4)
            for i, p in enumerate(probs)
        },
    }


def format_multi_label(
    logits: torch.Tensor,
    labels_schema: Any,
    threshold: float = 0.5,
) -> dict:
    """Format multi-label classification output."""
    probs = torch.sigmoid(logits).cpu().numpy()
    
    predictions = []
    all_scores = {}
    
    for i, p in enumerate(probs):
        label = labels_schema.id2label[i]
        all_scores[label] = round(float(p), 4)
        if p >= threshold:
            predictions.append(label)
    
    return {
        "predictions": predictions,
        "all_scores": all_scores,
    }


# =============================================================================
# Inference Engine
# =============================================================================


class MultiTaskInferenceEngine:
    """
    Inference engine for the multi-task model.
    
    Handles model loading, tokenization, inference, and post-processing
    for all supported capabilities.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the model checkpoint
            device: Device to use (auto, cpu, cuda, cuda:0, etc.)
        """
        self.model_path = Path(model_path)
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading model from {model_path}...")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        
        # Load model
        self.model = ModernBertMultiTaskModel.load_checkpoint(
            str(self.model_path),
            device=self.device,
        )
        self.model.eval()
        
        # Get available capabilities
        self.capabilities = list(self.model.heads.keys())
        logger.info(f"Available capabilities: {self.capabilities}")
    
    @torch.no_grad()
    def infer(
        self,
        text: str,
        capability: str | Capability,
        premise: str | None = None,
        hypothesis: str | None = None,
    ) -> dict:
        """
        Run inference for a single capability.
        
        Args:
            text: Input text
            capability: Which capability to use
            premise: For NLI - the premise text
            hypothesis: For NLI - the hypothesis text
            
        Returns:
            Dictionary with results
        """
        if isinstance(capability, str):
            capability = Capability(capability)
        
        cap_str = capability.value
        if cap_str not in self.capabilities:
            raise ValueError(
                f"Capability '{cap_str}' not available. "
                f"Available: {self.capabilities}"
            )
        
        # Tokenize based on task type
        if capability == Capability.NLI:
            if premise is None or hypothesis is None:
                raise ValueError("NLI requires both premise and hypothesis")
            inputs = self.tokenizer(
                premise,
                hypothesis,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
        else:
            inputs = self.tokenizer(
                text,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            capability=capability,
        )
        
        logits = outputs.logits
        
        # Post-process based on capability
        return self._postprocess(
            capability=capability,
            logits=logits,
            inputs=inputs,
            text=text,
        )
    
    def _postprocess(
        self,
        capability: Capability,
        logits: torch.Tensor,
        inputs: dict,
        text: str,
    ) -> dict:
        """Post-process model outputs based on capability."""
        labels_schema = CAPABILITY_TO_LABELS.get(capability)
        
        # Token classification tasks
        if capability in [Capability.NER_GENERAL, Capability.NER_FAMILY, Capability.TEMPORAL]:
            # Get predictions per token
            pred_ids = torch.argmax(logits, dim=-1)[0].cpu().numpy()
            
            # Convert to labels
            tokens = self.tokenizer.convert_ids_to_tokens(
                inputs["input_ids"][0].cpu().numpy()
            )
            pred_labels = [labels_schema.id2label[int(i)] for i in pred_ids]
            
            # Extract entities
            entities = extract_entities_from_bio(
                tokens=tokens,
                labels=pred_labels,
                input_ids=inputs["input_ids"][0].cpu().numpy().tolist(),
                tokenizer=self.tokenizer,
            )
            
            return {
                "capability": capability.value,
                "entities": entities,
                "token_labels": list(zip(tokens, pred_labels))[1:-1],  # Skip [CLS]/[SEP]
            }
        
        # Embedding
        elif capability == Capability.EMBEDDING:
            embedding = logits[0].cpu().numpy()
            return {
                "capability": capability.value,
                "embedding": embedding.tolist(),
                "embedding_dim": len(embedding),
                "norm": float(np.linalg.norm(embedding)),
            }
        
        # Multi-label tasks
        elif capability in [Capability.EMOTIONS, Capability.SAFETY_GENERIC]:
            return {
                "capability": capability.value,
                **format_multi_label(logits[0], labels_schema),
            }
        
        # Safety FamilyOS (special handling)
        elif capability == Capability.SAFETY_FAMILYOS:
            return {
                "capability": capability.value,
                **get_safety_band(logits[0], labels_schema),
            }
        
        # Single-label classification
        else:
            return {
                "capability": capability.value,
                **format_single_label(logits[0], labels_schema),
            }
    
    def infer_all(
        self,
        text: str,
        capabilities: list[str] | None = None,
        premise: str | None = None,
        hypothesis: str | None = None,
    ) -> dict:
        """
        Run inference for multiple capabilities.
        
        Args:
            text: Input text
            capabilities: List of capabilities (None = all available)
            premise: For NLI
            hypothesis: For NLI
            
        Returns:
            Dictionary with results per capability
        """
        if capabilities is None:
            capabilities = self.capabilities
        
        results = {"text": text, "results": {}}
        
        for cap in capabilities:
            try:
                if cap == "nli" and (premise is None or hypothesis is None):
                    # Skip NLI if no premise/hypothesis
                    continue
                result = self.infer(
                    text=text,
                    capability=cap,
                    premise=premise,
                    hypothesis=hypothesis,
                )
                results["results"][cap] = result
            except Exception as e:
                results["results"][cap] = {"error": str(e)}
        
        return results


# =============================================================================
# Output Formatting
# =============================================================================


def print_pretty(results: dict) -> None:
    """Pretty print results with colors."""
    text = results.get("text", "")
    print(f"\n{'='*60}")
    print(f"Input: {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"{'='*60}\n")
    
    for cap, result in results.get("results", {}).items():
        print(f"📊 {cap.upper()}")
        print("-" * 40)
        
        if "error" in result:
            print(f"  ❌ Error: {result['error']}")
        elif "entities" in result:
            # NER/Temporal
            if result["entities"]:
                for ent in result["entities"]:
                    print(f"  • {ent['text']} [{ent['label']}]")
            else:
                print("  (no entities found)")
        elif "embedding" in result:
            # Embedding
            print(f"  Dimension: {result['embedding_dim']}")
            print(f"  Norm: {result['norm']:.4f}")
            print(f"  First 5 dims: {result['embedding'][:5]}")
        elif "band" in result:
            # Safety FamilyOS
            band = result["band"]
            conf = result["confidence"]
            band_emoji = {"GREEN": "🟢", "AMBER": "🟡", "RED": "🔴", "CRISIS": "🚨"}.get(band, "⚪")
            print(f"  {band_emoji} {band} (confidence: {conf:.2%})")
        elif "predictions" in result:
            # Multi-label
            if result["predictions"]:
                for pred in result["predictions"]:
                    score = result["all_scores"].get(pred, 0)
                    print(f"  ✓ {pred}: {score:.2%}")
            else:
                print("  (no predictions above threshold)")
        elif "prediction" in result:
            # Single-label
            pred = result["prediction"]
            conf = result["confidence"]
            print(f"  → {pred} (confidence: {conf:.2%})")
        
        print()


# =============================================================================
# Interactive Mode
# =============================================================================


def run_interactive(engine: MultiTaskInferenceEngine) -> None:
    """Run interactive REPL mode."""
    print("\n" + "="*60)
    print("🚀 FamilyOS Multi-Task Model - Interactive Mode")
    print("="*60)
    print(f"\nAvailable capabilities: {', '.join(engine.capabilities)}")
    print("\nCommands:")
    print("  text <message>  - Set input text")
    print("  run <caps>      - Run specific capabilities (comma-separated)")
    print("  run all         - Run all capabilities")
    print("  nli             - Enter NLI mode (premise + hypothesis)")
    print("  quit            - Exit")
    print()
    
    current_text = ""
    
    while True:
        try:
            cmd = input(">>> ").strip()
            
            if not cmd:
                continue
            
            if cmd.lower() in ("quit", "exit", "q"):
                print("Goodbye! 👋")
                break
            
            if cmd.lower().startswith("text "):
                current_text = cmd[5:].strip()
                print(f"✓ Text set: {current_text[:50]}...")
            
            elif cmd.lower().startswith("run "):
                if not current_text:
                    print("❌ Set text first with: text <message>")
                    continue
                
                caps_str = cmd[4:].strip()
                if caps_str.lower() == "all":
                    caps = None
                else:
                    caps = [c.strip() for c in caps_str.split(",")]
                
                results = engine.infer_all(current_text, caps)
                print_pretty(results)
            
            elif cmd.lower() == "nli":
                premise = input("Premise: ").strip()
                hypothesis = input("Hypothesis: ").strip()
                if premise and hypothesis:
                    result = engine.infer(
                        text="",
                        capability="nli",
                        premise=premise,
                        hypothesis=hypothesis,
                    )
                    print(f"\n📊 NLI Result:")
                    print(f"  → {result['prediction']} ({result['confidence']:.2%})")
                    print()
            
            else:
                # Treat as text + run all
                current_text = cmd
                results = engine.infer_all(current_text)
                print_pretty(results)
        
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


# =============================================================================
# Batch Processing
# =============================================================================


def run_batch(
    engine: MultiTaskInferenceEngine,
    input_file: str,
    output_file: str,
    capabilities: list[str] | None = None,
) -> None:
    """
    Process a batch of texts from a JSONL file.
    
    Expected input format:
        {"text": "...", "id": "optional_id"}
        
    For NLI:
        {"premise": "...", "hypothesis": "...", "id": "optional_id"}
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    results = []
    
    with open(input_path) as f:
        lines = f.readlines()
    
    logger.info(f"Processing {len(lines)} samples...")
    
    for i, line in enumerate(lines):
        try:
            sample = json.loads(line.strip())
            
            if "premise" in sample and "hypothesis" in sample:
                # NLI mode
                result = engine.infer(
                    text="",
                    capability="nli",
                    premise=sample["premise"],
                    hypothesis=sample["hypothesis"],
                )
                result["id"] = sample.get("id", i)
            else:
                # Standard mode
                text = sample.get("text", "")
                result = engine.infer_all(text, capabilities)
                result["id"] = sample.get("id", i)
            
            results.append(result)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(lines)}")
        
        except Exception as e:
            logger.warning(f"Error on line {i}: {e}")
            results.append({"id": i, "error": str(e)})
    
    # Save results
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    logger.info(f"Results saved to {output_file}")


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Multi-task inference with FamilyOS unified encoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python scripts/infer.py --model outputs/modernbert-multitask-v0 --interactive
  
  # Single text inference
  python scripts/infer.py --model outputs/modernbert-multitask-v0 \\
      --text "Had a wonderful dinner with mom and dad yesterday"
  
  # Specific capabilities only
  python scripts/infer.py --model outputs/modernbert-multitask-v0 \\
      --text "Feeling anxious today" --tasks sentiment,emotions,safety_familyos
  
  # NLI inference
  python scripts/infer.py --model outputs/modernbert-multitask-v0 \\
      --premise "The restaurant was full" --hypothesis "It was crowded" --tasks nli
  
  # Batch processing
  python scripts/infer.py --model outputs/modernbert-multitask-v0 \\
      --input test_samples.jsonl --output predictions.jsonl
        """,
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        help="Single text to analyze",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help="Comma-separated list of capabilities or 'all'",
    )
    parser.add_argument(
        "--premise",
        type=str,
        help="Premise text for NLI",
    )
    parser.add_argument(
        "--hypothesis",
        type=str,
        help="Hypothesis text for NLI",
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input JSONL file for batch processing",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output JSONL file for batch results",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive REPL mode",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["json", "pretty", "numpy"],
        default="pretty",
        help="Output format (default: pretty)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda)",
    )
    
    args = parser.parse_args()
    
    # Load model
    engine = MultiTaskInferenceEngine(
        model_path=args.model,
        device=args.device,
    )
    
    # Parse capabilities
    if args.tasks.lower() == "all":
        capabilities = None
    else:
        capabilities = [c.strip() for c in args.tasks.split(",")]
    
    # Run appropriate mode
    if args.interactive:
        run_interactive(engine)
    
    elif args.input and args.output:
        run_batch(engine, args.input, args.output, capabilities)
    
    elif args.text:
        results = engine.infer_all(
            text=args.text,
            capabilities=capabilities,
            premise=args.premise,
            hypothesis=args.hypothesis,
        )
        
        if args.output_format == "json":
            print(json.dumps(results, indent=2))
        elif args.output_format == "numpy":
            # For embedding extraction
            if "embedding" in results.get("results", {}):
                emb = results["results"]["embedding"]["embedding"]
                print(f"Embedding shape: ({len(emb)},)")
                np.save("embedding.npy", np.array(emb))
                print("Saved to embedding.npy")
            else:
                print(json.dumps(results, indent=2))
        else:
            print_pretty(results)
    
    elif args.premise and args.hypothesis:
        # NLI only
        result = engine.infer(
            text="",
            capability="nli",
            premise=args.premise,
            hypothesis=args.hypothesis,
        )
        print(f"\n📊 NLI Result:")
        print(f"  Premise: {args.premise}")
        print(f"  Hypothesis: {args.hypothesis}")
        print(f"  → {result['prediction']} ({result['confidence']:.2%})")
    
    else:
        parser.print_help()
        print("\n❌ Provide --text, --input/--output, or --interactive")


if __name__ == "__main__":
    main()
