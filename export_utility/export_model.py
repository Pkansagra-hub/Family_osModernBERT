#!/usr/bin/env python3
"""
Model Export Script

Export trained models to production-ready formats with full configuration.

Supports:
    - HuggingFace format (safetensors)
    - capabilities.json export
    - Calibration config export
    - Model card generation

Usage:
    # Export full model
    python export_utility/export_model.py \
        --model outputs/modernbert-multitask-v0 \
        --output exports/familyos-unified-v1 \
        --format safetensors

    # Export with specific heads only
    python export_utility/export_model.py \
        --model outputs/modernbert-multitask-v0 \
        --output exports/familyos-safety-only \
        --heads safety_familyos safety_generic

    # Export embedding-only model
    python export_utility/export_model.py \
        --model outputs/modernbert-multitask-v0 \
        --output exports/familyos-embedder \
        --heads embedding \
        --name "FamilyOS Embedder"

    # Include calibration config
    python export_utility/export_model.py \
        --model outputs/modernbert-multitask-v0 \
        --output exports/familyos-unified-v1 \
        --calibration configs/calibration/safety_thresholds.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer

# =============================================================================
# Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Export formats
SUPPORTED_FORMATS = ["safetensors", "pytorch", "huggingface"]

# Model card template
MODEL_CARD_TEMPLATE = """---
language:
- en
license: apache-2.0
library_name: transformers
tags:
- modernbert
- multi-task
- familyos
- nlp
- ner
- sentiment
- safety
- embedding
pipeline_tag: text-classification
---

# {model_name}

{description}

## Model Description

This is a multi-task NLP model based on ModernBERT, trained for the FamilyOS platform.
It supports {num_capabilities} capabilities in a single unified model.

## Capabilities

{capabilities_list}

## Usage

```python
from modeling_studio.models import ModernBertMultiTaskModel
from transformers import AutoTokenizer

# Load model
model = ModernBertMultiTaskModel.from_pretrained("{model_path}")
tokenizer = AutoTokenizer.from_pretrained("{model_path}")

# Inference
text = "Had a wonderful dinner with mom and dad yesterday"
inputs = tokenizer(text, return_tensors="pt")

# Get sentiment
outputs = model(**inputs, task="sentiment")
print(f"Sentiment: {{outputs['logits'].argmax(-1).item()}}")

# Get NER
outputs = model(**inputs, task="ner_family")
print(f"NER: {{outputs['logits'].argmax(-1)}}")
```

## Training Details

- **Base Model:** {base_model}
- **Training Date:** {training_date}
- **Framework:** PyTorch + HuggingFace Transformers

## Evaluation Results

{eval_results}

## License

Apache 2.0
"""


# =============================================================================
# Export Functions
# =============================================================================


def load_model_and_tokenizer(model_path: str | Path) -> tuple:
    """Load model and tokenizer from path."""
    from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

    model_path = Path(model_path)
    logger.info(f"Loading model from {model_path}")

    # Load model
    model = ModernBertMultiTaskModel.from_pretrained(str(model_path))
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    logger.info(f"Model loaded with {len(model.heads)} heads: {list(model.heads.keys())}")

    return model, tokenizer


def filter_heads(model, heads_to_keep: list[str] | None) -> None:
    """Remove heads not in the keep list (in-place modification)."""
    if heads_to_keep is None:
        return

    heads_to_remove = [h for h in model.heads.keys() if h not in heads_to_keep]

    for head_name in heads_to_remove:
        del model.heads[head_name]
        logger.info(f"Removed head: {head_name}")

    logger.info(f"Remaining heads: {list(model.heads.keys())}")


def export_capabilities_json(
    model, output_dir: Path, calibration_config: dict | None = None
) -> None:
    """Export capabilities.json with head configurations."""
    capabilities = {}

    for head_name, head in model.heads.items():
        cap_info = {
            "type": head.__class__.__name__,
            "num_labels": getattr(head, "num_labels", None),
            "hidden_size": getattr(head, "hidden_size", model.config.hidden_size),
        }

        # Add label names if available
        if hasattr(head, "label_names"):
            cap_info["label_names"] = head.label_names
        elif hasattr(head, "config") and hasattr(head.config, "id2label"):
            cap_info["label_names"] = list(head.config.id2label.values())

        # Add calibration info if available
        if calibration_config and head_name in calibration_config:
            cap_info["calibration"] = calibration_config[head_name]

        capabilities[head_name] = cap_info

    # Write capabilities.json
    caps_path = output_dir / "capabilities.json"
    with open(caps_path, "w") as f:
        json.dump(capabilities, f, indent=2)

    logger.info(f"Exported capabilities.json with {len(capabilities)} capabilities")


def export_training_config(model, output_dir: Path, source_path: Path) -> None:
    """Export training configuration if available."""
    # Check for training_config.json in source
    training_config_path = source_path / "training_config.json"
    if training_config_path.exists():
        shutil.copy(training_config_path, output_dir / "training_config.json")
        logger.info("Copied training_config.json")

    # Check for training_args.json
    training_args_path = source_path / "training_args.json"
    if training_args_path.exists():
        shutil.copy(training_args_path, output_dir / "training_args.json")
        logger.info("Copied training_args.json")


def load_calibration_config(calibration_path: str | Path | None) -> dict | None:
    """Load calibration config from YAML file."""
    if calibration_path is None:
        return None

    calibration_path = Path(calibration_path)
    if not calibration_path.exists():
        logger.warning(f"Calibration file not found: {calibration_path}")
        return None

    with open(calibration_path) as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded calibration config from {calibration_path}")
    return config


def export_calibration_config(calibration_config: dict | None, output_dir: Path) -> None:
    """Export calibration config to output directory."""
    if calibration_config is None:
        return

    calibration_path = output_dir / "calibration_config.yaml"
    with open(calibration_path, "w") as f:
        yaml.dump(calibration_config, f, default_flow_style=False)

    logger.info("Exported calibration_config.yaml")


def generate_model_card(
    model,
    output_dir: Path,
    model_name: str,
    description: str,
    base_model: str,
    eval_results: dict | None = None,
) -> None:
    """Generate README.md model card."""
    # Build capabilities list
    capabilities_list = "\n".join(
        [f"- **{cap}**: {head.__class__.__name__}" for cap, head in model.heads.items()]
    )

    # Format eval results
    if eval_results:
        eval_lines = ["| Task | Metric | Value |", "|------|--------|-------|"]
        for task, metrics in eval_results.items():
            for metric, value in metrics.items():
                if isinstance(value, float):
                    eval_lines.append(f"| {task} | {metric} | {value:.4f} |")
        eval_results_str = "\n".join(eval_lines)
    else:
        eval_results_str = "Evaluation results not available."

    # Generate model card
    card_content = MODEL_CARD_TEMPLATE.format(
        model_name=model_name,
        description=description,
        num_capabilities=len(model.heads),
        capabilities_list=capabilities_list,
        model_path=output_dir.name,
        base_model=base_model,
        training_date=datetime.now().strftime("%Y-%m-%d"),
        eval_results=eval_results_str,
    )

    readme_path = output_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(card_content)

    logger.info("Generated README.md model card")


def export_model_safetensors(model, tokenizer, output_dir: Path) -> None:
    """Export model in safetensors format."""
    from safetensors.torch import save_file

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get state dict
    state_dict = model.state_dict()

    # Convert any non-tensor values
    tensors_dict = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            # Ensure contiguous tensors
            tensors_dict[key] = value.contiguous()

    # Save model weights
    model_path = output_dir / "model.safetensors"
    save_file(tensors_dict, str(model_path))
    logger.info(f"Saved model weights to {model_path}")

    # Save config
    config_path = output_dir / "config.json"
    model.config.save_pretrained(str(output_dir))
    logger.info(f"Saved config to {config_path}")

    # Save tokenizer
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Saved tokenizer files")


def export_model_pytorch(model, tokenizer, output_dir: Path) -> None:
    """Export model in PyTorch format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full model
    model_path = output_dir / "pytorch_model.bin"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model weights to {model_path}")

    # Save config
    model.config.save_pretrained(str(output_dir))
    logger.info("Saved config")

    # Save tokenizer
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Saved tokenizer files")


def export_model_huggingface(model, tokenizer, output_dir: Path) -> None:
    """Export model using HuggingFace's save_pretrained."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use model's save_pretrained if available
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(output_dir))
        logger.info("Saved model using save_pretrained")
    else:
        # Fallback to manual save
        export_model_safetensors(model, tokenizer, output_dir)

    # Save tokenizer
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Saved tokenizer files")


def verify_export(output_dir: Path, expected_heads: list[str] | None = None) -> bool:
    """Verify exported model can be loaded correctly."""
    from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

    logger.info("Verifying exported model...")

    try:
        # Load model
        model = ModernBertMultiTaskModel.from_pretrained(str(output_dir))
        tokenizer = AutoTokenizer.from_pretrained(str(output_dir))

        # Check heads
        if expected_heads:
            missing = set(expected_heads) - set(model.heads.keys())
            if missing:
                logger.error(f"Missing heads: {missing}")
                return False

        # Quick inference test
        model.eval()
        with torch.no_grad():
            inputs = tokenizer("Test input", return_tensors="pt")
            for head_name in list(model.heads.keys())[:1]:  # Test first head
                outputs = model(**inputs, task=head_name)
                if "logits" not in outputs and "embeddings" not in outputs:
                    logger.error(f"No output from head {head_name}")
                    return False

        logger.info("✅ Export verification passed")
        return True

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False


def get_export_stats(output_dir: Path) -> dict:
    """Get statistics about exported model."""
    stats = {
        "total_size_mb": 0,
        "files": {},
    }

    for file_path in output_dir.iterdir():
        if file_path.is_file():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            stats["files"][file_path.name] = f"{size_mb:.2f} MB"
            stats["total_size_mb"] += size_mb

    stats["total_size_mb"] = f"{stats['total_size_mb']:.2f} MB"
    return stats


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Export trained models to production formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export full model
  python export_utility/export_model.py \\
      --model outputs/modernbert-multitask-v0 \\
      --output exports/familyos-unified-v1

  # Export specific heads only
  python export_utility/export_model.py \\
      --model outputs/modernbert-multitask-v0 \\
      --output exports/familyos-safety \\
      --heads safety_familyos safety_generic

  # Include calibration config
  python export_utility/export_model.py \\
      --model outputs/modernbert-multitask-v0 \\
      --output exports/familyos-unified-v1 \\
      --calibration configs/calibration/safety_thresholds.yaml
        """,
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Path to source model directory",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to output directory",
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=SUPPORTED_FORMATS,
        default="safetensors",
        help="Export format (default: safetensors)",
    )
    parser.add_argument(
        "--heads",
        type=str,
        nargs="+",
        default=None,
        help="Specific heads to export (default: all)",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default=None,
        help="Path to calibration config YAML",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="FamilyOS Unified Model",
        help="Model name for model card",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="Multi-task NLP model for FamilyOS platform.",
        help="Model description for model card",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="answerdotai/ModernBERT-base",
        help="Base model name",
    )
    parser.add_argument(
        "--eval-results",
        type=str,
        default=None,
        help="Path to eval_results.json for model card",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip export verification",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output directory",
    )

    args = parser.parse_args()

    # Validate paths
    model_path = Path(args.model)
    output_path = Path(args.output)

    if not model_path.exists():
        logger.error(f"Model path does not exist: {model_path}")
        sys.exit(1)

    if output_path.exists() and not args.force:
        logger.error(f"Output path exists: {output_path}. Use --force to overwrite.")
        sys.exit(1)

    if output_path.exists() and args.force:
        logger.warning(f"Overwriting existing output: {output_path}")
        shutil.rmtree(output_path)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)

    # Filter heads if specified
    if args.heads:
        filter_heads(model, args.heads)

    # Load calibration config
    calibration_config = load_calibration_config(args.calibration)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Export model
    logger.info(f"Exporting model in {args.format} format...")

    if args.format == "safetensors":
        export_model_safetensors(model, tokenizer, output_path)
    elif args.format == "pytorch":
        export_model_pytorch(model, tokenizer, output_path)
    else:  # huggingface
        export_model_huggingface(model, tokenizer, output_path)

    # Export capabilities.json
    export_capabilities_json(model, output_path, calibration_config)

    # Export training config
    export_training_config(model, output_path, model_path)

    # Export calibration config
    export_calibration_config(calibration_config, output_path)

    # Load eval results if provided
    eval_results = None
    if args.eval_results:
        eval_path = Path(args.eval_results)
        if eval_path.exists():
            with open(eval_path) as f:
                eval_data = json.load(f)
                eval_results = eval_data.get("tasks", eval_data)

    # Generate model card
    generate_model_card(
        model=model,
        output_dir=output_path,
        model_name=args.name,
        description=args.description,
        base_model=args.base_model,
        eval_results=eval_results,
    )

    # Verify export
    if not args.skip_verify:
        if not verify_export(output_path, args.heads):
            logger.error("Export verification failed!")
            sys.exit(1)

    # Print statistics
    stats = get_export_stats(output_path)
    logger.info("=" * 50)
    logger.info("EXPORT COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Output: {output_path}")
    logger.info(f"Format: {args.format}")
    logger.info(f"Total size: {stats['total_size_mb']}")
    logger.info("Files:")
    for filename, size in stats["files"].items():
        logger.info(f"  - {filename}: {size}")

    logger.info("=" * 50)
    logger.info("✅ Export successful!")


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
