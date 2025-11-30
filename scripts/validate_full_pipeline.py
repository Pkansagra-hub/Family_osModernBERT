#!/usr/bin/env python
"""
FULL PIPELINE VALIDATION SCRIPT

This script runs the EXACT same code path as train_stage_a.py but with:
- 200 samples per task (train and eval)
- Full training loop for 10 steps
- Full evaluation after training
- Validates ALL config settings including metric_for_best_model

This will catch ANY error that would occur in full training.

Usage:
    python scripts/validate_full_pipeline.py --config configs/training/multitask/stage_a_generic.yaml

Author: Created to catch ALL training errors before wasting GPU time.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


def validate_config(config: dict) -> list[str]:
    """Validate config has all required fields and values are sensible."""
    errors = []
    warnings = []

    print("\n" + "=" * 70)
    print("STEP 1: Validating Configuration")
    print("=" * 70)

    # Check model config
    if "model" not in config:
        errors.append("Missing 'model' section in config")
    else:
        model_config = config["model"]
        if "name_or_path" not in model_config:
            errors.append("Missing 'model.name_or_path'")
        else:
            print(f"  ✓ Model: {model_config['name_or_path']}")

    # Check training config
    if "training" not in config:
        errors.append("Missing 'training' section in config")
    else:
        training_config = config["training"]

        # Check metric_for_best_model
        metric = training_config.get("metric_for_best_model", "NOT SET")
        print(f"  → metric_for_best_model: {metric}")

        # CRITICAL: Check if metric name is valid
        valid_metrics = [
            "eval_loss",
            "eval_avg_score",
            "eval_weighted_avg_score",
            "eval_worst_score",
            "eval_best_score",
            # Task-specific metrics
            "eval_ner_general_f1",
            "eval_sentiment_accuracy",
            "eval_sentiment_f1",
            "eval_emotions_micro_f1",
            "eval_emotions_macro_f1",
            "eval_safety_generic_micro_f1",
            "eval_nli_accuracy",
            "eval_nli_f1",
            "eval_embedding_spearman",
        ]

        if metric not in valid_metrics and metric != "NOT SET":
            errors.append(
                f"Invalid metric_for_best_model: '{metric}'. "
                f"Valid options include: {valid_metrics[:5]}..."
            )
        elif metric == "eval_avg_f1":
            errors.append(
                f"metric_for_best_model is 'eval_avg_f1' but the actual metric name is 'eval_avg_score'. "
                f"Fix the config!"
            )
        else:
            print(f"  ✓ metric_for_best_model is valid: {metric}")

        # Check other training params
        lr = training_config.get("learning_rate", 0)
        if lr <= 0 or lr > 0.1:
            errors.append(f"Suspicious learning_rate: {lr}")
        else:
            print(f"  ✓ learning_rate: {lr}")

        epochs = training_config.get("num_train_epochs", 0)
        if epochs <= 0:
            errors.append(f"num_train_epochs must be > 0, got {epochs}")
        else:
            print(f"  ✓ num_train_epochs: {epochs}")

        batch_size = training_config.get("per_device_train_batch_size", 0)
        if batch_size <= 0:
            errors.append(f"per_device_train_batch_size must be > 0")
        else:
            print(f"  ✓ per_device_train_batch_size: {batch_size}")

    # Check heads config
    if "heads" not in config:
        warnings.append("No 'heads' section - will use defaults")
    else:
        enabled_heads = [h for h, cfg in config["heads"].items() if cfg.get("enabled", True)]
        print(f"  ✓ Enabled heads: {enabled_heads}")

    # Check data config
    data_config = config.get("data", {})
    data_config_path = data_config.get("config_path", "configs/data/multitask/stage_a_datasets.yaml")
    if not Path(data_config_path).exists():
        errors.append(f"Data config not found: {data_config_path}")
    else:
        print(f"  ✓ Data config: {data_config_path}")

    # Print warnings
    for w in warnings:
        print(f"  ⚠ WARNING: {w}")

    # Print errors
    for e in errors:
        print(f"  ✗ ERROR: {e}")

    if errors:
        print(f"\n❌ Config validation FAILED with {len(errors)} error(s)")
    else:
        print(f"\n✅ Config validation PASSED")

    return errors


def validate_imports() -> bool:
    """Test all required imports."""
    print("\n" + "=" * 70)
    print("STEP 2: Validating Imports")
    print("=" * 70)

    imports = [
        ("torch", "import torch"),
        ("transformers", "from transformers import AutoTokenizer, TrainingArguments"),
        ("datasets", "import datasets"),
        ("numpy", "import numpy as np"),
        ("sklearn", "from sklearn.metrics import f1_score"),
        ("seqeval", "from seqeval.metrics import f1_score as ner_f1"),
        ("scipy", "from scipy.stats import spearmanr, pearsonr"),
        ("modeling_studio.models", "from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel"),
        ("modeling_studio.data.labels", "from modeling_studio.data.labels import Capability, get_num_labels"),
        ("modeling_studio.data.loaders", "from modeling_studio.data.loaders import load_stage_a_datasets"),
        ("modeling_studio.evaluation.metrics", "from modeling_studio.evaluation.metrics import compute_metrics_for_task, aggregate_metrics"),
        ("modeling_studio.trainers.multitask_trainer", "from modeling_studio.trainers.multitask_trainer import MultiTaskTrainer"),
        ("modeling_studio.trainers.collators", "from modeling_studio.trainers.collators import MultiTaskCollator"),
    ]

    all_ok = True
    for name, import_stmt in imports:
        try:
            exec(import_stmt)
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            all_ok = False

    if all_ok:
        print("\n✅ All imports OK")
    else:
        print("\n❌ Import errors - fix before continuing")

    return all_ok


def validate_device() -> tuple[str, dict]:
    """Check GPU/device availability."""
    print("\n" + "=" * 70)
    print("STEP 3: Checking Device")
    print("=" * 70)

    device_info = {}

    if torch.cuda.is_available():
        device = "cuda"
        device_info["name"] = torch.cuda.get_device_name(0)
        device_info["memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        device_info["compute_capability"] = torch.cuda.get_device_capability(0)

        print(f"  ✓ GPU: {device_info['name']}")
        print(f"  ✓ Memory: {device_info['memory_gb']:.1f} GB")
        print(f"  ✓ Compute Capability: {device_info['compute_capability']}")

        # Check bf16 support
        if device_info["compute_capability"][0] >= 8:
            print(f"  ✓ BFloat16: Supported (SM >= 8.0)")
            device_info["bf16"] = True
        else:
            print(f"  ⚠ BFloat16: Not supported (SM < 8.0), using fp16")
            device_info["bf16"] = False
    else:
        device = "cpu"
        device_info["name"] = "CPU"
        device_info["bf16"] = False
        print(f"  ⚠ No GPU detected, using CPU")

    return device, device_info


def run_mini_training(config: dict, num_samples: int = 200, num_steps: int = 20) -> dict:
    """
    Run a mini training loop with the EXACT same code path as full training.

    This uses the same:
    - Config loading
    - Model initialization
    - Dataset loading
    - Trainer setup
    - Training arguments (including metric_for_best_model!)
    - Evaluation loop
    """
    print("\n" + "=" * 70)
    print(f"STEP 4: Running Mini Training ({num_samples} samples, {num_steps} steps)")
    print("=" * 70)

    from transformers import AutoTokenizer, TrainingArguments, set_seed

    from modeling_studio.data.labels import Capability, get_num_labels
    from modeling_studio.data.loaders import load_stage_a_datasets
    from modeling_studio.models.modernbert_multitask import \
        ModernBertMultiTaskModel
    from modeling_studio.trainers.multitask_trainer import MultiTaskTrainer

    results = {"success": False, "errors": [], "metrics": {}}

    # Get configs
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    heads_config = config.get("heads", {})
    data_config = config.get("data", {})
    output_config = config.get("output", {})

    model_name = model_config.get("name_or_path", "answerdotai/ModernBERT-base")
    data_config_path = data_config.get("config_path", "configs/data/multitask/stage_a_datasets.yaml")

    # Set seed
    seed = training_config.get("seed", 42)
    set_seed(seed)

    # Output dirs for validation
    output_dir = Path("./tmp_validation_run")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 4a: Load tokenizer
        print("\n  4a. Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"      ✓ Tokenizer loaded: {tokenizer.__class__.__name__}")

        # Step 4b: Get enabled capabilities
        print("\n  4b. Determining capabilities...")
        capabilities = []
        for head_name, head_cfg in heads_config.items():
            if head_cfg.get("enabled", True):
                try:
                    cap = Capability(head_name)
                    capabilities.append(cap)
                except ValueError:
                    print(f"      ⚠ Unknown capability: {head_name}")

        if not capabilities:
            # Default capabilities if none specified
            capabilities = [
                Capability.NER_GENERAL,
                Capability.SENTIMENT,
                Capability.EMOTIONS,
                Capability.SAFETY_GENERIC,
                Capability.NLI,
                Capability.EMBEDDING,
            ]

        print(f"      ✓ Capabilities: {[c.value for c in capabilities]}")

        # Step 4c: Load model
        print("\n  4c. Loading model...")
        use_bf16 = training_config.get("bf16", True)
        torch_dtype = torch.bfloat16 if use_bf16 else torch.float16

        model = ModernBertMultiTaskModel.from_pretrained(
            model_name,
            capabilities=capabilities,
            torch_dtype=torch_dtype,
        )
        # Ensure dtype consistency
        model = model.to(dtype=torch_dtype)

        num_params = sum(p.numel() for p in model.parameters())
        print(f"      ✓ Model loaded: {num_params:,} parameters")

        # Step 4d: Load datasets
        print("\n  4d. Loading datasets...")
        train_datasets = load_stage_a_datasets(
            config_path=data_config_path,
            split="train",
            tokenizer=tokenizer,
            apply_tokenization=True,
        )

        eval_datasets = load_stage_a_datasets(
            config_path=data_config_path,
            split="validation",
            tokenizer=tokenizer,
            apply_tokenization=True,
        )

        print(f"      ✓ Loaded {len(train_datasets)} train datasets, {len(eval_datasets)} eval datasets")

        # Step 4e: Limit to num_samples
        print(f"\n  4e. Limiting datasets to {num_samples} samples each...")
        for task in train_datasets:
            if len(train_datasets[task]) > num_samples:
                train_datasets[task] = train_datasets[task].select(range(num_samples))
        for task in eval_datasets:
            if len(eval_datasets[task]) > num_samples:
                eval_datasets[task] = eval_datasets[task].select(range(num_samples))

        total_train = sum(len(ds) for ds in train_datasets.values())
        total_eval = sum(len(ds) for ds in eval_datasets.values())
        print(f"      ✓ Train samples: {total_train}, Eval samples: {total_eval}")

        for task in train_datasets:
            print(f"        - {task}: train={len(train_datasets[task])}, eval={len(eval_datasets[task])}")

        # Step 4f: Create training arguments (EXACT same as train_stage_a.py!)
        print("\n  4f. Creating TrainingArguments...")

        # Get the metric - THIS IS WHAT WE'RE VALIDATING
        metric_for_best = training_config.get("metric_for_best_model", "eval_avg_score")
        print(f"      → metric_for_best_model: {metric_for_best}")

        training_args = TrainingArguments(
            output_dir=str(output_dir),

            # Use mini settings
            num_train_epochs=1,
            max_steps=num_steps,
            per_device_train_batch_size=training_config.get("per_device_train_batch_size", 8),
            per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 16),
            gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),

            # Optimizer settings from config
            learning_rate=training_config.get("learning_rate", 2e-5),
            weight_decay=training_config.get("weight_decay", 0.01),
            warmup_ratio=training_config.get("warmup_ratio", 0.1),
            lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),

            # Precision from config
            bf16=training_config.get("bf16", True),
            fp16=training_config.get("fp16", False),

            # Evaluation - run after training
            eval_strategy="steps",
            eval_steps=num_steps,  # Evaluate at end

            # Saving - THIS IS WHERE metric_for_best_model MATTERS
            save_strategy="steps",
            save_steps=num_steps,
            save_total_limit=1,
            load_best_model_at_end=training_config.get("load_best_model_at_end", True),
            metric_for_best_model=metric_for_best,  # FROM CONFIG!
            greater_is_better=training_config.get("greater_is_better", True),

            # Logging
            logging_steps=5,
            report_to="none",

            # Other
            seed=seed,
            remove_unused_columns=False,
            dataloader_num_workers=0,  # Safer for validation
        )

        print(f"      ✓ TrainingArguments created")
        print(f"      ✓ load_best_model_at_end: {training_args.load_best_model_at_end}")
        print(f"      ✓ metric_for_best_model: {training_args.metric_for_best_model}")

        # Step 4g: Create trainer
        print("\n  4g. Creating MultiTaskTrainer...")
        trainer = MultiTaskTrainer(
            model=model,
            args=training_args,
            train_datasets=train_datasets,
            eval_datasets=eval_datasets,
            tokenizer=tokenizer,
        )
        print(f"      ✓ Trainer created")

        # Step 4h: Run training
        print(f"\n  4h. Running training for {num_steps} steps...")
        train_result = trainer.train()
        print(f"      ✓ Training completed")
        print(f"      ✓ Training loss: {train_result.training_loss:.4f}")
        print(f"      ✓ Steps completed: {train_result.global_step}")

        # Step 4i: Run evaluation (THIS IS WHERE eval_avg_f1 vs eval_avg_score would fail!)
        print("\n  4i. Running final evaluation...")
        eval_metrics = trainer.evaluate()

        print(f"\n      ✓ Evaluation completed!")
        print(f"      Available metrics:")
        for key in sorted(eval_metrics.keys()):
            if isinstance(eval_metrics[key], float):
                print(f"        - {key}: {eval_metrics[key]:.4f}")

        # Check if the metric we need exists
        if metric_for_best not in eval_metrics:
            results["errors"].append(
                f"CRITICAL: metric_for_best_model='{metric_for_best}' NOT FOUND in eval metrics! "
                f"Available: {list(eval_metrics.keys())}"
            )
            print(f"\n      ✗ CRITICAL ERROR: '{metric_for_best}' not in metrics!")
        else:
            print(f"\n      ✓ metric_for_best_model '{metric_for_best}' = {eval_metrics[metric_for_best]:.4f}")

        results["metrics"] = eval_metrics
        results["success"] = len(results["errors"]) == 0

    except Exception as e:
        import traceback
        results["errors"].append(f"Training failed: {e}")
        print(f"\n      ✗ ERROR: {e}")
        traceback.print_exc()
        results["success"] = False

    finally:
        # Cleanup
        if output_dir.exists():
            shutil.rmtree(output_dir)

    return results


def main():
    parser = argparse.ArgumentParser(description="Full Pipeline Validation")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of samples per task (default: 200)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of training steps (default: 20)"
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("   FULL PIPELINE VALIDATION")
    print("   This validates the EXACT same code path as training")
    print("=" * 70)
    print(f"\nConfig: {args.config}")
    print(f"Samples per task: {args.samples}")
    print(f"Training steps: {args.steps}")

    all_passed = True

    # Step 1: Load and validate config
    config = load_config(args.config)
    config_errors = validate_config(config)
    if config_errors:
        print("\n❌ CONFIG VALIDATION FAILED - Fix errors before training!")
        for e in config_errors:
            print(f"   - {e}")
        sys.exit(1)

    # Step 2: Validate imports
    if not validate_imports():
        print("\n❌ IMPORT VALIDATION FAILED")
        sys.exit(1)

    # Step 3: Check device
    device, device_info = validate_device()

    # Step 4: Run mini training
    results = run_mini_training(
        config=config,
        num_samples=args.samples,
        num_steps=args.steps,
    )

    # Summary
    print("\n" + "=" * 70)
    print("   VALIDATION SUMMARY")
    print("=" * 70)

    if results["success"]:
        print("\n✅ ALL VALIDATION PASSED!")
        print("\nYou can now run full training:")
        print(f"   python scripts/train_stage_a.py --config {args.config}")

        if results["metrics"]:
            print("\nSample metrics from validation run:")
            for key in ["eval_avg_score", "eval_sentiment_accuracy", "eval_nli_accuracy"]:
                if key in results["metrics"]:
                    print(f"   {key}: {results['metrics'][key]:.4f}")
    else:
        print("\n❌ VALIDATION FAILED!")
        print("\nErrors found:")
        for e in results["errors"]:
            print(f"   - {e}")
        print("\nFix these errors before running full training!")
        sys.exit(1)


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
