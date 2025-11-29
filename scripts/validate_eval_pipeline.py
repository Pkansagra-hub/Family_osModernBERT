#!/usr/bin/env python
"""
Evaluation Pipeline Validation Script

Quick test to verify the evaluation pipeline works BEFORE running full training.
Tests all task types with minimal data to catch errors early.

Usage:
    python scripts/validate_eval_pipeline.py
    python scripts/validate_eval_pipeline.py --config configs/training/multitask/stage_a_generic.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict:
    """Load YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def test_imports():
    """Test all required imports."""
    print("\n" + "=" * 60)
    print("STEP 1: Testing Imports")
    print("=" * 60)

    imports = [
        ("numpy", "import numpy as np"),
        ("torch", "import torch"),
        ("transformers", "from transformers import AutoTokenizer"),
        ("sklearn", "from sklearn.metrics import f1_score"),
        ("seqeval", "from seqeval.metrics import f1_score as ner_f1"),
        (
            "modeling_studio.models",
            "from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel",
        ),
        (
            "modeling_studio.data.labels",
            "from modeling_studio.data.labels import Capability, get_num_labels, NER_GENERAL_LABELS",
        ),
        (
            "modeling_studio.data.loaders",
            "from modeling_studio.data.loaders import load_stage_a_datasets",
        ),
        (
            "modeling_studio.evaluation.metrics",
            "from modeling_studio.evaluation.metrics import compute_metrics_for_task, get_task_problem_type, aggregate_metrics",
        ),
        (
            "modeling_studio.trainers",
            "from modeling_studio.trainers.multitask_trainer import MultiTaskTrainer",
        ),
        (
            "modeling_studio.trainers.collators",
            "from modeling_studio.trainers.collators import get_task_collator",
        ),
    ]

    all_ok = True
    for name, import_stmt in imports:
        try:
            exec(import_stmt)
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            all_ok = False

    if not all_ok:
        print("\n❌ Import errors detected! Fix before proceeding.")
        sys.exit(1)

    print("\n✅ All imports OK")
    return True


def test_metrics_computation():
    """Test metrics computation for each task type."""
    print("\n" + "=" * 60)
    print("STEP 2: Testing Metrics Computation")
    print("=" * 60)

    import numpy as np

    from modeling_studio.data.labels import NER_GENERAL_LABELS
    from modeling_studio.evaluation.metrics import compute_metrics_for_task

    tasks_tested = []

    # Test 1: NER (token classification)
    print("\n  Testing NER metrics...")
    try:
        # Simulate NER predictions: [batch, seq_len]
        batch_size, seq_len, num_labels = 4, 32, 17
        predictions = np.random.randint(0, num_labels, (batch_size, seq_len))
        labels = np.random.randint(0, num_labels, (batch_size, seq_len))
        # Add some -100 padding
        labels[:, -5:] = -100

        label_list = [NER_GENERAL_LABELS.id2label[i] for i in range(NER_GENERAL_LABELS.num_labels)]

        metrics = compute_metrics_for_task(
            task="ner_general",
            predictions=predictions,
            labels=labels,
            label_list=label_list,
        )
        print(f"    ✓ NER metrics: {list(metrics.keys())}")
        tasks_tested.append("ner_general")
    except Exception as e:
        print(f"    ✗ NER metrics failed: {e}")
        raise

    # Test 2: Classification (single-label)
    print("\n  Testing classification metrics...")
    try:
        batch_size, num_labels = 16, 5
        predictions = np.random.randint(0, num_labels, (batch_size,))
        labels = np.random.randint(0, num_labels, (batch_size,))

        metrics = compute_metrics_for_task(
            task="sentiment",
            predictions=predictions,
            labels=labels,
        )
        print(f"    ✓ Classification metrics: {list(metrics.keys())}")
        tasks_tested.append("sentiment")
    except Exception as e:
        print(f"    ✗ Classification metrics failed: {e}")
        raise

    # Test 3: Multi-label classification
    print("\n  Testing multi-label metrics...")
    try:
        batch_size, num_labels = 16, 8
        # Logits (will be sigmoided)
        predictions = np.random.randn(batch_size, num_labels).astype(np.float32)
        labels = np.random.randint(0, 2, (batch_size, num_labels))

        metrics = compute_metrics_for_task(
            task="safety_generic",
            predictions=predictions,
            labels=labels,
        )
        print(f"    ✓ Multi-label metrics: {list(metrics.keys())}")
        tasks_tested.append("safety_generic")
    except Exception as e:
        print(f"    ✗ Multi-label metrics failed: {e}")
        raise

    # Test 4: NLI
    print("\n  Testing NLI metrics...")
    try:
        batch_size, num_labels = 16, 3
        predictions = np.random.randint(0, num_labels, (batch_size,))
        labels = np.random.randint(0, num_labels, (batch_size,))

        metrics = compute_metrics_for_task(
            task="nli",
            predictions=predictions,
            labels=labels,
        )
        print(f"    ✓ NLI metrics: {list(metrics.keys())}")
        tasks_tested.append("nli")
    except Exception as e:
        print(f"    ✗ NLI metrics failed: {e}")
        raise

    # Test 5: Emotions (multi-label)
    print("\n  Testing emotions metrics...")
    try:
        batch_size, num_labels = 16, 32
        predictions = np.random.randn(batch_size, num_labels).astype(np.float32)
        labels = np.random.randint(0, 2, (batch_size, num_labels))

        metrics = compute_metrics_for_task(
            task="emotions",
            predictions=predictions,
            labels=labels,
        )
        print(f"    ✓ Emotions metrics: {list(metrics.keys())}")
        tasks_tested.append("emotions")
    except Exception as e:
        print(f"    ✗ Emotions metrics failed: {e}")
        raise

    print(f"\n✅ All {len(tasks_tested)} task metrics compute successfully")
    return tasks_tested


def test_bf16_to_numpy():
    """Test BFloat16 to numpy conversion (the actual bug we fixed)."""
    print("\n" + "=" * 60)
    print("STEP 3: Testing BFloat16 → NumPy Conversion")
    print("=" * 60)

    # Create bf16 tensors like we get from model output
    print("\n  Creating BFloat16 tensors...")
    predictions = torch.randn(8, 32, 17, dtype=torch.bfloat16)
    labels = torch.randint(0, 17, (8, 32), dtype=torch.long)

    print(f"    predictions dtype: {predictions.dtype}")
    print(f"    predictions shape: {predictions.shape}")

    # Test the WRONG way (should fail)
    print("\n  Testing WRONG conversion (direct .numpy())...")
    try:
        _ = predictions.cpu().numpy()
        print("    ✗ Unexpectedly succeeded - numpy may support bf16 now?")
    except TypeError as e:
        print(f"    ✓ Correctly failed: {e}")

    # Test the RIGHT way (our fix)
    print("\n  Testing CORRECT conversion (.float().numpy())...")
    try:
        pred_np = predictions.cpu().float().numpy()
        print(f"    ✓ Conversion succeeded, dtype: {pred_np.dtype}")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        raise

    print("\n✅ BFloat16 conversion test passed")
    return True


def test_tensor_concatenation():
    """Test tensor concatenation with variable sequence lengths."""
    print("\n" + "=" * 60)
    print("STEP 4: Testing Tensor Concatenation (Variable Seq Lengths)")
    print("=" * 60)

    from modeling_studio.trainers.multitask_trainer import MultiTaskTrainer

    # Create a minimal trainer just to test _concatenate_tensors
    class MockTrainer:
        pass

    # Monkey-patch the method
    mock = MockTrainer()
    mock._concatenate_tensors = MultiTaskTrainer._concatenate_tensors.__get__(mock, MockTrainer)

    # Test 1D (classification labels)
    print("\n  Testing 1D tensors (classification)...")
    tensors_1d = [torch.tensor([0, 1, 2]), torch.tensor([3, 4])]
    result = mock._concatenate_tensors(tensors_1d)
    print(f"    ✓ 1D concat: {result.shape}")

    # Test 2D with same seq_len
    print("\n  Testing 2D tensors (same seq_len)...")
    tensors_2d_same = [torch.randint(0, 10, (4, 32)), torch.randint(0, 10, (4, 32))]
    result = mock._concatenate_tensors(tensors_2d_same)
    print(f"    ✓ 2D same len: {result.shape}")

    # Test 2D with DIFFERENT seq_len (the bug case)
    print("\n  Testing 2D tensors (DIFFERENT seq_len - bug case)...")
    tensors_2d_diff = [
        torch.randint(0, 10, (4, 32)),
        torch.randint(0, 10, (4, 48)),
        torch.randint(0, 10, (4, 24)),
    ]
    result = mock._concatenate_tensors(tensors_2d_diff)
    print(f"    ✓ 2D diff len: {result.shape} (padded to max)")

    # Test 3D with different seq_len (NER logits)
    print("\n  Testing 3D tensors (NER logits, diff seq_len)...")
    tensors_3d_diff = [
        torch.randn(4, 32, 17),
        torch.randn(4, 48, 17),
        torch.randn(4, 24, 17),
    ]
    result = mock._concatenate_tensors(tensors_3d_diff)
    print(f"    ✓ 3D diff len: {result.shape} (padded to max)")

    print("\n✅ Tensor concatenation tests passed")
    return True


def test_model_forward_and_metrics(config: dict | None = None):
    """Test model forward pass and full metrics pipeline."""
    print("\n" + "=" * 60)
    print("STEP 5: Testing Model Forward + Metrics Pipeline")
    print("=" * 60)

    import numpy as np
    from transformers import AutoTokenizer

    from modeling_studio.data.labels import NER_GENERAL_LABELS, Capability, get_num_labels
    from modeling_studio.evaluation.metrics import compute_metrics_for_task
    from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Using device: {device}")

    # Get model path from config or use default
    model_path = "answerdotai/ModernBERT-base"
    if config and "model" in config:
        model_path = config["model"].get("name_or_path", model_path)

    print(f"  Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Test tasks
    test_tasks = [
        ("ner_general", "token_classification"),
        ("sentiment", "single_label_classification"),
        ("nli", "single_label_classification"),
        ("safety_generic", "multi_label_classification"),
    ]

    # Load model with capabilities
    capabilities = [t[0] for t in test_tasks]
    print(f"\n  Loading model with capabilities: {capabilities}")

    model = ModernBertMultiTaskModel.from_pretrained(
        model_path,
        capabilities=capabilities,
        torch_dtype=torch.bfloat16,
    )
    # Move to device AND ensure all parameters have consistent dtype
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    print(f"  Model loaded, params: {sum(p.numel() for p in model.parameters()):,}")

    # Test each task
    for task, task_type in test_tasks:
        print(f"\n  Testing {task} ({task_type})...")

        # Create fake batch
        batch_size = 4
        seq_len = 64

        input_ids = torch.randint(1000, 30000, (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)

        # Create appropriate labels
        num_labels = get_num_labels(Capability(task))

        if task_type == "token_classification":
            labels = torch.randint(0, num_labels, (batch_size, seq_len), device=device)
            labels[:, -10:] = -100  # Padding
        elif task_type == "multi_label_classification":
            labels = torch.randint(0, 2, (batch_size, num_labels), device=device).float()
        else:
            labels = torch.randint(0, num_labels, (batch_size,), device=device)

        # Forward pass
        with torch.no_grad():
            outputs = model(
                capability=task,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        loss = outputs.loss
        logits = outputs.logits

        print(f"    Forward pass OK - loss: {loss.item():.4f}, logits shape: {logits.shape}")

        # Convert to numpy (THE BUG FIX TEST)
        # Use .float() for logits (they can be bf16), but keep labels as int for classification
        logits_np = logits.detach().cpu().float().numpy()

        # Labels need to stay as int for classification/NER, float only for multi-label
        if task_type == "multi_label_classification":
            labels_np = labels.detach().cpu().float().numpy()
        else:
            labels_np = labels.detach().cpu().numpy()  # Keep as int

        print(f"    NumPy conversion OK - logits: {logits_np.dtype}, labels: {labels_np.dtype}")

        # Compute predictions
        if task_type == "token_classification":
            predictions = np.argmax(logits_np, axis=-1)
        elif task_type == "multi_label_classification":
            predictions = logits_np  # Keep as logits, metrics will threshold
        else:
            predictions = np.argmax(logits_np, axis=-1)

        # Compute metrics
        label_list = None
        if task == "ner_general":
            label_list = [
                NER_GENERAL_LABELS.id2label[i] for i in range(NER_GENERAL_LABELS.num_labels)
            ]

        metrics = compute_metrics_for_task(
            task=task,
            predictions=predictions,
            labels=(
                labels_np.astype(int) if task_type == "multi_label_classification" else labels_np
            ),
            label_list=label_list,
        )

        print(f"    ✓ Metrics computed: {list(metrics.keys())[:4]}...")

    print("\n✅ Model forward + metrics pipeline OK")
    return True


def test_full_eval_loop(config: dict | None = None):
    """Test the full evaluation loop from trainer."""
    print("\n" + "=" * 60)
    print("STEP 6: Testing Full Trainer Evaluation Loop")
    print("=" * 60)

    from torch.utils.data import Dataset
    from transformers import AutoTokenizer, TrainingArguments

    from modeling_studio.data.labels import Capability, get_num_labels
    from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel
    from modeling_studio.trainers.multitask_trainer import MultiTaskTrainer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get model path
    model_path = "answerdotai/ModernBERT-base"
    if config and "model" in config:
        model_path = config["model"].get("name_or_path", model_path)

    print("\n  Creating minimal datasets...")

    # Create minimal fake datasets for testing
    # The collators expect raw python lists, not tensors
    class FakeDataset(Dataset):
        def __init__(self, task: str, size: int = 20):
            self.task = task
            self.size = size
            self.num_labels = get_num_labels(Capability(task))

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            import random

            seq_len = 64

            # Return lists, not tensors - collator will handle conversion
            item = {
                "input_ids": [random.randint(1000, 30000) for _ in range(seq_len)],
                "attention_mask": [1] * seq_len,
                "task": self.task,
            }

            # Task-specific labels
            if self.task in ["ner_general", "temporal"]:
                # NER: list of int labels, with -100 for padding positions
                labels = [random.randint(0, self.num_labels - 1) for _ in range(seq_len)]
                for i in range(-10, 0):
                    labels[i] = -100
                item["labels"] = labels
            elif self.task in ["emotions", "safety_generic"]:
                # Multi-label: list of 0/1 for each label
                item["labels"] = [random.randint(0, 1) for _ in range(self.num_labels)]
            else:
                # Single-label: just an int
                item["labels"] = random.randint(0, self.num_labels - 1)

            return item

    # Tasks to test
    tasks = ["ner_general", "sentiment", "nli", "safety_generic"]

    # Create eval datasets
    eval_datasets = {task: FakeDataset(task, size=16) for task in tasks}
    train_datasets = {task: FakeDataset(task, size=16) for task in tasks}  # Needed for trainer init

    print(f"  Created datasets for: {list(eval_datasets.keys())}")

    # Load model
    print("\n  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = ModernBertMultiTaskModel.from_pretrained(
        model_path,
        capabilities=tasks,
        torch_dtype=torch.bfloat16,
    )
    # Ensure dtype consistency across all modules
    model = model.to(dtype=torch.bfloat16)

    # Create training args
    args = TrainingArguments(
        output_dir="./tmp_eval_test",
        per_device_eval_batch_size=8,
        do_train=False,
        do_eval=True,
        report_to="none",
        disable_tqdm=True,
    )

    # Create trainer
    print("\n  Creating trainer...")
    trainer = MultiTaskTrainer(
        model=model,
        args=args,
        train_datasets=train_datasets,
        eval_datasets=eval_datasets,
        tokenizer=tokenizer,
    )

    # Run evaluation
    print("\n  Running evaluation...")
    try:
        metrics = trainer.evaluate()

        print("\n  Evaluation completed! Metrics:")
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                print(f"    {key}: {value:.4f}")
            else:
                print(f"    {key}: {value}")

        print("\n✅ Full evaluation loop passed!")
        return True

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        # Cleanup
        import shutil

        if Path("./tmp_eval_test").exists():
            shutil.rmtree("./tmp_eval_test")


def test_real_data_eval(config: dict | None = None):
    """Test evaluation with REAL tokenized datasets - 200 samples per task."""
    print("\n" + "=" * 60)
    print("STEP 7: Testing with REAL TOKENIZED DATA (200 samples/task)")
    print("=" * 60)

    import shutil

    from transformers import AutoTokenizer, TrainingArguments

    from modeling_studio.data.loaders import load_stage_a_datasets
    from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel
    from modeling_studio.trainers.multitask_trainer import MultiTaskTrainer

    # Get model path
    model_path = "answerdotai/ModernBERT-base"
    if config and "model" in config:
        model_path = config["model"].get("name_or_path", model_path)

    print(f"\n  Loading tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("\n  Loading REAL datasets with tokenization...")
    try:
        train_ds = load_stage_a_datasets(
            config_path="configs/data/multitask/stage_a_datasets.yaml",
            split="train",
            tokenizer=tokenizer,
            apply_tokenization=True,
        )
        eval_ds = load_stage_a_datasets(
            config_path="configs/data/multitask/stage_a_datasets.yaml",
            split="validation",
            tokenizer=tokenizer,
            apply_tokenization=True,
        )
    except Exception as e:
        print(f"\n  ❌ Failed to load datasets: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Show what we loaded
    print("\n  Datasets loaded:")
    for task, ds in eval_ds.items():
        print(f"    {task}: {len(ds)} samples, columns: {ds.column_names[:5]}...")

    # Limit to 200 samples each for quick testing
    print("\n  Limiting to 200 eval samples per task...")
    for task in eval_ds:
        if len(eval_ds[task]) > 200:
            eval_ds[task] = eval_ds[task].select(range(200))

    # Also limit train for trainer init
    train_small = {}
    for task in train_ds:
        if len(train_ds[task]) > 100:
            train_small[task] = train_ds[task].select(range(100))
        else:
            train_small[task] = train_ds[task]

    print("\n  Final dataset sizes:")
    for task in eval_ds:
        print(f"    {task}: train={len(train_small[task])}, eval={len(eval_ds[task])}")

    # Load model
    print(f"\n  Loading model with {len(eval_ds)} capabilities...")
    model = ModernBertMultiTaskModel.from_pretrained(
        model_path,
        capabilities=list(eval_ds.keys()),
        torch_dtype=torch.bfloat16,
    )
    # CRITICAL: Move entire model to bf16 to ensure heads have same dtype as encoder
    model = model.to(dtype=torch.bfloat16)

    # Create training args
    args = TrainingArguments(
        output_dir="./tmp_real_eval_test",
        per_device_eval_batch_size=8,
        do_train=False,
        do_eval=True,
        report_to="none",
        disable_tqdm=False,
    )

    # Create trainer
    print("\n  Creating trainer...")
    trainer = MultiTaskTrainer(
        model=model,
        args=args,
        train_datasets=train_small,
        eval_datasets=eval_ds,
        tokenizer=tokenizer,
    )

    # Run evaluation
    print("\n  Running evaluation on REAL data...")
    try:
        metrics = trainer.evaluate()

        print("\n  " + "=" * 50)
        print("  REAL DATA EVALUATION RESULTS")
        print("  " + "=" * 50)
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                print(f"    {key}: {value:.4f}")
            else:
                print(f"    {key}: {value}")

        print("\n✅ REAL DATA evaluation completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ REAL DATA evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if Path("./tmp_real_eval_test").exists():
            shutil.rmtree("./tmp_real_eval_test")


def main():
    parser = argparse.ArgumentParser(description="Validate evaluation pipeline")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--skip-model", action="store_true", help="Skip model loading tests")
    parser.add_argument("--real-data", action="store_true", help="Test with real tokenized data")
    args = parser.parse_args()

    config = None
    if args.config:
        config = load_config(args.config)
        print(f"Loaded config from: {args.config}")

    print("\n" + "=" * 60)
    print("EVALUATION PIPELINE VALIDATION")
    print("=" * 60)
    print("This script validates the evaluation pipeline BEFORE training")
    print("to catch errors early and avoid wasting training time.")

    results = {}

    # Step 1: Imports
    results["imports"] = test_imports()

    # Step 2: Metrics computation
    results["metrics"] = test_metrics_computation()

    # Step 3: BFloat16 conversion
    results["bf16"] = test_bf16_to_numpy()

    # Step 4: Tensor concatenation
    results["concat"] = test_tensor_concatenation()

    if not args.skip_model:
        # Step 5: Model forward + metrics
        results["model"] = test_model_forward_and_metrics(config)

        # Step 6: Full eval loop with fake data
        results["eval_loop"] = test_full_eval_loop(config)

        # Step 7: Real data test (if requested)
        if args.real_data:
            results["real_data"] = test_real_data_eval(config)
    else:
        print("\n⏭️  Skipping model tests (--skip-model)")

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n" + "=" * 60)
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("=" * 60)
        print("\nYou can now run training with confidence:")
        print(
            "  python scripts/train_stage_a.py --config configs/training/multitask/stage_a_generic.yaml"
        )
    else:
        print("\n" + "=" * 60)
        print("💥 VALIDATION FAILED - FIX ERRORS BEFORE TRAINING")
        print("=" * 60)
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
