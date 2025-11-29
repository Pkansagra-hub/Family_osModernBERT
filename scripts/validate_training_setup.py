#!/usr/bin/env python
"""
Validate Training Setup

Run this BEFORE training to verify:
1. All imports work
2. Model initializes correctly
3. Each dataset loads and tokenizes properly
4. Collators work with sample data
5. A single forward pass works for each task
6. GPU/device is available

Usage:
    python scripts/validate_training_setup.py
    python scripts/validate_training_setup.py --verbose
    python scripts/validate_training_setup.py --task sentiment  # Test single task
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class Colors:
    """ANSI colors for terminal output."""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def ok(msg: str) -> None:
    print(f"  {Colors.GREEN}✓{Colors.RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {Colors.RED}✗{Colors.RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {Colors.YELLOW}!{Colors.RESET} {msg}")


def info(msg: str) -> None:
    print(f"  {Colors.BLUE}→{Colors.RESET} {msg}")


def header(msg: str) -> None:
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{msg}{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")


def subheader(msg: str) -> None:
    print(f"\n{Colors.BLUE}--- {msg} ---{Colors.RESET}")


def validate_imports() -> bool:
    """Test that all required imports work."""
    header("Step 1: Validating Imports")

    imports_ok = True

    # Core imports
    try:
        import torch

        ok(f"torch {torch.__version__}")
    except ImportError as e:
        fail(f"torch: {e}")
        imports_ok = False

    try:
        import transformers

        ok(f"transformers {transformers.__version__}")
    except ImportError as e:
        fail(f"transformers: {e}")
        imports_ok = False

    try:
        import datasets

        ok(f"datasets {datasets.__version__}")
    except ImportError as e:
        fail(f"datasets: {e}")
        imports_ok = False

    # Project imports
    try:
        from modeling_studio.data.labels import Capability, get_num_labels

        ok("modeling_studio.data.labels")
    except ImportError as e:
        fail(f"modeling_studio.data.labels: {e}")
        imports_ok = False

    try:
        from modeling_studio.data.loaders import load_stage_a_datasets

        ok("modeling_studio.data.loaders")
    except ImportError as e:
        fail(f"modeling_studio.data.loaders: {e}")
        imports_ok = False

    try:
        from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

        ok("modeling_studio.models.modernbert_multitask")
    except ImportError as e:
        fail(f"modeling_studio.models.modernbert_multitask: {e}")
        imports_ok = False

    try:
        from modeling_studio.trainers.multitask_trainer import MultiTaskTrainer

        ok("modeling_studio.trainers.multitask_trainer")
    except ImportError as e:
        fail(f"modeling_studio.trainers.multitask_trainer: {e}")
        imports_ok = False

    try:
        from modeling_studio.trainers.collators import MultiTaskCollator

        ok("modeling_studio.trainers.collators")
    except ImportError as e:
        fail(f"modeling_studio.trainers.collators: {e}")
        imports_ok = False

    return imports_ok


def validate_device() -> tuple[bool, str]:
    """Check GPU/device availability."""
    header("Step 2: Checking Device")

    import torch

    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        ok(f"CUDA available: {gpu_name} ({gpu_mem:.1f} GB)")
        return True, device
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        ok("MPS (Apple Silicon) available")
        return True, device
    else:
        device = "cpu"
        warn("No GPU available, using CPU (training will be slow)")
        return True, device


def validate_tokenizer(verbose: bool = False) -> tuple[bool, any]:
    """Test tokenizer loading."""
    header("Step 3: Loading Tokenizer")

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "answerdotai/ModernBERT-base",
            trust_remote_code=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        ok(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
        ok(f"Vocab size: {tokenizer.vocab_size:,}")

        # Quick test
        test_text = "Hello, this is a test sentence."
        encoded = tokenizer(test_text, return_tensors="pt")
        ok(f"Test encode: '{test_text}' -> {encoded['input_ids'].shape}")

        return True, tokenizer
    except Exception as e:
        fail(f"Tokenizer error: {e}")
        if verbose:
            traceback.print_exc()
        return False, None


def validate_model(verbose: bool = False) -> tuple[bool, any]:
    """Test model initialization."""
    header("Step 4: Initializing Model")

    try:
        from modeling_studio.data.labels import Capability
        from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

        # Only test with a subset of capabilities for speed
        capabilities = [
            Capability.SENTIMENT,
            Capability.NER_GENERAL,
            Capability.NLI,
            Capability.EMBEDDING,
        ]

        info(f"Capabilities: {[c.value for c in capabilities]}")

        model = ModernBertMultiTaskModel.from_pretrained(
            "answerdotai/ModernBERT-base",
            capabilities=capabilities,
            torch_dtype="float32",  # Use float32 for validation
            trust_remote_code=True,
        )

        num_params = sum(p.numel() for p in model.parameters())
        ok(f"Model initialized: {num_params:,} parameters")

        # List heads
        for cap in capabilities:
            head = model.get_head(cap)
            ok(f"  Head '{cap.value}': {head.__class__.__name__}")

        return True, model
    except Exception as e:
        fail(f"Model error: {e}")
        if verbose:
            traceback.print_exc()
        return False, None


def validate_single_dataset(
    task: str,
    tokenizer: any,
    verbose: bool = False,
    max_samples: int = 100,
) -> bool:
    """Validate a single dataset loads and tokenizes correctly."""
    subheader(f"Task: {task}")

    from modeling_studio.data.loaders import load_stage_a_datasets

    try:
        # Load raw dataset (small sample)
        info(f"Loading {task} dataset...")
        datasets = load_stage_a_datasets(
            split="train",
            tokenizer=tokenizer,
            apply_tokenization=True,
        )

        if task not in datasets:
            warn(f"Task '{task}' not in loaded datasets")
            return False

        ds = datasets[task]
        ok(f"Loaded {len(ds):,} samples")

        # Check columns
        columns = ds.column_names
        ok(f"Columns: {columns}")

        # Check required columns exist (task-specific)
        if task == "embedding":
            # Embedding uses anchor/positive format
            required = {
                "anchor_input_ids",
                "anchor_attention_mask",
                "positive_input_ids",
                "positive_attention_mask",
            }
        else:
            required = {"input_ids", "attention_mask"}

        if not required.issubset(set(columns)):
            fail(f"Missing required columns: {required - set(columns)}")
            return False

        # Sample a few examples
        sample = ds[0]
        ok(f"Sample keys: {list(sample.keys())}")

        if verbose:
            if "input_ids" in sample:
                info(f"Sample input_ids length: {len(sample['input_ids'])}")
            elif "anchor_input_ids" in sample:
                info(f"Sample anchor_input_ids length: {len(sample['anchor_input_ids'])}")
            if "labels" in sample:
                info(f"Sample labels: {sample['labels']}")

        return True

    except Exception as e:
        fail(f"Dataset error: {e}")
        if verbose:
            traceback.print_exc()
        return False


def validate_datasets(
    tokenizer: any, verbose: bool = False, single_task: str | None = None
) -> bool:
    """Validate all datasets load correctly."""
    header("Step 5: Validating Datasets")

    tasks = ["ner_general", "sentiment", "emotions", "safety_generic", "nli", "embedding"]

    if single_task:
        tasks = [single_task]

    all_ok = True
    for task in tasks:
        try:
            if not validate_single_dataset(task, tokenizer, verbose):
                all_ok = False
        except Exception as e:
            fail(f"{task}: {e}")
            all_ok = False

    return all_ok


def validate_collators(tokenizer: any, verbose: bool = False) -> bool:
    """Test collators with sample data."""
    header("Step 6: Testing Collators")

    try:
        from modeling_studio.trainers.collators import (
            EmbeddingCollator,
            MultiTaskCollator,
            SequenceClassificationCollator,
            TokenClassificationCollator,
        )

        # Sequence classification
        subheader("SequenceClassificationCollator")
        sc_collator = SequenceClassificationCollator(
            tokenizer=tokenizer,
            max_length=128,
        )

        samples = [
            {
                "input_ids": [1, 2, 3, 4],
                "attention_mask": [1, 1, 1, 1],
                "labels": 0,
                "task": "sentiment",
            },
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": 1, "task": "sentiment"},
        ]
        batch = sc_collator(samples)
        ok(f"Batch shape: input_ids={batch['input_ids'].shape}, labels={batch['labels'].shape}")

        # Token classification
        subheader("TokenClassificationCollator")
        tc_collator = TokenClassificationCollator(
            tokenizer=tokenizer,
            max_length=128,
        )

        samples = [
            {
                "input_ids": [1, 2, 3, 4],
                "attention_mask": [1, 1, 1, 1],
                "labels": [0, 1, 0, 0],
                "task": "ner_general",
            },
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "labels": [0, 1, 2],
                "task": "ner_general",
            },
        ]
        batch = tc_collator(samples)
        ok(f"Batch shape: input_ids={batch['input_ids'].shape}, labels={batch['labels'].shape}")

        # Embedding
        subheader("EmbeddingCollator")
        emb_collator = EmbeddingCollator(
            tokenizer=tokenizer,
            max_length=128,
        )

        samples = [
            {
                "anchor_input_ids": [1, 2, 3],
                "anchor_attention_mask": [1, 1, 1],
                "positive_input_ids": [4, 5, 6],
                "positive_attention_mask": [1, 1, 1],
                "task": "embedding",
            },
            {
                "anchor_input_ids": [7, 8],
                "anchor_attention_mask": [1, 1],
                "positive_input_ids": [9, 10],
                "positive_attention_mask": [1, 1],
                "task": "embedding",
            },
        ]
        batch = emb_collator(samples)
        ok(
            f"Batch shape: anchor={batch['anchor_input_ids'].shape}, positive={batch['positive_input_ids'].shape}"
        )

        # MultiTaskCollator
        subheader("MultiTaskCollator")
        _ = MultiTaskCollator(tokenizer=tokenizer)
        ok("MultiTaskCollator initialized")

        return True

    except Exception as e:
        fail(f"Collator error: {e}")
        if verbose:
            traceback.print_exc()
        return False


def validate_forward_pass(model: any, tokenizer: any, device: str, verbose: bool = False) -> bool:
    """Test forward pass for each task."""
    header("Step 7: Testing Forward Pass")

    import torch

    from modeling_studio.data.labels import Capability

    model = model.to(device)
    model.eval()

    all_ok = True

    # Test sentiment (sequence classification)
    subheader("Sentiment (Sequence Classification)")
    try:
        inputs = tokenizer("I love this!", return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(
                capability=Capability.SENTIMENT,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
        ok(f"Output logits shape: {outputs.logits.shape}")
    except Exception as e:
        fail(f"Forward pass error: {e}")
        if verbose:
            traceback.print_exc()
        all_ok = False

    # Test NER (token classification)
    subheader("NER (Token Classification)")
    try:
        inputs = tokenizer("John lives in New York.", return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(
                capability=Capability.NER_GENERAL,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
        ok(f"Output logits shape: {outputs.logits.shape}")
    except Exception as e:
        fail(f"Forward pass error: {e}")
        if verbose:
            traceback.print_exc()
        all_ok = False

    # Test NLI (sequence pair classification)
    subheader("NLI (Sequence Pair Classification)")
    try:
        inputs = tokenizer(
            "A man is playing guitar.",
            "A person is making music.",
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(
                capability=Capability.NLI,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
        ok(f"Output logits shape: {outputs.logits.shape}")
    except Exception as e:
        fail(f"Forward pass error: {e}")
        if verbose:
            traceback.print_exc()
        all_ok = False

    # Test Embedding
    subheader("Embedding")
    try:
        inputs = tokenizer("This is a test sentence.", return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(
                capability=Capability.EMBEDDING,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
        ok(f"Embedding shape: {outputs.logits.shape}")
    except Exception as e:
        fail(f"Forward pass error: {e}")
        if verbose:
            traceback.print_exc()
        all_ok = False

    return all_ok


def validate_training_step(model: any, tokenizer: any, device: str, verbose: bool = False) -> bool:
    """Test a single training step."""
    header("Step 8: Testing Training Step")

    import torch

    from modeling_studio.data.labels import Capability

    model = model.to(device)
    model.train()

    try:
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        # Sentiment task
        subheader("Training step: Sentiment")
        inputs = tokenizer("I love this movie!", return_tensors="pt").to(device)
        labels = torch.tensor([1], device=device)  # Positive

        outputs = model(
            capability=Capability.SENTIMENT,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels,
        )

        loss = outputs.loss
        ok(f"Loss: {loss.item():.4f}")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        ok("Backward pass successful")

        # NER task
        subheader("Training step: NER")
        inputs = tokenizer(
            ["John", "lives", "in", "Paris"],
            is_split_into_words=True,
            return_tensors="pt",
        ).to(device)
        # Create labels matching sequence length
        seq_len = inputs["input_ids"].shape[1]
        labels = torch.zeros(1, seq_len, dtype=torch.long, device=device)

        outputs = model(
            capability=Capability.NER_GENERAL,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels,
        )

        loss = outputs.loss
        ok(f"Loss: {loss.item():.4f}")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        ok("Backward pass successful")

        return True

    except Exception as e:
        fail(f"Training step error: {e}")
        if verbose:
            traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate training setup")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--task", type=str, help="Test single task only")
    parser.add_argument("--skip-datasets", action="store_true", help="Skip dataset validation")
    parser.add_argument("--skip-training", action="store_true", help="Skip training step test")
    args = parser.parse_args()

    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}    Training Setup Validation{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")

    results = {}

    # Step 1: Imports
    results["imports"] = validate_imports()
    if not results["imports"]:
        print(f"\n{Colors.RED}FAILED: Fix import errors before continuing{Colors.RESET}")
        return 1

    # Step 2: Device
    results["device"], device = validate_device()

    # Step 3: Tokenizer
    results["tokenizer"], tokenizer = validate_tokenizer(args.verbose)
    if not results["tokenizer"]:
        print(f"\n{Colors.RED}FAILED: Fix tokenizer errors before continuing{Colors.RESET}")
        return 1

    # Step 4: Model
    results["model"], model = validate_model(args.verbose)
    if not results["model"]:
        print(f"\n{Colors.RED}FAILED: Fix model errors before continuing{Colors.RESET}")
        return 1

    # Step 5: Datasets
    if not args.skip_datasets:
        results["datasets"] = validate_datasets(tokenizer, args.verbose, args.task)
    else:
        results["datasets"] = None
        warn("Skipping dataset validation")

    # Step 6: Collators
    results["collators"] = validate_collators(tokenizer, args.verbose)

    # Step 7: Forward pass
    results["forward"] = validate_forward_pass(model, tokenizer, device, args.verbose)

    # Step 8: Training step
    if not args.skip_training:
        results["training"] = validate_training_step(model, tokenizer, device, args.verbose)
    else:
        results["training"] = None
        warn("Skipping training step test")

    # Summary
    header("Summary")

    all_passed = True
    for name, result in results.items():
        if result is None:
            print(f"  {Colors.YELLOW}○{Colors.RESET} {name}: skipped")
        elif result:
            print(f"  {Colors.GREEN}✓{Colors.RESET} {name}: passed")
        else:
            print(f"  {Colors.RED}✗{Colors.RESET} {name}: FAILED")
            all_passed = False

    if all_passed:
        print(f"\n{Colors.GREEN}{Colors.BOLD}All validations passed! Ready to train.{Colors.RESET}")
        print("\nRun training with:")
        print(
            "  python scripts/train_stage_a.py --config configs/training/multitask/stage_a_generic.yaml --debug"
        )
        return 0
    else:
        print(
            f"\n{Colors.RED}{Colors.BOLD}Some validations failed. Fix errors before training.{Colors.RESET}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
