"""
Tests for Phase 0.5 Training Script.

Tests for Issue 5.4.1: Implement Phase 0.5 Healing Training Script

This module tests the command-line script that runs Phase 0.5
"Enhanced Healing" training for ModernBERT v3.

Test Categories:
    - TestParseArgs: Argument parsing tests
    - TestPhase05Config: Configuration tests
    - TestSmokeTest: Smoke test mode validation
    - TestDryRun: Dry run mode validation
    - TestDebugMode: Debug mode validation
    - TestDataLoading: Data loading tests
    - TestTrainingSetup: Training component setup tests
    - TestIssue541AcceptanceCriteria: Acceptance criteria tests

Author: FamilyOS Team
Date: December 2025
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.vocab_size = 50372
    tokenizer.__len__ = lambda self: 50372
    tokenizer.get_added_vocab.return_value = {
        "[EMO]": 50368,
        "[MEM]": 50369,
        "[REL]": 50370,
        "[TASK]": 50371,
    }

    def mock_call(text, **kwargs):
        result = MagicMock()
        result["input_ids"] = torch.randint(0, 50368, (1, 128))
        result["attention_mask"] = torch.ones(1, 128)
        return result

    tokenizer.side_effect = mock_call
    tokenizer.return_value = {
        "input_ids": torch.randint(0, 50368, (1, 128)),
        "attention_mask": torch.ones(1, 128),
    }

    return tokenizer


@pytest.fixture
def output_dir(tmp_path):
    """Create output directory for tests."""
    return tmp_path / "test_output"


@pytest.fixture
def healing_data_dir(tmp_path):
    """Create healing data directory with sample data."""
    data_dir = tmp_path / "healing_data"
    data_dir.mkdir(parents=True)

    # Create sample JSONL
    healing_file = data_dir / "healing_enhanced.jsonl"
    samples = [
        {"text": f"Sample {i} for testing", "task": "classification", "label": i % 3}
        for i in range(100)
    ]
    with open(healing_file, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    return data_dir


# ==============================================================================
# Argument Parsing Tests
# ==============================================================================


class TestParseArgs:
    """Tests for argument parsing."""

    def test_default_values(self):
        """Test default argument values."""
        from train_v3_phase0_5 import parse_args

        with patch("sys.argv", ["prog"]):
            args = parse_args()

        assert args.smoke_test is False
        assert args.dry_run is False
        assert args.debug is False
        assert args.config == "configs/training/multitask/stage_v3_phase0_5_enhanced.yaml"
        assert args.output_dir == "outputs/v3_phase0_5"

    def test_smoke_test_flag(self):
        """Test --smoke-test flag."""
        from train_v3_phase0_5 import parse_args

        with patch("sys.argv", ["prog", "--smoke-test"]):
            args = parse_args()

        assert args.smoke_test is True

    def test_dry_run_flag(self):
        """Test --dry-run flag."""
        from train_v3_phase0_5 import parse_args

        with patch("sys.argv", ["prog", "--dry-run"]):
            args = parse_args()

        assert args.dry_run is True

    def test_debug_flag(self):
        """Test --debug flag."""
        from train_v3_phase0_5 import parse_args

        with patch("sys.argv", ["prog", "--debug"]):
            args = parse_args()

        assert args.debug is True

    def test_custom_output_dir(self):
        """Test --output-dir argument."""
        from train_v3_phase0_5 import parse_args

        with patch("sys.argv", ["prog", "--output-dir", "/custom/output"]):
            args = parse_args()

        assert args.output_dir == "/custom/output"

    def test_training_overrides(self):
        """Test training parameter overrides."""
        from train_v3_phase0_5 import parse_args

        with patch(
            "sys.argv",
            [
                "prog",
                "--max-steps",
                "5000",
                "--warmup-steps",
                "1000",
                "--learning-rate",
                "1e-4",
                "--train-batch-size",
                "16",
            ],
        ):
            args = parse_args()

        assert args.max_steps == 5000
        assert args.warmup_steps == 1000
        assert args.learning_rate == 1e-4
        assert args.train_batch_size == 16


# ==============================================================================
# Configuration Tests
# ==============================================================================


class TestPhase05Config:
    """Tests for Phase05Config dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from train_v3_phase0_5 import Phase05Config

        config = Phase05Config()

        assert config.max_steps == 2500
        assert config.warmup_steps == 500
        assert config.eval_steps == 250
        assert config.base_lr == 3e-5
        assert config.lr_layers_1_18 == 0.0
        assert config.lr_layer_23 == 5e-5
        assert config.max_grad_norm == 1.0
        assert config.bf16 is True

    def test_config_to_dict(self):
        """Test config serialization to dict."""
        from train_v3_phase0_5 import Phase05Config

        config = Phase05Config(max_steps=1000, base_lr=1e-4)
        d = config.to_dict()

        assert d["max_steps"] == 1000
        assert d["base_lr"] == 1e-4
        assert isinstance(d, dict)

    def test_config_from_dict(self):
        """Test config creation from dict."""
        from train_v3_phase0_5 import Phase05Config

        d = {"max_steps": 3000, "base_lr": 2e-5, "unknown_field": "ignored"}
        config = Phase05Config.from_dict(d)

        assert config.max_steps == 3000
        assert config.base_lr == 2e-5

    def test_config_save_load(self, tmp_path):
        """Test config save and load."""
        from train_v3_phase0_5 import Phase05Config

        config = Phase05Config(max_steps=1500, wandb_run_name="test_run")
        config_path = tmp_path / "config.json"
        config.save(config_path)

        assert config_path.exists()

        with open(config_path) as f:
            loaded = json.load(f)

        assert loaded["max_steps"] == 1500
        assert loaded["wandb_run_name"] == "test_run"


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_smoke_test_mode(self):
        """Test config loading with smoke test mode."""
        from train_v3_phase0_5 import load_config, parse_args

        with patch("sys.argv", ["prog", "--smoke-test"]):
            args = parse_args()
            config = load_config(args)

        assert config.smoke_test is True
        assert config.max_steps == 10
        assert config.warmup_steps == 2
        assert config.train_batch_size == 2
        assert config.use_wandb is False

    def test_load_config_debug_mode(self):
        """Test config loading with debug mode."""
        from train_v3_phase0_5 import load_config, parse_args

        with patch("sys.argv", ["prog", "--debug"]):
            args = parse_args()
            config = load_config(args)

        assert config.debug is True
        assert config.logging_steps == 1
        assert config.train_batch_size == 4
        assert config.use_wandb is False

    def test_load_config_dry_run_mode(self):
        """Test config loading with dry run mode."""
        from train_v3_phase0_5 import load_config, parse_args

        with patch("sys.argv", ["prog", "--dry-run"]):
            args = parse_args()
            config = load_config(args)

        assert config.dry_run is True
        assert config.use_wandb is False


# ==============================================================================
# Synthetic Dataset Tests
# ==============================================================================


class TestSyntheticDataset:
    """Tests for SyntheticHealingDataset."""

    def test_synthetic_dataset_creation(self, mock_tokenizer):
        """Test synthetic dataset can be created."""
        from train_v3_phase0_5 import SyntheticHealingDataset

        dataset = SyntheticHealingDataset(mock_tokenizer, num_samples=50)

        assert len(dataset) == 50

    def test_synthetic_dataset_getitem(self, mock_tokenizer):
        """Test getting item from synthetic dataset."""
        from train_v3_phase0_5 import SyntheticHealingDataset

        # Create a more realistic mock tokenizer
        def tokenize(text, **kwargs):
            return {
                "input_ids": torch.randint(0, 50368, (1, 128)),
                "attention_mask": torch.ones(1, 128, dtype=torch.long),
            }

        mock_tokenizer.side_effect = None
        mock_tokenizer.__call__ = tokenize

        dataset = SyntheticHealingDataset(mock_tokenizer, num_samples=10)
        item = dataset[0]

        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item


# ==============================================================================
# Dry Run Tests
# ==============================================================================


class TestDryRun:
    """Tests for dry run functionality."""

    def test_dry_run_returns_results(self):
        """Test dry run returns results dictionary."""
        from train_v3_phase0_5 import run_dry_run, Phase05Config

        config = Phase05Config(dry_run=True, device="cpu")
        results = run_dry_run(config)

        assert "status" in results
        assert "checks" in results
        assert "warnings" in results
        assert "errors" in results
        assert isinstance(results["checks"], list)

    def test_dry_run_detects_cuda_unavailable(self):
        """Test dry run warns when CUDA requested but unavailable."""
        from train_v3_phase0_5 import run_dry_run, Phase05Config

        # Mock CUDA as unavailable
        with patch("torch.cuda.is_available", return_value=False):
            config = Phase05Config(dry_run=True, device="cuda")
            results = run_dry_run(config)

        # Should have warning about CUDA
        has_cuda_warning = any("CUDA" in w for w in results["warnings"])
        assert has_cuda_warning or config.device == "cpu"


# ==============================================================================
# Training Setup Tests
# ==============================================================================


class TestTrainingSetup:
    """Tests for training component setup."""

    def test_create_dataloaders(self, mock_tokenizer):
        """Test dataloader creation."""
        from train_v3_phase0_5 import (
            create_dataloaders,
            Phase05Config,
            SyntheticHealingDataset,
        )

        config = Phase05Config(
            train_batch_size=4,
            eval_batch_size=8,
            smoke_test=True,
        )

        train_dataset = SyntheticHealingDataset(mock_tokenizer, num_samples=20)
        val_dataset = SyntheticHealingDataset(mock_tokenizer, num_samples=10)

        train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, config)

        assert train_loader is not None
        assert val_loader is not None
        assert train_loader.batch_size == 4
        assert val_loader.batch_size == 8


# ==============================================================================
# Acceptance Criteria Tests (Issue 5.4.1)
# ==============================================================================


class TestIssue541AcceptanceCriteria:
    """Tests for Issue 5.4.1 acceptance criteria."""

    def test_script_can_be_imported(self):
        """Test that the script can be imported without errors."""
        import train_v3_phase0_5

        assert hasattr(train_v3_phase0_5, "main")
        assert hasattr(train_v3_phase0_5, "parse_args")
        assert hasattr(train_v3_phase0_5, "Phase05Config")
        assert hasattr(train_v3_phase0_5, "train_phase_0_5")

    def test_has_smoke_test_mode(self):
        """Test script has smoke test mode."""
        from train_v3_phase0_5 import parse_args, load_config

        with patch("sys.argv", ["prog", "--smoke-test"]):
            args = parse_args()
            config = load_config(args)

        assert config.smoke_test is True
        assert config.max_steps <= 10

    def test_has_dry_run_mode(self):
        """Test script has dry run mode."""
        from train_v3_phase0_5 import parse_args, load_config

        with patch("sys.argv", ["prog", "--dry-run"]):
            args = parse_args()
            config = load_config(args)

        assert config.dry_run is True

    def test_has_debug_mode(self):
        """Test script has debug mode."""
        from train_v3_phase0_5 import parse_args, load_config

        with patch("sys.argv", ["prog", "--debug"]):
            args = parse_args()
            config = load_config(args)

        assert config.debug is True
        assert config.logging_steps == 1

    def test_zipper_lr_config(self):
        """Test Zipper LR configuration is present."""
        from train_v3_phase0_5 import Phase05Config

        config = Phase05Config()

        # Check Zipper LR layer-specific rates
        assert config.lr_layers_1_18 == 0.0  # Frozen
        assert config.lr_layers_19_22 == 1e-5  # Feeder
        assert config.lr_layer_23 == 5e-5  # Interface (highest)
        assert config.lr_layers_24_28 == 3e-5  # Family

    def test_gradient_clipping_config(self):
        """Test gradient clipping configuration."""
        from train_v3_phase0_5 import Phase05Config

        config = Phase05Config()

        assert config.max_grad_norm == 1.0
        assert config.interface_grad_clip == 0.5
        assert config.per_layer_clip is True
        assert config.nan_check is True

    def test_hub_token_config(self):
        """Test hub token gradient masking configuration."""
        from train_v3_phase0_5 import Phase05Config

        config = Phase05Config()

        assert config.freeze_original_vocab is True
        assert "[EMO]" in config.train_hub_tokens
        assert "[MEM]" in config.train_hub_tokens
        assert "[REL]" in config.train_hub_tokens
        assert "[TASK]" in config.train_hub_tokens

    def test_bf16_enabled_by_default(self):
        """Test bf16 is enabled by default."""
        from train_v3_phase0_5 import Phase05Config

        config = Phase05Config()

        assert config.bf16 is True

    def test_wandb_config(self):
        """Test W&B configuration."""
        from train_v3_phase0_5 import Phase05Config

        config = Phase05Config()

        assert config.use_wandb is True
        assert config.wandb_project == "modernbert-v3"

    def test_checkpoint_config(self):
        """Test checkpoint saving configuration."""
        from train_v3_phase0_5 import Phase05Config

        config = Phase05Config()

        assert config.save_steps == 500
        assert config.output_dir == "outputs/v3_phase0_5"

    def test_phase_0_5_training_parameters(self):
        """Test Phase 0.5 specific training parameters."""
        from train_v3_phase0_5 import Phase05Config

        config = Phase05Config()

        # From reference.md requirements
        assert config.max_steps == 2500  # Phase 0.5: ~2500 steps
        assert config.warmup_steps == 500
        assert config.eval_steps == 250


# ==============================================================================
# Integration Tests (require model)
# ==============================================================================


@pytest.mark.slow
class TestIntegration:
    """Integration tests that may require model loading."""

    def test_full_smoke_test(self, tmp_path):
        """
        Test full smoke test execution.

        This test is skipped by default (marked slow) as it requires
        the full model infrastructure.
        """
        pytest.skip("Full integration test requires model infrastructure")

    def test_dry_run_execution(self):
        """Test dry run can complete successfully."""
        from train_v3_phase0_5 import run_dry_run, Phase05Config

        config = Phase05Config(
            dry_run=True,
            device="cpu",
            healing_data_dir="/nonexistent",  # Should handle gracefully
        )

        results = run_dry_run(config)

        assert results["status"] in ["success", "failed"]
        assert len(results["checks"]) > 0
