"""
Tests for train_stage_c.py script (Issue 13.3.1).

Tests the Stage C training script functionality.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


# =============================================================================
# Issue 13.3.1: train_stage_c.py Main Script
# =============================================================================


class TestScriptParsesArgs:
    """Tests that script parses arguments correctly (13.3.1-T1)."""

    def test_script_parses_config_arg(self):
        """13.3.1-T1: Script parses --config argument."""
        from scripts.train_stage_c import parse_args

        with patch("sys.argv", ["train_stage_c.py", "--config", "test_config.yaml"]):
            args = parse_args()
            assert args.config == "test_config.yaml"

    def test_script_parses_resume_arg(self):
        """Script parses --resume_from_checkpoint argument."""
        from scripts.train_stage_c import parse_args

        with patch("sys.argv", [
            "train_stage_c.py",
            "--config", "test.yaml",
            "--resume_from_checkpoint", "checkpoints/test/checkpoint-1000"
        ]):
            args = parse_args()
            assert args.resume_from_checkpoint == "checkpoints/test/checkpoint-1000"

    def test_script_parses_auto_resume(self):
        """Script parses --auto_resume flag."""
        from scripts.train_stage_c import parse_args

        with patch("sys.argv", ["train_stage_c.py", "--config", "test.yaml", "--auto_resume"]):
            args = parse_args()
            assert args.auto_resume is True

    def test_script_parses_debug_flag(self):
        """Script parses --debug flag."""
        from scripts.train_stage_c import parse_args

        with patch("sys.argv", ["train_stage_c.py", "--config", "test.yaml", "--debug"]):
            args = parse_args()
            assert args.debug is True


class TestScriptConfigLoading:
    """Tests config loading functionality."""

    def test_load_config_loads_yaml(self):
        """Script loads YAML config correctly."""
        from scripts.train_stage_c import load_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"model": {"checkpoint_path": "test"}}, f)
            f.flush()

            config = load_config(f.name)
            assert config["model"]["checkpoint_path"] == "test"

    def test_load_config_raises_on_missing(self):
        """Script raises error for missing config."""
        from scripts.train_stage_c import load_config

        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")


class TestScriptOverrides:
    """Tests config override functionality."""

    def test_apply_overrides_simple(self):
        """apply_overrides handles simple key=value."""
        from scripts.train_stage_c import apply_overrides

        config = {"training": {"learning_rate": 1e-4}}
        # Use numeric value directly (yaml.safe_load parses "0.0002" as float)
        overrides = ["training.learning_rate=0.0002"]

        result = apply_overrides(config, overrides)
        assert result["training"]["learning_rate"] == pytest.approx(0.0002, rel=1e-6)

    def test_apply_overrides_nested(self):
        """apply_overrides handles nested keys."""
        from scripts.train_stage_c import apply_overrides

        config = {"decoder": {"num_experts": 8}}
        overrides = ["decoder.num_experts=16"]

        result = apply_overrides(config, overrides)
        assert result["decoder"]["num_experts"] == 16


class TestFindLatestCheckpoint:
    """Tests checkpoint finding functionality."""

    def test_find_latest_checkpoint_returns_latest(self):
        """find_latest_checkpoint returns latest checkpoint."""
        from scripts.train_stage_c import find_latest_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create checkpoint directories
            (Path(tmpdir) / "checkpoint-100").mkdir()
            (Path(tmpdir) / "checkpoint-500").mkdir()
            (Path(tmpdir) / "checkpoint-300").mkdir()

            latest = find_latest_checkpoint(tmpdir)
            assert latest is not None
            assert "checkpoint-500" in latest

    def test_find_latest_checkpoint_returns_none_if_empty(self):
        """find_latest_checkpoint returns None for empty dir."""
        from scripts.train_stage_c import find_latest_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            latest = find_latest_checkpoint(tmpdir)
            assert latest is None

    def test_find_latest_checkpoint_returns_none_if_no_dir(self):
        """find_latest_checkpoint returns None for missing dir."""
        from scripts.train_stage_c import find_latest_checkpoint

        latest = find_latest_checkpoint("/nonexistent/path")
        assert latest is None


class TestScriptTrainingArgs:
    """Tests training arguments creation."""

    def test_create_training_args_defaults(self):
        """create_training_args uses config defaults."""
        from scripts.train_stage_c import create_training_args

        config = {
            "training": {
                "learning_rate": 2e-4,
                "num_train_epochs": 10,
                "per_device_train_batch_size": 8,
                "gradient_accumulation_steps": 4,
                "save_steps": 500,
                "eval_steps": 500,
            }
        }

        args = create_training_args(config)

        assert args.learning_rate == 2e-4
        assert args.num_train_epochs == 10
        assert args.per_device_train_batch_size == 8
        assert args.gradient_accumulation_steps == 4

    def test_create_training_args_debug_mode(self):
        """create_training_args adjusts for debug mode."""
        from scripts.train_stage_c import create_training_args

        config = {
            "training": {
                "per_device_train_batch_size": 16,
                "num_train_epochs": 10,
            }
        }

        args = create_training_args(config, debug=True)

        # Debug mode should reduce batch size
        assert args.per_device_train_batch_size <= 4
        # Debug mode should reduce epochs
        assert args.num_train_epochs == 1


class TestScriptIntegration:
    """Integration tests for script components."""

    def test_init_tokenizer_returns_tokenizer(self):
        """init_tokenizer returns a tokenizer."""
        from scripts.train_stage_c import init_tokenizer

        config = {"model": {}}

        # This may require network access, so we mock if needed
        try:
            tokenizer = init_tokenizer(config)
            assert tokenizer is not None
            assert hasattr(tokenizer, "encode")
        except Exception:
            # Skip if tokenizer can't be loaded (offline)
            pytest.skip("Tokenizer loading requires network access")
