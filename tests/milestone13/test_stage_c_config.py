"""
Tests for Stage C configuration (Issue 13.1.1).

Tests the stage_c_decoder.yaml configuration file.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config_path():
    """Get path to Stage C config."""
    return Path("configs/training/multitask/stage_c_decoder.yaml")


@pytest.fixture
def config(config_path):
    """Load Stage C config."""
    if not config_path.exists():
        pytest.skip(f"Config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# =============================================================================
# Issue 13.1.1: Stage C Config File
# =============================================================================


class TestStageConfigLoads:
    """Tests that config loads without errors (13.1.1-T1)."""

    def test_config_loads(self, config):
        """13.1.1-T1: Config loads without errors."""
        assert config is not None
        assert isinstance(config, dict)

    def test_config_has_required_sections(self, config):
        """Config has all required sections."""
        required_sections = ["model", "decoder", "data", "training"]
        for section in required_sections:
            assert section in config, f"Missing section: {section}"


class TestStageConfigDecoderParams:
    """Tests decoder params match architecture (13.1.1-T2)."""

    def test_config_decoder_hidden_size(self, config):
        """13.1.1-T2: Decoder hidden_size is 1280."""
        decoder = config.get("decoder", {})
        assert decoder.get("hidden_size") == 1280

    def test_config_decoder_num_layers(self, config):
        """Decoder num_layers is 8."""
        decoder = config.get("decoder", {})
        assert decoder.get("num_layers") == 8

    def test_config_decoder_num_experts(self, config):
        """Decoder num_experts is 8."""
        decoder = config.get("decoder", {})
        assert decoder.get("num_experts") == 8

    def test_config_decoder_top_k(self, config):
        """Decoder num_experts_per_token is 2 (top-2)."""
        decoder = config.get("decoder", {})
        assert decoder.get("num_experts_per_token") == 2

    def test_config_decoder_shared_expert(self, config):
        """Decoder use_shared_expert is True."""
        decoder = config.get("decoder", {})
        assert decoder.get("use_shared_expert") is True

    def test_config_decoder_attention_heads(self, config):
        """Decoder attention config matches architecture."""
        decoder = config.get("decoder", {})
        assert decoder.get("num_attention_heads") == 20
        assert decoder.get("num_kv_heads") == 4

    def test_config_decoder_moe_loss_weights(self, config):
        """Decoder MoE loss weights are configured."""
        decoder = config.get("decoder", {})
        assert decoder.get("load_balancing_loss_weight") == 0.01
        assert decoder.get("router_z_loss_weight") == 0.001


class TestStageConfigFreezingFlags:
    """Tests freezing flags are correct (13.1.1-T3)."""

    def test_config_freeze_encoder(self, config):
        """13.1.1-T3: freeze_encoder is True."""
        model = config.get("model", {})
        assert model.get("freeze_encoder") is True

    def test_config_freeze_existing_heads(self, config):
        """freeze_existing_heads is True."""
        model = config.get("model", {})
        assert model.get("freeze_existing_heads") is True


class TestStageConfigTraining:
    """Tests training hyperparams are configured."""

    def test_config_training_epochs(self, config):
        """Training epochs is configured."""
        training = config.get("training", {})
        assert training.get("num_train_epochs") == 10

    def test_config_training_batch_size(self, config):
        """Batch size is configured for Colab."""
        training = config.get("training", {})
        assert training.get("per_device_train_batch_size") == 8
        assert training.get("gradient_accumulation_steps") == 4

    def test_config_training_learning_rate(self, config):
        """Learning rate is configured."""
        training = config.get("training", {})
        lr = training.get("learning_rate")
        # Handle both string and float
        if isinstance(lr, str):
            lr = float(lr)
        assert lr == 2e-4 or lr == 0.0002

    def test_config_save_steps_for_colab(self, config):
        """Save steps is frequent enough for Colab (18 hour limit)."""
        training = config.get("training", {})
        save_steps = training.get("save_steps", 1000)
        # Should save at least every 500 steps for Colab safety
        assert save_steps <= 500, "save_steps should be <= 500 for Colab"

    def test_config_gradient_checkpointing(self, config):
        """Gradient checkpointing enabled for memory."""
        training = config.get("training", {})
        assert training.get("gradient_checkpointing") is True


class TestStageConfigData:
    """Tests data configuration."""

    def test_config_data_path(self, config):
        """Data path is configured."""
        data = config.get("data", {})
        assert data.get("train_path") is not None

    def test_config_embeddings_mode(self, config):
        """Embeddings mode is precomputed."""
        data = config.get("data", {})
        assert data.get("embeddings_mode") == "precomputed"

    def test_config_sequence_lengths(self, config):
        """Sequence lengths are configured."""
        data = config.get("data", {})
        assert data.get("max_input_length", 256) <= 512
        assert data.get("max_output_length", 256) <= 512
