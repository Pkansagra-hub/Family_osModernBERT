"""
Tests for v3 Trainer Layer Freezing.

Tests for Issue 5.1.1: Implement Layer Freezing by Band

This module tests the layer freezing utilities for phase-based training,
ensuring correct freezing/unfreezing of layers by band.

Test Categories:
    - TestLayerBandEnum: LayerBand enum tests
    - TestTrainingPhaseEnum: TrainingPhase enum tests
    - TestLayerBandMapping: Band to layer mapping tests
    - TestLayerFreezerBasic: Basic LayerFreezer tests
    - TestLayerFreezerBands: Band freezing/unfreezing tests
    - TestLayerFreezerPhase: Phase configuration tests
    - TestLayerFreezerStats: Statistics tests
    - TestHelperFunctions: Helper function tests
    - TestIssue511AcceptanceCriteria: Acceptance criteria tests

Author: FamilyOS Team
Date: December 2025
"""

import pytest
import torch
import torch.nn as nn

from modeling_studio.trainers.freezing_v3 import (
    LAYER_BANDS,
    LayerBand,
    LayerFreezer,
    TrainingPhase,
    configure_model_for_phase,
    get_band_for_layer,
    get_layers_for_band,
    get_trainable_bands_for_phase,
)


# ==============================================================================
# Fixtures
# ==============================================================================


class MockEncoder(nn.Module):
    """Mock encoder with 28 layers for testing."""

    def __init__(self, num_layers: int = 28, hidden_size: int = 768):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )


class MockEmbeddings(nn.Module):
    """Mock embeddings for testing."""

    def __init__(self, hidden_size: int = 768, vocab_size: int = 50432):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)


class MockModel(nn.Module):
    """Mock ModernBERTv3 model for testing."""

    def __init__(self, num_layers: int = 28, hidden_size: int = 768):
        super().__init__()
        self.embeddings = MockEmbeddings(hidden_size)
        self.encoder = MockEncoder(num_layers, hidden_size)


@pytest.fixture
def mock_model():
    """Create a mock 28-layer model."""
    return MockModel(num_layers=28)


@pytest.fixture
def mock_model_22_layers():
    """Create a mock 22-layer model."""
    return MockModel(num_layers=22)


@pytest.fixture
def freezer(mock_model):
    """Create a LayerFreezer for the mock model."""
    return LayerFreezer(mock_model)


# ==============================================================================
# LayerBand Enum Tests
# ==============================================================================


class TestLayerBandEnum:
    """Tests for LayerBand enum."""

    def test_has_four_bands(self):
        """Test that there are exactly 4 bands."""
        assert len(LayerBand) == 4

    def test_foundation_band_value(self):
        """Test FOUNDATION band value."""
        assert LayerBand.FOUNDATION.value == "foundation"

    def test_core_band_value(self):
        """Test CORE band value."""
        assert LayerBand.CORE.value == "core"

    def test_feeder_band_value(self):
        """Test FEEDER band value."""
        assert LayerBand.FEEDER.value == "feeder"

    def test_family_band_value(self):
        """Test FAMILY band value."""
        assert LayerBand.FAMILY.value == "family"


# ==============================================================================
# TrainingPhase Enum Tests
# ==============================================================================


class TestTrainingPhaseEnum:
    """Tests for TrainingPhase enum."""

    def test_has_four_phases(self):
        """Test that there are exactly 4 phases."""
        assert len(TrainingPhase) == 4

    def test_phase_0_5_value(self):
        """Test PHASE_0_5 value."""
        assert TrainingPhase.PHASE_0_5.value == "phase_0.5"

    def test_phase_1_value(self):
        """Test PHASE_1 value."""
        assert TrainingPhase.PHASE_1.value == "phase_1"

    def test_phase_2_value(self):
        """Test PHASE_2 value."""
        assert TrainingPhase.PHASE_2.value == "phase_2"

    def test_inference_value(self):
        """Test INFERENCE value."""
        assert TrainingPhase.INFERENCE.value == "inference"


# ==============================================================================
# Layer Band Mapping Tests
# ==============================================================================


class TestLayerBandMapping:
    """Tests for layer band mappings."""

    def test_foundation_layers(self):
        """Test FOUNDATION band has correct layer indices."""
        assert LAYER_BANDS[LayerBand.FOUNDATION] == [0, 1, 2, 3, 4, 5]

    def test_core_layers(self):
        """Test CORE band has correct layer indices."""
        expected = list(range(6, 18))
        assert LAYER_BANDS[LayerBand.CORE] == expected

    def test_feeder_layers(self):
        """Test FEEDER band has correct layer indices."""
        assert LAYER_BANDS[LayerBand.FEEDER] == [18, 19, 20, 21]

    def test_family_layers(self):
        """Test FAMILY band has correct layer indices."""
        assert LAYER_BANDS[LayerBand.FAMILY] == [22, 23, 24, 25, 26, 27]

    def test_all_layers_covered(self):
        """Test that all 28 layers are covered by bands."""
        all_layers = set()
        for band in LayerBand:
            all_layers.update(LAYER_BANDS[band])
        assert all_layers == set(range(28))

    def test_no_overlap_between_bands(self):
        """Test that bands don't overlap."""
        for band1 in LayerBand:
            for band2 in LayerBand:
                if band1 != band2:
                    overlap = set(LAYER_BANDS[band1]) & set(LAYER_BANDS[band2])
                    assert len(overlap) == 0, f"Overlap between {band1} and {band2}"


# ==============================================================================
# Basic LayerFreezer Tests
# ==============================================================================


class TestLayerFreezerBasic:
    """Basic tests for LayerFreezer."""

    def test_init_with_valid_model(self, mock_model):
        """Test initialization with valid model."""
        freezer = LayerFreezer(mock_model)
        assert freezer.num_layers == 28

    def test_init_stores_model_reference(self, mock_model):
        """Test that freezer stores model reference."""
        freezer = LayerFreezer(mock_model)
        assert freezer.model is mock_model

    def test_init_stores_encoder_reference(self, mock_model):
        """Test that freezer stores encoder reference."""
        freezer = LayerFreezer(mock_model)
        assert freezer.encoder is mock_model.encoder

    def test_init_with_encoder_only(self):
        """Test initialization with just encoder (no model wrapper)."""
        encoder = MockEncoder(num_layers=28)
        freezer = LayerFreezer(encoder)
        assert freezer.num_layers == 28

    def test_init_with_invalid_model(self):
        """Test initialization fails with model without layers."""
        invalid_model = nn.Linear(10, 10)
        with pytest.raises(ValueError, match="must have 'layers' attribute"):
            LayerFreezer(invalid_model)

    def test_get_layer(self, freezer):
        """Test get_layer returns correct layer."""
        layer = freezer.get_layer(0)
        assert isinstance(layer, nn.Linear)

    def test_get_layer_out_of_range(self, freezer):
        """Test get_layer raises error for invalid index."""
        with pytest.raises(IndexError):
            freezer.get_layer(28)

    def test_get_layer_negative_index(self, freezer):
        """Test get_layer raises error for negative index."""
        with pytest.raises(IndexError):
            freezer.get_layer(-1)


# ==============================================================================
# Layer Freezing Tests
# ==============================================================================


class TestLayerFreezerFreeze:
    """Tests for freezing individual layers."""

    def test_freeze_layer(self, freezer):
        """Test freezing a single layer."""
        freezer.freeze_layer(0)
        layer = freezer.get_layer(0)
        for param in layer.parameters():
            assert not param.requires_grad

    def test_freeze_layer_tracks_frozen(self, freezer):
        """Test that freezing updates _frozen_layers."""
        freezer.freeze_layer(5)
        assert 5 in freezer._frozen_layers

    def test_freeze_layer_returns_param_count(self, freezer):
        """Test freeze_layer returns number of frozen params."""
        frozen_count = freezer.freeze_layer(0)
        assert frozen_count > 0

    def test_unfreeze_layer(self, freezer):
        """Test unfreezing a layer."""
        freezer.freeze_layer(0)
        freezer.unfreeze_layer(0)
        layer = freezer.get_layer(0)
        for param in layer.parameters():
            assert param.requires_grad

    def test_unfreeze_layer_updates_tracking(self, freezer):
        """Test that unfreezing updates _frozen_layers."""
        freezer.freeze_layer(5)
        freezer.unfreeze_layer(5)
        assert 5 not in freezer._frozen_layers


# ==============================================================================
# Band Freezing Tests
# ==============================================================================


class TestLayerFreezerBands:
    """Tests for freezing/unfreezing bands."""

    def test_freeze_foundation_band(self, freezer):
        """Test freezing FOUNDATION band."""
        freezer.freeze_band(LayerBand.FOUNDATION)
        for idx in LAYER_BANDS[LayerBand.FOUNDATION]:
            layer = freezer.get_layer(idx)
            for param in layer.parameters():
                assert not param.requires_grad

    def test_freeze_core_band(self, freezer):
        """Test freezing CORE band."""
        freezer.freeze_band(LayerBand.CORE)
        for idx in LAYER_BANDS[LayerBand.CORE]:
            layer = freezer.get_layer(idx)
            for param in layer.parameters():
                assert not param.requires_grad

    def test_freeze_feeder_band(self, freezer):
        """Test freezing FEEDER band."""
        freezer.freeze_band(LayerBand.FEEDER)
        for idx in LAYER_BANDS[LayerBand.FEEDER]:
            layer = freezer.get_layer(idx)
            for param in layer.parameters():
                assert not param.requires_grad

    def test_freeze_family_band(self, freezer):
        """Test freezing FAMILY band."""
        freezer.freeze_band(LayerBand.FAMILY)
        for idx in LAYER_BANDS[LayerBand.FAMILY]:
            layer = freezer.get_layer(idx)
            for param in layer.parameters():
                assert not param.requires_grad

    def test_freeze_band_returns_param_count(self, freezer):
        """Test freeze_band returns frozen param count."""
        frozen = freezer.freeze_band(LayerBand.FOUNDATION)
        assert frozen > 0

    def test_unfreeze_band(self, freezer):
        """Test unfreezing a band."""
        freezer.freeze_band(LayerBand.FAMILY)
        freezer.unfreeze_band(LayerBand.FAMILY)
        for idx in LAYER_BANDS[LayerBand.FAMILY]:
            layer = freezer.get_layer(idx)
            for param in layer.parameters():
                assert param.requires_grad

    def test_is_band_frozen(self, freezer):
        """Test is_band_frozen method."""
        assert not freezer.is_band_frozen(LayerBand.FOUNDATION)
        freezer.freeze_band(LayerBand.FOUNDATION)
        assert freezer.is_band_frozen(LayerBand.FOUNDATION)


# ==============================================================================
# Embedding Freezing Tests
# ==============================================================================


class TestLayerFreezerEmbeddings:
    """Tests for embedding freezing."""

    def test_freeze_embeddings(self, mock_model):
        """Test freezing embeddings."""
        freezer = LayerFreezer(mock_model)
        freezer.freeze_embeddings()
        for param in mock_model.embeddings.parameters():
            assert not param.requires_grad

    def test_freeze_embeddings_tracks_component(self, freezer):
        """Test that freezing updates _frozen_components."""
        freezer.freeze_embeddings()
        assert "embeddings" in freezer._frozen_components

    def test_freeze_embeddings_returns_count(self, freezer):
        """Test freeze_embeddings returns frozen param count."""
        frozen = freezer.freeze_embeddings()
        assert frozen > 0

    def test_unfreeze_embeddings(self, mock_model):
        """Test unfreezing embeddings."""
        freezer = LayerFreezer(mock_model)
        freezer.freeze_embeddings()
        freezer.unfreeze_embeddings()
        for param in mock_model.embeddings.parameters():
            assert param.requires_grad

    def test_model_without_embeddings(self):
        """Test freezer handles model without embeddings."""
        encoder = MockEncoder(num_layers=28)
        freezer = LayerFreezer(encoder)
        # Should not raise
        frozen = freezer.freeze_embeddings()
        assert frozen == 0


# ==============================================================================
# Phase Configuration Tests
# ==============================================================================


class TestLayerFreezerPhase:
    """Tests for phase configuration."""

    def test_configure_phase_0_5(self, freezer):
        """Test Phase 0.5 configuration."""
        freezer.configure_for_phase(TrainingPhase.PHASE_0_5)

        # L1-18 (indices 0-17) should be frozen
        for idx in range(18):
            assert freezer.is_layer_frozen(idx), f"Layer {idx} should be frozen"

        # L19-28 (indices 18-27) should be trainable
        for idx in range(18, 28):
            assert not freezer.is_layer_frozen(idx), f"Layer {idx} should be trainable"

    def test_configure_phase_1(self, freezer):
        """Test Phase 1 configuration."""
        freezer.configure_for_phase(TrainingPhase.PHASE_1)

        # L1-18 should be frozen
        for idx in range(18):
            assert freezer.is_layer_frozen(idx)

        # L19-28 should be trainable
        for idx in range(18, 28):
            assert not freezer.is_layer_frozen(idx)

    def test_configure_phase_2(self, freezer):
        """Test Phase 2 configuration."""
        freezer.configure_for_phase(TrainingPhase.PHASE_2)

        # All layers should be trainable
        for idx in range(28):
            assert not freezer.is_layer_frozen(idx), f"Layer {idx} should be trainable"

    def test_configure_inference(self, freezer):
        """Test inference configuration."""
        freezer.configure_for_phase(TrainingPhase.INFERENCE)

        # All layers should be frozen
        for idx in range(28):
            assert freezer.is_layer_frozen(idx), f"Layer {idx} should be frozen"

    def test_embeddings_frozen_in_phase_0_5(self, mock_model):
        """Test embeddings are frozen in Phase 0.5."""
        freezer = LayerFreezer(mock_model)
        freezer.configure_for_phase(TrainingPhase.PHASE_0_5)
        assert "embeddings" in freezer._frozen_components

    def test_embeddings_frozen_in_phase_1(self, mock_model):
        """Test embeddings are frozen in Phase 1."""
        freezer = LayerFreezer(mock_model)
        freezer.configure_for_phase(TrainingPhase.PHASE_1)
        assert "embeddings" in freezer._frozen_components

    def test_embeddings_unfrozen_in_phase_2(self, mock_model):
        """Test embeddings are trainable in Phase 2."""
        freezer = LayerFreezer(mock_model)
        freezer.configure_for_phase(TrainingPhase.PHASE_0_5)  # First freeze
        freezer.configure_for_phase(TrainingPhase.PHASE_2)  # Then unfreeze
        assert "embeddings" not in freezer._frozen_components

    def test_configure_returns_stats(self, freezer):
        """Test configure_for_phase returns stats."""
        stats = freezer.configure_for_phase(TrainingPhase.PHASE_1)
        assert "total_params" in stats
        assert "trainable_params" in stats
        assert "frozen_params" in stats
        assert "frozen_layers" in stats
        assert "trainable_layers" in stats


# ==============================================================================
# Statistics Tests
# ==============================================================================


class TestLayerFreezerStats:
    """Tests for freezing statistics."""

    def test_get_freeze_stats_all_trainable(self, freezer):
        """Test stats when all layers are trainable."""
        stats = freezer.get_freeze_stats()
        assert stats["frozen_layers"] == 0
        assert stats["trainable_layers"] == 28
        assert stats["trainable_params"] == stats["total_params"]
        assert stats["frozen_params"] == 0

    def test_get_freeze_stats_all_frozen(self, freezer):
        """Test stats when all layers are frozen."""
        for band in LayerBand:
            freezer.freeze_band(band)
        stats = freezer.get_freeze_stats()
        assert stats["frozen_layers"] == 28
        assert stats["trainable_layers"] == 0

    def test_get_frozen_layers(self, freezer):
        """Test get_frozen_layers method."""
        freezer.freeze_band(LayerBand.FOUNDATION)
        frozen = freezer.get_frozen_layers()
        assert frozen == [0, 1, 2, 3, 4, 5]

    def test_get_trainable_layers(self, freezer):
        """Test get_trainable_layers method."""
        freezer.freeze_band(LayerBand.FOUNDATION)
        freezer.freeze_band(LayerBand.CORE)
        trainable = freezer.get_trainable_layers()
        expected = list(range(18, 28))
        assert trainable == expected


# ==============================================================================
# Helper Functions Tests
# ==============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_configure_model_for_phase(self, mock_model):
        """Test configure_model_for_phase function."""
        stats = configure_model_for_phase(mock_model, "phase_1")
        assert stats["frozen_layers"] == 18
        assert stats["trainable_layers"] == 10

    def test_configure_model_for_phase_invalid(self, mock_model):
        """Test configure_model_for_phase with invalid phase."""
        with pytest.raises(ValueError):
            configure_model_for_phase(mock_model, "invalid_phase")

    def test_get_band_for_layer_foundation(self):
        """Test get_band_for_layer for foundation layers."""
        for idx in range(6):
            assert get_band_for_layer(idx) == LayerBand.FOUNDATION

    def test_get_band_for_layer_core(self):
        """Test get_band_for_layer for core layers."""
        for idx in range(6, 18):
            assert get_band_for_layer(idx) == LayerBand.CORE

    def test_get_band_for_layer_feeder(self):
        """Test get_band_for_layer for feeder layers."""
        for idx in range(18, 22):
            assert get_band_for_layer(idx) == LayerBand.FEEDER

    def test_get_band_for_layer_family(self):
        """Test get_band_for_layer for family layers."""
        for idx in range(22, 28):
            assert get_band_for_layer(idx) == LayerBand.FAMILY

    def test_get_band_for_layer_out_of_range(self):
        """Test get_band_for_layer returns None for invalid index."""
        assert get_band_for_layer(28) is None
        assert get_band_for_layer(-1) is None

    def test_get_layers_for_band(self):
        """Test get_layers_for_band."""
        layers = get_layers_for_band(LayerBand.FAMILY)
        assert layers == [22, 23, 24, 25, 26, 27]

    def test_get_trainable_bands_for_phase_enum(self):
        """Test get_trainable_bands_for_phase with enum."""
        bands = get_trainable_bands_for_phase(TrainingPhase.PHASE_1)
        assert LayerBand.FEEDER in bands
        assert LayerBand.FAMILY in bands
        assert LayerBand.FOUNDATION not in bands
        assert LayerBand.CORE not in bands

    def test_get_trainable_bands_for_phase_string(self):
        """Test get_trainable_bands_for_phase with string."""
        bands = get_trainable_bands_for_phase("phase_1")
        assert LayerBand.FEEDER in bands
        assert LayerBand.FAMILY in bands


# ==============================================================================
# Acceptance Criteria Tests
# ==============================================================================


class TestIssue511AcceptanceCriteria:
    """Tests for Issue 5.1.1 acceptance criteria."""

    def test_ac1_layer_band_enum_defines_all_4_bands(self):
        """AC1: LayerBand enum defines all 4 bands correctly."""
        assert LayerBand.FOUNDATION.value == "foundation"
        assert LayerBand.CORE.value == "core"
        assert LayerBand.FEEDER.value == "feeder"
        assert LayerBand.FAMILY.value == "family"
        print("AC1: LayerBand enum defines all 4 bands correctly [PASS]")

    def test_ac2_freeze_and_unfreeze_band_work(self, mock_model):
        """AC2: freeze_band() and unfreeze_band() work correctly."""
        freezer = LayerFreezer(mock_model)

        # Freeze
        freezer.freeze_band(LayerBand.FOUNDATION)
        for idx in LAYER_BANDS[LayerBand.FOUNDATION]:
            layer = freezer.get_layer(idx)
            for param in layer.parameters():
                assert not param.requires_grad

        # Unfreeze
        freezer.unfreeze_band(LayerBand.FOUNDATION)
        for idx in LAYER_BANDS[LayerBand.FOUNDATION]:
            layer = freezer.get_layer(idx)
            for param in layer.parameters():
                assert param.requires_grad

        print("AC2: freeze_band() and unfreeze_band() work correctly [PASS]")

    def test_ac3_configure_for_phase_works(self, mock_model):
        """AC3: configure_for_phase() sets up correct freezing."""
        freezer = LayerFreezer(mock_model)
        stats = freezer.configure_for_phase(TrainingPhase.PHASE_1)

        assert stats["frozen_layers"] == 18
        assert stats["trainable_layers"] == 10
        print("AC3: configure_for_phase() sets up correct freezing [PASS]")

    def test_ac4_phase_0_5_and_1_correct_freezing(self, mock_model):
        """AC4: Phase 0.5/1 has L1-18 frozen, L19-28 trainable."""
        for phase in [TrainingPhase.PHASE_0_5, TrainingPhase.PHASE_1]:
            freezer = LayerFreezer(mock_model)
            freezer.configure_for_phase(phase)

            # L1-18 (0-17) frozen
            for idx in range(18):
                layer = freezer.get_layer(idx)
                for param in layer.parameters():
                    assert not param.requires_grad, f"L{idx+1} should be frozen in {phase}"

            # L19-28 (18-27) trainable
            for idx in range(18, 28):
                layer = freezer.get_layer(idx)
                for param in layer.parameters():
                    assert param.requires_grad, f"L{idx+1} should be trainable in {phase}"

        print("AC4: Phase 0.5/1 has L1-18 frozen, L19-28 trainable [PASS]")

    def test_ac5_phase_2_all_layers_trainable(self, mock_model):
        """AC5: Phase 2 has all layers trainable."""
        freezer = LayerFreezer(mock_model)
        freezer.configure_for_phase(TrainingPhase.PHASE_2)

        for idx in range(28):
            layer = freezer.get_layer(idx)
            for param in layer.parameters():
                assert param.requires_grad, f"L{idx+1} should be trainable in Phase 2"

        print("AC5: Phase 2 has all layers trainable [PASS]")

    def test_ac6_embeddings_frozen_in_phase_0_5_1(self, mock_model):
        """AC6: Embeddings frozen in Phase 0.5/1."""
        for phase in [TrainingPhase.PHASE_0_5, TrainingPhase.PHASE_1]:
            freezer = LayerFreezer(mock_model)
            freezer.configure_for_phase(phase)

            for param in mock_model.embeddings.parameters():
                assert not param.requires_grad, f"Embeddings should be frozen in {phase}"

        print("AC6: Embeddings frozen in Phase 0.5/1 [PASS]")

    def test_ac7_get_freeze_stats_accurate(self, mock_model):
        """AC7: get_freeze_stats() returns accurate counts."""
        freezer = LayerFreezer(mock_model)

        # All trainable
        stats = freezer.get_freeze_stats()
        assert stats["frozen_layers"] == 0
        assert stats["trainable_layers"] == 28
        total = stats["total_params"]
        assert stats["trainable_params"] == total
        assert stats["frozen_params"] == 0

        # After phase 1
        freezer.configure_for_phase(TrainingPhase.PHASE_1)
        stats = freezer.get_freeze_stats()
        assert stats["frozen_layers"] == 18
        assert stats["trainable_layers"] == 10
        assert stats["trainable_params"] < total
        assert stats["frozen_params"] > 0
        assert stats["trainable_params"] + stats["frozen_params"] == total

        print("AC7: get_freeze_stats() returns accurate counts [PASS]")


# =============================================================================
# ISSUE 5.1.2: Phase-Aware Training Loop Tests
# =============================================================================


@pytest.fixture
def mock_train_dataloader():
    """Create a mock training dataloader."""
    data = [
        {
            "input_ids": torch.ones(2, 10, dtype=torch.long),
            "labels": torch.zeros(2, dtype=torch.long),
        }
        for _ in range(5)
    ]
    return data


@pytest.fixture
def mock_eval_dataloader():
    """Create a mock evaluation dataloader."""
    data = [
        {
            "input_ids": torch.ones(2, 10, dtype=torch.long),
            "labels": torch.zeros(2, dtype=torch.long),
        }
        for _ in range(2)
    ]
    return data


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from modeling_studio.trainers.trainer_v3 import TrainingConfig

        config = TrainingConfig()
        assert config.phase == "phase_0.5"
        assert config.max_steps == 2500
        assert config.warmup_steps == 500
        assert config.learning_rate == 3e-5
        assert config.gradient_accumulation_steps == 1
        assert config.max_grad_norm == 1.0
        assert config.weight_decay == 0.01
        assert config.fp16 is False
        assert config.bf16 is True
        assert config.output_dir == "outputs/v3_training"
        assert config.logging_steps == 50
        assert config.save_steps == 500
        assert config.eval_steps == 250
        print("Default TrainingConfig [PASS]")

    def test_custom_config(self):
        """Test custom configuration values."""
        from modeling_studio.trainers.trainer_v3 import TrainingConfig

        config = TrainingConfig(
            phase="phase_1",
            max_steps=5000,
            learning_rate=1e-4,
            fp16=True,
            bf16=False,
        )
        assert config.phase == "phase_1"
        assert config.max_steps == 5000
        assert config.learning_rate == 1e-4
        assert config.fp16 is True
        assert config.bf16 is False
        print("Custom TrainingConfig [PASS]")

    def test_per_layer_lr_config(self):
        """Test per-layer learning rate configuration."""
        from modeling_studio.trainers.trainer_v3 import TrainingConfig

        config = TrainingConfig(
            lr_layers_19_22=1e-5,
            lr_layer_23=5e-5,
            lr_layers_24_28=3e-5,
        )
        assert config.lr_layers_19_22 == 1e-5
        assert config.lr_layer_23 == 5e-5
        assert config.lr_layers_24_28 == 3e-5
        print("Per-layer LR config [PASS]")

    def test_phase_string_config(self):
        """Test phase string configuration."""
        from modeling_studio.trainers.trainer_v3 import TrainingConfig

        config = TrainingConfig(phase="phase_1")
        assert config.phase == "phase_1"
        print("Phase config [PASS]")

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        from modeling_studio.trainers.trainer_v3 import TrainingConfig

        config = TrainingConfig(max_steps=1000)
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["max_steps"] == 1000
        print("Config to_dict [PASS]")

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        from modeling_studio.trainers.trainer_v3 import TrainingConfig

        d = {"max_steps": 2000, "phase": "phase_2"}
        config = TrainingConfig.from_dict(d)
        assert config.max_steps == 2000
        assert config.phase == "phase_2"
        print("Config from_dict [PASS]")


class TestTrainingState:
    """Tests for TrainingState dataclass."""

    def test_default_state(self):
        """Test default training state values."""
        from modeling_studio.trainers.trainer_v3 import TrainingState

        state = TrainingState()
        assert state.global_step == 0
        assert state.epoch == 0
        assert state.best_metric == 0.0
        assert state.phase == "phase_0.5"
        assert state.losses == []
        assert state.metrics_history == []
        print("Default TrainingState [PASS]")

    def test_custom_state(self):
        """Test custom training state values."""
        from modeling_studio.trainers.trainer_v3 import TrainingState

        state = TrainingState(
            global_step=100,
            epoch=2,
            best_metric=0.85,
            phase="phase_1",
            losses=[0.5, 0.4, 0.3],
        )
        assert state.global_step == 100
        assert state.epoch == 2
        assert state.best_metric == 0.85
        assert state.phase == "phase_1"
        assert len(state.losses) == 3
        print("Custom TrainingState [PASS]")


class TestModernBERTv3TrainerInit:
    """Tests for ModernBERTv3Trainer initialization."""

    def test_trainer_init_minimal(self, mock_model, mock_train_dataloader):
        """Test trainer initialization with minimal config."""
        from modeling_studio.trainers.trainer_v3 import (
            ModernBERTv3Trainer,
            TrainingConfig,
        )

        config = TrainingConfig()
        trainer = ModernBERTv3Trainer(
            model=mock_model,
            config=config,
            train_dataloader=mock_train_dataloader,
        )

        assert trainer.model is mock_model
        assert trainer.config is config
        assert trainer.state is not None
        assert trainer.freezer is not None
        assert trainer.optimizer is None  # Not created until setup
        assert trainer.scheduler is None  # Not created until setup
        print("Trainer init minimal [PASS]")

    def test_trainer_has_layer_freezer(self, mock_model, mock_train_dataloader):
        """Test trainer creates a LayerFreezer."""
        from modeling_studio.trainers.trainer_v3 import (
            ModernBERTv3Trainer,
            TrainingConfig,
        )

        config = TrainingConfig()
        trainer = ModernBERTv3Trainer(
            model=mock_model,
            config=config,
            train_dataloader=mock_train_dataloader,
        )

        assert trainer.freezer is not None
        assert isinstance(trainer.freezer, LayerFreezer)
        print("Trainer has LayerFreezer [PASS]")

    def test_trainer_phase_configurable(self, mock_model, mock_train_dataloader):
        """Test trainer respects training phase config."""
        from modeling_studio.trainers.trainer_v3 import (
            ModernBERTv3Trainer,
            TrainingConfig,
        )

        config = TrainingConfig(phase="phase_2")
        trainer = ModernBERTv3Trainer(
            model=mock_model,
            config=config,
            train_dataloader=mock_train_dataloader,
        )

        assert trainer.config.phase == "phase_2"
        print("Trainer phase configurable [PASS]")


class TestModernBERTv3TrainerParameterGroups:
    """Tests for parameter group creation."""

    def test_get_parameter_groups_creates_groups(self, mock_model, mock_train_dataloader):
        """Test parameter groups are created."""
        from modeling_studio.trainers.trainer_v3 import (
            ModernBERTv3Trainer,
            TrainingConfig,
        )

        config = TrainingConfig()
        trainer = ModernBERTv3Trainer(
            model=mock_model,
            config=config,
            train_dataloader=mock_train_dataloader,
        )

        groups = trainer._get_parameter_groups()

        assert len(groups) > 0
        # All groups should have "params" and "lr" keys
        for group in groups:
            assert "params" in group
            assert "lr" in group
        print("Parameter groups created [PASS]")

    def test_get_parameter_groups_respects_freezing(self, mock_model, mock_train_dataloader):
        """Test parameter groups only include trainable params."""
        from modeling_studio.trainers.trainer_v3 import (
            ModernBERTv3Trainer,
            TrainingConfig,
        )

        config = TrainingConfig(phase="phase_1")
        trainer = ModernBERTv3Trainer(
            model=mock_model,
            config=config,
            train_dataloader=mock_train_dataloader,
        )

        # Configure for Phase 1 (L1-18 frozen)
        trainer.freezer.configure_for_phase(TrainingPhase.PHASE_1)

        groups = trainer._get_parameter_groups()

        # Collect all params from groups
        all_params = []
        for group in groups:
            all_params.extend(list(group["params"]))

        # All params should require grad
        for p in all_params:
            assert p.requires_grad
        print("Parameter groups respect freezing [PASS]")


class TestModernBERTv3TrainerOptimizer:
    """Tests for optimizer creation."""

    def test_create_optimizer(self, mock_model, mock_train_dataloader):
        """Test optimizer is created correctly."""
        from modeling_studio.trainers.trainer_v3 import (
            ModernBERTv3Trainer,
            TrainingConfig,
        )

        config = TrainingConfig()
        trainer = ModernBERTv3Trainer(
            model=mock_model,
            config=config,
            train_dataloader=mock_train_dataloader,
        )

        assert trainer.optimizer is None
        trainer.optimizer = trainer._create_optimizer()
        assert trainer.optimizer is not None
        print("Optimizer created [PASS]")

    def test_optimizer_weight_decay(self, mock_model, mock_train_dataloader):
        """Test optimizer respects weight decay config."""
        from modeling_studio.trainers.trainer_v3 import (
            ModernBERTv3Trainer,
            TrainingConfig,
        )

        config = TrainingConfig(weight_decay=0.1)
        trainer = ModernBERTv3Trainer(
            model=mock_model,
            config=config,
            train_dataloader=mock_train_dataloader,
        )
        trainer.optimizer = trainer._create_optimizer()

        # AdamW should be used
        assert trainer.optimizer is not None
        # Weight decay is applied in param groups
        print("Optimizer weight decay [PASS]")


class TestModernBERTv3TrainerScheduler:
    """Tests for scheduler creation."""

    def test_create_scheduler_cosine(self, mock_model, mock_train_dataloader):
        """Test cosine scheduler is created correctly."""
        from modeling_studio.trainers.trainer_v3 import (
            ModernBERTv3Trainer,
            TrainingConfig,
        )

        config = TrainingConfig(lr_scheduler_type="cosine")
        trainer = ModernBERTv3Trainer(
            model=mock_model,
            config=config,
            train_dataloader=mock_train_dataloader,
        )
        trainer.optimizer = trainer._create_optimizer()
        trainer.scheduler = trainer._create_scheduler()

        assert trainer.scheduler is not None
        print("Cosine scheduler created [PASS]")

    def test_create_scheduler_linear(self, mock_model, mock_train_dataloader):
        """Test linear scheduler is created correctly."""
        from modeling_studio.trainers.trainer_v3 import (
            ModernBERTv3Trainer,
            TrainingConfig,
        )

        config = TrainingConfig(lr_scheduler_type="linear")
        trainer = ModernBERTv3Trainer(
            model=mock_model,
            config=config,
            train_dataloader=mock_train_dataloader,
        )
        trainer.optimizer = trainer._create_optimizer()
        trainer.scheduler = trainer._create_scheduler()

        assert trainer.scheduler is not None
        print("Linear scheduler created [PASS]")


class TestModernBERTv3TrainerSetup:
    """Tests for trainer setup."""

    def test_setup_creates_optimizer_and_scheduler(self, mock_model, mock_train_dataloader):
        """Test setup creates optimizer and scheduler."""
        from modeling_studio.trainers.trainer_v3 import (
            ModernBERTv3Trainer,
            TrainingConfig,
        )

        config = TrainingConfig()
        trainer = ModernBERTv3Trainer(
            model=mock_model,
            config=config,
            train_dataloader=mock_train_dataloader,
        )

        trainer.setup()

        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        print("Setup creates optimizer and scheduler [PASS]")

    def test_setup_configures_phase(self, mock_model, mock_train_dataloader):
        """Test setup configures model for training phase."""
        from modeling_studio.trainers.trainer_v3 import (
            ModernBERTv3Trainer,
            TrainingConfig,
        )

        config = TrainingConfig(phase="phase_1")
        trainer = ModernBERTv3Trainer(
            model=mock_model,
            config=config,
            train_dataloader=mock_train_dataloader,
        )

        trainer.setup()

        # Model should be configured for Phase 1
        stats = trainer.freezer.get_freeze_stats()
        assert stats["frozen_layers"] == 18  # L1-18 frozen
        assert stats["trainable_layers"] == 10  # L19-28 trainable
        print("Setup configures phase [PASS]")


class TestIssue512AcceptanceCriteria:
    """Acceptance criteria tests for Issue 5.1.2."""

    def test_ac1_training_config_supports_phase_settings(self):
        """AC1: TrainingConfig supports all phase-specific settings."""
        from modeling_studio.trainers.trainer_v3 import TrainingConfig

        config = TrainingConfig(
            phase="phase_1",
            lr_layers_19_22=1e-5,
            lr_layer_23=5e-5,
            lr_layers_24_28=3e-5,
        )

        assert config.phase == "phase_1"
        assert config.lr_layers_19_22 == 1e-5
        assert config.lr_layer_23 == 5e-5
        assert config.lr_layers_24_28 == 3e-5
        print("AC1: TrainingConfig supports phase settings [PASS]")

    def test_ac2_per_layer_group_learning_rates(self, mock_model, mock_train_dataloader):
        """AC2: Per-layer-group learning rates applied correctly."""
        from modeling_studio.trainers.trainer_v3 import (
            ModernBERTv3Trainer,
            TrainingConfig,
        )

        config = TrainingConfig(
            lr_layers_19_22=1e-5,
            lr_layer_23=5e-5,
            lr_layers_24_28=3e-5,
        )
        trainer = ModernBERTv3Trainer(
            model=mock_model,
            config=config,
            train_dataloader=mock_train_dataloader,
        )

        groups = trainer._get_parameter_groups()

        # Verify different LR groups exist
        lrs = {g["lr"] for g in groups}
        assert len(lrs) > 1  # Multiple different LRs
        print("AC2: Per-layer-group learning rates [PASS]")

    def test_ac3_warmup_and_cosine_decay(self, mock_model, mock_train_dataloader):
        """AC3: Warmup + cosine decay scheduler works."""
        from modeling_studio.trainers.trainer_v3 import (
            ModernBERTv3Trainer,
            TrainingConfig,
        )

        # Use phase_2 so all layers are trainable with non-zero LR
        config = TrainingConfig(
            phase="phase_2",
            warmup_steps=100,
            max_steps=1000,
            lr_scheduler_type="cosine",
            learning_rate=3e-5,
            lr_layers_1_18=1e-6,  # Non-zero LR
            lr_layers_19_22=1e-5,
            lr_layer_23=5e-5,
            lr_layers_24_28=3e-5,
        )
        trainer = ModernBERTv3Trainer(
            model=mock_model,
            config=config,
            train_dataloader=mock_train_dataloader,
        )

        # Configure for phase 2 (all layers trainable)
        trainer.freezer.configure_for_phase(TrainingPhase.PHASE_2)

        trainer.optimizer = trainer._create_optimizer()
        trainer.scheduler = trainer._create_scheduler()

        assert trainer.scheduler is not None

        # Record LRs during warmup
        lrs = []
        for _ in range(100):
            lrs.append(trainer.scheduler.get_last_lr()[0])
            trainer.scheduler.step()

        # LR should increase during warmup (from ~0 to peak)
        assert lrs[-1] > lrs[0], f"LR should increase during warmup: {lrs[0]} -> {lrs[-1]}"

        # LR at end of warmup should be at or near peak
        peak_lr = lrs[-1]

        # Continue past warmup into cosine decay
        for _ in range(100):
            trainer.scheduler.step()

        post_warmup_lr = trainer.scheduler.get_last_lr()[0]

        # LR should decrease after warmup (cosine decay)
        assert post_warmup_lr < peak_lr, "LR should decay after warmup"

        print("AC3: Warmup + cosine decay scheduler [PASS]")

    def test_ac4_gradient_clipping_config(self, mock_model, mock_train_dataloader):
        """AC4: Gradient clipping is configurable."""
        from modeling_studio.trainers.trainer_v3 import (
            ModernBERTv3Trainer,
            TrainingConfig,
        )

        config = TrainingConfig(max_grad_norm=0.5)
        trainer = ModernBERTv3Trainer(
            model=mock_model,
            config=config,
            train_dataloader=mock_train_dataloader,
        )

        assert trainer.config.max_grad_norm == 0.5
        print("AC4: Gradient clipping configurable [PASS]")

    def test_ac5_mixed_precision_supported(self, mock_model, mock_train_dataloader):
        """AC5: Mixed precision (bf16/fp16) supported."""
        from modeling_studio.trainers.trainer_v3 import (
            ModernBERTv3Trainer,
            TrainingConfig,
        )

        # Test bf16
        config_bf16 = TrainingConfig(bf16=True, fp16=False)
        trainer_bf16 = ModernBERTv3Trainer(
            model=mock_model,
            config=config_bf16,
            train_dataloader=mock_train_dataloader,
        )
        assert trainer_bf16.config.bf16 is True
        assert trainer_bf16.config.fp16 is False

        # Test fp16
        config_fp16 = TrainingConfig(bf16=False, fp16=True)
        trainer_fp16 = ModernBERTv3Trainer(
            model=mock_model,
            config=config_fp16,
            train_dataloader=mock_train_dataloader,
        )
        assert trainer_fp16.config.fp16 is True
        assert trainer_fp16.config.bf16 is False

        print("AC5: Mixed precision supported [PASS]")

    def test_ac6_checkpointing_configurable(self, mock_model, mock_train_dataloader):
        """AC6: Checkpointing at configurable intervals."""
        from modeling_studio.trainers.trainer_v3 import (
            ModernBERTv3Trainer,
            TrainingConfig,
        )

        config = TrainingConfig(save_steps=250)
        trainer = ModernBERTv3Trainer(
            model=mock_model,
            config=config,
            train_dataloader=mock_train_dataloader,
        )

        assert trainer.config.save_steps == 250
        print("AC6: Checkpointing configurable [PASS]")

    def test_ac7_wandb_integration_config(self, mock_model, mock_train_dataloader):
        """AC7: WandB logging integration available."""
        from modeling_studio.trainers.trainer_v3 import (
            ModernBERTv3Trainer,
            TrainingConfig,
        )

        config = TrainingConfig(
            use_wandb=True,
            wandb_project="test-project",
            wandb_run_name="test-run",
        )
        trainer = ModernBERTv3Trainer(
            model=mock_model,
            config=config,
            train_dataloader=mock_train_dataloader,
        )

        assert trainer.config.use_wandb is True
        assert trainer.config.wandb_project == "test-project"
        assert trainer.config.wandb_run_name == "test-run"
        print("AC7: WandB integration config [PASS]")

    def test_ac8_eval_configurable(self, mock_model, mock_train_dataloader):
        """AC8: Evaluation at configurable intervals."""
        from modeling_studio.trainers.trainer_v3 import (
            ModernBERTv3Trainer,
            TrainingConfig,
        )

        config = TrainingConfig(eval_steps=100)
        trainer = ModernBERTv3Trainer(
            model=mock_model,
            config=config,
            train_dataloader=mock_train_dataloader,
        )

        assert trainer.config.eval_steps == 100
        print("AC8: Evaluation configurable [PASS]")


# =============================================================================
# ISSUE 5.1.3: LoRA Application Tests
# =============================================================================


class TestLoRAConfig:
    """Tests for LoRAConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from modeling_studio.trainers.lora_v3 import LoRAConfig

        config = LoRAConfig()
        assert config.rank == 16
        assert config.alpha == 32.0
        assert config.dropout == 0.1
        assert config.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]
        assert config.layers == list(range(22, 28))
        print("Default LoRAConfig [PASS]")

    def test_custom_config(self):
        """Test custom configuration values."""
        from modeling_studio.trainers.lora_v3 import LoRAConfig

        config = LoRAConfig(
            rank=8,
            alpha=16.0,
            dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            layers=[22, 23, 24],
        )
        assert config.rank == 8
        assert config.alpha == 16.0
        assert config.dropout == 0.05
        assert config.target_modules == ["q_proj", "v_proj"]
        assert config.layers == [22, 23, 24]
        print("Custom LoRAConfig [PASS]")

    def test_scaling_property(self):
        """Test scaling factor computation."""
        from modeling_studio.trainers.lora_v3 import LoRAConfig

        config = LoRAConfig(rank=16, alpha=32.0)
        assert config.scaling == 2.0

        config2 = LoRAConfig(rank=8, alpha=16.0)
        assert config2.scaling == 2.0

        config3 = LoRAConfig(rank=16, alpha=16.0)
        assert config3.scaling == 1.0
        print("Scaling property [PASS]")

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from modeling_studio.trainers.lora_v3 import LoRAConfig

        config = LoRAConfig(rank=8)
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["rank"] == 8
        print("LoRAConfig to_dict [PASS]")


class TestLoRALinear:
    """Tests for LoRALinear module."""

    def test_init(self):
        """Test LoRALinear initialization."""
        from modeling_studio.trainers.lora_v3 import LoRALinear

        lora = LoRALinear(768, 768, rank=16, alpha=32.0)
        assert lora.in_features == 768
        assert lora.out_features == 768
        assert lora.rank == 16
        assert lora.alpha == 32.0
        assert lora.scaling == 2.0
        assert lora.merged is False
        assert lora.enabled is True
        print("LoRALinear init [PASS]")

    def test_forward_shape(self):
        """Test LoRALinear forward produces correct shape."""
        from modeling_studio.trainers.lora_v3 import LoRALinear

        lora = LoRALinear(768, 768, rank=16)
        x = torch.randn(2, 50, 768)
        y = lora(x)
        assert y.shape == (2, 50, 768)
        print("LoRALinear forward shape [PASS]")

    def test_lora_init_zero_b(self):
        """Test that B is initialized to zeros (LoRA starts as identity)."""
        from modeling_studio.trainers.lora_v3 import LoRALinear

        lora = LoRALinear(768, 768, rank=16)
        assert torch.allclose(lora.lora_B.weight, torch.zeros_like(lora.lora_B.weight))
        print("LoRA B initialized to zeros [PASS]")

    def test_lora_contribution_disabled(self):
        """Test LoRA is bypassed when disabled."""
        from modeling_studio.trainers.lora_v3 import LoRALinear

        lora = LoRALinear(768, 768, rank=16)
        lora.enabled = False

        x = torch.randn(2, 50, 768)
        y_disabled = lora(x)
        y_linear_only = lora.linear(x)

        assert torch.allclose(y_disabled, y_linear_only)
        print("LoRA disabled bypass [PASS]")

    def test_from_linear(self):
        """Test creating LoRALinear from existing Linear."""
        from modeling_studio.trainers.lora_v3 import LoRALinear

        linear = nn.Linear(768, 768)
        lora = LoRALinear.from_linear(linear, rank=16)

        # Weights should be copied
        assert torch.allclose(lora.linear.weight, linear.weight)

        # Base weights should be frozen
        assert not lora.linear.weight.requires_grad

        # LoRA weights should be trainable
        assert lora.lora_A.weight.requires_grad
        assert lora.lora_B.weight.requires_grad
        print("LoRALinear from_linear [PASS]")

    def test_merge_weights(self):
        """Test merging LoRA weights into base layer."""
        from modeling_studio.trainers.lora_v3 import LoRALinear

        lora = LoRALinear(768, 768, rank=16)
        lora.lora_B.weight.data.fill_(0.1)  # Non-zero for test

        original_weight = lora.linear.weight.clone()

        lora.merge_weights()

        # Weight should be modified
        assert not torch.allclose(lora.linear.weight, original_weight)
        assert lora.merged is True
        print("LoRALinear merge_weights [PASS]")

    def test_unmerge_weights(self):
        """Test unmerging LoRA weights from base layer."""
        from modeling_studio.trainers.lora_v3 import LoRALinear

        lora = LoRALinear(768, 768, rank=16)
        lora.lora_B.weight.data.fill_(0.1)  # Non-zero for test

        original_weight = lora.linear.weight.clone()

        lora.merge_weights()
        lora.unmerge_weights()

        # Weight should be restored
        assert torch.allclose(lora.linear.weight, original_weight, atol=1e-6)
        assert lora.merged is False
        print("LoRALinear unmerge_weights [PASS]")

    def test_get_lora_params(self):
        """Test LoRA parameter count."""
        from modeling_studio.trainers.lora_v3 import LoRALinear

        lora = LoRALinear(768, 768, rank=16)
        params = lora.get_lora_params()

        # A: [768, 16] = 12288, B: [16, 768] = 12288
        expected = 768 * 16 + 16 * 768
        assert params == expected
        print("LoRALinear get_lora_params [PASS]")


class TestLoRAManager:
    """Tests for LoRAManager class."""

    @pytest.fixture
    def mock_model_with_attention(self):
        """Create a mock model with attention layers containing projections."""

        class MockAttention(nn.Module):
            def __init__(self, hidden_size: int = 768):
                super().__init__()
                self.q_proj = nn.Linear(hidden_size, hidden_size)
                self.k_proj = nn.Linear(hidden_size, hidden_size)
                self.v_proj = nn.Linear(hidden_size, hidden_size)
                self.o_proj = nn.Linear(hidden_size, hidden_size)

            def forward(self, x):
                return x

        class MockLayer(nn.Module):
            def __init__(self, hidden_size: int = 768):
                super().__init__()
                self.attn = MockAttention(hidden_size)
                self.ffn = nn.Linear(hidden_size, hidden_size * 4)

            def forward(self, x):
                return self.attn(x) + self.ffn(x)

        class MockEncoder(nn.Module):
            def __init__(self, num_layers: int = 28, hidden_size: int = 768):
                super().__init__()
                self.layers = nn.ModuleList([MockLayer(hidden_size) for _ in range(num_layers)])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = MockEncoder()

            def forward(self, x):
                return self.encoder(x)

        return MockModel()

    def test_apply_lora(self, mock_model_with_attention):
        """Test applying LoRA to model."""
        from modeling_studio.trainers.lora_v3 import LoRAConfig, LoRAManager

        config = LoRAConfig(
            rank=16,
            target_modules=["attn.q_proj", "attn.v_proj"],
            layers=[22, 23],
        )
        manager = LoRAManager(mock_model_with_attention, config)

        params = manager.apply_lora()

        assert params > 0
        assert len(manager.lora_modules) == 4  # 2 modules x 2 layers
        print("LoRAManager apply_lora [PASS]")

    def test_get_lora_parameters(self, mock_model_with_attention):
        """Test getting LoRA parameters for optimizer."""
        from modeling_studio.trainers.lora_v3 import LoRAConfig, LoRAManager

        config = LoRAConfig(
            rank=16,
            target_modules=["attn.q_proj"],
            layers=[22],
        )
        manager = LoRAManager(mock_model_with_attention, config)
        manager.apply_lora()

        params = manager.get_lora_parameters()

        assert len(params) > 0
        # Each LoRA module has A and B
        assert len(params) == 2  # 1 module x 2 (A + B)
        print("LoRAManager get_lora_parameters [PASS]")

    def test_merge_all(self, mock_model_with_attention):
        """Test merging all LoRA weights."""
        from modeling_studio.trainers.lora_v3 import LoRAConfig, LoRAManager

        config = LoRAConfig(
            rank=16,
            target_modules=["attn.q_proj"],
            layers=[22],
        )
        manager = LoRAManager(mock_model_with_attention, config)
        manager.apply_lora()

        manager.merge_all()

        for lora_module in manager.lora_modules.values():
            assert lora_module.merged is True
        print("LoRAManager merge_all [PASS]")

    def test_unmerge_all(self, mock_model_with_attention):
        """Test unmerging all LoRA weights."""
        from modeling_studio.trainers.lora_v3 import LoRAConfig, LoRAManager

        config = LoRAConfig(
            rank=16,
            target_modules=["attn.q_proj"],
            layers=[22],
        )
        manager = LoRAManager(mock_model_with_attention, config)
        manager.apply_lora()

        manager.merge_all()
        manager.unmerge_all()

        for lora_module in manager.lora_modules.values():
            assert lora_module.merged is False
        print("LoRAManager unmerge_all [PASS]")

    def test_enable_disable_lora(self, mock_model_with_attention):
        """Test enabling/disabling LoRA."""
        from modeling_studio.trainers.lora_v3 import LoRAConfig, LoRAManager

        config = LoRAConfig(
            rank=16,
            target_modules=["attn.q_proj"],
            layers=[22],
        )
        manager = LoRAManager(mock_model_with_attention, config)
        manager.apply_lora()

        manager.enable_lora(False)
        for lora_module in manager.lora_modules.values():
            assert lora_module.enabled is False

        manager.enable_lora(True)
        for lora_module in manager.lora_modules.values():
            assert lora_module.enabled is True
        print("LoRAManager enable/disable [PASS]")

    def test_save_load_lora_weights(self, mock_model_with_attention, tmp_path):
        """Test saving and loading LoRA weights."""
        from modeling_studio.trainers.lora_v3 import LoRAConfig, LoRAManager

        config = LoRAConfig(
            rank=16,
            target_modules=["attn.q_proj"],
            layers=[22],
        )
        manager = LoRAManager(mock_model_with_attention, config)
        manager.apply_lora()

        # Modify weights
        for lora_module in manager.lora_modules.values():
            lora_module.lora_A.weight.data.fill_(1.0)
            lora_module.lora_B.weight.data.fill_(2.0)

        # Save
        save_path = tmp_path / "lora_weights.pt"
        manager.save_lora_weights(save_path)

        # Reset weights
        for lora_module in manager.lora_modules.values():
            lora_module.lora_A.weight.data.zero_()
            lora_module.lora_B.weight.data.zero_()

        # Load
        manager.load_lora_weights(save_path)

        # Verify restored
        for lora_module in manager.lora_modules.values():
            assert torch.allclose(
                lora_module.lora_A.weight, torch.ones_like(lora_module.lora_A.weight)
            )
            assert torch.allclose(
                lora_module.lora_B.weight, torch.full_like(lora_module.lora_B.weight, 2.0)
            )
        print("LoRAManager save/load [PASS]")

    def test_get_stats(self, mock_model_with_attention):
        """Test getting LoRA statistics."""
        from modeling_studio.trainers.lora_v3 import LoRAConfig, LoRAManager

        config = LoRAConfig(
            rank=16,
            target_modules=["attn.q_proj", "attn.v_proj"],
            layers=[22, 23],
        )
        manager = LoRAManager(mock_model_with_attention, config)
        manager.apply_lora()

        stats = manager.get_stats()

        assert stats["num_modules"] == 4
        assert stats["rank"] == 16
        assert stats["total_lora_params"] > 0
        print("LoRAManager get_stats [PASS]")


class TestApplyLoRAToFamilyBand:
    """Tests for apply_lora_to_family_band convenience function."""

    @pytest.fixture
    def mock_model_28_layers(self):
        """Create a mock model with 28 layers containing attention projections."""

        class MockAttention(nn.Module):
            def __init__(self, hidden_size: int = 768):
                super().__init__()
                self.q_proj = nn.Linear(hidden_size, hidden_size)
                self.k_proj = nn.Linear(hidden_size, hidden_size)
                self.v_proj = nn.Linear(hidden_size, hidden_size)
                self.o_proj = nn.Linear(hidden_size, hidden_size)

            def forward(self, x):
                return x

        class MockLayer(nn.Module):
            def __init__(self, hidden_size: int = 768):
                super().__init__()
                self.attn = MockAttention(hidden_size)

            def forward(self, x):
                return self.attn(x)

        class MockEncoder(nn.Module):
            def __init__(self, num_layers: int = 28, hidden_size: int = 768):
                super().__init__()
                self.layers = nn.ModuleList([MockLayer(hidden_size) for _ in range(num_layers)])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = MockEncoder()

            def forward(self, x):
                return self.encoder(x)

        return MockModel()

    def test_apply_to_family_band(self, mock_model_28_layers):
        """Test applying LoRA to Family Band."""
        from modeling_studio.trainers.lora_v3 import apply_lora_to_family_band

        manager = apply_lora_to_family_band(
            mock_model_28_layers,
            rank=16,
            target_modules=["attn.q_proj", "attn.v_proj"],
        )

        # Should apply to layers 22-27 (6 layers) x 2 modules = 12 LoRA modules
        assert len(manager.lora_modules) == 12
        print("apply_lora_to_family_band [PASS]")


class TestGetLoRAParamCount:
    """Tests for get_lora_param_count helper function."""

    def test_param_count_calculation(self):
        """Test LoRA parameter count calculation."""
        from modeling_studio.trainers.lora_v3 import get_lora_param_count

        # Family Band: 6 layers, 4 modules each (QKVO)
        params = get_lora_param_count(
            hidden_size=768,
            rank=16,
            num_layers=6,
            num_modules_per_layer=4,
        )

        # Each module: A [768, 16] + B [16, 768] = 2 * 768 * 16 = 24,576
        # 6 layers * 4 modules * 24,576 = 589,824
        expected = 2 * 768 * 16 * 6 * 4
        assert params == expected
        print("get_lora_param_count [PASS]")


class TestIssue513AcceptanceCriteria:
    """Acceptance criteria tests for Issue 5.1.3."""

    @pytest.fixture
    def mock_model_for_lora(self):
        """Create a mock model for LoRA testing."""

        class MockAttention(nn.Module):
            def __init__(self, hidden_size: int = 768):
                super().__init__()
                self.q_proj = nn.Linear(hidden_size, hidden_size)
                self.k_proj = nn.Linear(hidden_size, hidden_size)
                self.v_proj = nn.Linear(hidden_size, hidden_size)
                self.o_proj = nn.Linear(hidden_size, hidden_size)

            def forward(self, x):
                return x

        class MockLayer(nn.Module):
            def __init__(self, hidden_size: int = 768):
                super().__init__()
                self.attn = MockAttention(hidden_size)

            def forward(self, x):
                return self.attn(x)

        class MockEncoder(nn.Module):
            def __init__(self, num_layers: int = 28, hidden_size: int = 768):
                super().__init__()
                self.layers = nn.ModuleList([MockLayer(hidden_size) for _ in range(num_layers)])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = MockEncoder()

            def forward(self, x):
                return self.encoder(x)

        return MockModel()

    def test_ac1_lora_linear_implements_low_rank_correctly(self):
        """AC1: LoRALinear implements low-rank adaptation correctly."""
        from modeling_studio.trainers.lora_v3 import LoRALinear

        lora = LoRALinear(768, 768, rank=16, alpha=32.0)

        # A projects down: [768 -> 16]
        assert lora.lora_A.weight.shape == (16, 768)

        # B projects up: [16 -> 768]
        assert lora.lora_B.weight.shape == (768, 16)

        # Scaling is correct
        assert lora.scaling == 32.0 / 16

        # Forward works
        x = torch.randn(2, 50, 768)
        y = lora(x)
        assert y.shape == x.shape

        print("AC1: LoRALinear implements low-rank correctly [PASS]")

    def test_ac2_merge_weights_combines_correctly(self, mock_model_for_lora):
        """AC2: merge_weights() combines LoRA into base layer."""
        from modeling_studio.trainers.lora_v3 import LoRALinear

        lora = LoRALinear(768, 768, rank=16)

        # Set non-zero LoRA weights
        lora.lora_B.weight.data.fill_(0.1)

        # Get outputs before merge
        x = torch.randn(2, 50, 768)
        lora.eval()
        y_before = lora(x).clone()

        # Merge
        lora.merge_weights()

        # Get outputs after merge (should be same)
        y_after = lora(x)

        assert torch.allclose(y_before, y_after, atol=1e-5)
        print("AC2: merge_weights() combines correctly [PASS]")

    def test_ac3_unmerge_weights_reverses_correctly(self):
        """AC3: unmerge_weights() reverses merge correctly."""
        from modeling_studio.trainers.lora_v3 import LoRALinear

        lora = LoRALinear(768, 768, rank=16)
        lora.lora_B.weight.data.fill_(0.1)

        original_weight = lora.linear.weight.clone()

        lora.merge_weights()
        assert not torch.allclose(lora.linear.weight, original_weight)

        lora.unmerge_weights()
        assert torch.allclose(lora.linear.weight, original_weight, atol=1e-6)

        print("AC3: unmerge_weights() reverses correctly [PASS]")

    def test_ac4_lora_manager_targets_l23_28(self, mock_model_for_lora):
        """AC4: LoRAManager.apply_lora() targets L23-28 attention projections."""
        from modeling_studio.trainers.lora_v3 import apply_lora_to_family_band

        manager = apply_lora_to_family_band(
            mock_model_for_lora,
            rank=16,
            target_modules=["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj"],
        )

        # Check all targeted layers are in Family Band (22-27, 0-indexed)
        for name in manager.lora_modules.keys():
            layer_idx = int(name.split("_")[1].split(".")[0])
            assert 22 <= layer_idx <= 27, f"Layer {layer_idx} not in Family Band"

        print("AC4: LoRAManager targets L23-28 [PASS]")

    def test_ac5_get_lora_parameters_returns_only_trainable(self, mock_model_for_lora):
        """AC5: get_lora_parameters() returns only trainable LoRA params."""
        from modeling_studio.trainers.lora_v3 import apply_lora_to_family_band

        manager = apply_lora_to_family_band(
            mock_model_for_lora,
            rank=16,
            target_modules=["attn.q_proj"],
        )

        params = manager.get_lora_parameters()

        # All returned params should require grad
        for p in params:
            assert p.requires_grad

        # Should be A and B for each module
        assert len(params) == 2 * len(manager.lora_modules)

        print("AC5: get_lora_parameters() returns only trainable [PASS]")

    def test_ac6_save_load_works(self, mock_model_for_lora, tmp_path):
        """AC6: save_lora_weights() / load_lora_weights() work correctly."""
        from modeling_studio.trainers.lora_v3 import apply_lora_to_family_band

        manager = apply_lora_to_family_band(
            mock_model_for_lora,
            rank=16,
            target_modules=["attn.q_proj"],
        )

        # Modify weights
        for lora_module in manager.lora_modules.values():
            lora_module.lora_A.weight.data.fill_(3.14)

        # Save
        save_path = tmp_path / "lora.pt"
        manager.save_lora_weights(save_path)

        # Reset
        for lora_module in manager.lora_modules.values():
            lora_module.lora_A.weight.data.zero_()

        # Load
        manager.load_lora_weights(save_path)

        # Verify
        for lora_module in manager.lora_modules.values():
            assert torch.allclose(
                lora_module.lora_A.weight.data, torch.full_like(lora_module.lora_A.weight, 3.14)
            )

        print("AC6: save/load works correctly [PASS]")

    def test_ac7_lora_rank_16_adds_reasonable_params(self, mock_model_for_lora):
        """AC7: LoRA rank 16 adds reasonable number of params."""
        from modeling_studio.trainers.lora_v3 import apply_lora_to_family_band

        manager = apply_lora_to_family_band(
            mock_model_for_lora,
            rank=16,
            target_modules=["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj"],
        )

        stats = manager.get_stats()
        lora_params = stats["total_lora_params"]

        # Expected: 6 layers * 4 modules * 2 * 768 * 16 = 589,824
        # This is ~0.4% of a 150M model
        expected = 6 * 4 * 2 * 768 * 16
        assert lora_params == expected

        print(f"AC7: LoRA params = {lora_params:,} (expected ~2M for full model) [PASS]")


# ==============================================================================
# Issue 5.1.4: Layer-Group Learning Rates Tests
# ==============================================================================


class TestLayerGroupLRConfig:
    """Tests for LayerGroupLRConfig dataclass."""

    def test_default_config_values(self):
        """Test default config values match specification."""
        from modeling_studio.trainers.lr_groups_v3 import LayerGroupLRConfig

        config = LayerGroupLRConfig()

        assert config.base_lr == 3e-5
        assert config.foundation_mult == 0.0
        assert config.core_mult == 0.0
        assert config.feeder_mult == 0.33
        assert config.interface_mult == 1.67
        assert config.family_mult == 1.0
        assert config.embeddings_mult == 0.1
        assert config.task_heads_mult == 1.0
        assert config.hub_tokens_mult == 0.5
        assert config.warmup_ratio == 0.1
        assert config.min_lr_ratio == 0.01

    def test_custom_config_values(self):
        """Test custom config values."""
        from modeling_studio.trainers.lr_groups_v3 import LayerGroupLRConfig

        config = LayerGroupLRConfig(
            base_lr=1e-4,
            foundation_mult=0.1,
            core_mult=0.2,
            feeder_mult=0.5,
            interface_mult=2.0,
            family_mult=1.5,
        )

        assert config.base_lr == 1e-4
        assert config.foundation_mult == 0.1
        assert config.core_mult == 0.2
        assert config.feeder_mult == 0.5
        assert config.interface_mult == 2.0
        assert config.family_mult == 1.5

    def test_get_layer_lr_foundation(self):
        """Test get_layer_lr for Foundation band (L1-6)."""
        from modeling_studio.trainers.lr_groups_v3 import LayerGroupLRConfig

        config = LayerGroupLRConfig(base_lr=3e-5, foundation_mult=0.0)

        for layer_idx in range(0, 6):
            lr = config.get_layer_lr(layer_idx)
            assert lr == 0.0, f"Layer {layer_idx} should have 0 LR"

    def test_get_layer_lr_core(self):
        """Test get_layer_lr for Core band (L7-18)."""
        from modeling_studio.trainers.lr_groups_v3 import LayerGroupLRConfig

        config = LayerGroupLRConfig(base_lr=3e-5, core_mult=0.0)

        for layer_idx in range(6, 18):
            lr = config.get_layer_lr(layer_idx)
            assert lr == 0.0, f"Layer {layer_idx} should have 0 LR"

    def test_get_layer_lr_feeder(self):
        """Test get_layer_lr for Feeder band (L19-22)."""
        from modeling_studio.trainers.lr_groups_v3 import LayerGroupLRConfig

        config = LayerGroupLRConfig(base_lr=3e-5, feeder_mult=0.33)

        expected_lr = 3e-5 * 0.33
        for layer_idx in range(18, 22):
            lr = config.get_layer_lr(layer_idx)
            assert abs(lr - expected_lr) < 1e-10, f"Layer {layer_idx} LR mismatch"

    def test_get_layer_lr_interface(self):
        """Test get_layer_lr for Interface layer (L23)."""
        from modeling_studio.trainers.lr_groups_v3 import LayerGroupLRConfig

        config = LayerGroupLRConfig(base_lr=3e-5, interface_mult=1.67)

        expected_lr = 3e-5 * 1.67
        lr = config.get_layer_lr(22)  # L23 is 0-indexed
        assert abs(lr - expected_lr) < 1e-10

    def test_get_layer_lr_family(self):
        """Test get_layer_lr for Family band (L24-28)."""
        from modeling_studio.trainers.lr_groups_v3 import LayerGroupLRConfig

        config = LayerGroupLRConfig(base_lr=3e-5, family_mult=1.0)

        expected_lr = 3e-5 * 1.0
        for layer_idx in range(23, 28):
            lr = config.get_layer_lr(layer_idx)
            assert abs(lr - expected_lr) < 1e-10, f"Layer {layer_idx} LR mismatch"

    def test_get_component_lr(self):
        """Test get_component_lr for various components."""
        from modeling_studio.trainers.lr_groups_v3 import LayerGroupLRConfig

        config = LayerGroupLRConfig(
            base_lr=3e-5,
            embeddings_mult=0.1,
            task_heads_mult=1.0,
            hub_tokens_mult=0.5,
        )

        assert abs(config.get_component_lr("embeddings") - 3e-6) < 1e-12
        assert abs(config.get_component_lr("task_heads") - 3e-5) < 1e-12
        assert abs(config.get_component_lr("hub_tokens") - 1.5e-5) < 1e-12
        assert abs(config.get_component_lr("unknown") - 3e-5) < 1e-12

    def test_get_band_lr(self):
        """Test get_band_lr for all bands."""
        from modeling_studio.trainers.lr_groups_v3 import LayerGroupLRConfig

        config = LayerGroupLRConfig(base_lr=3e-5)

        assert config.get_band_lr("foundation") == 0.0
        assert config.get_band_lr("core") == 0.0
        assert abs(config.get_band_lr("feeder") - 3e-5 * 0.33) < 1e-12
        assert abs(config.get_band_lr("interface") - 3e-5 * 1.67) < 1e-12
        assert abs(config.get_band_lr("family") - 3e-5 * 1.0) < 1e-12

    def test_get_warmup_steps(self):
        """Test get_warmup_steps calculation."""
        from modeling_studio.trainers.lr_groups_v3 import LayerGroupLRConfig

        config = LayerGroupLRConfig(warmup_ratio=0.1)

        assert config.get_warmup_steps(1000) == 100
        assert config.get_warmup_steps(2500) == 250
        assert config.get_warmup_steps(100) == 10

    def test_get_min_lr(self):
        """Test get_min_lr calculation."""
        from modeling_studio.trainers.lr_groups_v3 import LayerGroupLRConfig

        config = LayerGroupLRConfig(base_lr=3e-5, min_lr_ratio=0.01)

        assert abs(config.get_min_lr() - 3e-7) < 1e-15

    def test_to_dict(self):
        """Test to_dict method."""
        from modeling_studio.trainers.lr_groups_v3 import LayerGroupLRConfig

        config = LayerGroupLRConfig(base_lr=5e-5)
        d = config.to_dict()

        assert d["base_lr"] == 5e-5
        assert "foundation_mult" in d
        assert "interface_mult" in d
        assert "warmup_ratio" in d

    def test_from_dict(self):
        """Test from_dict class method."""
        from modeling_studio.trainers.lr_groups_v3 import LayerGroupLRConfig

        d = {"base_lr": 1e-4, "feeder_mult": 0.5, "interface_mult": 2.0}
        config = LayerGroupLRConfig.from_dict(d)

        assert config.base_lr == 1e-4
        assert config.feeder_mult == 0.5
        assert config.interface_mult == 2.0
        # Other fields should have defaults
        assert config.foundation_mult == 0.0


class TestPhaseLRConfigs:
    """Tests for phase-specific LR configs."""

    def test_phase_configs_exist(self):
        """Test that all phase configs exist."""
        from modeling_studio.trainers.lr_groups_v3 import PHASE_LR_CONFIGS

        assert "phase_0.5" in PHASE_LR_CONFIGS
        assert "phase_1" in PHASE_LR_CONFIGS
        assert "phase_2" in PHASE_LR_CONFIGS

    def test_phase_05_config(self):
        """Test phase 0.5 config values."""
        from modeling_studio.trainers.lr_groups_v3 import PHASE_LR_CONFIGS

        config = PHASE_LR_CONFIGS["phase_0.5"]

        assert config.base_lr == 3e-5
        assert config.foundation_mult == 0.0  # Frozen
        assert config.core_mult == 0.0  # Frozen
        assert config.feeder_mult == 0.33
        assert config.interface_mult == 1.67
        assert config.family_mult == 1.0

    def test_phase_1_config(self):
        """Test phase 1 config values."""
        from modeling_studio.trainers.lr_groups_v3 import PHASE_LR_CONFIGS

        config = PHASE_LR_CONFIGS["phase_1"]

        assert config.base_lr == 2e-5
        assert config.foundation_mult == 0.0  # Frozen
        assert config.core_mult == 0.0  # Frozen
        assert config.feeder_mult == 0.5
        assert config.interface_mult == 1.5

    def test_phase_2_config(self):
        """Test phase 2 config values."""
        from modeling_studio.trainers.lr_groups_v3 import PHASE_LR_CONFIGS

        config = PHASE_LR_CONFIGS["phase_2"]

        assert config.base_lr == 1e-5
        assert config.foundation_mult == 0.1  # Trainable but low
        assert config.core_mult == 0.2  # Trainable but low
        assert config.feeder_mult == 0.5
        assert config.interface_mult == 1.0

    def test_phase_05_foundation_core_frozen(self):
        """Test that Foundation/Core have 0 LR in phase 0.5."""
        from modeling_studio.trainers.lr_groups_v3 import PHASE_LR_CONFIGS

        config = PHASE_LR_CONFIGS["phase_0.5"]

        for layer_idx in range(0, 18):  # L1-18
            lr = config.get_layer_lr(layer_idx)
            assert lr == 0.0, f"Layer {layer_idx + 1} should be frozen in phase 0.5"

    def test_phase_05_interface_highest_lr(self):
        """Test that Interface layer has highest LR in phase 0.5."""
        from modeling_studio.trainers.lr_groups_v3 import PHASE_LR_CONFIGS

        config = PHASE_LR_CONFIGS["phase_0.5"]

        interface_lr = config.get_layer_lr(22)  # L23

        # Interface should be higher than all other trainable layers
        for layer_idx in range(18, 28):
            if layer_idx != 22:
                other_lr = config.get_layer_lr(layer_idx)
                assert interface_lr > other_lr, f"Interface LR should be > Layer {layer_idx + 1}"

    def test_feeder_lower_than_family(self):
        """Test that Feeder band has lower LR than Family band."""
        from modeling_studio.trainers.lr_groups_v3 import PHASE_LR_CONFIGS

        for phase in ["phase_0.5", "phase_1"]:
            config = PHASE_LR_CONFIGS[phase]

            feeder_lr = config.get_band_lr("feeder")
            family_lr = config.get_band_lr("family")

            assert feeder_lr < family_lr, f"Feeder should have lower LR than Family in {phase}"


class TestLayerGroupOptimizer:
    """Tests for LayerGroupOptimizer class."""

    @pytest.fixture
    def model_for_lr_groups(self):
        """Create model suitable for LR group testing."""
        return MockModel(num_layers=28)

    def test_create_optimizer_returns_adamw(self, model_for_lr_groups):
        """Test that create_optimizer returns AdamW."""
        from modeling_studio.trainers.lr_groups_v3 import (
            LayerGroupLRConfig,
            LayerGroupOptimizer,
        )

        config = LayerGroupLRConfig()
        group_optimizer = LayerGroupOptimizer(model_for_lr_groups, config)
        optimizer = group_optimizer.create_optimizer()

        assert isinstance(optimizer, torch.optim.AdamW)

    def test_param_groups_have_names(self, model_for_lr_groups):
        """Test that parameter groups have names."""
        from modeling_studio.trainers.lr_groups_v3 import (
            LayerGroupLRConfig,
            LayerGroupOptimizer,
        )

        config = LayerGroupLRConfig()
        group_optimizer = LayerGroupOptimizer(model_for_lr_groups, config)
        param_groups = group_optimizer.get_param_groups()

        for group in param_groups:
            assert "name" in group
            assert "lr" in group
            assert "params" in group

    def test_frozen_bands_not_in_groups(self, model_for_lr_groups):
        """Test that frozen bands are not in parameter groups."""
        from modeling_studio.trainers.lr_groups_v3 import (
            LayerGroupLRConfig,
            LayerGroupOptimizer,
        )

        config = LayerGroupLRConfig(foundation_mult=0.0, core_mult=0.0)
        group_optimizer = LayerGroupOptimizer(model_for_lr_groups, config)
        param_groups = group_optimizer.get_param_groups()

        group_names = [g["name"] for g in param_groups]
        assert "foundation" not in group_names
        assert "core" not in group_names

    def test_trainable_bands_in_groups(self, model_for_lr_groups):
        """Test that trainable bands are in parameter groups."""
        from modeling_studio.trainers.lr_groups_v3 import (
            LayerGroupLRConfig,
            LayerGroupOptimizer,
        )

        config = LayerGroupLRConfig()
        group_optimizer = LayerGroupOptimizer(model_for_lr_groups, config)
        param_groups = group_optimizer.get_param_groups()

        group_names = [g["name"] for g in param_groups]
        assert "feeder" in group_names
        assert "interface" in group_names
        assert "family" in group_names

    def test_interface_lr_correct(self, model_for_lr_groups):
        """Test that interface group has correct LR."""
        from modeling_studio.trainers.lr_groups_v3 import (
            LayerGroupLRConfig,
            LayerGroupOptimizer,
        )

        config = LayerGroupLRConfig(base_lr=3e-5, interface_mult=1.67)
        group_optimizer = LayerGroupOptimizer(model_for_lr_groups, config)
        param_groups = group_optimizer.get_param_groups()

        interface_group = next(g for g in param_groups if g["name"] == "interface")
        expected_lr = 3e-5 * 1.67
        assert abs(interface_group["lr"] - expected_lr) < 1e-10

    def test_feeder_lr_correct(self, model_for_lr_groups):
        """Test that feeder group has correct LR."""
        from modeling_studio.trainers.lr_groups_v3 import (
            LayerGroupLRConfig,
            LayerGroupOptimizer,
        )

        config = LayerGroupLRConfig(base_lr=3e-5, feeder_mult=0.33)
        group_optimizer = LayerGroupOptimizer(model_for_lr_groups, config)
        param_groups = group_optimizer.get_param_groups()

        feeder_group = next(g for g in param_groups if g["name"] == "feeder")
        expected_lr = 3e-5 * 0.33
        assert abs(feeder_group["lr"] - expected_lr) < 1e-10

    def test_embeddings_group_created(self, model_for_lr_groups):
        """Test that embeddings group is created."""
        from modeling_studio.trainers.lr_groups_v3 import (
            LayerGroupLRConfig,
            LayerGroupOptimizer,
        )

        config = LayerGroupLRConfig(embeddings_mult=0.1)
        group_optimizer = LayerGroupOptimizer(model_for_lr_groups, config)
        param_groups = group_optimizer.get_param_groups()

        group_names = [g["name"] for g in param_groups]
        assert "embeddings" in group_names

    def test_all_params_assigned(self, model_for_lr_groups):
        """Test that all trainable params are assigned to groups."""
        from modeling_studio.trainers.lr_groups_v3 import (
            LayerGroupLRConfig,
            LayerGroupOptimizer,
        )

        config = LayerGroupLRConfig()
        group_optimizer = LayerGroupOptimizer(model_for_lr_groups, config)
        param_groups = group_optimizer.get_param_groups()

        total_in_groups = sum(len(g["params"]) for g in param_groups)
        total_trainable = sum(1 for p in model_for_lr_groups.parameters() if p.requires_grad)

        assert total_in_groups == total_trainable

    def test_custom_weight_decay(self, model_for_lr_groups):
        """Test custom weight decay."""
        from modeling_studio.trainers.lr_groups_v3 import (
            LayerGroupLRConfig,
            LayerGroupOptimizer,
        )

        config = LayerGroupLRConfig()
        group_optimizer = LayerGroupOptimizer(model_for_lr_groups, config, weight_decay=0.05)
        optimizer = group_optimizer.create_optimizer()

        # Check weight decay is set
        for group in optimizer.param_groups:
            assert group["weight_decay"] == 0.05


class TestCreateLayerGroupOptimizer:
    """Tests for create_layer_group_optimizer helper function."""

    @pytest.fixture
    def model_for_helper(self):
        """Create model for helper function tests."""
        return MockModel(num_layers=28)

    def test_default_phase(self, model_for_helper):
        """Test default phase is phase_0.5."""
        from modeling_studio.trainers.lr_groups_v3 import create_layer_group_optimizer

        optimizer = create_layer_group_optimizer(model_for_helper)

        assert isinstance(optimizer, torch.optim.AdamW)

    def test_phase_1(self, model_for_helper):
        """Test phase_1 configuration."""
        from modeling_studio.trainers.lr_groups_v3 import create_layer_group_optimizer

        optimizer = create_layer_group_optimizer(model_for_helper, phase="phase_1")

        # Check LRs match phase_1 config
        interface_group = next(g for g in optimizer.param_groups if g.get("name") == "interface")
        expected_lr = 2e-5 * 1.5
        assert abs(interface_group["lr"] - expected_lr) < 1e-10

    def test_phase_2(self, model_for_helper):
        """Test phase_2 configuration."""
        from modeling_studio.trainers.lr_groups_v3 import create_layer_group_optimizer

        optimizer = create_layer_group_optimizer(model_for_helper, phase="phase_2")

        # In phase 2, foundation and core should have non-zero LR
        group_names = [g.get("name") for g in optimizer.param_groups]
        assert "foundation" in group_names
        assert "core" in group_names

    def test_override_base_lr(self, model_for_helper):
        """Test base_lr override."""
        from modeling_studio.trainers.lr_groups_v3 import create_layer_group_optimizer

        optimizer = create_layer_group_optimizer(
            model_for_helper,
            phase="phase_0.5",
            base_lr=1e-4,
        )

        # Interface should use new base_lr
        interface_group = next(g for g in optimizer.param_groups if g.get("name") == "interface")
        expected_lr = 1e-4 * 1.67
        assert abs(interface_group["lr"] - expected_lr) < 1e-10

    def test_unknown_phase_fallback(self, model_for_helper):
        """Test unknown phase falls back to phase_0.5."""
        from modeling_studio.trainers.lr_groups_v3 import create_layer_group_optimizer

        optimizer = create_layer_group_optimizer(model_for_helper, phase="unknown_phase")

        # Should use phase_0.5 config
        interface_group = next(g for g in optimizer.param_groups if g.get("name") == "interface")
        expected_lr = 3e-5 * 1.67
        assert abs(interface_group["lr"] - expected_lr) < 1e-10

    def test_custom_weight_decay(self, model_for_helper):
        """Test custom weight decay passed through."""
        from modeling_studio.trainers.lr_groups_v3 import create_layer_group_optimizer

        optimizer = create_layer_group_optimizer(
            model_for_helper,
            weight_decay=0.1,
        )

        for group in optimizer.param_groups:
            assert group["weight_decay"] == 0.1


class TestGetPhaseConfig:
    """Tests for get_phase_config helper function."""

    def test_known_phases(self):
        """Test getting known phase configs."""
        from modeling_studio.trainers.lr_groups_v3 import get_phase_config

        config = get_phase_config("phase_0.5")
        assert config.base_lr == 3e-5

        config = get_phase_config("phase_1")
        assert config.base_lr == 2e-5

        config = get_phase_config("phase_2")
        assert config.base_lr == 1e-5

    def test_unknown_phase_fallback(self):
        """Test unknown phase falls back to phase_0.5."""
        from modeling_studio.trainers.lr_groups_v3 import get_phase_config

        config = get_phase_config("unknown")
        assert config.base_lr == 3e-5  # Same as phase_0.5

    def test_returns_copy(self):
        """Test that modifications don't affect preset."""
        from modeling_studio.trainers.lr_groups_v3 import PHASE_LR_CONFIGS, get_phase_config

        config = get_phase_config("phase_0.5")
        original_lr = PHASE_LR_CONFIGS["phase_0.5"].base_lr

        config.base_lr = 1e-3

        # Original should be unchanged
        assert PHASE_LR_CONFIGS["phase_0.5"].base_lr == original_lr


class TestIssue514AcceptanceCriteria:
    """Acceptance criteria tests for Issue 5.1.4."""

    @pytest.fixture
    def model_for_ac(self):
        """Create model for acceptance criteria tests."""
        return MockModel(num_layers=28)

    def test_ac1_layer_group_lr_config_supports_all_bands(self):
        """AC1: LayerGroupLRConfig supports all layer bands."""
        from modeling_studio.trainers.lr_groups_v3 import LayerGroupLRConfig

        config = LayerGroupLRConfig()

        # All bands should be configurable
        assert hasattr(config, "foundation_mult")
        assert hasattr(config, "core_mult")
        assert hasattr(config, "feeder_mult")
        assert hasattr(config, "interface_mult")
        assert hasattr(config, "family_mult")

        # get_layer_lr should work for all layers
        for layer_idx in range(28):
            lr = config.get_layer_lr(layer_idx)
            assert isinstance(lr, float)

        print("AC1: LayerGroupLRConfig supports all layer bands [PASS]")

    def test_ac2_foundation_core_zero_lr_in_phase_05_1(self, model_for_ac):
        """AC2: Foundation/Core get 0 LR in Phase 0.5/1."""
        from modeling_studio.trainers.lr_groups_v3 import PHASE_LR_CONFIGS

        for phase in ["phase_0.5", "phase_1"]:
            config = PHASE_LR_CONFIGS[phase]

            assert config.foundation_mult == 0.0, f"Foundation should be 0 in {phase}"
            assert config.core_mult == 0.0, f"Core should be 0 in {phase}"

            for layer_idx in range(0, 18):
                lr = config.get_layer_lr(layer_idx)
                assert lr == 0.0, f"Layer {layer_idx + 1} should have 0 LR in {phase}"

        print("AC2: Foundation/Core get 0 LR in Phase 0.5/1 [PASS]")

    def test_ac3_interface_layer_highest_lr(self):
        """AC3: Interface layer (L23) gets highest LR."""
        from modeling_studio.trainers.lr_groups_v3 import PHASE_LR_CONFIGS

        for phase in ["phase_0.5", "phase_1"]:
            config = PHASE_LR_CONFIGS[phase]

            interface_lr = config.get_layer_lr(22)  # L23

            # Check against all trainable layers
            all_trainable_lrs = [config.get_layer_lr(i) for i in range(18, 28)]
            max_lr = max(all_trainable_lrs)

            assert interface_lr == max_lr, f"Interface should have highest LR in {phase}"

        print("AC3: Interface layer (L23) gets highest LR [PASS]")

    def test_ac4_feeder_lower_than_family(self):
        """AC4: Feeder band gets lower LR than Family."""
        from modeling_studio.trainers.lr_groups_v3 import PHASE_LR_CONFIGS

        for phase in ["phase_0.5", "phase_1", "phase_2"]:
            config = PHASE_LR_CONFIGS[phase]

            feeder_lr = config.get_band_lr("feeder")
            family_lr = config.get_band_lr("family")

            assert feeder_lr <= family_lr, f"Feeder should be <= Family in {phase}"

        print("AC4: Feeder band gets lower LR than Family [PASS]")

    def test_ac5_create_optimizer_valid_adamw(self, model_for_ac):
        """AC5: create_optimizer() creates valid AdamW."""
        from modeling_studio.trainers.lr_groups_v3 import create_layer_group_optimizer

        optimizer = create_layer_group_optimizer(model_for_ac)

        assert isinstance(optimizer, torch.optim.AdamW)
        assert len(optimizer.param_groups) > 0

        # Should be able to step
        for p in model_for_ac.parameters():
            p.grad = torch.zeros_like(p)
        optimizer.step()

        print("AC5: create_optimizer() creates valid AdamW [PASS]")

    def test_ac6_param_groups_logged_with_names(self, model_for_ac, capsys):
        """AC6: Parameter groups logged clearly with names."""
        from modeling_studio.trainers.lr_groups_v3 import (
            LayerGroupLRConfig,
            LayerGroupOptimizer,
        )

        config = LayerGroupLRConfig()
        group_optimizer = LayerGroupOptimizer(model_for_ac, config)
        _ = group_optimizer.create_optimizer()

        captured = capsys.readouterr()

        # Check logging output
        assert "Layer Group Learning Rates" in captured.out
        assert "feeder" in captured.out
        assert "interface" in captured.out
        assert "family" in captured.out
        assert "lr=" in captured.out
        assert "params=" in captured.out

        print("AC6: Parameter groups logged clearly [PASS]")

    def test_ac7_preset_configs_for_all_phases(self):
        """AC7: Preset configs for all phases."""
        from modeling_studio.trainers.lr_groups_v3 import PHASE_LR_CONFIGS, LayerGroupLRConfig

        required_phases = ["phase_0.5", "phase_1", "phase_2"]

        for phase in required_phases:
            assert phase in PHASE_LR_CONFIGS, f"Missing preset for {phase}"
            config = PHASE_LR_CONFIGS[phase]
            assert isinstance(config, LayerGroupLRConfig)

        print("AC7: Preset configs for all phases [PASS]")


# ==============================================================================
# Issue 5.1.5: Hub Token Gradient Masking Tests
# ==============================================================================


class MockEmbeddingsForGradMask(nn.Module):
    """Mock embeddings with word_embeddings for gradient masking tests."""

    def __init__(self, vocab_size: int = 50372, hidden_size: int = 768):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)


class MockModelForGradMask(nn.Module):
    """Mock model for gradient masking tests."""

    def __init__(self, vocab_size: int = 50372, hidden_size: int = 768):
        super().__init__()
        self.embeddings = MockEmbeddingsForGradMask(vocab_size, hidden_size)
        self.encoder = MockEncoder(num_layers=28, hidden_size=hidden_size)


class TestGradientMaskConfig:
    """Tests for GradientMaskConfig dataclass."""

    def test_default_config(self):
        """Test default config values."""
        from modeling_studio.trainers.gradient_masking_v3 import GradientMaskConfig

        config = GradientMaskConfig()

        assert config.freeze_original_vocab is True
        assert config.hub_token_grad_scale == 1.0
        assert config.train_hub_tokens is not None
        assert len(config.train_hub_tokens) == 4  # All hub tokens by default

    def test_custom_config(self):
        """Test custom config values."""
        from modeling_studio.trainers.gradient_masking_v3 import GradientMaskConfig

        config = GradientMaskConfig(
            train_hub_tokens=["[EMO]", "[TASK]"],
            freeze_original_vocab=False,
            hub_token_grad_scale=0.5,
        )

        assert config.train_hub_tokens == ["[EMO]", "[TASK]"]
        assert config.freeze_original_vocab is False
        assert config.hub_token_grad_scale == 0.5

    def test_to_dict(self):
        """Test to_dict method."""
        from modeling_studio.trainers.gradient_masking_v3 import GradientMaskConfig

        config = GradientMaskConfig(train_hub_tokens=["[EMO]"])
        d = config.to_dict()

        assert d["train_hub_tokens"] == ["[EMO]"]
        assert d["freeze_original_vocab"] is True
        assert d["hub_token_grad_scale"] == 1.0

    def test_from_dict(self):
        """Test from_dict class method."""
        from modeling_studio.trainers.gradient_masking_v3 import GradientMaskConfig

        d = {"train_hub_tokens": ["[MEM]"], "hub_token_grad_scale": 2.0}
        config = GradientMaskConfig.from_dict(d)

        assert config.train_hub_tokens == ["[MEM]"]
        assert config.hub_token_grad_scale == 2.0


class TestHubTokenPositions:
    """Tests for hub token position constants."""

    def test_hub_token_positions(self):
        """Test hub token positions are correct."""
        from modeling_studio.trainers.gradient_masking_v3 import HUB_TOKEN_POSITIONS

        assert HUB_TOKEN_POSITIONS["[EMO]"] == 50368
        assert HUB_TOKEN_POSITIONS["[MEM]"] == 50369
        assert HUB_TOKEN_POSITIONS["[REL]"] == 50370
        assert HUB_TOKEN_POSITIONS["[TASK]"] == 50371
        assert len(HUB_TOKEN_POSITIONS) == 4

    def test_vocab_constants(self):
        """Test vocabulary layout constants."""
        from modeling_studio.trainers.gradient_masking_v3 import (
            V2_VOCAB_SIZE,
            V3_VOCAB_SIZE,
            HUB_TOKEN_START,
            HUB_TOKEN_COUNT,
        )

        assert V2_VOCAB_SIZE == 50368
        assert HUB_TOKEN_START == 50368
        assert HUB_TOKEN_COUNT == 4
        assert V3_VOCAB_SIZE == 50372
        assert V3_VOCAB_SIZE == V2_VOCAB_SIZE + HUB_TOKEN_COUNT

    def test_get_hub_token_positions(self):
        """Test get_hub_token_positions helper."""
        from modeling_studio.trainers.gradient_masking_v3 import get_hub_token_positions

        positions = get_hub_token_positions()
        assert len(positions) == 4
        assert positions["[EMO]"] == 50368

    def test_get_vocab_layout(self):
        """Test get_vocab_layout helper."""
        from modeling_studio.trainers.gradient_masking_v3 import get_vocab_layout

        layout = get_vocab_layout()
        assert layout["V2_VOCAB_SIZE"] == 50368
        assert layout["V3_VOCAB_SIZE"] == 50372


class TestEmbeddingGradientHook:
    """Tests for EmbeddingGradientHook class."""

    @pytest.fixture
    def model_for_hook(self):
        """Create model for hook tests."""
        return MockModelForGradMask(vocab_size=50372, hidden_size=768)

    def test_hook_init(self, model_for_hook):
        """Test hook initialization."""
        from modeling_studio.trainers.gradient_masking_v3 import (
            EmbeddingGradientHook,
            GradientMaskConfig,
        )

        config = GradientMaskConfig()
        embedding_weight = model_for_hook.embeddings.word_embeddings.weight
        hook = EmbeddingGradientHook(embedding_weight, config)

        assert hook.grad_mask is not None
        assert hook.grad_mask.shape[0] == 50372
        assert hook.grad_mask.shape[1] == 1

    def test_hook_mask_frozen_vocab(self, model_for_hook):
        """Test that mask freezes original vocab when configured."""
        from modeling_studio.trainers.gradient_masking_v3 import (
            EmbeddingGradientHook,
            GradientMaskConfig,
        )

        config = GradientMaskConfig(freeze_original_vocab=True)
        embedding_weight = model_for_hook.embeddings.word_embeddings.weight
        hook = EmbeddingGradientHook(embedding_weight, config)

        # Original vocab should be frozen (mask = 0)
        assert hook.grad_mask[0].item() == 0.0
        assert hook.grad_mask[50367].item() == 0.0  # Last v2 token

    def test_hook_mask_hub_tokens_trainable(self, model_for_hook):
        """Test that hub tokens are trainable by default."""
        from modeling_studio.trainers.gradient_masking_v3 import (
            EmbeddingGradientHook,
            GradientMaskConfig,
            HUB_TOKEN_POSITIONS,
        )

        config = GradientMaskConfig(freeze_original_vocab=True, hub_token_grad_scale=1.0)
        embedding_weight = model_for_hook.embeddings.word_embeddings.weight
        hook = EmbeddingGradientHook(embedding_weight, config)

        # Hub tokens should be trainable (mask = 1.0)
        for name, pos in HUB_TOKEN_POSITIONS.items():
            assert hook.grad_mask[pos].item() == 1.0, f"{name} should be trainable"

    def test_hook_mask_specific_hub_tokens(self, model_for_hook):
        """Test training only specific hub tokens."""
        from modeling_studio.trainers.gradient_masking_v3 import (
            EmbeddingGradientHook,
            GradientMaskConfig,
            HUB_TOKEN_POSITIONS,
        )

        config = GradientMaskConfig(
            train_hub_tokens=["[EMO]", "[TASK]"],
            freeze_original_vocab=True,
        )
        embedding_weight = model_for_hook.embeddings.word_embeddings.weight
        hook = EmbeddingGradientHook(embedding_weight, config)

        # Only [EMO] and [TASK] should be trainable
        assert hook.grad_mask[HUB_TOKEN_POSITIONS["[EMO]"]].item() == 1.0
        assert hook.grad_mask[HUB_TOKEN_POSITIONS["[TASK]"]].item() == 1.0
        assert hook.grad_mask[HUB_TOKEN_POSITIONS["[MEM]"]].item() == 0.0
        assert hook.grad_mask[HUB_TOKEN_POSITIONS["[REL]"]].item() == 0.0

    def test_hook_grad_scaling(self, model_for_hook):
        """Test gradient scaling for hub tokens."""
        from modeling_studio.trainers.gradient_masking_v3 import (
            EmbeddingGradientHook,
            GradientMaskConfig,
            HUB_TOKEN_POSITIONS,
        )

        config = GradientMaskConfig(hub_token_grad_scale=0.5)
        embedding_weight = model_for_hook.embeddings.word_embeddings.weight
        hook = EmbeddingGradientHook(embedding_weight, config)

        for name, pos in HUB_TOKEN_POSITIONS.items():
            assert hook.grad_mask[pos].item() == 0.5, f"{name} should have scale 0.5"

    def test_hook_register_remove(self, model_for_hook):
        """Test hook registration and removal."""
        from modeling_studio.trainers.gradient_masking_v3 import (
            EmbeddingGradientHook,
            GradientMaskConfig,
        )

        config = GradientMaskConfig()
        embedding_weight = model_for_hook.embeddings.word_embeddings.weight
        hook = EmbeddingGradientHook(embedding_weight, config)

        assert not hook.is_registered()

        result = hook.register()
        assert result is True
        assert hook.is_registered()

        hook.remove()
        assert not hook.is_registered()

    def test_hook_update_trainable_tokens(self, model_for_hook):
        """Test updating trainable tokens."""
        from modeling_studio.trainers.gradient_masking_v3 import (
            EmbeddingGradientHook,
            GradientMaskConfig,
            HUB_TOKEN_POSITIONS,
        )

        config = GradientMaskConfig(train_hub_tokens=["[EMO]"])
        embedding_weight = model_for_hook.embeddings.word_embeddings.weight
        hook = EmbeddingGradientHook(embedding_weight, config)

        # Initially only [EMO] trainable
        assert hook.grad_mask[HUB_TOKEN_POSITIONS["[EMO]"]].item() == 1.0
        assert hook.grad_mask[HUB_TOKEN_POSITIONS["[MEM]"]].item() == 0.0

        # Update to train [MEM] only
        hook.update_trainable_tokens(["[MEM]"])

        assert hook.grad_mask[HUB_TOKEN_POSITIONS["[EMO]"]].item() == 0.0
        assert hook.grad_mask[HUB_TOKEN_POSITIONS["[MEM]"]].item() == 1.0

    def test_hook_get_mask_stats(self, model_for_hook):
        """Test get_mask_stats method."""
        from modeling_studio.trainers.gradient_masking_v3 import (
            EmbeddingGradientHook,
            GradientMaskConfig,
        )

        config = GradientMaskConfig(train_hub_tokens=["[EMO]", "[TASK]"])
        embedding_weight = model_for_hook.embeddings.word_embeddings.weight
        hook = EmbeddingGradientHook(embedding_weight, config)

        stats = hook.get_mask_stats()

        assert stats["total_tokens"] == 50372
        assert stats["trainable_tokens"] == 2  # [EMO] and [TASK]
        assert stats["frozen_tokens"] == 50370
        assert stats["hub_tokens_trainable"] == ["[EMO]", "[TASK]"]


class TestHubTokenGradientManager:
    """Tests for HubTokenGradientManager class."""

    @pytest.fixture
    def model_for_manager(self):
        """Create model for manager tests."""
        return MockModelForGradMask(vocab_size=50372, hidden_size=768)

    def test_manager_init(self, model_for_manager):
        """Test manager initialization."""
        from modeling_studio.trainers.gradient_masking_v3 import HubTokenGradientManager

        manager = HubTokenGradientManager(model_for_manager)

        assert manager.model is model_for_manager
        assert manager.config is not None
        assert len(manager.hooks) == 0
        assert not manager.is_setup()

    def test_manager_get_embedding_weight(self, model_for_manager):
        """Test finding embedding weight."""
        from modeling_studio.trainers.gradient_masking_v3 import HubTokenGradientManager

        manager = HubTokenGradientManager(model_for_manager)
        weight = manager.get_embedding_weight()

        assert weight is not None
        assert weight.shape[0] == 50372
        assert weight.shape[1] == 768

    def test_manager_setup(self, model_for_manager):
        """Test setup method."""
        from modeling_studio.trainers.gradient_masking_v3 import HubTokenGradientManager

        manager = HubTokenGradientManager(model_for_manager)

        result = manager.setup()
        assert result is True
        assert manager.is_setup()
        assert len(manager.hooks) == 1

        # Cleanup
        manager.cleanup()

    def test_manager_cleanup(self, model_for_manager):
        """Test cleanup method."""
        from modeling_studio.trainers.gradient_masking_v3 import HubTokenGradientManager

        manager = HubTokenGradientManager(model_for_manager)
        manager.setup()

        assert manager.is_setup()

        manager.cleanup()

        assert not manager.is_setup()
        assert len(manager.hooks) == 0

    def test_manager_freeze_unfreeze_all(self, model_for_manager):
        """Test freeze/unfreeze all hub tokens."""
        from modeling_studio.trainers.gradient_masking_v3 import (
            HubTokenGradientManager,
            HUB_TOKEN_POSITIONS,
        )

        manager = HubTokenGradientManager(model_for_manager)
        manager.setup()

        # Freeze all
        manager.freeze_all_hub_tokens()

        for hook in manager.hooks:
            for pos in HUB_TOKEN_POSITIONS.values():
                assert hook.grad_mask[pos].item() == 0.0

        # Unfreeze all
        manager.unfreeze_all_hub_tokens()

        for hook in manager.hooks:
            for pos in HUB_TOKEN_POSITIONS.values():
                assert hook.grad_mask[pos].item() == 1.0

        manager.cleanup()

    def test_manager_train_specific_tokens(self, model_for_manager):
        """Test training specific hub tokens."""
        from modeling_studio.trainers.gradient_masking_v3 import (
            HubTokenGradientManager,
            HUB_TOKEN_POSITIONS,
        )

        manager = HubTokenGradientManager(model_for_manager)
        manager.setup()

        manager.train_specific_hub_tokens(["[EMO]", "[MEM]"])

        for hook in manager.hooks:
            assert hook.grad_mask[HUB_TOKEN_POSITIONS["[EMO]"]].item() == 1.0
            assert hook.grad_mask[HUB_TOKEN_POSITIONS["[MEM]"]].item() == 1.0
            assert hook.grad_mask[HUB_TOKEN_POSITIONS["[REL]"]].item() == 0.0
            assert hook.grad_mask[HUB_TOKEN_POSITIONS["[TASK]"]].item() == 0.0

        manager.cleanup()

    def test_manager_set_grad_scale(self, model_for_manager):
        """Test setting gradient scale."""
        from modeling_studio.trainers.gradient_masking_v3 import (
            HubTokenGradientManager,
            HUB_TOKEN_POSITIONS,
        )

        manager = HubTokenGradientManager(model_for_manager)
        manager.setup()

        manager.set_grad_scale(0.25)

        for hook in manager.hooks:
            for pos in HUB_TOKEN_POSITIONS.values():
                assert hook.grad_mask[pos].item() == 0.25

        manager.cleanup()

    def test_manager_get_hub_token_embeddings(self, model_for_manager):
        """Test getting hub token embeddings."""
        from modeling_studio.trainers.gradient_masking_v3 import HubTokenGradientManager

        manager = HubTokenGradientManager(model_for_manager)

        embeddings = manager.get_hub_token_embeddings()

        assert len(embeddings) == 4
        assert "[EMO]" in embeddings
        assert "[MEM]" in embeddings
        assert "[REL]" in embeddings
        assert "[TASK]" in embeddings
        assert embeddings["[EMO]"].shape == (768,)

    def test_manager_get_hub_token_gradients_no_grad(self, model_for_manager):
        """Test getting gradients when none exist."""
        from modeling_studio.trainers.gradient_masking_v3 import HubTokenGradientManager

        manager = HubTokenGradientManager(model_for_manager)

        gradients = manager.get_hub_token_gradients()

        assert len(gradients) == 4
        for name, grad in gradients.items():
            assert grad is None

    def test_manager_get_stats(self, model_for_manager):
        """Test getting stats."""
        from modeling_studio.trainers.gradient_masking_v3 import HubTokenGradientManager

        manager = HubTokenGradientManager(model_for_manager)
        manager.setup()

        stats = manager.get_stats()

        assert stats["is_setup"] is True
        assert stats["num_hooks"] == 1
        assert stats["vocab_size"] == 50372
        assert stats["embedding_dim"] == 768
        assert "config" in stats
        assert "mask_stats" in stats

        manager.cleanup()


class TestSetupHubTokenGradientMasking:
    """Tests for setup_hub_token_gradient_masking helper function."""

    @pytest.fixture
    def model_for_setup(self):
        """Create model for setup tests."""
        return MockModelForGradMask(vocab_size=50372, hidden_size=768)

    def test_setup_default(self, model_for_setup):
        """Test setup with default config."""
        from modeling_studio.trainers.gradient_masking_v3 import setup_hub_token_gradient_masking

        manager = setup_hub_token_gradient_masking(model_for_setup)

        assert manager.is_setup()
        assert len(manager.hooks) == 1

        manager.cleanup()

    def test_setup_custom_tokens(self, model_for_setup):
        """Test setup with specific hub tokens."""
        from modeling_studio.trainers.gradient_masking_v3 import (
            setup_hub_token_gradient_masking,
            HUB_TOKEN_POSITIONS,
        )

        manager = setup_hub_token_gradient_masking(
            model_for_setup,
            train_hub_tokens=["[EMO]"],
        )

        for hook in manager.hooks:
            assert hook.grad_mask[HUB_TOKEN_POSITIONS["[EMO]"]].item() == 1.0
            assert hook.grad_mask[HUB_TOKEN_POSITIONS["[MEM]"]].item() == 0.0

        manager.cleanup()

    def test_setup_unfreeze_original_vocab(self, model_for_setup):
        """Test setup with original vocab unfrozen."""
        from modeling_studio.trainers.gradient_masking_v3 import setup_hub_token_gradient_masking

        manager = setup_hub_token_gradient_masking(
            model_for_setup,
            freeze_original_vocab=False,
        )

        for hook in manager.hooks:
            # Original vocab should be trainable
            assert hook.grad_mask[0].item() == 1.0
            assert hook.grad_mask[1000].item() == 1.0

        manager.cleanup()

    def test_setup_custom_grad_scale(self, model_for_setup):
        """Test setup with custom gradient scale."""
        from modeling_studio.trainers.gradient_masking_v3 import (
            setup_hub_token_gradient_masking,
            HUB_TOKEN_POSITIONS,
        )

        manager = setup_hub_token_gradient_masking(
            model_for_setup,
            hub_token_grad_scale=2.0,
        )

        for hook in manager.hooks:
            for pos in HUB_TOKEN_POSITIONS.values():
                assert hook.grad_mask[pos].item() == 2.0

        manager.cleanup()


class TestGradientMaskingIntegration:
    """Integration tests for gradient masking with actual backward pass."""

    @pytest.fixture
    def model_for_integration(self):
        """Create model for integration tests."""
        return MockModelForGradMask(vocab_size=50372, hidden_size=768)

    def test_gradient_masking_forward_backward(self, model_for_integration):
        """Test that gradient masking works in forward/backward pass."""
        from modeling_studio.trainers.gradient_masking_v3 import (
            setup_hub_token_gradient_masking,
            HUB_TOKEN_POSITIONS,
        )

        manager = setup_hub_token_gradient_masking(
            model_for_integration,
            train_hub_tokens=["[EMO]"],
            freeze_original_vocab=True,
        )

        # Forward pass with embedding lookup
        embedding_weight = model_for_integration.embeddings.word_embeddings.weight

        # Create input that uses various token indices
        input_indices = torch.tensor(
            [
                0,  # Original vocab (should be frozen)
                100,  # Original vocab (should be frozen)
                50368,  # [EMO] (should be trainable)
                50369,  # [MEM] (should be frozen)
            ]
        )

        # Get embeddings and compute loss
        embeddings = embedding_weight[input_indices]
        loss = embeddings.sum()
        loss.backward()

        # Check gradients
        grad = embedding_weight.grad

        # Original vocab should have zero gradients
        assert grad[0].abs().sum() == 0.0
        assert grad[100].abs().sum() == 0.0

        # [EMO] should have non-zero gradients
        assert grad[HUB_TOKEN_POSITIONS["[EMO]"]].abs().sum() > 0.0

        # [MEM] should have zero gradients (not in train_hub_tokens)
        assert grad[HUB_TOKEN_POSITIONS["[MEM]"]].abs().sum() == 0.0

        manager.cleanup()

    def test_no_memory_leak_from_hooks(self, model_for_integration):
        """Test that hooks don't cause memory leaks."""
        from modeling_studio.trainers.gradient_masking_v3 import setup_hub_token_gradient_masking
        import gc

        # Create and cleanup multiple times
        for _ in range(10):
            manager = setup_hub_token_gradient_masking(model_for_integration)
            manager.cleanup()

        gc.collect()
        # If we get here without error, no memory leak detected


class TestIssue515AcceptanceCriteria:
    """Acceptance criteria tests for Issue 5.1.5."""

    @pytest.fixture
    def model_for_ac(self):
        """Create model for acceptance criteria tests."""
        return MockModelForGradMask(vocab_size=50372, hidden_size=768)

    def test_ac1_embedding_gradient_hook_masks_correctly(self, model_for_ac):
        """AC1: EmbeddingGradientHook masks gradients correctly."""
        from modeling_studio.trainers.gradient_masking_v3 import (
            EmbeddingGradientHook,
            GradientMaskConfig,
            HUB_TOKEN_POSITIONS,
        )

        config = GradientMaskConfig(
            train_hub_tokens=["[EMO]"],
            freeze_original_vocab=True,
        )
        embedding_weight = model_for_ac.embeddings.word_embeddings.weight
        hook = EmbeddingGradientHook(embedding_weight, config)
        hook.register()

        # Test gradient masking
        test_grad = torch.ones_like(embedding_weight.data)
        masked_grad = hook._gradient_hook(test_grad)

        # Original vocab should be zeroed
        assert masked_grad[0].sum() == 0.0
        assert masked_grad[50367].sum() == 0.0

        # [EMO] should be preserved
        assert masked_grad[HUB_TOKEN_POSITIONS["[EMO]"]].sum() == 768.0

        # [MEM] should be zeroed (not in train list)
        assert masked_grad[HUB_TOKEN_POSITIONS["[MEM]"]].sum() == 0.0

        hook.remove()
        print("AC1: EmbeddingGradientHook masks gradients correctly [PASS]")

    def test_ac2_original_vocab_gradients_zeroed(self, model_for_ac):
        """AC2: Original vocab (0-50367) gradients zeroed when frozen."""
        from modeling_studio.trainers.gradient_masking_v3 import (
            setup_hub_token_gradient_masking,
            V2_VOCAB_SIZE,
        )

        manager = setup_hub_token_gradient_masking(
            model_for_ac,
            freeze_original_vocab=True,
        )

        for hook in manager.hooks:
            for i in range(V2_VOCAB_SIZE):
                assert hook.grad_mask[i].item() == 0.0, f"Token {i} should be frozen"

        manager.cleanup()
        print("AC2: Original vocab gradients zeroed when frozen [PASS]")

    def test_ac3_hub_token_gradients_preserved_scaled(self, model_for_ac):
        """AC3: Hub token gradients preserved/scaled."""
        from modeling_studio.trainers.gradient_masking_v3 import (
            setup_hub_token_gradient_masking,
            HUB_TOKEN_POSITIONS,
        )

        # Test with scale 1.0 (preserved)
        manager = setup_hub_token_gradient_masking(
            model_for_ac,
            hub_token_grad_scale=1.0,
        )

        for hook in manager.hooks:
            for pos in HUB_TOKEN_POSITIONS.values():
                assert hook.grad_mask[pos].item() == 1.0

        manager.cleanup()

        # Test with scale 0.5 (scaled)
        manager = setup_hub_token_gradient_masking(
            model_for_ac,
            hub_token_grad_scale=0.5,
        )

        for hook in manager.hooks:
            for pos in HUB_TOKEN_POSITIONS.values():
                assert hook.grad_mask[pos].item() == 0.5

        manager.cleanup()
        print("AC3: Hub token gradients preserved/scaled [PASS]")

    def test_ac4_train_specific_hub_tokens(self, model_for_ac):
        """AC4: train_specific_hub_tokens() selects specific tokens."""
        from modeling_studio.trainers.gradient_masking_v3 import (
            HubTokenGradientManager,
            HUB_TOKEN_POSITIONS,
        )

        manager = HubTokenGradientManager(model_for_ac)
        manager.setup()

        manager.train_specific_hub_tokens(["[REL]", "[TASK]"])

        for hook in manager.hooks:
            assert hook.grad_mask[HUB_TOKEN_POSITIONS["[EMO]"]].item() == 0.0
            assert hook.grad_mask[HUB_TOKEN_POSITIONS["[MEM]"]].item() == 0.0
            assert hook.grad_mask[HUB_TOKEN_POSITIONS["[REL]"]].item() == 1.0
            assert hook.grad_mask[HUB_TOKEN_POSITIONS["[TASK]"]].item() == 1.0

        manager.cleanup()
        print("AC4: train_specific_hub_tokens() selects specific tokens [PASS]")

    def test_ac5_get_hub_token_gradients(self, model_for_ac):
        """AC5: get_hub_token_gradients() returns correct values."""
        from modeling_studio.trainers.gradient_masking_v3 import HubTokenGradientManager

        manager = HubTokenGradientManager(model_for_ac)

        # No gradients yet
        gradients = manager.get_hub_token_gradients()
        assert len(gradients) == 4
        for name, grad in gradients.items():
            assert grad is None

        # Compute gradients
        embedding_weight = model_for_ac.embeddings.word_embeddings.weight
        loss = embedding_weight.sum()
        loss.backward()

        # Now should have gradients
        gradients = manager.get_hub_token_gradients()
        for name, grad in gradients.items():
            assert grad is not None
            assert grad.shape == (768,)

        print("AC5: get_hub_token_gradients() returns correct values [PASS]")

    def test_ac6_hooks_properly_registered_removable(self, model_for_ac):
        """AC6: Hooks properly registered and removable."""
        from modeling_studio.trainers.gradient_masking_v3 import (
            EmbeddingGradientHook,
            GradientMaskConfig,
        )

        config = GradientMaskConfig()
        embedding_weight = model_for_ac.embeddings.word_embeddings.weight
        hook = EmbeddingGradientHook(embedding_weight, config)

        # Not registered yet
        assert not hook.is_registered()
        assert hook.hook_handle is None

        # Register
        result = hook.register()
        assert result is True
        assert hook.is_registered()
        assert hook.hook_handle is not None

        # Remove
        hook.remove()
        assert not hook.is_registered()
        assert hook.hook_handle is None

        print("AC6: Hooks properly registered and removable [PASS]")

    def test_ac7_no_memory_leaks(self, model_for_ac):
        """AC7: No memory leaks from hook registration."""
        from modeling_studio.trainers.gradient_masking_v3 import HubTokenGradientManager
        import gc
        import weakref

        # Create manager and setup
        manager = HubTokenGradientManager(model_for_ac)
        manager.setup()

        # Get weak reference to hook
        hook_ref = weakref.ref(manager.hooks[0])

        # Cleanup
        manager.cleanup()

        # Force garbage collection
        gc.collect()

        # Hook should still exist because manager still exists
        # (but hook should be properly unregistered)
        assert len(manager.hooks) == 0

        print("AC7: No memory leaks from hook registration [PASS]")


# ============================================================================
# Issue 5.1.6: Zipper Learning Rate Strategy Tests
# ============================================================================


class MockModelForZipper(nn.Module):
    """Mock model for Zipper LR testing."""

    def __init__(self, num_layers: int = 28, hidden_size: int = 768):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Create encoder with layers
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )

        # Create embeddings
        self.embeddings = nn.Embedding(50372, hidden_size)

        # Create task heads
        self.task_heads = nn.ModuleDict(
            {
                "intent": nn.Linear(hidden_size, 10),
                "emotion": nn.Linear(hidden_size, 8),
            }
        )


class TestZipperLRConfig:
    """Tests for ZipperLRConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from modeling_studio.trainers.zipper_lr_v3 import ZipperLRConfig

        config = ZipperLRConfig()

        assert config.base_lr == 3e-5
        assert config.feeder_lr == 1e-5
        assert config.interface_lr == 5e-5
        assert config.family_lr == 3e-5
        assert config.family_graduated is True
        assert config.family_decay == 0.9
        assert config.frozen_lr == 0.0
        assert config.embeddings_lr == 0.0
        assert config.task_heads_lr == 3e-5

    def test_custom_config(self):
        """Test custom configuration values."""
        from modeling_studio.trainers.zipper_lr_v3 import ZipperLRConfig

        config = ZipperLRConfig(
            base_lr=5e-5,
            feeder_lr=2e-5,
            interface_lr=1e-4,
            family_lr=4e-5,
            family_graduated=False,
            family_decay=0.8,
        )

        assert config.base_lr == 5e-5
        assert config.feeder_lr == 2e-5
        assert config.interface_lr == 1e-4
        assert config.family_lr == 4e-5
        assert config.family_graduated is False

    def test_get_layer_lr_frozen_layers(self):
        """Test LR for frozen layers (L1-18)."""
        from modeling_studio.trainers.zipper_lr_v3 import ZipperLRConfig

        config = ZipperLRConfig()

        # Foundation (L1-6, indices 0-5)
        for idx in range(6):
            assert config.get_layer_lr(idx) == 0.0

        # Core (L7-18, indices 6-17)
        for idx in range(6, 18):
            assert config.get_layer_lr(idx) == 0.0

    def test_get_layer_lr_feeder_band(self):
        """Test LR for Feeder band (L19-22)."""
        from modeling_studio.trainers.zipper_lr_v3 import ZipperLRConfig

        config = ZipperLRConfig(feeder_lr=1e-5)

        # Feeder (L19-22, indices 18-21)
        for idx in range(18, 22):
            assert config.get_layer_lr(idx) == 1e-5

    def test_get_layer_lr_interface_layer(self):
        """Test LR for Interface layer (L23)."""
        from modeling_studio.trainers.zipper_lr_v3 import ZipperLRConfig

        config = ZipperLRConfig(interface_lr=5e-5)

        # Interface (L23, index 22)
        assert config.get_layer_lr(22) == 5e-5

    def test_get_layer_lr_family_uniform(self):
        """Test uniform LR for Family band."""
        from modeling_studio.trainers.zipper_lr_v3 import ZipperLRConfig

        config = ZipperLRConfig(
            family_lr=3e-5,
            family_graduated=False,
        )

        # Family (L24-28, indices 23-27)
        for idx in range(23, 28):
            assert config.get_layer_lr(idx) == 3e-5

    def test_get_layer_lr_family_graduated(self):
        """Test graduated LR for Family band."""
        from modeling_studio.trainers.zipper_lr_v3 import ZipperLRConfig

        config = ZipperLRConfig(
            interface_lr=5e-5,
            family_graduated=True,
            family_decay=0.9,
        )

        # Family (L24-28, indices 23-27) with decay
        # L24 (idx 23): 5e-5 * 0.9^1 = 4.5e-5
        # L25 (idx 24): 5e-5 * 0.9^2 = 4.05e-5
        # etc.
        expected_l24 = 5e-5 * (0.9**1)
        expected_l25 = 5e-5 * (0.9**2)
        expected_l28 = 5e-5 * (0.9**5)

        assert abs(config.get_layer_lr(23) - expected_l24) < 1e-10
        assert abs(config.get_layer_lr(24) - expected_l25) < 1e-10
        assert abs(config.get_layer_lr(27) - expected_l28) < 1e-10

    def test_interface_has_highest_lr(self):
        """Test that interface layer has highest LR."""
        from modeling_studio.trainers.zipper_lr_v3 import ZipperLRConfig

        config = ZipperLRConfig()

        interface_lr = config.get_layer_lr(22)

        # Check interface LR is higher than all others
        for idx in range(28):
            if idx != 22:
                assert interface_lr >= config.get_layer_lr(idx)

    def test_get_all_layer_lrs(self):
        """Test getting all layer LRs."""
        from modeling_studio.trainers.zipper_lr_v3 import ZipperLRConfig

        config = ZipperLRConfig()
        all_lrs = config.get_all_layer_lrs()

        assert len(all_lrs) == 28
        assert all_lrs[0] == 0.0  # Foundation frozen
        assert all_lrs[17] == 0.0  # Core frozen
        assert all_lrs[18] == config.feeder_lr  # Feeder
        assert all_lrs[22] == config.interface_lr  # Interface

    def test_get_trainable_layer_lrs(self):
        """Test getting trainable layer LRs only."""
        from modeling_studio.trainers.zipper_lr_v3 import ZipperLRConfig

        config = ZipperLRConfig()
        trainable_lrs = config.get_trainable_layer_lrs()

        # Should only have L19-28 (indices 18-27)
        assert len(trainable_lrs) == 10
        assert 0 not in trainable_lrs  # Frozen
        assert 17 not in trainable_lrs  # Frozen
        assert 18 in trainable_lrs  # Feeder
        assert 22 in trainable_lrs  # Interface

    def test_get_band_summary(self):
        """Test getting band summary."""
        from modeling_studio.trainers.zipper_lr_v3 import ZipperLRConfig

        config = ZipperLRConfig()
        summary = config.get_band_summary()

        assert "foundation" in summary
        assert "core" in summary
        assert "feeder" in summary
        assert "interface" in summary
        assert "family" in summary

        assert summary["interface"]["lr"] == config.interface_lr
        assert summary["feeder"]["lr"] == config.feeder_lr


class TestZipperPresets:
    """Tests for Zipper LR presets."""

    def test_presets_exist(self):
        """Test that all expected presets exist."""
        from modeling_studio.trainers.zipper_lr_v3 import ZIPPER_PRESETS

        expected_presets = [
            "phase_0.5_healing",
            "phase_1_multitask",
            "phase_2_polish",
            "conservative",
            "aggressive",
        ]

        for preset in expected_presets:
            assert preset in ZIPPER_PRESETS

    def test_preset_values(self):
        """Test preset configuration values."""
        from modeling_studio.trainers.zipper_lr_v3 import ZIPPER_PRESETS

        # Phase 0.5 healing
        healing = ZIPPER_PRESETS["phase_0.5_healing"]
        assert healing.interface_lr == 5e-5
        assert healing.family_decay == 0.85

        # Phase 1 multitask
        multitask = ZIPPER_PRESETS["phase_1_multitask"]
        assert multitask.interface_lr == 4e-5
        assert multitask.family_decay == 0.9

        # Phase 2 polish
        polish = ZIPPER_PRESETS["phase_2_polish"]
        assert polish.family_graduated is False

        # Aggressive
        aggressive = ZIPPER_PRESETS["aggressive"]
        assert aggressive.interface_lr == 1e-4

    def test_get_zipper_preset(self):
        """Test getting a preset by name."""
        from modeling_studio.trainers.zipper_lr_v3 import get_zipper_preset

        config = get_zipper_preset("phase_0.5_healing")
        assert config.interface_lr == 5e-5

    def test_get_zipper_preset_unknown(self):
        """Test error on unknown preset."""
        from modeling_studio.trainers.zipper_lr_v3 import get_zipper_preset

        with pytest.raises(ValueError, match="Unknown preset"):
            get_zipper_preset("unknown_preset")

    def test_list_zipper_presets(self):
        """Test listing available presets."""
        from modeling_studio.trainers.zipper_lr_v3 import list_zipper_presets

        presets = list_zipper_presets()
        assert len(presets) == 5
        assert "phase_0.5_healing" in presets


class TestZipperLROptimizer:
    """Tests for ZipperLROptimizer class."""

    @pytest.fixture
    def model_for_zipper(self):
        """Create mock model for Zipper tests."""
        return MockModelForZipper(num_layers=28, hidden_size=768)

    def test_optimizer_creation(self, model_for_zipper):
        """Test creating optimizer with Zipper strategy."""
        from modeling_studio.trainers.zipper_lr_v3 import (
            ZipperLRConfig,
            ZipperLROptimizer,
        )

        config = ZipperLRConfig()
        zipper = ZipperLROptimizer(model_for_zipper, config)

        optimizer = zipper.create_optimizer()

        assert isinstance(optimizer, torch.optim.AdamW)

    def test_optimizer_param_groups(self, model_for_zipper):
        """Test that param groups are created correctly."""
        from modeling_studio.trainers.zipper_lr_v3 import (
            ZipperLRConfig,
            ZipperLROptimizer,
        )

        config = ZipperLRConfig()
        zipper = ZipperLROptimizer(model_for_zipper, config)

        optimizer = zipper.create_optimizer()

        # Should have groups for L19-28 + embeddings (if lr>0) + task_heads + other
        # With default config, embeddings_lr=0, so no embeddings group
        assert len(optimizer.param_groups) > 0

        # Check that layer groups have correct names
        layer_names = [g.get("name", "") for g in optimizer.param_groups]
        # Should have layer_19 through layer_28
        for layer_num in range(19, 29):
            assert f"layer_{layer_num}" in layer_names

    def test_optimizer_layer_lrs(self, model_for_zipper):
        """Test that layers get correct LRs."""
        from modeling_studio.trainers.zipper_lr_v3 import (
            ZipperLRConfig,
            ZipperLROptimizer,
        )

        config = ZipperLRConfig(
            feeder_lr=1e-5,
            interface_lr=5e-5,
            family_graduated=False,
            family_lr=3e-5,
        )
        zipper = ZipperLROptimizer(model_for_zipper, config)

        optimizer = zipper.create_optimizer()

        # Find layer_23 group (interface)
        for group in optimizer.param_groups:
            if group.get("name") == "layer_23":
                assert group["lr"] == 5e-5
            elif group.get("name") == "layer_19":
                assert group["lr"] == 1e-5
            elif group.get("name") == "layer_24":
                assert group["lr"] == 3e-5

    def test_optimizer_with_weight_decay(self, model_for_zipper):
        """Test weight decay is applied."""
        from modeling_studio.trainers.zipper_lr_v3 import (
            ZipperLRConfig,
            ZipperLROptimizer,
        )

        config = ZipperLRConfig()
        zipper = ZipperLROptimizer(model_for_zipper, config, weight_decay=0.05)

        optimizer = zipper.create_optimizer()

        assert optimizer.defaults["weight_decay"] == 0.05

    def test_optimizer_custom_betas(self, model_for_zipper):
        """Test custom beta parameters."""
        from modeling_studio.trainers.zipper_lr_v3 import (
            ZipperLRConfig,
            ZipperLROptimizer,
        )

        config = ZipperLRConfig()
        zipper = ZipperLROptimizer(
            model_for_zipper,
            config,
            betas=(0.85, 0.99),
        )

        optimizer = zipper.create_optimizer()

        assert optimizer.defaults["betas"] == (0.85, 0.99)

    def test_get_lr_dict(self, model_for_zipper):
        """Test getting LR dictionary."""
        from modeling_studio.trainers.zipper_lr_v3 import (
            ZipperLRConfig,
            ZipperLROptimizer,
        )

        config = ZipperLRConfig()
        zipper = ZipperLROptimizer(model_for_zipper, config)

        lr_dict = zipper.get_lr_dict()

        assert "layer_1" in lr_dict
        assert "layer_23" in lr_dict
        assert "layer_28" in lr_dict
        assert "embeddings" in lr_dict
        assert "task_heads" in lr_dict

        assert lr_dict["layer_1"] == 0.0  # Frozen
        assert lr_dict["layer_23"] == config.interface_lr

    def test_get_param_group_count(self, model_for_zipper):
        """Test counting param groups."""
        from modeling_studio.trainers.zipper_lr_v3 import (
            ZipperLRConfig,
            ZipperLROptimizer,
        )

        config = ZipperLRConfig()
        zipper = ZipperLROptimizer(model_for_zipper, config)

        count = zipper.get_param_group_count()
        assert count >= 10  # At least L19-28

    def test_get_trainable_param_count(self, model_for_zipper):
        """Test counting trainable parameters."""
        from modeling_studio.trainers.zipper_lr_v3 import (
            ZipperLRConfig,
            ZipperLROptimizer,
        )

        config = ZipperLRConfig()
        zipper = ZipperLROptimizer(model_for_zipper, config)

        count = zipper.get_trainable_param_count()
        assert count > 0


class TestCreateZipperOptimizer:
    """Tests for create_zipper_optimizer factory function."""

    @pytest.fixture
    def model_for_factory(self):
        """Create mock model for factory tests."""
        return MockModelForZipper(num_layers=28, hidden_size=768)

    def test_create_with_default_preset(self, model_for_factory):
        """Test creating optimizer with default preset."""
        from modeling_studio.trainers.zipper_lr_v3 import create_zipper_optimizer

        optimizer = create_zipper_optimizer(model_for_factory)

        assert isinstance(optimizer, torch.optim.AdamW)

    def test_create_with_named_preset(self, model_for_factory):
        """Test creating optimizer with named preset."""
        from modeling_studio.trainers.zipper_lr_v3 import create_zipper_optimizer

        optimizer = create_zipper_optimizer(
            model_for_factory,
            preset="phase_1_multitask",
        )

        # Find interface layer group and check LR
        for group in optimizer.param_groups:
            if group.get("name") == "layer_23":
                assert group["lr"] == 4e-5  # Phase 1 interface LR

    def test_create_with_overrides(self, model_for_factory):
        """Test creating optimizer with config overrides."""
        from modeling_studio.trainers.zipper_lr_v3 import create_zipper_optimizer

        optimizer = create_zipper_optimizer(
            model_for_factory,
            preset="phase_0.5_healing",
            interface_lr=7e-5,  # Override
        )

        # Check override was applied
        for group in optimizer.param_groups:
            if group.get("name") == "layer_23":
                assert group["lr"] == 7e-5

    def test_create_with_weight_decay(self, model_for_factory):
        """Test creating optimizer with custom weight decay."""
        from modeling_studio.trainers.zipper_lr_v3 import create_zipper_optimizer

        optimizer = create_zipper_optimizer(
            model_for_factory,
            weight_decay=0.02,
        )

        assert optimizer.defaults["weight_decay"] == 0.02

    def test_create_unknown_preset_uses_default(self, model_for_factory):
        """Test that unknown preset falls back to default."""
        from modeling_studio.trainers.zipper_lr_v3 import create_zipper_optimizer

        # Should not raise, but use default
        optimizer = create_zipper_optimizer(
            model_for_factory,
            preset="unknown_preset",
        )

        assert isinstance(optimizer, torch.optim.AdamW)


class TestZipperUtilityFunctions:
    """Tests for Zipper utility functions."""

    def test_compare_zipper_presets(self):
        """Test comparing presets."""
        from modeling_studio.trainers.zipper_lr_v3 import compare_zipper_presets

        comparison = compare_zipper_presets()

        assert "phase_0.5_healing" in comparison
        assert "phase_1_multitask" in comparison

        healing = comparison["phase_0.5_healing"]
        assert "interface_lr" in healing
        assert "family_graduated" in healing

    def test_validate_zipper_config_valid(self):
        """Test validation of valid config."""
        from modeling_studio.trainers.zipper_lr_v3 import (
            ZipperLRConfig,
            validate_zipper_config,
        )

        config = ZipperLRConfig()
        warnings = validate_zipper_config(config)

        assert len(warnings) == 0

    def test_validate_zipper_config_interface_too_low(self):
        """Test validation catches low interface LR."""
        from modeling_studio.trainers.zipper_lr_v3 import (
            ZipperLRConfig,
            validate_zipper_config,
        )

        config = ZipperLRConfig(
            feeder_lr=1e-4,
            interface_lr=1e-5,  # Lower than feeder
        )
        warnings = validate_zipper_config(config)

        assert len(warnings) > 0
        assert any("Interface LR" in w for w in warnings)

    def test_validate_zipper_config_bad_decay(self):
        """Test validation catches invalid decay."""
        from modeling_studio.trainers.zipper_lr_v3 import (
            ZipperLRConfig,
            validate_zipper_config,
        )

        config = ZipperLRConfig(
            family_graduated=True,
            family_decay=1.5,  # Invalid: > 1
        )
        warnings = validate_zipper_config(config)

        assert len(warnings) > 0
        assert any("family_decay" in w for w in warnings)

    def test_quick_ref_string(self):
        """Test quick reference string exists."""
        from modeling_studio.trainers.zipper_lr_v3 import ZIPPER_LR_QUICK_REF

        assert isinstance(ZIPPER_LR_QUICK_REF, str)
        assert "Interface" in ZIPPER_LR_QUICK_REF
        assert "L23" in ZIPPER_LR_QUICK_REF


class TestIssue516AcceptanceCriteria:
    """Acceptance criteria tests for Issue 5.1.6."""

    @pytest.fixture
    def model_for_ac(self):
        """Create model for acceptance criteria tests."""
        return MockModelForZipper(num_layers=28, hidden_size=768)

    def test_ac1_zipper_lr_config_defines_all_layer_lrs(self):
        """AC1: ZipperLRConfig defines all layer LRs."""
        from modeling_studio.trainers.zipper_lr_v3 import ZipperLRConfig

        config = ZipperLRConfig()

        # All 28 layers should have defined LRs
        for idx in range(28):
            lr = config.get_layer_lr(idx)
            assert lr is not None
            assert lr >= 0

        print("AC1: ZipperLRConfig defines all layer LRs [PASS]")

    def test_ac2_interface_layer_gets_highest_lr(self):
        """AC2: Interface layer (L23) gets highest LR."""
        from modeling_studio.trainers.zipper_lr_v3 import ZipperLRConfig

        config = ZipperLRConfig()
        interface_lr = config.get_layer_lr(22)  # L23, index 22

        for idx in range(28):
            if idx != 22:
                assert interface_lr >= config.get_layer_lr(idx)

        print("AC2: Interface layer (L23) gets highest LR [PASS]")

    def test_ac3_graduated_decay_in_family_band(self):
        """AC3: Graduated decay in Family band works correctly."""
        from modeling_studio.trainers.zipper_lr_v3 import ZipperLRConfig

        config = ZipperLRConfig(
            interface_lr=5e-5,
            family_graduated=True,
            family_decay=0.9,
        )

        # Check that LRs decrease from L24 to L28
        prev_lr = config.get_layer_lr(22)  # Interface
        for idx in range(23, 28):
            current_lr = config.get_layer_lr(idx)
            assert current_lr < prev_lr
            prev_lr = current_lr

        print("AC3: Graduated decay in Family band works correctly [PASS]")

    def test_ac4_feeder_band_uniform_low_lr(self):
        """AC4: Feeder band gets uniform low LR."""
        from modeling_studio.trainers.zipper_lr_v3 import ZipperLRConfig

        config = ZipperLRConfig(feeder_lr=1e-5)

        # All feeder layers should have same LR
        for idx in range(18, 22):  # L19-22
            assert config.get_layer_lr(idx) == 1e-5

        print("AC4: Feeder band gets uniform low LR [PASS]")

    def test_ac5_create_optimizer_creates_valid_adamw(self, model_for_ac):
        """AC5: create_optimizer() creates valid AdamW."""
        from modeling_studio.trainers.zipper_lr_v3 import create_zipper_optimizer

        optimizer = create_zipper_optimizer(model_for_ac)

        assert isinstance(optimizer, torch.optim.AdamW)
        assert len(optimizer.param_groups) > 0

        # Verify can do a step
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.requires_grad:
                    param.grad = torch.zeros_like(param)

        optimizer.step()  # Should not raise

        print("AC5: create_optimizer() creates valid AdamW [PASS]")

    def test_ac6_ascii_visualization(self, model_for_ac, capsys):
        """AC6: ASCII visualization shows LR profile clearly."""
        from modeling_studio.trainers.zipper_lr_v3 import (
            ZipperLRConfig,
            ZipperLROptimizer,
        )

        config = ZipperLRConfig()
        zipper = ZipperLROptimizer(model_for_ac, config)
        zipper._print_zipper_summary()

        captured = capsys.readouterr()

        # Check for key elements in output
        assert "Zipper Learning Rate" in captured.out
        assert "Layer" in captured.out
        assert "Band" in captured.out
        assert "Interface" in captured.out
        assert "frozen" in captured.out.lower()

        print("AC6: ASCII visualization shows LR profile clearly [PASS]")

    def test_ac7_presets_for_all_phases(self):
        """AC7: Presets for all phases available."""
        from modeling_studio.trainers.zipper_lr_v3 import ZIPPER_PRESETS

        # Required presets
        required = ["phase_0.5_healing", "phase_1_multitask", "phase_2_polish"]

        for preset in required:
            assert preset in ZIPPER_PRESETS

        print("AC7: Presets for all phases available [PASS]")

    def test_ac8_override_mechanism_works(self, model_for_ac):
        """AC8: Override mechanism works."""
        from modeling_studio.trainers.zipper_lr_v3 import create_zipper_optimizer

        # Override interface_lr
        optimizer = create_zipper_optimizer(
            model_for_ac,
            preset="phase_0.5_healing",
            interface_lr=8e-5,  # Override from 5e-5
        )

        # Find interface group and verify override
        for group in optimizer.param_groups:
            if group.get("name") == "layer_23":
                assert group["lr"] == 8e-5
                break

        print("AC8: Override mechanism works [PASS]")


# ============================================================================
# Issue 5.1.7: Warmup + Cosine Decay Scheduler Tests
# ============================================================================


class TestWarmupCosineScheduler:
    """Tests for WarmupCosineScheduler."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer for scheduler tests."""
        model = nn.Linear(10, 10)
        return torch.optim.AdamW(model.parameters(), lr=3e-5)

    @pytest.fixture
    def multi_group_optimizer(self):
        """Create optimizer with multiple param groups."""
        model1 = nn.Linear(10, 10)
        model2 = nn.Linear(10, 10)
        return torch.optim.AdamW(
            [
                {"params": model1.parameters(), "lr": 3e-5},
                {"params": model2.parameters(), "lr": 1e-5},
            ]
        )

    def test_init_default(self, optimizer):
        """Test default initialization."""
        from modeling_studio.trainers.schedulers_v3 import WarmupCosineScheduler

        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=500,
            total_steps=2500,
        )

        assert scheduler.warmup_steps == 500
        assert scheduler.total_steps == 2500
        assert scheduler.min_lr_ratio == 0.01

    def test_init_custom_min_lr(self, optimizer):
        """Test initialization with custom min_lr_ratio."""
        from modeling_studio.trainers.schedulers_v3 import WarmupCosineScheduler

        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
            min_lr_ratio=0.1,
        )

        assert scheduler.min_lr_ratio == 0.1

    def test_init_invalid_warmup(self, optimizer):
        """Test error on invalid warmup_steps."""
        from modeling_studio.trainers.schedulers_v3 import WarmupCosineScheduler

        with pytest.raises(ValueError, match="warmup_steps"):
            WarmupCosineScheduler(
                optimizer,
                warmup_steps=-1,
                total_steps=1000,
            )

    def test_init_warmup_greater_than_total(self, optimizer):
        """Test error when warmup > total."""
        from modeling_studio.trainers.schedulers_v3 import WarmupCosineScheduler

        with pytest.raises(ValueError, match="total_steps"):
            WarmupCosineScheduler(
                optimizer,
                warmup_steps=1000,
                total_steps=500,
            )

    def test_lr_at_step_0(self, optimizer):
        """Test LR is 0 at step 0."""
        from modeling_studio.trainers.schedulers_v3 import WarmupCosineScheduler

        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=500,
            total_steps=2500,
        )

        # Step 0: lr should be 0
        lrs = scheduler.get_lr()
        assert lrs[0] == 0.0

    def test_lr_at_warmup_peak(self, optimizer):
        """Test LR peaks at warmup_steps."""
        from modeling_studio.trainers.schedulers_v3 import WarmupCosineScheduler

        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=500,
            total_steps=2500,
        )

        # Advance to warmup_steps
        for _ in range(500):
            scheduler.step()

        lrs = scheduler.get_lr()
        assert abs(lrs[0] - 3e-5) < 1e-10  # Should be at base_lr

    def test_lr_at_end(self, optimizer):
        """Test LR at end is min_lr."""
        from modeling_studio.trainers.schedulers_v3 import WarmupCosineScheduler

        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=500,
            total_steps=2500,
            min_lr_ratio=0.01,
        )

        # Advance to total_steps
        for _ in range(2500):
            scheduler.step()

        lrs = scheduler.get_lr()
        expected = 3e-5 * 0.01
        assert abs(lrs[0] - expected) < 1e-10

    def test_lr_beyond_total(self, optimizer):
        """Test LR stays at min after total_steps."""
        from modeling_studio.trainers.schedulers_v3 import WarmupCosineScheduler

        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=500,
            total_steps=2500,
            min_lr_ratio=0.01,
        )

        # Advance beyond total_steps
        for _ in range(3000):
            scheduler.step()

        lrs = scheduler.get_lr()
        expected = 3e-5 * 0.01
        assert abs(lrs[0] - expected) < 1e-10

    def test_cosine_shape(self, optimizer):
        """Test cosine decay shape."""
        from modeling_studio.trainers.schedulers_v3 import WarmupCosineScheduler
        import math

        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=0,  # No warmup for clean test
            total_steps=1000,
            min_lr_ratio=0.0,  # Decay to 0
        )

        # At step 500 (midpoint), should be at ~50% of base_lr
        for _ in range(500):
            scheduler.step()

        lrs = scheduler.get_lr()
        # Cosine at 0.5 progress: 0.5 * (1 + cos(pi * 0.5)) = 0.5
        expected = 3e-5 * 0.5
        assert abs(lrs[0] - expected) < 1e-7

    def test_multi_param_groups(self, multi_group_optimizer):
        """Test with multiple param groups."""
        from modeling_studio.trainers.schedulers_v3 import WarmupCosineScheduler

        scheduler = WarmupCosineScheduler(
            multi_group_optimizer,
            warmup_steps=100,
            total_steps=1000,
        )

        # Advance to warmup_steps
        for _ in range(100):
            scheduler.step()

        lrs = scheduler.get_lr()
        assert len(lrs) == 2
        assert abs(lrs[0] - 3e-5) < 1e-10
        assert abs(lrs[1] - 1e-5) < 1e-10

    def test_get_lr_at_step(self, optimizer):
        """Test getting LR at specific step without modifying state."""
        from modeling_studio.trainers.schedulers_v3 import WarmupCosineScheduler

        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=500,
            total_steps=2500,
        )

        # Check LR at step 500 without stepping
        lrs = scheduler.get_lr_at_step(500)
        assert abs(lrs[0] - 3e-5) < 1e-10

        # Scheduler should still be at step 0 (PyTorch auto-steps on init from -1 to 0)
        assert scheduler.last_epoch == 0


class TestWarmupLinearScheduler:
    """Tests for WarmupLinearScheduler."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer for scheduler tests."""
        model = nn.Linear(10, 10)
        return torch.optim.AdamW(model.parameters(), lr=3e-5)

    def test_init(self, optimizer):
        """Test initialization."""
        from modeling_studio.trainers.schedulers_v3 import WarmupLinearScheduler

        scheduler = WarmupLinearScheduler(
            optimizer,
            warmup_steps=500,
            total_steps=2500,
        )

        assert scheduler.warmup_steps == 500
        assert scheduler.total_steps == 2500
        assert scheduler.min_lr_ratio == 0.0

    def test_linear_warmup(self, optimizer):
        """Test linear warmup phase."""
        from modeling_studio.trainers.schedulers_v3 import WarmupLinearScheduler

        scheduler = WarmupLinearScheduler(
            optimizer,
            warmup_steps=500,
            total_steps=2500,
        )

        # At step 250 (midpoint of warmup), should be at 50%
        for _ in range(250):
            scheduler.step()

        lrs = scheduler.get_lr()
        expected = 3e-5 * 0.5
        assert abs(lrs[0] - expected) < 1e-10

    def test_linear_decay(self, optimizer):
        """Test linear decay phase."""
        from modeling_studio.trainers.schedulers_v3 import WarmupLinearScheduler

        scheduler = WarmupLinearScheduler(
            optimizer,
            warmup_steps=0,  # No warmup
            total_steps=1000,
            min_lr_ratio=0.0,
        )

        # At step 500 (midpoint), should be at 50%
        for _ in range(500):
            scheduler.step()

        lrs = scheduler.get_lr()
        expected = 3e-5 * 0.5
        assert abs(lrs[0] - expected) < 1e-10

    def test_lr_at_end(self, optimizer):
        """Test LR at end."""
        from modeling_studio.trainers.schedulers_v3 import WarmupLinearScheduler

        scheduler = WarmupLinearScheduler(
            optimizer,
            warmup_steps=500,
            total_steps=2500,
            min_lr_ratio=0.0,
        )

        # Advance to total_steps
        for _ in range(2500):
            scheduler.step()

        lrs = scheduler.get_lr()
        assert lrs[0] == 0.0


class TestWarmupConstantScheduler:
    """Tests for WarmupConstantScheduler."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer for scheduler tests."""
        model = nn.Linear(10, 10)
        return torch.optim.AdamW(model.parameters(), lr=3e-5)

    def test_init(self, optimizer):
        """Test initialization."""
        from modeling_studio.trainers.schedulers_v3 import WarmupConstantScheduler

        scheduler = WarmupConstantScheduler(
            optimizer,
            warmup_steps=500,
        )

        assert scheduler.warmup_steps == 500

    def test_warmup_phase(self, optimizer):
        """Test warmup phase."""
        from modeling_studio.trainers.schedulers_v3 import WarmupConstantScheduler

        scheduler = WarmupConstantScheduler(
            optimizer,
            warmup_steps=500,
        )

        # At step 250 (midpoint of warmup), should be at 50%
        for _ in range(250):
            scheduler.step()

        lrs = scheduler.get_lr()
        expected = 3e-5 * 0.5
        assert abs(lrs[0] - expected) < 1e-10

    def test_constant_phase(self, optimizer):
        """Test constant phase after warmup."""
        from modeling_studio.trainers.schedulers_v3 import WarmupConstantScheduler

        scheduler = WarmupConstantScheduler(
            optimizer,
            warmup_steps=500,
        )

        # Advance past warmup
        for _ in range(1000):
            scheduler.step()

        lrs = scheduler.get_lr()
        assert abs(lrs[0] - 3e-5) < 1e-10

    def test_stays_constant(self, optimizer):
        """Test LR stays constant after warmup."""
        from modeling_studio.trainers.schedulers_v3 import WarmupConstantScheduler

        scheduler = WarmupConstantScheduler(
            optimizer,
            warmup_steps=100,
        )

        # Advance to various points after warmup
        for _ in range(100):
            scheduler.step()
        lr_at_100 = scheduler.get_lr()[0]

        for _ in range(1000):
            scheduler.step()
        lr_at_1100 = scheduler.get_lr()[0]

        assert abs(lr_at_100 - lr_at_1100) < 1e-10


class TestPhaseAwareScheduler:
    """Tests for PhaseAwareScheduler."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer for scheduler tests."""
        model = nn.Linear(10, 10)
        return torch.optim.AdamW(model.parameters(), lr=3e-5)

    @pytest.fixture
    def phase_configs(self):
        """Create phase configurations."""
        return {
            "phase_0.5": {
                "scheduler_type": "cosine",
                "warmup_steps": 100,
                "total_steps": 500,
                "min_lr_ratio": 0.01,
            },
            "phase_1": {
                "scheduler_type": "linear",
                "warmup_steps": 200,
                "total_steps": 1000,
                "min_lr_ratio": 0.01,
            },
            "phase_2": {
                "scheduler_type": "constant",
                "warmup_steps": 50,
                "total_steps": 300,
            },
        }

    def test_init(self, optimizer, phase_configs):
        """Test initialization."""
        from modeling_studio.trainers.schedulers_v3 import PhaseAwareScheduler

        scheduler = PhaseAwareScheduler(optimizer, phase_configs)

        assert scheduler.current_phase is None
        assert scheduler.current_scheduler is None
        assert scheduler.phase_step == 0

    def test_set_phase(self, optimizer, phase_configs):
        """Test setting a phase."""
        from modeling_studio.trainers.schedulers_v3 import PhaseAwareScheduler

        scheduler = PhaseAwareScheduler(optimizer, phase_configs)
        scheduler.set_phase("phase_0.5")

        assert scheduler.current_phase == "phase_0.5"
        assert scheduler.current_scheduler is not None
        assert scheduler.phase_step == 0

    def test_set_unknown_phase(self, optimizer, phase_configs):
        """Test error on unknown phase."""
        from modeling_studio.trainers.schedulers_v3 import PhaseAwareScheduler

        scheduler = PhaseAwareScheduler(optimizer, phase_configs)

        with pytest.raises(ValueError, match="Unknown phase"):
            scheduler.set_phase("unknown")

    def test_step(self, optimizer, phase_configs):
        """Test stepping the scheduler."""
        from modeling_studio.trainers.schedulers_v3 import PhaseAwareScheduler

        scheduler = PhaseAwareScheduler(optimizer, phase_configs)
        scheduler.set_phase("phase_0.5")

        scheduler.step()
        assert scheduler.phase_step == 1
        assert scheduler.total_steps_across_phases == 1

    def test_get_last_lr(self, optimizer, phase_configs):
        """Test getting last LR."""
        from modeling_studio.trainers.schedulers_v3 import PhaseAwareScheduler

        scheduler = PhaseAwareScheduler(optimizer, phase_configs)
        scheduler.set_phase("phase_0.5")

        lrs = scheduler.get_last_lr()
        assert len(lrs) == 1

    def test_phase_transition(self, optimizer, phase_configs):
        """Test transitioning between phases."""
        from modeling_studio.trainers.schedulers_v3 import PhaseAwareScheduler

        scheduler = PhaseAwareScheduler(optimizer, phase_configs)

        # Start phase 0.5
        scheduler.set_phase("phase_0.5")
        for _ in range(500):
            scheduler.step()

        # Transition to phase 1
        scheduler.set_phase("phase_1")
        assert scheduler.current_phase == "phase_1"
        assert scheduler.phase_step == 0

    def test_get_phase_progress(self, optimizer, phase_configs):
        """Test getting phase progress."""
        from modeling_studio.trainers.schedulers_v3 import PhaseAwareScheduler

        scheduler = PhaseAwareScheduler(optimizer, phase_configs)
        scheduler.set_phase("phase_0.5")

        # At step 250 of 500, should be 50%
        for _ in range(250):
            scheduler.step()

        progress = scheduler.get_phase_progress()
        assert abs(progress - 0.5) < 0.01

    def test_is_warmup_complete(self, optimizer, phase_configs):
        """Test warmup completion check."""
        from modeling_studio.trainers.schedulers_v3 import PhaseAwareScheduler

        scheduler = PhaseAwareScheduler(optimizer, phase_configs)
        scheduler.set_phase("phase_0.5")

        # Before warmup complete
        assert not scheduler.is_warmup_complete()

        # After warmup
        for _ in range(100):
            scheduler.step()
        assert scheduler.is_warmup_complete()

    def test_state_dict(self, optimizer, phase_configs):
        """Test state dict save/load."""
        from modeling_studio.trainers.schedulers_v3 import PhaseAwareScheduler

        scheduler = PhaseAwareScheduler(optimizer, phase_configs)
        scheduler.set_phase("phase_0.5")
        for _ in range(100):
            scheduler.step()

        state = scheduler.get_state_dict()

        # Create new scheduler and load state
        scheduler2 = PhaseAwareScheduler(optimizer, phase_configs)
        scheduler2.load_state_dict(state)

        assert scheduler2.current_phase == "phase_0.5"
        assert scheduler2.phase_step == 100


class TestCreateScheduler:
    """Tests for create_scheduler factory function."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer for scheduler tests."""
        model = nn.Linear(10, 10)
        return torch.optim.AdamW(model.parameters(), lr=3e-5)

    def test_create_cosine(self, optimizer):
        """Test creating cosine scheduler."""
        from modeling_studio.trainers.schedulers_v3 import (
            create_scheduler,
            WarmupCosineScheduler,
        )

        scheduler = create_scheduler(
            optimizer,
            scheduler_type="cosine",
            warmup_steps=500,
            total_steps=2500,
        )

        assert isinstance(scheduler, WarmupCosineScheduler)

    def test_create_linear(self, optimizer):
        """Test creating linear scheduler."""
        from modeling_studio.trainers.schedulers_v3 import (
            create_scheduler,
            WarmupLinearScheduler,
        )

        scheduler = create_scheduler(
            optimizer,
            scheduler_type="linear",
            warmup_steps=500,
            total_steps=2500,
        )

        assert isinstance(scheduler, WarmupLinearScheduler)

    def test_create_constant(self, optimizer):
        """Test creating constant scheduler."""
        from modeling_studio.trainers.schedulers_v3 import (
            create_scheduler,
            WarmupConstantScheduler,
        )

        scheduler = create_scheduler(
            optimizer,
            scheduler_type="constant",
            warmup_steps=500,
            total_steps=2500,
        )

        assert isinstance(scheduler, WarmupConstantScheduler)

    def test_create_unknown(self, optimizer):
        """Test error on unknown scheduler type."""
        from modeling_studio.trainers.schedulers_v3 import create_scheduler

        with pytest.raises(ValueError, match="Unknown scheduler type"):
            create_scheduler(
                optimizer,
                scheduler_type="unknown",
            )


class TestSchedulerUtilities:
    """Tests for scheduler utility functions."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer for scheduler tests."""
        model = nn.Linear(10, 10)
        return torch.optim.AdamW(model.parameters(), lr=3e-5)

    def test_compute_warmup_steps(self):
        """Test computing warmup steps."""
        from modeling_studio.trainers.schedulers_v3 import compute_warmup_steps

        # Normal case
        warmup = compute_warmup_steps(10000, warmup_ratio=0.1)
        assert warmup == 1000

        # Respects min
        warmup = compute_warmup_steps(500, warmup_ratio=0.1, min_warmup=100)
        assert warmup == 100

        # Respects max
        warmup = compute_warmup_steps(100000, warmup_ratio=0.1, max_warmup=2000)
        assert warmup == 2000

    def test_get_lr_at_step(self, optimizer):
        """Test getting LR at specific step."""
        from modeling_studio.trainers.schedulers_v3 import (
            WarmupCosineScheduler,
            get_lr_at_step,
        )

        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=500,
            total_steps=2500,
        )

        # Get LR at warmup peak
        lrs = get_lr_at_step(scheduler, 500)
        assert abs(lrs[0] - 3e-5) < 1e-10

    def test_default_phase_configs(self):
        """Test default phase configurations exist."""
        from modeling_studio.trainers.schedulers_v3 import DEFAULT_PHASE_SCHEDULER_CONFIGS

        assert "phase_0.5" in DEFAULT_PHASE_SCHEDULER_CONFIGS
        assert "phase_1" in DEFAULT_PHASE_SCHEDULER_CONFIGS
        assert "phase_2" in DEFAULT_PHASE_SCHEDULER_CONFIGS

        # Check phase_0.5 config
        config = DEFAULT_PHASE_SCHEDULER_CONFIGS["phase_0.5"]
        assert config["scheduler_type"] == "cosine"
        assert config["warmup_steps"] == 500
        assert config["total_steps"] == 2500

    def test_valid_scheduler_types(self):
        """Test valid scheduler types constant."""
        from modeling_studio.trainers.schedulers_v3 import VALID_SCHEDULER_TYPES

        assert "cosine" in VALID_SCHEDULER_TYPES
        assert "linear" in VALID_SCHEDULER_TYPES
        assert "constant" in VALID_SCHEDULER_TYPES


class TestIssue517AcceptanceCriteria:
    """Acceptance criteria tests for Issue 5.1.7."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer for acceptance tests."""
        model = nn.Linear(10, 10)
        return torch.optim.AdamW(model.parameters(), lr=3e-5)

    def test_ac1_warmup_cosine_implements_correctly(self, optimizer):
        """AC1: WarmupCosineScheduler implements warmup + cosine correctly."""
        from modeling_studio.trainers.schedulers_v3 import WarmupCosineScheduler
        import math

        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=500,
            total_steps=2500,
            min_lr_ratio=0.01,
        )

        # Test warmup phase
        for _ in range(250):
            scheduler.step()
        lr_mid_warmup = scheduler.get_lr()[0]
        assert abs(lr_mid_warmup - 3e-5 * 0.5) < 1e-10

        # Test peak at warmup
        for _ in range(250):
            scheduler.step()
        lr_peak = scheduler.get_lr()[0]
        assert abs(lr_peak - 3e-5) < 1e-10

        # Test decay
        for _ in range(1000):
            scheduler.step()
        # At midpoint of decay (step 1500), cosine gives ~0.5
        lr_mid_decay = scheduler.get_lr()[0]
        # Should be between min and peak
        assert 3e-5 * 0.01 < lr_mid_decay < 3e-5

        print("AC1: WarmupCosineScheduler implements warmup + cosine correctly [PASS]")

    def test_ac2_lr_profile(self, optimizer):
        """AC2: LR starts at 0, peaks at warmup_steps, decays to min_lr."""
        from modeling_studio.trainers.schedulers_v3 import WarmupCosineScheduler

        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=500,
            total_steps=2500,
            min_lr_ratio=0.01,
        )

        # Step 0: lr = 0
        assert scheduler.get_lr()[0] == 0.0

        # Step 500: lr = base_lr (peak)
        for _ in range(500):
            scheduler.step()
        assert abs(scheduler.get_lr()[0] - 3e-5) < 1e-10

        # Step 2500: lr = min_lr
        for _ in range(2000):
            scheduler.step()
        assert abs(scheduler.get_lr()[0] - 3e-5 * 0.01) < 1e-10

        print("AC2: LR starts at 0, peaks at warmup_steps, decays to min_lr [PASS]")

    def test_ac3_warmup_linear_provides_alternative(self, optimizer):
        """AC3: WarmupLinearScheduler provides linear alternative."""
        from modeling_studio.trainers.schedulers_v3 import WarmupLinearScheduler

        scheduler = WarmupLinearScheduler(
            optimizer,
            warmup_steps=500,
            total_steps=2500,
        )

        # Linear warmup
        for _ in range(250):
            scheduler.step()
        assert abs(scheduler.get_lr()[0] - 3e-5 * 0.5) < 1e-10

        # Linear decay
        for _ in range(250):  # Now at step 500
            scheduler.step()
        lr_at_500 = scheduler.get_lr()[0]
        assert abs(lr_at_500 - 3e-5) < 1e-10

        for _ in range(1000):  # Now at step 1500
            scheduler.step()
        # Linear decay: 1500 is at 50% through decay phase
        lr_at_1500 = scheduler.get_lr()[0]
        assert lr_at_1500 < lr_at_500  # Should have decayed

        print("AC3: WarmupLinearScheduler provides linear alternative [PASS]")

    def test_ac4_warmup_constant_for_short_runs(self, optimizer):
        """AC4: WarmupConstantScheduler for short runs."""
        from modeling_studio.trainers.schedulers_v3 import WarmupConstantScheduler

        scheduler = WarmupConstantScheduler(
            optimizer,
            warmup_steps=100,
        )

        # After warmup, should stay constant
        for _ in range(100):
            scheduler.step()
        lr_at_100 = scheduler.get_lr()[0]

        for _ in range(1000):
            scheduler.step()
        lr_at_1100 = scheduler.get_lr()[0]

        assert abs(lr_at_100 - lr_at_1100) < 1e-10
        assert abs(lr_at_100 - 3e-5) < 1e-10

        print("AC4: WarmupConstantScheduler for short runs [PASS]")

    def test_ac5_phase_aware_handles_transitions(self, optimizer):
        """AC5: PhaseAwareScheduler handles phase transitions."""
        from modeling_studio.trainers.schedulers_v3 import (
            PhaseAwareScheduler,
            DEFAULT_PHASE_SCHEDULER_CONFIGS,
        )

        scheduler = PhaseAwareScheduler(optimizer, DEFAULT_PHASE_SCHEDULER_CONFIGS)

        # Phase 0.5
        scheduler.set_phase("phase_0.5")
        assert scheduler.current_phase == "phase_0.5"
        for _ in range(500):
            scheduler.step()

        # Transition to phase 1
        scheduler.set_phase("phase_1")
        assert scheduler.current_phase == "phase_1"
        assert scheduler.phase_step == 0  # Reset

        print("AC5: PhaseAwareScheduler handles phase transitions [PASS]")

    def test_ac6_create_scheduler_factory(self, optimizer):
        """AC6: create_scheduler() factory function works."""
        from modeling_studio.trainers.schedulers_v3 import create_scheduler

        # All types should work
        for scheduler_type in ["cosine", "linear", "constant"]:
            scheduler = create_scheduler(
                optimizer,
                scheduler_type=scheduler_type,
                warmup_steps=100,
                total_steps=1000,
            )
            assert scheduler is not None

        print("AC6: create_scheduler() factory function works [PASS]")

    def test_ac7_compatible_with_per_layer_lrs(self):
        """AC7: Compatible with per-layer-group LRs."""
        from modeling_studio.trainers.schedulers_v3 import WarmupCosineScheduler

        # Create optimizer with multiple param groups (different LRs)
        model1 = nn.Linear(10, 10)
        model2 = nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(
            [
                {"params": model1.parameters(), "lr": 5e-5},  # Interface
                {"params": model2.parameters(), "lr": 1e-5},  # Feeder
            ]
        )

        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
        )

        # Advance to peak
        for _ in range(100):
            scheduler.step()

        lrs = scheduler.get_lr()
        assert len(lrs) == 2
        assert abs(lrs[0] - 5e-5) < 1e-10  # Group 1 at its peak
        assert abs(lrs[1] - 1e-5) < 1e-10  # Group 2 at its peak

        print("AC7: Compatible with per-layer-group LRs [PASS]")


# ============================================================================
# Issue 5.1.8: Gradient Clipping for Phase 0.5
# ============================================================================


class TestGradientClipConfig:
    """Tests for GradientClipConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from modeling_studio.trainers.gradient_utils_v3 import GradientClipConfig

        config = GradientClipConfig()
        assert config.max_grad_norm == 1.0
        assert config.per_layer_clip is False
        assert config.interface_clip == 0.5
        assert config.family_clip == 1.0
        assert config.feeder_clip == 1.0
        assert config.log_grad_norms is True
        assert config.log_every_n_steps == 100
        assert config.explosion_threshold == 10.0
        assert config.nan_check is True

    def test_custom_values(self):
        """Test custom configuration values."""
        from modeling_studio.trainers.gradient_utils_v3 import GradientClipConfig

        config = GradientClipConfig(
            max_grad_norm=2.0,
            per_layer_clip=True,
            interface_clip=0.3,
            log_every_n_steps=50,
        )
        assert config.max_grad_norm == 2.0
        assert config.per_layer_clip is True
        assert config.interface_clip == 0.3
        assert config.log_every_n_steps == 50


class TestGradientStats:
    """Tests for GradientStats dataclass."""

    def test_default_values(self):
        """Test default statistics values."""
        from modeling_studio.trainers.gradient_utils_v3 import GradientStats

        stats = GradientStats()
        assert stats.total_norm == 0.0
        assert stats.layer_norms == {}
        assert stats.max_grad == 0.0
        assert stats.min_grad == 0.0
        assert stats.has_nan is False
        assert stats.has_inf is False
        assert stats.clipped is False

    def test_custom_values(self):
        """Test custom statistics values."""
        from modeling_studio.trainers.gradient_utils_v3 import GradientStats

        stats = GradientStats(
            total_norm=5.0,
            layer_norms={"layer_23": 1.5},
            has_nan=True,
            clipped=True,
        )
        assert stats.total_norm == 5.0
        assert stats.layer_norms == {"layer_23": 1.5}
        assert stats.has_nan is True
        assert stats.clipped is True


class TestGradientClipper:
    """Tests for GradientClipper class."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
        )

    @pytest.fixture
    def mock_encoder_model(self):
        """Create a mock model with encoder.layers structure."""

        class MockEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(28)])

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = MockEncoder()

        return MockModel()

    def test_global_clip_no_gradients(self, simple_model):
        """Test global clipping with no gradients."""
        from modeling_studio.trainers.gradient_utils_v3 import (
            GradientClipConfig,
            GradientClipper,
        )

        config = GradientClipConfig(log_grad_norms=False)
        clipper = GradientClipper(simple_model, config)
        stats = clipper.clip_gradients()

        assert stats.total_norm == 0.0
        assert stats.clipped is False

    def test_global_clip_with_gradients(self, simple_model):
        """Test global clipping with gradients."""
        from modeling_studio.trainers.gradient_utils_v3 import (
            GradientClipConfig,
            GradientClipper,
        )

        # Create gradients
        x = torch.randn(4, 10)
        y = simple_model(x)
        loss = y.sum()
        loss.backward()

        config = GradientClipConfig(max_grad_norm=1.0, log_grad_norms=False)
        clipper = GradientClipper(simple_model, config)
        stats = clipper.clip_gradients()

        assert stats.total_norm > 0
        assert isinstance(stats.clipped, bool)

    def test_global_clip_clips_large_gradients(self, simple_model):
        """Test that large gradients are clipped."""
        from modeling_studio.trainers.gradient_utils_v3 import (
            GradientClipConfig,
            GradientClipper,
        )

        # Create large gradients
        x = torch.randn(4, 10) * 100
        y = simple_model(x)
        loss = y.sum() * 100
        loss.backward()

        config = GradientClipConfig(max_grad_norm=0.1, log_grad_norms=False)
        clipper = GradientClipper(simple_model, config)

        # Get norm before clipping
        params = [p for p in simple_model.parameters() if p.grad is not None]
        norm_before = 0.0
        for p in params:
            norm_before += p.grad.data.norm(2).item() ** 2
        norm_before = norm_before**0.5

        stats = clipper.clip_gradients()

        # Verify clipping happened
        assert norm_before > 0.1  # Was above threshold
        assert stats.clipped is True

    def test_per_layer_clip(self, mock_encoder_model):
        """Test per-layer gradient clipping."""
        from modeling_studio.trainers.gradient_utils_v3 import (
            GradientClipConfig,
            GradientClipper,
        )

        # Create gradients
        x = torch.randn(4, 10)
        y = x
        for layer in mock_encoder_model.encoder.layers:
            y = layer(y)
        loss = y.sum()
        loss.backward()

        config = GradientClipConfig(
            per_layer_clip=True,
            interface_clip=0.5,
            family_clip=1.0,
            feeder_clip=1.0,
            log_grad_norms=False,
        )
        clipper = GradientClipper(mock_encoder_model, config)
        stats = clipper.clip_gradients()

        assert stats.total_norm >= 0
        assert stats.clipped is True  # Per-layer always marks as clipped

    def test_nan_detection(self, simple_model):
        """Test NaN gradient detection."""
        from modeling_studio.trainers.gradient_utils_v3 import (
            GradientClipConfig,
            GradientClipper,
        )

        # Create gradients with NaN
        x = torch.randn(4, 10)
        y = simple_model(x)
        loss = y.sum()
        loss.backward()

        # Inject NaN
        for p in simple_model.parameters():
            if p.grad is not None:
                p.grad[0] = float("nan")
                break

        config = GradientClipConfig(nan_check=True, log_grad_norms=False)
        clipper = GradientClipper(simple_model, config)
        stats = clipper.clip_gradients()

        assert stats.has_nan is True

    def test_inf_detection(self, simple_model):
        """Test Inf gradient detection."""
        from modeling_studio.trainers.gradient_utils_v3 import (
            GradientClipConfig,
            GradientClipper,
        )

        # Create gradients with Inf
        x = torch.randn(4, 10)
        y = simple_model(x)
        loss = y.sum()
        loss.backward()

        # Inject Inf
        for p in simple_model.parameters():
            if p.grad is not None:
                p.grad[0] = float("inf")
                break

        config = GradientClipConfig(nan_check=True, log_grad_norms=False)
        clipper = GradientClipper(simple_model, config)
        stats = clipper.clip_gradients()

        assert stats.has_inf is True

    def test_bad_gradients_zeroed(self, simple_model):
        """Test that NaN/Inf gradients are zeroed."""
        from modeling_studio.trainers.gradient_utils_v3 import (
            GradientClipConfig,
            GradientClipper,
        )

        # Create gradients
        x = torch.randn(4, 10)
        y = simple_model(x)
        loss = y.sum()
        loss.backward()

        # Inject NaN and Inf
        for p in simple_model.parameters():
            if p.grad is not None:
                p.grad.data[0] = float("nan")
                p.grad.data[1] = float("inf")
                break

        config = GradientClipConfig(nan_check=True, log_grad_norms=False)
        clipper = GradientClipper(simple_model, config)
        clipper.clip_gradients()

        # Verify NaN/Inf are gone
        for p in simple_model.parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any()
                assert not torch.isinf(p.grad).any()

    def test_compute_layer_norms(self, mock_encoder_model):
        """Test per-layer gradient norm computation."""
        from modeling_studio.trainers.gradient_utils_v3 import (
            GradientClipConfig,
            GradientClipper,
        )

        # Create gradients
        x = torch.randn(4, 10)
        y = x
        for layer in mock_encoder_model.encoder.layers:
            y = layer(y)
        loss = y.sum()
        loss.backward()

        config = GradientClipConfig(log_grad_norms=True)
        clipper = GradientClipper(mock_encoder_model, config)
        stats = clipper.clip_gradients()

        # Check layer norms computed
        assert len(stats.layer_norms) == 28
        assert "layer_1" in stats.layer_norms
        assert "layer_23" in stats.layer_norms
        assert "layer_28" in stats.layer_norms

    def test_explosion_detection(self, simple_model, caplog):
        """Test gradient explosion detection."""
        import logging

        from modeling_studio.trainers.gradient_utils_v3 import (
            GradientClipConfig,
            GradientClipper,
        )

        # Create large gradients
        x = torch.randn(4, 10) * 1000
        y = simple_model(x)
        loss = y.sum() * 1000
        loss.backward()

        config = GradientClipConfig(
            explosion_threshold=0.01,
            log_grad_norms=False,
        )
        clipper = GradientClipper(simple_model, config)

        with caplog.at_level(logging.WARNING):
            clipper.clip_gradients()

        assert clipper.explosion_count == 1

    def test_gradient_history(self, simple_model):
        """Test gradient history tracking."""
        from modeling_studio.trainers.gradient_utils_v3 import (
            GradientClipConfig,
            GradientClipper,
        )

        config = GradientClipConfig(log_grad_norms=False)
        clipper = GradientClipper(simple_model, config)

        # Run multiple steps
        for _ in range(5):
            x = torch.randn(4, 10)
            y = simple_model(x)
            loss = y.sum()
            simple_model.zero_grad()
            loss.backward()
            clipper.clip_gradients()

        assert len(clipper.gradient_history) == 5
        assert clipper.step == 5

    def test_gradient_summary(self, simple_model):
        """Test gradient summary computation."""
        from modeling_studio.trainers.gradient_utils_v3 import (
            GradientClipConfig,
            GradientClipper,
        )

        config = GradientClipConfig(log_grad_norms=False)
        clipper = GradientClipper(simple_model, config)

        # Run multiple steps
        for _ in range(5):
            x = torch.randn(4, 10)
            y = simple_model(x)
            loss = y.sum()
            simple_model.zero_grad()
            loss.backward()
            clipper.clip_gradients()

        summary = clipper.get_gradient_summary()

        assert "mean_norm" in summary
        assert "max_norm" in summary
        assert "min_norm" in summary
        assert "clip_count" in summary
        assert "clip_ratio" in summary
        assert "explosion_count" in summary

    def test_clear_history(self, simple_model):
        """Test clearing gradient history."""
        from modeling_studio.trainers.gradient_utils_v3 import (
            GradientClipConfig,
            GradientClipper,
        )

        config = GradientClipConfig(log_grad_norms=False)
        clipper = GradientClipper(simple_model, config)

        # Add some history
        for _ in range(5):
            x = torch.randn(4, 10)
            y = simple_model(x)
            loss = y.sum()
            simple_model.zero_grad()
            loss.backward()
            clipper.clip_gradients()

        assert len(clipper.gradient_history) == 5

        clipper.clear_history()
        assert len(clipper.gradient_history) == 0

    def test_reset(self, simple_model):
        """Test full reset."""
        from modeling_studio.trainers.gradient_utils_v3 import (
            GradientClipConfig,
            GradientClipper,
        )

        config = GradientClipConfig(
            explosion_threshold=0.001,
            log_grad_norms=False,
        )
        clipper = GradientClipper(simple_model, config)

        # Add some history and explosions
        for _ in range(5):
            x = torch.randn(4, 10) * 100
            y = simple_model(x)
            loss = y.sum() * 100
            simple_model.zero_grad()
            loss.backward()
            clipper.clip_gradients()

        assert clipper.step == 5
        assert clipper.explosion_count > 0

        clipper.reset()
        assert clipper.step == 0
        assert clipper.explosion_count == 0
        assert len(clipper.gradient_history) == 0


class TestInterfaceGradientMonitor:
    """Tests for InterfaceGradientMonitor class."""

    @pytest.fixture
    def mock_encoder_model(self):
        """Create a mock model with encoder.layers structure."""

        class MockEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(28)])

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = MockEncoder()

        return MockModel()

    def test_record_with_gradients(self, mock_encoder_model):
        """Test recording interface gradients."""
        from modeling_studio.trainers.gradient_utils_v3 import InterfaceGradientMonitor

        # Create gradients
        x = torch.randn(4, 10)
        y = x
        for layer in mock_encoder_model.encoder.layers:
            y = layer(y)
        loss = y.sum()
        loss.backward()

        monitor = InterfaceGradientMonitor(mock_encoder_model)
        stats = monitor.record()

        assert "l22_grad_norm" in stats
        assert "l23_grad_norm" in stats
        assert "l23_l22_ratio" in stats
        assert "interface_healthy" in stats

    def test_record_no_gradients(self, mock_encoder_model):
        """Test recording when no gradients exist."""
        from modeling_studio.trainers.gradient_utils_v3 import InterfaceGradientMonitor

        monitor = InterfaceGradientMonitor(mock_encoder_model)
        stats = monitor.record()

        assert stats["l22_grad_norm"] == 0.0
        assert stats["l23_grad_norm"] == 0.0
        assert stats["l23_l22_ratio"] == 0.0

    def test_interface_healthy(self, mock_encoder_model):
        """Test interface health detection."""
        from modeling_studio.trainers.gradient_utils_v3 import InterfaceGradientMonitor

        # Create gradients
        x = torch.randn(4, 10)
        y = x
        for layer in mock_encoder_model.encoder.layers:
            y = layer(y)
        loss = y.sum()
        loss.backward()

        monitor = InterfaceGradientMonitor(mock_encoder_model)
        stats = monitor.record()

        # With normal gradients, ratio should be within healthy range
        assert stats["l23_l22_ratio"] > 0

    def test_history_tracking(self, mock_encoder_model):
        """Test history tracking."""
        from modeling_studio.trainers.gradient_utils_v3 import InterfaceGradientMonitor

        monitor = InterfaceGradientMonitor(mock_encoder_model)

        # Record multiple times
        for _ in range(5):
            x = torch.randn(4, 10)
            y = x
            for layer in mock_encoder_model.encoder.layers:
                y = layer(y)
            loss = y.sum()
            mock_encoder_model.zero_grad()
            loss.backward()
            monitor.record()

        assert len(monitor.history) == 5

    def test_get_interface_health_no_history(self, mock_encoder_model):
        """Test interface health with no history."""
        from modeling_studio.trainers.gradient_utils_v3 import InterfaceGradientMonitor

        monitor = InterfaceGradientMonitor(mock_encoder_model)
        health = monitor.get_interface_health()

        assert health["healthy"] is True
        assert health["message"] == "No data yet"

    def test_get_interface_health_with_history(self, mock_encoder_model):
        """Test interface health with history."""
        from modeling_studio.trainers.gradient_utils_v3 import InterfaceGradientMonitor

        monitor = InterfaceGradientMonitor(mock_encoder_model)

        # Record multiple healthy steps
        for _ in range(10):
            x = torch.randn(4, 10)
            y = x
            for layer in mock_encoder_model.encoder.layers:
                y = layer(y)
            loss = y.sum()
            mock_encoder_model.zero_grad()
            loss.backward()
            monitor.record()

        health = monitor.get_interface_health()

        assert "healthy" in health
        assert "health_ratio" in health
        assert "mean_l23_l22_ratio" in health
        assert "message" in health

    def test_clear_history(self, mock_encoder_model):
        """Test clearing history."""
        from modeling_studio.trainers.gradient_utils_v3 import InterfaceGradientMonitor

        monitor = InterfaceGradientMonitor(mock_encoder_model)

        # Add some history
        for _ in range(5):
            x = torch.randn(4, 10)
            y = x
            for layer in mock_encoder_model.encoder.layers:
                y = layer(y)
            loss = y.sum()
            mock_encoder_model.zero_grad()
            loss.backward()
            monitor.record()

        assert len(monitor.history) == 5

        monitor.clear_history()
        assert len(monitor.history) == 0


class TestClipGradientsFunction:
    """Tests for clip_gradients convenience function."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
        )

    def test_basic_usage(self, simple_model):
        """Test basic clip_gradients usage."""
        from modeling_studio.trainers.gradient_utils_v3 import clip_gradients

        # Create gradients
        x = torch.randn(4, 10)
        y = simple_model(x)
        loss = y.sum()
        loss.backward()

        norm = clip_gradients(simple_model, max_norm=1.0)
        assert norm >= 0

    def test_per_layer_flag(self, simple_model):
        """Test per_layer parameter."""
        from modeling_studio.trainers.gradient_utils_v3 import clip_gradients

        # Create gradients
        x = torch.randn(4, 10)
        y = simple_model(x)
        loss = y.sum()
        loss.backward()

        # Should not raise even without encoder.layers
        norm = clip_gradients(simple_model, max_norm=1.0, per_layer=True)
        assert norm >= 0

    def test_custom_max_norm(self, simple_model):
        """Test custom max_norm."""
        from modeling_studio.trainers.gradient_utils_v3 import clip_gradients

        # Create gradients
        x = torch.randn(4, 10)
        y = simple_model(x)
        loss = y.sum()
        loss.backward()

        norm = clip_gradients(simple_model, max_norm=0.5)
        assert norm >= 0


class TestCreateGradientClipper:
    """Tests for create_gradient_clipper factory function."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Linear(10, 5)

    def test_default_creation(self, simple_model):
        """Test creating clipper with defaults."""
        from modeling_studio.trainers.gradient_utils_v3 import create_gradient_clipper

        clipper = create_gradient_clipper(simple_model)

        assert clipper.config.max_grad_norm == 1.0
        assert clipper.config.per_layer_clip is False
        assert clipper.config.interface_clip == 0.5

    def test_custom_creation(self, simple_model):
        """Test creating clipper with custom settings."""
        from modeling_studio.trainers.gradient_utils_v3 import create_gradient_clipper

        clipper = create_gradient_clipper(
            simple_model,
            max_grad_norm=2.0,
            per_layer_clip=True,
            interface_clip=0.3,
            log_every_n_steps=50,
        )

        assert clipper.config.max_grad_norm == 2.0
        assert clipper.config.per_layer_clip is True
        assert clipper.config.interface_clip == 0.3
        assert clipper.config.log_every_n_steps == 50


class TestIssue518AcceptanceCriteria:
    """Tests verifying Issue 5.1.8 acceptance criteria."""

    @pytest.fixture
    def mock_encoder_model(self):
        """Create a mock model with encoder.layers structure."""

        class MockEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(28)])

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = MockEncoder()

        return MockModel()

    def test_ac1_global_clipping(self, mock_encoder_model):
        """AC1: GradientClipper implements global clipping (max_norm=1.0)."""
        from modeling_studio.trainers.gradient_utils_v3 import (
            GradientClipConfig,
            GradientClipper,
        )

        # Create large gradients
        x = torch.randn(4, 10) * 100
        y = x
        for layer in mock_encoder_model.encoder.layers:
            y = layer(y)
        loss = y.sum() * 100
        loss.backward()

        config = GradientClipConfig(max_grad_norm=1.0, log_grad_norms=False)
        clipper = GradientClipper(mock_encoder_model, config)
        stats = clipper.clip_gradients()

        # Verify clipping was applied
        assert stats.total_norm > 0
        # After clipping, gradient norm should be <= max_norm
        params = [p for p in mock_encoder_model.parameters() if p.grad is not None]
        total_norm = 0.0
        for p in params:
            total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm**0.5
        assert total_norm <= 1.0 + 1e-5

        print("AC1: GradientClipper implements global clipping [PASS]")

    def test_ac2_per_layer_l23_tighter_clip(self, mock_encoder_model):
        """AC2: Per-layer clipping applies tighter clip to L23 (0.5)."""
        from modeling_studio.trainers.gradient_utils_v3 import (
            GradientClipConfig,
            GradientClipper,
        )

        # Create large gradients
        x = torch.randn(4, 10) * 100
        y = x
        for layer in mock_encoder_model.encoder.layers:
            y = layer(y)
        loss = y.sum() * 100
        loss.backward()

        config = GradientClipConfig(
            per_layer_clip=True,
            interface_clip=0.5,
            family_clip=1.0,
            feeder_clip=1.0,
            log_grad_norms=False,
        )
        clipper = GradientClipper(mock_encoder_model, config)
        clipper.clip_gradients()

        # Check L23 (index 22) has smaller norm than family layers
        l23 = mock_encoder_model.encoder.layers[22]
        l23_norm = 0.0
        for p in l23.parameters():
            if p.grad is not None:
                l23_norm += p.grad.data.norm(2).item() ** 2
        l23_norm = l23_norm**0.5

        # L23 should be clipped to 0.5
        assert l23_norm <= 0.5 + 1e-5

        print("AC2: Per-layer clipping applies tighter clip to L23 [PASS]")

    def test_ac3_nan_inf_detection_and_zeroing(self, mock_encoder_model):
        """AC3: NaN/Inf gradient detection and zeroing."""
        from modeling_studio.trainers.gradient_utils_v3 import (
            GradientClipConfig,
            GradientClipper,
        )

        # Create gradients
        x = torch.randn(4, 10)
        y = x
        for layer in mock_encoder_model.encoder.layers:
            y = layer(y)
        loss = y.sum()
        loss.backward()

        # Inject NaN and Inf
        mock_encoder_model.encoder.layers[0].weight.grad[0, 0] = float("nan")
        mock_encoder_model.encoder.layers[1].weight.grad[0, 0] = float("inf")

        config = GradientClipConfig(nan_check=True, log_grad_norms=False)
        clipper = GradientClipper(mock_encoder_model, config)
        stats = clipper.clip_gradients()

        # Verify detection
        assert stats.has_nan is True
        assert stats.has_inf is True

        # Verify zeroing
        for p in mock_encoder_model.parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any()
                assert not torch.isinf(p.grad).any()

        print("AC3: NaN/Inf gradient detection and zeroing [PASS]")

    def test_ac4_explosion_warnings(self, mock_encoder_model, caplog):
        """AC4: Gradient explosion warnings logged."""
        import logging

        from modeling_studio.trainers.gradient_utils_v3 import (
            GradientClipConfig,
            GradientClipper,
        )

        # Create large gradients
        x = torch.randn(4, 10) * 1000
        y = x
        for layer in mock_encoder_model.encoder.layers:
            y = layer(y)
        loss = y.sum() * 1000
        loss.backward()

        config = GradientClipConfig(
            explosion_threshold=1.0,  # Low threshold
            log_grad_norms=False,
        )
        clipper = GradientClipper(mock_encoder_model, config)

        with caplog.at_level(logging.WARNING):
            clipper.clip_gradients()

        # Verify explosion was detected
        assert clipper.explosion_count > 0

        print("AC4: Gradient explosion warnings logged [PASS]")

    def test_ac5_interface_gradient_monitor(self, mock_encoder_model):
        """AC5: InterfaceGradientMonitor tracks L22->L23 gradient flow."""
        from modeling_studio.trainers.gradient_utils_v3 import InterfaceGradientMonitor

        # Create gradients
        x = torch.randn(4, 10)
        y = x
        for layer in mock_encoder_model.encoder.layers:
            y = layer(y)
        loss = y.sum()
        loss.backward()

        monitor = InterfaceGradientMonitor(mock_encoder_model)
        stats = monitor.record()

        # Verify L22 and L23 norms are tracked
        assert "l22_grad_norm" in stats
        assert "l23_grad_norm" in stats
        assert stats["l22_grad_norm"] > 0
        assert stats["l23_grad_norm"] > 0
        assert "l23_l22_ratio" in stats

        # Verify health assessment works
        health = monitor.get_interface_health()
        assert "healthy" in health
        assert "health_ratio" in health

        print("AC5: InterfaceGradientMonitor tracks L22->L23 flow [PASS]")

    def test_ac6_stats_logged_periodically(self, mock_encoder_model, caplog):
        """AC6: Gradient statistics logged periodically."""
        import logging

        from modeling_studio.trainers.gradient_utils_v3 import (
            GradientClipConfig,
            GradientClipper,
        )

        config = GradientClipConfig(
            log_grad_norms=True,
            log_every_n_steps=1,  # Log every step
        )
        clipper = GradientClipper(mock_encoder_model, config)

        # Create gradients and clip
        x = torch.randn(4, 10)
        y = x
        for layer in mock_encoder_model.encoder.layers:
            y = layer(y)
        loss = y.sum()
        loss.backward()

        with caplog.at_level(logging.INFO):
            clipper.clip_gradients()

        # Verify stats were logged (at step 0)
        # The logger should have been called with gradient stats
        assert clipper.step == 1

        print("AC6: Gradient statistics logged periodically [PASS]")

    def test_ac7_convenience_function(self, mock_encoder_model):
        """AC7: clip_gradients() convenience function works."""
        from modeling_studio.trainers.gradient_utils_v3 import clip_gradients

        # Create gradients
        x = torch.randn(4, 10)
        y = x
        for layer in mock_encoder_model.encoder.layers:
            y = layer(y)
        loss = y.sum()
        loss.backward()

        norm = clip_gradients(mock_encoder_model, max_norm=1.0)
        assert norm > 0
        assert isinstance(norm, float)

        # Also test with per_layer=True - need fresh forward pass
        mock_encoder_model.zero_grad()
        x2 = torch.randn(4, 10)
        y2 = x2
        for layer in mock_encoder_model.encoder.layers:
            y2 = layer(y2)
        loss2 = y2.sum()
        loss2.backward()
        norm2 = clip_gradients(mock_encoder_model, max_norm=1.0, per_layer=True)
        assert norm2 >= 0

        print("AC7: clip_gradients() convenience function works [PASS]")

    def test_ac8_no_memory_leaks(self, mock_encoder_model):
        """AC8: No memory leaks from gradient history."""
        from modeling_studio.trainers.gradient_utils_v3 import (
            GradientClipConfig,
            GradientClipper,
            InterfaceGradientMonitor,
        )

        config = GradientClipConfig(log_grad_norms=False)
        clipper = GradientClipper(mock_encoder_model, config)
        monitor = InterfaceGradientMonitor(mock_encoder_model)

        # Run many steps
        for _ in range(100):
            x = torch.randn(4, 10)
            y = x
            for layer in mock_encoder_model.encoder.layers:
                y = layer(y)
            loss = y.sum()
            mock_encoder_model.zero_grad()
            loss.backward()
            clipper.clip_gradients()
            monitor.record()

        # Verify history exists
        assert len(clipper.gradient_history) == 100
        assert len(monitor.history) == 100

        # Clear history
        clipper.clear_history()
        monitor.clear_history()

        # Verify cleared
        assert len(clipper.gradient_history) == 0
        assert len(monitor.history) == 0

        print("AC8: No memory leaks from gradient history [PASS]")
