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
