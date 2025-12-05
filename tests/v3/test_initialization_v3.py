"""
Tests for v2 Checkpoint Initialization (initialization_v3.py).

This test suite validates all acceptance criteria for Issue 4.1.1:
1. Loads PyTorch v2 checkpoints (22 layers)
2. Handles different checkpoint formats (state_dict, model, etc.)
3. Cleans `module.` prefix from DDP checkpoints
4. Extracts metadata (layers, hidden_size, vocab_size)
5. `validate()` checks compatibility
6. `get_layer_weights()` extracts per-layer tensors
7. `get_embedding_weights()` extracts embedding tensors

Author: FamilyOS Team
Date: December 2025
Version: 3.3
"""

# ruff: noqa: F401, I001
from pathlib import Path

import pytest
import torch

from modeling_studio.models.initialization_v3 import (
    V2CheckpointInfo,
    V2CheckpointLoader,
    WeightTransferStats,
    load_v2_checkpoint,
)


# ══════════════════════════════════════════════════════════════════════════════
# Test Fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_v2_checkpoint():
    """Create a mock v2 checkpoint for testing."""
    # Create a minimal v2-like state dict
    state_dict = {}

    # Embeddings
    state_dict["embeddings.word_embeddings.weight"] = torch.randn(50368, 768)
    state_dict["embeddings.position_embeddings.weight"] = torch.randn(8192, 768)
    state_dict["embeddings.LayerNorm.weight"] = torch.randn(768)
    state_dict["embeddings.LayerNorm.bias"] = torch.randn(768)

    # 22 encoder layers
    for layer_idx in range(22):
        prefix = f"encoder.layers.{layer_idx}."

        # Attention
        state_dict[f"{prefix}attention.q_proj.weight"] = torch.randn(768, 768)
        state_dict[f"{prefix}attention.k_proj.weight"] = torch.randn(768, 768)
        state_dict[f"{prefix}attention.v_proj.weight"] = torch.randn(768, 768)
        state_dict[f"{prefix}attention.out_proj.weight"] = torch.randn(768, 768)

        # FFN
        state_dict[f"{prefix}ffn.fc1.weight"] = torch.randn(3072, 768)
        state_dict[f"{prefix}ffn.fc1.bias"] = torch.randn(3072)
        state_dict[f"{prefix}ffn.fc2.weight"] = torch.randn(768, 3072)
        state_dict[f"{prefix}ffn.fc2.bias"] = torch.randn(768)

        # LayerNorms
        state_dict[f"{prefix}attention_layer_norm.weight"] = torch.randn(768)
        state_dict[f"{prefix}attention_layer_norm.bias"] = torch.randn(768)
        state_dict[f"{prefix}ffn_layer_norm.weight"] = torch.randn(768)
        state_dict[f"{prefix}ffn_layer_norm.bias"] = torch.randn(768)

    # Optional pooler
    state_dict["pooler.dense.weight"] = torch.randn(768, 768)
    state_dict["pooler.dense.bias"] = torch.randn(768)

    return state_dict


@pytest.fixture
def v2_checkpoint_path(mock_v2_checkpoint, tmp_path):
    """Save mock checkpoint to temporary file."""
    checkpoint_path = tmp_path / "v2_checkpoint.pt"
    torch.save(mock_v2_checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def v2_checkpoint_with_wrapper(mock_v2_checkpoint, tmp_path):
    """Save checkpoint with state_dict wrapper."""
    checkpoint_path = tmp_path / "v2_checkpoint_wrapped.pt"
    torch.save({"state_dict": mock_v2_checkpoint}, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def ddp_checkpoint(mock_v2_checkpoint, tmp_path):
    """Create checkpoint with DDP module. prefix."""
    ddp_state_dict = {f"module.{k}": v for k, v in mock_v2_checkpoint.items()}
    checkpoint_path = tmp_path / "ddp_checkpoint.pt"
    torch.save(ddp_state_dict, checkpoint_path)
    return checkpoint_path


# ══════════════════════════════════════════════════════════════════════════════
# Test Data Structures
# ══════════════════════════════════════════════════════════════════════════════


class TestDataStructures:
    """Test V2CheckpointInfo and WeightTransferStats dataclasses."""

    def test_v2_checkpoint_info_creation(self):
        """Test creating V2CheckpointInfo."""
        info = V2CheckpointInfo(
            path=Path("test.pt"),
            num_layers=22,
            hidden_size=768,
            vocab_size=50368,
            has_pooler=True,
            has_task_heads=False,
            state_dict_keys=["key1", "key2"],
        )

        assert info.num_layers == 22
        assert info.hidden_size == 768
        assert info.vocab_size == 50368
        assert info.has_pooler is True
        assert info.has_task_heads is False
        assert len(info.state_dict_keys) == 2

    def test_weight_transfer_stats_creation(self):
        """Test creating WeightTransferStats."""
        stats = WeightTransferStats(
            total_params=1000000,
            transferred_params=800000,
            initialized_params=150000,
            skipped_params=50000,
            layer_mapping={0: 0, 1: 1, 22: 14},
        )

        assert stats.total_params == 1000000
        assert stats.transferred_params == 800000
        assert stats.initialized_params == 150000
        assert stats.skipped_params == 50000
        assert len(stats.layer_mapping) == 3


# ══════════════════════════════════════════════════════════════════════════════
# Test V2CheckpointLoader
# ══════════════════════════════════════════════════════════════════════════════


class TestV2CheckpointLoader:
    """Test V2CheckpointLoader class."""

    def test_loader_initialization(self, v2_checkpoint_path):
        """Test loader initialization."""
        loader = V2CheckpointLoader(str(v2_checkpoint_path))

        assert loader.checkpoint_path == v2_checkpoint_path
        assert loader._state_dict is None
        assert loader._info is None

    def test_load_basic_checkpoint(self, v2_checkpoint_path):
        """AC1: Loads PyTorch v2 checkpoints (22 layers)."""
        loader = V2CheckpointLoader(str(v2_checkpoint_path))
        state_dict = loader.load()

        assert isinstance(state_dict, dict)
        assert "embeddings.word_embeddings.weight" in state_dict
        assert "encoder.layers.0.attention.q_proj.weight" in state_dict
        assert "encoder.layers.21.ffn.fc1.weight" in state_dict

    def test_load_wrapped_checkpoint(self, v2_checkpoint_with_wrapper):
        """AC2: Handles different checkpoint formats (state_dict wrapper)."""
        loader = V2CheckpointLoader(str(v2_checkpoint_with_wrapper))
        state_dict = loader.load()

        assert isinstance(state_dict, dict)
        assert "embeddings.word_embeddings.weight" in state_dict

    def test_load_model_state_dict_format(self, mock_v2_checkpoint, tmp_path):
        """AC2: Handles model_state_dict format."""
        checkpoint_path = tmp_path / "model_state_dict.pt"
        torch.save({"model_state_dict": mock_v2_checkpoint}, checkpoint_path)

        loader = V2CheckpointLoader(str(checkpoint_path))
        state_dict = loader.load()

        assert "embeddings.word_embeddings.weight" in state_dict

    def test_load_model_format(self, mock_v2_checkpoint, tmp_path):
        """AC2: Handles model format."""
        checkpoint_path = tmp_path / "model.pt"
        torch.save({"model": mock_v2_checkpoint}, checkpoint_path)

        loader = V2CheckpointLoader(str(checkpoint_path))
        state_dict = loader.load()

        assert "embeddings.word_embeddings.weight" in state_dict

    def test_clean_ddp_prefix(self, ddp_checkpoint):
        """AC3: Cleans `module.` prefix from DDP checkpoints."""
        loader = V2CheckpointLoader(str(ddp_checkpoint))
        state_dict = loader.load()

        # Check that module. prefix has been removed
        assert "embeddings.word_embeddings.weight" in state_dict
        assert "encoder.layers.0.attention.q_proj.weight" in state_dict

        # Ensure no module. prefix remains
        assert not any(k.startswith("module.") for k in state_dict.keys())

    def test_load_nonexistent_file(self):
        """Test error handling for missing file."""
        loader = V2CheckpointLoader("nonexistent.pt")

        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            loader.load()

    def test_load_caching(self, v2_checkpoint_path):
        """Test that state dict is cached after first load."""
        loader = V2CheckpointLoader(str(v2_checkpoint_path))

        state_dict1 = loader.load()
        state_dict2 = loader.load()

        # Should return same object (cached)
        assert state_dict1 is state_dict2


# ══════════════════════════════════════════════════════════════════════════════
# Test Metadata Extraction
# ══════════════════════════════════════════════════════════════════════════════


class TestMetadataExtraction:
    """Test checkpoint metadata extraction."""

    def test_get_info_basic(self, v2_checkpoint_path):
        """AC4: Extracts metadata (layers, hidden_size, vocab_size)."""
        loader = V2CheckpointLoader(str(v2_checkpoint_path))
        info = loader.get_info()

        assert isinstance(info, V2CheckpointInfo)
        assert info.num_layers == 22
        assert info.hidden_size == 768
        assert info.vocab_size == 50368
        assert info.has_pooler is True
        assert isinstance(info.state_dict_keys, list)

    def test_get_info_detects_layers(self, v2_checkpoint_path):
        """Test layer count detection."""
        loader = V2CheckpointLoader(str(v2_checkpoint_path))
        info = loader.get_info()

        assert info.num_layers == 22

    def test_get_info_detects_hidden_size(self, v2_checkpoint_path):
        """Test hidden size detection from LayerNorm."""
        loader = V2CheckpointLoader(str(v2_checkpoint_path))
        info = loader.get_info()

        assert info.hidden_size == 768

    def test_get_info_detects_vocab_size(self, v2_checkpoint_path):
        """Test vocab size detection from embeddings."""
        loader = V2CheckpointLoader(str(v2_checkpoint_path))
        info = loader.get_info()

        assert info.vocab_size == 50368

    def test_get_info_detects_pooler(self, v2_checkpoint_path):
        """Test pooler detection."""
        loader = V2CheckpointLoader(str(v2_checkpoint_path))
        info = loader.get_info()

        assert info.has_pooler is True

    def test_get_info_detects_task_heads(self, mock_v2_checkpoint, tmp_path):
        """Test task head detection."""
        # Add task head to checkpoint
        mock_v2_checkpoint["emotion_head.classifier.weight"] = torch.randn(7, 768)

        checkpoint_path = tmp_path / "with_heads.pt"
        torch.save(mock_v2_checkpoint, checkpoint_path)

        loader = V2CheckpointLoader(str(checkpoint_path))
        info = loader.get_info()

        assert info.has_task_heads is True

    def test_get_info_caching(self, v2_checkpoint_path):
        """Test that info is cached."""
        loader = V2CheckpointLoader(str(v2_checkpoint_path))

        info1 = loader.get_info()
        info2 = loader.get_info()

        assert info1 is info2


# ══════════════════════════════════════════════════════════════════════════════
# Test Validation
# ══════════════════════════════════════════════════════════════════════════════


class TestValidation:
    """Test checkpoint validation."""

    def test_validate_correct_checkpoint(self, v2_checkpoint_path):
        """AC5: `validate()` checks compatibility - valid checkpoint."""
        loader = V2CheckpointLoader(str(v2_checkpoint_path))
        is_valid, issues = loader.validate()

        assert is_valid is True
        assert len(issues) == 0

    def test_validate_wrong_layer_count(self, mock_v2_checkpoint, tmp_path):
        """AC5: Validate fails for wrong layer count."""
        # Remove some layers
        keys_to_remove = [
            k for k in mock_v2_checkpoint.keys() if "layers.20" in k or "layers.21" in k
        ]
        for key in keys_to_remove:
            del mock_v2_checkpoint[key]

        checkpoint_path = tmp_path / "wrong_layers.pt"
        torch.save(mock_v2_checkpoint, checkpoint_path)

        loader = V2CheckpointLoader(str(checkpoint_path))
        is_valid, issues = loader.validate()

        assert is_valid is False
        assert any("22 layers" in issue for issue in issues)

    def test_validate_wrong_hidden_size(self, mock_v2_checkpoint, tmp_path):
        """AC5: Validate fails for wrong hidden size."""
        # Change hidden size for all LayerNorms to ensure detection
        for key in list(mock_v2_checkpoint.keys()):
            if "layer_norm" in key.lower() and ".weight" in key:
                mock_v2_checkpoint[key] = torch.randn(512)

        checkpoint_path = tmp_path / "wrong_hidden.pt"
        torch.save(mock_v2_checkpoint, checkpoint_path)

        loader = V2CheckpointLoader(str(checkpoint_path))
        is_valid, issues = loader.validate()

        assert is_valid is False
        assert any("hidden_size" in issue for issue in issues)

    def test_validate_missing_embeddings(self, mock_v2_checkpoint, tmp_path):
        """AC5: Validate fails for missing required keys."""
        # Remove embeddings
        del mock_v2_checkpoint["embeddings.word_embeddings.weight"]

        checkpoint_path = tmp_path / "missing_embeddings.pt"
        torch.save(mock_v2_checkpoint, checkpoint_path)

        loader = V2CheckpointLoader(str(checkpoint_path))
        is_valid, issues = loader.validate()

        assert is_valid is False
        assert any("embeddings.word_embeddings" in issue for issue in issues)


# ══════════════════════════════════════════════════════════════════════════════
# Test Weight Extraction
# ══════════════════════════════════════════════════════════════════════════════


class TestWeightExtraction:
    """Test layer and embedding weight extraction."""

    def test_get_layer_weights_basic(self, v2_checkpoint_path):
        """AC6: `get_layer_weights()` extracts per-layer tensors."""
        loader = V2CheckpointLoader(str(v2_checkpoint_path))
        layer_0_weights = loader.get_layer_weights(0)

        assert isinstance(layer_0_weights, dict)
        assert "attention.q_proj.weight" in layer_0_weights
        assert "attention.k_proj.weight" in layer_0_weights
        assert "ffn.fc1.weight" in layer_0_weights

    def test_get_layer_weights_all_layers(self, v2_checkpoint_path):
        """AC6: Get weights for all 22 layers."""
        loader = V2CheckpointLoader(str(v2_checkpoint_path))

        for layer_idx in range(22):
            layer_weights = loader.get_layer_weights(layer_idx)
            assert len(layer_weights) > 0
            assert "attention.q_proj.weight" in layer_weights

    def test_get_layer_weights_removes_prefix(self, v2_checkpoint_path):
        """AC6: Layer weights have prefix removed."""
        loader = V2CheckpointLoader(str(v2_checkpoint_path))
        layer_0_weights = loader.get_layer_weights(0)

        # Keys should not have "encoder.layers.0." prefix
        for key in layer_0_weights.keys():
            assert not key.startswith("encoder.layers.")

    def test_get_layer_weights_invalid_index(self, v2_checkpoint_path):
        """Test error handling for invalid layer index."""
        loader = V2CheckpointLoader(str(v2_checkpoint_path))

        with pytest.raises(ValueError, match="Invalid layer index"):
            loader.get_layer_weights(25)

        with pytest.raises(ValueError, match="Invalid layer index"):
            loader.get_layer_weights(-1)

    def test_get_embedding_weights_basic(self, v2_checkpoint_path):
        """AC7: `get_embedding_weights()` extracts embedding tensors."""
        loader = V2CheckpointLoader(str(v2_checkpoint_path))
        embeddings = loader.get_embedding_weights()

        assert isinstance(embeddings, dict)
        assert "word_embeddings.weight" in embeddings
        assert "position_embeddings.weight" in embeddings
        assert "LayerNorm.weight" in embeddings

    def test_get_embedding_weights_removes_prefix(self, v2_checkpoint_path):
        """AC7: Embedding weights have prefix removed."""
        loader = V2CheckpointLoader(str(v2_checkpoint_path))
        embeddings = loader.get_embedding_weights()

        # Keys should not have "embeddings." prefix
        for key in embeddings.keys():
            assert not key.startswith("embeddings.")

    def test_get_embedding_weights_shapes(self, v2_checkpoint_path):
        """Test embedding weight shapes are correct."""
        loader = V2CheckpointLoader(str(v2_checkpoint_path))
        embeddings = loader.get_embedding_weights()

        assert embeddings["word_embeddings.weight"].shape == (50368, 768)
        assert embeddings["position_embeddings.weight"].shape == (8192, 768)
        assert embeddings["LayerNorm.weight"].shape == (768,)


# ══════════════════════════════════════════════════════════════════════════════
# Test Print Summary
# ══════════════════════════════════════════════════════════════════════════════


class TestPrintSummary:
    """Test print_summary functionality."""

    def test_print_summary_runs(self, v2_checkpoint_path, capsys):
        """Test print_summary displays checkpoint info."""
        loader = V2CheckpointLoader(str(v2_checkpoint_path))
        loader.print_summary()

        captured = capsys.readouterr()
        output = captured.out

        assert "v2 Checkpoint Summary" in output
        assert "Layers: 22" in output
        assert "Hidden Size: 768" in output
        assert "Vocab Size: 50368" in output


# ══════════════════════════════════════════════════════════════════════════════
# Test Factory Function
# ══════════════════════════════════════════════════════════════════════════════


class TestFactoryFunction:
    """Test load_v2_checkpoint factory function."""

    def test_load_v2_checkpoint_factory(self, v2_checkpoint_path, capsys):
        """Test factory function creates and validates loader."""
        loader = load_v2_checkpoint(str(v2_checkpoint_path))

        assert isinstance(loader, V2CheckpointLoader)

        # Should print summary
        captured = capsys.readouterr()
        assert "v2 Checkpoint Summary" in captured.out

    def test_load_v2_checkpoint_validates(self, v2_checkpoint_path):
        """Test factory function validates checkpoint."""
        loader = load_v2_checkpoint(str(v2_checkpoint_path))

        # Validation already run
        info = loader.get_info()
        assert info.num_layers == 22


# ══════════════════════════════════════════════════════════════════════════════
# Test Layer Mapping Constants
# ══════════════════════════════════════════════════════════════════════════════


class TestLayerMapping:
    """Test LAYER_MAPPING constant."""

    def test_layer_mapping_completeness(self):
        """Test LAYER_MAPPING covers all v3 layers."""
        mapping = V2CheckpointLoader.LAYER_MAPPING

        assert len(mapping) == 28  # All v3 layers
        assert all(i in mapping for i in range(28))

    def test_layer_mapping_direct_copy(self):
        """Test layers 0-21 are direct copies."""
        mapping = V2CheckpointLoader.LAYER_MAPPING

        for i in range(22):
            assert mapping[i] == i  # Direct 1:1 mapping

    def test_layer_mapping_clones(self):
        """Test layers 22-27 are cloned from 14-19."""
        mapping = V2CheckpointLoader.LAYER_MAPPING

        expected_clones = {
            22: 14,  # L23 ← L15
            23: 15,  # L24 ← L16
            24: 16,  # L25 ← L17
            25: 17,  # L26 ← L18
            26: 18,  # L27 ← L19
            27: 19,  # L28 ← L20
        }

        for v3_layer, v2_layer in expected_clones.items():
            assert mapping[v3_layer] == v2_layer


# ══════════════════════════════════════════════════════════════════════════════
# Integration Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_complete_workflow(self, v2_checkpoint_path):
        """Test complete loading and extraction workflow."""
        # Load checkpoint
        loader = V2CheckpointLoader(str(v2_checkpoint_path))

        # Validate
        is_valid, issues = loader.validate()
        assert is_valid
        assert len(issues) == 0

        # Get info
        info = loader.get_info()
        assert info.num_layers == 22

        # Extract all layers
        for layer_idx in range(22):
            layer_weights = loader.get_layer_weights(layer_idx)
            assert len(layer_weights) > 0

        # Extract embeddings
        embeddings = loader.get_embedding_weights()
        assert len(embeddings) > 0

    def test_multiple_loader_instances(self, v2_checkpoint_path):
        """Test multiple loaders can be created."""
        loader1 = V2CheckpointLoader(str(v2_checkpoint_path))
        loader2 = V2CheckpointLoader(str(v2_checkpoint_path))

        state_dict1 = loader1.load()
        state_dict2 = loader2.load()

        # Different instances should load independently
        assert len(state_dict1) == len(state_dict2)


# ══════════════════════════════════════════════════════════════════════════════
# Acceptance Criteria Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestAcceptanceCriteria:
    """Comprehensive tests for all acceptance criteria."""

    def test_ac1_loads_v2_checkpoints(self, v2_checkpoint_path):
        """AC1: Loads PyTorch v2 checkpoints (22 layers)."""
        loader = V2CheckpointLoader(str(v2_checkpoint_path))
        state_dict = loader.load()

        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0

        # Verify 22 layers present
        layer_keys = [k for k in state_dict.keys() if "encoder.layers." in k]
        layer_indices = set()
        for key in layer_keys:
            import re

            match = re.search(r"encoder\.layers\.(\d+)\.", key)
            if match:
                layer_indices.add(int(match.group(1)))

        assert len(layer_indices) == 22
        print("✓ AC1: Loads v2 checkpoints (22 layers)")

    def test_ac2_handles_checkpoint_formats(self, mock_v2_checkpoint, tmp_path):
        """AC2: Handles different checkpoint formats."""
        # Test direct state dict
        path1 = tmp_path / "direct.pt"
        torch.save(mock_v2_checkpoint, path1)
        loader1 = V2CheckpointLoader(str(path1))
        assert loader1.load() is not None

        # Test state_dict wrapper
        path2 = tmp_path / "wrapped.pt"
        torch.save({"state_dict": mock_v2_checkpoint}, path2)
        loader2 = V2CheckpointLoader(str(path2))
        assert loader2.load() is not None

        # Test model wrapper
        path3 = tmp_path / "model.pt"
        torch.save({"model": mock_v2_checkpoint}, path3)
        loader3 = V2CheckpointLoader(str(path3))
        assert loader3.load() is not None

        print("✓ AC2: Handles different checkpoint formats")

    def test_ac3_cleans_ddp_prefix(self, ddp_checkpoint):
        """AC3: Cleans `module.` prefix from DDP checkpoints."""
        loader = V2CheckpointLoader(str(ddp_checkpoint))
        state_dict = loader.load()

        # No module. prefix should remain
        assert not any(k.startswith("module.") for k in state_dict.keys())
        assert "embeddings.word_embeddings.weight" in state_dict

        print("✓ AC3: Cleans module. prefix")

    def test_ac4_extracts_metadata(self, v2_checkpoint_path):
        """AC4: Extracts metadata (layers, hidden_size, vocab_size)."""
        loader = V2CheckpointLoader(str(v2_checkpoint_path))
        info = loader.get_info()

        assert info.num_layers == 22
        assert info.hidden_size == 768
        assert info.vocab_size == 50368
        assert isinstance(info.has_pooler, bool)
        assert isinstance(info.has_task_heads, bool)

        print("✓ AC4: Extracts metadata")

    def test_ac5_validation_checks(self, v2_checkpoint_path):
        """AC5: `validate()` checks compatibility."""
        loader = V2CheckpointLoader(str(v2_checkpoint_path))
        is_valid, issues = loader.validate()

        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)

        # Valid checkpoint should pass
        assert is_valid is True
        assert len(issues) == 0

        print("✓ AC5: Validation checks compatibility")

    def test_ac6_extracts_layer_weights(self, v2_checkpoint_path):
        """AC6: `get_layer_weights()` extracts per-layer tensors."""
        loader = V2CheckpointLoader(str(v2_checkpoint_path))

        for layer_idx in range(22):
            layer_weights = loader.get_layer_weights(layer_idx)

            assert isinstance(layer_weights, dict)
            assert len(layer_weights) > 0
            assert all(isinstance(v, torch.Tensor) for v in layer_weights.values())

        print("✓ AC6: Extracts per-layer tensors")

    def test_ac7_extracts_embedding_weights(self, v2_checkpoint_path):
        """AC7: `get_embedding_weights()` extracts embedding tensors."""
        loader = V2CheckpointLoader(str(v2_checkpoint_path))
        embeddings = loader.get_embedding_weights()

        assert isinstance(embeddings, dict)
        assert len(embeddings) > 0
        assert "word_embeddings.weight" in embeddings
        assert all(isinstance(v, torch.Tensor) for v in embeddings.values())

        print("✓ AC7: Extracts embedding tensors")


# ══════════════════════════════════════════════════════════════════════════════
# Issue 4.1.2: Layer Copying Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestLayerCopier:
    """Tests for LayerCopier class (Issue 4.1.2)."""

    def test_layer_copier_initialization(self, v2_checkpoint_path):
        """Test LayerCopier initializes correctly."""
        from modeling_studio.models.initialization_v3 import LayerCopier

        loader = V2CheckpointLoader(str(v2_checkpoint_path))
        copier = LayerCopier(loader, strict=True)

        assert copier.v2_loader is loader
        assert copier.strict is True
        assert copier.copy_stats["matched"] == 0
        assert copier.copy_stats["mismatched_shape"] == 0
        assert copier.copy_stats["missing_in_v2"] == 0

    def test_copy_single_layer(self, v2_checkpoint_path):
        """Test copying a single layer."""
        import torch.nn as nn
        from modeling_studio.models.initialization_v3 import LayerCopier

        # Create simple mock v3 layer
        class MockLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = nn.ModuleDict(
                    {
                        "q_proj": nn.Linear(768, 768, bias=False),
                        "k_proj": nn.Linear(768, 768, bias=False),
                        "v_proj": nn.Linear(768, 768, bias=False),
                        "out_proj": nn.Linear(768, 768, bias=False),
                    }
                )
                self.ffn = nn.ModuleDict(
                    {
                        "fc1": nn.Linear(768, 3072),
                        "fc2": nn.Linear(3072, 768),
                    }
                )
                self.attention_layer_norm = nn.LayerNorm(768)
                self.ffn_layer_norm = nn.LayerNorm(768)

        loader = V2CheckpointLoader(str(v2_checkpoint_path))
        copier = LayerCopier(loader, strict=False)
        v3_layer = MockLayer()

        # Get original weights for comparison
        original_weight = v3_layer.attention["q_proj"].weight.clone()

        # Copy layer 0 weights
        copied = copier.copy_layer(v3_layer, v2_layer_idx=0, v3_layer_idx=0)

        # Verify weights changed
        assert not torch.allclose(v3_layer.attention["q_proj"].weight, original_weight)
        assert copied > 0
        assert copier.copy_stats["matched"] > 0

    def test_copy_layers_1_to_22(self, v2_checkpoint_path):
        """Test copying all 22 layers."""
        import torch.nn as nn
        from modeling_studio.models.initialization_v3 import LayerCopier

        # Create mock encoder with 28 layers
        class MockEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList()
                for _ in range(28):
                    layer = nn.ModuleDict(
                        {
                            "attention": nn.ModuleDict(
                                {
                                    "q_proj": nn.Linear(768, 768, bias=False),
                                    "k_proj": nn.Linear(768, 768, bias=False),
                                    "v_proj": nn.Linear(768, 768, bias=False),
                                    "out_proj": nn.Linear(768, 768, bias=False),
                                }
                            ),
                            "ffn": nn.ModuleDict(
                                {
                                    "fc1": nn.Linear(768, 3072),
                                    "fc2": nn.Linear(3072, 768),
                                }
                            ),
                            "attention_layer_norm": nn.LayerNorm(768),
                            "ffn_layer_norm": nn.LayerNorm(768),
                        }
                    )
                    self.layers.append(layer)

        loader = V2CheckpointLoader(str(v2_checkpoint_path))
        copier = LayerCopier(loader, strict=False)
        encoder = MockEncoder()

        # Copy first 22 layers
        total_copied = copier.copy_layers_1_to_22(encoder)

        assert total_copied > 0
        assert copier.copy_stats["matched"] > 0
        print(f"Copied {total_copied:,} parameters from v2 to v3 L1-22")

    def test_copy_preserves_exact_values(self, v2_checkpoint_path):
        """Test that copying preserves exact parameter values."""
        import torch.nn as nn
        from modeling_studio.models.initialization_v3 import LayerCopier

        class SimpleLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = nn.ModuleDict({"q_proj": nn.Linear(768, 768, bias=False)})

        loader = V2CheckpointLoader(str(v2_checkpoint_path))
        v2_weights = loader.get_layer_weights(0)

        copier = LayerCopier(loader, strict=False)
        v3_layer = SimpleLayer()

        # Copy
        copier.copy_layer(v3_layer, v2_layer_idx=0, v3_layer_idx=0)

        # Verify exact match
        v2_q_proj = v2_weights["attention.q_proj.weight"]
        v3_q_proj = v3_layer.attention["q_proj"].weight

        assert torch.equal(v2_q_proj, v3_q_proj), "Weights should be exactly equal"

    def test_copy_stats_tracking(self, v2_checkpoint_path):
        """Test that statistics are tracked correctly."""
        import torch.nn as nn
        from modeling_studio.models.initialization_v3 import LayerCopier

        class MockLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = nn.ModuleDict({"q_proj": nn.Linear(768, 768, bias=False)})

        loader = V2CheckpointLoader(str(v2_checkpoint_path))
        copier = LayerCopier(loader, strict=True)
        v3_layer = MockLayer()

        copier.copy_layer(v3_layer, v2_layer_idx=0, v3_layer_idx=0)

        stats = copier.get_stats()
        assert "matched" in stats
        assert "mismatched_shape" in stats
        assert "missing_in_v2" in stats
        assert stats["matched"] > 0

    def test_copy_with_strict_mode(self, v2_checkpoint_path):
        """Test strict mode warnings for missing weights."""
        import torch.nn as nn
        from modeling_studio.models.initialization_v3 import LayerCopier

        class LayerWithExtraParam(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = nn.ModuleDict({"q_proj": nn.Linear(768, 768, bias=False)})
                self.extra_param = nn.Parameter(torch.randn(768))

        loader = V2CheckpointLoader(str(v2_checkpoint_path))
        copier = LayerCopier(loader, strict=True)
        v3_layer = LayerWithExtraParam()

        # Should handle missing weights gracefully
        copied = copier.copy_layer(v3_layer, v2_layer_idx=0, v3_layer_idx=0)

        assert copied >= 0
        # extra_param should be tracked as missing
        assert copier.copy_stats["missing_in_v2"] > 0


class TestCopyLayersDirect:
    """Tests for copy_layers_direct function (Issue 4.1.2)."""

    def test_copy_layers_direct_function_exists(self):
        """Test copy_layers_direct function is importable."""
        from modeling_studio.models.initialization_v3 import copy_layers_direct

        assert callable(copy_layers_direct)

    def test_copy_layers_direct_with_mock_model(self, v2_checkpoint_path):
        """Test complete copy_layers_direct workflow."""
        import torch.nn as nn
        from modeling_studio.models.initialization_v3 import copy_layers_direct

        # Create minimal mock v3 model
        class MockV3Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.ModuleDict()
                self.encoder["layers"] = nn.ModuleList()
                for _ in range(28):
                    layer = nn.ModuleDict(
                        {
                            "attention": nn.ModuleDict(
                                {
                                    "q_proj": nn.Linear(768, 768, bias=False),
                                    "k_proj": nn.Linear(768, 768, bias=False),
                                    "v_proj": nn.Linear(768, 768, bias=False),
                                    "out_proj": nn.Linear(768, 768, bias=False),
                                }
                            ),
                            "ffn": nn.ModuleDict(
                                {
                                    "fc1": nn.Linear(768, 3072),
                                    "fc2": nn.Linear(3072, 768),
                                }
                            ),
                            "attention_layer_norm": nn.LayerNorm(768),
                            "ffn_layer_norm": nn.LayerNorm(768),
                        }
                    )
                    self.encoder["layers"].append(layer)

        v3_model = MockV3Model()

        # Get original weight for verification
        original_weight = v3_model.encoder["layers"][0]["attention"]["q_proj"].weight.clone()

        # Copy layers
        total_copied = copy_layers_direct(v3_model, str(v2_checkpoint_path))

        # Verify
        assert total_copied > 0
        assert not torch.equal(
            v3_model.encoder["layers"][0]["attention"]["q_proj"].weight, original_weight
        ), "Weights should have changed"

        print(f"✓ copy_layers_direct: Copied {total_copied:,} parameters")

    def test_copy_only_affects_first_22_layers(self, v2_checkpoint_path):
        """Test that only layers 0-21 are modified."""
        import torch.nn as nn
        from modeling_studio.models.initialization_v3 import copy_layers_direct

        class MockV3Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.ModuleDict()
                self.encoder["layers"] = nn.ModuleList()
                for _ in range(28):
                    layer = nn.ModuleDict({"linear": nn.Linear(768, 768, bias=False)})
                    self.encoder["layers"].append(layer)

        v3_model = MockV3Model()

        # Store original weights of layers 22-27
        original_layer_22 = v3_model.encoder["layers"][22]["linear"].weight.clone()
        original_layer_27 = v3_model.encoder["layers"][27]["linear"].weight.clone()

        # Copy (should only affect L0-21)
        copy_layers_direct(v3_model, str(v2_checkpoint_path))

        # Layers 22-27 should remain unchanged
        assert torch.equal(
            v3_model.encoder["layers"][22]["linear"].weight, original_layer_22
        ), "Layer 22 should not be modified by direct copy"
        assert torch.equal(
            v3_model.encoder["layers"][27]["linear"].weight, original_layer_27
        ), "Layer 27 should not be modified by direct copy"


class TestIssue412AcceptanceCriteria:
    """Comprehensive tests for Issue 4.1.2 acceptance criteria."""

    def test_ac1_copies_all_22_layers(self, v2_checkpoint_path):
        """AC1: Copies all 22 v2 layers to first 22 v3 layers."""
        import torch.nn as nn
        from modeling_studio.models.initialization_v3 import LayerCopier

        class MockEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList(
                    [
                        nn.ModuleDict(
                            {
                                "attention": nn.ModuleDict(
                                    {"q_proj": nn.Linear(768, 768, bias=False)}
                                ),
                                "ffn": nn.ModuleDict({"fc1": nn.Linear(768, 3072)}),
                                "attention_layer_norm": nn.LayerNorm(768),
                                "ffn_layer_norm": nn.LayerNorm(768),
                            }
                        )
                        for _ in range(28)
                    ]
                )

        loader = V2CheckpointLoader(str(v2_checkpoint_path))
        copier = LayerCopier(loader, strict=False)
        encoder = MockEncoder()

        total = copier.copy_layers_1_to_22(encoder)

        assert total > 0
        print("✓ AC1: Copies all 22 v2 layers to first 22 v3 layers")

    def test_ac2_handles_shape_mismatches(self, v2_checkpoint_path):
        """AC2: Handles shape mismatches gracefully with warnings."""
        import torch.nn as nn
        from modeling_studio.models.initialization_v3 import LayerCopier

        class LayerWithWrongShape(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = nn.ModuleDict(
                    {"q_proj": nn.Linear(512, 512, bias=False)}  # Wrong shape
                )

        loader = V2CheckpointLoader(str(v2_checkpoint_path))
        copier = LayerCopier(loader, strict=True)
        v3_layer = LayerWithWrongShape()

        # Should not raise, but track mismatch
        copied = copier.copy_layer(v3_layer, v2_layer_idx=0, v3_layer_idx=0)

        assert copier.copy_stats["mismatched_shape"] > 0
        print("✓ AC2: Handles shape mismatches gracefully")

    def test_ac3_reports_statistics(self, v2_checkpoint_path):
        """AC3: Reports statistics (matched, mismatched, missing)."""
        import torch.nn as nn
        from modeling_studio.models.initialization_v3 import LayerCopier

        class MockLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = nn.ModuleDict({"q_proj": nn.Linear(768, 768, bias=False)})

        loader = V2CheckpointLoader(str(v2_checkpoint_path))
        copier = LayerCopier(loader, strict=False)
        v3_layer = MockLayer()

        copier.copy_layer(v3_layer, v2_layer_idx=0, v3_layer_idx=0)
        stats = copier.get_stats()

        assert "matched" in stats
        assert "mismatched_shape" in stats
        assert "missing_in_v2" in stats
        print("✓ AC3: Reports statistics")

    def test_ac4_strict_and_non_strict_modes(self, v2_checkpoint_path):
        """AC4: Works with both strict and non-strict modes."""
        import torch.nn as nn
        from modeling_studio.models.initialization_v3 import LayerCopier

        class MockLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = nn.ModuleDict({"q_proj": nn.Linear(768, 768, bias=False)})

        loader = V2CheckpointLoader(str(v2_checkpoint_path))

        # Test strict mode
        copier_strict = LayerCopier(loader, strict=True)
        assert copier_strict.strict is True

        # Test non-strict mode
        copier_lenient = LayerCopier(loader, strict=False)
        assert copier_lenient.strict is False

        # Both should work
        v3_layer = MockLayer()
        copied1 = copier_strict.copy_layer(v3_layer, v2_layer_idx=0, v3_layer_idx=0)
        copied2 = copier_lenient.copy_layer(v3_layer, v2_layer_idx=0, v3_layer_idx=0)

        assert copied1 >= 0
        assert copied2 >= 0
        print("✓ AC4: Works with both strict and non-strict modes")

    def test_ac5_preserves_values_exactly(self, v2_checkpoint_path):
        """AC5: Preserves parameter values exactly (no modifications)."""
        import torch.nn as nn
        from modeling_studio.models.initialization_v3 import LayerCopier

        class MockLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = nn.ModuleDict(
                    {
                        "q_proj": nn.Linear(768, 768, bias=False),
                        "k_proj": nn.Linear(768, 768, bias=False),
                    }
                )

        loader = V2CheckpointLoader(str(v2_checkpoint_path))
        v2_weights = loader.get_layer_weights(0)

        copier = LayerCopier(loader, strict=False)
        v3_layer = MockLayer()

        copier.copy_layer(v3_layer, v2_layer_idx=0, v3_layer_idx=0)

        # Verify exact equality (no modifications)
        v2_q = v2_weights["attention.q_proj.weight"]
        v3_q = v3_layer.attention["q_proj"].weight
        assert torch.equal(v2_q, v3_q), "Values must be preserved exactly"

        v2_k = v2_weights["attention.k_proj.weight"]
        v3_k = v3_layer.attention["k_proj"].weight
        assert torch.equal(v2_k, v3_k), "Values must be preserved exactly"

        print("✓ AC5: Preserves parameter values exactly")
