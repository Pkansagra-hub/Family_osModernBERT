"""
Tests for ModernBERT v3 Configuration.

Tests the configuration dataclass for correct defaults, validation,
and layer band mappings according to enhanced_design_v3.md specifications.
"""

import pytest
from pathlib import Path
from omegaconf import OmegaConf

from modeling_studio.models.config_v3 import (
    ModernBERTv3Config,
    LayerSource,
    LayerMapping,
    get_layer_source_mapping,
    print_layer_source_mapping,
)


def test_config_defaults():
    """Test that configuration has correct default values."""
    config = ModernBERTv3Config()

    # Architecture
    assert config.hidden_size == 768
    assert config.num_layers == 28
    assert config.num_attention_heads == 12
    assert config.intermediate_size == 3072
    assert config.max_position_embeddings == 8192
    assert config.vocab_size == 50432  # 256-aligned

    # Hub Tokens
    assert config.hub_tokens == ["[EMO]", "[MEM]", "[REL]", "[TASK]"]
    assert config.hub_token_positions == {
        "[CLS]": 0,
        "[EMO]": 1,
        "[MEM]": 2,
        "[REL]": 3,
        "[TASK]": 4,
    }
    assert config.global_attention_positions == [0, 1, 2, 3, 4]

    # Window Sizes
    assert config.window_sizes == {
        "foundation": 64,
        "context": 128,
        "semantic": 256,
        "family": 512,
    }

    # Layer Bands
    assert config.layer_bands["foundation"] == list(range(1, 7))  # 1-6
    assert config.layer_bands["context"] == list(range(7, 19))  # 7-18
    assert config.layer_bands["semantic"] == list(range(19, 23))  # 19-22
    assert config.layer_bands["family"] == list(range(23, 29))  # 23-28

    # LoRA
    assert config.lora_enabled is True
    assert config.lora_r == 16
    assert config.lora_alpha == 16
    assert config.lora_dropout == 0.05
    assert config.lora_target_layers == [23, 24, 25, 26, 27, 28]

    # Pair Encoder
    assert config.pair_encoder_enabled is True
    assert config.pair_encoder_heads == 8
    assert config.pair_encoder_dropout == 0.1

    # Training
    assert config.frozen_layers_phase1 == list(range(1, 19))  # L1-18 frozen

    # FFN
    assert config.ffn_activation == "gelu"
    assert config.hidden_dropout_prob == 0.1
    assert config.attention_probs_dropout_prob == 0.1


def test_layer_bands_sum_to_28():
    """Test that layer bands correctly sum to 28 total layers."""
    config = ModernBERTv3Config()

    total_layers = sum(len(band) for band in config.layer_bands.values())
    assert total_layers == 28

    # Verify specific counts
    assert len(config.layer_bands["foundation"]) == 6  # L1-6
    assert len(config.layer_bands["context"]) == 12  # L7-18
    assert len(config.layer_bands["semantic"]) == 4  # L19-22
    assert len(config.layer_bands["family"]) == 6  # L23-28


def test_hub_token_positions():
    """Test that hub token positions are 0-indexed correctly."""
    config = ModernBERTv3Config()

    # CLS is at position 0
    assert config.hub_token_positions["[CLS]"] == 0

    # Hub tokens are at positions 1-4
    assert config.hub_token_positions["[EMO]"] == 1
    assert config.hub_token_positions["[MEM]"] == 2
    assert config.hub_token_positions["[REL]"] == 3
    assert config.hub_token_positions["[TASK]"] == 4

    # Global attention includes all positions 0-4
    assert config.global_attention_positions == [0, 1, 2, 3, 4]


def test_get_layer_band():
    """Test that get_layer_band returns correct band for each layer."""
    config = ModernBERTv3Config()

    # Foundation (L1-6)
    for layer in range(1, 7):
        assert config.get_layer_band(layer) == "foundation"

    # Context (L7-18)
    for layer in range(7, 19):
        assert config.get_layer_band(layer) == "context"

    # Semantic (L19-22)
    for layer in range(19, 23):
        assert config.get_layer_band(layer) == "semantic"

    # Family (L23-28)
    for layer in range(23, 29):
        assert config.get_layer_band(layer) == "family"

    # Invalid layer should raise error
    with pytest.raises(ValueError, match="not found in any band"):
        config.get_layer_band(0)

    with pytest.raises(ValueError, match="not found in any band"):
        config.get_layer_band(29)


def test_get_window_size():
    """Test that get_window_size returns correct window for each layer."""
    config = ModernBERTv3Config()

    # Foundation layers: window 64
    for layer in range(1, 7):
        assert config.get_window_size(layer) == 64

    # Context layers: window 128
    for layer in range(7, 19):
        assert config.get_window_size(layer) == 128

    # Semantic layers: window 256
    for layer in range(19, 23):
        assert config.get_window_size(layer) == 256

    # Family layers: window 512
    for layer in range(23, 29):
        assert config.get_window_size(layer) == 512


def test_get_trainable_layers():
    """Test that get_trainable_layers returns correct layers for each phase."""
    config = ModernBERTv3Config()

    # Phase 0 (healing): Only L23-28 trainable
    phase0_trainable = config.get_trainable_layers("phase0")
    assert phase0_trainable == list(range(23, 29))
    assert len(phase0_trainable) == 6

    # Phase 1: L19-28 trainable (semantic + family)
    phase1_trainable = config.get_trainable_layers("phase1")
    assert phase1_trainable == list(range(19, 23)) + list(range(23, 29))
    assert len(phase1_trainable) == 10

    # Phase 2: All layers trainable
    phase2_trainable = config.get_trainable_layers("phase2")
    assert phase2_trainable == list(range(1, 29))
    assert len(phase2_trainable) == 28

    # Invalid phase
    with pytest.raises(ValueError, match="Unknown training phase"):
        config.get_trainable_layers("phase99")


def test_get_lora_layers():
    """Test that get_lora_layers returns correct LoRA-enabled layers."""
    config = ModernBERTv3Config()

    # LoRA enabled: should return L23-28
    assert config.get_lora_layers() == [23, 24, 25, 26, 27, 28]

    # Disable LoRA
    config.lora_enabled = False
    assert config.get_lora_layers() == []


def test_lora_target_layers_match_family_band():
    """Test that LoRA target layers match family band layers."""
    config = ModernBERTv3Config()

    assert set(config.lora_target_layers) == set(config.layer_bands["family"])


def test_frozen_layers_match_foundation_context():
    """Test that frozen layers match foundation + context bands."""
    config = ModernBERTv3Config()

    expected_frozen = config.layer_bands["foundation"] + config.layer_bands["context"]
    assert set(config.frozen_layers_phase1) == set(expected_frozen)
    assert len(config.frozen_layers_phase1) == 18  # L1-18


def test_to_dict():
    """Test that to_dict returns complete configuration dictionary."""
    config = ModernBERTv3Config()
    config_dict = config.to_dict()

    # Verify all keys are present
    assert "hidden_size" in config_dict
    assert "num_layers" in config_dict
    assert "hub_tokens" in config_dict
    assert "hub_token_positions" in config_dict
    assert "window_sizes" in config_dict
    assert "layer_bands" in config_dict
    assert "lora_enabled" in config_dict
    assert "pair_encoder_enabled" in config_dict

    # Verify values match
    assert config_dict["hidden_size"] == 768
    assert config_dict["num_layers"] == 28
    assert config_dict["vocab_size"] == 50432  # 256-aligned


def test_validation_layer_count_mismatch():
    """Test that validation catches layer count mismatches."""
    # This should fail because layer bands don't sum to num_layers
    with pytest.raises(ValueError, match="Layer bands sum to"):
        ModernBERTv3Config(
            num_layers=30,  # Wrong count
            layer_bands={
                "foundation": list(range(1, 7)),
                "context": list(range(7, 19)),
                "semantic": list(range(19, 23)),
                "family": list(range(23, 29)),
            },
        )


def test_validation_hub_token_positions():
    """Test that validation catches incorrect hub token positions."""
    with pytest.raises(ValueError, match="Hub token positions mismatch"):
        ModernBERTv3Config(
            hub_token_positions={
                "[CLS]": 0,
                "[EMO]": 2,
                "[MEM]": 1,
                "[REL]": 3,
                "[TASK]": 4,  # Wrong order
            }
        )


def test_validation_global_attention_positions():
    """Test that validation catches incorrect global attention positions."""
    with pytest.raises(ValueError, match="Global attention positions must be"):
        ModernBERTv3Config(global_attention_positions=[0, 1, 2, 3])  # Missing position 4


def test_validation_window_sizes():
    """Test that validation catches missing window sizes."""
    with pytest.raises(ValueError, match="Window size not defined"):
        ModernBERTv3Config(
            window_sizes={
                "foundation": 64,
                "context": 128,
                # Missing "semantic" and "family"
            }
        )


def test_validation_lora_target_layers():
    """Test that validation catches LoRA layers not matching family band."""
    with pytest.raises(ValueError, match="LoRA target layers .* must match family band"):
        ModernBERTv3Config(lora_target_layers=[20, 21, 22, 23, 24, 25])  # Wrong layers


def test_validation_frozen_layers():
    """Test that validation catches frozen layers not matching foundation + context."""
    with pytest.raises(ValueError, match="Frozen layers .* must match foundation \\+ context"):
        ModernBERTv3Config(frozen_layers_phase1=list(range(1, 10)))  # Wrong range


# Tests for Issue 1.1.2: YAML Configuration


def test_yaml_loading():
    """Test that YAML configuration loads correctly via OmegaConf."""
    from omegaconf import OmegaConf
    from pathlib import Path

    # Load YAML config
    yaml_path = Path("configs/model/encoder/modernbert_v3_ultra.yaml")
    if not yaml_path.exists():
        pytest.skip(f"YAML config not found at {yaml_path}")

    cfg = OmegaConf.load(yaml_path)

    # Verify basic structure
    assert cfg.name == "ModernBERTv3Ultra"
    assert cfg.version == "3.3"
    assert cfg.codename == "Ultra"

    # Verify architecture
    assert cfg.architecture.hidden_size == 768
    assert cfg.architecture.num_layers == 28
    assert cfg.architecture.num_attention_heads == 12
    assert cfg.architecture.intermediate_size == 3072
    assert cfg.architecture.max_position_embeddings == 8192
    assert cfg.architecture.vocab_size == 50432  # 256-aligned

    # Verify hub tokens
    assert cfg.hub_tokens.enabled is True
    assert cfg.hub_tokens.tokens == ["[EMO]", "[MEM]", "[REL]", "[TASK]"]
    assert cfg.hub_tokens.positions["[CLS]"] == 0
    assert cfg.hub_tokens.positions["[EMO]"] == 1
    assert cfg.hub_tokens.positions["[MEM]"] == 2
    assert cfg.hub_tokens.positions["[REL]"] == 3
    assert cfg.hub_tokens.positions["[TASK]"] == 4
    assert cfg.hub_tokens.global_attention == [0, 1, 2, 3, 4]

    # Verify window sizes
    assert cfg.attention.window_sizes.foundation == 64
    assert cfg.attention.window_sizes.context == 128
    assert cfg.attention.window_sizes.semantic == 256
    assert cfg.attention.window_sizes.family == 512

    # Verify layer bands sum to 28
    total_layers = (
        len(cfg.layer_bands.foundation)
        + len(cfg.layer_bands.context)
        + len(cfg.layer_bands.semantic)
        + len(cfg.layer_bands.family)
    )
    assert total_layers == 28

    # Verify LoRA config
    assert cfg.lora.enabled is True
    assert cfg.lora.r == 16
    assert cfg.lora.alpha == 16
    assert cfg.lora.target_layers == [23, 24, 25, 26, 27, 28]


def test_yaml_matches_dataclass_defaults():
    """Test that YAML values match dataclass defaults."""
    from omegaconf import OmegaConf
    from pathlib import Path

    yaml_path = Path("configs/model/encoder/modernbert_v3_ultra.yaml")
    if not yaml_path.exists():
        pytest.skip(f"YAML config not found at {yaml_path}")

    cfg = OmegaConf.load(yaml_path)
    config = ModernBERTv3Config()

    # Compare architecture values
    assert cfg.architecture.hidden_size == config.hidden_size
    assert cfg.architecture.num_layers == config.num_layers
    assert cfg.architecture.num_attention_heads == config.num_attention_heads
    assert cfg.architecture.intermediate_size == config.intermediate_size
    assert cfg.architecture.vocab_size == config.vocab_size

    # Compare hub token configuration
    assert cfg.hub_tokens.tokens == config.hub_tokens

    # Compare LoRA configuration
    assert cfg.lora.r == config.lora_r
    assert cfg.lora.alpha == config.lora_alpha


# Tests for Issue 1.1.3: Layer Source Mapping


def test_layer_source_mapping():
    """Test that layer source mapping is correctly defined."""
    from modeling_studio.models.config_v3 import get_layer_source_mapping, LayerSource

    mapping = get_layer_source_mapping()

    # Verify all 28 layers are mapped
    assert len(mapping) == 28
    assert all(i in mapping for i in range(1, 29))

    # Verify layers 1-22 are COPY operations
    for i in range(1, 23):
        assert mapping[i].v3_layer == i
        assert mapping[i].source == LayerSource.COPY
        assert mapping[i].v2_layer == i

    # Verify layers 23-28 are CLONE operations from v2 layers 15-20
    expected_clones = {23: 15, 24: 16, 25: 17, 26: 18, 27: 19, 28: 20}
    for v3_layer, v2_layer in expected_clones.items():
        assert mapping[v3_layer].v3_layer == v3_layer
        assert mapping[v3_layer].source == LayerSource.CLONE
        assert mapping[v3_layer].v2_layer == v2_layer


def test_layer_source_mapping_copy_layers():
    """Test that layers 1-22 map to COPY from same v2 layer."""
    from modeling_studio.models.config_v3 import get_layer_source_mapping, LayerSource

    mapping = get_layer_source_mapping()

    for i in range(1, 23):
        layer_map = mapping[i]
        assert layer_map.source == LayerSource.COPY
        assert layer_map.v2_layer == i, f"Layer {i} should copy from v2 layer {i}"


def test_layer_source_mapping_clone_layers():
    """Test that layers 23-28 map to CLONE from v2 layers 15-20."""
    from modeling_studio.models.config_v3 import get_layer_source_mapping, LayerSource

    mapping = get_layer_source_mapping()

    # Layer 23 clones from v2 layer 15
    assert mapping[23].source == LayerSource.CLONE
    assert mapping[23].v2_layer == 15

    # Layer 24 clones from v2 layer 16
    assert mapping[24].source == LayerSource.CLONE
    assert mapping[24].v2_layer == 16

    # Layer 25 clones from v2 layer 17
    assert mapping[25].source == LayerSource.CLONE
    assert mapping[25].v2_layer == 17

    # Layer 26 clones from v2 layer 18
    assert mapping[26].source == LayerSource.CLONE
    assert mapping[26].v2_layer == 18

    # Layer 27 clones from v2 layer 19
    assert mapping[27].source == LayerSource.CLONE
    assert mapping[27].v2_layer == 19

    # Layer 28 clones from v2 layer 20
    assert mapping[28].source == LayerSource.CLONE
    assert mapping[28].v2_layer == 20


def test_layer_source_mapping_family_band():
    """Test that family band (L23-28) uses cloned layers."""
    from modeling_studio.models.config_v3 import get_layer_source_mapping, LayerSource

    config = ModernBERTv3Config()
    mapping = get_layer_source_mapping()

    # All family band layers should be cloned
    for layer in config.layer_bands["family"]:
        assert mapping[layer].source == LayerSource.CLONE


def test_layer_source_mapping_foundation_context_semantic():
    """Test that foundation/context/semantic bands use copied layers."""
    from modeling_studio.models.config_v3 import get_layer_source_mapping, LayerSource

    config = ModernBERTv3Config()
    mapping = get_layer_source_mapping()

    # Foundation band should be copied
    for layer in config.layer_bands["foundation"]:
        assert mapping[layer].source == LayerSource.COPY

    # Context band should be copied
    for layer in config.layer_bands["context"]:
        assert mapping[layer].source == LayerSource.COPY

    # Semantic band should be copied
    for layer in config.layer_bands["semantic"]:
        assert mapping[layer].source == LayerSource.COPY


def test_print_layer_source_mapping():
    """Test that print_layer_source_mapping runs without errors."""
    from modeling_studio.models.config_v3 import print_layer_source_mapping
    import io
    import sys

    # Capture stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    try:
        print_layer_source_mapping()
        output = captured_output.getvalue()

        # Verify output contains expected information
        assert "ModernBERT v3 Layer Source Mapping" in output
        assert "Foundation (L1-6)" in output
        assert "Context (L7-18)" in output
        assert "Semantic (L19-22)" in output
        assert "Family (L23-28)" in output
        assert "COPY" in output
        assert "CLONE" in output
        assert "function-preserving" in output
    finally:
        sys.stdout = sys.__stdout__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
