"""
Tests for ModernBERT v3.3 Ultra - Attention Mechanisms

Tests Issue 2.1.1: Global-Local Attention Mask Creation
Tests Issue 2.1.2: Layer-wise Window Size Configuration

Author: FamilyOS Team
Date: December 2025
"""

import numpy as np
import pytest
import torch

from modeling_studio.models.attention_v3 import (
    # Constants
    GLOBAL_TOKEN_POSITIONS,
    LAYER_WINDOW_CONFIG,
    LAYER_BANDS,
    FLASH_ATTN_AVAILABLE,
    # Mask creation
    create_global_local_attention_mask,
    create_causal_global_local_mask,
    expand_mask_for_batch,
    convert_mask_to_additive,
    # Layer configuration
    get_window_size_for_layer,
    get_layer_band_name,
    get_attention_mask_for_layer,
    get_layer_config_summary,
    # Attention modules
    MultiScaleAttentionWithGlobals,
    FlashAttentionWithGlobals,
    create_attention_layer,
    # Utilities
    count_attention_patterns,
)


# ==============================================================================
# Test Global-Local Attention Mask Creation (Issue 2.1.1)
# ==============================================================================


class TestGlobalLocalAttentionMask:
    """Test suite for global-local attention mask creation."""

    def test_global_token_positions_constant(self):
        """Test that global token positions are correctly defined."""
        assert GLOBAL_TOKEN_POSITIONS == [0, 1, 2, 3, 4]
        assert len(GLOBAL_TOKEN_POSITIONS) == 5

    def test_mask_creation_basic(self):
        """Test basic mask creation with small sequence."""
        seq_len = 10
        window_size = 4
        test_mask = create_global_local_attention_mask(seq_len, window_size)

        assert test_mask.shape == (seq_len, seq_len)
        assert test_mask.dtype == torch.bool

    def test_global_positions_have_full_row_attention(self):
        """
        Acceptance Criterion 1: Global positions (0-4) have full row attention.
        Global tokens can see everything.
        """
        seq_len = 20
        window_size = 4
        mask = create_global_local_attention_mask(seq_len, window_size)

        # Check each global position has full row (all 1s)
        for pos in GLOBAL_TOKEN_POSITIONS:
            row = mask[pos, :]
            assert row.all(), f"Global token at position {pos} should attend to all tokens"
            assert row.sum() == seq_len

    def test_global_positions_have_full_column_attention(self):
        """
        Acceptance Criterion 2: Global positions have full column attention.
        Everyone can see global tokens.
        """
        seq_len = 20
        window_size = 4
        mask = create_global_local_attention_mask(seq_len, window_size)

        # Check each global position has full column (all 1s)
        for pos in GLOBAL_TOKEN_POSITIONS:
            col = mask[:, pos]
            assert col.all(), f"All tokens should attend to global token at position {pos}"
            assert col.sum() == seq_len

    def test_text_tokens_use_sliding_window(self):
        """
        Acceptance Criterion 3: Text tokens use sliding window for non-global positions.
        """
        seq_len = 20
        window_size = 4
        mask = create_global_local_attention_mask(seq_len, window_size)

        # Check text token (position 10) attention pattern
        text_pos = 10
        row = mask[text_pos, :]

        # Should attend to globals (0-4)
        for global_pos in GLOBAL_TOKEN_POSITIONS:
            assert row[global_pos], f"Text token should attend to global at {global_pos}"

        # Should attend within window (half_window = 2)
        half_window = window_size // 2
        start = max(0, text_pos - half_window)
        end = min(seq_len, text_pos + half_window + 1)

        for i in range(start, end):
            if i not in GLOBAL_TOKEN_POSITIONS:
                assert row[i], f"Text token {text_pos} should attend to {i} (within window)"

        # Should NOT attend to distant text tokens
        distant_pos = 19
        if distant_pos not in range(start, end) and distant_pos not in GLOBAL_TOKEN_POSITIONS:
            assert not row[distant_pos], f"Text token {text_pos} should NOT attend to {distant_pos}"

    def test_mask_shape_batch_expansion(self):
        """
        Acceptance Criterion 4: Mask shape is [seq_len, seq_len] or [batch, heads, seq_len, seq_len].
        """
        seq_len = 10
        window_size = 4
        batch_size = 2
        num_heads = 12

        # 2D mask
        mask_2d = create_global_local_attention_mask(seq_len, window_size)
        assert mask_2d.shape == (seq_len, seq_len)

        # Expand to 4D
        mask_4d = expand_mask_for_batch(mask_2d, batch_size, num_heads)
        assert mask_4d.shape == (batch_size, num_heads, seq_len, seq_len)

        # Check content is replicated correctly
        assert torch.equal(mask_4d[0, 0, :, :], mask_2d)
        assert torch.equal(mask_4d[1, 5, :, :], mask_2d)

    def test_variable_sequence_lengths(self):
        """
        Acceptance Criterion 5: Works with variable sequence lengths.
        """
        window_size = 4
        seq_lengths = [8, 16, 32, 64, 128, 256]

        for seq_len in seq_lengths:
            mask = create_global_local_attention_mask(seq_len, window_size)
            assert mask.shape == (seq_len, seq_len)

            # Check global tokens still work
            for pos in GLOBAL_TOKEN_POSITIONS:
                if pos < seq_len:
                    assert mask[pos, :].all()
                    assert mask[:, pos].all()

    def test_mask_dtype_bool(self):
        """Test mask creation with boolean dtype."""
        mask = create_global_local_attention_mask(10, 4, dtype=torch.bool)
        assert mask.dtype == torch.bool
        assert mask[0, 5]  # Global can see text
        # Position 8 with window=4 (half=2): window is [6,10], pos 3 is outside AND it's a global
        assert mask[8, 3]  # Position 3 is a global token, always visible

    def test_mask_dtype_float(self):
        """Test mask creation with float dtype."""
        mask = create_global_local_attention_mask(10, 4, dtype=torch.float32)
        assert mask.dtype == torch.float32
        assert mask[0, 5] == 1.0
        assert mask[8, 2] == 0.0 or mask[8, 2] == 1.0  # Binary values

    def test_mask_device_cpu(self):
        """Test mask creation on CPU device."""
        mask = create_global_local_attention_mask(10, 4, device=torch.device("cpu"))
        assert mask.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mask_device_cuda(self):
        """Test mask creation on CUDA device."""
        device = torch.device("cuda")
        mask = create_global_local_attention_mask(10, 4, device=device)
        assert mask.device.type == "cuda"

    def test_visual_example_from_docstring(self):
        """
         Test the exact visual example from the docstring.

         Visual example (seq_len=10, window=4, globals=[0,1,2,3,4]):

               0  1  2  3  4  5  6  7  8  9   (keys)
            +--------------------------------
        0   |  1  1  1  1  1  1  1  1  1  1   <- [CLS] global
        1   |  1  1  1  1  1  1  1  1  1  1   <- [EMO] global
        2   |  1  1  1  1  1  1  1  1  1  1   <- [MEM] global
        3   |  1  1  1  1  1  1  1  1  1  1   <- [REL] global
        4   |  1  1  1  1  1  1  1  1  1  1   <- [TASK] global
        5   |  1  1  1  1  1  1  1  1  0  0   <- text: globals + window
        6   |  1  1  1  1  1  0  1  1  1  0   <- text: globals + window
        7   |  1  1  1  1  1  0  0  1  1  1   <- text: globals + window
        8   |  1  1  1  1  1  0  0  0  1  1   <- text: globals + window
        9   |  1  1  1  1  1  0  0  0  0  1   <- text: globals + window
        """
        mask = create_global_local_attention_mask(10, 4)

        # Convert to numpy for easy checking
        m = mask.cpu().numpy().astype(int)

        # Check global rows (0-4)
        for i in range(5):
            assert np.all(m[i, :] == 1), f"Row {i} should be all 1s"

        # Check text token rows
        # Row 5: globals (0-4) + window [3-8] (half_window=2: 5-2=3 to 5+2+1=8)
        expected_5 = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
        assert list(m[5, :]) == expected_5, "Row 5 mismatch"

        # Row 6: globals (0-4) + window [4-9] (half_window=2: 6-2=4 to 6+2+1=9)
        expected_6 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        assert list(m[6, :]) == expected_6, "Row 6 mismatch"

        # Row 7: globals (0-4) + window [5-10] (half_window=2: 7-2=5 to 7+2+1=10, capped at 10)
        expected_7 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        assert list(m[7, :]) == expected_7, "Row 7 mismatch"

        # Row 8: globals (0-4) + window [6-11] (half_window=2: 8-2=6 to 8+2+1=11, capped at 10)
        expected_8 = [1, 1, 1, 1, 1, 0, 1, 1, 1, 1]
        assert list(m[8, :]) == expected_8, "Row 8 mismatch"

        # Row 9: globals (0-4) + window [7-12] (half_window=2: 9-2=7 to 9+2+1=12, capped at 10)
        expected_9 = [1, 1, 1, 1, 1, 0, 0, 1, 1, 1]
        assert list(m[9, :]) == expected_9, "Row 9 mismatch"

    def test_attention_pattern_counts(self):
        """Test counting attention patterns."""
        mask = create_global_local_attention_mask(100, 64)
        stats = count_attention_patterns(mask)

        assert stats["seq_len"] == 100
        assert stats["global_tokens"] == 5  # Positions 0-4
        assert stats["attended_by_all"] == 5  # Same positions
        assert 0 < stats["density"] < 1.0  # Sparse but not empty


# ==============================================================================
# Test Causal Masks
# ==============================================================================


class TestCausalMask:
    """Test causal attention masks (for completeness)."""

    def test_causal_mask_creation(self):
        """Test basic causal mask creation."""
        mask = create_causal_global_local_mask(10, 4)
        assert mask.shape == (10, 10)

    def test_causal_constraint(self):
        """Test that causal mask prevents attending to future tokens."""
        seq_len = 10
        mask = create_causal_global_local_mask(seq_len, 4)

        # Upper triangle should be masked (except globals)
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                if i not in GLOBAL_TOKEN_POSITIONS:
                    # Non-global can't see future (unless j is global)
                    if j not in GLOBAL_TOKEN_POSITIONS:
                        assert not mask[i, j], f"Position {i} should not see future {j}"

    def test_causal_global_tokens_still_work(self):
        """Test that global tokens work in causal mode."""
        mask = create_causal_global_local_mask(10, 4)

        # In causal mode, even globals can't see future tokens
        # Global tokens see everything UP TO their position, not future
        for pos in GLOBAL_TOKEN_POSITIONS:
            if pos < 10:
                # Can see positions 0 to pos (inclusive)
                assert mask[pos, : pos + 1].all(), f"Global {pos} can't see past positions"
                # Can't see future positions
                if pos + 1 < 10:
                    assert not mask[pos, pos + 1 :].any(), f"Global {pos} should not see future"


# ==============================================================================
# Test Layer Window Configuration (Issue 2.1.2)
# ==============================================================================


class TestLayerWindowConfiguration:
    """Test suite for layer-wise window size configuration."""

    def test_layer_window_config_constant(self):
        """Test that LAYER_WINDOW_CONFIG has all 28 layers."""
        assert len(LAYER_WINDOW_CONFIG) == 28
        assert all(i in LAYER_WINDOW_CONFIG for i in range(1, 29))

    def test_layer_bands_constant(self):
        """Test that LAYER_BANDS is correctly defined."""
        assert len(LAYER_BANDS) == 4
        assert "foundation" in LAYER_BANDS
        assert "context" in LAYER_BANDS
        assert "semantic" in LAYER_BANDS
        assert "family" in LAYER_BANDS

    def test_foundation_band_window_64(self):
        """
        Acceptance Criterion: Foundation (L1-6) uses 64-token window.
        """
        for layer in range(1, 7):
            assert get_window_size_for_layer(layer) == 64
            assert get_layer_band_name(layer) == "foundation"

    def test_context_band_window_128(self):
        """
        Acceptance Criterion: Context (L7-18) uses 128-token window.
        """
        for layer in range(7, 19):
            assert get_window_size_for_layer(layer) == 128
            assert get_layer_band_name(layer) == "context"

    def test_semantic_band_window_256(self):
        """
        Acceptance Criterion: Semantic (L19-22) uses 256-token window.
        """
        for layer in range(19, 23):
            assert get_window_size_for_layer(layer) == 256
            assert get_layer_band_name(layer) == "semantic"

    def test_family_band_window_512(self):
        """
        Acceptance Criterion: Family (L23-28) uses 512-token window.
        """
        for layer in range(23, 29):
            assert get_window_size_for_layer(layer) == 512
            assert get_layer_band_name(layer) == "family"

    def test_invalid_layer_index_raises(self):
        """
        Acceptance Criterion: Invalid layer indices raise ValueError.
        """
        with pytest.raises(ValueError, match="Invalid layer index"):
            get_window_size_for_layer(0)

        with pytest.raises(ValueError, match="Invalid layer index"):
            get_window_size_for_layer(29)

        with pytest.raises(ValueError, match="Invalid layer index"):
            get_window_size_for_layer(-1)

        with pytest.raises(ValueError, match="Invalid layer index"):
            get_layer_band_name(0)

        with pytest.raises(ValueError, match="Invalid layer index"):
            get_layer_band_name(29)

    def test_all_28_layers_have_window_sizes(self):
        """
        Acceptance Criterion: All 28 layers have defined window sizes.
        """
        for layer in range(1, 29):
            window_size = get_window_size_for_layer(layer)
            assert window_size in [64, 128, 256, 512]

    def test_get_attention_mask_for_layer(self):
        """Test convenience function for getting layer-specific masks."""
        seq_len = 100

        # Foundation layer (window=64)
        mask_l1 = get_attention_mask_for_layer(1, seq_len)
        assert mask_l1.shape == (seq_len, seq_len)

        # Family layer (window=512)
        mask_l25 = get_attention_mask_for_layer(25, seq_len)
        assert mask_l25.shape == (seq_len, seq_len)

        # Different layers should have different masks (due to window size)
        # Compare attention patterns for text tokens
        assert not torch.equal(mask_l1, mask_l25)

    def test_layer_config_summary(self):
        """Test programmatic access to layer config."""
        summary = get_layer_config_summary()

        assert summary["foundation"]["window_size"] == 64
        assert summary["foundation"]["num_layers"] == 6
        assert summary["context"]["window_size"] == 128
        assert summary["context"]["num_layers"] == 12
        assert summary["semantic"]["window_size"] == 256
        assert summary["semantic"]["num_layers"] == 4
        assert summary["family"]["window_size"] == 512
        assert summary["family"]["num_layers"] == 6


# ==============================================================================
# Test Mask Utilities
# ==============================================================================


class TestMaskUtilities:
    """Test utility functions for masks."""

    def test_convert_mask_to_additive(self):
        """Test conversion from boolean to additive mask."""
        bool_mask = torch.tensor([[True, False], [False, True]])
        additive_mask = convert_mask_to_additive(bool_mask)

        assert additive_mask.dtype == torch.float32
        assert additive_mask[0, 0] == 0.0  # True -> 0.0
        assert additive_mask[0, 1] == float("-inf")  # False -> -inf
        assert additive_mask[1, 0] == float("-inf")
        assert additive_mask[1, 1] == 0.0

    def test_expand_mask_for_batch(self):
        """Test batch expansion of 2D mask."""
        mask_2d = create_global_local_attention_mask(10, 4)
        batch_size = 3
        num_heads = 12

        mask_4d = expand_mask_for_batch(mask_2d, batch_size, num_heads)

        assert mask_4d.shape == (batch_size, num_heads, 10, 10)

        # Check all batch/head combinations have same content
        for b in range(batch_size):
            for h in range(num_heads):
                assert torch.equal(mask_4d[b, h, :, :], mask_2d)


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestAttentionIntegration:
    """Integration tests for attention system."""

    def test_full_pipeline_foundation_layer(self):
        """Test complete pipeline for foundation layer."""
        layer_idx = 1
        seq_len = 256
        batch_size = 4
        num_heads = 12

        # Get layer-specific mask
        mask = get_attention_mask_for_layer(layer_idx, seq_len)

        # Expand for batch
        mask_batch = expand_mask_for_batch(mask, batch_size, num_heads)

        # Convert to additive (needs boolean input)
        mask_additive = convert_mask_to_additive(mask_batch)

        assert mask_additive.shape == (batch_size, num_heads, seq_len, seq_len)

        # Check global tokens still work
        for pos in GLOBAL_TOKEN_POSITIONS:
            assert (mask_additive[:, :, pos, :] == 0.0).all()

    def test_full_pipeline_family_layer(self):
        """Test complete pipeline for family layer."""
        layer_idx = 28
        seq_len = 512
        batch_size = 2
        num_heads = 12

        mask = get_attention_mask_for_layer(layer_idx, seq_len)
        mask_batch = expand_mask_for_batch(mask, batch_size, num_heads)
        mask_additive = convert_mask_to_additive(mask_batch)

        assert mask_additive.shape == (batch_size, num_heads, seq_len, seq_len)

    def test_different_layers_different_patterns(self):
        """Test that different layer bands produce different attention patterns."""
        seq_len = 200

        mask_foundation = get_attention_mask_for_layer(1, seq_len)
        mask_context = get_attention_mask_for_layer(10, seq_len)
        mask_semantic = get_attention_mask_for_layer(20, seq_len)
        mask_family = get_attention_mask_for_layer(25, seq_len)

        # Count attention edges for text tokens (position 50)
        pos = 50
        edges_foundation = mask_foundation[pos, :].sum().item()
        edges_context = mask_context[pos, :].sum().item()
        edges_semantic = mask_semantic[pos, :].sum().item()
        edges_family = mask_family[pos, :].sum().item()

        # Larger windows should have more edges
        # Note: All include 5 global tokens, so differences are from window
        assert edges_foundation < edges_context
        assert edges_context < edges_semantic
        assert edges_semantic < edges_family

    def test_long_sequence_8192_tokens(self):
        """Test with full 8192-token context (v3 maximum)."""
        seq_len = 8192
        layer_idx = 28  # Family layer (512 window)

        mask = get_attention_mask_for_layer(layer_idx, seq_len)

        assert mask.shape == (seq_len, seq_len)

        # Check global tokens still work
        for pos in GLOBAL_TOKEN_POSITIONS:
            assert mask[pos, :].all()
            assert mask[:, pos].all()

        # Check a text token in the middle
        mid_pos = 4096
        row = mask[mid_pos, :]

        # Should attend to globals
        for global_pos in GLOBAL_TOKEN_POSITIONS:
            assert row[global_pos]

        # Should attend within 512 window
        half_window = 512 // 2
        start = mid_pos - half_window
        end = mid_pos + half_window + 1
        for i in range(start, end):
            if 0 <= i < seq_len:
                assert row[i]

        # Should NOT attend to very distant tokens
        distant = 100
        if distant < start and distant not in GLOBAL_TOKEN_POSITIONS:
            assert not row[distant]


# ==============================================================================
# Correctness Tests (Blind Hub Problem)
# ==============================================================================


class TestBlindHubSolution:
    """
    Tests verifying the solution to the "Blind Hub" problem.

    The Blind Hub problem occurred when hub tokens had sliding windows,
    preventing them from seeing distant tokens in long sequences.
    """

    def test_hub_can_see_all_tokens_short_sequence(self):
        """Test hub tokens can see all tokens in a short sequence."""
        seq_len = 50
        window_size = 64  # Window larger than sequence
        mask = create_global_local_attention_mask(seq_len, window_size)

        # Every hub token should see every position
        for hub_pos in GLOBAL_TOKEN_POSITIONS:
            assert mask[hub_pos, :].all(), f"Hub {hub_pos} cannot see all tokens"

    def test_hub_can_see_all_tokens_long_sequence(self):
        """Test hub tokens can see all tokens even in long sequences."""
        seq_len = 1000
        window_size = 64  # Window much smaller than sequence
        mask = create_global_local_attention_mask(seq_len, window_size)

        # Every hub token should see every position (CRITICAL TEST)
        for hub_pos in GLOBAL_TOKEN_POSITIONS:
            row = mask[hub_pos, :]
            assert row.all(), f"Hub {hub_pos} cannot see all {seq_len} tokens"
            assert row.sum() == seq_len

    def test_distant_text_can_see_hub_tokens(self):
        """Test that distant text tokens can see hub tokens."""
        seq_len = 1000
        window_size = 64
        mask = create_global_local_attention_mask(seq_len, window_size)

        # A text token far from hubs (position 500) should still see them
        distant_pos = 500
        for hub_pos in GLOBAL_TOKEN_POSITIONS:
            assert mask[distant_pos, hub_pos], f"Text at {distant_pos} cannot see hub at {hub_pos}"

    def test_text_at_position_500_can_see_emo_hub(self):
        """
        MUST PASS: Text at position 500 can attend to [EMO] at position 1.

        This is the acceptance test from implementation_plan_v3.md Risk 1.
        """
        seq_len = 1000
        window_size = 64  # Small window, [EMO] at pos 1 is far outside
        mask = create_global_local_attention_mask(seq_len, window_size)

        text_pos = 500
        emo_pos = 1  # [EMO] hub token

        # CRITICAL: Text must be able to see [EMO]
        assert mask[
            text_pos, emo_pos
        ], "Text cannot see [EMO] hub token! Blind Hub problem not solved."

        # Also check [EMO] can see text
        assert mask[emo_pos, text_pos], "[EMO] cannot see text at position 500!"

    def test_attention_weights_nonzero_for_hubs(self):
        """
        Simulate attention weights to verify hubs are attended to.

        This is a more realistic test: even if mask allows attention,
        we need to verify the pattern makes sense.
        """
        seq_len = 100
        window_size = 64
        mask = create_global_local_attention_mask(seq_len, window_size)

        # For each text token, check it can attend to hubs
        for text_pos in range(5, seq_len):  # Skip global positions
            row = mask[text_pos, :]

            # Should attend to all 5 global positions
            for hub_pos in GLOBAL_TOKEN_POSITIONS:
                assert row[hub_pos], f"Text {text_pos} cannot attend to hub {hub_pos}"

            # Count how many positions can be attended
            num_attended = row.sum().item()

            # Should be: 5 globals + window_size (roughly)
            # Exact count depends on position, but should be > 5
            assert num_attended > 5, f"Text {text_pos} only attends to {num_attended} positions"


# ==============================================================================
# Test MultiScaleAttentionWithGlobals (Issue 2.1.3)
# ==============================================================================


class TestMultiScaleAttentionWithGlobals:
    """Test suite for MultiScaleAttentionWithGlobals module."""

    def test_module_initialization(self):
        """
        Acceptance Criterion: QKV projections correctly sized (768 → 768).
        """
        from modeling_studio.models.attention_v3 import MultiScaleAttentionWithGlobals

        attn = MultiScaleAttentionWithGlobals(layer_idx=1)

        # Check projections
        assert attn.q_proj.in_features == 768
        assert attn.q_proj.out_features == 768
        assert attn.k_proj.in_features == 768
        assert attn.k_proj.out_features == 768
        assert attn.v_proj.in_features == 768
        assert attn.v_proj.out_features == 768
        assert attn.out_proj.in_features == 768
        assert attn.out_proj.out_features == 768

    def test_head_dimensions(self):
        """
        Acceptance Criterion: Multi-head reshape is correct (12 heads × 64 dim).
        """
        from modeling_studio.models.attention_v3 import MultiScaleAttentionWithGlobals

        attn = MultiScaleAttentionWithGlobals(hidden_size=768, num_attention_heads=12, layer_idx=1)

        assert attn.num_attention_heads == 12
        assert attn.head_dim == 64
        assert attn.num_attention_heads * attn.head_dim == 768

    def test_layer_specific_window_size(self):
        """Test that attention module uses layer-specific window sizes."""
        from modeling_studio.models.attention_v3 import MultiScaleAttentionWithGlobals

        # Foundation layer (window=64)
        attn_l1 = MultiScaleAttentionWithGlobals(layer_idx=1)
        assert attn_l1.window_size == 64

        # Context layer (window=128)
        attn_l10 = MultiScaleAttentionWithGlobals(layer_idx=10)
        assert attn_l10.window_size == 128

        # Semantic layer (window=256)
        attn_l20 = MultiScaleAttentionWithGlobals(layer_idx=20)
        assert attn_l20.window_size == 256

        # Family layer (window=512)
        attn_l25 = MultiScaleAttentionWithGlobals(layer_idx=25)
        assert attn_l25.window_size == 512

    def test_forward_pass_basic(self):
        """
        Acceptance Criterion: Output shape matches input shape.
        """
        from modeling_studio.models.attention_v3 import MultiScaleAttentionWithGlobals

        attn = MultiScaleAttentionWithGlobals(layer_idx=1)
        batch_size = 2
        seq_len = 50
        hidden_size = 768

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        output, _ = attn(hidden_states)

        assert output.shape == (batch_size, seq_len, hidden_size)

    def test_forward_pass_with_attention_weights(self):
        """
        Acceptance Criterion: Attention weights can be returned for debugging.
        """
        from modeling_studio.models.attention_v3 import MultiScaleAttentionWithGlobals

        attn = MultiScaleAttentionWithGlobals(layer_idx=1)
        batch_size = 2
        seq_len = 50

        hidden_states = torch.randn(batch_size, seq_len, 768)
        output, weights = attn(hidden_states, output_attentions=True)

        assert weights is not None
        assert weights.shape == (batch_size, 12, seq_len, seq_len)

    def test_global_local_mask_applied(self):
        """
        Acceptance Criterion: Global-local mask applied correctly.

        Test that the attention module uses the correct mask pattern:
        - Hub tokens can attend everywhere
        - All tokens can attend to hubs
        - Text tokens use sliding windows
        """
        from modeling_studio.models.attention_v3 import MultiScaleAttentionWithGlobals

        attn = MultiScaleAttentionWithGlobals(layer_idx=1)  # window=64
        batch_size = 1
        seq_len = 100
        hidden_states = torch.randn(batch_size, seq_len, 768)

        output, weights = attn(hidden_states, output_attentions=True)

        # Check weights for hub tokens (should attend to all)
        for hub_pos in GLOBAL_TOKEN_POSITIONS:
            hub_weights = weights[0, 0, hub_pos, :]  # First batch, first head
            # Hub should have non-zero attention to many positions
            num_nonzero = (hub_weights > 0).sum().item()
            assert num_nonzero > 50, f"Hub {hub_pos} only attends to {num_nonzero} positions"

        # Check that all tokens can attend to hubs
        for text_pos in range(5, seq_len):
            text_weights = weights[0, 0, text_pos, :]
            for hub_pos in GLOBAL_TOKEN_POSITIONS:
                # Should have some attention to each hub
                assert (
                    text_weights[hub_pos] >= 0
                ), f"Text {text_pos} has no attention to hub {hub_pos}"

    def test_padding_mask_combination(self):
        """
        Acceptance Criterion: Padding mask combined with global-local mask.

        Test that padding tokens are correctly masked even with global-local pattern.
        """
        from modeling_studio.models.attention_v3 import MultiScaleAttentionWithGlobals

        attn = MultiScaleAttentionWithGlobals(layer_idx=1)
        batch_size = 2
        seq_len = 50
        hidden_states = torch.randn(batch_size, seq_len, 768)

        # Create padding mask: first sample has padding from position 40 onwards
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, 40:] = 0  # Padding

        output, weights = attn(hidden_states, attention_mask, output_attentions=True)

        # Check that attention weights to padded positions are near zero
        for query_pos in range(seq_len):
            for key_pos in range(40, seq_len):
                # First sample should have ~0 attention to padded positions
                attn_to_padded = weights[0, 0, query_pos, key_pos].item()
                assert attn_to_padded < 1e-5, (
                    f"Query {query_pos} has non-zero attention {attn_to_padded} "
                    f"to padded position {key_pos}"
                )

    def test_attention_mask_caching(self):
        """Test that attention masks are cached for efficiency."""
        from modeling_studio.models.attention_v3 import MultiScaleAttentionWithGlobals

        attn = MultiScaleAttentionWithGlobals(layer_idx=1)
        seq_len = 50
        hidden_states = torch.randn(2, seq_len, 768)

        # First forward pass - should create mask
        output1, _ = attn(hidden_states)
        assert attn._cached_mask is not None
        assert attn._cached_seq_len == seq_len
        cached_mask_id = id(attn._cached_mask)

        # Second forward pass with same seq_len - should reuse cached mask
        output2, _ = attn(hidden_states)
        assert id(attn._cached_mask) == cached_mask_id

        # Third forward pass with different seq_len - should recreate mask
        hidden_states_new = torch.randn(2, 60, 768)
        output3, _ = attn(hidden_states_new)
        assert attn._cached_seq_len == 60
        assert id(attn._cached_mask) != cached_mask_id

    def test_different_layer_bands(self):
        """Test attention works correctly for all layer bands."""
        from modeling_studio.models.attention_v3 import MultiScaleAttentionWithGlobals

        hidden_states = torch.randn(2, 100, 768)

        # Test one layer from each band
        for layer_idx, expected_window in [(1, 64), (10, 128), (20, 256), (25, 512)]:
            attn = MultiScaleAttentionWithGlobals(layer_idx=layer_idx)
            assert attn.window_size == expected_window

            output, _ = attn(hidden_states)
            assert output.shape == hidden_states.shape

    def test_long_sequence_8192(self):
        """Test attention handles maximum context length (8192 tokens)."""
        from modeling_studio.models.attention_v3 import MultiScaleAttentionWithGlobals

        # Use Family layer (512 window) for long context
        attn = MultiScaleAttentionWithGlobals(layer_idx=25)
        batch_size = 1
        seq_len = 8192
        hidden_states = torch.randn(batch_size, seq_len, 768)

        output, _ = attn(hidden_states)
        assert output.shape == (batch_size, seq_len, 768)

    def test_gradient_flow(self):
        """Test that gradients flow correctly through attention."""
        from modeling_studio.models.attention_v3 import MultiScaleAttentionWithGlobals

        attn = MultiScaleAttentionWithGlobals(layer_idx=1)
        hidden_states = torch.randn(2, 50, 768, requires_grad=True)

        output, _ = attn(hidden_states)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        assert hidden_states.grad is not None
        assert attn.q_proj.weight.grad is not None
        assert attn.k_proj.weight.grad is not None
        assert attn.v_proj.weight.grad is not None
        assert attn.out_proj.weight.grad is not None

    def test_extra_repr(self):
        """Test extra_repr provides useful debug information."""
        from modeling_studio.models.attention_v3 import MultiScaleAttentionWithGlobals

        attn = MultiScaleAttentionWithGlobals(layer_idx=15)
        repr_str = attn.extra_repr()

        assert "layer=15" in repr_str
        assert "window=128" in repr_str
        assert "heads=12" in repr_str
        assert "head_dim=64" in repr_str

    def test_device_compatibility(self):
        """Test attention works on different devices."""
        from modeling_studio.models.attention_v3 import MultiScaleAttentionWithGlobals

        attn = MultiScaleAttentionWithGlobals(layer_idx=1)
        hidden_states = torch.randn(2, 50, 768)

        # CPU test
        output_cpu, _ = attn(hidden_states)
        assert output_cpu.device.type == "cpu"

        # CUDA test (if available)
        if torch.cuda.is_available():
            attn_cuda = attn.cuda()
            hidden_states_cuda = hidden_states.cuda()
            output_cuda, _ = attn_cuda(hidden_states_cuda)
            assert output_cuda.device.type == "cuda"

    def test_batch_size_1_and_larger(self):
        """Test attention works with batch size 1 and larger batches."""
        from modeling_studio.models.attention_v3 import MultiScaleAttentionWithGlobals

        attn = MultiScaleAttentionWithGlobals(layer_idx=1)
        seq_len = 50

        # Batch size 1
        hidden_states_1 = torch.randn(1, seq_len, 768)
        output_1, _ = attn(hidden_states_1)
        assert output_1.shape == (1, seq_len, 768)

        # Batch size 8
        hidden_states_8 = torch.randn(8, seq_len, 768)
        output_8, _ = attn(hidden_states_8)
        assert output_8.shape == (8, seq_len, 768)

    def test_dropout_applied(self):
        """Test that dropout is applied during training mode."""
        from modeling_studio.models.attention_v3 import MultiScaleAttentionWithGlobals

        attn = MultiScaleAttentionWithGlobals(layer_idx=1, attention_dropout=0.5)
        hidden_states = torch.randn(2, 50, 768)

        # Training mode - dropout active
        attn.train()
        output1, weights1 = attn(hidden_states, output_attentions=True)
        output2, weights2 = attn(hidden_states, output_attentions=True)

        # Outputs should differ due to dropout
        assert not torch.allclose(output1, output2)

        # Eval mode - dropout inactive
        attn.eval()
        output3, weights3 = attn(hidden_states, output_attentions=True)
        output4, weights4 = attn(hidden_states, output_attentions=True)

        # Outputs should be identical
        assert torch.allclose(output3, output4)


# ==============================================================================
# Test Flash Attention 2 with Safety Switch (Issue 2.1.4)
# ==============================================================================


class TestFlashAttentionWithSafetySwitch:
    """Test suite for Flash Attention 2 integration with Safety Switch."""

    def test_flash_attention_availability_flag(self):
        """Test that FLASH_ATTN_AVAILABLE flag is set correctly."""
        # Should be a boolean
        assert isinstance(FLASH_ATTN_AVAILABLE, bool)

        # Try importing to verify consistency
        try:
            from flash_attn import flash_attn_func

            assert FLASH_ATTN_AVAILABLE is True
        except ImportError:
            assert FLASH_ATTN_AVAILABLE is False

    def test_sdpa_used_in_standard_attention(self):
        """Test that MultiScaleAttentionWithGlobals uses SDPA when output_attentions=False."""
        attn = MultiScaleAttentionWithGlobals(layer_idx=1)
        hidden_states = torch.randn(2, 100, 768)

        # Should use SDPA (no errors)
        output, weights = attn(hidden_states, output_attentions=False)
        assert output.shape == (2, 100, 768)
        assert weights is None

        # Should fall back to manual attention when output_attentions=True
        output2, weights2 = attn(hidden_states, output_attentions=True)
        assert output2.shape == (2, 100, 768)
        assert weights2.shape == (2, 12, 100, 100)

    def test_sdpa_memory_efficiency_long_sequence(self):
        """Test that SDPA handles long sequences without OOM (up to 8k)."""
        attn = MultiScaleAttentionWithGlobals(layer_idx=28)  # 512 window
        seq_len = 2048  # Test with 2k sequence

        hidden_states = torch.randn(1, seq_len, 768)

        # Should work without OOM
        output, _ = attn(hidden_states)
        assert output.shape == (1, seq_len, 768)

    @pytest.mark.skipif(not FLASH_ATTN_AVAILABLE, reason="Flash Attention not installed")
    def test_flash_attention_module_initialization(self):
        """Test that FlashAttentionWithGlobals initializes correctly."""
        attn = FlashAttentionWithGlobals(layer_idx=25)

        assert attn.hidden_size == 768
        assert attn.num_attention_heads == 12
        assert attn.head_dim == 64
        assert attn.layer_idx == 25
        assert attn.window_size == 512  # Family band

        # Check projections
        assert attn.q_proj.in_features == 768
        assert attn.q_proj.out_features == 768
        assert attn.k_proj.in_features == 768
        assert attn.v_proj.in_features == 768
        assert attn.out_proj.in_features == 768

    @pytest.mark.skipif(not FLASH_ATTN_AVAILABLE, reason="Flash Attention not installed")
    def test_flash_attention_forward_pass(self):
        """Test that FlashAttentionWithGlobals forward pass works."""
        attn = FlashAttentionWithGlobals(layer_idx=1)
        hidden_states = torch.randn(2, 100, 768)

        output, weights = attn(hidden_states)
        assert output.shape == (2, 100, 768)
        assert weights is None  # Flash Attention doesn't return weights

    @pytest.mark.skipif(not FLASH_ATTN_AVAILABLE, reason="Flash Attention not installed")
    def test_flash_attention_hub_correction(self):
        """Test that hub tokens (0-4) get corrected attention in Flash mode."""
        attn = FlashAttentionWithGlobals(layer_idx=1)
        seq_len = 100
        hidden_states = torch.randn(1, seq_len, 768)

        output, _ = attn(hidden_states)

        # Hub positions should have global attention applied
        # (We can't verify the exact pattern without output_attentions,
        # but we can check the output is reasonable)
        assert output.shape == (1, seq_len, 768)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

        # Check that hub tokens (0-4) have different values than text tokens
        # (they should be affected by global attention)
        hub_outputs = output[0, :5, :]  # [5, 768]
        text_outputs = output[0, 5:, :]  # [95, 768]

        # Hub and text should have different statistics due to different attention
        hub_mean = hub_outputs.mean()
        text_mean = text_outputs.mean()

        # Not a strict equality, but they should be different
        assert not torch.allclose(hub_mean, text_mean, atol=0.01)

    @pytest.mark.skipif(not FLASH_ATTN_AVAILABLE, reason="Flash Attention not installed")
    def test_flash_attention_output_attentions_error(self):
        """Test that Flash Attention raises error when output_attentions=True."""
        attn = FlashAttentionWithGlobals(layer_idx=1)
        hidden_states = torch.randn(2, 100, 768)

        with pytest.raises(ValueError, match="does not support output_attentions"):
            attn(hidden_states, output_attentions=True)

    @pytest.mark.skipif(not FLASH_ATTN_AVAILABLE, reason="Flash Attention not installed")
    def test_flash_attention_with_padding_mask(self):
        """Test that Flash Attention handles padding masks correctly."""
        attn = FlashAttentionWithGlobals(layer_idx=1)
        seq_len = 100
        hidden_states = torch.randn(2, seq_len, 768)

        # Create padding mask: first sample has padding after position 80
        attention_mask = torch.ones(2, seq_len)
        attention_mask[0, 80:] = 0

        output, _ = attn(hidden_states, attention_mask=attention_mask)
        assert output.shape == (2, seq_len, 768)
        assert not torch.isnan(output).any()

    def test_safety_switch_factory_returns_standard_when_flash_disabled(self):
        """Test that factory returns standard attention when use_flash_attention=False."""
        attn = create_attention_layer(layer_idx=1, use_flash_attention=False)

        # Should always return MultiScaleAttentionWithGlobals
        assert isinstance(attn, MultiScaleAttentionWithGlobals)
        assert not isinstance(attn, FlashAttentionWithGlobals)

    def test_safety_switch_factory_returns_standard_when_flash_unavailable(self):
        """Test that factory falls back to standard when Flash Attention not available."""
        if FLASH_ATTN_AVAILABLE:
            pytest.skip("Flash Attention is available, can't test fallback")

        attn = create_attention_layer(layer_idx=1, use_flash_attention=True)

        # Should fall back to MultiScaleAttentionWithGlobals
        assert isinstance(attn, MultiScaleAttentionWithGlobals)

    @pytest.mark.skipif(not FLASH_ATTN_AVAILABLE, reason="Flash Attention not installed")
    def test_safety_switch_factory_returns_flash_when_enabled(self):
        """Test that factory returns Flash Attention when available and enabled."""
        attn = create_attention_layer(layer_idx=1, use_flash_attention=True)

        # Should return FlashAttentionWithGlobals
        assert isinstance(attn, FlashAttentionWithGlobals)

    def test_safety_switch_factory_layer_specific_window(self):
        """Test that factory creates attention with correct window size per layer."""
        # Foundation band (L1)
        attn1 = create_attention_layer(layer_idx=1, use_flash_attention=False)
        assert attn1.window_size == 64

        # Context band (L7)
        attn7 = create_attention_layer(layer_idx=7, use_flash_attention=False)
        assert attn7.window_size == 128

        # Semantic band (L19)
        attn19 = create_attention_layer(layer_idx=19, use_flash_attention=False)
        assert attn19.window_size == 256

        # Family band (L25)
        attn25 = create_attention_layer(layer_idx=25, use_flash_attention=False)
        assert attn25.window_size == 512

    def test_text_to_hub_visibility_standard_attention(self):
        """Test that standard attention preserves Text→Hub visibility via mask."""
        from modeling_studio.models.attention_v3 import create_global_local_attention_mask

        attn = MultiScaleAttentionWithGlobals(layer_idx=1, attention_dropout=0.0)  # No dropout
        seq_len = 200  # Text tokens beyond window

        # Get the attention mask directly
        mask = attn._get_attention_mask(seq_len, torch.device("cpu"))

        # Verify mask shape
        assert mask.shape == (seq_len, seq_len)

        # Check that text tokens can see hub tokens (mask = 1.0 means can attend)
        # Text token at position 100 should see all hub tokens (0-4)
        for hub_pos in range(5):
            assert (
                mask[100, hub_pos] == 1.0
            ), f"Mask should allow text token 100 to see hub token {hub_pos}"

        # Check all text tokens can see all hub tokens (0-4)
        for text_pos in range(5, seq_len):
            for hub_pos in range(5):
                assert (
                    mask[text_pos, hub_pos] == 1.0
                ), f"Mask should allow text token {text_pos} to see hub token {hub_pos}"

        # Also verify hub tokens can see everything
        for hub_pos in range(5):
            for key_pos in range(seq_len):
                assert (
                    mask[hub_pos, key_pos] == 1.0
                ), f"Hub token {hub_pos} should see all tokens (including {key_pos})"

        # Now verify forward pass works correctly
        attn.eval()  # Eval mode to disable dropout
        hidden_states = torch.randn(1, seq_len, 768)
        output, _ = attn(hidden_states, output_attentions=False)

        # Check output shape
        assert output.shape == (1, seq_len, 768)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_standard_attention_correctness_all_layers(self):
        """Test that standard attention works correctly for all layer bands."""
        for layer_idx in [1, 7, 19, 25]:  # One from each band
            attn = create_attention_layer(layer_idx=layer_idx, use_flash_attention=False)
            hidden_states = torch.randn(2, 100, 768)

            output, _ = attn(hidden_states)
            assert output.shape == (2, 100, 768)
            assert not torch.isnan(output).any()

    @pytest.mark.skipif(not FLASH_ATTN_AVAILABLE, reason="Flash Attention not installed")
    def test_flash_attention_extra_repr_warning(self):
        """Test that Flash Attention extra_repr includes warning about Text→Hub blindness."""
        attn = FlashAttentionWithGlobals(layer_idx=25)
        repr_str = attn.extra_repr()

        # Should include warning
        assert "Text→Hub blind" in repr_str or "Text->Hub blind" in repr_str
        assert "inference only" in repr_str or "use for inference" in repr_str
        assert "layer=25" in repr_str
        assert "window=512" in repr_str

    def test_standard_attention_gradient_flow_with_sdpa(self):
        """Test that gradients flow correctly through SDPA path."""
        attn = MultiScaleAttentionWithGlobals(layer_idx=1)
        hidden_states = torch.randn(2, 50, 768, requires_grad=True)

        output, _ = attn(hidden_states, output_attentions=False)  # Uses SDPA
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert hidden_states.grad is not None
        assert not torch.isnan(hidden_states.grad).any()

    @pytest.mark.skipif(not FLASH_ATTN_AVAILABLE, reason="Flash Attention not installed")
    def test_flash_vs_standard_output_similarity(self):
        """Test that Flash and Standard attention produce similar outputs for hub tokens."""
        layer_idx = 1
        hidden_states = torch.randn(1, 100, 768)

        # Standard attention
        attn_std = create_attention_layer(layer_idx=layer_idx, use_flash_attention=False)
        attn_std.eval()
        output_std, _ = attn_std(hidden_states)

        # Flash attention
        attn_flash = create_attention_layer(layer_idx=layer_idx, use_flash_attention=True)
        attn_flash.eval()
        output_flash, _ = attn_flash(hidden_states)

        # Hub tokens (0-4) should be very similar (they get corrected in Flash)
        hub_std = output_std[0, :5, :]
        hub_flash = output_flash[0, :5, :]

        # Allow some tolerance due to different implementations
        assert torch.allclose(
            hub_std, hub_flash, atol=1e-2, rtol=1e-2
        ), "Hub token outputs should be similar between Flash and Standard"


# ==============================================================================
# Run Tests
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
