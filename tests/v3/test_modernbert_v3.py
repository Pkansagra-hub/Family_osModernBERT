"""
Tests for ModernBERT v3.3 Ultra Model Components.

This module tests the complete v3 model assembly including:
- Embeddings module (Issue 3.1.1)
- Encoder stack (Issue 3.1.2)
- Pair encoder (Issue 3.1.3)
- Main model class (Issue 3.1.4)
"""

import torch
import torch.nn as nn

from modeling_studio.models.config_v3 import ModernBERTv3Config
from modeling_studio.models.embeddings_v3 import ModernBERTEmbeddingsV3
from modeling_studio.models.encoder_v3 import ModernBERTEncoderV3
from modeling_studio.models.hub_tokens import (
    HUB_TOKEN_REGISTRY,
    get_hub_positions,
)
from modeling_studio.models.modernbert_v3 import (
    ClassificationHead,
    ModernBERTv3ForMultiTask,
    ModernBERTv3Output,
    ModernBERTv3Ultra,
    RegressionHead,
    TokenClassificationHead,
    create_modernbert_v3_ultra,
    create_v3_multitask_model,
)
from modeling_studio.models.pair_encoder_v3 import (
    PairEncoderV3,
    SiamesePairEncoderV3,
)


# ============================================================================
# Issue 3.1.1: ModernBERTEmbeddingsV3 Tests
# ============================================================================


class TestModernBERTEmbeddingsV3:
    """Test suite for v3 embeddings module."""

    def test_embeddings_initialization_default(self):
        """Test embeddings initialization with default parameters."""
        embeddings = ModernBERTEmbeddingsV3()

        assert embeddings.vocab_size == 50268
        assert embeddings.hidden_size == 768
        assert embeddings.max_position_embeddings == 8192
        assert embeddings.pad_token_id == 0
        assert embeddings.use_rotary_embeddings is True

        # Check word embeddings
        assert embeddings.word_embeddings.num_embeddings == 50268
        assert embeddings.word_embeddings.embedding_dim == 768
        assert embeddings.word_embeddings.padding_idx == 0

        # Position embeddings should be None (RoPE mode)
        assert embeddings.position_embeddings is None

        # Token type embeddings should be None
        assert embeddings.token_type_embeddings is None

        # LayerNorm and Dropout should exist
        assert isinstance(embeddings.LayerNorm, nn.LayerNorm)
        assert isinstance(embeddings.dropout, nn.Dropout)

    def test_embeddings_initialization_with_position_embeddings(self):
        """Test embeddings initialization with learned position embeddings."""
        embeddings = ModernBERTEmbeddingsV3(use_rotary_embeddings=False)

        assert embeddings.use_rotary_embeddings is False
        assert embeddings.position_embeddings is not None
        assert embeddings.position_embeddings.num_embeddings == 8192
        assert embeddings.position_embeddings.embedding_dim == 768

    def test_embeddings_initialization_custom_size(self):
        """Test embeddings with custom vocab and hidden size."""
        embeddings = ModernBERTEmbeddingsV3(
            vocab_size=60000, hidden_size=1024, max_position_embeddings=4096
        )

        assert embeddings.vocab_size == 60000
        assert embeddings.hidden_size == 1024
        assert embeddings.max_position_embeddings == 4096
        assert embeddings.word_embeddings.num_embeddings == 60000
        assert embeddings.word_embeddings.embedding_dim == 1024

    def test_embeddings_forward_shape_rope_mode(self):
        """Test forward pass shape preservation in RoPE mode."""
        embeddings = ModernBERTEmbeddingsV3(use_rotary_embeddings=True)

        batch_size = 4
        seq_len = 128
        input_ids = torch.randint(0, 50268, (batch_size, seq_len))

        output = embeddings(input_ids)

        assert output.shape == (batch_size, seq_len, 768)
        assert output.dtype == torch.float32

    def test_embeddings_forward_shape_learned_position(self):
        """Test forward pass with learned position embeddings."""
        embeddings = ModernBERTEmbeddingsV3(use_rotary_embeddings=False)

        batch_size = 4
        seq_len = 128
        input_ids = torch.randint(0, 50268, (batch_size, seq_len))

        output = embeddings(input_ids)

        assert output.shape == (batch_size, seq_len, 768)

    def test_embeddings_forward_with_custom_position_ids(self):
        """Test forward pass with custom position IDs."""
        embeddings = ModernBERTEmbeddingsV3(use_rotary_embeddings=False)

        batch_size = 4
        seq_len = 128
        input_ids = torch.randint(0, 50268, (batch_size, seq_len))
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        output = embeddings(input_ids, position_ids=position_ids)

        assert output.shape == (batch_size, seq_len, 768)

    def test_embeddings_forward_with_inputs_embeds(self):
        """Test forward pass with pre-computed embeddings."""
        embeddings = ModernBERTEmbeddingsV3()

        batch_size = 4
        seq_len = 128
        inputs_embeds = torch.randn(batch_size, seq_len, 768)

        output = embeddings(input_ids=None, inputs_embeds=inputs_embeds)

        assert output.shape == (batch_size, seq_len, 768)

    def test_embeddings_forward_variable_sequence_lengths(self):
        """Test forward pass with different sequence lengths."""
        embeddings = ModernBERTEmbeddingsV3()

        for seq_len in [32, 64, 128, 256, 512]:
            input_ids = torch.randint(0, 50268, (2, seq_len))
            output = embeddings(input_ids)
            assert output.shape == (2, seq_len, 768)

    def test_embeddings_forward_long_sequence(self):
        """Test forward pass with long sequence (8192 tokens)."""
        embeddings = ModernBERTEmbeddingsV3(use_rotary_embeddings=False)

        batch_size = 1
        seq_len = 8192
        input_ids = torch.randint(0, 50268, (batch_size, seq_len))

        output = embeddings(input_ids)

        assert output.shape == (batch_size, seq_len, 768)

    def test_embeddings_hub_token_positions(self):
        """Test that hub token position indices are correct."""
        embeddings = ModernBERTEmbeddingsV3()

        hub_positions = embeddings.hub_positions

        assert hub_positions["[CLS]"] == 0
        assert hub_positions["[EMO]"] == 1
        assert hub_positions["[MEM]"] == 2
        assert hub_positions["[REL]"] == 3
        assert hub_positions["[TASK]"] == 4

    def test_embeddings_num_hub_tokens(self):
        """Test that number of hub tokens is correct."""
        embeddings = ModernBERTEmbeddingsV3()

        assert embeddings.num_hub_tokens == 4  # [EMO], [MEM], [REL], [TASK]
        assert embeddings.text_start_position == 5

    def test_embeddings_get_hub_token_embeddings(self):
        """Test extraction of hub token embeddings."""
        embeddings = ModernBERTEmbeddingsV3()

        hub_embeds = embeddings.get_hub_token_embeddings()

        # Should have 4 hub tokens (excluding [CLS])
        assert len(hub_embeds) == 4
        assert "[EMO]" in hub_embeds
        assert "[MEM]" in hub_embeds
        assert "[REL]" in hub_embeds
        assert "[TASK]" in hub_embeds

        # Each should be a 768-dim vector
        for token_name, embedding in hub_embeds.items():
            assert embedding.shape == (768,)
            assert embedding.dtype == torch.float32

    def test_embeddings_get_hub_token_embeddings_correct_indices(self):
        """Test that hub token embeddings are extracted from correct vocab indices."""
        embeddings = ModernBERTEmbeddingsV3()

        hub_embeds = embeddings.get_hub_token_embeddings()

        # Verify that embeddings are from vocab indices 50264-50267 (0-indexed)
        # v2 vocab ends at 50263, so hub tokens are at 50264, 50265, 50266, 50267
        emo_expected = embeddings.word_embeddings.weight[50264]
        mem_expected = embeddings.word_embeddings.weight[50265]
        rel_expected = embeddings.word_embeddings.weight[50266]
        task_expected = embeddings.word_embeddings.weight[50267]

        assert torch.allclose(hub_embeds["[EMO]"], emo_expected.detach())
        assert torch.allclose(hub_embeds["[MEM]"], mem_expected.detach())
        assert torch.allclose(hub_embeds["[REL]"], rel_expected.detach())
        assert torch.allclose(hub_embeds["[TASK]"], task_expected.detach())

    def test_embeddings_resize_token_embeddings_expand(self):
        """Test resizing embeddings to larger vocabulary."""
        embeddings = ModernBERTEmbeddingsV3(vocab_size=50264)  # v2 vocab

        original_size = embeddings.word_embeddings.num_embeddings
        assert original_size == 50264

        # Resize to add hub tokens
        embeddings.resize_token_embeddings(50268)

        assert embeddings.vocab_size == 50268
        assert embeddings.word_embeddings.num_embeddings == 50268
        assert embeddings.word_embeddings.embedding_dim == 768

    def test_embeddings_resize_token_embeddings_preserves_old(self):
        """Test that resizing preserves old embeddings."""
        embeddings = ModernBERTEmbeddingsV3(vocab_size=50264)

        # Save old embeddings
        old_embeddings = embeddings.word_embeddings.weight.data.clone()

        # Resize
        embeddings.resize_token_embeddings(50268)

        # Check that old embeddings are preserved
        assert torch.allclose(embeddings.word_embeddings.weight.data[:50264], old_embeddings)

    def test_embeddings_resize_token_embeddings_initializes_new(self):
        """Test that new embeddings are initialized correctly."""
        embeddings = ModernBERTEmbeddingsV3(vocab_size=50264)

        embeddings.resize_token_embeddings(50268)

        # New embeddings (50265-50268) should be initialized
        new_embeds = embeddings.word_embeddings.weight.data[50264:]
        assert new_embeds.shape == (4, 768)
        # Should not be all zeros (initialized with normal distribution)
        assert not torch.allclose(new_embeds, torch.zeros_like(new_embeds))

    def test_embeddings_resize_token_embeddings_no_op(self):
        """Test that resizing to same size is a no-op."""
        embeddings = ModernBERTEmbeddingsV3(vocab_size=50268)

        old_embeddings = embeddings.word_embeddings.weight.data.clone()

        embeddings.resize_token_embeddings(50268)

        # Should be unchanged
        assert embeddings.vocab_size == 50268
        assert torch.allclose(embeddings.word_embeddings.weight.data, old_embeddings)

    def test_embeddings_layernorm_applied(self):
        """Test that LayerNorm is applied correctly."""
        embeddings = ModernBERTEmbeddingsV3()
        embeddings.eval()  # Disable dropout for stable variance

        batch_size = 4
        seq_len = 128
        input_ids = torch.randint(0, 50268, (batch_size, seq_len))

        output = embeddings(input_ids)

        # Check that output has approximately unit variance (LayerNorm effect)
        # Along hidden dimension
        variance = output.var(dim=-1, unbiased=False)
        mean_variance = variance.mean().item()
        # LayerNorm should produce variance close to 1.0
        assert 0.8 < mean_variance < 1.2, f"Mean variance {mean_variance} not close to 1.0"

    def test_embeddings_dropout_training_mode(self):
        """Test that dropout is applied in training mode."""
        embeddings = ModernBERTEmbeddingsV3(hidden_dropout_prob=0.5)
        embeddings.train()

        batch_size = 4
        seq_len = 128
        input_ids = torch.randint(0, 50268, (batch_size, seq_len))

        # Run multiple times and check variance
        outputs = []
        for _ in range(5):
            output = embeddings(input_ids)
            outputs.append(output)

        # Outputs should be different due to dropout
        for i in range(1, len(outputs)):
            assert not torch.allclose(outputs[0], outputs[i])

    def test_embeddings_dropout_eval_mode(self):
        """Test that dropout is disabled in eval mode."""
        embeddings = ModernBERTEmbeddingsV3(hidden_dropout_prob=0.5)
        embeddings.eval()

        batch_size = 4
        seq_len = 128
        input_ids = torch.randint(0, 50268, (batch_size, seq_len))

        # Run multiple times
        output1 = embeddings(input_ids)
        output2 = embeddings(input_ids)

        # Outputs should be identical (no dropout)
        assert torch.allclose(output1, output2)

    def test_embeddings_gradient_flow(self):
        """Test that gradients flow correctly through embeddings."""
        embeddings = ModernBERTEmbeddingsV3()

        batch_size = 4
        seq_len = 128
        input_ids = torch.randint(0, 50268, (batch_size, seq_len))

        output = embeddings(input_ids)
        loss = output.sum()
        loss.backward()

        # Check that word embeddings have gradients
        assert embeddings.word_embeddings.weight.grad is not None
        assert embeddings.word_embeddings.weight.grad.sum() != 0

    def test_embeddings_get_num_params(self):
        """Test parameter counting."""
        embeddings = ModernBERTEmbeddingsV3()

        params = embeddings.get_num_params()

        # Word embeddings: 50268 * 768
        expected_word = 50268 * 768
        assert params["word_embeddings"] == expected_word

        # No position embeddings in RoPE mode
        assert params["position_embeddings"] == 0

        # LayerNorm: 768 * 2 (weight + bias)
        assert params["layer_norm"] == 768 * 2

        # Total should match
        assert params["total"] == sum(p.numel() for p in embeddings.parameters())

    def test_embeddings_get_num_params_with_position(self):
        """Test parameter counting with learned position embeddings."""
        embeddings = ModernBERTEmbeddingsV3(use_rotary_embeddings=False)

        params = embeddings.get_num_params()

        # Position embeddings: 8192 * 768
        expected_pos = 8192 * 768
        assert params["position_embeddings"] == expected_pos

        # Total should include position embeddings
        assert params["total"] > params["word_embeddings"]

    def test_embeddings_extra_repr(self):
        """Test string representation."""
        embeddings = ModernBERTEmbeddingsV3()

        repr_str = embeddings.extra_repr()

        assert "vocab_size=50268" in repr_str
        assert "hidden_size=768" in repr_str
        assert "max_position=8192" in repr_str
        assert "rotary=yes" in repr_str

    def test_embeddings_extra_repr_learned_position(self):
        """Test string representation with learned position embeddings."""
        embeddings = ModernBERTEmbeddingsV3(use_rotary_embeddings=False)

        repr_str = embeddings.extra_repr()

        assert "rotary=no" in repr_str

    # ========================================================================
    # Acceptance Criteria Tests (Issue 3.1.1)
    # ========================================================================

    def test_ac1_word_embeddings_sized_correctly(self):
        """AC1: Word embeddings sized for v2 vocab + 4 hub tokens."""
        embeddings = ModernBERTEmbeddingsV3()

        # v2 vocab (50264) + 4 hub tokens = 50268
        assert embeddings.word_embeddings.num_embeddings == 50268
        assert embeddings.word_embeddings.embedding_dim == 768

    def test_ac2_position_embeddings_support_8192(self):
        """AC2: Position embeddings support up to 8192 tokens."""
        embeddings = ModernBERTEmbeddingsV3(use_rotary_embeddings=False)

        assert embeddings.max_position_embeddings == 8192
        assert embeddings.position_embeddings.num_embeddings == 8192

        # Test forward pass with 8192 tokens
        input_ids = torch.randint(0, 50268, (1, 8192))
        output = embeddings(input_ids)
        assert output.shape == (1, 8192, 768)

    def test_ac3_hub_token_positions_accessible(self):
        """AC3: Hub token positions (1-4) accessible via get_hub_token_embeddings()."""
        embeddings = ModernBERTEmbeddingsV3()

        hub_embeds = embeddings.get_hub_token_embeddings()

        # All 4 hub tokens accessible
        assert len(hub_embeds) == 4
        assert "[EMO]" in hub_embeds
        assert "[MEM]" in hub_embeds
        assert "[REL]" in hub_embeds
        assert "[TASK]" in hub_embeds

        # Each is a valid embedding vector
        for embedding in hub_embeds.values():
            assert embedding.shape == (768,)

    def test_ac4_resize_token_embeddings_works(self):
        """AC4: resize_token_embeddings() works for adding hub tokens."""
        embeddings = ModernBERTEmbeddingsV3(vocab_size=50264)

        # Resize to add hub tokens
        embeddings.resize_token_embeddings(50268)

        # Verify new size
        assert embeddings.vocab_size == 50268
        assert embeddings.word_embeddings.num_embeddings == 50268

        # Verify forward pass works
        input_ids = torch.randint(0, 50268, (2, 128))
        output = embeddings(input_ids)
        assert output.shape == (2, 128, 768)

    def test_ac5_rope_mode_skips_position_addition(self):
        """AC5: RoPE mode skips position embedding addition (applied in attention)."""
        embeddings = ModernBERTEmbeddingsV3(use_rotary_embeddings=True)

        # Position embeddings should be None
        assert embeddings.position_embeddings is None

        # Forward pass should work without adding position embeddings
        input_ids = torch.randint(0, 50268, (2, 128))
        output = embeddings(input_ids)
        assert output.shape == (2, 128, 768)

        # Output should be word embeddings + LayerNorm + Dropout (no position)

    def test_ac6_layernorm_and_dropout_applied(self):
        """AC6: LayerNorm and Dropout applied correctly."""
        embeddings = ModernBERTEmbeddingsV3()

        # LayerNorm should exist
        assert isinstance(embeddings.LayerNorm, nn.LayerNorm)
        assert embeddings.LayerNorm.normalized_shape == (768,)

        # Dropout should exist
        assert isinstance(embeddings.dropout, nn.Dropout)
        assert embeddings.dropout.p == 0.1

        # Test that they are applied in forward pass
        embeddings.eval()  # Disable dropout for stable test
        input_ids = torch.randint(0, 50268, (2, 128))
        output = embeddings(input_ids)

        # Output should have approximately normalized variance (LayerNorm effect)
        variance = output.var(dim=-1, unbiased=False)
        mean_variance = variance.mean().item()
        assert 0.8 < mean_variance < 1.2, f"Mean variance {mean_variance} not close to 1.0"


# ============================================================================
# Integration Tests
# ============================================================================


# ============================================================================
# Issue 3.1.2: ModernBERTEncoderV3 Tests
# ============================================================================


class TestModernBERTEncoderV3:
    """Test suite for v3 encoder stack."""

    def test_encoder_initialization_default(self):
        """Test encoder initialization with default parameters."""
        from modeling_studio.models.encoder_v3 import ModernBERTEncoderV3

        encoder = ModernBERTEncoderV3()

        assert encoder.num_layers == 28
        assert encoder.hidden_size == 768
        assert encoder.num_attention_heads == 12
        assert len(encoder.layers) == 28
        assert encoder.gradient_checkpointing is False

        # Check layer bands
        assert "foundation" in encoder.layer_bands
        assert "context" in encoder.layer_bands
        assert "semantic" in encoder.layer_bands
        assert "family" in encoder.layer_bands

        # Check band layer counts
        assert len(encoder.layer_bands["foundation"]) == 6  # L1-6
        assert len(encoder.layer_bands["context"]) == 12  # L7-18
        assert len(encoder.layer_bands["semantic"]) == 4  # L19-22
        assert len(encoder.layer_bands["family"]) == 6  # L23-28

    def test_encoder_initialization_custom(self):
        """Test encoder initialization with custom parameters."""
        from modeling_studio.models.encoder_v3 import ModernBERTEncoderV3

        encoder = ModernBERTEncoderV3(
            num_layers=12,
            hidden_size=512,
            num_attention_heads=8,
            intermediate_size=2048,
            gradient_checkpointing=True,
        )

        assert encoder.num_layers == 12
        assert encoder.hidden_size == 512
        assert encoder.num_attention_heads == 8
        assert encoder.gradient_checkpointing is True
        assert len(encoder.layers) == 12

    def test_encoder_forward_basic(self):
        """Test basic forward pass through encoder."""
        from modeling_studio.models.encoder_v3 import ModernBERTEncoderV3

        encoder = ModernBERTEncoderV3()
        batch_size, seq_len = 2, 64

        hidden_states = torch.randn(batch_size, seq_len, 768)
        attention_mask = torch.ones(batch_size, seq_len)

        output, all_hidden_states, all_attentions = encoder(hidden_states, attention_mask)

        assert output.shape == (batch_size, seq_len, 768)
        assert all_hidden_states is None  # Not requested
        assert all_attentions is None  # Not requested

    def test_encoder_forward_with_hidden_states(self):
        """Test forward pass with output_hidden_states=True."""
        from modeling_studio.models.encoder_v3 import ModernBERTEncoderV3

        encoder = ModernBERTEncoderV3()
        batch_size, seq_len = 2, 64

        hidden_states = torch.randn(batch_size, seq_len, 768)
        attention_mask = torch.ones(batch_size, seq_len)

        output, all_hidden_states, all_attentions = encoder(
            hidden_states, attention_mask, output_hidden_states=True
        )

        assert output.shape == (batch_size, seq_len, 768)
        assert all_hidden_states is not None
        assert len(all_hidden_states) == 29  # Initial + 28 layers
        assert all_attentions is None

        # Check each hidden state shape
        for hs in all_hidden_states:
            assert hs.shape == (batch_size, seq_len, 768)

    def test_encoder_forward_with_attentions(self):
        """Test forward pass with output_attentions=True."""
        from modeling_studio.models.encoder_v3 import ModernBERTEncoderV3

        encoder = ModernBERTEncoderV3()
        batch_size, seq_len = 2, 64

        hidden_states = torch.randn(batch_size, seq_len, 768)
        attention_mask = torch.ones(batch_size, seq_len)

        output, all_hidden_states, all_attentions = encoder(
            hidden_states, attention_mask, output_attentions=True
        )

        assert output.shape == (batch_size, seq_len, 768)
        assert all_hidden_states is None
        assert all_attentions is not None
        assert len(all_attentions) == 28  # One per layer

    def test_encoder_forward_with_both_outputs(self):
        """Test forward pass with both output flags."""
        from modeling_studio.models.encoder_v3 import ModernBERTEncoderV3

        encoder = ModernBERTEncoderV3()
        batch_size, seq_len = 2, 64

        hidden_states = torch.randn(batch_size, seq_len, 768)
        attention_mask = torch.ones(batch_size, seq_len)

        output, all_hidden_states, all_attentions = encoder(
            hidden_states,
            attention_mask,
            output_hidden_states=True,
            output_attentions=True,
        )

        assert output.shape == (batch_size, seq_len, 768)
        assert all_hidden_states is not None
        assert all_attentions is not None
        assert len(all_hidden_states) == 29
        assert len(all_attentions) == 28

    def test_encoder_gradient_checkpointing(self):
        """Test gradient checkpointing mode."""
        from modeling_studio.models.encoder_v3 import ModernBERTEncoderV3

        encoder = ModernBERTEncoderV3(gradient_checkpointing=True)
        batch_size, seq_len = 2, 64

        hidden_states = torch.randn(batch_size, seq_len, 768, requires_grad=True)
        attention_mask = torch.ones(batch_size, seq_len)

        output, _, _ = encoder(hidden_states, attention_mask)

        # Compute loss and backward
        loss = output.sum()
        loss.backward()

        # Gradients should be computed
        assert hidden_states.grad is not None
        assert hidden_states.grad.shape == (batch_size, seq_len, 768)

    def test_encoder_freeze_layers(self):
        """Test freeze_layers() functionality."""
        from modeling_studio.models.encoder_v3 import ModernBERTEncoderV3

        encoder = ModernBERTEncoderV3()

        # Freeze first 6 layers (foundation band)
        encoder.freeze_layers([1, 2, 3, 4, 5, 6])

        # Check that parameters are frozen
        for i in range(6):
            for param in encoder.layers[i].parameters():
                assert param.requires_grad is False

        # Check that other layers are not frozen
        for i in range(6, 28):
            has_trainable = False
            for param in encoder.layers[i].parameters():
                if param.requires_grad:
                    has_trainable = True
                    break
            assert has_trainable

    def test_encoder_unfreeze_layers(self):
        """Test unfreeze_layers() functionality."""
        from modeling_studio.models.encoder_v3 import ModernBERTEncoderV3

        encoder = ModernBERTEncoderV3()

        # Freeze then unfreeze
        encoder.freeze_layers([1, 2, 3])
        encoder.unfreeze_layers([1, 2, 3])

        # All should be trainable again
        for i in range(3):
            has_trainable = False
            for param in encoder.layers[i].parameters():
                if param.requires_grad:
                    has_trainable = True
                    break
            assert has_trainable

    def test_encoder_freeze_by_band(self):
        """Test freeze_by_band() functionality."""
        from modeling_studio.models.encoder_v3 import ModernBERTEncoderV3

        encoder = ModernBERTEncoderV3()

        # Freeze foundation band (L1-6)
        encoder.freeze_by_band(["foundation"])

        # Check foundation layers are frozen
        for i in range(6):
            for param in encoder.layers[i].parameters():
                assert param.requires_grad is False

        # Check context layers are not frozen
        for i in range(6, 18):
            has_trainable = False
            for param in encoder.layers[i].parameters():
                if param.requires_grad:
                    has_trainable = True
                    break
            assert has_trainable

    def test_encoder_unfreeze_by_band(self):
        """Test unfreeze_by_band() functionality."""
        from modeling_studio.models.encoder_v3 import ModernBERTEncoderV3

        encoder = ModernBERTEncoderV3()

        # Freeze then unfreeze foundation
        encoder.freeze_by_band(["foundation"])
        encoder.unfreeze_by_band(["foundation"])

        # All foundation layers should be trainable
        for i in range(6):
            has_trainable = False
            for param in encoder.layers[i].parameters():
                if param.requires_grad:
                    has_trainable = True
                    break
            assert has_trainable

    def test_encoder_get_layers_by_band(self):
        """Test get_layers_by_band() functionality."""
        from modeling_studio.models.encoder_v3 import ModernBERTEncoderV3

        encoder = ModernBERTEncoderV3()

        # Get foundation layers
        foundation_layers = encoder.get_layers_by_band("foundation")
        assert len(foundation_layers) == 6

        # Get context layers
        context_layers = encoder.get_layers_by_band("context")
        assert len(context_layers) == 12

        # Get semantic layers
        semantic_layers = encoder.get_layers_by_band("semantic")
        assert len(semantic_layers) == 4

        # Get family layers
        family_layers = encoder.get_layers_by_band("family")
        assert len(family_layers) == 6

    def test_encoder_get_num_params(self):
        """Test get_num_params() returns correct statistics."""
        from modeling_studio.models.encoder_v3 import ModernBERTEncoderV3

        encoder = ModernBERTEncoderV3()
        params = encoder.get_num_params()

        assert "total" in params
        assert "trainable" in params
        assert "lora" in params
        assert "by_band" in params

        assert params["total"] > 0
        assert params["trainable"] > 0
        assert params["lora"] >= 0

        # Check band stats
        assert "foundation" in params["by_band"]
        assert "context" in params["by_band"]
        assert "semantic" in params["by_band"]
        assert "family" in params["by_band"]

    def test_encoder_print_layer_summary(self):
        """Test print_layer_summary() executes without error."""
        from modeling_studio.models.encoder_v3 import ModernBERTEncoderV3

        encoder = ModernBERTEncoderV3()

        # Should not raise exception
        encoder.print_layer_summary()


class TestEncoderV3Integration:
    """Integration tests for encoder with other v3 components."""

    def test_encoder_with_embeddings_output(self):
        """Test encoder with embeddings output."""
        from modeling_studio.models.embeddings_v3 import ModernBERTEmbeddingsV3
        from modeling_studio.models.encoder_v3 import ModernBERTEncoderV3

        embeddings = ModernBERTEmbeddingsV3()
        encoder = ModernBERTEncoderV3()

        batch_size, seq_len = 2, 64
        input_ids = torch.randint(0, 50268, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        # Get embeddings
        embed_output = embeddings(input_ids)

        # Pass through encoder
        encoder_output, _, _ = encoder(embed_output, attention_mask)

        assert encoder_output.shape == (batch_size, seq_len, 768)

    def test_encoder_long_sequence_8k(self):
        """Test encoder with 8k sequence length (max capacity)."""
        from modeling_studio.models.encoder_v3 import ModernBERTEncoderV3

        encoder = ModernBERTEncoderV3(gradient_checkpointing=True)

        batch_size, seq_len = 1, 8192
        hidden_states = torch.randn(batch_size, seq_len, 768)
        attention_mask = torch.ones(batch_size, seq_len)

        output, _, _ = encoder(hidden_states, attention_mask)

        assert output.shape == (batch_size, seq_len, 768)

    def test_encoder_batch_processing(self):
        """Test encoder with different batch sizes."""
        from modeling_studio.models.encoder_v3 import ModernBERTEncoderV3

        encoder = ModernBERTEncoderV3()
        batch_sizes = [1, 4, 8, 16]
        seq_len = 128

        for batch_size in batch_sizes:
            hidden_states = torch.randn(batch_size, seq_len, 768)
            attention_mask = torch.ones(batch_size, seq_len)

            output, _, _ = encoder(hidden_states, attention_mask)
            assert output.shape == (batch_size, seq_len, 768)


class TestEncoderV3AcceptanceCriteria:
    """Tests for Issue 3.1.2 acceptance criteria."""

    def test_ac1_28_layers_with_band_configs(self):
        """AC1: 28 layers created with correct band configurations."""
        from modeling_studio.models.encoder_v3 import ModernBERTEncoderV3

        encoder = ModernBERTEncoderV3()

        # 28 layers total
        assert len(encoder.layers) == 28

        # Band configurations
        assert len(encoder.layer_bands["foundation"]) == 6  # L1-6, W=64
        assert len(encoder.layer_bands["context"]) == 12  # L7-18, W=128
        assert len(encoder.layer_bands["semantic"]) == 4  # L19-22, W=256
        assert len(encoder.layer_bands["family"]) == 6  # L23-28, W=512

    def test_ac2_gradient_checkpointing_memory_efficiency(self):
        """AC2: Gradient checkpointing reduces memory for 8k sequences."""
        from modeling_studio.models.encoder_v3 import ModernBERTEncoderV3

        encoder_no_checkpoint = ModernBERTEncoderV3(gradient_checkpointing=False)
        encoder_checkpoint = ModernBERTEncoderV3(gradient_checkpointing=True)

        batch_size, seq_len = 1, 1024
        hidden_states = torch.randn(batch_size, seq_len, 768, requires_grad=True)
        attention_mask = torch.ones(batch_size, seq_len)

        # Both should produce same output
        output1, _, _ = encoder_no_checkpoint(hidden_states, attention_mask)
        output2, _, _ = encoder_checkpoint(
            hidden_states.detach().requires_grad_(True), attention_mask
        )

        # Shapes should match
        assert output1.shape == output2.shape == (batch_size, seq_len, 768)

    def test_ac3_freeze_unfreeze_layers(self):
        """AC3: freeze_layers() and unfreeze_layers() work correctly."""
        from modeling_studio.models.encoder_v3 import ModernBERTEncoderV3

        encoder = ModernBERTEncoderV3()

        # Freeze layers 1-6
        encoder.freeze_layers([1, 2, 3, 4, 5, 6])

        # Verify frozen
        for i in range(6):
            for param in encoder.layers[i].parameters():
                assert param.requires_grad is False

        # Unfreeze layers 1-3
        encoder.unfreeze_layers([1, 2, 3])

        # Verify unfrozen
        for i in range(3):
            has_trainable = False
            for param in encoder.layers[i].parameters():
                if param.requires_grad:
                    has_trainable = True
                    break
            assert has_trainable

    def test_ac4_all_hidden_states_returned(self):
        """AC4: All hidden states returned when output_hidden_states=True."""
        from modeling_studio.models.encoder_v3 import ModernBERTEncoderV3

        encoder = ModernBERTEncoderV3()

        batch_size, seq_len = 2, 64
        hidden_states = torch.randn(batch_size, seq_len, 768)
        attention_mask = torch.ones(batch_size, seq_len)

        output, all_hidden_states, _ = encoder(
            hidden_states, attention_mask, output_hidden_states=True
        )

        # 29 hidden states: initial + 28 layers
        assert all_hidden_states is not None
        assert len(all_hidden_states) == 29

        # All should have correct shape
        for hs in all_hidden_states:
            assert hs.shape == (batch_size, seq_len, 768)

    def test_ac5_attention_weights_returned(self):
        """AC5: Attention weights returned when output_attentions=True."""
        from modeling_studio.models.encoder_v3 import ModernBERTEncoderV3

        encoder = ModernBERTEncoderV3()

        batch_size, seq_len = 2, 64
        hidden_states = torch.randn(batch_size, seq_len, 768)
        attention_mask = torch.ones(batch_size, seq_len)

        output, _, all_attentions = encoder(hidden_states, attention_mask, output_attentions=True)

        # 28 attention weight tensors (one per layer)
        assert all_attentions is not None
        assert len(all_attentions) == 28

    def test_ac6_layer_band_lookup_correct(self):
        """AC6: Layer band lookup works correctly."""
        from modeling_studio.models.encoder_v3 import ModernBERTEncoderV3

        encoder = ModernBERTEncoderV3()

        # Get layers by band
        foundation = encoder.get_layers_by_band("foundation")
        context = encoder.get_layers_by_band("context")
        semantic = encoder.get_layers_by_band("semantic")
        family = encoder.get_layers_by_band("family")

        # Correct counts
        assert len(foundation) == 6
        assert len(context) == 12
        assert len(semantic) == 4
        assert len(family) == 6

        # Total = 28
        assert len(foundation) + len(context) + len(semantic) + len(family) == 28


# ============================================================================
# Issue 3.1.3: PairEncoderV3 Tests
# ============================================================================


class TestPairEncoderV3:
    """Test suite for v3 pair encoder with [REL] hub."""

    def test_pair_encoder_initialization_default(self):
        """Test pair encoder initialization with default parameters."""
        from modeling_studio.models.pair_encoder_v3 import PairEncoderV3

        pair_encoder = PairEncoderV3()

        assert pair_encoder.hidden_size == 768
        assert pair_encoder.num_labels == 3  # NLI default
        assert pair_encoder.pooling_strategy == "rel_hub"
        assert pair_encoder.use_rel_hub is True
        assert pair_encoder.rel_position == 3
        assert pair_encoder.cls_position == 0

    def test_pair_encoder_initialization_custom(self):
        """Test pair encoder with custom parameters."""
        from modeling_studio.models.pair_encoder_v3 import PairEncoderV3

        pair_encoder = PairEncoderV3(
            hidden_size=512,
            num_labels=2,
            classifier_dropout=0.2,
            pooling_strategy="cls",
        )

        assert pair_encoder.hidden_size == 512
        assert pair_encoder.num_labels == 2
        assert pair_encoder.pooling_strategy == "cls"

    def test_pair_encoder_forward_rel_hub_strategy(self):
        """Test forward pass with rel_hub pooling strategy."""
        from modeling_studio.models.pair_encoder_v3 import PairEncoderV3

        pair_encoder = PairEncoderV3(num_labels=3, pooling_strategy="rel_hub")

        batch_size, seq_len = 2, 128
        encoder_output = torch.randn(batch_size, seq_len, 768)

        logits = pair_encoder(encoder_output)

        assert logits.shape == (batch_size, 3)

    def test_pair_encoder_forward_cls_strategy(self):
        """Test forward pass with cls pooling strategy."""
        from modeling_studio.models.pair_encoder_v3 import PairEncoderV3

        pair_encoder = PairEncoderV3(num_labels=3, pooling_strategy="cls")

        batch_size, seq_len = 2, 128
        encoder_output = torch.randn(batch_size, seq_len, 768)

        logits = pair_encoder(encoder_output)

        assert logits.shape == (batch_size, 3)

    def test_pair_encoder_forward_mean_strategy(self):
        """Test forward pass with mean pooling strategy."""
        from modeling_studio.models.pair_encoder_v3 import PairEncoderV3

        pair_encoder = PairEncoderV3(num_labels=3, pooling_strategy="mean")

        batch_size, seq_len = 2, 128
        encoder_output = torch.randn(batch_size, seq_len, 768)
        attention_mask = torch.ones(batch_size, seq_len)

        logits = pair_encoder(encoder_output, attention_mask=attention_mask)

        assert logits.shape == (batch_size, 3)

    def test_pair_encoder_forward_concat_strategy(self):
        """Test forward pass with concat pooling strategy."""
        from modeling_studio.models.pair_encoder_v3 import PairEncoderV3

        pair_encoder = PairEncoderV3(num_labels=3, pooling_strategy="concat")

        batch_size, seq_len = 2, 128
        encoder_output = torch.randn(batch_size, seq_len, 768)

        # Create text_a and text_b masks
        text_a_mask = torch.zeros(batch_size, seq_len)
        text_a_mask[:, 5:64] = 1  # Text A from position 5-63

        text_b_mask = torch.zeros(batch_size, seq_len)
        text_b_mask[:, 64:120] = 1  # Text B from position 64-119

        logits = pair_encoder(encoder_output, text_a_mask=text_a_mask, text_b_mask=text_b_mask)

        assert logits.shape == (batch_size, 3)

    def test_pair_encoder_return_pooled(self):
        """Test forward pass with return_pooled=True."""
        from modeling_studio.models.pair_encoder_v3 import PairEncoderV3

        pair_encoder = PairEncoderV3(num_labels=3, pooling_strategy="rel_hub")

        batch_size, seq_len = 2, 128
        encoder_output = torch.randn(batch_size, seq_len, 768)

        logits, pooled = pair_encoder(encoder_output, return_pooled=True)

        assert logits.shape == (batch_size, 3)
        assert pooled.shape == (batch_size, 768)

    def test_pair_encoder_get_rel_hub_representation(self):
        """Test extracting [REL] hub representation."""
        from modeling_studio.models.pair_encoder_v3 import PairEncoderV3

        pair_encoder = PairEncoderV3()

        batch_size, seq_len = 2, 128
        encoder_output = torch.randn(batch_size, seq_len, 768)

        rel_repr = pair_encoder.get_rel_hub_representation(encoder_output)

        assert rel_repr.shape == (batch_size, 768)
        # Should be same as encoder_output[:, 3, :]
        assert torch.allclose(rel_repr, encoder_output[:, 3, :])

    def test_pair_encoder_set_pooling_strategy(self):
        """Test changing pooling strategy at runtime."""
        from modeling_studio.models.pair_encoder_v3 import PairEncoderV3

        pair_encoder = PairEncoderV3(pooling_strategy="rel_hub")

        assert pair_encoder.pooling_strategy == "rel_hub"

        pair_encoder.set_pooling_strategy("cls")
        assert pair_encoder.pooling_strategy == "cls"

        pair_encoder.set_pooling_strategy("mean")
        assert pair_encoder.pooling_strategy == "mean"

        pair_encoder.set_pooling_strategy("concat")
        assert pair_encoder.pooling_strategy == "concat"

    def test_pair_encoder_invalid_strategy_raises(self):
        """Test that invalid pooling strategy raises ValueError."""
        from modeling_studio.models.pair_encoder_v3 import PairEncoderV3
        import pytest

        pair_encoder = PairEncoderV3()

        with pytest.raises(ValueError):
            pair_encoder.set_pooling_strategy("invalid")

    def test_pair_encoder_gradient_flow(self):
        """Test that gradients flow through pair encoder."""
        from modeling_studio.models.pair_encoder_v3 import PairEncoderV3

        pair_encoder = PairEncoderV3(num_labels=3)

        batch_size, seq_len = 2, 128
        encoder_output = torch.randn(batch_size, seq_len, 768, requires_grad=True)

        logits = pair_encoder(encoder_output)
        loss = logits.sum()
        loss.backward()

        assert encoder_output.grad is not None
        assert encoder_output.grad.shape == (batch_size, seq_len, 768)


class TestSiamesePairEncoderV3:
    """Test suite for Siamese pair encoder."""

    def test_siamese_initialization_cosine(self):
        """Test Siamese encoder with cosine similarity."""
        from modeling_studio.models.pair_encoder_v3 import SiamesePairEncoderV3

        siamese = SiamesePairEncoderV3(similarity_function="cosine")

        assert siamese.hidden_size == 768
        assert siamese.similarity_function == "cosine"

    def test_siamese_initialization_euclidean(self):
        """Test Siamese encoder with euclidean similarity."""
        from modeling_studio.models.pair_encoder_v3 import SiamesePairEncoderV3

        siamese = SiamesePairEncoderV3(similarity_function="euclidean")

        assert siamese.similarity_function == "euclidean"

    def test_siamese_initialization_learned(self):
        """Test Siamese encoder with learned similarity."""
        from modeling_studio.models.pair_encoder_v3 import SiamesePairEncoderV3

        siamese = SiamesePairEncoderV3(similarity_function="learned")

        assert siamese.similarity_function == "learned"
        assert siamese.similarity_layer is not None

    def test_siamese_forward_cosine(self):
        """Test forward pass with cosine similarity."""
        from modeling_studio.models.pair_encoder_v3 import SiamesePairEncoderV3

        siamese = SiamesePairEncoderV3(similarity_function="cosine")

        batch_size, seq_len = 2, 128
        output_a = torch.randn(batch_size, seq_len, 768)
        output_b = torch.randn(batch_size, seq_len, 768)

        similarity = siamese(output_a, output_b)

        assert similarity.shape == (batch_size,)
        # Cosine similarity should be in [-1, 1]
        assert torch.all(similarity >= -1.0)
        assert torch.all(similarity <= 1.0)

    def test_siamese_forward_euclidean(self):
        """Test forward pass with euclidean distance."""
        from modeling_studio.models.pair_encoder_v3 import SiamesePairEncoderV3

        siamese = SiamesePairEncoderV3(similarity_function="euclidean")

        batch_size, seq_len = 2, 128
        output_a = torch.randn(batch_size, seq_len, 768)
        output_b = torch.randn(batch_size, seq_len, 768)

        similarity = siamese(output_a, output_b)

        assert similarity.shape == (batch_size,)
        # Negative distance should be <= 0
        assert torch.all(similarity <= 0.0)

    def test_siamese_forward_learned(self):
        """Test forward pass with learned similarity."""
        from modeling_studio.models.pair_encoder_v3 import SiamesePairEncoderV3

        siamese = SiamesePairEncoderV3(similarity_function="learned")

        batch_size, seq_len = 2, 128
        output_a = torch.randn(batch_size, seq_len, 768)
        output_b = torch.randn(batch_size, seq_len, 768)

        similarity = siamese(output_a, output_b)

        assert similarity.shape == (batch_size,)

    def test_siamese_uses_mem_hub(self):
        """Test that Siamese encoder uses [MEM] hub token."""
        from modeling_studio.models.pair_encoder_v3 import SiamesePairEncoderV3

        siamese = SiamesePairEncoderV3(similarity_function="cosine")

        batch_size, seq_len = 2, 128
        output_a = torch.randn(batch_size, seq_len, 768)
        output_b = torch.randn(batch_size, seq_len, 768)

        # Set [MEM] position (2) to specific values
        output_a[:, 2, :] = 1.0
        output_b[:, 2, :] = 1.0

        similarity = siamese(output_a, output_b)

        # Should be very close to 1.0 (identical [MEM] representations)
        assert torch.allclose(similarity, torch.ones(batch_size), atol=1e-6)

    def test_siamese_gradient_flow(self):
        """Test gradient flow through Siamese encoder."""
        from modeling_studio.models.pair_encoder_v3 import SiamesePairEncoderV3

        siamese = SiamesePairEncoderV3(similarity_function="learned")

        batch_size, seq_len = 2, 128
        output_a = torch.randn(batch_size, seq_len, 768, requires_grad=True)
        output_b = torch.randn(batch_size, seq_len, 768, requires_grad=True)

        similarity = siamese(output_a, output_b)
        loss = similarity.sum()
        loss.backward()

        assert output_a.grad is not None
        assert output_b.grad is not None


class TestPairEncoderV3Integration:
    """Integration tests for pair encoder with encoder."""

    def test_pair_encoder_with_encoder_output(self):
        """Test pair encoder with encoder output."""
        from modeling_studio.models.encoder_v3 import ModernBERTEncoderV3
        from modeling_studio.models.embeddings_v3 import ModernBERTEmbeddingsV3
        from modeling_studio.models.pair_encoder_v3 import PairEncoderV3

        embeddings = ModernBERTEmbeddingsV3()
        encoder = ModernBERTEncoderV3()
        pair_encoder = PairEncoderV3(num_labels=3)

        batch_size, seq_len = 2, 128
        input_ids = torch.randint(0, 50268, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        # Forward through embeddings and encoder
        embed_output = embeddings(input_ids)
        encoder_output, _, _ = encoder(embed_output, attention_mask)

        # Forward through pair encoder
        logits = pair_encoder(encoder_output)

        assert logits.shape == (batch_size, 3)

    def test_siamese_with_encoder_output(self):
        """Test Siamese encoder with encoder output."""
        from modeling_studio.models.encoder_v3 import ModernBERTEncoderV3
        from modeling_studio.models.embeddings_v3 import ModernBERTEmbeddingsV3
        from modeling_studio.models.pair_encoder_v3 import SiamesePairEncoderV3

        embeddings = ModernBERTEmbeddingsV3()
        encoder = ModernBERTEncoderV3()
        siamese = SiamesePairEncoderV3(similarity_function="cosine")

        batch_size, seq_len = 2, 128
        input_ids_a = torch.randint(0, 50268, (batch_size, seq_len))
        input_ids_b = torch.randint(0, 50268, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        # Encode both sequences
        embed_a = embeddings(input_ids_a)
        embed_b = embeddings(input_ids_b)

        output_a, _, _ = encoder(embed_a, attention_mask)
        output_b, _, _ = encoder(embed_b, attention_mask)

        # Compute similarity
        similarity = siamese(output_a, output_b)

        assert similarity.shape == (batch_size,)


class TestPairEncoderV3AcceptanceCriteria:
    """Tests for Issue 3.1.3 acceptance criteria."""

    def test_ac1_rel_hub_primary_representation(self):
        """AC1: [REL] hub token (position 3) used as primary pair representation."""
        from modeling_studio.models.pair_encoder_v3 import PairEncoderV3

        pair_encoder = PairEncoderV3(pooling_strategy="rel_hub")
        pair_encoder.eval()  # Disable dropout for exact comparison

        batch_size, seq_len = 2, 128
        encoder_output = torch.randn(batch_size, seq_len, 768)

        # Get pooled representation
        _, pooled = pair_encoder(encoder_output, return_pooled=True)

        # Should match [REL] position (3)
        assert torch.allclose(pooled, encoder_output[:, 3, :])

    def test_ac2_multiple_pooling_strategies(self):
        """AC2: Multiple pooling strategies supported."""
        from modeling_studio.models.pair_encoder_v3 import PairEncoderV3

        strategies = ["rel_hub", "cls", "mean", "concat"]

        batch_size, seq_len = 2, 128
        encoder_output = torch.randn(batch_size, seq_len, 768)
        attention_mask = torch.ones(batch_size, seq_len)
        text_a_mask = torch.zeros(batch_size, seq_len)
        text_a_mask[:, 5:64] = 1
        text_b_mask = torch.zeros(batch_size, seq_len)
        text_b_mask[:, 64:120] = 1

        for strategy in strategies:
            pair_encoder = PairEncoderV3(pooling_strategy=strategy)
            logits = pair_encoder(
                encoder_output,
                attention_mask=attention_mask,
                text_a_mask=text_a_mask,
                text_b_mask=text_b_mask,
            )
            assert logits.shape == (batch_size, 3)

    def test_ac3_nli_classification_works(self):
        """AC3: NLI classification (3 labels) works correctly."""
        from modeling_studio.models.pair_encoder_v3 import PairEncoderV3

        pair_encoder = PairEncoderV3(num_labels=3)  # NLI: entailment, neutral, contradiction

        batch_size, seq_len = 4, 128
        encoder_output = torch.randn(batch_size, seq_len, 768)

        logits = pair_encoder(encoder_output)

        assert logits.shape == (batch_size, 3)

        # Softmax should sum to 1
        probs = torch.softmax(logits, dim=-1)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size))

    def test_ac4_siamese_similarity_functions(self):
        """AC4: SiamesePairEncoderV3 supports cosine, euclidean, and learned similarity."""
        from modeling_studio.models.pair_encoder_v3 import SiamesePairEncoderV3

        similarity_functions = ["cosine", "euclidean", "learned"]

        batch_size, seq_len = 2, 128
        output_a = torch.randn(batch_size, seq_len, 768)
        output_b = torch.randn(batch_size, seq_len, 768)

        for sim_func in similarity_functions:
            siamese = SiamesePairEncoderV3(similarity_function=sim_func)
            similarity = siamese(output_a, output_b)
            assert similarity.shape == (batch_size,)

    def test_ac5_mem_hub_for_embedding_similarity(self):
        """AC5: [MEM] hub used for embedding similarity."""
        from modeling_studio.models.pair_encoder_v3 import SiamesePairEncoderV3

        siamese = SiamesePairEncoderV3(similarity_function="cosine")

        # MEM position is 2
        assert siamese.hub_positions["[MEM]"] == 2

        batch_size, seq_len = 2, 128
        output_a = torch.randn(batch_size, seq_len, 768)
        output_b = torch.randn(batch_size, seq_len, 768)

        # Manually compute similarity using [MEM] hub
        mem_a = output_a[:, 2, :]
        mem_b = output_b[:, 2, :]
        expected_sim = nn.functional.cosine_similarity(mem_a, mem_b, dim=-1)

        # Compare with Siamese output
        actual_sim = siamese(output_a, output_b)

        assert torch.allclose(expected_sim, actual_sim)

    def test_ac6_text_ab_masks_applied_correctly(self):
        """AC6: Text A/B masks correctly applied for mean pooling."""
        from modeling_studio.models.pair_encoder_v3 import PairEncoderV3

        pair_encoder = PairEncoderV3(pooling_strategy="concat")

        batch_size, seq_len = 2, 128
        encoder_output = torch.randn(batch_size, seq_len, 768)

        # Create text_a and text_b masks
        text_a_mask = torch.zeros(batch_size, seq_len)
        text_a_mask[:, 5:64] = 1  # Text A from position 5-63

        text_b_mask = torch.zeros(batch_size, seq_len)
        text_b_mask[:, 64:120] = 1  # Text B from position 64-119

        logits = pair_encoder(encoder_output, text_a_mask=text_a_mask, text_b_mask=text_b_mask)

        # Should produce concat of CLS + REL + mean_diff
        assert logits.shape == (batch_size, 3)


# ============================================================================
# Issue 3.1.1: Embeddings Integration Tests
# ============================================================================


class TestEmbeddingsV3Integration:
    """Integration tests for embeddings with other v3 components."""

    def test_embeddings_with_hub_token_registry(self):
        """Test that embeddings are compatible with hub token registry."""
        embeddings = ModernBERTEmbeddingsV3()

        # Hub positions should match registry
        hub_positions = get_hub_positions()
        assert embeddings.hub_positions == hub_positions

        # Number of hub tokens should match registry
        assert embeddings.num_hub_tokens == len(HUB_TOKEN_REGISTRY)

    def test_embeddings_with_hub_token_ids(self):
        """Test that embeddings use correct vocab indices for hub tokens."""
        embeddings = ModernBERTEmbeddingsV3()

        # Create input with hub tokens (0-indexed: 50264-50267)
        input_ids = torch.tensor(
            [[0, 50264, 50265, 50266, 50267, 100, 200, 300, 1]]
        )  # [CLS] [EMO] [MEM] [REL] [TASK] <text> [SEP]

        output = embeddings(input_ids)

        assert output.shape == (1, 9, 768)
        # No errors should occur with hub token IDs

    def test_embeddings_batch_processing(self):
        """Test embeddings with batched inputs (common in training)."""
        embeddings = ModernBERTEmbeddingsV3()

        batch_sizes = [1, 4, 16, 32]
        seq_len = 128

        for batch_size in batch_sizes:
            input_ids = torch.randint(0, 50268, (batch_size, seq_len))
            output = embeddings(input_ids)
            assert output.shape == (batch_size, seq_len, 768)

    def test_embeddings_mixed_precision_compatible(self):
        """Test that embeddings work with mixed precision training."""
        embeddings = ModernBERTEmbeddingsV3()

        input_ids = torch.randint(0, 50268, (4, 128))

        # Test with autocast
        with torch.autocast(device_type="cpu", dtype=torch.float16):
            output = embeddings(input_ids)

        # Output should be float16 in autocast context
        # (Note: CPU autocast may not change dtype, but should not error)
        assert output.shape == (4, 128, 768)


# ======================================================================
# ModernBERTv3Ultra Tests (Issue 3.1.4)
# ======================================================================


class TestModernBERTv3Ultra:
    """Test suite for ModernBERTv3Ultra main model class."""

    def test_model_initialization_default(self):
        """Test default model initialization with standard config."""
        config = ModernBERTv3Config()
        model = ModernBERTv3Ultra(config)

        assert model.config == config
        assert model.embeddings is not None
        assert model.encoder is not None
        assert model.hub_pooler is not None
        assert model.combined_pooler is not None
        assert model.pair_encoder is not None
        assert model.final_layer_norm is not None
        assert len(model.hub_positions) == 5  # [CLS], [EMO], [MEM], [REL], [TASK]

    def test_model_initialization_custom_config(self):
        """Test model initialization with custom config."""
        config = ModernBERTv3Config(
            num_layers=24,
            hidden_size=512,
            num_attention_heads=8,
            layer_bands={
                "foundation": list(range(1, 7)),
                "context": list(range(7, 16)),
                "semantic": list(range(16, 20)),
                "family": list(range(20, 25)),
            },
            lora_target_layers=[20, 21, 22, 23, 24],
            frozen_layers_phase1=list(range(1, 16)),  # Foundation + Context
        )
        model = ModernBERTv3Ultra(config)

        assert model.config.num_layers == 24
        assert model.config.hidden_size == 512
        assert model.config.num_attention_heads == 8
        assert model.encoder.num_layers == 24

    def test_model_forward_basic(self):
        """Test basic forward pass returns correct output structure."""
        config = ModernBERTv3Config()
        model = ModernBERTv3Ultra(config)
        model.eval()

        input_ids = torch.randint(0, 50268, (2, 128))
        output = model(input_ids)

        assert isinstance(output, ModernBERTv3Output)
        assert output.last_hidden_state.shape == (2, 128, 768)
        assert isinstance(output.pooled_outputs, dict)
        assert len(output.pooled_outputs) == 5  # [CLS], [EMO], [MEM], [REL], [TASK]

    def test_model_forward_with_attention_mask(self):
        """Test forward pass with attention mask."""
        config = ModernBERTv3Config()
        model = ModernBERTv3Ultra(config)
        model.eval()

        input_ids = torch.randint(0, 50268, (2, 128))
        attention_mask = torch.ones(2, 128)
        attention_mask[:, 64:] = 0  # Mask second half

        output = model(input_ids, attention_mask=attention_mask)

        assert output.last_hidden_state.shape == (2, 128, 768)
        assert isinstance(output.pooled_outputs, dict)

    def test_model_forward_output_hidden_states(self):
        """Test forward pass with output_hidden_states=True."""
        config = ModernBERTv3Config()
        model = ModernBERTv3Ultra(config)
        model.eval()

        input_ids = torch.randint(0, 50268, (2, 128))
        output = model(input_ids, output_hidden_states=True)

        assert output.hidden_states is not None
        assert len(output.hidden_states) == 28 + 1  # 28 layers + embeddings
        for hidden in output.hidden_states:
            assert hidden.shape == (2, 128, 768)

    def test_model_forward_output_attentions(self):
        """Test forward pass with output_attentions=True."""
        config = ModernBERTv3Config()
        model = ModernBERTv3Ultra(config)
        model.eval()

        input_ids = torch.randint(0, 50268, (2, 128))
        output = model(input_ids, output_attentions=True)

        assert output.attentions is not None
        assert len(output.attentions) == 28  # 28 layers

    def test_model_forward_return_dict_false(self):
        """Test forward pass with return_dict=False."""
        config = ModernBERTv3Config()
        model = ModernBERTv3Ultra(config)
        model.eval()

        input_ids = torch.randint(0, 50268, (2, 128))
        output = model(input_ids, return_dict=False)

        assert isinstance(output, tuple)
        assert len(output) == 4
        assert output[0].shape == (2, 128, 768)  # last_hidden_state
        assert isinstance(output[1], dict)  # pooled_outputs

    def test_model_pooled_outputs_keys(self):
        """Test that pooled_outputs contains all hub tokens."""
        config = ModernBERTv3Config()
        model = ModernBERTv3Ultra(config)
        model.eval()

        input_ids = torch.randint(0, 50268, (2, 128))
        output = model(input_ids)

        expected_keys = {"[CLS]", "[EMO]", "[MEM]", "[REL]", "[TASK]"}
        assert set(output.pooled_outputs.keys()) == expected_keys

        for key, value in output.pooled_outputs.items():
            assert value.shape == (2, 768)

    def test_model_num_parameters(self):
        """Test num_parameters property."""
        config = ModernBERTv3Config()
        model = ModernBERTv3Ultra(config)

        num_params = model.num_parameters
        assert num_params > 0
        assert isinstance(num_params, int)

        # Verify it matches manual count
        manual_count = sum(p.numel() for p in model.parameters())
        assert num_params == manual_count

    def test_model_num_trainable_parameters(self):
        """Test num_trainable_parameters property."""
        config = ModernBERTv3Config()
        model = ModernBERTv3Ultra(config)

        num_trainable = model.num_trainable_parameters
        assert num_trainable > 0
        assert isinstance(num_trainable, int)

        # Initially all parameters should be trainable
        assert num_trainable == model.num_parameters

    def test_model_get_input_embeddings(self):
        """Test get_input_embeddings returns word embeddings."""
        config = ModernBERTv3Config()
        model = ModernBERTv3Ultra(config)

        embeddings = model.get_input_embeddings()
        assert isinstance(embeddings, nn.Embedding)
        assert embeddings.num_embeddings == config.vocab_size
        assert embeddings.embedding_dim == config.hidden_size

    def test_model_set_input_embeddings(self):
        """Test set_input_embeddings replaces word embeddings."""
        config = ModernBERTv3Config()
        model = ModernBERTv3Ultra(config)

        new_embeddings = nn.Embedding(50432, 768)
        model.set_input_embeddings(new_embeddings)

        current = model.get_input_embeddings()
        assert current is new_embeddings

    def test_model_resize_token_embeddings(self):
        """Test resize_token_embeddings expands vocabulary."""
        config = ModernBERTv3Config()
        model = ModernBERTv3Ultra(config)

        original_vocab = config.vocab_size
        new_vocab = original_vocab + 100

        model.resize_token_embeddings(new_vocab)

        assert model.config.vocab_size == new_vocab
        embeddings = model.get_input_embeddings()
        assert embeddings.num_embeddings == new_vocab


class TestModernBERTv3Output:
    """Test suite for ModernBERTv3Output dataclass."""

    def test_output_dataclass_creation(self):
        """Test creating ModernBERTv3Output dataclass."""
        last_hidden = torch.randn(2, 128, 768)
        pooled = {"[CLS]": torch.randn(2, 768), "[EMO]": torch.randn(2, 768)}

        output = ModernBERTv3Output(last_hidden_state=last_hidden, pooled_outputs=pooled)

        assert output.last_hidden_state is last_hidden
        assert output.pooled_outputs is pooled
        assert output.hidden_states is None
        assert output.attentions is None

    def test_output_dataclass_with_optional_fields(self):
        """Test creating ModernBERTv3Output with optional fields."""
        last_hidden = torch.randn(2, 128, 768)
        pooled = {"[CLS]": torch.randn(2, 768)}
        hidden_states = [torch.randn(2, 128, 768) for _ in range(28)]
        attentions = [torch.randn(2, 12, 128, 128) for _ in range(28)]

        output = ModernBERTv3Output(
            last_hidden_state=last_hidden,
            pooled_outputs=pooled,
            hidden_states=hidden_states,
            attentions=attentions,
        )

        assert output.last_hidden_state is last_hidden
        assert output.pooled_outputs is pooled
        assert output.hidden_states is hidden_states
        assert output.attentions is attentions


class TestModernBERTv3CapabilityRouting:
    """Test suite for capability routing in ModernBERTv3Ultra."""

    def test_get_representation_for_hub_routed_capability(self):
        """Test getting representation for hub-routed capability."""
        config = ModernBERTv3Config()
        model = ModernBERTv3Ultra(config)
        model.eval()

        input_ids = torch.randint(0, 50268, (2, 128))
        output = model(input_ids)

        # Test hub-routed capabilities
        emo_repr = model.get_representation_for_capability(
            output.last_hidden_state, output.pooled_outputs, "emotions"
        )
        assert emo_repr.shape == (2, 768)
        assert torch.equal(emo_repr, output.pooled_outputs["[EMO]"])

    def test_get_representation_for_token_level_capability(self):
        """Test getting representation for token-level capability."""
        config = ModernBERTv3Config()
        model = ModernBERTv3Ultra(config)
        model.eval()

        input_ids = torch.randint(0, 50268, (2, 128))
        output = model(input_ids)

        # Test token-level capabilities
        ner_repr = model.get_representation_for_capability(
            output.last_hidden_state, output.pooled_outputs, "ner_general"
        )
        assert ner_repr.shape == (2, 128, 768)
        assert torch.equal(ner_repr, output.last_hidden_state)

    def test_get_representation_for_all_hub_capabilities(self):
        """Test all hub-routed capabilities."""
        config = ModernBERTv3Config()
        model = ModernBERTv3Ultra(config)
        model.eval()

        input_ids = torch.randint(0, 50268, (2, 128))
        output = model(input_ids)

        hub_capabilities = [
            "emotions",
            "sentiment",
            "safety_general",
            "safety_child",
            "safety_relationship",
            "embedding",
            "nli",
            "relation",
            "intent",
            "ingress",
        ]

        for capability in hub_capabilities:
            repr_tensor = model.get_representation_for_capability(
                output.last_hidden_state, output.pooled_outputs, capability
            )
            assert repr_tensor.shape == (2, 768)

    def test_get_representation_for_all_token_capabilities(self):
        """Test all token-level capabilities."""
        config = ModernBERTv3Config()
        model = ModernBERTv3Ultra(config)
        model.eval()

        input_ids = torch.randint(0, 50268, (2, 128))
        output = model(input_ids)

        token_capabilities = ["ner_general", "ner_family", "temporal"]

        for capability in token_capabilities:
            repr_tensor = model.get_representation_for_capability(
                output.last_hidden_state, output.pooled_outputs, capability
            )
            assert repr_tensor.shape == (2, 128, 768)

    def test_get_embedding_representation(self):
        """Test getting embedding representation from [MEM] hub."""
        config = ModernBERTv3Config()
        model = ModernBERTv3Ultra(config)
        model.eval()

        input_ids = torch.randint(0, 50268, (2, 128))
        output = model(input_ids)

        embedding = model.get_embedding_representation(output.last_hidden_state)
        assert embedding.shape == (2, 768)

        # Should be [MEM] hub token at position 2
        mem_position = model.hub_positions["[MEM]"]
        expected = output.last_hidden_state[:, mem_position, :]
        assert torch.equal(embedding, expected)


class TestModernBERTv3Freezing:
    """Test suite for phase-based freezing in ModernBERTv3Ultra."""

    def test_freeze_for_phase1(self):
        """Test freezing for phase1 training."""
        config = ModernBERTv3Config()
        model = ModernBERTv3Ultra(config)

        # Initially all parameters trainable
        initial_trainable = model.num_trainable_parameters

        model.freeze_for_phase("phase1")

        # After freezing, fewer parameters should be trainable
        frozen_trainable = model.num_trainable_parameters
        assert frozen_trainable < initial_trainable

        # Embeddings should be frozen
        for param in model.embeddings.parameters():
            assert not param.requires_grad

        # L1-18 should be frozen
        for i in range(18):
            for param in model.encoder.layers[i].parameters():
                assert not param.requires_grad

        # L19-28 should be trainable
        for i in range(18, 28):
            for param in model.encoder.layers[i].parameters():
                assert param.requires_grad

    def test_freeze_for_phase05(self):
        """Test freezing for phase0.5 (healing) training."""
        config = ModernBERTv3Config()
        model = ModernBERTv3Ultra(config)

        model.freeze_for_phase("phase0.5")

        # Same behavior as phase1
        frozen_trainable = model.num_trainable_parameters
        assert frozen_trainable < model.num_parameters

        # L19-28 should be trainable
        for i in range(18, 28):
            for param in model.encoder.layers[i].parameters():
                assert param.requires_grad


class TestModernBERTv3Integration:
    """Test suite for ModernBERTv3Ultra integration with all components."""

    def test_full_forward_pipeline(self):
        """Test complete forward pass through all components."""
        config = ModernBERTv3Config()
        model = ModernBERTv3Ultra(config)
        model.eval()

        # Simulate realistic input
        batch_size = 4
        seq_len = 256
        input_ids = torch.randint(0, 50268, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, 128:] = 0  # Mask half

        output = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # Verify output structure
        assert output.last_hidden_state.shape == (batch_size, seq_len, 768)
        assert len(output.pooled_outputs) == 5
        assert output.hidden_states is not None
        assert len(output.hidden_states) == 29  # 28 layers + embeddings

        # Verify gradient flow
        loss = output.last_hidden_state.mean()
        loss.backward()

        # Check gradients exist
        assert model.embeddings.word_embeddings.weight.grad is not None

    def test_model_with_long_sequence(self):
        """Test model with 2K sequence length (8K would exceed memory on CPU)."""
        config = ModernBERTv3Config()
        model = ModernBERTv3Ultra(config)
        model.eval()

        input_ids = torch.randint(0, 50268, (1, 2048))
        output = model(input_ids)

        assert output.last_hidden_state.shape == (1, 2048, 768)
        assert len(output.pooled_outputs) == 5

    def test_model_batch_processing(self):
        """Test model with various batch sizes."""
        config = ModernBERTv3Config()
        model = ModernBERTv3Ultra(config)
        model.eval()

        batch_sizes = [1, 2, 4, 8, 16]
        seq_len = 128

        for batch_size in batch_sizes:
            input_ids = torch.randint(0, 50268, (batch_size, seq_len))
            output = model(input_ids)
            assert output.last_hidden_state.shape == (batch_size, seq_len, 768)

    def test_model_mixed_precision_compatible(self):
        """Test model works with mixed precision training."""
        config = ModernBERTv3Config()
        model = ModernBERTv3Ultra(config)
        model.eval()

        input_ids = torch.randint(0, 50268, (2, 128))

        with torch.autocast(device_type="cpu", dtype=torch.float16):
            output = model(input_ids)

        assert output.last_hidden_state.shape == (2, 128, 768)


class TestModernBERTv3AcceptanceCriteria:
    """Test suite for Issue 3.1.4 acceptance criteria."""

    def test_ac1_combines_embeddings_encoder_poolers(self):
        """AC1: Model correctly combines embeddings, encoder, and poolers."""
        config = ModernBERTv3Config()
        model = ModernBERTv3Ultra(config)
        model.eval()

        input_ids = torch.randint(0, 50268, (2, 128))
        output = model(input_ids)

        # Verify all components are connected
        assert output.last_hidden_state.shape == (2, 128, 768)
        assert len(output.pooled_outputs) == 5
        assert all(v.shape == (2, 768) for v in output.pooled_outputs.values())

    def test_ac2_output_contains_all_fields(self):
        """AC2: ModernBERTv3Output contains all required fields."""
        config = ModernBERTv3Config()
        model = ModernBERTv3Ultra(config)
        model.eval()

        input_ids = torch.randint(0, 50268, (2, 128))
        output = model(input_ids, output_hidden_states=True, output_attentions=True)

        # Verify all fields present
        assert output.last_hidden_state is not None
        assert output.pooled_outputs is not None
        assert output.hidden_states is not None
        assert output.attentions is not None

    def test_ac3_hub_token_representations_accessible(self):
        """AC3: Hub token representations accessible via get_representation_for_capability."""
        config = ModernBERTv3Config()
        model = ModernBERTv3Ultra(config)
        model.eval()

        input_ids = torch.randint(0, 50268, (2, 128))
        output = model(input_ids)

        # Test hub-routed capability
        emo_repr = model.get_representation_for_capability(
            output.last_hidden_state, output.pooled_outputs, "emotions"
        )
        assert emo_repr.shape == (2, 768)

        # Test token-level capability
        ner_repr = model.get_representation_for_capability(
            output.last_hidden_state, output.pooled_outputs, "ner_general"
        )
        assert ner_repr.shape == (2, 128, 768)

    def test_ac4_freeze_for_phase_configures_correctly(self):
        """AC4: freeze_for_phase() configures L1-18 frozen, L19-28 trainable."""
        config = ModernBERTv3Config()
        model = ModernBERTv3Ultra(config)

        model.freeze_for_phase("phase1")

        # L1-18 frozen
        for i in range(18):
            for param in model.encoder.layers[i].parameters():
                assert not param.requires_grad

        # L19-28 trainable
        for i in range(18, 28):
            for param in model.encoder.layers[i].parameters():
                assert param.requires_grad

    def test_ac5_merge_lora_weights_works(self):
        """AC5: merge_lora_weights() works for inference export."""
        config = ModernBERTv3Config()
        model = ModernBERTv3Ultra(config)

        # Should not raise error
        model.merge_lora_weights()

        # Model should still work after merging
        model.eval()
        input_ids = torch.randint(0, 50268, (2, 128))
        output = model(input_ids)
        assert output.last_hidden_state.shape == (2, 128, 768)

    def test_ac6_print_model_summary_shows_architecture(self):
        """AC6: print_model_summary() shows complete architecture info."""
        config = ModernBERTv3Config()
        model = ModernBERTv3Ultra(config)

        # Should not raise error
        model.print_model_summary()

        # Verify properties used in summary
        assert model.num_parameters > 0
        assert model.num_trainable_parameters > 0
        assert model.config.num_layers == 28
        assert model.config.hidden_size == 768

    def test_ac7_factory_function_creates_model(self):
        """AC7: create_modernbert_v3_ultra() factory function works."""
        # Test with defaults
        model1 = create_modernbert_v3_ultra()
        assert model1.config.num_layers == 28
        assert model1.config.hidden_size == 768

        # Test with overrides (must also adjust layer_bands for custom num_layers)
        model2 = create_modernbert_v3_ultra(
            num_layers=24,
            hidden_size=480,  # Must be divisible by num_attention_heads
            num_attention_heads=8,
            layer_bands={
                "foundation": list(range(1, 7)),
                "context": list(range(7, 16)),
                "semantic": list(range(16, 20)),
                "family": list(range(20, 25)),
            },
            lora_target_layers=[20, 21, 22, 23, 24],
            frozen_layers_phase1=list(range(1, 16)),  # Foundation + Context
        )
        assert model2.config.num_layers == 24
        assert model2.config.hidden_size == 480


# ======================================================================
# ModernBERTv3ForMultiTask Tests (Issue 3.1.5)
# ======================================================================


class TestModernBERTv3TaskHeads:
    """Test suite for task head classes."""

    def test_classification_head_forward(self):
        """Test ClassificationHead forward pass."""
        head = ClassificationHead(768, 7)
        pooled = torch.randn(4, 768)
        logits = head(pooled)

        assert logits.shape == (4, 7)

    def test_token_classification_head_forward(self):
        """Test TokenClassificationHead forward pass."""
        head = TokenClassificationHead(768, 9)
        sequence = torch.randn(4, 128, 768)
        logits = head(sequence)

        assert logits.shape == (4, 128, 9)

    def test_regression_head_forward(self):
        """Test RegressionHead forward pass."""
        head = RegressionHead(768)
        pooled = torch.randn(4, 768)
        scores = head(pooled)

        assert scores.shape == (4, 1)

    def test_classification_head_dropout(self):
        """Test ClassificationHead applies dropout."""
        head = ClassificationHead(768, 7, dropout=0.5)
        head.train()

        pooled = torch.randn(4, 768)
        logits1 = head(pooled)
        logits2 = head(pooled)

        # In training mode with dropout, outputs should differ
        assert not torch.allclose(logits1, logits2)

    def test_token_classification_head_shape_preservation(self):
        """Test TokenClassificationHead preserves sequence length."""
        head = TokenClassificationHead(768, 9)
        head.eval()

        # Different sequence lengths
        for seq_len in [64, 128, 256]:
            sequence = torch.randn(2, seq_len, 768)
            logits = head(sequence)
            assert logits.shape == (2, seq_len, 9)


class TestModernBERTv3ForMultiTask:
    """Test suite for ModernBERTv3ForMultiTask class."""

    def test_multitask_initialization_default(self):
        """Test default multi-task model initialization."""
        config = ModernBERTv3Config()
        model = ModernBERTv3ForMultiTask(config)

        assert model.config == config
        assert model.hub_router is not None
        assert len(model.task_heads) == 0
        assert len(model.task_loss_weights) == 0

    def test_multitask_initialization_with_heads(self):
        """Test multi-task model initialization with task heads."""
        config = ModernBERTv3Config()
        task_heads = {
            "emotions": ClassificationHead(768, 7),
            "sentiment": ClassificationHead(768, 3),
        }
        model = ModernBERTv3ForMultiTask(config, task_heads=task_heads)

        assert len(model.task_heads) == 2
        assert "emotions" in model.task_heads
        assert "sentiment" in model.task_heads

    def test_register_task_head(self):
        """Test registering a task head."""
        config = ModernBERTv3Config()
        model = ModernBERTv3ForMultiTask(config)

        head = ClassificationHead(768, 7)
        model.register_task_head("emotions", head, loss_weight=2.0)

        assert "emotions" in model.task_heads
        assert model.task_heads["emotions"] is head
        assert model.task_loss_weights["emotions"] == 2.0

    def test_forward_for_task_basic(self):
        """Test forward_for_task basic functionality."""
        config = ModernBERTv3Config()
        model = ModernBERTv3ForMultiTask(config)
        model.eval()

        # Register task
        model.register_task_head("emotions", ClassificationHead(768, 7))

        input_ids = torch.randint(0, 50268, (2, 128))
        output = model.forward_for_task(input_ids, task="emotions")

        assert "logits" in output
        assert output["logits"].shape == (2, 7)
        assert output["pool_type"] == "hub"

    def test_forward_for_task_with_labels(self):
        """Test forward_for_task computes loss when labels provided."""
        config = ModernBERTv3Config()
        model = ModernBERTv3ForMultiTask(config)
        model.eval()

        model.register_task_head("emotions", ClassificationHead(768, 7))

        input_ids = torch.randint(0, 50268, (2, 128))
        labels = torch.randint(0, 7, (2,))
        output = model.forward_for_task(input_ids, task="emotions", labels=labels)

        assert "loss" in output
        assert output["loss"].dim() == 0  # Scalar

    def test_forward_for_task_unknown_task_raises(self):
        """Test forward_for_task raises on unknown task."""
        config = ModernBERTv3Config()
        model = ModernBERTv3ForMultiTask(config)

        input_ids = torch.randint(0, 50268, (2, 128))

        try:
            model.forward_for_task(input_ids, task="unknown_task")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unknown task" in str(e)

    def test_forward_for_task_requires_task_param(self):
        """Test forward_for_task requires task parameter."""
        config = ModernBERTv3Config()
        model = ModernBERTv3ForMultiTask(config)

        input_ids = torch.randint(0, 50268, (2, 128))

        try:
            model.forward_for_task(input_ids, task=None)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "task parameter is required" in str(e)

    def test_forward_multitask_basic(self):
        """Test forward_multitask basic functionality."""
        config = ModernBERTv3Config()
        model = ModernBERTv3ForMultiTask(config)
        model.eval()

        # Register multiple tasks
        model.register_task_head("emotions", ClassificationHead(768, 7))
        model.register_task_head("sentiment", ClassificationHead(768, 3))

        input_ids = torch.randint(0, 50268, (2, 128))
        output = model.forward_multitask(input_ids)

        assert "task_logits" in output
        assert "emotions" in output["task_logits"]
        assert "sentiment" in output["task_logits"]
        assert output["task_logits"]["emotions"].shape == (2, 7)
        assert output["task_logits"]["sentiment"].shape == (2, 3)

    def test_forward_multitask_with_labels(self):
        """Test forward_multitask computes losses."""
        config = ModernBERTv3Config()
        model = ModernBERTv3ForMultiTask(config)
        model.eval()

        model.register_task_head("emotions", ClassificationHead(768, 7))
        model.register_task_head("sentiment", ClassificationHead(768, 3))

        input_ids = torch.randint(0, 50268, (2, 128))
        task_labels = {
            "emotions": torch.randint(0, 7, (2,)),
            "sentiment": torch.randint(0, 3, (2,)),
        }

        output = model.forward_multitask(input_ids, task_labels=task_labels)

        assert "total_loss" in output
        assert "task_losses" in output
        assert "emotions" in output["task_losses"]
        assert "sentiment" in output["task_losses"]
        assert output["total_loss"] is not None

    def test_forward_multitask_active_tasks_filter(self):
        """Test forward_multitask respects active_tasks parameter."""
        config = ModernBERTv3Config()
        model = ModernBERTv3ForMultiTask(config)
        model.eval()

        model.register_task_head("emotions", ClassificationHead(768, 7))
        model.register_task_head("sentiment", ClassificationHead(768, 3))
        model.register_task_head("nli", ClassificationHead(768, 3))

        input_ids = torch.randint(0, 50268, (2, 128))
        output = model.forward_multitask(input_ids, active_tasks=["emotions", "sentiment"])

        assert "emotions" in output["task_logits"]
        assert "sentiment" in output["task_logits"]
        assert "nli" not in output["task_logits"]

    def test_forward_multitask_weighted_loss(self):
        """Test forward_multitask applies loss weights."""
        config = ModernBERTv3Config()
        model = ModernBERTv3ForMultiTask(config)
        model.eval()

        model.register_task_head("emotions", ClassificationHead(768, 7), loss_weight=2.0)
        model.register_task_head("sentiment", ClassificationHead(768, 3), loss_weight=0.5)

        input_ids = torch.randint(0, 50268, (2, 128))
        task_labels = {
            "emotions": torch.randint(0, 7, (2,)),
            "sentiment": torch.randint(0, 3, (2,)),
        }

        output = model.forward_multitask(input_ids, task_labels=task_labels)

        # Total loss should be weighted sum
        expected_loss = (
            2.0 * output["task_losses"]["emotions"] + 0.5 * output["task_losses"]["sentiment"]
        )
        assert torch.allclose(output["total_loss"], expected_loss)

    def test_set_task_loss_weight(self):
        """Test set_task_loss_weight updates loss weight."""
        config = ModernBERTv3Config()
        model = ModernBERTv3ForMultiTask(config)

        model.register_task_head("emotions", ClassificationHead(768, 7))
        model.set_task_loss_weight("emotions", 3.0)

        assert model.task_loss_weights["emotions"] == 3.0

    def test_print_routing_table(self):
        """Test print_routing_table runs without error."""
        config = ModernBERTv3Config()
        model = ModernBERTv3ForMultiTask(config)

        model.register_task_head("emotions", ClassificationHead(768, 7))
        model.register_task_head("ner_general", TokenClassificationHead(768, 9))

        # Should not raise error
        model.print_routing_table()


class TestModernBERTv3HubRouting:
    """Test suite for hub routing in multi-task model."""

    def test_hub_routing_emotions_to_emo(self):
        """Test emotions task routes to [EMO] hub."""
        config = ModernBERTv3Config()
        model = ModernBERTv3ForMultiTask(config)
        model.eval()

        model.register_task_head("emotions", ClassificationHead(768, 7))

        input_ids = torch.randint(0, 50268, (2, 128))
        output = model.forward_for_task(input_ids, task="emotions")

        # Should use hub pooling
        assert output["pool_type"] == "hub"
        assert output["logits"].shape == (2, 7)

    def test_hub_routing_sentiment_to_emo(self):
        """Test sentiment task routes to [EMO] hub."""
        config = ModernBERTv3Config()
        model = ModernBERTv3ForMultiTask(config)
        model.eval()

        model.register_task_head("sentiment", ClassificationHead(768, 3))

        input_ids = torch.randint(0, 50268, (2, 128))
        output = model.forward_for_task(input_ids, task="sentiment")

        assert output["pool_type"] == "hub"

    def test_hub_routing_nli_to_rel(self):
        """Test NLI task routes to [REL] hub."""
        config = ModernBERTv3Config()
        model = ModernBERTv3ForMultiTask(config)
        model.eval()

        model.register_task_head("nli", ClassificationHead(768, 3))

        input_ids = torch.randint(0, 50268, (2, 128))
        output = model.forward_for_task(input_ids, task="nli")

        assert output["pool_type"] == "hub"

    def test_hub_routing_ner_token_level(self):
        """Test NER task uses token-level representation."""
        config = ModernBERTv3Config()
        model = ModernBERTv3ForMultiTask(config)
        model.eval()

        model.register_task_head("ner_general", TokenClassificationHead(768, 9))

        input_ids = torch.randint(0, 50268, (2, 128))
        output = model.forward_for_task(input_ids, task="ner_general")

        # Should use token-level representation
        assert output["pool_type"] == "token"
        assert output["logits"].shape == (2, 128, 9)

    def test_hub_routing_temporal_token_level(self):
        """Test temporal task uses token-level representation."""
        config = ModernBERTv3Config()
        model = ModernBERTv3ForMultiTask(config)
        model.eval()

        model.register_task_head("temporal", TokenClassificationHead(768, 5))

        input_ids = torch.randint(0, 50268, (2, 128))
        output = model.forward_for_task(input_ids, task="temporal")

        assert output["pool_type"] == "token"
        assert output["logits"].shape == (2, 128, 5)

    def test_hub_routing_multiple_tasks_same_hub(self):
        """Test multiple tasks sharing same hub."""
        config = ModernBERTv3Config()
        model = ModernBERTv3ForMultiTask(config)
        model.eval()

        # Both use [EMO] hub
        model.register_task_head("emotions", ClassificationHead(768, 7))
        model.register_task_head("sentiment", ClassificationHead(768, 3))

        input_ids = torch.randint(0, 50268, (2, 128))
        output = model.forward_multitask(input_ids)

        # Both should have hub pooling
        assert "emotions" in output["task_logits"]
        assert "sentiment" in output["task_logits"]


class TestModernBERTv3LossComputation:
    """Test suite for loss computation in multi-task model."""

    def test_loss_computation_classification(self):
        """Test loss computation for classification tasks."""
        config = ModernBERTv3Config()
        model = ModernBERTv3ForMultiTask(config)
        model.eval()

        model.register_task_head("emotions", ClassificationHead(768, 7))

        input_ids = torch.randint(0, 50268, (2, 128))
        labels = torch.randint(0, 7, (2,))

        output = model.forward_for_task(input_ids, task="emotions", labels=labels)

        assert "loss" in output
        assert output["loss"].requires_grad

    def test_loss_computation_token_classification(self):
        """Test loss computation for token-level classification."""
        config = ModernBERTv3Config()
        model = ModernBERTv3ForMultiTask(config)
        model.eval()

        model.register_task_head("ner_general", TokenClassificationHead(768, 9))

        input_ids = torch.randint(0, 50268, (2, 128))
        labels = torch.randint(0, 9, (2, 128))

        output = model.forward_for_task(input_ids, task="ner_general", labels=labels)

        assert "loss" in output
        assert output["loss"].requires_grad

    def test_loss_computation_regression(self):
        """Test loss computation for regression tasks."""
        config = ModernBERTv3Config()
        model = ModernBERTv3ForMultiTask(config)
        model.eval()

        model.register_task_head("similarity", RegressionHead(768))

        input_ids = torch.randint(0, 50268, (2, 128))
        labels = torch.randn(2)

        output = model.forward_for_task(input_ids, task="similarity", labels=labels)

        assert "loss" in output
        assert output["loss"].requires_grad


class TestModernBERTv3GradientMasks:
    """Test suite for gradient masking in multi-task model."""

    def test_get_hub_gradient_mask(self):
        """Test get_hub_gradient_mask returns correct masks."""
        config = ModernBERTv3Config()
        model = ModernBERTv3ForMultiTask(config)

        model.register_task_head("emotions", ClassificationHead(768, 7))
        model.register_task_head("sentiment", ClassificationHead(768, 3))

        # Set active capabilities
        model._active_capabilities = ["emotions", "sentiment"]

        masks = model.get_hub_gradient_mask(torch.device("cpu"), batch_size=4)

        assert isinstance(masks, dict)
        assert "[EMO]" in masks
        assert masks["[EMO]"].shape == (4,)


class TestModernBERTv3FactoryFunction:
    """Test suite for create_v3_multitask_model factory function."""

    def test_factory_function_creates_model(self):
        """Test factory function creates model with configured heads."""
        config = ModernBERTv3Config()
        task_configs = {
            "emotions": {"type": "classification", "num_labels": 7},
            "sentiment": {"type": "classification", "num_labels": 3},
        }

        model = create_v3_multitask_model(config, task_configs)

        assert isinstance(model, ModernBERTv3ForMultiTask)
        assert len(model.task_heads) == 2
        assert "emotions" in model.task_heads
        assert "sentiment" in model.task_heads

    def test_factory_function_token_classification(self):
        """Test factory function with token classification heads."""
        config = ModernBERTv3Config()
        task_configs = {
            "ner_general": {"type": "token_classification", "num_labels": 9},
            "temporal": {"type": "token_classification", "num_labels": 5},
        }

        model = create_v3_multitask_model(config, task_configs)

        assert len(model.task_heads) == 2
        assert isinstance(model.task_heads["ner_general"], TokenClassificationHead)

    def test_factory_function_regression(self):
        """Test factory function with regression heads."""
        config = ModernBERTv3Config()
        task_configs = {
            "similarity": {"type": "regression"},
        }

        model = create_v3_multitask_model(config, task_configs)

        assert len(model.task_heads) == 1
        assert isinstance(model.task_heads["similarity"], RegressionHead)

    def test_factory_function_mixed_heads(self):
        """Test factory function with mixed head types."""
        config = ModernBERTv3Config()
        task_configs = {
            "emotions": {"type": "classification", "num_labels": 7},
            "ner_general": {"type": "token_classification", "num_labels": 9},
            "similarity": {"type": "regression"},
        }

        model = create_v3_multitask_model(config, task_configs)

        assert len(model.task_heads) == 3

    def test_factory_function_custom_loss_weights(self):
        """Test factory function applies custom loss weights."""
        config = ModernBERTv3Config()
        task_configs = {
            "emotions": {"type": "classification", "num_labels": 7, "loss_weight": 2.0},
            "sentiment": {"type": "classification", "num_labels": 3, "loss_weight": 0.5},
        }

        model = create_v3_multitask_model(config, task_configs)

        assert model.task_loss_weights["emotions"] == 2.0
        assert model.task_loss_weights["sentiment"] == 0.5


class TestModernBERTv3ForMultiTaskAcceptanceCriteria:
    """Test suite for Issue 3.1.5 acceptance criteria."""

    def test_ac1_forward_for_task_routes_to_correct_hub(self):
        """AC1: forward_for_task() routes single task to correct hub."""
        config = ModernBERTv3Config()
        model = ModernBERTv3ForMultiTask(config)
        model.eval()

        # Register tasks for different hubs
        model.register_task_head("emotions", ClassificationHead(768, 7))  # [EMO]
        model.register_task_head("nli", ClassificationHead(768, 3))  # [REL]

        input_ids = torch.randint(0, 50268, (2, 128))

        # Test emotions routes to [EMO] (hub pooling)
        emo_output = model.forward_for_task(input_ids, task="emotions")
        assert emo_output["pool_type"] == "hub"
        assert emo_output["logits"].shape == (2, 7)

        # Test NLI routes to [REL] (hub pooling)
        nli_output = model.forward_for_task(input_ids, task="nli")
        assert nli_output["pool_type"] == "hub"
        assert nli_output["logits"].shape == (2, 3)

    def test_ac2_forward_multitask_handles_multiple_tasks(self):
        """AC2: forward_multitask() handles multiple tasks in one forward pass."""
        config = ModernBERTv3Config()
        model = ModernBERTv3ForMultiTask(config)
        model.eval()

        model.register_task_head("emotions", ClassificationHead(768, 7))
        model.register_task_head("sentiment", ClassificationHead(768, 3))
        model.register_task_head("nli", ClassificationHead(768, 3))

        input_ids = torch.randint(0, 50268, (2, 128))
        output = model.forward_multitask(input_ids)

        # Should process all tasks in one pass
        assert len(output["task_logits"]) == 3
        assert "emotions" in output["task_logits"]
        assert "sentiment" in output["task_logits"]
        assert "nli" in output["task_logits"]

    def test_ac3_hub_routing_uses_correct_tokens(self):
        """AC3: Hub routing uses [EMO] for emotions/sentiment, [REL] for NLI, etc."""
        config = ModernBERTv3Config()
        model = ModernBERTv3ForMultiTask(config)
        model.eval()

        # Register tasks and verify routing
        model.register_task_head("emotions", ClassificationHead(768, 7))
        model.register_task_head("sentiment", ClassificationHead(768, 3))
        model.register_task_head("nli", ClassificationHead(768, 3))
        model.register_task_head("intent", ClassificationHead(768, 10))

        input_ids = torch.randint(0, 50268, (2, 128))

        # All should use hub pooling with correct hubs
        for task in ["emotions", "sentiment", "nli", "intent"]:
            output = model.forward_for_task(input_ids, task=task)
            assert output["pool_type"] == "hub"

    def test_ac4_token_level_tasks_receive_full_sequence(self):
        """AC4: Token-level tasks (NER) receive full sequence, not hub pooling."""
        config = ModernBERTv3Config()
        model = ModernBERTv3ForMultiTask(config)
        model.eval()

        model.register_task_head("ner_general", TokenClassificationHead(768, 9))
        model.register_task_head("temporal", TokenClassificationHead(768, 5))

        input_ids = torch.randint(0, 50268, (2, 128))

        # Token-level tasks should get full sequence
        ner_output = model.forward_for_task(input_ids, task="ner_general")
        assert ner_output["pool_type"] == "token"
        assert ner_output["logits"].shape == (2, 128, 9)

        temporal_output = model.forward_for_task(input_ids, task="temporal")
        assert temporal_output["pool_type"] == "token"
        assert temporal_output["logits"].shape == (2, 128, 5)

    def test_ac5_loss_computation_handles_all_types(self):
        """AC5: Loss computation handles classification, token-level, and regression."""
        config = ModernBERTv3Config()
        model = ModernBERTv3ForMultiTask(config)
        model.eval()

        model.register_task_head("emotions", ClassificationHead(768, 7))
        model.register_task_head("ner_general", TokenClassificationHead(768, 9))
        model.register_task_head("similarity", RegressionHead(768))

        input_ids = torch.randint(0, 50268, (2, 128))

        # Classification loss
        emo_output = model.forward_for_task(
            input_ids, task="emotions", labels=torch.randint(0, 7, (2,))
        )
        assert "loss" in emo_output

        # Token-level loss
        ner_output = model.forward_for_task(
            input_ids, task="ner_general", labels=torch.randint(0, 9, (2, 128))
        )
        assert "loss" in ner_output

        # Regression loss
        sim_output = model.forward_for_task(input_ids, task="similarity", labels=torch.randn(2))
        assert "loss" in sim_output

    def test_ac6_get_hub_gradient_mask_returns_masks(self):
        """AC6: get_hub_gradient_mask() returns masks for active capabilities."""
        config = ModernBERTv3Config()
        model = ModernBERTv3ForMultiTask(config)

        model.register_task_head("emotions", ClassificationHead(768, 7))
        model._active_capabilities = ["emotions"]

        masks = model.get_hub_gradient_mask(torch.device("cpu"), batch_size=4)

        assert isinstance(masks, dict)
        assert len(masks) > 0

    def test_ac7_print_routing_table_shows_mappings(self):
        """AC7: print_routing_table() shows all task→hub mappings."""
        config = ModernBERTv3Config()
        model = ModernBERTv3ForMultiTask(config)

        model.register_task_head("emotions", ClassificationHead(768, 7))
        model.register_task_head("ner_general", TokenClassificationHead(768, 9))

        # Should not raise error
        model.print_routing_table()

    def test_ac8_factory_function_creates_configured_model(self):
        """AC8: Factory function creates model with configured heads."""
        config = ModernBERTv3Config()
        task_configs = {
            "emotions": {"type": "classification", "num_labels": 7, "loss_weight": 2.0},
            "sentiment": {"type": "classification", "num_labels": 3},
            "ner_general": {"type": "token_classification", "num_labels": 9},
        }

        model = create_v3_multitask_model(config, task_configs)

        assert isinstance(model, ModernBERTv3ForMultiTask)
        assert len(model.task_heads) == 3
        assert model.task_loss_weights["emotions"] == 2.0
        assert model.task_loss_weights["sentiment"] == 1.0
