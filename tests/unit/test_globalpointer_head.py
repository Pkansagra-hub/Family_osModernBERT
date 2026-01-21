"""
Unit Tests for GlobalPointer NER Head.

Tests the GlobalPointerNERHead that performs span-based NER,
outputting (B, num_labels, L, L) span scores.

Author: FamilyOS Team
Date: January 2026
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from modeling_studio.models.heads import (
    GlobalPointerNERHead,
    create_globalpointer_head,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def device():
    """Get device for tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def hidden_size():
    """Standard hidden size."""
    return 768


@pytest.fixture
def head_size():
    """Standard head size."""
    return 64


@pytest.fixture
def num_labels():
    """Number of entity types for ner_general."""
    return 4


@pytest.fixture
def batch_size():
    """Standard batch size."""
    return 2


@pytest.fixture
def seq_len():
    """Standard sequence length."""
    return 32


@pytest.fixture
def basic_head(hidden_size, num_labels, head_size, device):
    """Create a basic GlobalPointerNERHead."""
    head = GlobalPointerNERHead(
        hidden_size=hidden_size,
        num_labels=num_labels,
        head_size=head_size,
        dropout=0.0,  # No dropout for deterministic tests
        use_rope=True,
    )
    return head.to(device)


@pytest.fixture
def hidden_states(batch_size, seq_len, hidden_size, device):
    """Create random hidden states."""
    return torch.randn(batch_size, seq_len, hidden_size, device=device)


@pytest.fixture
def attention_mask(batch_size, seq_len, device):
    """Create attention mask with some padding."""
    mask = torch.ones(batch_size, seq_len, device=device)
    # Mask last 4 tokens as padding
    mask[:, -4:] = 0
    return mask


# =============================================================================
# Test Initialization
# =============================================================================


class TestGlobalPointerNERHeadInit:
    """Tests for head initialization."""

    def test_init_default_params(self, hidden_size):
        """Head initializes with default parameters."""
        head = GlobalPointerNERHead(hidden_size=hidden_size)

        assert head.hidden_size == hidden_size
        assert head.num_labels == 4  # default
        assert head.head_size == 64  # default
        assert head.use_rope is True
        assert isinstance(head.q_proj, nn.Linear)
        assert isinstance(head.k_proj, nn.Linear)

    def test_init_custom_labels(self, hidden_size):
        """Head respects num_labels parameter."""
        head = GlobalPointerNERHead(hidden_size=hidden_size, num_labels=10)
        assert head.num_labels == 10

        # Check projection dimensions
        expected_out_dim = 10 * 64 * 2  # num_labels * head_size * 2
        assert head.q_proj.out_features == expected_out_dim
        assert head.k_proj.out_features == expected_out_dim

    def test_init_custom_head_size(self, hidden_size, num_labels):
        """Head respects head_size parameter."""
        head = GlobalPointerNERHead(
            hidden_size=hidden_size,
            num_labels=num_labels,
            head_size=32,
        )
        assert head.head_size == 32

        expected_out_dim = num_labels * 32 * 2
        assert head.q_proj.out_features == expected_out_dim

    def test_init_without_rope(self, hidden_size):
        """Head can be created without RoPE."""
        head = GlobalPointerNERHead(hidden_size=hidden_size, use_rope=False)
        assert head.use_rope is False
        assert head.cos_cached is None
        assert head.sin_cached is None

    def test_init_rope_buffers_created(self, hidden_size):
        """RoPE buffers are created when enabled."""
        head = GlobalPointerNERHead(hidden_size=hidden_size, use_rope=True)
        assert head.cos_cached is not None
        assert head.sin_cached is not None
        assert head.cos_cached.shape[0] == 512  # default max_seq_len


# =============================================================================
# Test Forward Pass
# =============================================================================


class TestGlobalPointerNERHeadForward:
    """Tests for forward pass."""

    def test_forward_shape(self, basic_head, hidden_states, num_labels, batch_size, seq_len):
        """Output shape is (B, num_labels, L, L)."""
        output = basic_head(hidden_states)

        assert "logits" in output
        assert output["logits"].shape == (batch_size, num_labels, seq_len, seq_len)

    def test_forward_with_mask(self, basic_head, hidden_states, attention_mask):
        """Padding positions are masked to -inf."""
        output = basic_head(hidden_states, attention_mask=attention_mask)
        logits = output["logits"]

        # Last 4 positions should be masked (set to very negative value)
        # Check that masked positions have very low scores
        assert logits[:, :, :, -4:].max() < -1e10
        assert logits[:, :, -4:, :].max() < -1e10

    def test_upper_triangular_constraint(self, basic_head, hidden_states, batch_size, num_labels, seq_len):
        """Only upper triangle has valid scores (i <= j)."""
        output = basic_head(hidden_states)
        logits = output["logits"]

        # Lower triangle should be masked (very negative)
        for i in range(seq_len):
            for j in range(i):  # j < i means lower triangle
                assert logits[0, 0, i, j].item() < -1e10, f"Position [{i},{j}] should be masked"

    def test_forward_no_rope(self, hidden_size, hidden_states, device):
        """Forward works without RoPE."""
        head = GlobalPointerNERHead(hidden_size=hidden_size, use_rope=False).to(device)
        output = head(hidden_states)

        assert "logits" in output
        assert not torch.isnan(output["logits"]).any()

    def test_forward_deterministic(self, hidden_size, hidden_states, device):
        """Forward is deterministic with same input (no dropout)."""
        head = GlobalPointerNERHead(hidden_size=hidden_size, dropout=0.0).to(device)
        head.eval()

        out1 = head(hidden_states)["logits"]
        out2 = head(hidden_states)["logits"]

        assert torch.allclose(out1, out2)


# =============================================================================
# Test Loss Computation
# =============================================================================


class TestGlobalPointerNERHeadLoss:
    """Tests for loss computation."""

    def test_compute_loss_shape(self, basic_head, hidden_states, num_labels, batch_size, seq_len, device):
        """Loss is scalar tensor."""
        # Create fake span labels
        span_labels = torch.zeros(batch_size, num_labels, seq_len, seq_len, device=device)
        # Add a few positive spans (upper triangular only)
        span_labels[0, 0, 2, 5] = 1  # Entity of type 0 from token 2 to 5
        span_labels[0, 1, 10, 12] = 1  # Entity of type 1 from token 10 to 12

        output = basic_head(hidden_states, span_labels=span_labels)

        assert "loss" in output
        assert output["loss"].dim() == 0  # scalar
        assert output["loss"].requires_grad

    def test_compute_loss_with_mask(self, basic_head, hidden_states, attention_mask, num_labels, batch_size, seq_len, device):
        """Loss respects attention mask."""
        span_labels = torch.zeros(batch_size, num_labels, seq_len, seq_len, device=device)
        span_labels[0, 0, 2, 5] = 1

        output = basic_head(hidden_states, attention_mask=attention_mask, span_labels=span_labels)

        assert "loss" in output
        assert not torch.isnan(output["loss"])
        assert output["loss"].item() >= 0

    def test_loss_zero_with_no_entities(self, hidden_size, hidden_states, num_labels, batch_size, seq_len, device):
        """Loss is computed correctly with no positive entities."""
        head = GlobalPointerNERHead(hidden_size=hidden_size, num_labels=num_labels).to(device)

        span_labels = torch.zeros(batch_size, num_labels, seq_len, seq_len, device=device)
        output = head(hidden_states, span_labels=span_labels)

        # Should not be NaN
        assert not torch.isnan(output["loss"])


# =============================================================================
# Test Decoding
# =============================================================================


class TestGlobalPointerNERHeadDecode:
    """Tests for span decoding."""

    def test_decode_basic(self, basic_head, num_labels, batch_size, seq_len, device):
        """Decode extracts correct spans."""
        # Create scores with clear positive spans
        scores = torch.full((batch_size, num_labels, seq_len, seq_len), -10.0, device=device)
        # Add a positive span: entity type 0 from token 5 to 8
        scores[0, 0, 5, 8] = 5.0  # High positive score

        id2label = {0: "PER", 1: "ORG", 2: "LOC", 3: "MISC"}
        entities = basic_head.decode(scores, threshold=0.0, id2label=id2label)

        assert len(entities) == batch_size
        assert len(entities[0]) >= 1

        # Find the PER entity
        per_entities = [e for e in entities[0] if e["label"] == "PER"]
        assert len(per_entities) >= 1
        assert per_entities[0]["start"] == 5
        assert per_entities[0]["end"] == 8

    def test_decode_threshold(self, basic_head, num_labels, batch_size, seq_len, device):
        """Decode respects threshold parameter."""
        scores = torch.full((batch_size, num_labels, seq_len, seq_len), -10.0, device=device)
        # Add a borderline span
        scores[0, 0, 5, 8] = 0.5  # Slightly positive

        # With low threshold, should find entity
        entities_low = basic_head.decode(scores, threshold=-1.0)
        assert any(e["start"] == 5 and e["end"] == 8 for e in entities_low[0])

        # With high threshold, should not find entity
        entities_high = basic_head.decode(scores, threshold=2.0)
        assert not any(e["start"] == 5 and e["end"] == 8 for e in entities_high[0])

    def test_decode_with_mask(self, basic_head, num_labels, batch_size, seq_len, device):
        """Decode respects attention mask."""
        scores = torch.full((batch_size, num_labels, seq_len, seq_len), -10.0, device=device)
        # Add entity in padding region
        scores[0, 0, seq_len-2, seq_len-1] = 5.0

        # Create mask that marks last 4 as padding
        mask = torch.ones(batch_size, seq_len, device=device)
        mask[:, -4:] = 0

        entities = basic_head.decode(scores, attention_mask=mask, threshold=0.0)

        # Should not find entity in padding region
        for ent in entities[0]:
            assert ent["start"] < seq_len - 4
            assert ent["end"] < seq_len - 4

    def test_decode_multiple_entities(self, basic_head, num_labels, batch_size, seq_len, device):
        """Decode finds multiple entities."""
        scores = torch.full((batch_size, num_labels, seq_len, seq_len), -10.0, device=device)
        # Add multiple entities
        scores[0, 0, 2, 4] = 5.0   # PER
        scores[0, 1, 10, 12] = 4.0  # ORG
        scores[0, 2, 20, 20] = 3.0  # LOC (single token)

        id2label = {0: "PER", 1: "ORG", 2: "LOC", 3: "MISC"}
        entities = basic_head.decode(scores, threshold=0.0, id2label=id2label)

        assert len(entities[0]) >= 3
        labels_found = {e["label"] for e in entities[0]}
        assert "PER" in labels_found
        assert "ORG" in labels_found
        assert "LOC" in labels_found


# =============================================================================
# Test RoPE Integration
# =============================================================================


class TestGlobalPointerNERHeadRoPE:
    """Tests for Rotary Position Encoding."""

    def test_rope_applied(self, hidden_size, num_labels, hidden_states, device):
        """RoPE modifies Q/K differently than no RoPE."""
        head_with_rope = GlobalPointerNERHead(
            hidden_size=hidden_size,
            num_labels=num_labels,
            use_rope=True,
            dropout=0.0,
        ).to(device)
        head_without_rope = GlobalPointerNERHead(
            hidden_size=hidden_size,
            num_labels=num_labels,
            use_rope=False,
            dropout=0.0,
        ).to(device)

        # Copy weights for fair comparison
        head_without_rope.q_proj.weight.data = head_with_rope.q_proj.weight.data.clone()
        head_without_rope.q_proj.bias.data = head_with_rope.q_proj.bias.data.clone()
        head_without_rope.k_proj.weight.data = head_with_rope.k_proj.weight.data.clone()
        head_without_rope.k_proj.bias.data = head_with_rope.k_proj.bias.data.clone()

        out_with_rope = head_with_rope(hidden_states)["logits"]
        out_without_rope = head_without_rope(hidden_states)["logits"]

        # Outputs should be different due to RoPE
        assert not torch.allclose(out_with_rope, out_without_rope)

    def test_rope_position_sensitivity(self, hidden_size, hidden_states, device):
        """RoPE should make model position-sensitive."""
        head = GlobalPointerNERHead(hidden_size=hidden_size, use_rope=True, dropout=0.0).to(device)

        # Get output for original input
        out1 = head(hidden_states)["logits"]

        # Shuffle positions and get output
        shuffled = hidden_states[:, torch.randperm(hidden_states.shape[1]), :]
        out2 = head(shuffled)["logits"]

        # Outputs should be different
        assert not torch.allclose(out1, out2)


# =============================================================================
# Test Factory Function
# =============================================================================


class TestCreateGlobalPointerHead:
    """Tests for factory function."""

    def test_factory_ner_general(self, hidden_size):
        """Factory creates correct head for ner_general."""
        head = create_globalpointer_head("ner_general", hidden_size=hidden_size)

        assert isinstance(head, GlobalPointerNERHead)
        assert head.num_labels == 4  # PER, ORG, LOC, MISC

    def test_factory_ner_family(self, hidden_size):
        """Factory creates correct head for ner_family."""
        head = create_globalpointer_head("ner_family", hidden_size=hidden_size)

        assert isinstance(head, GlobalPointerNERHead)
        assert head.num_labels == 10  # KINSHIP, NICKNAME, etc.

    def test_factory_temporal(self, hidden_size):
        """Factory creates correct head for temporal."""
        head = create_globalpointer_head("temporal", hidden_size=hidden_size)

        assert isinstance(head, GlobalPointerNERHead)
        assert head.num_labels == 5  # DATE_ABS, DATE_REL, etc.

    def test_factory_unknown_capability(self, hidden_size):
        """Factory raises error for unknown capability."""
        with pytest.raises(ValueError, match="Unknown capability"):
            create_globalpointer_head("unknown", hidden_size=hidden_size)

    def test_factory_custom_head_size(self, hidden_size):
        """Factory respects head_size parameter."""
        head = create_globalpointer_head("ner_general", hidden_size=hidden_size, head_size=32)
        assert head.head_size == 32


# =============================================================================
# Test Backward Pass
# =============================================================================


class TestGlobalPointerNERHeadBackward:
    """Tests for gradient flow."""

    def test_backward_pass(self, basic_head, hidden_states, num_labels, batch_size, seq_len, device):
        """Gradients flow correctly."""
        hidden_states.requires_grad = True

        span_labels = torch.zeros(batch_size, num_labels, seq_len, seq_len, device=device)
        span_labels[0, 0, 2, 5] = 1

        output = basic_head(hidden_states, span_labels=span_labels)
        output["loss"].backward()

        assert hidden_states.grad is not None
        assert not torch.isnan(hidden_states.grad).any()

    def test_gradients_flow_to_projections(self, basic_head, hidden_states, num_labels, batch_size, seq_len, device):
        """Gradients flow to Q/K projections."""
        span_labels = torch.zeros(batch_size, num_labels, seq_len, seq_len, device=device)
        span_labels[0, 0, 2, 5] = 1

        output = basic_head(hidden_states, span_labels=span_labels)
        output["loss"].backward()

        assert basic_head.q_proj.weight.grad is not None
        assert basic_head.k_proj.weight.grad is not None
        assert not torch.isnan(basic_head.q_proj.weight.grad).any()
        assert not torch.isnan(basic_head.k_proj.weight.grad).any()


# =============================================================================
# Test Freeze/Unfreeze
# =============================================================================


class TestGlobalPointerNERHeadFreeze:
    """Tests for freeze/unfreeze functionality."""

    def test_freeze(self, basic_head):
        """Freeze disables gradients."""
        basic_head.freeze()

        for param in basic_head.parameters():
            assert not param.requires_grad

    def test_unfreeze(self, basic_head):
        """Unfreeze enables gradients."""
        basic_head.freeze()
        basic_head.unfreeze()

        for param in basic_head.parameters():
            assert param.requires_grad


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestGlobalPointerNERHeadEdgeCases:
    """Tests for edge cases."""

    def test_single_token_sequence(self, hidden_size, num_labels, device):
        """Handle single-token sequence."""
        head = GlobalPointerNERHead(hidden_size=hidden_size, num_labels=num_labels).to(device)
        hidden = torch.randn(1, 1, hidden_size, device=device)

        output = head(hidden)

        assert output["logits"].shape == (1, num_labels, 1, 1)
        assert not torch.isnan(output["logits"]).any()

    def test_max_length_sequence(self, hidden_size, num_labels, device):
        """Handle max length sequence (512)."""
        head = GlobalPointerNERHead(hidden_size=hidden_size, num_labels=num_labels).to(device)
        hidden = torch.randn(1, 512, hidden_size, device=device)

        output = head(hidden)

        assert output["logits"].shape == (1, num_labels, 512, 512)

    def test_all_padding(self, basic_head, hidden_size, device):
        """Handle all-padding input."""
        hidden = torch.randn(1, 16, hidden_size, device=device)
        mask = torch.zeros(1, 16, device=device)  # All padding

        output = basic_head(hidden, attention_mask=mask)

        # All positions should be masked
        assert (output["logits"] < -1e10).all()

    def test_empty_span_labels(self, basic_head, hidden_states, num_labels, batch_size, seq_len, device):
        """Handle span labels with no entities."""
        span_labels = torch.zeros(batch_size, num_labels, seq_len, seq_len, device=device)

        output = basic_head(hidden_states, span_labels=span_labels)

        assert "loss" in output
        assert not torch.isnan(output["loss"])

    def test_extra_repr(self, basic_head):
        """extra_repr returns expected string."""
        repr_str = basic_head.extra_repr()

        assert "hidden_size=" in repr_str
        assert "num_labels=" in repr_str
        assert "head_size=" in repr_str
        assert "use_rope=" in repr_str


# =============================================================================
# Test Loss Type Selection
# =============================================================================


class TestGlobalPointerNERHeadLossType:
    """Tests for loss_type parameter."""

    @pytest.fixture
    def hidden_size(self):
        return 256

    @pytest.fixture
    def num_labels(self):
        return 4

    @pytest.fixture
    def batch_size(self):
        return 2

    @pytest.fixture
    def seq_len(self):
        return 32

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_default_loss_type_is_globalpointer(self, hidden_size, device):
        """Default loss type is 'globalpointer'."""
        head = GlobalPointerNERHead(hidden_size=hidden_size).to(device)

        assert head.loss_type == "globalpointer"
        assert head.loss_fn is not None
        assert head.loss_fn.__class__.__name__ == "GlobalPointerLoss"

    def test_loss_type_globalpointer_explicit(self, hidden_size, device):
        """Explicit globalpointer loss type works."""
        head = GlobalPointerNERHead(
            hidden_size=hidden_size,
            loss_type="globalpointer",
        ).to(device)

        assert head.loss_type == "globalpointer"
        assert head.loss_fn is not None

    def test_loss_type_focal_globalpointer(self, hidden_size, device):
        """Focal globalpointer loss type works."""
        head = GlobalPointerNERHead(
            hidden_size=hidden_size,
            loss_type="focal_globalpointer",
        ).to(device)

        assert head.loss_type == "focal_globalpointer"
        assert head.loss_fn is not None
        assert head.loss_fn.__class__.__name__ == "FocalGlobalPointerLoss"

    def test_loss_type_bce_fallback(self, hidden_size, device):
        """BCE fallback works when loss_type is 'bce'."""
        head = GlobalPointerNERHead(
            hidden_size=hidden_size,
            loss_type="bce",
        ).to(device)

        assert head.loss_type == "bce"
        assert head.loss_fn is None  # BCE is computed inline

    def test_loss_computation_with_globalpointer_loss(
        self, hidden_size, num_labels, batch_size, seq_len, device
    ):
        """Loss computation works with GlobalPointerLoss."""
        head = GlobalPointerNERHead(
            hidden_size=hidden_size,
            num_labels=num_labels,
            loss_type="globalpointer",
        ).to(device)

        hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device)
        span_labels = torch.zeros(batch_size, num_labels, seq_len, seq_len, device=device)
        span_labels[0, 0, 2, 5] = 1  # Entity of type 0 from token 2 to 5

        output = head(hidden_states, span_labels=span_labels)

        assert "loss" in output
        assert output["loss"].dim() == 0
        assert not torch.isnan(output["loss"])
        assert output["loss"].requires_grad

    def test_loss_computation_with_bce_fallback(
        self, hidden_size, num_labels, batch_size, seq_len, device
    ):
        """Loss computation works with BCE fallback."""
        head = GlobalPointerNERHead(
            hidden_size=hidden_size,
            num_labels=num_labels,
            loss_type="bce",
        ).to(device)

        hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device)
        span_labels = torch.zeros(batch_size, num_labels, seq_len, seq_len, device=device)
        span_labels[0, 0, 2, 5] = 1

        output = head(hidden_states, span_labels=span_labels)

        assert "loss" in output
        assert output["loss"].dim() == 0
        assert not torch.isnan(output["loss"])

    def test_globalpointer_loss_vs_bce_different(
        self, hidden_size, num_labels, batch_size, seq_len, device
    ):
        """GlobalPointerLoss and BCE give different loss values."""
        torch.manual_seed(42)

        head_gp = GlobalPointerNERHead(
            hidden_size=hidden_size,
            num_labels=num_labels,
            loss_type="globalpointer",
            dropout=0.0,
        ).to(device)
        head_bce = GlobalPointerNERHead(
            hidden_size=hidden_size,
            num_labels=num_labels,
            loss_type="bce",
            dropout=0.0,
        ).to(device)

        # Copy weights
        head_bce.q_proj.weight.data = head_gp.q_proj.weight.data.clone()
        head_bce.q_proj.bias.data = head_gp.q_proj.bias.data.clone()
        head_bce.k_proj.weight.data = head_gp.k_proj.weight.data.clone()
        head_bce.k_proj.bias.data = head_gp.k_proj.bias.data.clone()

        hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device)
        span_labels = torch.zeros(batch_size, num_labels, seq_len, seq_len, device=device)
        span_labels[0, 0, 2, 5] = 1
        span_labels[0, 1, 10, 15] = 1

        head_gp.eval()
        head_bce.eval()

        loss_gp = head_gp(hidden_states, span_labels=span_labels)["loss"]
        loss_bce = head_bce(hidden_states, span_labels=span_labels)["loss"]

        # Different loss functions should give different values
        assert not torch.allclose(loss_gp, loss_bce, rtol=1e-2)


# =============================================================================
# Test NMS Functions (Issue 2.3.2)
# =============================================================================


class TestGlobalPointerNMSFunctions:
    """Tests for NMS helper functions."""

    @pytest.fixture
    def hidden_size(self):
        return 256

    @pytest.fixture
    def basic_head(self, hidden_size):
        return GlobalPointerNERHead(hidden_size=hidden_size)

    def test_spans_overlap_true(self, basic_head):
        """Overlapping spans correctly detected."""
        a = {"start": 0, "end": 5, "label": "PER", "score": 0.9}
        b = {"start": 3, "end": 8, "label": "PER", "score": 0.7}

        assert basic_head._spans_overlap(a, b) is True
        assert basic_head._spans_overlap(b, a) is True

    def test_spans_overlap_false(self, basic_head):
        """Non-overlapping spans correctly detected."""
        a = {"start": 0, "end": 5, "label": "PER", "score": 0.9}
        b = {"start": 6, "end": 10, "label": "PER", "score": 0.7}

        assert basic_head._spans_overlap(a, b) is False
        assert basic_head._spans_overlap(b, a) is False

    def test_spans_overlap_adjacent(self, basic_head):
        """Adjacent spans (touching at boundary) are NOT overlapping."""
        # In our token-based spans, end is INCLUSIVE
        # So span (0,5) covers tokens 0,1,2,3,4,5 and span (6,10) covers tokens 6,7,8,9,10
        # These are adjacent but not overlapping
        a = {"start": 0, "end": 5, "label": "PER", "score": 0.9}
        b = {"start": 6, "end": 10, "label": "PER", "score": 0.7}

        assert basic_head._spans_overlap(a, b) is False

        # But if they share the same token, they overlap
        # (0,5) and (5,10) share token 5
        c = {"start": 5, "end": 10, "label": "PER", "score": 0.7}
        assert basic_head._spans_overlap(a, c) is True

    def test_spans_overlap_contained(self, basic_head):
        """Contained spans are overlapping."""
        outer = {"start": 0, "end": 10, "label": "PER", "score": 0.9}
        inner = {"start": 2, "end": 5, "label": "PER", "score": 0.7}

        assert basic_head._spans_overlap(outer, inner) is True
        assert basic_head._spans_overlap(inner, outer) is True

    def test_calculate_iou_full_overlap(self, basic_head):
        """IoU of identical spans is 1.0."""
        a = {"start": 0, "end": 5, "label": "PER", "score": 0.9}
        b = {"start": 0, "end": 5, "label": "PER", "score": 0.7}

        assert basic_head._calculate_iou(a, b) == 1.0

    def test_calculate_iou_partial_overlap(self, basic_head):
        """IoU of partial overlap calculated correctly."""
        # Span a: tokens 0-5 (6 tokens), Span b: tokens 3-8 (6 tokens)
        # Intersection: tokens 3-5 (3 tokens)
        # Union: 6 + 6 - 3 = 9 tokens
        # IoU = 3/9 = 0.333...
        a = {"start": 0, "end": 5, "label": "PER", "score": 0.9}
        b = {"start": 3, "end": 8, "label": "PER", "score": 0.7}

        iou = basic_head._calculate_iou(a, b)
        assert abs(iou - 3/9) < 0.01

    def test_calculate_iou_no_overlap(self, basic_head):
        """IoU of non-overlapping spans is 0.0."""
        a = {"start": 0, "end": 5, "label": "PER", "score": 0.9}
        b = {"start": 10, "end": 15, "label": "PER", "score": 0.7}

        assert basic_head._calculate_iou(a, b) == 0.0

    def test_nms_empty_input(self, basic_head):
        """NMS handles empty input."""
        result = basic_head.nms_spans([])
        assert result == []

    def test_nms_single_entity(self, basic_head):
        """NMS keeps single entity."""
        entities = [{"start": 0, "end": 5, "label": "PER", "score": 0.9}]
        result = basic_head.nms_spans(entities)

        assert len(result) == 1
        assert result[0]["start"] == 0

    def test_nms_no_overlap(self, basic_head):
        """NMS keeps all non-overlapping entities."""
        entities = [
            {"start": 0, "end": 3, "label": "PER", "score": 0.9},
            {"start": 10, "end": 15, "label": "PER", "score": 0.7},
            {"start": 20, "end": 25, "label": "ORG", "score": 0.8},
        ]
        result = basic_head.nms_spans(entities)

        assert len(result) == 3

    def test_nms_same_type_overlap_higher_score_wins(self, basic_head):
        """NMS suppresses lower-score overlapping same-type spans."""
        entities = [
            {"start": 0, "end": 5, "label": "PER", "score": 0.9},
            {"start": 3, "end": 8, "label": "PER", "score": 0.7},
        ]
        result = basic_head.nms_spans(entities)

        assert len(result) == 1
        assert result[0]["score"] == 0.9

    def test_nms_different_type_overlap_both_kept(self, basic_head):
        """NMS keeps overlapping entities of different types by default."""
        entities = [
            {"start": 0, "end": 5, "label": "PER", "score": 0.9},
            {"start": 0, "end": 5, "label": "ORG", "score": 0.7},
        ]
        result = basic_head.nms_spans(entities, cross_type=False)

        assert len(result) == 2

    def test_nms_cross_type_enabled(self, basic_head):
        """NMS suppresses across types when cross_type=True."""
        entities = [
            {"start": 0, "end": 5, "label": "PER", "score": 0.9},
            {"start": 0, "end": 5, "label": "ORG", "score": 0.7},
        ]
        result = basic_head.nms_spans(entities, cross_type=True)

        assert len(result) == 1
        assert result[0]["label"] == "PER"  # Higher score

    def test_nms_iou_threshold(self, basic_head):
        """NMS respects IoU threshold."""
        # Small overlap should be allowed with high threshold
        entities = [
            {"start": 0, "end": 5, "label": "PER", "score": 0.9},
            {"start": 4, "end": 10, "label": "PER", "score": 0.7},
        ]

        # With threshold 0.0, suppress any overlap
        result_strict = basic_head.nms_spans(entities, iou_threshold=0.0)
        assert len(result_strict) == 1

        # With high threshold, allow partial overlap
        result_relaxed = basic_head.nms_spans(entities, iou_threshold=0.8)
        assert len(result_relaxed) == 2

    def test_nms_three_way_overlap(self, basic_head):
        """NMS handles three overlapping entities correctly."""
        entities = [
            {"start": 0, "end": 5, "label": "PER", "score": 0.7},
            {"start": 3, "end": 8, "label": "PER", "score": 0.9},  # Highest
            {"start": 6, "end": 12, "label": "PER", "score": 0.5},
        ]
        result = basic_head.nms_spans(entities)

        # Keep highest score (3-8), suppress others that overlap
        assert len(result) == 1
        assert result[0]["start"] == 3
        assert result[0]["score"] == 0.9


# =============================================================================
# Test Token-to-Char Mapping (Issue 2.3.4)
# =============================================================================


class TestTokenToCharMapping:
    """Tests for token to character span conversion."""

    @pytest.fixture
    def hidden_size(self):
        return 256

    @pytest.fixture
    def basic_head(self, hidden_size):
        return GlobalPointerNERHead(hidden_size=hidden_size)

    def test_token_to_char_single_token(self, basic_head):
        """Single token span maps correctly."""
        # "Emma" at char 0-4
        offset_mapping = [(0, 0), (0, 4), (5, 10), (0, 0)]  # CLS, Emma, lives, SEP

        char_start, char_end = basic_head._token_to_char_span(offset_mapping, 1, 1)

        assert char_start == 0
        assert char_end == 4

    def test_token_to_char_multi_token(self, basic_head):
        """Multi-token span maps correctly."""
        # "New York" at tokens 1-2
        offset_mapping = [(0, 0), (0, 3), (4, 8), (9, 14), (0, 0)]

        char_start, char_end = basic_head._token_to_char_span(offset_mapping, 1, 2)

        assert char_start == 0
        assert char_end == 8


# =============================================================================
# Test decode_with_nms (Issue 2.3.5)
# =============================================================================


class TestDecodeWithNMS:
    """Tests for the full decode_with_nms pipeline."""

    @pytest.fixture
    def hidden_size(self):
        return 256

    @pytest.fixture
    def num_labels(self):
        return 4

    @pytest.fixture
    def seq_len(self):
        return 32

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def basic_head(self, hidden_size, num_labels, device):
        return GlobalPointerNERHead(
            hidden_size=hidden_size,
            num_labels=num_labels,
            dropout=0.0,
        ).to(device)

    def test_decode_with_nms_basic(self, basic_head, num_labels, seq_len, device):
        """Basic decode_with_nms works."""
        scores = torch.full((1, num_labels, seq_len, seq_len), -10.0, device=device)
        scores[0, 0, 2, 5] = 5.0  # One entity

        id2label = {0: "PER", 1: "ORG", 2: "LOC", 3: "MISC"}

        entities = basic_head.decode_with_nms(
            scores, threshold=0.0, id2label=id2label
        )

        assert len(entities) == 1
        assert len(entities[0]) >= 1
        assert entities[0][0]["label"] == "PER"

    def test_decode_with_nms_returns_confidence(self, basic_head, num_labels, seq_len, device):
        """decode_with_nms includes confidence when requested."""
        scores = torch.full((1, num_labels, seq_len, seq_len), -10.0, device=device)
        scores[0, 0, 2, 5] = 2.0  # sigmoid(2.0) ~ 0.88

        entities = basic_head.decode_with_nms(
            scores, threshold=0.0, return_probabilities=True
        )

        assert "confidence" in entities[0][0]
        assert 0.8 < entities[0][0]["confidence"] < 0.95

    def test_decode_with_nms_temperature(self, basic_head, num_labels, seq_len, device):
        """Temperature affects confidence values."""
        scores = torch.full((1, num_labels, seq_len, seq_len), -10.0, device=device)
        scores[0, 0, 2, 5] = 2.0

        entities_t1 = basic_head.decode_with_nms(
            scores, threshold=0.0, return_probabilities=True, temperature=1.0
        )
        entities_t2 = basic_head.decode_with_nms(
            scores, threshold=0.0, return_probabilities=True, temperature=2.0
        )

        # Higher temperature -> lower confidence
        assert entities_t1[0][0]["confidence"] > entities_t2[0][0]["confidence"]

    def test_decode_with_nms_applies_nms(self, basic_head, num_labels, seq_len, device):
        """decode_with_nms actually applies NMS."""
        scores = torch.full((1, num_labels, seq_len, seq_len), -10.0, device=device)
        # Two overlapping entities
        scores[0, 0, 2, 5] = 5.0   # Higher score
        scores[0, 0, 3, 6] = 3.0   # Lower score, overlaps

        entities = basic_head.decode_with_nms(
            scores, threshold=0.0, nms_threshold=0.0
        )

        # Only one should survive NMS
        assert len(entities[0]) == 1
        assert entities[0][0]["score"] == 5.0

    def test_decode_with_nms_char_spans(self, basic_head, num_labels, seq_len, device):
        """decode_with_nms returns char spans when offset_mapping provided."""
        scores = torch.full((1, num_labels, seq_len, seq_len), -10.0, device=device)
        scores[0, 0, 1, 2] = 5.0  # Tokens 1-2

        # Mock offset_mapping: "New York lives..."
        offset_mapping = [[(0, 0)] + [(i*4, i*4+3) for i in range(seq_len-1)]]

        entities = basic_head.decode_with_nms(
            scores, threshold=0.0, offset_mapping=offset_mapping
        )

        assert "char_start" in entities[0][0]
        assert "char_end" in entities[0][0]
        assert "token_start" in entities[0][0]
        assert "token_end" in entities[0][0]

    def test_decode_with_nms_batch(self, basic_head, num_labels, seq_len, device):
        """decode_with_nms handles batch correctly."""
        batch_size = 3
        scores = torch.full((batch_size, num_labels, seq_len, seq_len), -10.0, device=device)
        scores[0, 0, 2, 5] = 5.0
        scores[1, 1, 10, 12] = 4.0
        scores[2, 2, 1, 3] = 3.0

        entities = basic_head.decode_with_nms(scores, threshold=0.0)

        assert len(entities) == batch_size
        assert len(entities[0]) >= 1
        assert len(entities[1]) >= 1
        assert len(entities[2]) >= 1

    def test_decode_with_nms_no_entities(self, basic_head, num_labels, seq_len, device):
        """decode_with_nms handles no entities gracefully."""
        scores = torch.full((1, num_labels, seq_len, seq_len), -10.0, device=device)
        # No high scores -> no entities

        entities = basic_head.decode_with_nms(scores, threshold=0.0)

        assert len(entities) == 1
        assert entities[0] == []
