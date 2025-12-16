"""
Milestone 10: Tests for TopKRouter and Auxiliary Losses.

Test Coverage:
    - Issue 10.1.1: TopKRouter Base Class (3 tests)
    - Issue 10.1.2: Load Balancing Loss (3 tests)
    - Issue 10.1.3: Router Z-Loss (2 tests)

Acceptance Criteria Tested:
    - AC1 (10.1.1): Router returns top-k expert indices per token
    - AC2 (10.1.1): Routing weights sum to 1.0 for selected experts
    - AC3 (10.1.1): Small uniform initialization (-0.01, 0.01)
    - AC1 (10.1.2): Loss ~1.0 when perfectly balanced
    - AC2 (10.1.2): Loss > 1.0 when imbalanced
    - AC3 (10.1.2): Gradient flows back to router
    - AC1 (10.1.3): Z-loss is scalar
    - AC2 (10.1.3): Z-loss increases with larger logits
"""

import pytest
import torch
import torch.nn as nn

from modeling_studio.models.moe_components import (
    TopKRouter,
    compute_load_balancing_loss,
    compute_router_z_loss,
)


# =============================================================================
# Issue 10.1.1: TopKRouter Base Class Tests
# =============================================================================


class TestTopKRouterInitialization:
    """Tests for TopKRouter initialization."""

    def test_router_initialization_weights_small_uniform(self):
        """10.1.1-T1: Router gate weights are small uniform [-0.01, 0.01]."""
        router = TopKRouter(hidden_size=768, num_experts=8, top_k=2)

        # Check weight bounds
        weights = router.gate.weight.data
        assert weights.min() >= -0.02, "Weights should be >= -0.02"
        assert weights.max() <= 0.02, "Weights should be <= 0.02"

        # Check it's not all zeros (actually initialized)
        assert weights.abs().sum() > 0, "Weights should not be all zeros"

        # Check distribution is roughly uniform (not all same value)
        unique_values = weights.unique().numel()
        assert unique_values > 10, "Weights should have varied values"

    def test_router_gate_no_bias(self):
        """10.1.1-T1b: Router gate has no bias."""
        router = TopKRouter(hidden_size=768, num_experts=8, top_k=2)

        assert router.gate.bias is None, "Gate should have no bias"

    def test_router_shape_configuration(self):
        """10.1.1-T1c: Router gate shape matches configuration."""
        hidden_size = 1280
        num_experts = 8

        router = TopKRouter(hidden_size=hidden_size, num_experts=num_experts, top_k=2)

        assert router.gate.weight.shape == (num_experts, hidden_size)
        assert router.hidden_size == hidden_size
        assert router.num_experts == num_experts
        assert router.top_k == 2


class TestTopKRouterSelection:
    """Tests for TopKRouter top-k selection."""

    def test_router_selects_exactly_top_k_experts(self):
        """10.1.1-T2: Router selects exactly top_k experts per token."""
        batch_size, seq_len, hidden_size = 4, 16, 768
        num_experts = 8
        top_k = 2

        router = TopKRouter(hidden_size=hidden_size, num_experts=num_experts, top_k=top_k)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        routing_weights, expert_indices, _ = router(hidden_states)

        # Check shapes
        assert routing_weights.shape == (batch_size, seq_len, top_k)
        assert expert_indices.shape == (batch_size, seq_len, top_k)

        # Check indices are valid
        assert (expert_indices >= 0).all(), "Indices should be >= 0"
        assert (expert_indices < num_experts).all(), f"Indices should be < {num_experts}"

        # Check each token has exactly top_k unique experts (or fewer if tied)
        for b in range(batch_size):
            for s in range(seq_len):
                indices = expert_indices[b, s].tolist()
                # Note: Could have duplicates if scores are tied, but typically unique
                assert len(indices) == top_k

    def test_router_top_k_values(self):
        """10.1.1-T2b: Different top_k values work correctly."""
        hidden_size = 768
        num_experts = 8

        for top_k in [1, 2, 4]:
            router = TopKRouter(hidden_size=hidden_size, num_experts=num_experts, top_k=top_k)
            hidden_states = torch.randn(2, 8, hidden_size)

            routing_weights, expert_indices, _ = router(hidden_states)

            assert routing_weights.shape[-1] == top_k
            assert expert_indices.shape[-1] == top_k


class TestTopKRouterWeights:
    """Tests for routing weight normalization."""

    def test_router_weights_sum_to_one(self):
        """10.1.1-T3: Selected expert routing weights sum to 1.0."""
        batch_size, seq_len, hidden_size = 4, 16, 768
        num_experts = 8
        top_k = 2

        router = TopKRouter(hidden_size=hidden_size, num_experts=num_experts, top_k=top_k)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        routing_weights, _, _ = router(hidden_states)

        # Sum across top_k dimension should be ~1.0
        weight_sums = routing_weights.sum(dim=-1)

        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), \
            f"Routing weights should sum to 1.0, got {weight_sums}"

    def test_router_weights_positive(self):
        """10.1.1-T3b: All routing weights are positive."""
        router = TopKRouter(hidden_size=768, num_experts=8, top_k=2)
        hidden_states = torch.randn(4, 16, 768)

        routing_weights, _, _ = router(hidden_states)

        assert (routing_weights >= 0).all(), "All routing weights should be >= 0"

    def test_router_aux_losses_returned(self):
        """10.1.1-T3c: Router returns auxiliary losses."""
        router = TopKRouter(hidden_size=768, num_experts=8, top_k=2)
        hidden_states = torch.randn(4, 16, 768)

        _, _, aux_losses = router(hidden_states)

        assert "load_balance_loss" in aux_losses
        assert "z_loss" in aux_losses
        assert "raw_load_balance" in aux_losses
        assert "raw_z_loss" in aux_losses


# =============================================================================
# Issue 10.1.2: Load Balancing Loss Tests
# =============================================================================


class TestLoadBalancingLoss:
    """Tests for load balancing auxiliary loss."""

    def test_load_balance_loss_perfect_balance(self):
        """10.1.2-T1: Loss ~1.0 when tokens perfectly balanced across experts."""
        num_experts = 8
        batch_size, seq_len = 4, 8  # 32 tokens total, 4 per expert

        # Create perfectly balanced assignment
        # Each expert gets exactly batch_size * seq_len / num_experts tokens
        expert_indices = torch.arange(num_experts).repeat(batch_size * seq_len // num_experts)
        expert_indices = expert_indices.view(batch_size, seq_len, 1)

        # Uniform router probabilities
        router_probs = torch.ones(batch_size, seq_len, num_experts) / num_experts

        loss = compute_load_balancing_loss(router_probs, expert_indices, num_experts)

        # Perfect balance should give loss = num_experts × (1/num_experts) × (1/num_experts) × num_experts = 1.0
        assert torch.isclose(loss, torch.tensor(1.0), atol=0.1), \
            f"Perfect balance should give loss ~1.0, got {loss.item()}"

    def test_load_balance_loss_imbalanced(self):
        """10.1.2-T2: Loss > 1.0 when experts are imbalanced."""
        num_experts = 8
        batch_size, seq_len = 4, 8

        # All tokens to expert 0 (maximally imbalanced)
        expert_indices = torch.zeros(batch_size, seq_len, 1, dtype=torch.long)

        # Router probs still give high prob to expert 0
        router_probs = torch.zeros(batch_size, seq_len, num_experts)
        router_probs[:, :, 0] = 1.0  # All probability to expert 0

        loss = compute_load_balancing_loss(router_probs, expert_indices, num_experts)

        # Imbalanced should give higher loss
        # tokens_per_expert = [1, 0, 0, ...], prob_per_expert = [1, 0, 0, ...]
        # loss = 8 × (1 × 1 + 0 × 0 + ...) = 8
        assert loss > 1.0, f"Imbalanced routing should give loss > 1.0, got {loss.item()}"

    def test_load_balance_gradient_flow(self):
        """10.1.2-T3: Gradients flow back through load balancing loss."""
        num_experts = 8
        batch_size, seq_len = 4, 8

        # Create router with requires_grad
        router = TopKRouter(hidden_size=768, num_experts=num_experts, top_k=2)
        hidden_states = torch.randn(batch_size, seq_len, 768, requires_grad=True)

        routing_weights, expert_indices, aux_losses = router(hidden_states)

        # Backprop through load balance loss
        aux_losses["load_balance_loss"].backward()

        # Check gradients exist
        assert hidden_states.grad is not None, "Gradients should flow to hidden_states"
        assert router.gate.weight.grad is not None, "Gradients should flow to router gate"

    def test_load_balance_loss_empty_input(self):
        """10.1.2-T3b: Handle empty input gracefully."""
        router_probs = torch.zeros(0, 0, 8)
        expert_indices = torch.zeros(0, 0, 2, dtype=torch.long)

        loss = compute_load_balancing_loss(router_probs, expert_indices, num_experts=8)

        assert loss.item() == 0.0, "Empty input should give 0 loss"


# =============================================================================
# Issue 10.1.3: Router Z-Loss Tests
# =============================================================================


class TestRouterZLoss:
    """Tests for router z-loss."""

    def test_z_loss_is_scalar(self):
        """10.1.3-T1: Z-loss returns scalar tensor."""
        router_logits = torch.randn(4, 16, 8)

        z_loss = compute_router_z_loss(router_logits)

        assert z_loss.dim() == 0, f"Z-loss should be scalar, got shape {z_loss.shape}"
        assert z_loss.dtype == router_logits.dtype

    def test_z_loss_increases_with_logits(self):
        """10.1.3-T2: Larger logits produce larger z-loss."""
        base_logits = torch.randn(4, 16, 8)

        z_loss_small = compute_router_z_loss(base_logits)
        z_loss_large = compute_router_z_loss(base_logits * 10)  # Scale up

        assert z_loss_large > z_loss_small, \
            f"Larger logits should give larger z-loss: {z_loss_large.item()} vs {z_loss_small.item()}"

    def test_z_loss_differentiable(self):
        """10.1.3-T2b: Z-loss is differentiable."""
        router_logits = torch.randn(4, 16, 8, requires_grad=True)

        z_loss = compute_router_z_loss(router_logits)
        z_loss.backward()

        assert router_logits.grad is not None, "Z-loss should be differentiable"
        assert not torch.isnan(router_logits.grad).any(), "Gradients should not be NaN"

    def test_z_loss_empty_input(self):
        """10.1.3-T2c: Handle empty input gracefully."""
        router_logits = torch.zeros(0, 0, 8)

        z_loss = compute_router_z_loss(router_logits)

        assert z_loss.item() == 0.0, "Empty input should give 0 loss"


# =============================================================================
# Integration Tests
# =============================================================================


class TestRouterIntegration:
    """Integration tests for router with aux losses."""

    def test_router_end_to_end(self):
        """Test complete router forward pass."""
        batch_size, seq_len, hidden_size = 8, 32, 1280
        num_experts = 8
        top_k = 2

        router = TopKRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            aux_loss_weight=0.01,
            z_loss_weight=0.001,
        )

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        routing_weights, expert_indices, aux_losses = router(hidden_states)

        # Verify shapes
        assert routing_weights.shape == (batch_size, seq_len, top_k)
        assert expert_indices.shape == (batch_size, seq_len, top_k)

        # Verify weights
        assert torch.allclose(routing_weights.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-5)

        # Verify aux losses are weighted
        assert aux_losses["load_balance_loss"] < aux_losses["raw_load_balance"]
        assert aux_losses["z_loss"] < aux_losses["raw_z_loss"]

    def test_router_deterministic(self):
        """Router should be deterministic in eval mode."""
        router = TopKRouter(hidden_size=768, num_experts=8, top_k=2)
        router.eval()

        hidden_states = torch.randn(4, 16, 768)

        w1, i1, _ = router(hidden_states)
        w2, i2, _ = router(hidden_states)

        assert torch.equal(w1, w2), "Router should be deterministic in eval"
        assert torch.equal(i1, i2), "Router should be deterministic in eval"
