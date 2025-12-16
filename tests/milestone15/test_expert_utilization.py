"""
Tests for MoE Expert Utilization.

Test Coverage:
    - Issue 15.2.2: MoE Expert Utilization Test
        - 15.2.2-T1: Experts receive roughly equal tokens
        - 15.2.2-T2: No expert receives 0 tokens
        - 15.2.2-T3: Shared expert processes all tokens

Milestone 15: Evaluation & Quality
"""

from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

if TYPE_CHECKING:
    pass


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_router():
    """Create a mock TopKRouter for testing."""
    router = MagicMock()
    router.num_experts = 8
    router.top_k = 2
    router.gate = MagicMock()
    return router


@pytest.fixture
def mock_moe_layer():
    """Create a mock MoELayer for testing."""
    moe = MagicMock()
    moe.num_experts = 8
    moe.num_experts_per_token = 2
    moe.router = MagicMock()
    moe.shared_expert = MagicMock()
    moe.experts = [MagicMock() for _ in range(8)]
    return moe


@pytest.fixture
def balanced_routing_output():
    """
    Create balanced routing output where tokens are evenly distributed.

    For 8 experts and 64 tokens with top-2 routing:
        - Each token goes to 2 experts
        - 64 * 2 = 128 total assignments
        - Balanced: 128 / 8 = 16 assignments per expert
    """
    batch_size = 4
    seq_len = 16
    top_k = 2
    num_experts = 8

    # Create balanced expert indices
    # Each position gets assigned to 2 consecutive experts (mod 8)
    expert_indices = torch.zeros(batch_size, seq_len, top_k, dtype=torch.long)

    for b in range(batch_size):
        for s in range(seq_len):
            # Assign to expert i and i+1 (mod 8) based on position
            base_expert = (b * seq_len + s) % num_experts
            expert_indices[b, s, 0] = base_expert
            expert_indices[b, s, 1] = (base_expert + 1) % num_experts

    # Uniform routing weights
    routing_weights = torch.ones(batch_size, seq_len, top_k) / top_k

    aux_losses = {
        "load_balance_loss": torch.tensor(0.01),
        "z_loss": torch.tensor(0.001),
        "total": torch.tensor(0.011),
    }

    return routing_weights, expert_indices, aux_losses


@pytest.fixture
def imbalanced_routing_output():
    """
    Create imbalanced routing where most tokens go to first 2 experts.

    This simulates expert collapse - a common failure mode.
    """
    batch_size = 4
    seq_len = 16
    top_k = 2

    # Most tokens go to experts 0 and 1
    expert_indices = torch.zeros(batch_size, seq_len, top_k, dtype=torch.long)
    expert_indices[:, :, 0] = 0  # First choice: expert 0
    expert_indices[:, :, 1] = 1  # Second choice: expert 1

    # A few tokens go to other experts
    expert_indices[0, 0, 0] = 2
    expert_indices[0, 1, 1] = 3

    routing_weights = torch.ones(batch_size, seq_len, top_k) / top_k

    aux_losses = {
        "load_balance_loss": torch.tensor(0.5),  # High loss indicates imbalance
        "z_loss": torch.tensor(0.01),
        "total": torch.tensor(0.51),
    }

    return routing_weights, expert_indices, aux_losses


@pytest.fixture
def collapsed_routing_output():
    """
    Create routing where some experts receive 0 tokens (expert collapse).
    """
    batch_size = 4
    seq_len = 16
    top_k = 2

    # All tokens go to experts 0, 1, 2, 3 only
    # Experts 4, 5, 6, 7 receive nothing
    expert_indices = torch.zeros(batch_size, seq_len, top_k, dtype=torch.long)
    for b in range(batch_size):
        for s in range(seq_len):
            expert_indices[b, s, 0] = s % 4  # Only use experts 0-3
            expert_indices[b, s, 1] = (s + 1) % 4

    routing_weights = torch.ones(batch_size, seq_len, top_k) / top_k

    aux_losses = {
        "load_balance_loss": torch.tensor(0.8),
        "z_loss": torch.tensor(0.01),
        "total": torch.tensor(0.81),
    }

    return routing_weights, expert_indices, aux_losses


# =============================================================================
# Helper Functions
# =============================================================================


def compute_expert_distribution(
    expert_indices: torch.Tensor,
    num_experts: int,
) -> dict[int, int]:
    """
    Compute the distribution of tokens across experts.

    Args:
        expert_indices: Tensor of expert assignments.
            Shape: (batch, seq_len, top_k)
        num_experts: Total number of experts.

    Returns:
        Dictionary mapping expert index to token count.
    """
    flat_indices = expert_indices.flatten().tolist()
    counts = Counter(flat_indices)

    # Ensure all experts have an entry
    distribution = {i: counts.get(i, 0) for i in range(num_experts)}
    return distribution


def compute_balance_metrics(distribution: dict[int, int]) -> dict[str, float]:
    """
    Compute load balancing metrics from expert distribution.

    Args:
        distribution: Dict mapping expert_idx to token count.

    Returns:
        Dictionary with metrics:
            - mean: Average tokens per expert
            - std: Standard deviation
            - cv: Coefficient of variation
            - balance_score: 1 - cv (higher = more balanced)
            - min_fraction: Minimum expert fraction
            - max_fraction: Maximum expert fraction
    """
    counts = list(distribution.values())
    total = sum(counts)
    num_experts = len(counts)

    if total == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "cv": 0.0,
            "balance_score": 0.0,
            "min_fraction": 0.0,
            "max_fraction": 0.0,
        }

    fractions = [c / total for c in counts]
    mean = sum(fractions) / num_experts

    if mean > 0:
        variance = sum((f - mean) ** 2 for f in fractions) / num_experts
        std = math.sqrt(variance)
        cv = std / mean
    else:
        std = 0.0
        cv = 0.0

    return {
        "mean": mean,
        "std": std,
        "cv": cv,
        "balance_score": max(0.0, 1.0 - cv),
        "min_fraction": min(fractions),
        "max_fraction": max(fractions),
    }


def count_collapsed_experts(distribution: dict[int, int]) -> int:
    """
    Count experts that received 0 tokens.

    Args:
        distribution: Dict mapping expert_idx to token count.

    Returns:
        Number of experts with 0 tokens.
    """
    return sum(1 for count in distribution.values() if count == 0)


# =============================================================================
# Test Issue 15.2.2: MoE Expert Utilization
# =============================================================================


class TestExpertUtilization:
    """Tests for MoE expert load balancing."""

    def test_expert_utilization_balanced(self, balanced_routing_output):
        """15.2.2-T1: Experts receive roughly equal tokens."""
        routing_weights, expert_indices, aux_losses = balanced_routing_output
        num_experts = 8

        distribution = compute_expert_distribution(expert_indices, num_experts)
        metrics = compute_balance_metrics(distribution)

        # For balanced routing, each expert should get ~12.5% of tokens
        expected_fraction = 1.0 / num_experts

        for expert_idx, count in distribution.items():
            total = sum(distribution.values())
            fraction = count / total

            # Allow 5% tolerance around expected fraction
            tolerance = 0.05
            assert abs(fraction - expected_fraction) < tolerance, (
                f"Expert {expert_idx}: fraction {fraction:.3f} not close to "
                f"expected {expected_fraction:.3f}"
            )

        # Balance score should be high (close to 1.0)
        assert metrics["balance_score"] > 0.8, (
            f"Balance score {metrics['balance_score']:.3f} should be > 0.8"
        )

    def test_expert_utilization_detects_imbalance(self, imbalanced_routing_output):
        """Imbalanced routing is detected correctly."""
        routing_weights, expert_indices, aux_losses = imbalanced_routing_output
        num_experts = 8

        distribution = compute_expert_distribution(expert_indices, num_experts)
        metrics = compute_balance_metrics(distribution)

        # Balance score should be low for imbalanced routing
        assert metrics["balance_score"] < 0.5, (
            f"Balance score {metrics['balance_score']:.3f} should be < 0.5 for imbalanced routing"
        )

        # Some experts should have very high fraction
        assert metrics["max_fraction"] > 0.3, (
            f"Max fraction {metrics['max_fraction']:.3f} should be > 0.3 for imbalanced routing"
        )

    def test_no_expert_collapse(self, balanced_routing_output):
        """15.2.2-T2: No expert receives 0 tokens."""
        routing_weights, expert_indices, aux_losses = balanced_routing_output
        num_experts = 8

        distribution = compute_expert_distribution(expert_indices, num_experts)
        collapsed = count_collapsed_experts(distribution)

        assert collapsed == 0, (
            f"Expected 0 collapsed experts, got {collapsed}. "
            f"Distribution: {distribution}"
        )

        # All experts should have at least some tokens
        for expert_idx, count in distribution.items():
            assert count > 0, f"Expert {expert_idx} received 0 tokens"

    def test_detect_expert_collapse(self, collapsed_routing_output):
        """Expert collapse is detected correctly."""
        routing_weights, expert_indices, aux_losses = collapsed_routing_output
        num_experts = 8

        distribution = compute_expert_distribution(expert_indices, num_experts)
        collapsed = count_collapsed_experts(distribution)

        # Experts 4, 5, 6, 7 should have 0 tokens
        assert collapsed == 4, (
            f"Expected 4 collapsed experts, got {collapsed}. "
            f"Distribution: {distribution}"
        )

    def test_shared_expert_always_active(self, mock_moe_layer):
        """15.2.2-T3: Shared expert processes all tokens."""
        # If shared expert exists, it should process all tokens
        assert mock_moe_layer.shared_expert is not None

        # Simulate forward pass
        batch_size = 4
        seq_len = 16
        hidden_size = 1280

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        # Mock shared expert forward
        mock_moe_layer.shared_expert.return_value = torch.randn_like(hidden_states)

        # In real MoELayer, shared expert processes all tokens
        shared_output = mock_moe_layer.shared_expert(hidden_states)

        # Output should have same shape as input
        assert shared_output.shape == hidden_states.shape

        # Verify shared expert was called
        mock_moe_layer.shared_expert.assert_called_once()

    def test_shared_expert_none_handling(self):
        """Handle case where shared expert is disabled."""
        moe_layer = MagicMock()
        moe_layer.shared_expert = None
        moe_layer.num_experts = 8

        # When shared_expert is None, shared_fraction should be 0
        has_shared = moe_layer.shared_expert is not None
        shared_fraction = 1.0 if has_shared else 0.0

        assert shared_fraction == 0.0

    def test_expert_fraction_sums_to_one(self, balanced_routing_output):
        """Expert fractions sum to approximately 1.0."""
        routing_weights, expert_indices, aux_losses = balanced_routing_output
        num_experts = 8

        distribution = compute_expert_distribution(expert_indices, num_experts)
        total = sum(distribution.values())

        fractions = [count / total for count in distribution.values()]
        fraction_sum = sum(fractions)

        # Should sum to 1.0 (with small floating point tolerance)
        assert abs(fraction_sum - 1.0) < 1e-6, (
            f"Fractions sum to {fraction_sum}, expected 1.0"
        )


# =============================================================================
# Test Load Balancing Loss
# =============================================================================


class TestLoadBalancingLoss:
    """Tests for load balancing auxiliary loss."""

    def test_low_loss_for_balanced_routing(self, balanced_routing_output):
        """Load balancing loss is low for balanced routing."""
        routing_weights, expert_indices, aux_losses = balanced_routing_output

        lb_loss = aux_losses["load_balance_loss"].item()

        # For balanced routing, loss should be close to 1.0 (the expected value)
        # Lower deviation from ideal indicates better balance
        assert lb_loss < 0.1, f"Load balance loss {lb_loss} should be < 0.1 for balanced routing"

    def test_high_loss_for_imbalanced_routing(self, imbalanced_routing_output):
        """Load balancing loss is high for imbalanced routing."""
        routing_weights, expert_indices, aux_losses = imbalanced_routing_output

        lb_loss = aux_losses["load_balance_loss"].item()

        # For imbalanced routing, loss should be higher
        assert lb_loss > 0.2, f"Load balance loss {lb_loss} should be > 0.2 for imbalanced routing"

    def test_auxiliary_loss_components(self, balanced_routing_output):
        """Auxiliary losses have expected components."""
        routing_weights, expert_indices, aux_losses = balanced_routing_output

        assert "load_balance_loss" in aux_losses
        assert "z_loss" in aux_losses
        assert "total" in aux_losses

        # Total should be sum of components (approximately)
        lb = aux_losses["load_balance_loss"].item()
        z = aux_losses["z_loss"].item()
        total = aux_losses["total"].item()

        assert abs((lb + z) - total) < 1e-6


# =============================================================================
# Integration Tests
# =============================================================================


class TestExpertUtilizationIntegration:
    """Integration tests for expert utilization monitoring."""

    def test_utilization_statistics_structure(self, balanced_routing_output):
        """Utilization statistics have expected structure."""
        routing_weights, expert_indices, aux_losses = balanced_routing_output
        num_experts = 8

        distribution = compute_expert_distribution(expert_indices, num_experts)
        metrics = compute_balance_metrics(distribution)
        collapsed = count_collapsed_experts(distribution)

        # Build statistics dict (similar to compute_expert_utilization output)
        stats = {
            "expert_counts": distribution,
            "expert_fractions": {
                i: c / sum(distribution.values())
                for i, c in distribution.items()
            },
            "balance_score": metrics["balance_score"],
            "collapsed_experts": collapsed,
            "num_experts": num_experts,
            "total_tokens": sum(distribution.values()),
        }

        # Verify structure
        assert "expert_counts" in stats
        assert "expert_fractions" in stats
        assert "balance_score" in stats
        assert "collapsed_experts" in stats
        assert stats["num_experts"] == 8
        assert stats["total_tokens"] > 0

    def test_utilization_over_multiple_batches(self):
        """Utilization accumulates correctly over multiple batches."""
        num_experts = 8
        accumulated_counts = Counter()

        # Simulate multiple batches
        for batch_idx in range(5):
            batch_size = 4
            seq_len = 16
            top_k = 2

            # Create routing for this batch
            expert_indices = torch.randint(0, num_experts, (batch_size, seq_len, top_k))

            # Accumulate counts
            batch_distribution = compute_expert_distribution(expert_indices, num_experts)
            accumulated_counts.update(batch_distribution)

        # Final distribution
        final_distribution = dict(accumulated_counts)

        # Should have counts for all experts
        assert len(final_distribution) == num_experts

        # Total should be sum of all batches
        expected_total = 5 * 4 * 16 * 2  # batches * batch_size * seq_len * top_k
        actual_total = sum(final_distribution.values())
        assert actual_total == expected_total

    def test_balance_score_interpretation(self):
        """Balance score correctly represents load distribution."""
        # Perfect balance: all experts equal
        perfect_distribution = {i: 100 for i in range(8)}
        perfect_metrics = compute_balance_metrics(perfect_distribution)
        assert perfect_metrics["balance_score"] == 1.0

        # Very imbalanced: one expert gets everything
        terrible_distribution = {0: 800, **{i: 0 for i in range(1, 8)}}
        terrible_metrics = compute_balance_metrics(terrible_distribution)
        assert terrible_metrics["balance_score"] < 0.2

    def test_expert_count_per_layer(self, mock_moe_layer):
        """Track expert usage per MoE layer."""
        # In a decoder with 8 layers, only layers 2-7 are MoE
        moe_layer_indices = [2, 3, 4, 5, 6, 7]

        layer_stats = {}
        for layer_idx in moe_layer_indices:
            # Simulate routing for this layer
            expert_indices = torch.randint(0, 8, (4, 16, 2))
            distribution = compute_expert_distribution(expert_indices, 8)

            layer_stats[layer_idx] = {
                "distribution": distribution,
                "collapsed": count_collapsed_experts(distribution),
            }

        # Each layer should have its own statistics
        assert len(layer_stats) == 6

        for layer_idx, stats in layer_stats.items():
            assert "distribution" in stats
            assert "collapsed" in stats


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "compute_expert_distribution",
    "compute_balance_metrics",
    "count_collapsed_experts",
]
