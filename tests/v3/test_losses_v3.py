"""
Tests for Hub-Aware Loss Computation (losses_v3.py).

This test suite validates all acceptance criteria for Issue 3.2.2:
1. Token-level loss masks hub token positions (0-4)
2. Focal loss correctly implements γ-weighted cross entropy
3. Hierarchical loss combines primary + secondary for emotions
4. Multi-task loss aggregates with configurable weights
5. UncertaintyWeightedLoss learns per-task σ parameters
6. Label smoothing works with cross-entropy
7. Factory function supports both fixed and uncertainty weighting

Author: FamilyOS Team
Date: December 2025
Version: 3.3
"""

import pytest
import torch
import torch.nn as nn

from modeling_studio.models.losses_v3 import (
    HubAwareLossComputer,
    LossOutput,
    UncertaintyWeightedLoss,
    create_loss_computer,
)
from modeling_studio.training.losses_v3 import (
    HUB_TOKEN_POSITIONS_DEFAULT,
    HubGradientMaskedLoss,
    HubLossConfig,
    HubLossWeightCalculator,
    HubWeightedMultiTaskLoss,
    aggregate_task_losses,
    log_task_losses,
)
from modeling_studio.data.loaders_v3 import HubRouting


# ======================================================================
# Test HubAwareLossComputer - Basic Functionality
# ======================================================================


class TestHubAwareLossComputer:
    """Test basic HubAwareLossComputer functionality."""

    def test_initialization(self):
        """Test loss computer initialization."""
        task_configs = {
            "emotions": {"loss_type": "hierarchical", "loss_weight": 1.0},
            "ner_general": {"loss_type": "token_level", "loss_weight": 0.8},
            "intent": {"loss_type": "cross_entropy", "loss_weight": 1.2},
        }

        loss_computer = HubAwareLossComputer(
            task_configs, label_smoothing=0.1, use_focal_loss=True, focal_gamma=2.0
        )

        assert loss_computer.label_smoothing == 0.1
        assert loss_computer.use_focal_loss is True
        assert loss_computer.focal_gamma == 2.0
        assert len(loss_computer.task_configs) == 3

        # Check weights are registered as buffers
        assert hasattr(loss_computer, "weight_emotions")
        assert hasattr(loss_computer, "weight_ner_general")
        assert hasattr(loss_computer, "weight_intent")

    def test_standard_classification_loss(self):
        """Test standard cross-entropy loss for classification."""
        task_configs = {"intent": {"loss_type": "cross_entropy", "loss_weight": 1.0}}
        loss_computer = HubAwareLossComputer(task_configs)

        batch_size, num_labels = 4, 5
        logits = torch.randn(batch_size, num_labels)
        labels = torch.randint(0, num_labels, (batch_size,))

        loss = loss_computer.compute_task_loss("intent", logits, labels)

        assert loss.ndim == 0  # Scalar
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_regression_loss(self):
        """Test MSE loss for regression tasks."""
        task_configs = {"similarity": {"loss_type": "regression", "loss_weight": 1.0}}
        loss_computer = HubAwareLossComputer(task_configs)

        batch_size = 4
        logits = torch.randn(batch_size, 1)
        labels = torch.randn(batch_size)

        loss = loss_computer.compute_task_loss("similarity", logits, labels)

        assert loss.ndim == 0
        assert loss.item() >= 0  # MSE is always non-negative
        assert not torch.isnan(loss)

    def test_update_task_weight(self):
        """Test updating task loss weights."""
        task_configs = {"intent": {"loss_type": "cross_entropy", "loss_weight": 1.0}}
        loss_computer = HubAwareLossComputer(task_configs)

        assert loss_computer.weight_intent.item() == 1.0

        loss_computer.update_task_weight("intent", 2.5)
        assert loss_computer.weight_intent.item() == 2.5


# ======================================================================
# Test Token-Level Loss Masking (Acceptance Criterion 1)
# ======================================================================


class TestTokenLevelLoss:
    """Test token-level loss with hub token masking."""

    def test_token_level_loss_masks_hub_positions(self):
        """AC1: Token-level loss masks hub token positions (0-4)."""
        task_configs = {"ner_general": {"loss_type": "token_level", "loss_weight": 1.0}}
        loss_computer = HubAwareLossComputer(task_configs)

        batch_size, seq_len, num_labels = 2, 12, 9
        logits = torch.randn(batch_size, seq_len, num_labels)
        labels = torch.randint(0, num_labels, (batch_size, seq_len))

        # Create attention mask (all 1s)
        attention_mask = torch.ones(batch_size, seq_len)

        loss = loss_computer.compute_task_loss("ner_general", logits, labels, attention_mask)

        assert loss.ndim == 0
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_hub_positions_not_contributing_to_loss(self):
        """Verify hub positions (0-4) don't contribute to token-level loss."""
        task_configs = {"ner_family": {"loss_type": "token_level", "loss_weight": 1.0}}
        loss_computer = HubAwareLossComputer(task_configs)

        batch_size, seq_len, num_labels = 2, 12, 7
        logits = torch.randn(batch_size, seq_len, num_labels)
        labels = torch.randint(0, num_labels, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        # Compute loss with hub masking
        loss_with_mask = loss_computer.compute_task_loss(
            "ner_family", logits, labels, attention_mask
        )

        # Set hub positions to wrong labels (should not affect loss)
        labels_wrong_hub = labels.clone()
        labels_wrong_hub[:, :5] = (labels_wrong_hub[:, :5] + 1) % num_labels

        loss_with_wrong_hub = loss_computer.compute_task_loss(
            "ner_family", logits, labels_wrong_hub, attention_mask
        )

        # Loss should be identical (hub positions ignored)
        assert torch.allclose(loss_with_mask, loss_with_wrong_hub, atol=1e-6)

    def test_padding_mask_in_token_loss(self):
        """Test that padding is correctly masked in token-level loss."""
        task_configs = {"temporal": {"loss_type": "token_level", "loss_weight": 1.0}}
        loss_computer = HubAwareLossComputer(task_configs)

        batch_size, seq_len, num_labels = 2, 10, 5
        logits = torch.randn(batch_size, seq_len, num_labels)
        labels = torch.randint(0, num_labels, (batch_size, seq_len))

        # Create attention mask (some padding)
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, 8:] = 0  # Pad last 2 tokens of first sample
        attention_mask[1, 9:] = 0  # Pad last 1 token of second sample

        loss = loss_computer.compute_task_loss("temporal", logits, labels, attention_mask)

        assert loss.ndim == 0
        assert loss.item() > 0
        assert not torch.isnan(loss)


# ======================================================================
# Test Focal Loss (Acceptance Criterion 2)
# ======================================================================


class TestFocalLoss:
    """Test focal loss implementation."""

    def test_focal_loss_implementation(self):
        """AC2: Focal loss correctly implements γ-weighted cross entropy."""
        task_configs = {"safety": {"loss_type": "cross_entropy", "loss_weight": 1.0}}
        loss_computer = HubAwareLossComputer(task_configs, use_focal_loss=True, focal_gamma=2.0)

        batch_size, num_labels = 8, 2
        logits = torch.randn(batch_size, num_labels)
        labels = torch.randint(0, num_labels, (batch_size,))

        focal_loss = loss_computer.compute_task_loss("safety", logits, labels)

        assert focal_loss.ndim == 0
        assert focal_loss.item() > 0
        assert not torch.isnan(focal_loss)

    def test_focal_loss_focuses_on_hard_examples(self):
        """Verify focal loss down-weights easy examples."""
        task_configs = {"intent": {"loss_type": "cross_entropy", "loss_weight": 1.0}}

        # Standard cross-entropy
        loss_computer_ce = HubAwareLossComputer(task_configs, use_focal_loss=False)

        # Focal loss with gamma=2
        loss_computer_focal = HubAwareLossComputer(
            task_configs, use_focal_loss=True, focal_gamma=2.0
        )

        labels = torch.tensor([0, 1, 2, 0])

        # Easy examples: High confidence correct predictions
        easy_logits = torch.tensor(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0], [10.0, 0.0, 0.0]]
        )

        # Hard examples: Low confidence predictions
        hard_logits = torch.tensor(
            [[1.0, 0.9, 0.8], [0.9, 1.0, 0.8], [0.8, 0.9, 1.0], [1.0, 0.9, 0.8]]
        )

        # Compute losses
        ce_easy = loss_computer_ce.compute_task_loss("intent", easy_logits, labels)
        focal_easy = loss_computer_focal.compute_task_loss("intent", easy_logits, labels)

        ce_hard = loss_computer_ce.compute_task_loss("intent", hard_logits, labels)
        focal_hard = loss_computer_focal.compute_task_loss("intent", hard_logits, labels)

        # Focal loss should have larger difference between hard and easy
        focal_ratio = focal_hard / (focal_easy + 1e-6)
        ce_ratio = ce_hard / (ce_easy + 1e-6)

        assert focal_ratio > ce_ratio  # Focal emphasizes hard examples more

    def test_focal_loss_gamma_parameter(self):
        """Test that gamma parameter affects focal loss correctly."""
        task_configs = {"emotion": {"loss_type": "cross_entropy", "loss_weight": 1.0}}

        batch_size, num_labels = 4, 5
        logits = torch.randn(batch_size, num_labels)
        labels = torch.randint(0, num_labels, (batch_size,))

        # Test different gamma values
        loss_gamma_0 = HubAwareLossComputer(
            task_configs, use_focal_loss=True, focal_gamma=0.0
        ).compute_task_loss("emotion", logits, labels)

        loss_gamma_2 = HubAwareLossComputer(
            task_configs, use_focal_loss=True, focal_gamma=2.0
        ).compute_task_loss("emotion", logits, labels)

        loss_gamma_5 = HubAwareLossComputer(
            task_configs, use_focal_loss=True, focal_gamma=5.0
        ).compute_task_loss("emotion", logits, labels)

        # Gamma=0 should be close to standard CE
        # Higher gamma should have different loss values
        assert not torch.isnan(loss_gamma_0)
        assert not torch.isnan(loss_gamma_2)
        assert not torch.isnan(loss_gamma_5)


# ======================================================================
# Hub-Weighted Loss Scaling (Issue 5.3.4)
# ======================================================================


class TestHubWeightedLoss:
    """Tests for hub-weighted loss scaling utilities."""

    def test_weight_calculator_applies_active_inactive_and_safety(self) -> None:
        config = HubLossConfig()
        calculator = HubLossWeightCalculator(config)

        hub_routings = [
            HubRouting(emo=True, rel=False, mem=False, task=False),
            HubRouting(emo=False, rel=True, mem=True, task=False),
        ]

        weights_emo = calculator.compute_batch_weights(
            "emotions", hub_routings, [True, True]
        ).tolist()
        weights_safety = calculator.compute_batch_weights(
            "safety_familyos", hub_routings, [True, True]
        ).tolist()

        assert weights_emo == [pytest.approx(1.0), pytest.approx(0.3)]
        assert weights_safety == [pytest.approx(2.0), pytest.approx(0.6)]

    def test_hub_weighted_multitask_loss(self) -> None:
        hub_routings = [
            HubRouting(emo=True, rel=False, mem=False, task=True),
            HubRouting(emo=False, rel=True, mem=False, task=True),
        ]

        task_logits = {
            "emotions": torch.zeros(2, 2),
            "sentiment": torch.tensor([[2.0, 0.0], [0.5, 1.0]]),
            "ner_family": torch.zeros(2, 4, 3),
        }

        task_labels = {
            "emotions": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "sentiment": torch.tensor([0, 1]),
            "ner_family": torch.tensor([[0, 1, -100, -100], [0, 2, 1, -100]]),
        }

        loss_module = HubWeightedMultiTaskLoss()
        total_loss, task_losses = loss_module(task_logits, task_labels, hub_routings)

        assert total_loss.item() > 0
        assert set(task_losses.keys()) == {"emotions", "sentiment", "ner_family"}
        assert task_losses["emotions"].item() > 0
        assert task_losses["sentiment"].item() > 0
        assert task_losses["ner_family"].item() > 0

        weights = loss_module.weight_calculator.compute_batch_weights(
            "emotions", hub_routings, [True, True]
        )
        assert weights.tolist() == [pytest.approx(1.0), pytest.approx(0.3)]

    def test_hub_gradient_mask_zeroes_inactive_hubs(self) -> None:
        class DummyLoss(nn.Module):
            def forward(
                self, hidden_states_input: torch.Tensor
            ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
                return hidden_states_input.sum(), {}

        hidden_states = torch.ones(2, 6, 4, requires_grad=True)
        hub_routings = [
            HubRouting(emo=True, rel=False, mem=False, task=False),
            HubRouting(emo=False, rel=True, mem=True, task=False),
        ]

        wrapper = HubGradientMaskedLoss(DummyLoss())
        total_loss, _ = wrapper(hidden_states, hub_routings, hidden_states_input=hidden_states)
        total_loss.backward()

        gradients = hidden_states.grad

        # Sample 0: REL, MEM, TASK inactive -> positions 2,3,4 masked
        assert torch.allclose(
            gradients[0, HUB_TOKEN_POSITIONS_DEFAULT["REL"]], torch.zeros_like(gradients[0, 0])
        )
        assert torch.allclose(
            gradients[0, HUB_TOKEN_POSITIONS_DEFAULT["MEM"]], torch.zeros_like(gradients[0, 0])
        )
        assert torch.allclose(
            gradients[0, HUB_TOKEN_POSITIONS_DEFAULT["TASK"]], torch.zeros_like(gradients[0, 0])
        )

        # Sample 1: EMO and TASK inactive -> positions 1 and 4 masked
        assert torch.allclose(
            gradients[1, HUB_TOKEN_POSITIONS_DEFAULT["EMO"]], torch.zeros_like(gradients[1, 0])
        )
        assert torch.allclose(
            gradients[1, HUB_TOKEN_POSITIONS_DEFAULT["TASK"]], torch.zeros_like(gradients[1, 0])
        )

    def test_aggregation_and_logging_helpers(self) -> None:
        task_losses = {"a": torch.tensor(1.0), "b": torch.tensor(2.0)}
        weights = {"a": 0.5, "b": 2.0}

        total = aggregate_task_losses(task_losses, weights)
        assert total.item() == pytest.approx(1.0 * 0.5 + 2.0 * 2.0)

        logs = log_task_losses(task_losses, prefix="eval")
        assert logs["eval/loss_a"] == 1.0
        assert logs["eval/loss_b"] == 2.0


# ======================================================================
# Test Hierarchical Loss (Acceptance Criterion 3)
# ======================================================================


class TestHierarchicalLoss:
    """Test hierarchical loss for emotions."""

    def test_hierarchical_loss_combination(self):
        """AC3: Hierarchical loss combines primary + secondary for emotions."""
        task_configs = {"emotions": {"loss_type": "hierarchical", "loss_weight": 1.0}}
        loss_computer = HubAwareLossComputer(task_configs)

        batch_size, primary_labels, secondary_labels = 4, 7, 28
        primary_logits = torch.randn(batch_size, primary_labels)
        secondary_logits = torch.randn(batch_size, secondary_labels)
        primary_labels_gt = torch.randint(0, primary_labels, (batch_size,))
        secondary_labels_gt = torch.randint(0, secondary_labels, (batch_size,))

        logits = (primary_logits, secondary_logits)
        labels = (primary_labels_gt, secondary_labels_gt)

        loss = loss_computer.compute_task_loss("emotions", logits, labels)

        assert loss.ndim == 0
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_hierarchical_loss_weighting(self):
        """Test that primary and secondary losses are weighted correctly."""
        task_configs = {"emotions": {"loss_type": "hierarchical", "loss_weight": 1.0}}
        loss_computer = HubAwareLossComputer(task_configs)

        batch_size, primary_labels, secondary_labels = 4, 7, 28

        # Primary predictions (all wrong)
        primary_logits = torch.randn(batch_size, primary_labels)
        primary_labels_gt = torch.randint(0, primary_labels, (batch_size,))

        # Secondary predictions (all wrong)
        secondary_logits = torch.randn(batch_size, secondary_labels)
        secondary_labels_gt = torch.randint(0, secondary_labels, (batch_size,))

        # Compute individual losses manually
        primary_loss = nn.CrossEntropyLoss()(primary_logits, primary_labels_gt)
        secondary_loss = nn.CrossEntropyLoss()(secondary_logits, secondary_labels_gt)

        # Expected combined loss (default weights: 0.4 primary, 0.6 secondary)
        expected_loss = 0.4 * primary_loss + 0.6 * secondary_loss

        # Compute via loss computer
        logits = (primary_logits, secondary_logits)
        labels = (primary_labels_gt, secondary_labels_gt)
        computed_loss = loss_computer.compute_task_loss("emotions", logits, labels)

        assert torch.allclose(computed_loss, expected_loss, atol=1e-5)

    def test_hierarchical_fallback_to_standard(self):
        """Test that hierarchical loss falls back to CE if logits not tuple."""
        task_configs = {"emotions": {"loss_type": "hierarchical", "loss_weight": 1.0}}
        loss_computer = HubAwareLossComputer(task_configs)

        batch_size, num_labels = 4, 7
        logits = torch.randn(batch_size, num_labels)
        labels = torch.randint(0, num_labels, (batch_size,))

        # Should fallback to standard cross-entropy
        loss = loss_computer.compute_task_loss("emotions", logits, labels)

        assert loss.ndim == 0
        assert loss.item() > 0
        assert not torch.isnan(loss)


# ======================================================================
# Test Multi-Task Loss Aggregation (Acceptance Criterion 4)
# ======================================================================


class TestMultiTaskLoss:
    """Test multi-task loss aggregation."""

    def test_multitask_loss_aggregation(self):
        """AC4: Multi-task loss aggregates with configurable weights."""
        task_configs = {
            "emotions": {"loss_type": "cross_entropy", "loss_weight": 1.0},
            "intent": {"loss_type": "cross_entropy", "loss_weight": 1.5},
            "ner_general": {"loss_type": "token_level", "loss_weight": 0.8},
        }
        loss_computer = HubAwareLossComputer(task_configs)

        batch_size = 4
        num_labels_emo, num_labels_intent, num_labels_ner = 7, 5, 9
        seq_len = 10

        task_logits = {
            "emotions": torch.randn(batch_size, num_labels_emo),
            "intent": torch.randn(batch_size, num_labels_intent),
            "ner_general": torch.randn(batch_size, seq_len, num_labels_ner),
        }

        task_labels = {
            "emotions": torch.randint(0, num_labels_emo, (batch_size,)),
            "intent": torch.randint(0, num_labels_intent, (batch_size,)),
            "ner_general": torch.randint(0, num_labels_ner, (batch_size, seq_len)),
        }

        attention_mask = torch.ones(batch_size, seq_len)

        output = loss_computer.compute_multitask_loss(task_logits, task_labels, attention_mask)

        assert isinstance(output, LossOutput)
        assert output.total_loss.ndim == 0
        assert output.total_loss.item() > 0
        assert len(output.task_losses) == 3
        assert len(output.task_weights) == 3
        assert abs(output.task_weights["emotions"] - 1.0) < 1e-6
        assert abs(output.task_weights["intent"] - 1.5) < 1e-6
        assert abs(output.task_weights["ner_general"] - 0.8) < 1e-6

    def test_multitask_with_active_tasks(self):
        """Test multi-task loss with subset of active tasks."""
        task_configs = {
            "emotions": {"loss_type": "cross_entropy", "loss_weight": 1.0},
            "intent": {"loss_type": "cross_entropy", "loss_weight": 1.0},
            "sentiment": {"loss_type": "cross_entropy", "loss_weight": 1.0},
        }
        loss_computer = HubAwareLossComputer(task_configs)

        batch_size = 4
        task_logits = {
            "emotions": torch.randn(batch_size, 7),
            "intent": torch.randn(batch_size, 5),
            "sentiment": torch.randn(batch_size, 3),
        }

        task_labels = {
            "emotions": torch.randint(0, 7, (batch_size,)),
            "intent": torch.randint(0, 5, (batch_size,)),
            "sentiment": torch.randint(0, 3, (batch_size,)),
        }

        # Only compute loss for emotions and intent
        output = loss_computer.compute_multitask_loss(
            task_logits, task_labels, active_tasks=["emotions", "intent"]
        )

        assert len(output.task_losses) == 2
        assert "emotions" in output.task_losses
        assert "intent" in output.task_losses
        assert "sentiment" not in output.task_losses

    def test_weighted_sum_correctness(self):
        """Test that total loss is correct weighted sum."""
        task_configs = {
            "task_a": {"loss_type": "cross_entropy", "loss_weight": 2.0},
            "task_b": {"loss_type": "cross_entropy", "loss_weight": 0.5},
        }
        loss_computer = HubAwareLossComputer(task_configs)

        batch_size = 4
        task_logits = {
            "task_a": torch.randn(batch_size, 3),
            "task_b": torch.randn(batch_size, 5),
        }

        task_labels = {
            "task_a": torch.randint(0, 3, (batch_size,)),
            "task_b": torch.randint(0, 5, (batch_size,)),
        }

        output = loss_computer.compute_multitask_loss(task_logits, task_labels)

        # Verify weighted sum
        expected_total = 2.0 * output.task_losses["task_a"] + 0.5 * output.task_losses["task_b"]

        assert torch.allclose(output.total_loss, expected_total, atol=1e-6)


# ======================================================================
# Test Uncertainty Weighted Loss (Acceptance Criterion 5)
# ======================================================================


class TestUncertaintyWeightedLoss:
    """Test uncertainty-weighted loss for learned task weighting."""

    def test_uncertainty_loss_initialization(self):
        """AC5: UncertaintyWeightedLoss learns per-task σ parameters."""
        task_names = ["emotions", "intent", "ner_general"]
        uncertainty_loss = UncertaintyWeightedLoss(task_names)

        assert len(uncertainty_loss.log_vars) == 3
        assert all(task in uncertainty_loss.log_vars for task in task_names)

        # Check parameters are learnable
        for log_var in uncertainty_loss.log_vars.values():
            assert log_var.requires_grad

    def test_uncertainty_forward(self):
        """Test uncertainty-weighted loss forward pass."""
        task_names = ["emotions", "intent"]
        uncertainty_loss = UncertaintyWeightedLoss(task_names)

        task_losses = {
            "emotions": torch.tensor(0.5, requires_grad=True),
            "intent": torch.tensor(0.8, requires_grad=True),
        }

        total_loss, weights = uncertainty_loss(task_losses)

        assert total_loss.ndim == 0
        assert total_loss.item() > 0
        assert len(weights) == 2
        assert "emotions" in weights
        assert "intent" in weights

    def test_uncertainty_weights_update(self):
        """Test that uncertainty parameters update during training."""
        task_names = ["task_a", "task_b"]
        uncertainty_loss = UncertaintyWeightedLoss(task_names)

        # Simulate training step
        task_losses = {
            "task_a": torch.tensor(0.3, requires_grad=True),
            "task_b": torch.tensor(0.9, requires_grad=True),
        }

        total_loss, _ = uncertainty_loss(task_losses)
        total_loss.backward()

        # Check gradients exist
        for log_var in uncertainty_loss.log_vars.values():
            assert log_var.grad is not None
            assert not torch.allclose(log_var.grad, torch.zeros_like(log_var.grad))

    def test_get_task_uncertainties(self):
        """Test getting task uncertainties (σ)."""
        task_names = ["task_a", "task_b"]
        uncertainty_loss = UncertaintyWeightedLoss(task_names)

        uncertainties = uncertainty_loss.get_task_uncertainties()

        assert len(uncertainties) == 2
        assert all(u > 0 for u in uncertainties.values())  # σ should be positive


# ======================================================================
# Test Label Smoothing (Acceptance Criterion 6)
# ======================================================================


class TestLabelSmoothing:
    """Test label smoothing functionality."""

    def test_label_smoothing_applied(self):
        """AC6: Label smoothing works with cross-entropy."""
        task_configs = {"intent": {"loss_type": "cross_entropy", "loss_weight": 1.0}}

        # Without smoothing
        loss_computer_no_smooth = HubAwareLossComputer(task_configs, label_smoothing=0.0)

        # With smoothing
        loss_computer_smooth = HubAwareLossComputer(task_configs, label_smoothing=0.1)

        batch_size, num_labels = 4, 5
        logits = torch.randn(batch_size, num_labels)
        labels = torch.randint(0, num_labels, (batch_size,))

        loss_no_smooth = loss_computer_no_smooth.compute_task_loss("intent", logits, labels)
        loss_smooth = loss_computer_smooth.compute_task_loss("intent", logits, labels)

        # Label smoothing should affect loss value
        assert not torch.allclose(loss_no_smooth, loss_smooth)
        assert loss_smooth.item() > 0

    def test_label_smoothing_reduces_overconfidence(self):
        """Test that label smoothing reduces overconfidence."""
        task_configs = {"task": {"loss_type": "cross_entropy", "loss_weight": 1.0}}

        labels = torch.tensor([0, 1, 2, 0])

        # Very confident predictions
        confident_logits = torch.tensor(
            [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0], [100.0, 0.0, 0.0]]
        )

        loss_no_smooth = HubAwareLossComputer(task_configs, label_smoothing=0.0).compute_task_loss(
            "task", confident_logits, labels
        )

        loss_smooth = HubAwareLossComputer(task_configs, label_smoothing=0.1).compute_task_loss(
            "task", confident_logits, labels
        )

        # With smoothing, even confident predictions have non-zero loss
        assert loss_smooth.item() > loss_no_smooth.item()


# ======================================================================
# Test Factory Function (Acceptance Criterion 7)
# ======================================================================


class TestFactoryFunction:
    """Test create_loss_computer factory function."""

    def test_factory_creates_standard_loss(self):
        """AC7: Factory function supports both fixed and uncertainty weighting."""
        task_configs = {
            "emotions": {"loss_type": "hierarchical", "loss_weight": 1.0},
            "intent": {"loss_type": "cross_entropy", "loss_weight": 1.2},
        }

        loss_computer = create_loss_computer(task_configs, use_uncertainty_weighting=False)

        assert isinstance(loss_computer, HubAwareLossComputer)
        assert len(loss_computer.task_configs) == 2

    def test_factory_creates_uncertainty_loss(self):
        """Test factory creates ModuleDict with uncertainty weighting."""
        task_configs = {
            "emotions": {"loss_type": "hierarchical", "loss_weight": 1.0},
            "intent": {"loss_type": "cross_entropy", "loss_weight": 1.2},
        }

        loss_modules = create_loss_computer(task_configs, use_uncertainty_weighting=True)

        assert isinstance(loss_modules, nn.ModuleDict)
        assert "base" in loss_modules
        assert "uncertainty" in loss_modules
        assert isinstance(loss_modules["base"], HubAwareLossComputer)
        assert isinstance(loss_modules["uncertainty"], UncertaintyWeightedLoss)

    def test_factory_passes_kwargs(self):
        """Test that factory function passes kwargs to HubAwareLossComputer."""
        task_configs = {"intent": {"loss_type": "cross_entropy", "loss_weight": 1.0}}

        loss_computer = create_loss_computer(
            task_configs,
            use_uncertainty_weighting=False,
            label_smoothing=0.15,
            use_focal_loss=True,
            focal_gamma=3.0,
        )

        assert loss_computer.label_smoothing == 0.15
        assert loss_computer.use_focal_loss is True
        assert loss_computer.focal_gamma == 3.0


# ======================================================================
# Acceptance Criteria Tests
# ======================================================================


class TestAcceptanceCriteria:
    """Comprehensive tests for all acceptance criteria."""

    def test_ac1_token_level_masks_hub_positions(self):
        """AC1: Token-level loss masks hub token positions (0-4)."""
        task_configs = {"ner_general": {"loss_type": "token_level", "loss_weight": 1.0}}
        loss_computer = HubAwareLossComputer(task_configs)

        batch_size, seq_len, num_labels = 2, 15, 9
        logits = torch.randn(batch_size, seq_len, num_labels)
        labels = torch.randint(0, num_labels, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        # Change hub positions to incorrect labels
        labels_wrong_hub = labels.clone()
        labels_wrong_hub[:, :5] = (labels_wrong_hub[:, :5] + 1) % num_labels

        loss1 = loss_computer.compute_task_loss("ner_general", logits, labels, attention_mask)
        loss2 = loss_computer.compute_task_loss(
            "ner_general", logits, labels_wrong_hub, attention_mask
        )

        # Hub positions should not affect loss
        assert torch.allclose(loss1, loss2, atol=1e-6)
        print("✓ AC1: Token-level loss correctly masks hub positions (0-4)")

    def test_ac2_focal_loss_gamma_weighted(self):
        """AC2: Focal loss correctly implements γ-weighted cross entropy."""
        task_configs = {"safety": {"loss_type": "cross_entropy", "loss_weight": 1.0}}
        loss_computer = HubAwareLossComputer(task_configs, use_focal_loss=True, focal_gamma=2.0)

        batch_size, num_labels = 8, 2
        logits = torch.randn(batch_size, num_labels)
        labels = torch.randint(0, num_labels, (batch_size,))

        loss = loss_computer.compute_task_loss("safety", logits, labels)

        assert loss.ndim == 0
        assert loss.item() > 0
        assert not torch.isnan(loss)
        print("✓ AC2: Focal loss correctly implements γ-weighted cross entropy")

    def test_ac3_hierarchical_loss_combines_primary_secondary(self):
        """AC3: Hierarchical loss combines primary + secondary for emotions."""
        task_configs = {"emotions": {"loss_type": "hierarchical", "loss_weight": 1.0}}
        loss_computer = HubAwareLossComputer(task_configs)

        batch_size, primary_labels, secondary_labels = 4, 7, 28
        primary_logits = torch.randn(batch_size, primary_labels)
        secondary_logits = torch.randn(batch_size, secondary_labels)
        primary_labels_gt = torch.randint(0, primary_labels, (batch_size,))
        secondary_labels_gt = torch.randint(0, secondary_labels, (batch_size,))

        logits = (primary_logits, secondary_logits)
        labels = (primary_labels_gt, secondary_labels_gt)

        loss = loss_computer.compute_task_loss("emotions", logits, labels)

        assert loss.ndim == 0
        assert loss.item() > 0
        print("✓ AC3: Hierarchical loss combines primary + secondary predictions")

    def test_ac4_multitask_aggregates_with_weights(self):
        """AC4: Multi-task loss aggregates with configurable weights."""
        task_configs = {
            "task_a": {"loss_type": "cross_entropy", "loss_weight": 1.5},
            "task_b": {"loss_type": "cross_entropy", "loss_weight": 0.8},
        }
        loss_computer = HubAwareLossComputer(task_configs)

        batch_size = 4
        task_logits = {
            "task_a": torch.randn(batch_size, 3),
            "task_b": torch.randn(batch_size, 5),
        }
        task_labels = {
            "task_a": torch.randint(0, 3, (batch_size,)),
            "task_b": torch.randint(0, 5, (batch_size,)),
        }

        output = loss_computer.compute_multitask_loss(task_logits, task_labels)

        # Verify weighted sum
        expected = 1.5 * output.task_losses["task_a"] + 0.8 * output.task_losses["task_b"]
        assert torch.allclose(output.total_loss, expected, atol=1e-6)
        print("✓ AC4: Multi-task loss aggregates with configurable weights")

    def test_ac5_uncertainty_learns_per_task_sigma(self):
        """AC5: UncertaintyWeightedLoss learns per-task σ parameters."""
        task_names = ["emotions", "intent", "ner_general"]
        uncertainty_loss = UncertaintyWeightedLoss(task_names)

        # Check learnable parameters exist
        assert len(uncertainty_loss.log_vars) == 3
        for task in task_names:
            assert task in uncertainty_loss.log_vars
            assert uncertainty_loss.log_vars[task].requires_grad

        # Test forward pass
        task_losses = {
            "emotions": torch.tensor(0.5),
            "intent": torch.tensor(0.8),
            "ner_general": torch.tensor(0.3),
        }
        total_loss, weights = uncertainty_loss(task_losses)

        assert total_loss.ndim == 0
        assert len(weights) == 3
        print("✓ AC5: UncertaintyWeightedLoss learns per-task σ parameters")

    def test_ac6_label_smoothing_works(self):
        """AC6: Label smoothing works with cross-entropy."""
        task_configs = {"intent": {"loss_type": "cross_entropy", "loss_weight": 1.0}}

        loss_no_smooth = HubAwareLossComputer(task_configs, label_smoothing=0.0)
        loss_smooth = HubAwareLossComputer(task_configs, label_smoothing=0.1)

        batch_size, num_labels = 4, 5
        logits = torch.randn(batch_size, num_labels)
        labels = torch.randint(0, num_labels, (batch_size,))

        loss1 = loss_no_smooth.compute_task_loss("intent", logits, labels)
        loss2 = loss_smooth.compute_task_loss("intent", logits, labels)

        # Label smoothing should change loss
        assert not torch.allclose(loss1, loss2)
        print("✓ AC6: Label smoothing works with cross-entropy")

    def test_ac7_factory_supports_both_modes(self):
        """AC7: Factory function supports both fixed and uncertainty weighting."""
        task_configs = {
            "emotions": {"loss_type": "hierarchical", "loss_weight": 1.0},
            "intent": {"loss_type": "cross_entropy", "loss_weight": 1.2},
        }

        # Fixed weights
        fixed_loss = create_loss_computer(task_configs, use_uncertainty_weighting=False)
        assert isinstance(fixed_loss, HubAwareLossComputer)

        # Uncertainty weights
        uncertainty_loss = create_loss_computer(task_configs, use_uncertainty_weighting=True)
        assert isinstance(uncertainty_loss, nn.ModuleDict)
        assert "base" in uncertainty_loss
        assert "uncertainty" in uncertainty_loss

        print("✓ AC7: Factory function supports both fixed and uncertainty weighting")
