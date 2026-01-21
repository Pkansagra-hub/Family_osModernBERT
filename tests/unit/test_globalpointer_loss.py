"""
Unit tests for GlobalPointerLoss and FocalGlobalPointerLoss.

Tests the multi-label categorical cross-entropy loss function designed
for span-based NER with the GlobalPointer architecture.
"""

import pytest
import torch

from modeling_studio.models.losses import FocalGlobalPointerLoss, GlobalPointerLoss


class TestGlobalPointerLoss:
    """Tests for GlobalPointerLoss class."""

    @pytest.fixture
    def loss_fn(self) -> GlobalPointerLoss:
        """Create a default loss function."""
        return GlobalPointerLoss(reduction="mean")

    @pytest.fixture
    def sample_data(self) -> dict:
        """Create sample input data for testing."""
        batch_size = 2
        num_labels = 4
        seq_len = 32

        scores = torch.randn(batch_size, num_labels, seq_len, seq_len)
        labels = torch.zeros(batch_size, num_labels, seq_len, seq_len)
        # Add some positive spans
        labels[0, 0, 5, 10] = 1  # Entity type 0, tokens 5-10
        labels[0, 1, 15, 18] = 1  # Entity type 1, tokens 15-18
        labels[1, 0, 3, 7] = 1  # Entity type 0, tokens 3-7 in second sample
        attention_mask = torch.ones(batch_size, seq_len)

        return {
            "scores": scores,
            "labels": labels,
            "attention_mask": attention_mask,
            "batch_size": batch_size,
            "num_labels": num_labels,
            "seq_len": seq_len,
        }

    def test_output_is_scalar(self, loss_fn: GlobalPointerLoss, sample_data: dict):
        """Loss with mean reduction should return a scalar tensor."""
        loss = loss_fn(
            sample_data["scores"],
            sample_data["labels"],
            sample_data["attention_mask"],
        )
        assert loss.dim() == 0, "Loss should be scalar (0-dimensional)"
        assert loss.numel() == 1, "Loss should have exactly 1 element"

    def test_output_requires_grad(self, loss_fn: GlobalPointerLoss, sample_data: dict):
        """Loss should maintain gradient computation."""
        sample_data["scores"].requires_grad = True
        loss = loss_fn(
            sample_data["scores"],
            sample_data["labels"],
            sample_data["attention_mask"],
        )
        assert loss.requires_grad, "Loss should require gradients"

    def test_gradient_flow(self, loss_fn: GlobalPointerLoss, sample_data: dict):
        """Gradients should flow back through the loss."""
        sample_data["scores"].requires_grad = True
        loss = loss_fn(
            sample_data["scores"],
            sample_data["labels"],
            sample_data["attention_mask"],
        )
        loss.backward()
        assert sample_data["scores"].grad is not None, "Gradients should be computed"
        assert not torch.isnan(sample_data["scores"].grad).any(), "Gradients should not be NaN"
        assert not torch.isinf(sample_data["scores"].grad).any(), "Gradients should not be Inf"

    def test_low_loss_matching_predictions(self, loss_fn: GlobalPointerLoss, sample_data: dict):
        """Loss should be lower when predictions match labels."""
        # Create scores that match labels (high where labels=1, low elsewhere)
        labels = sample_data["labels"]
        matching_scores = labels * 10.0 - (1 - labels) * 10.0

        loss_matching = loss_fn(
            matching_scores,
            labels,
            sample_data["attention_mask"],
        )

        # Random scores
        random_scores = torch.randn_like(labels)
        loss_random = loss_fn(
            random_scores,
            labels,
            sample_data["attention_mask"],
        )

        assert loss_matching < loss_random, "Matching predictions should have lower loss"

    def test_high_loss_inverted_predictions(self, loss_fn: GlobalPointerLoss, sample_data: dict):
        """Loss should be high when predictions are inverted."""
        labels = sample_data["labels"]

        # Inverted scores (high where labels=0, low where labels=1)
        inverted_scores = -labels * 10.0 + (1 - labels) * 10.0

        # Matching scores
        matching_scores = labels * 10.0 - (1 - labels) * 10.0

        loss_inverted = loss_fn(
            inverted_scores,
            labels,
            sample_data["attention_mask"],
        )
        loss_matching = loss_fn(
            matching_scores,
            labels,
            sample_data["attention_mask"],
        )

        assert loss_inverted > loss_matching, "Inverted predictions should have higher loss"

    def test_padding_mask_effect(self, loss_fn: GlobalPointerLoss, sample_data: dict):
        """Padding positions should not contribute to loss."""
        scores = sample_data["scores"]
        labels = sample_data["labels"]

        # Full mask (no padding)
        full_mask = torch.ones(sample_data["batch_size"], sample_data["seq_len"])
        loss_full = loss_fn(scores, labels, full_mask)

        # Partial mask (last half is padding)
        partial_mask = torch.ones(sample_data["batch_size"], sample_data["seq_len"])
        partial_mask[:, sample_data["seq_len"] // 2 :] = 0
        loss_partial = loss_fn(scores, labels, partial_mask)

        # Losses should be different since different positions contribute
        assert loss_full != loss_partial, "Padding should affect loss computation"

    def test_lower_triangular_masked(self, loss_fn: GlobalPointerLoss, sample_data: dict):
        """Lower triangular positions (invalid spans) should be masked."""
        batch_size = 2
        num_labels = 1
        seq_len = 8

        # Create scores with high values in lower triangle
        scores = torch.zeros(batch_size, num_labels, seq_len, seq_len)
        # Put large values in lower triangle (invalid spans where end < start)
        for i in range(seq_len):
            for j in range(i):
                scores[:, :, i, j] = 100.0  # Should be ignored

        labels = torch.zeros_like(scores)
        mask = torch.ones(batch_size, seq_len)

        # This should not cause high loss since lower triangle is masked
        loss = loss_fn(scores, labels, mask)
        assert loss < 50.0, "Lower triangular values should not contribute to loss"

    def test_no_attention_mask(self, loss_fn: GlobalPointerLoss, sample_data: dict):
        """Loss should work without attention mask."""
        loss = loss_fn(
            sample_data["scores"],
            sample_data["labels"],
            attention_mask=None,
        )
        assert not torch.isnan(loss), "Loss should be valid without attention mask"

    def test_empty_labels(self, loss_fn: GlobalPointerLoss):
        """Loss should handle all-zero labels (no entities)."""
        batch_size, num_labels, seq_len = 2, 4, 32
        scores = torch.randn(batch_size, num_labels, seq_len, seq_len)
        labels = torch.zeros(batch_size, num_labels, seq_len, seq_len)
        mask = torch.ones(batch_size, seq_len)

        loss = loss_fn(scores, labels, mask)
        assert not torch.isnan(loss), "Loss should be valid with no positive labels"
        assert not torch.isinf(loss), "Loss should be finite with no positive labels"

    def test_all_positive_labels(self, loss_fn: GlobalPointerLoss):
        """Loss should handle all-one labels (all spans are entities)."""
        batch_size, num_labels, seq_len = 2, 4, 8
        scores = torch.randn(batch_size, num_labels, seq_len, seq_len)
        labels = torch.ones(batch_size, num_labels, seq_len, seq_len)
        mask = torch.ones(batch_size, seq_len)

        loss = loss_fn(scores, labels, mask)
        assert not torch.isnan(loss), "Loss should be valid with all positive labels"
        assert not torch.isinf(loss), "Loss should be finite with all positive labels"

    def test_batch_consistency(self, loss_fn: GlobalPointerLoss):
        """Same sample repeated in batch should give consistent loss per sample."""
        num_labels, seq_len = 4, 16
        single_score = torch.randn(1, num_labels, seq_len, seq_len)
        single_label = torch.zeros(1, num_labels, seq_len, seq_len)
        single_label[0, 0, 3, 7] = 1

        # Single sample loss
        loss_single = loss_fn(
            single_score,
            single_label,
            torch.ones(1, seq_len),
        )

        # Same sample duplicated in batch
        batch_score = single_score.repeat(4, 1, 1, 1)
        batch_label = single_label.repeat(4, 1, 1, 1)
        loss_batch = loss_fn(
            batch_score,
            batch_label,
            torch.ones(4, seq_len),
        )

        # Mean reduction should give same result
        assert torch.allclose(
            loss_single, loss_batch, rtol=1e-4
        ), "Batch of same samples should have same mean loss"

    def test_numerical_stability_extreme_logits(self, loss_fn: GlobalPointerLoss):
        """Loss should not produce NaN/Inf with extreme logit values."""
        batch_size, num_labels, seq_len = 2, 4, 16
        mask = torch.ones(batch_size, seq_len)

        # Very large positive logits
        large_scores = torch.ones(batch_size, num_labels, seq_len, seq_len) * 100
        labels = torch.zeros(batch_size, num_labels, seq_len, seq_len)
        labels[0, 0, 2, 5] = 1

        loss_large = loss_fn(large_scores, labels, mask)
        assert not torch.isnan(loss_large), "Loss should be valid with large logits"
        assert not torch.isinf(loss_large), "Loss should be finite with large logits"

        # Very large negative logits
        small_scores = torch.ones(batch_size, num_labels, seq_len, seq_len) * -100
        loss_small = loss_fn(small_scores, labels, mask)
        assert not torch.isnan(loss_small), "Loss should be valid with small logits"
        assert not torch.isinf(loss_small), "Loss should be finite with small logits"

    def test_reduction_none(self):
        """Loss with reduction='none' should return per-sample losses."""
        loss_fn = GlobalPointerLoss(reduction="none")
        batch_size, num_labels, seq_len = 3, 4, 16

        scores = torch.randn(batch_size, num_labels, seq_len, seq_len)
        labels = torch.zeros(batch_size, num_labels, seq_len, seq_len)
        mask = torch.ones(batch_size, seq_len)

        loss = loss_fn(scores, labels, mask)
        expected_elements = batch_size * num_labels
        assert loss.numel() == expected_elements, f"Expected {expected_elements} loss values"

    def test_reduction_sum(self):
        """Loss with reduction='sum' should sum all losses."""
        loss_fn_sum = GlobalPointerLoss(reduction="sum")
        loss_fn_none = GlobalPointerLoss(reduction="none")
        batch_size, num_labels, seq_len = 2, 4, 16

        scores = torch.randn(batch_size, num_labels, seq_len, seq_len)
        labels = torch.zeros(batch_size, num_labels, seq_len, seq_len)
        mask = torch.ones(batch_size, seq_len)

        loss_sum = loss_fn_sum(scores, labels, mask)
        loss_none = loss_fn_none(scores, labels, mask)

        assert torch.allclose(
            loss_sum, loss_none.sum(), rtol=1e-4
        ), "Sum reduction should equal sum of none reduction"

    def test_mask_diagonal_option(self):
        """mask_diagonal=True should exclude single-token spans."""
        loss_fn_with_diag = GlobalPointerLoss(mask_diagonal=False)
        loss_fn_no_diag = GlobalPointerLoss(mask_diagonal=True)

        batch_size, num_labels, seq_len = 2, 4, 8
        scores = torch.randn(batch_size, num_labels, seq_len, seq_len)
        labels = torch.zeros(batch_size, num_labels, seq_len, seq_len)
        # Add entity on diagonal (single token)
        labels[0, 0, 3, 3] = 1
        mask = torch.ones(batch_size, seq_len)

        loss_with = loss_fn_with_diag(scores, labels, mask)
        loss_without = loss_fn_no_diag(scores, labels, mask)

        # Losses should differ since diagonal is treated differently
        assert loss_with != loss_without, "Diagonal masking should affect loss"

    def test_device_compatibility(self, loss_fn: GlobalPointerLoss, sample_data: dict):
        """Loss should work on both CPU and CUDA (if available)."""
        # CPU test
        loss_cpu = loss_fn(
            sample_data["scores"],
            sample_data["labels"],
            sample_data["attention_mask"],
        )
        assert loss_cpu.device.type == "cpu"

        # CUDA test if available
        if torch.cuda.is_available():
            scores_cuda = sample_data["scores"].cuda()
            labels_cuda = sample_data["labels"].cuda()
            mask_cuda = sample_data["attention_mask"].cuda()

            loss_cuda = loss_fn(scores_cuda, labels_cuda, mask_cuda)
            assert loss_cuda.device.type == "cuda"
            assert torch.allclose(
                loss_cpu, loss_cuda.cpu(), rtol=1e-4
            ), "CPU and CUDA should give same result"


class TestFocalGlobalPointerLoss:
    """Tests for FocalGlobalPointerLoss class."""

    @pytest.fixture
    def sample_data(self) -> dict:
        """Create sample input data for testing."""
        batch_size = 2
        num_labels = 4
        seq_len = 32

        scores = torch.randn(batch_size, num_labels, seq_len, seq_len)
        labels = torch.zeros(batch_size, num_labels, seq_len, seq_len)
        labels[0, 0, 5, 10] = 1
        labels[0, 1, 15, 18] = 1
        attention_mask = torch.ones(batch_size, seq_len)

        return {
            "scores": scores,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def test_output_is_scalar(self, sample_data: dict):
        """Loss should return a scalar."""
        loss_fn = FocalGlobalPointerLoss(gamma=2.0)
        loss = loss_fn(
            sample_data["scores"],
            sample_data["labels"],
            sample_data["attention_mask"],
        )
        assert loss.dim() == 0, "Loss should be scalar"

    def test_gamma_zero_similar_to_base(self, sample_data: dict):
        """Gamma=0 should give similar results to base GlobalPointerLoss."""
        base_loss_fn = GlobalPointerLoss()
        focal_loss_fn = FocalGlobalPointerLoss(gamma=0.0)

        base_loss = base_loss_fn(
            sample_data["scores"],
            sample_data["labels"],
            sample_data["attention_mask"],
        )
        focal_loss = focal_loss_fn(
            sample_data["scores"],
            sample_data["labels"],
            sample_data["attention_mask"],
        )

        # With gamma=0, focal weight = 1, so should be similar
        # Not exactly equal due to different computation paths
        ratio = focal_loss / base_loss
        assert 0.5 < ratio < 2.0, "Gamma=0 focal loss should be similar magnitude to base"

    def test_higher_gamma_affects_loss(self, sample_data: dict):
        """Higher gamma should produce different loss values."""
        loss_fn_low = FocalGlobalPointerLoss(gamma=0.5)
        loss_fn_high = FocalGlobalPointerLoss(gamma=3.0)

        loss_low = loss_fn_low(
            sample_data["scores"],
            sample_data["labels"],
            sample_data["attention_mask"],
        )
        loss_high = loss_fn_high(
            sample_data["scores"],
            sample_data["labels"],
            sample_data["attention_mask"],
        )

        assert loss_low != loss_high, "Different gamma values should give different losses"

    def test_gradient_flow(self, sample_data: dict):
        """Gradients should flow through focal loss."""
        loss_fn = FocalGlobalPointerLoss(gamma=2.0)
        sample_data["scores"].requires_grad = True

        loss = loss_fn(
            sample_data["scores"],
            sample_data["labels"],
            sample_data["attention_mask"],
        )
        loss.backward()

        assert sample_data["scores"].grad is not None
        assert not torch.isnan(sample_data["scores"].grad).any()

    def test_numerical_stability(self, sample_data: dict):
        """Focal loss should be numerically stable."""
        loss_fn = FocalGlobalPointerLoss(gamma=2.0)

        # Extreme scores
        extreme_scores = sample_data["scores"] * 100

        loss = loss_fn(
            extreme_scores,
            sample_data["labels"],
            sample_data["attention_mask"],
        )

        assert not torch.isnan(loss), "Focal loss should not be NaN"
        assert not torch.isinf(loss), "Focal loss should not be Inf"


class TestGlobalPointerLossEdgeCases:
    """Edge case tests for GlobalPointerLoss."""

    def test_single_sample_single_label(self):
        """Should handle minimal dimensions."""
        loss_fn = GlobalPointerLoss()
        scores = torch.randn(1, 1, 8, 8)
        labels = torch.zeros(1, 1, 8, 8)
        labels[0, 0, 2, 5] = 1
        mask = torch.ones(1, 8)

        loss = loss_fn(scores, labels, mask)
        assert loss.dim() == 0

    def test_very_long_sequence(self):
        """Should handle long sequences without memory issues."""
        loss_fn = GlobalPointerLoss()
        seq_len = 512
        scores = torch.randn(1, 2, seq_len, seq_len)
        labels = torch.zeros(1, 2, seq_len, seq_len)
        labels[0, 0, 10, 50] = 1
        mask = torch.ones(1, seq_len)

        loss = loss_fn(scores, labels, mask)
        assert not torch.isnan(loss)

    def test_many_labels(self):
        """Should handle many entity types."""
        loss_fn = GlobalPointerLoss()
        num_labels = 50
        scores = torch.randn(2, num_labels, 16, 16)
        labels = torch.zeros(2, num_labels, 16, 16)
        for i in range(num_labels):
            labels[0, i, 2, 5 + (i % 10)] = 1
        mask = torch.ones(2, 16)

        loss = loss_fn(scores, labels, mask)
        assert not torch.isnan(loss)

    def test_half_precision(self):
        """Should work with float16."""
        loss_fn = GlobalPointerLoss()
        scores = torch.randn(2, 4, 16, 16, dtype=torch.float16)
        labels = torch.zeros(2, 4, 16, 16, dtype=torch.float16)
        labels[0, 0, 3, 7] = 1
        mask = torch.ones(2, 16, dtype=torch.float16)

        loss = loss_fn(scores, labels, mask)
        # May have some precision issues but should not be NaN
        assert not torch.isnan(loss)
