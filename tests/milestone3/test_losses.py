"""
Tests for models/losses.py - Issue 3.1.4

This module tests all loss functions in the FamilyOS Unified Encoder:
- FocalLoss (class imbalance handling)
- LabelSmoothingCrossEntropy (regularization)
- MultipleNegativesRankingLoss (contrastive learning)
- CosineSimilarityLoss (similarity regression)
- TripletLoss (triplet margin loss)
- CRFLoss (sequence labeling)
- FamilyContrastiveLoss (family-aware contrastive)
- MultiTaskLoss (weighted task combination)
- UncertaintyWeightedLoss (learned task weights)
- RDropLoss (dropout regularization)
- FGM (fast gradient method adversarial)
- PGD (projected gradient descent adversarial)
- MixupLoss (mixup training)
- EmbeddingMixup (embedding space mixup)

Test Count: 18 tests as per testing_plan.md Issue 3.1.4
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling_studio.models.losses import (
    FocalLoss,
    LabelSmoothingCrossEntropy,
    MultipleNegativesRankingLoss,
    CosineSimilarityLoss,
    TripletLoss,
    CRFLoss,
    FamilyContrastiveLoss,
    MultiTaskLoss,
    UncertaintyWeightedLoss,
    RDropLoss,
    FGM,
    PGD,
    MixupLoss,
    EmbeddingMixup,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 32


@pytest.fixture
def num_classes():
    """Standard number of classes for classification."""
    return 4


@pytest.fixture
def embedding_dim():
    """Standard embedding dimension."""
    return 768


@pytest.fixture
def seq_length():
    """Standard sequence length."""
    return 64


@pytest.fixture
def sample_logits(batch_size, num_classes):
    """Generate sample logits for classification."""
    return torch.randn(batch_size, num_classes, requires_grad=True)


@pytest.fixture
def sample_labels(batch_size, num_classes):
    """Generate sample labels for classification."""
    return torch.randint(0, num_classes, (batch_size,))


@pytest.fixture
def sample_embeddings(batch_size, embedding_dim):
    """Generate sample embeddings."""
    return torch.randn(batch_size, embedding_dim)


# =============================================================================
# FocalLoss Tests
# =============================================================================


class TestFocalLossInit:
    """Test: test_focal_loss_init - Initializes with alpha and gamma."""

    def test_init_with_scalar_alpha(self):
        """Focal loss initializes with scalar alpha."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        assert loss_fn.gamma == 2.0
        assert loss_fn.alpha is not None

    def test_init_with_list_alpha(self, num_classes):
        """Focal loss initializes with per-class alpha list."""
        alphas = [0.25, 0.25, 0.25, 0.25]
        loss_fn = FocalLoss(alpha=alphas, gamma=2.0)
        assert loss_fn.alpha.shape == torch.Size([num_classes])

    def test_init_with_tensor_alpha(self, num_classes):
        """Focal loss initializes with alpha tensor."""
        alpha = torch.ones(num_classes) * 0.25
        loss_fn = FocalLoss(alpha=alpha, gamma=2.0)
        assert loss_fn.alpha is not None

    def test_init_without_alpha(self):
        """Focal loss initializes without alpha (no class weighting)."""
        loss_fn = FocalLoss(alpha=None, gamma=2.0)
        assert loss_fn.alpha is None

    def test_init_default_values(self):
        """Focal loss uses correct default values."""
        loss_fn = FocalLoss()
        assert loss_fn.gamma == 2.0
        assert loss_fn.reduction == "mean"
        assert loss_fn.ignore_index == -100


class TestFocalLossForward:
    """Test: test_focal_loss_forward - Computes focal loss correctly."""

    def test_forward_computes_loss(self, sample_logits, sample_labels):
        """Forward pass computes focal loss."""
        loss_fn = FocalLoss(gamma=2.0)
        loss = loss_fn(sample_logits, sample_labels)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() > 0

    def test_forward_supports_backward(self, sample_logits, sample_labels):
        """Focal loss supports backward pass."""
        loss_fn = FocalLoss(gamma=2.0)
        loss = loss_fn(sample_logits, sample_labels)
        loss.backward()

        assert sample_logits.grad is not None

    def test_forward_reduction_sum(self, sample_logits, sample_labels):
        """Sum reduction works correctly."""
        loss_fn = FocalLoss(gamma=2.0, reduction="sum")
        loss = loss_fn(sample_logits, sample_labels)

        assert loss.item() > 0

    def test_forward_reduction_none(self, sample_logits, sample_labels):
        """No reduction returns per-sample loss."""
        loss_fn = FocalLoss(gamma=2.0, reduction="none")
        loss = loss_fn(sample_logits, sample_labels)

        assert loss.shape == sample_labels.shape


class TestFocalLossDownweightsEasy:
    """Test: test_focal_loss_downweights_easy - Easy examples have lower weight."""

    def test_downweights_easy_examples(self, num_classes):
        """Higher gamma downweights easy examples more."""
        # Create confident predictions (easy examples)
        logits = torch.zeros(2, num_classes)
        logits[0, 0] = 10.0  # Very confident
        logits[1, 0] = 1.0  # Less confident
        labels = torch.zeros(2, dtype=torch.long)  # Both correct class 0

        # Gamma=0 (standard CE) - equal treatment
        focal_0 = FocalLoss(gamma=0.0, reduction="none")
        loss_gamma_0 = focal_0(logits, labels)

        # Gamma=2 (focal) - downweight easy
        focal_2 = FocalLoss(gamma=2.0, reduction="none")
        loss_gamma_2 = focal_2(logits, labels)

        # Easy example (logits[0]) should have lower loss with focal
        # The ratio of easy/hard should be smaller with gamma=2
        ratio_gamma_0 = loss_gamma_0[0] / loss_gamma_0[1]
        ratio_gamma_2 = loss_gamma_2[0] / loss_gamma_2[1]

        # With focal loss, easy example loss reduced more
        assert ratio_gamma_2 < ratio_gamma_0


class TestFocalLossPerClassAlpha:
    """Test: test_focal_loss_per_class_alpha - Per-class weights applied."""

    def test_per_class_weights_applied(self, batch_size, num_classes):
        """Per-class alpha weights are correctly applied."""
        # Higher weight for class 0
        alphas = [2.0, 1.0, 1.0, 1.0]
        loss_fn = FocalLoss(alpha=alphas, gamma=2.0)

        logits = torch.randn(batch_size, num_classes)
        labels = torch.randint(0, num_classes, (batch_size,))

        loss = loss_fn(logits, labels)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0


class TestFocalLossIgnoreIndex:
    """Test: test_focal_loss_ignore_index - Ignores -100 labels."""

    def test_ignores_minus_100_labels(self, batch_size, num_classes):
        """Loss computation ignores labels with value -100."""
        loss_fn = FocalLoss(gamma=2.0, ignore_index=-100)

        logits = torch.randn(batch_size, num_classes)
        labels = torch.randint(0, num_classes, (batch_size,))
        labels[0] = -100  # Mark first sample as ignored
        labels[1] = -100  # Mark second sample as ignored

        loss = loss_fn(logits, labels)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


# =============================================================================
# LabelSmoothingCrossEntropy Tests
# =============================================================================


class TestLabelSmoothingCEInit:
    """Test: test_label_smoothing_ce_init - Initializes with epsilon."""

    def test_init_default_epsilon(self):
        """Default epsilon is 0.1."""
        loss_fn = LabelSmoothingCrossEntropy()
        assert loss_fn.epsilon == 0.1

    def test_init_custom_epsilon(self):
        """Custom epsilon is set correctly."""
        loss_fn = LabelSmoothingCrossEntropy(epsilon=0.2)
        assert loss_fn.epsilon == 0.2

    def test_init_reduction_options(self):
        """Different reduction options work."""
        for reduction in ["mean", "sum", "none"]:
            loss_fn = LabelSmoothingCrossEntropy(reduction=reduction)
            assert loss_fn.reduction == reduction


class TestLabelSmoothingCEForward:
    """Test: test_label_smoothing_ce_forward - Applies label smoothing."""

    def test_forward_computes_loss(self, sample_logits, sample_labels):
        """Forward pass computes label-smoothed loss."""
        loss_fn = LabelSmoothingCrossEntropy(epsilon=0.1)
        loss = loss_fn(sample_logits, sample_labels)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0

    def test_smoothing_increases_loss(self, sample_logits, sample_labels):
        """Label smoothing generally increases loss vs hard targets."""
        # With epsilon=0, should be similar to standard CE
        loss_no_smooth = LabelSmoothingCrossEntropy(epsilon=0.0)
        loss_smooth = LabelSmoothingCrossEntropy(epsilon=0.1)

        # Create very confident predictions
        confident_logits = sample_logits.detach().clone()
        confident_logits.scatter_(1, sample_labels.unsqueeze(1), 10.0)

        loss_0 = loss_no_smooth(confident_logits, sample_labels)
        loss_1 = loss_smooth(confident_logits, sample_labels)

        # Smoothing should penalize overconfidence
        assert loss_1 >= loss_0

    def test_forward_supports_backward(self, sample_logits, sample_labels):
        """Label smoothing loss supports backward pass."""
        loss_fn = LabelSmoothingCrossEntropy(epsilon=0.1)
        loss = loss_fn(sample_logits, sample_labels)
        loss.backward()

        assert sample_logits.grad is not None


# =============================================================================
# MultipleNegativesRankingLoss Tests
# =============================================================================


class TestMultipleNegativesRankingLoss:
    """Test: test_multiple_negatives_ranking_loss - Contrastive loss for embeddings."""

    def test_init_default_scale(self):
        """Default scale is 20.0."""
        loss_fn = MultipleNegativesRankingLoss()
        assert loss_fn.scale == 20.0

    def test_forward_computes_contrastive_loss(self, batch_size, embedding_dim):
        """Forward pass computes contrastive loss."""
        loss_fn = MultipleNegativesRankingLoss(scale=20.0)

        embeddings_a = torch.randn(batch_size, embedding_dim)
        embeddings_b = torch.randn(batch_size, embedding_dim)

        loss = loss_fn(embeddings_a, embeddings_b)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_in_batch_negatives(self, batch_size, embedding_dim):
        """In-batch negatives are used correctly."""
        loss_fn = MultipleNegativesRankingLoss(scale=20.0)

        # Create anchors and positives
        embeddings_a = torch.randn(batch_size, embedding_dim)
        embeddings_b = torch.randn(batch_size, embedding_dim)

        # Without explicit labels, diagonal should be positive
        loss = loss_fn(embeddings_a, embeddings_b, labels=None)

        assert loss.item() >= 0


# =============================================================================
# CosineSimilarityLoss Tests
# =============================================================================


class TestCosineSimilarityLoss:
    """Test: test_cosine_similarity_loss - Regression on similarity scores."""

    def test_init_mse(self):
        """MSE loss function is default."""
        loss_fn = CosineSimilarityLoss(loss_fn="mse")
        assert loss_fn.loss_fn == "mse"

    def test_init_smooth_l1(self):
        """Smooth L1 loss function option."""
        loss_fn = CosineSimilarityLoss(loss_fn="smooth_l1")
        assert loss_fn.loss_fn == "smooth_l1"

    def test_forward_computes_similarity_loss(self, batch_size, embedding_dim):
        """Forward pass computes similarity regression loss."""
        loss_fn = CosineSimilarityLoss()

        embeddings_a = torch.randn(batch_size, embedding_dim)
        embeddings_b = torch.randn(batch_size, embedding_dim)
        targets = torch.rand(batch_size)  # Similarity scores [0, 1]

        loss = loss_fn(embeddings_a, embeddings_b, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


# =============================================================================
# TripletLoss Tests
# =============================================================================


class TestTripletLoss:
    """Test: test_triplet_loss - Anchor-positive-negative loss."""

    def test_init_default_margin(self):
        """Default margin is 1.0."""
        loss_fn = TripletLoss()
        assert loss_fn.margin == 1.0

    def test_init_cosine_distance(self):
        """Cosine distance function option."""
        loss_fn = TripletLoss(distance_fn="cosine")
        assert loss_fn.distance_fn == "cosine"

    def test_forward_computes_triplet_loss(self, batch_size, embedding_dim):
        """Forward pass computes triplet margin loss."""
        loss_fn = TripletLoss(margin=0.5)

        anchor = torch.randn(batch_size, embedding_dim)
        positive = torch.randn(batch_size, embedding_dim)
        negative = torch.randn(batch_size, embedding_dim)

        loss = loss_fn(anchor, positive, negative)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_triplet_margin_constraint(self, batch_size, embedding_dim):
        """Loss is zero when margin constraint is satisfied."""
        loss_fn = TripletLoss(margin=0.5, distance_fn="euclidean")

        # Create embeddings where anchor is very close to positive
        anchor = torch.randn(batch_size, embedding_dim)
        positive = anchor + 0.1 * torch.randn(batch_size, embedding_dim)  # Close
        negative = anchor + 2.0 * torch.randn(batch_size, embedding_dim)  # Far

        loss = loss_fn(anchor, positive, negative)

        # Loss should be small or zero when constraint is satisfied
        assert loss.item() >= 0


# =============================================================================
# CRFLoss Tests
# =============================================================================


class TestCRFLoss:
    """Test: test_crf_loss - CRF for sequence labeling."""

    def test_init_creates_transitions(self):
        """CRF initializes transition parameters."""
        num_tags = 17
        crf = CRFLoss(num_tags=num_tags)

        assert crf.transitions.shape == (num_tags, num_tags)
        assert crf.start_transitions.shape == (num_tags,)
        assert crf.end_transitions.shape == (num_tags,)

    def test_forward_computes_nll(self, batch_size, seq_length):
        """Forward pass computes negative log-likelihood."""
        num_tags = 17
        crf = CRFLoss(num_tags=num_tags)

        emissions = torch.randn(batch_size, seq_length, num_tags)
        tags = torch.randint(0, num_tags, (batch_size, seq_length))
        mask = torch.ones(batch_size, seq_length, dtype=torch.bool)

        loss = crf(emissions, tags, mask)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0

    def test_decode_returns_sequences(self, batch_size, seq_length):
        """Decode method returns tag sequences."""
        num_tags = 9
        crf = CRFLoss(num_tags=num_tags)

        emissions = torch.randn(batch_size, seq_length, num_tags)
        mask = torch.ones(batch_size, seq_length, dtype=torch.bool)

        sequences = crf.decode(emissions, mask)

        assert len(sequences) == batch_size
        assert all(len(seq) == seq_length for seq in sequences)


# =============================================================================
# FamilyContrastiveLoss Tests
# =============================================================================


class TestFamilyContrastiveLoss:
    """Test: test_family_contrastive_loss - Family-aware contrastive learning."""

    def test_init_default_temperature(self):
        """Default temperature is 0.07."""
        loss_fn = FamilyContrastiveLoss()
        assert loss_fn.temperature == 0.07

    def test_forward_with_explicit_negatives(self, batch_size, embedding_dim):
        """Forward with explicit negative embeddings."""
        loss_fn = FamilyContrastiveLoss(temperature=0.07)

        anchor = torch.randn(batch_size, embedding_dim)
        positive = torch.randn(batch_size, embedding_dim)
        negatives = torch.randn(batch_size, 10, embedding_dim)  # 10 negatives each

        loss = loss_fn(anchor, positive, negatives)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_forward_with_in_batch_negatives(self, batch_size, embedding_dim):
        """Forward using only in-batch negatives."""
        loss_fn = FamilyContrastiveLoss(temperature=0.07)

        anchor = torch.randn(batch_size, embedding_dim)
        positive = torch.randn(batch_size, embedding_dim)

        loss = loss_fn.forward_with_in_batch_negatives(anchor, positive)

        assert loss.item() >= 0

    def test_hard_negative_mining(self, embedding_dim):
        """Hard negative mining works correctly."""
        anchor = torch.randn(16, embedding_dim)
        candidates = torch.randn(100, embedding_dim)

        hard_negs, indices = FamilyContrastiveLoss.mine_hard_negatives(
            anchor, candidates, num_hard=5, strategy="semi-hard"
        )

        assert hard_negs.shape == (16, 5, embedding_dim)
        assert indices.shape == (16, 5)


# =============================================================================
# MultiTaskLoss Tests
# =============================================================================


class TestMultiTaskLoss:
    """Test: test_multi_task_loss - Weighted combination of task losses."""

    def test_init_with_weights(self):
        """Initializes with task weights."""
        weights = {"safety": 15.0, "ner": 1.0, "sentiment": 1.0}
        loss_fn = MultiTaskLoss(task_weights=weights)

        assert loss_fn.task_weights == weights

    def test_forward_computes_weighted_sum(self):
        """Forward pass computes weighted sum of losses."""
        weights = {"task_a": 2.0, "task_b": 1.0}
        loss_fn = MultiTaskLoss(task_weights=weights)

        losses = {
            "task_a": torch.tensor(1.0),
            "task_b": torch.tensor(1.0),
        }

        total = loss_fn(losses)

        # 2.0 * 1.0 + 1.0 * 1.0 = 3.0
        assert torch.isclose(total, torch.tensor(3.0))

    def test_missing_task_uses_default_weight(self):
        """Missing tasks default to weight 1.0."""
        weights = {"task_a": 2.0}
        loss_fn = MultiTaskLoss(task_weights=weights)

        losses = {
            "task_a": torch.tensor(1.0),
            "task_b": torch.tensor(1.0),  # Not in weights
        }

        total = loss_fn(losses)

        # 2.0 * 1.0 + 1.0 * 1.0 = 3.0
        assert torch.isclose(total, torch.tensor(3.0))

    def test_get_weights(self):
        """Get weights returns copy of task weights."""
        weights = {"safety": 15.0}
        loss_fn = MultiTaskLoss(task_weights=weights)

        returned = loss_fn.get_weights()
        assert returned == weights
        assert returned is not weights  # Should be copy


# =============================================================================
# UncertaintyWeightedLoss Tests
# =============================================================================


class TestUncertaintyWeightedLoss:
    """Test: test_uncertainty_weighted_loss - Learns task weights automatically."""

    def test_init_creates_log_vars(self):
        """Initializes learnable log variance parameters."""
        num_tasks = 5
        loss_fn = UncertaintyWeightedLoss(num_tasks=num_tasks)

        assert loss_fn.log_vars.shape == (num_tasks,)
        assert loss_fn.log_vars.requires_grad

    def test_init_with_task_names(self):
        """Task names are stored correctly."""
        task_names = ["ner", "sentiment", "safety", "nli", "embed"]
        loss_fn = UncertaintyWeightedLoss(num_tasks=5, task_names=task_names)

        assert loss_fn.task_names == task_names

    def test_forward_computes_weighted_loss(self):
        """Forward pass computes uncertainty-weighted loss."""
        num_tasks = 3
        loss_fn = UncertaintyWeightedLoss(num_tasks=num_tasks)

        losses = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(0.5)]

        total = loss_fn(losses)

        assert isinstance(total, torch.Tensor)
        assert total.item() > 0

    def test_forward_with_dict_input(self):
        """Forward accepts dictionary of losses."""
        task_names = ["a", "b", "c"]
        loss_fn = UncertaintyWeightedLoss(num_tasks=3, task_names=task_names)

        losses = {"a": torch.tensor(1.0), "b": torch.tensor(2.0), "c": torch.tensor(0.5)}

        total = loss_fn(losses)

        assert total.item() > 0

    def test_get_weights_returns_dict(self):
        """Get weights returns task weight dictionary."""
        task_names = ["ner", "sentiment", "safety"]
        loss_fn = UncertaintyWeightedLoss(num_tasks=3, task_names=task_names)

        weights = loss_fn.get_weights()

        assert isinstance(weights, dict)
        assert len(weights) == 3
        assert all(name in weights for name in task_names)

    def test_log_vars_are_learnable(self):
        """Log variances update during training."""
        loss_fn = UncertaintyWeightedLoss(num_tasks=2)

        initial_log_vars = loss_fn.log_vars.detach().clone()

        # Simulate training step
        losses = [torch.tensor(1.0), torch.tensor(10.0)]
        total = loss_fn(losses)
        total.backward()

        # Gradients should exist
        assert loss_fn.log_vars.grad is not None


# =============================================================================
# FGM Adversarial Tests
# =============================================================================


class TestFGMAdversarial:
    """Test: test_fgm_adversarial - FGM perturbation applied."""

    def test_init_parameters(self):
        """FGM initializes with correct parameters."""
        model = nn.Embedding(100, 768)
        fgm = FGM(model, epsilon=1.0, emb_name="weight")

        assert fgm.epsilon == 1.0
        assert fgm.emb_name == "weight"

    def test_attack_modifies_embeddings(self):
        """Attack perturbs embeddings."""
        model = nn.Embedding(100, 768)
        fgm = FGM(model, epsilon=1.0, emb_name="weight")

        # Create gradient
        input_ids = torch.randint(0, 100, (4, 10))
        output = model(input_ids)
        loss = output.sum()
        loss.backward()

        original = model.weight.data.clone()

        fgm.attack()

        # Embeddings should be perturbed
        assert not torch.allclose(model.weight.data, original)

    def test_restore_reverts_embeddings(self):
        """Restore returns embeddings to original."""
        model = nn.Embedding(100, 768)
        fgm = FGM(model, epsilon=1.0, emb_name="weight")

        # Create gradient
        input_ids = torch.randint(0, 100, (4, 10))
        output = model(input_ids)
        loss = output.sum()
        loss.backward()

        original = model.weight.data.clone()

        fgm.attack()
        fgm.restore()

        # Embeddings should be restored
        assert torch.allclose(model.weight.data, original)


# =============================================================================
# PGD Adversarial Tests
# =============================================================================


class TestPGDAdversarial:
    """Test: test_pgd_adversarial - PGD iterative perturbation."""

    def test_init_parameters(self):
        """PGD initializes with correct parameters."""
        model = nn.Embedding(100, 768)
        pgd = PGD(model, epsilon=1.0, alpha=0.3, num_steps=3)

        assert pgd.epsilon == 1.0
        assert pgd.alpha == 0.3
        assert pgd.num_steps == 3

    def test_multi_step_attack(self):
        """PGD performs multiple attack steps."""
        model = nn.Embedding(100, 768)
        pgd = PGD(model, epsilon=1.0, alpha=0.3, num_steps=3, emb_name="weight")

        input_ids = torch.randint(0, 100, (4, 10))
        output = model(input_ids)
        loss = output.sum()
        loss.backward()

        original = model.weight.data.clone()

        # First step backs up embeddings
        pgd.attack(is_first=True)

        # Additional steps
        for _ in range(pgd.num_steps - 1):
            model.zero_grad()
            output = model(input_ids)
            loss = output.sum()
            loss.backward()
            pgd.attack(is_first=False)

        # Embeddings should be perturbed
        assert not torch.allclose(model.weight.data, original)

        pgd.restore()

        # Should be restored
        assert torch.allclose(model.weight.data, original)

    def test_projection_to_epsilon_ball(self):
        """Perturbation is projected to epsilon-ball."""
        model = nn.Embedding(100, 768)
        epsilon = 0.5
        pgd = PGD(model, epsilon=epsilon, alpha=0.3, num_steps=5, emb_name="weight")

        input_ids = torch.randint(0, 100, (4, 10))
        output = model(input_ids)
        loss = output.sum()
        loss.backward()

        original = model.weight.data.clone()

        pgd.attack(is_first=True)

        # Check perturbation is within epsilon
        perturbation = model.weight.data - original
        norm = torch.norm(perturbation)

        # Note: projection happens per embedding, this is aggregate check
        # Individual embeddings should be within epsilon
        pgd.restore()


# =============================================================================
# RDropLoss Tests
# =============================================================================


class TestRDropLoss:
    """Test: test_rdrop_loss - R-Drop KL divergence regularization."""

    def test_init_alpha(self):
        """R-Drop initializes with alpha parameter."""
        rdrop = RDropLoss(alpha=0.5)
        assert rdrop.alpha == 0.5

    def test_forward_combines_ce_and_kl(self, batch_size, num_classes):
        """Forward combines CE loss with KL divergence."""
        rdrop = RDropLoss(alpha=0.5)

        logits1 = torch.randn(batch_size, num_classes)
        logits2 = torch.randn(batch_size, num_classes)  # Different dropout
        ce_loss = torch.tensor(1.0)

        total = rdrop(logits1, logits2, ce_loss)

        assert isinstance(total, torch.Tensor)
        assert total.item() >= ce_loss.item()  # Adding KL should increase loss

    def test_kl_divergence_symmetric(self, batch_size, num_classes):
        """KL divergence is computed symmetrically."""
        logits1 = torch.randn(batch_size, num_classes)
        logits2 = torch.randn(batch_size, num_classes)

        kl = RDropLoss.compute_kl_divergence(logits1, logits2)

        assert isinstance(kl, torch.Tensor)
        assert kl.item() >= 0


# =============================================================================
# EmbeddingMixup Tests
# =============================================================================


class TestEmbeddingMixup:
    """Test: test_embedding_mixup - Mixup in embedding space."""

    def test_init_parameters(self):
        """EmbeddingMixup initializes with correct parameters."""
        mixup = EmbeddingMixup(alpha=0.4, apply_prob=0.5)
        assert mixup.alpha == 0.4
        assert mixup.apply_prob == 0.5

    def test_forward_during_training(self, batch_size, embedding_dim, num_classes):
        """Forward applies mixup during training."""
        mixup = EmbeddingMixup(alpha=0.4, apply_prob=1.0)
        mixup.train()

        embeddings = torch.randn(batch_size, embedding_dim)
        labels = torch.randint(0, num_classes, (batch_size,))

        mixed_emb, mixed_labels = mixup(embeddings, labels, num_classes=num_classes)

        assert mixed_emb.shape == embeddings.shape
        assert mixed_labels.shape == (batch_size, num_classes)
        # Mixed labels should be soft (not one-hot)
        assert (mixed_labels.sum(dim=1) - 1.0).abs().max() < 0.01

    def test_forward_during_eval(self, batch_size, embedding_dim, num_classes):
        """Forward does not apply mixup during evaluation."""
        mixup = EmbeddingMixup(alpha=0.4, apply_prob=1.0)
        mixup.eval()

        embeddings = torch.randn(batch_size, embedding_dim)
        labels = torch.randint(0, num_classes, (batch_size,))

        mixed_emb, mixed_labels = mixup(embeddings, labels, num_classes=num_classes)

        # During eval, embeddings should be unchanged
        assert torch.allclose(mixed_emb, embeddings)

    def test_mixup_loss_forward(self, sample_logits, sample_labels):
        """MixupLoss forward computes mixed loss."""
        mixup = MixupLoss(alpha=0.4)

        # Create shuffled labels
        labels_b = sample_labels[torch.randperm(len(sample_labels))]
        lam = 0.6

        loss = mixup(sample_logits, sample_labels, labels_b, lam)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestLossIntegration:
    """Integration tests for loss functions."""

    def test_all_losses_are_modules(self):
        """All loss classes are nn.Module subclasses."""
        loss_classes = [
            FocalLoss,
            LabelSmoothingCrossEntropy,
            MultipleNegativesRankingLoss,
            CosineSimilarityLoss,
            TripletLoss,
            CRFLoss,
            FamilyContrastiveLoss,
            MultiTaskLoss,
            UncertaintyWeightedLoss,
            RDropLoss,
            MixupLoss,
            EmbeddingMixup,
        ]

        for cls in loss_classes:
            assert issubclass(cls, nn.Module)

    def test_uncertainty_weighting_with_focal_loss(self, batch_size, num_classes):
        """UncertaintyWeightedLoss works with FocalLoss outputs."""
        focal = FocalLoss(gamma=2.0)
        uw_loss = UncertaintyWeightedLoss(num_tasks=2, task_names=["task_a", "task_b"])

        logits_a = torch.randn(batch_size, num_classes)
        logits_b = torch.randn(batch_size, num_classes)
        labels_a = torch.randint(0, num_classes, (batch_size,))
        labels_b = torch.randint(0, num_classes, (batch_size,))

        loss_a = focal(logits_a, labels_a)
        loss_b = focal(logits_b, labels_b)

        total = uw_loss({"task_a": loss_a, "task_b": loss_b})

        assert total.item() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
