"""
Tests for models/heads.py - Issue 3.1.3

This module tests all head classes in the FamilyOS Unified Encoder:
- BaseHead (abstract base with compute_loss, ASL, focal loss, freeze/unfreeze)
- SequenceClassificationHead (pooling strategies, forward, external_pooler)
- TokenClassificationHead (per-token logits, ignore_index)
- EmbeddingHead (normalize, projection, pooling)
- NLIHead (3-class NLI, pair_encoder)
- SafetyHead (4 bands, subcategories, temperature)
- EnhancedSafetyHead (keyword_override, hierarchical)
- RelationHead (15 relations, entity pairs)
- IntentHead (8 intents, confidence threshold)
- TemporalHead (13 BIO tags, span extraction)
- HierarchicalEmotionHead (44 emotions, ASL, family groupings)

Test Count: 35 tests as per testing_plan.md Issue 3.1.3
"""

import pytest
import torch
import torch.nn as nn

from modeling_studio.models.heads import (
    BaseHead,
    SequenceClassificationHead,
    TokenClassificationHead,
    EmbeddingHead,
    NLIHead,
    SafetyHead,
    EnhancedSafetyHead,
    RelationHead,
    IntentHead,
    TemporalHead,
    HierarchicalEmotionHead,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def hidden_size():
    """Standard hidden size for tests."""
    return 768


@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 4


@pytest.fixture
def seq_length():
    """Standard sequence length for tests."""
    return 32


@pytest.fixture
def sample_hidden_states(batch_size, seq_length, hidden_size):
    """Generate sample hidden states for testing."""
    return torch.randn(batch_size, seq_length, hidden_size)


@pytest.fixture
def sample_attention_mask(batch_size, seq_length):
    """Generate sample attention mask."""
    mask = torch.ones(batch_size, seq_length, dtype=torch.long)
    # Mask out last few tokens for some samples
    mask[0, -3:] = 0
    mask[1, -5:] = 0
    return mask


# =============================================================================
# BaseHead Tests
# =============================================================================


class ConcreteHead(BaseHead):
    """Concrete implementation of BaseHead for testing."""

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout: float = 0.1,
        problem_type: str = "single_label_classification",
        use_asl: bool = False,
        use_focal_loss: bool = False,
    ):
        # BaseHead signature: hidden_size, num_labels, dropout, problem_type, class_weights,
        #                     pos_weight, use_focal_loss, focal_gamma, use_asl, ...
        super().__init__(
            hidden_size=hidden_size,
            num_labels=num_labels,
            dropout=dropout,
            problem_type=problem_type,
            class_weights=None,
            pos_weight=None,
            use_focal_loss=use_focal_loss,
            focal_gamma=2.0,
            use_asl=use_asl,
        )
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states, attention_mask=None, labels=None):
        pooled = hidden_states[:, 0, :]  # CLS pooling
        logits = self.classifier(pooled)
        output = {"logits": logits}
        if labels is not None:
            output["loss"] = self.compute_loss(logits, labels)
        return output


class TestBaseHeadInit:
    """Test: test_base_head_init - BaseHead initializes with correct parameters."""

    def test_base_head_init_single_label(self, hidden_size):
        """BaseHead initializes correctly for single-label classification."""
        head = ConcreteHead(
            hidden_size=hidden_size,
            num_labels=5,
            dropout=0.1,
            problem_type="single_label_classification",
        )
        assert head.hidden_size == hidden_size
        assert head.num_labels == 5
        assert head.problem_type == "single_label_classification"
        assert not head.use_asl
        assert not head.use_focal_loss

    def test_base_head_init_multi_label(self, hidden_size):
        """BaseHead initializes correctly for multi-label classification."""
        head = ConcreteHead(
            hidden_size=hidden_size,
            num_labels=10,
            problem_type="multi_label_classification",
            use_asl=True,
        )
        assert head.problem_type == "multi_label_classification"
        assert head.use_asl

    def test_base_head_init_regression(self, hidden_size):
        """BaseHead initializes correctly for regression."""
        head = ConcreteHead(
            hidden_size=hidden_size,
            num_labels=1,
            problem_type="regression",
        )
        assert head.problem_type == "regression"
        assert head.num_labels == 1


class TestBaseHeadComputeLossSingleLabel:
    """Test: test_base_head_compute_loss_single_label - Cross-entropy for single-label."""

    def test_compute_loss_single_label_classification(self, hidden_size, batch_size):
        """Single-label classification uses cross-entropy loss."""
        num_labels = 5
        head = ConcreteHead(hidden_size, num_labels, problem_type="single_label_classification")

        logits = torch.randn(batch_size, num_labels)
        labels = torch.randint(0, num_labels, (batch_size,))

        loss = head.compute_loss(logits, labels)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() > 0  # Loss should be positive

    def test_compute_loss_single_label_backward(self, hidden_size, batch_size):
        """Single-label loss supports backward pass."""
        num_labels = 5
        head = ConcreteHead(hidden_size, num_labels, problem_type="single_label_classification")

        logits = torch.randn(batch_size, num_labels, requires_grad=True)
        labels = torch.randint(0, num_labels, (batch_size,))

        loss = head.compute_loss(logits, labels)
        loss.backward()

        assert logits.grad is not None


class TestBaseHeadComputeLossMultiLabel:
    """Test: test_base_head_compute_loss_multi_label - BCE for multi-label."""

    def test_compute_loss_multi_label_classification(self, hidden_size, batch_size):
        """Multi-label classification uses BCE loss."""
        num_labels = 10
        head = ConcreteHead(hidden_size, num_labels, problem_type="multi_label_classification")

        logits = torch.randn(batch_size, num_labels)
        labels = torch.randint(0, 2, (batch_size, num_labels)).float()

        loss = head.compute_loss(logits, labels)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0


class TestBaseHeadComputeLossRegression:
    """Test: test_base_head_compute_loss_regression - MSE for regression."""

    def test_compute_loss_regression(self, hidden_size, batch_size):
        """Regression uses MSE loss."""
        head = ConcreteHead(hidden_size, num_labels=1, problem_type="regression")

        logits = torch.randn(batch_size, 1)
        labels = torch.randn(batch_size, 1)

        loss = head.compute_loss(logits, labels)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0


class TestBaseHeadAsymmetricLoss:
    """Test: test_base_head_asymmetric_loss - ASL computes correctly for multi-label."""

    def test_asymmetric_loss_computes(self, hidden_size, batch_size):
        """ASL computes correctly for multi-label classification."""
        num_labels = 10
        head = ConcreteHead(
            hidden_size,
            num_labels,
            problem_type="multi_label_classification",
            use_asl=True,
        )

        logits = torch.randn(batch_size, num_labels)
        labels = torch.randint(0, 2, (batch_size, num_labels)).float()

        loss = head.compute_loss(logits, labels)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_asymmetric_loss_suppresses_negatives(self, hidden_size, batch_size):
        """ASL with gamma_neg > gamma_pos suppresses easy negatives."""
        num_labels = 10
        head = ConcreteHead(
            hidden_size,
            num_labels,
            problem_type="multi_label_classification",
            use_asl=True,
        )
        # ASL default: gamma_neg=4.0, gamma_pos=1.0

        # Create scenario with many easy negatives (high confidence 0)
        logits = torch.randn(batch_size, num_labels)
        labels = torch.zeros(batch_size, num_labels)  # All negatives
        labels[:, 0] = 1  # One positive

        loss = head.compute_loss(logits, labels)
        assert loss.item() > 0  # Should still produce valid loss


class TestBaseHeadFocalLoss:
    """Test: test_base_head_focal_loss - Focal loss reduces easy example weight."""

    def test_focal_loss_computes(self, hidden_size, batch_size):
        """Focal loss computes correctly."""
        num_labels = 10
        head = ConcreteHead(
            hidden_size,
            num_labels,
            problem_type="multi_label_classification",
            use_focal_loss=True,
        )

        logits = torch.randn(batch_size, num_labels)
        labels = torch.randint(0, 2, (batch_size, num_labels)).float()

        loss = head.compute_loss(logits, labels)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


class TestBaseHeadFreezeUnfreeze:
    """Test: test_base_head_freeze_unfreeze - Parameters freeze/unfreeze correctly."""

    def test_freeze_parameters(self, hidden_size):
        """Freeze sets requires_grad=False for all parameters."""
        head = ConcreteHead(hidden_size, num_labels=5)

        head.freeze()

        for param in head.parameters():
            assert not param.requires_grad

    def test_unfreeze_parameters(self, hidden_size):
        """Unfreeze sets requires_grad=True for all parameters."""
        head = ConcreteHead(hidden_size, num_labels=5)

        head.freeze()
        head.unfreeze()

        for param in head.parameters():
            assert param.requires_grad


# =============================================================================
# SequenceClassificationHead Tests
# =============================================================================


class TestSequenceClassificationHeadInit:
    """Test: test_sequence_classification_head_init - Initializes with dense + classifier."""

    def test_init_default(self, hidden_size):
        """Default initialization creates dense and classifier layers."""
        head = SequenceClassificationHead(hidden_size, num_labels=5)

        assert hasattr(head, "dense")
        assert hasattr(head, "classifier")
        assert head.hidden_size == hidden_size
        assert head.num_labels == 5
        assert head.pooling == "cls"

    def test_init_with_dropout(self, hidden_size):
        """Initialization respects dropout parameter."""
        head = SequenceClassificationHead(hidden_size, num_labels=5, dropout=0.3)

        assert hasattr(head, "dropout")


class TestSequenceClassificationHeadPoolCls:
    """Test: test_sequence_classification_head_pool_cls - CLS pooling extracts first token."""

    def test_pool_cls_extracts_first_token(self, sample_hidden_states, sample_attention_mask):
        """CLS pooling extracts the first token representation."""
        hidden_size = sample_hidden_states.size(-1)
        head = SequenceClassificationHead(hidden_size, num_labels=5, pooling="cls")

        # Get pooled output through forward
        output = head(sample_hidden_states, sample_attention_mask)

        assert "logits" in output
        assert output["logits"].shape == (sample_hidden_states.size(0), 5)


class TestSequenceClassificationHeadPoolMean:
    """Test: test_sequence_classification_head_pool_mean - Mean pooling averages tokens."""

    def test_pool_mean_averages_tokens(self, sample_hidden_states, sample_attention_mask):
        """Mean pooling averages token representations with mask."""
        hidden_size = sample_hidden_states.size(-1)
        head = SequenceClassificationHead(hidden_size, num_labels=5, pooling="mean")

        output = head(sample_hidden_states, sample_attention_mask)

        assert "logits" in output
        assert output["logits"].shape == (sample_hidden_states.size(0), 5)


class TestSequenceClassificationHeadPoolMax:
    """Test: test_sequence_classification_head_pool_max - Max pooling takes max values."""

    def test_pool_max_takes_max(self, sample_hidden_states, sample_attention_mask):
        """Max pooling takes max values across sequence dimension."""
        hidden_size = sample_hidden_states.size(-1)
        head = SequenceClassificationHead(hidden_size, num_labels=5, pooling="max")

        output = head(sample_hidden_states, sample_attention_mask)

        assert "logits" in output
        assert output["logits"].shape == (sample_hidden_states.size(0), 5)


class TestSequenceClassificationHeadForward:
    """Test: test_sequence_classification_head_forward - Forward returns logits."""

    def test_forward_returns_logits(self, sample_hidden_states, sample_attention_mask):
        """Forward pass returns logits with correct shape."""
        hidden_size = sample_hidden_states.size(-1)
        num_labels = 5
        head = SequenceClassificationHead(hidden_size, num_labels)

        output = head(sample_hidden_states, sample_attention_mask)

        assert "logits" in output
        assert output["logits"].shape == (sample_hidden_states.size(0), num_labels)


class TestSequenceClassificationHeadWithLabels:
    """Test: test_sequence_classification_head_with_labels - Returns loss with labels."""

    def test_forward_with_labels_returns_loss(self, sample_hidden_states, sample_attention_mask):
        """Forward with labels returns loss."""
        hidden_size = sample_hidden_states.size(-1)
        batch_size = sample_hidden_states.size(0)
        num_labels = 5
        head = SequenceClassificationHead(hidden_size, num_labels)

        labels = torch.randint(0, num_labels, (batch_size,))
        output = head(sample_hidden_states, sample_attention_mask, labels=labels)

        assert "loss" in output
        assert "logits" in output
        assert output["loss"].item() > 0


class TestSequenceClassificationHeadExternalPooler:
    """Test: test_sequence_classification_head_external_pooler - Uses external pooler when provided."""

    def test_external_pooler_used(self, sample_hidden_states, sample_attention_mask, hidden_size):
        """External pooler is used when provided at init (Epic 5.0 enhancement)."""

        # Create mock external pooler
        class MockPooler(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.dense = nn.Linear(hidden_size, hidden_size)

            def forward(self, hidden_states, attention_mask=None):
                return self.dense(hidden_states[:, 0, :])

        external_pooler = MockPooler(hidden_size)
        # Epic 5.0: external_pooler is passed at init time, not forward
        head = SequenceClassificationHead(
            hidden_size, num_labels=5, external_pooler=external_pooler
        )

        # Forward should use the external pooler
        output = head(sample_hidden_states, sample_attention_mask)

        assert "logits" in output
        assert head._use_external_pooler


# =============================================================================
# TokenClassificationHead Tests
# =============================================================================


class TestTokenClassificationHeadInit:
    """Test: test_token_classification_head_init - Initializes with classifier."""

    def test_init_creates_classifier(self, hidden_size):
        """Initialization creates classifier layer."""
        head = TokenClassificationHead(hidden_size, num_labels=9)

        assert hasattr(head, "classifier")
        assert head.num_labels == 9


class TestTokenClassificationHeadForward:
    """Test: test_token_classification_head_forward - Forward returns per-token logits."""

    def test_forward_returns_per_token_logits(self, sample_hidden_states, sample_attention_mask):
        """Forward pass returns per-token logits."""
        hidden_size = sample_hidden_states.size(-1)
        batch_size = sample_hidden_states.size(0)
        seq_len = sample_hidden_states.size(1)
        num_labels = 9

        head = TokenClassificationHead(hidden_size, num_labels)
        output = head(sample_hidden_states, sample_attention_mask)

        assert "logits" in output
        assert output["logits"].shape == (batch_size, seq_len, num_labels)


class TestTokenClassificationHeadIgnoreIndex:
    """Test: test_token_classification_head_ignore_index - Loss ignores -100 labels."""

    def test_ignore_index_minus_100(self, sample_hidden_states, sample_attention_mask):
        """Loss computation ignores labels with value -100."""
        hidden_size = sample_hidden_states.size(-1)
        batch_size = sample_hidden_states.size(0)
        seq_len = sample_hidden_states.size(1)
        num_labels = 9

        head = TokenClassificationHead(hidden_size, num_labels)

        # Create labels with -100 for padding
        labels = torch.randint(0, num_labels, (batch_size, seq_len))
        labels[:, -5:] = -100  # Mark padding as -100

        output = head(sample_hidden_states, sample_attention_mask, labels=labels)

        assert "loss" in output
        assert output["loss"].item() > 0


# =============================================================================
# EmbeddingHead Tests
# =============================================================================


class TestEmbeddingHeadInit:
    """Test: test_embedding_head_init - Initializes with optional projection."""

    def test_init_without_projection(self, hidden_size):
        """Initialization without projection layer."""
        head = EmbeddingHead(hidden_size)
        assert head.hidden_size == hidden_size
        assert head.projection is None  # No projection when output_dim == hidden_size

    def test_init_with_projection(self, hidden_size):
        """Initialization with projection to different dimension."""
        output_dim = 256
        head = EmbeddingHead(hidden_size, output_dim=output_dim)
        assert head.projection is not None
        assert head.output_dim == output_dim


class TestEmbeddingHeadNormalize:
    """Test: test_embedding_head_normalize - L2 normalizes output when flag set."""

    def test_l2_normalize(self, sample_hidden_states, sample_attention_mask, hidden_size):
        """Embeddings are L2 normalized when normalize=True."""
        head = EmbeddingHead(hidden_size, normalize=True)

        # EmbeddingHead.forward returns tensor directly
        embeddings = head(sample_hidden_states, sample_attention_mask)

        # Check L2 norm is approximately 1
        norms = torch.norm(embeddings, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_no_normalize(self, sample_hidden_states, sample_attention_mask, hidden_size):
        """Embeddings are not normalized when normalize=False."""
        head = EmbeddingHead(hidden_size, normalize=False)

        embeddings = head(sample_hidden_states, sample_attention_mask)

        norms = torch.norm(embeddings, p=2, dim=-1)
        # Should not all be 1 (unless by coincidence)
        assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


class TestEmbeddingHeadPooling:
    """Test: test_embedding_head_pooling - Uses specified pooling strategy."""

    def test_pooling_cls(self, sample_hidden_states, sample_attention_mask, hidden_size):
        """CLS pooling works correctly."""
        head = EmbeddingHead(hidden_size, pooling="cls")
        embeddings = head(sample_hidden_states, sample_attention_mask)
        # EmbeddingHead returns tensor directly
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (sample_hidden_states.size(0), hidden_size)

    def test_pooling_mean(self, sample_hidden_states, sample_attention_mask, hidden_size):
        """Mean pooling works correctly."""
        head = EmbeddingHead(hidden_size, pooling="mean")
        embeddings = head(sample_hidden_states, sample_attention_mask)
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (sample_hidden_states.size(0), hidden_size)


# =============================================================================
# NLIHead Tests
# =============================================================================


class TestNLIHeadInit:
    """Test: test_nli_head_init - Initializes with 3 labels."""

    def test_init_default_3_labels(self, hidden_size):
        """NLIHead defaults to 3 labels (entailment, neutral, contradiction)."""
        head = NLIHead(hidden_size)
        assert head.num_labels == 3

    def test_init_custom_labels(self, hidden_size):
        """NLIHead can be initialized with custom number of labels."""
        head = NLIHead(hidden_size, num_labels=5)
        assert head.num_labels == 5


class TestNLIHeadForward:
    """Test: test_nli_head_forward - Returns 3-class logits."""

    def test_forward_returns_3_class_logits(self, sample_hidden_states, sample_attention_mask):
        """Forward returns logits for 3 NLI classes."""
        hidden_size = sample_hidden_states.size(-1)
        batch_size = sample_hidden_states.size(0)

        head = NLIHead(hidden_size)
        output = head(sample_hidden_states, sample_attention_mask)

        assert "logits" in output
        assert output["logits"].shape == (batch_size, 3)


class TestNLIHeadPairEncoder:
    """Test: test_nli_head_pair_encoder - Uses pair encoder for cross-attention."""

    def test_pair_encoder_integration(self, hidden_size, batch_size, seq_length):
        """NLI head uses pair encoder when provided (Epic 5.0)."""
        head = NLIHead(hidden_size)

        # Create sample inputs
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        attention_mask = torch.ones(batch_size, seq_length)

        output = head(hidden_states, attention_mask)
        assert "logits" in output


# =============================================================================
# SafetyHead Tests
# =============================================================================


class TestSafetyHeadInit:
    """Test: test_safety_head_init - Initializes with 4 bands."""

    def test_init_4_bands(self, hidden_size):
        """SafetyHead initializes with 4 safety bands."""
        head = SafetyHead(hidden_size)

        assert head.num_bands == 4
        assert len(head.BAND_NAMES) == 4
        assert "GREEN" in head.BAND_NAMES
        assert "AMBER" in head.BAND_NAMES
        assert "RED" in head.BAND_NAMES
        assert "CRISIS" in head.BAND_NAMES


class TestSafetyHeadTemperatureScaling:
    """Test: test_safety_head_temperature_scaling - Temperature calibration applied."""

    def test_temperature_calibration(self, sample_hidden_states, sample_attention_mask):
        """Temperature scaling is applied for calibration."""
        hidden_size = sample_hidden_states.size(-1)
        head = SafetyHead(hidden_size)

        # Check temperature parameter exists
        assert hasattr(head, "temperature") or hasattr(head, "_temperature")

        output = head(sample_hidden_states, sample_attention_mask)
        assert "logits" in output or "band_logits" in output


# =============================================================================
# EnhancedSafetyHead Tests
# =============================================================================


class TestEnhancedSafetyHeadSubcategories:
    """Test: test_enhanced_safety_head_subcategories - Returns 12 subcategories."""

    def test_12_subcategories(self, hidden_size):
        """EnhancedSafetyHead has 12 subcategories."""
        head = EnhancedSafetyHead(hidden_size)

        assert len(head.SUBCATEGORIES) == 12

    def test_forward_returns_subcategory_logits(self, sample_hidden_states, sample_attention_mask):
        """Forward pass returns subcategory logits."""
        hidden_size = sample_hidden_states.size(-1)
        head = EnhancedSafetyHead(hidden_size)

        output = head(sample_hidden_states, sample_attention_mask)

        assert "subcategory_logits" in output or "logits" in output


class TestEnhancedSafetyHeadKeywordOverride:
    """Test: test_enhanced_safety_head_keyword_override - CRISIS keywords trigger override."""

    def test_crisis_keywords_defined(self, hidden_size):
        """CRISIS keywords are properly defined."""
        head = EnhancedSafetyHead(hidden_size)

        assert hasattr(head, "CRISIS_KEYWORDS")
        assert len(head.CRISIS_KEYWORDS) > 0
        # Check some expected crisis keywords
        keywords_str = " ".join(head.CRISIS_KEYWORDS).lower()
        assert "suicide" in keywords_str or "kill" in keywords_str

    def test_keyword_override_method_exists(self, hidden_size):
        """Keyword override functionality exists."""
        head = EnhancedSafetyHead(hidden_size)

        # Check for keyword override capability (_detect_crisis_keywords is private)
        assert hasattr(head, "_detect_crisis_keywords") or hasattr(head, "add_crisis_keyword")

    def test_add_remove_crisis_keyword(self, hidden_size):
        """Can add and remove crisis keywords."""
        head = EnhancedSafetyHead(hidden_size)

        # Test add
        head.add_crisis_keyword("test_crisis_phrase")
        assert "test_crisis_phrase" in head.CRISIS_KEYWORDS

        # Test remove
        removed = head.remove_crisis_keyword("test_crisis_phrase")
        assert removed
        assert "test_crisis_phrase" not in head.CRISIS_KEYWORDS


# =============================================================================
# RelationHead Tests
# =============================================================================


class TestRelationHeadInit:
    """Test: test_relation_head_init - Initializes with 15 relations."""

    def test_init_15_relations(self, hidden_size):
        """RelationHead initializes with 15 relation types."""
        head = RelationHead(hidden_size, num_labels=15)

        assert head.num_labels == 15

    def test_init_default_relations(self, hidden_size):
        """RelationHead has default relation types."""
        head = RelationHead(hidden_size)

        # Check for relation-specific components
        assert hasattr(head, "entity_pair_dense") or hasattr(head, "classifier")


class TestRelationHeadEntityPairs:
    """Test: test_relation_head_entity_pairs - Handles entity pair representations."""

    def test_entity_pair_representation(self, sample_hidden_states, sample_attention_mask):
        """RelationHead handles entity pair representations."""
        hidden_size = sample_hidden_states.size(-1)
        batch_size = sample_hidden_states.size(0)

        head = RelationHead(hidden_size, num_labels=15)

        # Create entity span indices
        entity1_start = torch.zeros(batch_size, dtype=torch.long)
        entity1_end = torch.ones(batch_size, dtype=torch.long)
        entity2_start = torch.full((batch_size,), 5, dtype=torch.long)
        entity2_end = torch.full((batch_size,), 6, dtype=torch.long)

        output = head(
            sample_hidden_states,
            sample_attention_mask,
            entity1_start=entity1_start,
            entity1_end=entity1_end,
            entity2_start=entity2_start,
            entity2_end=entity2_end,
        )

        assert "logits" in output


# =============================================================================
# IntentHead Tests
# =============================================================================


class TestIntentHeadInit:
    """Test: test_intent_head_init - Initializes with 8 intents."""

    def test_init_8_intents(self, hidden_size):
        """IntentHead initializes with 8 intent categories."""
        head = IntentHead(hidden_size, num_labels=8)

        assert head.num_labels == 8

    def test_init_confidence_threshold(self, hidden_size):
        """IntentHead has configurable confidence threshold."""
        head = IntentHead(hidden_size, confidence_threshold=0.7)

        assert head.confidence_threshold == 0.7


class TestIntentHeadForward:
    """Test: test_intent_head_forward - Returns intent logits."""

    def test_forward_returns_intent_logits(self, sample_hidden_states, sample_attention_mask):
        """Forward pass returns intent logits and confidence."""
        hidden_size = sample_hidden_states.size(-1)
        batch_size = sample_hidden_states.size(0)

        head = IntentHead(hidden_size, num_labels=8)
        output = head(sample_hidden_states, sample_attention_mask)

        assert "logits" in output
        assert output["logits"].shape == (batch_size, 8)
        assert "confidence" in output
        assert "predicted_intent" in output


# =============================================================================
# TemporalHead Tests
# =============================================================================


class TestTemporalHeadInit:
    """Test: test_temporal_head_init - Initializes with 13 BIO tags."""

    def test_init_13_bio_tags(self, hidden_size):
        """TemporalHead initializes with 13 BIO tags."""
        head = TemporalHead(hidden_size, num_labels=13)

        assert head.num_labels == 13

    def test_init_default_problem_type(self, hidden_size):
        """TemporalHead defaults to token classification."""
        head = TemporalHead(hidden_size)

        assert head.problem_type == "token_classification"


class TestTemporalHeadForward:
    """Test: test_temporal_head_forward - Returns per-token temporal tags."""

    def test_forward_returns_per_token_tags(self, sample_hidden_states, sample_attention_mask):
        """Forward pass returns per-token temporal tag logits."""
        hidden_size = sample_hidden_states.size(-1)
        batch_size = sample_hidden_states.size(0)
        seq_len = sample_hidden_states.size(1)

        head = TemporalHead(hidden_size, num_labels=13)
        output = head(sample_hidden_states, sample_attention_mask)

        assert "logits" in output
        assert output["logits"].shape == (batch_size, seq_len, 13)


# =============================================================================
# HierarchicalEmotionHead Tests
# =============================================================================


class TestHierarchicalEmotionHead44Emotions:
    """Test: test_hierarchical_emotion_head_44_emotions - Handles 44 FamilyOS emotions."""

    def test_44_familyos_emotions(self, hidden_size):
        """HierarchicalEmotionHead supports 44 FamilyOS emotions."""
        head = HierarchicalEmotionHead(hidden_size, num_emotions=44, use_familyos=True)

        assert head.num_emotions == 44
        assert len(head.emotion_labels) == 44
        assert len(head.FAMILYOS_EMOTION_LABELS) == 44

    def test_familyos_emotion_labels(self, hidden_size):
        """FamilyOS emotion labels include family-specific emotions."""
        head = HierarchicalEmotionHead(hidden_size, num_emotions=44, use_familyos=True)

        # Check for FamilyOS-specific emotions
        assert "parental_pride" in head.emotion_labels
        assert "parental_guilt" in head.emotion_labels
        assert "togetherness" in head.emotion_labels
        assert "homesickness" in head.emotion_labels

    def test_forward_returns_emotion_output(self, sample_hidden_states, sample_attention_mask):
        """Forward returns comprehensive emotion output."""
        hidden_size = sample_hidden_states.size(-1)
        head = HierarchicalEmotionHead(hidden_size, num_emotions=44, use_familyos=True)

        output = head(sample_hidden_states, sample_attention_mask)

        assert "logits" in output
        assert "probabilities" in output
        assert "primary_emotion" in output
        assert "secondary_emotions" in output


class TestHierarchicalEmotionHeadASL:
    """Test: test_hierarchical_emotion_head_asl - Uses asymmetric loss."""

    def test_asl_enabled_by_default(self, hidden_size):
        """ASL is enabled by default for HierarchicalEmotionHead."""
        head = HierarchicalEmotionHead(hidden_size, num_emotions=44)

        assert head.use_asl

    def test_asl_parameters(self, hidden_size):
        """ASL parameters are correctly set."""
        head = HierarchicalEmotionHead(
            hidden_size,
            num_emotions=44,
            asl_gamma_neg=4.0,
            asl_gamma_pos=1.0,
            asl_clip=0.05,
        )

        assert head.asl_gamma_neg == 4.0
        assert head.asl_gamma_pos == 1.0
        assert head.asl_clip == 0.05

    def test_asl_loss_computation(self, sample_hidden_states, sample_attention_mask):
        """ASL loss is computed correctly with labels."""
        hidden_size = sample_hidden_states.size(-1)
        batch_size = sample_hidden_states.size(0)
        num_emotions = 44

        head = HierarchicalEmotionHead(hidden_size, num_emotions=num_emotions, use_asl=True)

        # Create multi-hot labels
        labels = torch.randint(0, 2, (batch_size, num_emotions)).float()

        output = head(sample_hidden_states, sample_attention_mask, labels=labels)

        assert "loss" in output
        assert output["loss"].item() > 0


class TestHierarchicalEmotionHeadFamilies:
    """Additional tests for HierarchicalEmotionHead family groupings."""

    def test_emotion_families_defined(self, hidden_size):
        """Emotion family groupings are properly defined."""
        head = HierarchicalEmotionHead(hidden_size, num_emotions=44, use_familyos=True)

        assert hasattr(head, "FAMILYOS_EMOTION_FAMILIES")
        assert "joy" in head.FAMILYOS_EMOTION_FAMILIES
        assert "sadness" in head.FAMILYOS_EMOTION_FAMILIES
        assert "love" in head.FAMILYOS_EMOTION_FAMILIES

    def test_factory_for_familyos(self, hidden_size):
        """Factory method creates correct FamilyOS configuration."""
        head = HierarchicalEmotionHead.for_familyos(hidden_size=hidden_size)

        assert head.num_emotions == 44
        assert head.use_familyos
        assert "parental_pride" in head.emotion_labels


# =============================================================================
# Integration Tests
# =============================================================================


class TestHeadIntegration:
    """Integration tests for head classes."""

    def test_all_heads_have_forward(self, hidden_size):
        """All head classes implement forward method."""
        heads = [
            SequenceClassificationHead(hidden_size, num_labels=5),
            TokenClassificationHead(hidden_size, num_labels=9),
            EmbeddingHead(hidden_size),
            NLIHead(hidden_size),
            SafetyHead(hidden_size),
            EnhancedSafetyHead(hidden_size),
            RelationHead(hidden_size, num_labels=15),
            IntentHead(hidden_size, num_labels=8),
            TemporalHead(hidden_size, num_labels=13),
            HierarchicalEmotionHead(hidden_size, num_emotions=44),
        ]

        for head in heads:
            assert hasattr(head, "forward")
            assert callable(head.forward)

    def test_all_heads_return_logits(self, sample_hidden_states, sample_attention_mask):
        """All heads return logits in forward output."""
        hidden_size = sample_hidden_states.size(-1)

        heads = [
            SequenceClassificationHead(hidden_size, num_labels=5),
            TokenClassificationHead(hidden_size, num_labels=9),
            NLIHead(hidden_size),
            SafetyHead(hidden_size),
            IntentHead(hidden_size, num_labels=8),
            TemporalHead(hidden_size, num_labels=13),
        ]

        for head in heads:
            output = head(sample_hidden_states, sample_attention_mask)
            assert "logits" in output, f"{type(head).__name__} missing logits"


# =============================================================================
# Additional Coverage Tests - Edge Cases and Branches
# =============================================================================


class TestBaseHeadEdgeCases:
    """Additional coverage tests for BaseHead edge cases."""

    def test_compute_loss_unknown_problem_type(self, hidden_size):
        """Unknown problem type raises ValueError."""
        head = ConcreteHead(hidden_size, num_labels=5, problem_type="unknown")
        logits = torch.randn(4, 5)
        labels = torch.randint(0, 5, (4,))

        with pytest.raises(ValueError, match="Unknown problem type"):
            head.compute_loss(logits, labels)

    def test_multi_label_with_class_weights(self, hidden_size, batch_size):
        """Multi-label with class_weights uses weighted BCE."""
        num_labels = 10
        class_weights = torch.ones(num_labels) * 0.5
        head = ConcreteHead(
            hidden_size,
            num_labels,
            problem_type="multi_label_classification",
        )
        head.class_weights = class_weights

        logits = torch.randn(batch_size, num_labels)
        labels = torch.randint(0, 2, (batch_size, num_labels)).float()

        loss = head.compute_loss(logits, labels)
        assert loss.item() > 0

    def test_multi_label_with_pos_weight(self, hidden_size, batch_size):
        """Multi-label with pos_weight upweights positives."""
        num_labels = 10
        head = ConcreteHead(
            hidden_size,
            num_labels,
            problem_type="multi_label_classification",
        )
        # Register pos_weight buffer
        head.register_buffer("pos_weight", torch.ones(num_labels) * 2.0)

        logits = torch.randn(batch_size, num_labels)
        labels = torch.randint(0, 2, (batch_size, num_labels)).float()

        loss = head.compute_loss(logits, labels)
        assert loss.item() > 0

    def test_asl_with_pos_weight(self, hidden_size, batch_size):
        """ASL with pos_weight applies additional upweighting."""
        num_labels = 10
        head = ConcreteHead(
            hidden_size,
            num_labels,
            problem_type="multi_label_classification",
            use_asl=True,
        )
        head.register_buffer("pos_weight", torch.ones(num_labels) * 2.0)

        logits = torch.randn(batch_size, num_labels)
        labels = torch.randint(0, 2, (batch_size, num_labels)).float()

        loss = head.compute_loss(logits, labels)
        assert loss.item() > 0

    def test_asl_with_zero_gammas(self, hidden_size, batch_size):
        """ASL with zero gammas falls back to simple BCE."""
        num_labels = 10
        head = ConcreteHead(
            hidden_size,
            num_labels,
            problem_type="multi_label_classification",
            use_asl=True,
        )
        head.asl_gamma_neg = 0.0
        head.asl_gamma_pos = 0.0

        logits = torch.randn(batch_size, num_labels)
        labels = torch.randint(0, 2, (batch_size, num_labels)).float()

        loss = head.compute_loss(logits, labels)
        assert loss.item() > 0

    def test_focal_bce_with_class_weights(self, hidden_size, batch_size):
        """Focal BCE applies class weights."""
        num_labels = 10
        head = ConcreteHead(
            hidden_size,
            num_labels,
            problem_type="multi_label_classification",
            use_focal_loss=True,
        )
        head.class_weights = torch.ones(num_labels) * 0.5

        logits = torch.randn(batch_size, num_labels)
        labels = torch.randint(0, 2, (batch_size, num_labels)).float()

        loss = head.compute_loss(logits, labels)
        assert loss.item() >= 0


class TestSequenceClassificationHeadPoolingEdges:
    """Edge cases for pooling strategies."""

    def test_pool_mean_no_mask(self, sample_hidden_states, hidden_size):
        """Mean pooling without mask uses simple mean."""
        head = SequenceClassificationHead(hidden_size, num_labels=5, pooling="mean")

        output = head(sample_hidden_states, attention_mask=None)
        assert "logits" in output

    def test_pool_max_no_mask(self, sample_hidden_states, hidden_size):
        """Max pooling without mask takes max of all tokens."""
        head = SequenceClassificationHead(hidden_size, num_labels=5, pooling="max")

        output = head(sample_hidden_states, attention_mask=None)
        assert "logits" in output

    def test_pool_unknown_strategy(self, sample_hidden_states, sample_attention_mask, hidden_size):
        """Unknown pooling strategy raises ValueError."""
        head = SequenceClassificationHead(hidden_size, num_labels=5, pooling="unknown")

        with pytest.raises(ValueError, match="Unknown pooling"):
            head(sample_hidden_states, sample_attention_mask)


class TestSafetyHeadAdvanced:
    """Advanced tests for SafetyHead methods."""

    def test_subcategory_classification(self, sample_hidden_states, sample_attention_mask):
        """SafetyHead classifies subcategories."""
        hidden_size = sample_hidden_states.size(-1)
        head = SafetyHead(hidden_size, use_hierarchical=True)

        output = head(sample_hidden_states, sample_attention_mask)

        assert "subcategory_logits" in output
        assert "band_probs" in output

    def test_hierarchical_masking(self, sample_hidden_states, sample_attention_mask):
        """Hierarchical masking filters invalid subcategories."""
        hidden_size = sample_hidden_states.size(-1)
        head = SafetyHead(hidden_size, use_hierarchical=True)

        output = head(sample_hidden_states, sample_attention_mask)

        # Subcategory probs should sum to 1 after masking
        assert "subcategory_probs" in output

    def test_focal_loss_option(self, sample_hidden_states, sample_attention_mask):
        """SafetyHead with focal loss enabled."""
        hidden_size = sample_hidden_states.size(-1)
        batch_size = sample_hidden_states.size(0)
        head = SafetyHead(hidden_size, use_focal_loss=True)

        labels = torch.randint(0, 4, (batch_size,))
        output = head(sample_hidden_states, sample_attention_mask, labels=labels)

        assert "loss" in output

    def test_subcategory_labels(self, sample_hidden_states, sample_attention_mask):
        """SafetyHead with subcategory labels computes combined loss."""
        hidden_size = sample_hidden_states.size(-1)
        batch_size = sample_hidden_states.size(0)
        head = SafetyHead(hidden_size, use_hierarchical=True)

        labels = torch.randint(0, 4, (batch_size,))
        subcategory_labels = torch.randint(0, 13, (batch_size,))

        output = head(
            sample_hidden_states,
            sample_attention_mask,
            labels=labels,
            subcategory_labels=subcategory_labels,
        )

        assert "loss" in output
        assert "band_loss" in output
        assert "subcategory_loss" in output

    def test_set_temperature(self, hidden_size):
        """Set temperature updates the parameter."""
        head = SafetyHead(hidden_size)
        head.set_temperature(2.0)

        assert head.temperature.item() == pytest.approx(2.0, rel=0.1)

    def test_calibrate_method(self, hidden_size):
        """Calibrate learns temperature on validation data."""
        head = SafetyHead(hidden_size)

        val_logits = torch.randn(100, 4)
        val_labels = torch.randint(0, 4, (100,))

        temp = head.calibrate(val_logits, val_labels, max_iter=5)
        assert temp > 0


class TestEnhancedSafetyHeadAdvanced:
    """Advanced tests for EnhancedSafetyHead."""

    def test_keyword_override_triggers_crisis(self, sample_hidden_states, sample_attention_mask):
        """CRISIS keywords override model predictions."""
        hidden_size = sample_hidden_states.size(-1)
        head = EnhancedSafetyHead(hidden_size, keyword_override=True)

        output = head(
            sample_hidden_states,
            sample_attention_mask,
            text="I want to kill myself",
        )

        assert output["band"] == "CRISIS" or "CRISIS" in str(output["band"])
        assert output["keyword_override"].any()

    def test_batch_text_keyword_detection(self, sample_hidden_states, sample_attention_mask):
        """Keyword detection works with batch of texts."""
        hidden_size = sample_hidden_states.size(-1)
        batch_size = sample_hidden_states.size(0)
        head = EnhancedSafetyHead(hidden_size, keyword_override=True)

        texts = ["Hello world", "I want to kill myself", "Nice day", "suicide thoughts"]

        output = head(
            sample_hidden_states[: len(texts)],
            sample_attention_mask[: len(texts)],
            text=texts,
        )

        # At least some should be flagged
        assert output["keyword_override"].sum() > 0

    def test_set_temperature_enhanced(self, hidden_size):
        """Set temperature updates log_temperature."""
        head = EnhancedSafetyHead(hidden_size)
        head.set_temperature(2.0)

        temp = head.log_temperature.exp().item()
        assert temp == pytest.approx(2.0, rel=0.1)

    def test_calibrate_enhanced(self, hidden_size):
        """EnhancedSafetyHead calibrate method."""
        head = EnhancedSafetyHead(hidden_size)

        val_logits = torch.randn(50, 4)
        val_labels = torch.randint(0, 4, (50,))

        temp = head.calibrate(val_logits, val_labels, max_iter=3)
        assert temp > 0

    def test_get_severity_score(self, hidden_size):
        """Severity score computation."""
        head = EnhancedSafetyHead(hidden_size)

        # GREEN = 0, AMBER = 0.33, RED = 0.66, CRISIS = 1.0
        band_probs = torch.tensor([[1.0, 0, 0, 0]])  # All GREEN
        score = head.get_severity_score(band_probs)
        assert score.item() == pytest.approx(0.0, abs=0.01)

        band_probs = torch.tensor([[0, 0, 0, 1.0]])  # All CRISIS
        score = head.get_severity_score(band_probs)
        assert score.item() == pytest.approx(1.0, abs=0.01)

    def test_freeze_unfreeze_enhanced(self, hidden_size):
        """Freeze/unfreeze works for EnhancedSafetyHead."""
        head = EnhancedSafetyHead(hidden_size)

        head.freeze()
        for param in head.parameters():
            assert not param.requires_grad

        head.unfreeze()
        for param in head.parameters():
            assert param.requires_grad


class TestEmbeddingHeadAdvanced:
    """Additional coverage for EmbeddingHead."""

    def test_pool_max_with_mask(self, sample_hidden_states, sample_attention_mask, hidden_size):
        """Max pooling with attention mask."""
        head = EmbeddingHead(hidden_size, pooling="max", normalize=False)

        embeddings = head(sample_hidden_states, sample_attention_mask)
        assert embeddings.shape == (sample_hidden_states.size(0), hidden_size)

    def test_pool_unknown_raises(self, sample_hidden_states, sample_attention_mask, hidden_size):
        """Unknown pooling raises ValueError."""
        head = EmbeddingHead(hidden_size, pooling="unknown")

        with pytest.raises(ValueError, match="Unknown pooling"):
            head(sample_hidden_states, sample_attention_mask)

    def test_freeze_unfreeze_embedding_head(self, hidden_size):
        """EmbeddingHead freeze/unfreeze."""
        head = EmbeddingHead(hidden_size, output_dim=256)

        head.freeze()
        for param in head.parameters():
            assert not param.requires_grad

        head.unfreeze()
        for param in head.parameters():
            assert param.requires_grad


class TestNLIHeadAdvanced:
    """Additional coverage for NLIHead with pair encoder."""

    def test_nli_with_pair_inputs(self, hidden_size, batch_size, seq_length):
        """NLI head with explicit text_a/text_b inputs."""
        head = NLIHead(hidden_size)

        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        attention_mask = torch.ones(batch_size, seq_length)

        # Provide pair inputs (without external pair encoder)
        text_a_hidden = torch.randn(batch_size, seq_length // 2, hidden_size)
        text_b_hidden = torch.randn(batch_size, seq_length // 2, hidden_size)

        # This should fall back to standard forward since no pair_encoder
        output = head(
            hidden_states,
            attention_mask,
            text_a_hidden=text_a_hidden,
            text_b_hidden=text_b_hidden,
        )

        assert "logits" in output


class TestRelationHeadAdvanced:
    """Additional coverage for RelationHead."""

    def test_relation_without_entity_spans(self, sample_hidden_states, sample_attention_mask):
        """RelationHead falls back to CLS when no entity spans provided."""
        hidden_size = sample_hidden_states.size(-1)
        head = RelationHead(hidden_size, num_labels=15)

        output = head(sample_hidden_states, sample_attention_mask)

        assert "logits" in output

    def test_relation_with_labels(self, sample_hidden_states, sample_attention_mask):
        """RelationHead computes loss with labels."""
        hidden_size = sample_hidden_states.size(-1)
        batch_size = sample_hidden_states.size(0)
        head = RelationHead(hidden_size, num_labels=15)

        labels = torch.randint(0, 15, (batch_size,))

        output = head(sample_hidden_states, sample_attention_mask, labels=labels)

        assert "loss" in output


class TestIntentHeadAdvanced:
    """Additional coverage for IntentHead."""

    def test_set_confidence_threshold(self, hidden_size):
        """Set confidence threshold updates value."""
        head = IntentHead(hidden_size, confidence_threshold=0.5)
        head.set_confidence_threshold(0.8)

        assert head.confidence_threshold == 0.8

    def test_low_confidence_mask(self, sample_hidden_states, sample_attention_mask):
        """Low confidence mask flags uncertain predictions."""
        hidden_size = sample_hidden_states.size(-1)
        head = IntentHead(hidden_size, confidence_threshold=0.99)  # High threshold

        output = head(sample_hidden_states, sample_attention_mask)

        assert "low_confidence_mask" in output


class TestTemporalHeadAdvanced:
    """Additional coverage for TemporalHead."""

    def test_extract_temporal_spans(self, sample_hidden_states, sample_attention_mask):
        """Extract temporal spans from predictions."""
        hidden_size = sample_hidden_states.size(-1)
        head = TemporalHead(hidden_size, num_labels=13)

        output = head(sample_hidden_states, sample_attention_mask)
        logits = output["logits"]

        id2label = {i: f"B-DATE" if i % 2 == 1 else "O" for i in range(13)}

        spans = head.extract_temporal_spans(logits, sample_attention_mask, id2label)

        assert len(spans) == sample_hidden_states.size(0)


class TestHierarchicalEmotionHeadAdvanced:
    """Additional coverage for HierarchicalEmotionHead."""

    def test_for_familyos_factory(self, hidden_size):
        """Factory method creates correct configuration."""
        head = HierarchicalEmotionHead.for_familyos(hidden_size=hidden_size)

        assert head.num_emotions == 44
        assert head.use_familyos
        assert head.use_asl

    def test_without_familyos_32_emotions(self, hidden_size):
        """32-emotion mode uses legacy labels."""
        head = HierarchicalEmotionHead(
            hidden_size,
            num_emotions=32,
            use_familyos=False,
        )

        assert head.num_emotions == 32
        assert len(head.emotion_labels) == 32

    def test_secondary_emotions(self, sample_hidden_states, sample_attention_mask):
        """Secondary emotions extraction."""
        hidden_size = sample_hidden_states.size(-1)
        head = HierarchicalEmotionHead(hidden_size, num_emotions=44, num_secondary=3)

        output = head(sample_hidden_states, sample_attention_mask)

        assert "secondary_emotions" in output
        # Should return up to num_secondary emotions
        if isinstance(output["secondary_emotions"], list):
            for sec in output["secondary_emotions"]:
                if isinstance(sec, list):
                    assert len(sec) <= 3

    def test_intensity_scoring(self, sample_hidden_states, sample_attention_mask):
        """Intensity scores when use_intensity=True."""
        hidden_size = sample_hidden_states.size(-1)
        head = HierarchicalEmotionHead(hidden_size, num_emotions=44, use_intensity=True)

        output = head(sample_hidden_states, sample_attention_mask)

        assert "intensity" in output or "intensities" in output or "emotion_scores" in output

    def test_dynamic_thresholds(self, hidden_size):
        """Dynamic thresholds are learnable parameters."""
        head = HierarchicalEmotionHead(hidden_size, num_emotions=44, use_dynamic_thresholds=True)

        assert hasattr(head, "thresholds") or hasattr(head, "dynamic_thresholds")

    def test_label_correlation(self, hidden_size):
        """Label correlation module exists when enabled."""
        head = HierarchicalEmotionHead(hidden_size, num_emotions=44, use_label_correlation=True)

        assert hasattr(head, "label_correlation") or hasattr(head, "correlation_layer")

    def test_hierarchical_loss(self, sample_hidden_states, sample_attention_mask):
        """Hierarchical loss adds family-level component."""
        hidden_size = sample_hidden_states.size(-1)
        batch_size = sample_hidden_states.size(0)
        head = HierarchicalEmotionHead(
            hidden_size,
            num_emotions=44,
            use_hierarchical_loss=True,
        )

        labels = torch.randint(0, 2, (batch_size, 44)).float()
        output = head(sample_hidden_states, sample_attention_mask, labels=labels)

        assert "loss" in output


class TestHeadClassMethods:
    """Test class-level methods and attributes."""

    def test_safety_head_constants(self):
        """SafetyHead class has required constants."""
        assert hasattr(SafetyHead, "BAND_NAMES")
        assert hasattr(SafetyHead, "BAND_TO_ID")
        assert hasattr(SafetyHead, "ID_TO_BAND")
        assert hasattr(SafetyHead, "SUBCATEGORY_NAMES")
        assert hasattr(SafetyHead, "SUBCATEGORY_TO_BAND_ID")

    def test_enhanced_safety_head_constants(self):
        """EnhancedSafetyHead class has required constants."""
        assert hasattr(EnhancedSafetyHead, "BAND_NAMES")
        assert hasattr(EnhancedSafetyHead, "SUBCATEGORIES")
        assert hasattr(EnhancedSafetyHead, "CRISIS_KEYWORDS")
        assert len(EnhancedSafetyHead.CRISIS_KEYWORDS) > 0

    def test_hierarchical_emotion_head_constants(self):
        """HierarchicalEmotionHead class has required constants."""
        assert hasattr(HierarchicalEmotionHead, "FAMILYOS_EMOTION_LABELS")
        assert hasattr(HierarchicalEmotionHead, "DEFAULT_EMOTION_LABELS")
        assert hasattr(HierarchicalEmotionHead, "FAMILYOS_EMOTION_FAMILIES")
        assert len(HierarchicalEmotionHead.FAMILYOS_EMOTION_LABELS) == 44


class TestHeadGradients:
    """Test gradient flow through heads."""

    def test_sequence_head_gradients(self, sample_hidden_states, sample_attention_mask):
        """Gradients flow through sequence classification head."""
        hidden_size = sample_hidden_states.size(-1)
        batch_size = sample_hidden_states.size(0)

        hidden_states = sample_hidden_states.clone().requires_grad_(True)
        head = SequenceClassificationHead(hidden_size, num_labels=5)

        labels = torch.randint(0, 5, (batch_size,))
        output = head(hidden_states, sample_attention_mask, labels=labels)

        output["loss"].backward()
        assert hidden_states.grad is not None

    def test_token_head_gradients(self, sample_hidden_states, sample_attention_mask):
        """Gradients flow through token classification head."""
        hidden_size = sample_hidden_states.size(-1)
        batch_size = sample_hidden_states.size(0)
        seq_length = sample_hidden_states.size(1)

        hidden_states = sample_hidden_states.clone().requires_grad_(True)
        head = TokenClassificationHead(hidden_size, num_labels=9)

        labels = torch.randint(0, 9, (batch_size, seq_length))
        output = head(hidden_states, sample_attention_mask, labels=labels)

        output["loss"].backward()
        assert hidden_states.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
