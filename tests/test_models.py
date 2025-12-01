"""
Tests for ModernBERT Multi-Task Model

Test coverage for:
    - Model initialization
    - Forward pass for each task
    - Loss computation
    - Head freezing/unfreezing
    - Checkpoint save/load
    - Epic 5.0 enhancements (poolers, adapters, pair_encoder)
"""

import torch

from modeling_studio.data.labels import Capability

# =============================================================================
# Epic 5.0 Component Tests
# =============================================================================


class TestEpic50Adapters:
    """Tests for Epic 5.0 adapter components."""

    def test_bottleneck_adapter(self):
        """Test BottleneckAdapter initialization and forward pass."""
        from modeling_studio.models.adapters import BottleneckAdapter

        adapter = BottleneckAdapter(
            hidden_size=768,
            bottleneck_size=64,
            activation="gelu",
            dropout=0.1,
        )

        x = torch.randn(2, 10, 768)
        out = adapter(x)

        assert out.shape == x.shape
        # Residual connection should preserve identity initially
        assert torch.allclose(out, x, atol=0.1)

    def test_task_group_adapter(self):
        """Test TaskGroupAdapter with multiple task groups."""
        from modeling_studio.models.adapters import TaskGroupAdapter

        adapter = TaskGroupAdapter(
            hidden_size=768,
            task_groups=["token_tasks", "sequence_tasks", "pair_tasks"],
            bottleneck_size=64,
        )

        x = torch.randn(2, 10, 768)

        # Test each task group
        for group in ["token_tasks", "sequence_tasks", "pair_tasks"]:
            out = adapter(x, task_group=group)
            assert out.shape == x.shape

    def test_lora_adapter(self):
        """Test LoRA adapter."""
        from modeling_studio.models.adapters import LoRAAdapter

        adapter = LoRAAdapter(
            in_features=768,
            out_features=768,
            r=8,  # LoRA rank
            alpha=16,  # LoRA alpha
        )

        x = torch.randn(2, 10, 768)
        out = adapter(x)

        assert out.shape == x.shape


class TestEpic50PairEncoder:
    """Tests for Epic 5.0 pair encoder components."""

    def test_cross_attention_pair_encoder(self):
        """Test CrossAttentionPairEncoder."""
        from modeling_studio.models.pair_encoder import CrossAttentionPairEncoder

        encoder = CrossAttentionPairEncoder(
            hidden_size=768,
            num_heads=8,
            num_layers=2,
            pooling_strategy="attention",
        )

        text_a = torch.randn(2, 10, 768)
        text_b = torch.randn(2, 8, 768)
        mask_a = torch.ones(2, 10)
        mask_b = torch.ones(2, 8)

        out = encoder(text_a, text_b, mask_a, mask_b)

        # Output should be combined representation
        assert out.shape == (2, 768)

    def test_concat_pair_encoder(self):
        """Test ConcatPairEncoder."""
        from modeling_studio.models.pair_encoder import ConcatPairEncoder

        encoder = ConcatPairEncoder(
            hidden_size=768,
            pooling_strategy="mean",
            output_size=768,
        )

        text_a = torch.randn(2, 10, 768)
        text_b = torch.randn(2, 8, 768)
        mask_a = torch.ones(2, 10)
        mask_b = torch.ones(2, 8)

        out = encoder(text_a, text_b, mask_a, mask_b)

        assert out.shape == (2, 768)


class TestEpic50Poolers:
    """Tests for Epic 5.0 pooler components."""

    def test_cls_mean_pooler(self):
        """Test CLSMeanPooler."""
        from modeling_studio.models.poolers import CLSMeanPooler

        pooler = CLSMeanPooler(hidden_size=768, dropout=0.1)

        hidden_states = torch.randn(2, 10, 768)
        attention_mask = torch.ones(2, 10)

        out = pooler(hidden_states, attention_mask)

        assert out.shape == (2, 768)

    def test_attention_pooler(self):
        """Test AttentionPooler."""
        from modeling_studio.models.poolers import AttentionPooler

        pooler = AttentionPooler(hidden_size=768)

        hidden_states = torch.randn(2, 10, 768)
        attention_mask = torch.ones(2, 10)

        out = pooler(hidden_states, attention_mask)

        assert out.shape == (2, 768)

    def test_get_pooler(self):
        """Test pooler factory function."""
        from modeling_studio.models.poolers import get_pooler

        pooler = get_pooler("cls_mean", hidden_size=768)
        assert pooler is not None

        pooler = get_pooler("attention", hidden_size=768)
        assert pooler is not None


class TestEpic50ModelIntegration:
    """Tests for Epic 5.0 integration in ModernBertMultiTaskModel."""

    def test_model_with_adapters_init(self):
        """Test model initialization with adapters enabled."""
        from transformers import AutoConfig

        from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

        config = AutoConfig.from_pretrained("answerdotai/ModernBERT-base")

        model = ModernBertMultiTaskModel(
            config=config,
            capabilities=[Capability.SENTIMENT, Capability.NER_GENERAL],
            use_adapters=True,
            adapter_bottleneck_size=64,
        )

        # Check adapters were created
        assert model.task_adapters is not None
        assert model._use_adapters is True

    def test_model_with_pair_encoder_init(self):
        """Test model initialization with pair encoder enabled."""
        from transformers import AutoConfig

        from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

        config = AutoConfig.from_pretrained("answerdotai/ModernBERT-base")

        model = ModernBertMultiTaskModel(
            config=config,
            capabilities=[Capability.NLI, Capability.RELATION],
            use_pair_encoder=True,
            pair_encoder_num_layers=2,
        )

        # Check pair encoder was created
        assert model.pair_encoder is not None
        assert model._use_pair_encoder is True

    def test_model_with_shared_pooler_init(self):
        """Test model initialization with shared pooler."""
        from transformers import AutoConfig

        from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

        config = AutoConfig.from_pretrained("answerdotai/ModernBERT-base")

        model = ModernBertMultiTaskModel(
            config=config,
            capabilities=[Capability.SENTIMENT],
            shared_pooler="cls_mean",
        )

        # Check shared pooler was created
        assert model.shared_pooler is not None
        assert model._shared_pooler_type == "cls_mean"

    def test_model_backward_compatibility(self):
        """Test that model works without Epic 5.0 features (backward compat)."""
        from transformers import AutoConfig

        from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

        config = AutoConfig.from_pretrained("answerdotai/ModernBERT-base")

        # No Epic 5.0 features
        model = ModernBertMultiTaskModel(
            config=config,
            capabilities=[Capability.SENTIMENT],
        )

        # Check no Epic 5.0 components
        assert model.task_adapters is None
        assert model.pair_encoder is None
        assert model.shared_pooler is None


class TestEpic50HeadEnhancements:
    """Tests for Epic 5.0 head enhancements (external pooler, pair encoder)."""

    def test_sequence_classification_with_external_pooler(self):
        """Test SequenceClassificationHead with external pooler (acceptance criteria)."""
        from modeling_studio.models.heads import SequenceClassificationHead
        from modeling_studio.models.poolers import CLSMeanPooler

        # Head with external pooler
        pooler = CLSMeanPooler(hidden_size=768)
        head = SequenceClassificationHead(
            hidden_size=768,
            num_labels=5,
            external_pooler=pooler,  # Use shared pooler
        )

        hidden_states = torch.randn(2, 128, 768)
        attention_mask = torch.ones(2, 128)
        output = head(hidden_states, attention_mask)

        assert output["logits"].shape == (2, 5)
        assert head._use_external_pooler is True
        print("✅ Head with external pooler works correctly")

    def test_sequence_classification_without_external_pooler(self):
        """Test backward compatibility - head works without external pooler."""
        from modeling_studio.models.heads import SequenceClassificationHead

        head = SequenceClassificationHead(
            hidden_size=768,
            num_labels=5,
            pooling="mean",
        )

        hidden_states = torch.randn(2, 128, 768)
        attention_mask = torch.ones(2, 128)
        output = head(hidden_states, attention_mask)

        assert output["logits"].shape == (2, 5)
        assert head._use_external_pooler is False
        assert head.external_pooler is None

    def test_sequence_classification_with_attention_pooler(self):
        """Test SequenceClassificationHead with AttentionPooler."""
        from modeling_studio.models.heads import SequenceClassificationHead
        from modeling_studio.models.poolers import AttentionPooler

        pooler = AttentionPooler(hidden_size=768)
        head = SequenceClassificationHead(
            hidden_size=768,
            num_labels=3,
            external_pooler=pooler,
        )

        hidden_states = torch.randn(4, 64, 768)
        attention_mask = torch.ones(4, 64)
        output = head(hidden_states, attention_mask)

        assert output["logits"].shape == (4, 3)

    def test_nli_head_with_pair_encoder(self):
        """Test NLIHead with CrossAttentionPairEncoder."""
        from modeling_studio.models.heads import NLIHead
        from modeling_studio.models.pair_encoder import CrossAttentionPairEncoder

        pair_encoder = CrossAttentionPairEncoder(
            hidden_size=768,
            num_heads=8,
            num_layers=1,
        )
        head = NLIHead(
            hidden_size=768,
            pair_encoder=pair_encoder,
        )

        # Test with pair encoding path
        text_a = torch.randn(2, 32, 768)
        text_b = torch.randn(2, 24, 768)
        mask_a = torch.ones(2, 32)
        mask_b = torch.ones(2, 24)

        output = head(
            hidden_states=text_a,  # Not used when pair encoding
            text_a_hidden=text_a,
            text_b_hidden=text_b,
            text_a_mask=mask_a,
            text_b_mask=mask_b,
        )

        assert output["logits"].shape == (2, 3)  # 3 NLI classes
        assert head._use_pair_encoder is True

    def test_nli_head_backward_compatibility(self):
        """Test NLIHead works without pair encoder (backward compat)."""
        from modeling_studio.models.heads import NLIHead

        head = NLIHead(hidden_size=768)

        hidden_states = torch.randn(2, 128, 768)
        attention_mask = torch.ones(2, 128)
        output = head(hidden_states, attention_mask)

        assert output["logits"].shape == (2, 3)
        assert head._use_pair_encoder is False

    def test_relation_head_with_pair_encoder(self):
        """Test RelationHead with CrossAttentionPairEncoder."""
        from modeling_studio.models.heads import RelationHead
        from modeling_studio.models.pair_encoder import CrossAttentionPairEncoder

        pair_encoder = CrossAttentionPairEncoder(
            hidden_size=768,
            num_heads=8,
            num_layers=1,
        )
        head = RelationHead(
            hidden_size=768,
            num_labels=15,
            pair_encoder=pair_encoder,
        )

        # Test with pair encoding path
        entity1_ctx = torch.randn(2, 16, 768)
        entity2_ctx = torch.randn(2, 16, 768)
        mask1 = torch.ones(2, 16)
        mask2 = torch.ones(2, 16)

        hidden_states = torch.randn(2, 128, 768)  # Not used when pair encoding

        output = head(
            hidden_states=hidden_states,
            entity1_context=entity1_ctx,
            entity2_context=entity2_ctx,
            entity1_mask=mask1,
            entity2_mask=mask2,
        )

        assert output["logits"].shape == (2, 15)
        assert head._use_pair_encoder is True

    def test_relation_head_backward_compatibility(self):
        """Test RelationHead works without pair encoder (backward compat)."""
        from modeling_studio.models.heads import RelationHead

        head = RelationHead(hidden_size=768, num_labels=15)

        hidden_states = torch.randn(2, 128, 768)
        entity1_start = torch.tensor([5, 10])
        entity2_start = torch.tensor([20, 30])

        output = head(
            hidden_states,
            entity1_start=entity1_start,
            entity2_start=entity2_start,
        )

        assert output["logits"].shape == (2, 15)
        assert head._use_pair_encoder is False


# TODO: Implement test fixtures
#   - sample_model: Small model for testing
#   - sample_tokenizer: Tokenizer instance
#   - sample_batch: Sample input batch


class TestModernBertMultiTaskModel:
    """Tests for the main multi-task model."""

    # TODO: test_model_initialization
    #   - Load from pretrained
    #   - Initialize heads from config
    #   - Verify head dimensions

    # TODO: test_forward_classification
    #   - Run forward pass for sentiment
    #   - Verify output shape
    #   - Verify loss computation

    # TODO: test_forward_ner
    #   - Run forward pass for NER
    #   - Verify per-token outputs
    #   - Verify label alignment

    # TODO: test_forward_embedding
    #   - Run forward pass for embedding
    #   - Verify output dimension
    #   - Verify normalization

    # TODO: test_forward_nli
    #   - Run forward pass with pairs
    #   - Verify output shape

    # TODO: test_multi_task_forward
    #   - Run multiple tasks in one call
    #   - Verify all outputs present

    # TODO: test_head_freezing
    #   - Freeze specific heads
    #   - Verify gradients don't flow
    #   - Unfreeze and verify gradients

    # TODO: test_save_load
    #   - Save model checkpoint
    #   - Load checkpoint
    #   - Verify outputs match

    pass


class TestTaskHeads:
    """Tests for individual task heads."""

    # TODO: test_sequence_classification_head
    # TODO: test_token_classification_head
    # TODO: test_embedding_head
    # TODO: test_nli_head
    # TODO: test_safety_head_with_calibration

    pass


class TestPoolers:
    """Tests for pooling strategies."""

    # TODO: test_cls_pooler
    # TODO: test_mean_pooler
    # TODO: test_max_pooler
    # TODO: test_pooler_with_attention_mask

    pass


class TestLosses:
    """Tests for loss functions."""

    # TODO: test_focal_loss
    # TODO: test_multiple_negatives_ranking_loss
    # TODO: test_multi_task_loss_aggregation

    pass
    """Tests for loss functions."""

    # TODO: test_focal_loss
    # TODO: test_multiple_negatives_ranking_loss
    # TODO: test_multi_task_loss_aggregation

    pass
