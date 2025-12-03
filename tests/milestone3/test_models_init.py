"""
Milestone 3: Models Tests
Issue 3.1.1: models/__init__.py

Tests for:
- Model class exports: ModernBertMultiTaskModel, MultiTaskOutput
- Mapping exports: CAPABILITY_TO_HEAD_TYPE
- Head class exports: BaseHead, SequenceClassificationHead, TokenClassificationHead, etc.
- Pooler class exports: CLSPooler, MeanPooler, AttentionPooler, etc.
- Adapter class exports: BottleneckAdapter, TaskGroupAdapter, LoRAAdapter, etc.
- Pair encoder exports: CrossAttentionPairEncoder, ConcatPairEncoder, etc.
"""

# =============================================================================
# Model Class Exports Tests
# =============================================================================


class TestModelClassExports:
    """Test ModernBertMultiTaskModel is exported."""

    def test_modernbert_multitask_model_exported(self):
        """ModernBertMultiTaskModel should be importable from models."""
        from modeling_studio.models import ModernBertMultiTaskModel

        assert ModernBertMultiTaskModel is not None

    def test_modernbert_multitask_model_is_class(self):
        """ModernBertMultiTaskModel should be a class."""
        from modeling_studio.models import ModernBertMultiTaskModel

        assert isinstance(ModernBertMultiTaskModel, type)

    def test_modernbert_in_all(self):
        """ModernBertMultiTaskModel should be in __all__."""
        from modeling_studio import models

        assert "ModernBertMultiTaskModel" in models.__all__


class TestMultiTaskOutputExported:
    """Test MultiTaskOutput is exported."""

    def test_multi_task_output_exported(self):
        """MultiTaskOutput should be importable from models."""
        from modeling_studio.models import MultiTaskOutput

        assert MultiTaskOutput is not None

    def test_multi_task_output_is_class(self):
        """MultiTaskOutput should be a class."""
        from modeling_studio.models import MultiTaskOutput

        assert isinstance(MultiTaskOutput, type)

    def test_multi_task_output_in_all(self):
        """MultiTaskOutput should be in __all__."""
        from modeling_studio import models

        assert "MultiTaskOutput" in models.__all__


class TestCapabilityToHeadTypeExported:
    """Test CAPABILITY_TO_HEAD_TYPE mapping is exported."""

    def test_capability_to_head_type_exported(self):
        """CAPABILITY_TO_HEAD_TYPE should be importable from models."""
        from modeling_studio.models import CAPABILITY_TO_HEAD_TYPE

        assert CAPABILITY_TO_HEAD_TYPE is not None

    def test_capability_to_head_type_is_dict(self):
        """CAPABILITY_TO_HEAD_TYPE should be a dictionary."""
        from modeling_studio.models import CAPABILITY_TO_HEAD_TYPE

        assert isinstance(CAPABILITY_TO_HEAD_TYPE, dict)

    def test_capability_to_head_type_in_all(self):
        """CAPABILITY_TO_HEAD_TYPE should be in __all__."""
        from modeling_studio import models

        assert "CAPABILITY_TO_HEAD_TYPE" in models.__all__

    def test_capability_to_head_type_has_12_entries(self):
        """CAPABILITY_TO_HEAD_TYPE should have 12 capability mappings."""
        from modeling_studio.models import CAPABILITY_TO_HEAD_TYPE

        assert len(CAPABILITY_TO_HEAD_TYPE) == 12


# =============================================================================
# Head Class Exports Tests
# =============================================================================


class TestAllHeadClassesExported:
    """Test all head classes are exported."""

    def test_base_head_exported(self):
        """BaseHead should be importable from models."""
        from modeling_studio.models import BaseHead

        assert BaseHead is not None

    def test_sequence_classification_head_exported(self):
        """SequenceClassificationHead should be importable from models."""
        from modeling_studio.models import SequenceClassificationHead

        assert SequenceClassificationHead is not None

    def test_token_classification_head_exported(self):
        """TokenClassificationHead should be importable from models."""
        from modeling_studio.models import TokenClassificationHead

        assert TokenClassificationHead is not None

    def test_embedding_head_exported(self):
        """EmbeddingHead should be importable from models."""
        from modeling_studio.models import EmbeddingHead

        assert EmbeddingHead is not None

    def test_nli_head_exported(self):
        """NLIHead should be importable from models."""
        from modeling_studio.models import NLIHead

        assert NLIHead is not None

    def test_safety_head_exported(self):
        """SafetyHead should be importable from models."""
        from modeling_studio.models import SafetyHead

        assert SafetyHead is not None

    def test_relation_head_exported(self):
        """RelationHead should be importable from models."""
        from modeling_studio.models import RelationHead

        assert RelationHead is not None

    def test_intent_head_exported(self):
        """IntentHead should be importable from models."""
        from modeling_studio.models import IntentHead

        assert IntentHead is not None

    def test_temporal_head_exported(self):
        """TemporalHead should be importable from models."""
        from modeling_studio.models import TemporalHead

        assert TemporalHead is not None

    def test_head_classes_in_all(self):
        """All head classes should be in __all__."""
        from modeling_studio import models

        assert "BaseHead" in models.__all__
        assert "SequenceClassificationHead" in models.__all__
        assert "TokenClassificationHead" in models.__all__
        assert "EmbeddingHead" in models.__all__
        assert "NLIHead" in models.__all__
        assert "SafetyHead" in models.__all__
        assert "RelationHead" in models.__all__
        assert "IntentHead" in models.__all__
        assert "TemporalHead" in models.__all__


# =============================================================================
# Pooler Class Exports Tests
# =============================================================================


class TestPoolerClassesExported:
    """Test pooler classes are exported."""

    def test_cls_pooler_exported(self):
        """CLSPooler should be importable from models."""
        from modeling_studio.models import CLSPooler

        assert CLSPooler is not None

    def test_mean_pooler_exported(self):
        """MeanPooler should be importable from models."""
        from modeling_studio.models import MeanPooler

        assert MeanPooler is not None

    def test_attention_pooler_exported(self):
        """AttentionPooler should be importable from models."""
        from modeling_studio.models import AttentionPooler

        assert AttentionPooler is not None

    def test_cls_mean_pooler_exported(self):
        """CLSMeanPooler should be importable from models."""
        from modeling_studio.models import CLSMeanPooler

        assert CLSMeanPooler is not None

    def test_get_pooler_exported(self):
        """get_pooler factory function should be importable from models."""
        from modeling_studio.models import get_pooler

        assert get_pooler is not None
        assert callable(get_pooler)

    def test_pooler_classes_in_all(self):
        """All pooler classes should be in __all__."""
        from modeling_studio import models

        assert "CLSPooler" in models.__all__
        assert "MeanPooler" in models.__all__
        assert "AttentionPooler" in models.__all__
        assert "CLSMeanPooler" in models.__all__
        assert "get_pooler" in models.__all__


# =============================================================================
# Adapter Class Exports Tests
# =============================================================================


class TestAdapterClassesExported:
    """Test adapter classes are exported."""

    def test_bottleneck_adapter_exported(self):
        """BottleneckAdapter should be importable from models."""
        from modeling_studio.models import BottleneckAdapter

        assert BottleneckAdapter is not None

    def test_task_group_adapter_exported(self):
        """TaskGroupAdapter should be importable from models."""
        from modeling_studio.models import TaskGroupAdapter

        assert TaskGroupAdapter is not None

    def test_lora_adapter_exported(self):
        """LoRAAdapter should be importable from models."""
        from modeling_studio.models import LoRAAdapter

        assert LoRAAdapter is not None

    def test_parallel_adapter_exported(self):
        """ParallelAdapter should be importable from models."""
        from modeling_studio.models import ParallelAdapter

        assert ParallelAdapter is not None

    def test_adapter_config_exported(self):
        """AdapterConfig should be importable from models."""
        from modeling_studio.models import AdapterConfig

        assert AdapterConfig is not None

    def test_task_group_config_exported(self):
        """TaskGroupConfig should be importable from models."""
        from modeling_studio.models import TaskGroupConfig

        assert TaskGroupConfig is not None

    def test_adapted_linear_exported(self):
        """AdaptedLinear should be importable from models."""
        from modeling_studio.models import AdaptedLinear

        assert AdaptedLinear is not None

    def test_create_adapter_exported(self):
        """create_adapter factory function should be importable from models."""
        from modeling_studio.models import create_adapter

        assert create_adapter is not None
        assert callable(create_adapter)

    def test_adapter_classes_in_all(self):
        """All adapter classes should be in __all__."""
        from modeling_studio import models

        assert "AdapterConfig" in models.__all__
        assert "TaskGroupConfig" in models.__all__
        assert "BottleneckAdapter" in models.__all__
        assert "TaskGroupAdapter" in models.__all__
        assert "ParallelAdapter" in models.__all__
        assert "LoRAAdapter" in models.__all__
        assert "AdaptedLinear" in models.__all__
        assert "create_adapter" in models.__all__


# =============================================================================
# Pair Encoder Exports Tests
# =============================================================================


class TestPairEncoderExported:
    """Test pair encoder classes are exported."""

    def test_cross_attention_pair_encoder_exported(self):
        """CrossAttentionPairEncoder should be importable from models."""
        from modeling_studio.models import CrossAttentionPairEncoder

        assert CrossAttentionPairEncoder is not None

    def test_concat_pair_encoder_exported(self):
        """ConcatPairEncoder should be importable from models."""
        from modeling_studio.models import ConcatPairEncoder

        assert ConcatPairEncoder is not None

    def test_cross_attention_layer_exported(self):
        """CrossAttentionLayer should be importable from models."""
        from modeling_studio.models import CrossAttentionLayer

        assert CrossAttentionLayer is not None

    def test_bidirectional_cross_attention_block_exported(self):
        """BidirectionalCrossAttentionBlock should be importable from models."""
        from modeling_studio.models import BidirectionalCrossAttentionBlock

        assert BidirectionalCrossAttentionBlock is not None

    def test_attention_pooling_exported(self):
        """AttentionPooling should be importable from models."""
        from modeling_studio.models import AttentionPooling

        assert AttentionPooling is not None

    def test_pair_encoder_config_exported(self):
        """PairEncoderConfig should be importable from models."""
        from modeling_studio.models import PairEncoderConfig

        assert PairEncoderConfig is not None

    def test_create_pair_encoder_exported(self):
        """create_pair_encoder factory function should be importable from models."""
        from modeling_studio.models import create_pair_encoder

        assert create_pair_encoder is not None
        assert callable(create_pair_encoder)

    def test_pair_encoder_classes_in_all(self):
        """All pair encoder classes should be in __all__."""
        from modeling_studio import models

        assert "PairEncoderConfig" in models.__all__
        assert "CrossAttentionPairEncoder" in models.__all__
        assert "CrossAttentionLayer" in models.__all__
        assert "BidirectionalCrossAttentionBlock" in models.__all__
        assert "AttentionPooling" in models.__all__
        assert "ConcatPairEncoder" in models.__all__
        assert "create_pair_encoder" in models.__all__


# =============================================================================
# Module Structure Tests
# =============================================================================


class TestModuleStructure:
    """Test the module structure is correct."""

    def test_all_exports_defined(self):
        """__all__ should be defined with all public APIs."""
        from modeling_studio import models

        assert hasattr(models, "__all__")
        assert len(models.__all__) > 0

    def test_get_problem_type_exported(self):
        """get_problem_type function should be exported."""
        from modeling_studio.models import get_problem_type

        assert get_problem_type is not None
        assert callable(get_problem_type)

    def test_get_problem_type_in_all(self):
        """get_problem_type should be in __all__."""
        from modeling_studio import models

        assert "get_problem_type" in models.__all__
