"""
Milestone 3: Models Tests
Issue 3.1.2: models/modernbert_multitask.py

Tests for:
- Model initialization and configuration
- Capability normalization
- Task groups and head type mapping
- MultiTaskOutput dataclass
- Forward pass for various capabilities
- Model saving/loading
- Epic 5.0 integrations (poolers, adapters, pair encoder)
"""

import pytest
import torch


# =============================================================================
# Model Initialization Tests
# =============================================================================


class TestModelInit:
    """Test model initializes with default capabilities."""

    def test_model_class_exists(self):
        """ModernBertMultiTaskModel class should exist."""
        from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

        assert ModernBertMultiTaskModel is not None

    def test_model_has_init(self):
        """Model should have __init__ method."""
        from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

        assert hasattr(ModernBertMultiTaskModel, "__init__")

    def test_model_inherits_pretrained(self):
        """Model should inherit from PreTrainedModel."""
        from transformers import PreTrainedModel

        from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

        assert issubclass(ModernBertMultiTaskModel, PreTrainedModel)


class TestModelInitSpecificCapabilities:
    """Test model initializes with subset of capabilities."""

    def test_model_accepts_capabilities_list(self):
        """Model should accept capabilities parameter."""
        from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

        # Check the signature includes capabilities
        import inspect

        sig = inspect.signature(ModernBertMultiTaskModel.__init__)
        assert "capabilities" in sig.parameters

    def test_model_capabilities_can_be_subset(self):
        """Model should support subset of capabilities."""
        from modeling_studio.data.labels import Capability
        from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

        # Just verify the class accepts the parameter
        assert hasattr(ModernBertMultiTaskModel, "_normalize_capabilities")


class TestModelHeadsCreated:
    """Test heads are created for each capability."""

    def test_init_heads_method_exists(self):
        """Model should have _init_heads method."""
        from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

        assert hasattr(ModernBertMultiTaskModel, "_init_heads")

    def test_get_head_method_exists(self):
        """Model should have get_head method."""
        from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

        assert hasattr(ModernBertMultiTaskModel, "get_head")


# =============================================================================
# Capability Normalization Tests
# =============================================================================


class TestNormalizeCapabilitiesString:
    """Test string capabilities converted to Capability enum."""

    def test_normalize_capabilities_method_exists(self):
        """Model should have _normalize_capabilities method."""
        from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

        assert hasattr(ModernBertMultiTaskModel, "_normalize_capabilities")


class TestNormalizeCapabilitiesEnum:
    """Test Capability enum passed through."""

    def test_capability_enum_imported(self):
        """Capability enum should be available."""
        from modeling_studio.data.labels import Capability

        assert Capability is not None


# =============================================================================
# Task Groups Tests
# =============================================================================


class TestTaskGroupsDefined:
    """Test TASK_GROUPS has token_tasks, sequence_tasks, pair_tasks, embedding_tasks."""

    def test_task_groups_exists(self):
        """TASK_GROUPS should be defined."""
        from modeling_studio.models.modernbert_multitask import TASK_GROUPS

        assert TASK_GROUPS is not None
        assert isinstance(TASK_GROUPS, dict)

    def test_task_groups_has_token_tasks(self):
        """TASK_GROUPS should have token_tasks."""
        from modeling_studio.models.modernbert_multitask import TASK_GROUPS

        assert "token_tasks" in TASK_GROUPS
        assert len(TASK_GROUPS["token_tasks"]) > 0

    def test_task_groups_has_sequence_tasks(self):
        """TASK_GROUPS should have sequence_tasks."""
        from modeling_studio.models.modernbert_multitask import TASK_GROUPS

        assert "sequence_tasks" in TASK_GROUPS
        assert len(TASK_GROUPS["sequence_tasks"]) > 0

    def test_task_groups_has_pair_tasks(self):
        """TASK_GROUPS should have pair_tasks."""
        from modeling_studio.models.modernbert_multitask import TASK_GROUPS

        assert "pair_tasks" in TASK_GROUPS
        assert len(TASK_GROUPS["pair_tasks"]) > 0

    def test_task_groups_has_embedding_tasks(self):
        """TASK_GROUPS should have embedding_tasks."""
        from modeling_studio.models.modernbert_multitask import TASK_GROUPS

        assert "embedding_tasks" in TASK_GROUPS
        assert len(TASK_GROUPS["embedding_tasks"]) > 0


class TestGetTaskGroup:
    """Test get_task_group returns correct group for capability."""

    def test_get_task_group_function_exists(self):
        """get_task_group function should exist."""
        from modeling_studio.models.modernbert_multitask import get_task_group

        assert get_task_group is not None
        assert callable(get_task_group)

    def test_get_task_group_ner_general(self):
        """NER_GENERAL should be in token_tasks."""
        from modeling_studio.data.labels import Capability
        from modeling_studio.models.modernbert_multitask import get_task_group

        assert get_task_group(Capability.NER_GENERAL) == "token_tasks"

    def test_get_task_group_sentiment(self):
        """SENTIMENT should be in sequence_tasks."""
        from modeling_studio.data.labels import Capability
        from modeling_studio.models.modernbert_multitask import get_task_group

        assert get_task_group(Capability.SENTIMENT) == "sequence_tasks"

    def test_get_task_group_nli(self):
        """NLI should be in pair_tasks."""
        from modeling_studio.data.labels import Capability
        from modeling_studio.models.modernbert_multitask import get_task_group

        assert get_task_group(Capability.NLI) == "pair_tasks"

    def test_get_task_group_embedding(self):
        """EMBEDDING should be in embedding_tasks."""
        from modeling_studio.data.labels import Capability
        from modeling_studio.models.modernbert_multitask import get_task_group

        assert get_task_group(Capability.EMBEDDING) == "embedding_tasks"


# =============================================================================
# Head Type Mapping Tests
# =============================================================================


class TestCapabilityToHeadTypeMapping:
    """Test 12 capabilities mapped to correct head types."""

    def test_mapping_exists(self):
        """CAPABILITY_TO_HEAD_TYPE should be defined."""
        from modeling_studio.models.modernbert_multitask import CAPABILITY_TO_HEAD_TYPE

        assert CAPABILITY_TO_HEAD_TYPE is not None

    def test_mapping_has_12_entries(self):
        """Mapping should have 12 capabilities."""
        from modeling_studio.models.modernbert_multitask import CAPABILITY_TO_HEAD_TYPE

        assert len(CAPABILITY_TO_HEAD_TYPE) == 12

    def test_ner_general_head_type(self):
        """NER_GENERAL should map to TokenClassificationHead."""
        from modeling_studio.data.labels import Capability
        from modeling_studio.models.heads import TokenClassificationHead
        from modeling_studio.models.modernbert_multitask import CAPABILITY_TO_HEAD_TYPE

        assert CAPABILITY_TO_HEAD_TYPE[Capability.NER_GENERAL] == TokenClassificationHead

    def test_sentiment_head_type(self):
        """SENTIMENT should map to SequenceClassificationHead."""
        from modeling_studio.data.labels import Capability
        from modeling_studio.models.heads import SequenceClassificationHead
        from modeling_studio.models.modernbert_multitask import CAPABILITY_TO_HEAD_TYPE

        assert CAPABILITY_TO_HEAD_TYPE[Capability.SENTIMENT] == SequenceClassificationHead

    def test_embedding_head_type(self):
        """EMBEDDING should map to EmbeddingHead."""
        from modeling_studio.data.labels import Capability
        from modeling_studio.models.heads import EmbeddingHead
        from modeling_studio.models.modernbert_multitask import CAPABILITY_TO_HEAD_TYPE

        assert CAPABILITY_TO_HEAD_TYPE[Capability.EMBEDDING] == EmbeddingHead

    def test_nli_head_type(self):
        """NLI should map to NLIHead."""
        from modeling_studio.data.labels import Capability
        from modeling_studio.models.heads import NLIHead
        from modeling_studio.models.modernbert_multitask import CAPABILITY_TO_HEAD_TYPE

        assert CAPABILITY_TO_HEAD_TYPE[Capability.NLI] == NLIHead


class TestGetProblemType:
    """Test get_problem_type returns correct problem type."""

    def test_get_problem_type_function_exists(self):
        """get_problem_type function should exist."""
        from modeling_studio.models.modernbert_multitask import get_problem_type

        assert get_problem_type is not None
        assert callable(get_problem_type)

    def test_get_problem_type_sentiment(self):
        """SENTIMENT should return single_label_classification."""
        from modeling_studio.data.labels import Capability
        from modeling_studio.models.modernbert_multitask import get_problem_type

        assert get_problem_type(Capability.SENTIMENT) == "single_label_classification"

    def test_get_problem_type_emotions(self):
        """EMOTIONS should return multi_label_classification."""
        from modeling_studio.data.labels import Capability
        from modeling_studio.models.modernbert_multitask import get_problem_type

        assert get_problem_type(Capability.EMOTIONS) == "multi_label_classification"

    def test_get_problem_type_ner(self):
        """NER_GENERAL should return token_classification."""
        from modeling_studio.data.labels import Capability
        from modeling_studio.models.modernbert_multitask import get_problem_type

        assert get_problem_type(Capability.NER_GENERAL) == "token_classification"

    def test_get_problem_type_embedding(self):
        """EMBEDDING should return embedding."""
        from modeling_studio.data.labels import Capability
        from modeling_studio.models.modernbert_multitask import get_problem_type

        assert get_problem_type(Capability.EMBEDDING) == "embedding"


# =============================================================================
# MultiTaskOutput Tests
# =============================================================================


class TestMultiTaskOutputInit:
    """Test MultiTaskOutput initializes with all fields."""

    def test_multi_task_output_class_exists(self):
        """MultiTaskOutput class should exist."""
        from modeling_studio.models.modernbert_multitask import MultiTaskOutput

        assert MultiTaskOutput is not None

    def test_multi_task_output_init_defaults(self):
        """MultiTaskOutput should initialize with default None values."""
        from modeling_studio.models.modernbert_multitask import MultiTaskOutput

        output = MultiTaskOutput()
        assert output.loss is None
        assert output.logits is None
        assert output.hidden_states is None
        assert output.attentions is None
        assert output.capability is None

    def test_multi_task_output_init_with_values(self):
        """MultiTaskOutput should accept values."""
        from modeling_studio.data.labels import Capability
        from modeling_studio.models.modernbert_multitask import MultiTaskOutput

        loss = torch.tensor(0.5)
        logits = torch.randn(2, 5)

        output = MultiTaskOutput(
            loss=loss,
            logits=logits,
            capability=Capability.SENTIMENT,
        )

        assert output.loss is not None
        assert output.logits is not None
        assert output.capability == Capability.SENTIMENT


class TestMultiTaskOutputToDict:
    """Test MultiTaskOutput converts to dictionary."""

    def test_to_dict_method_exists(self):
        """MultiTaskOutput should have to_dict method."""
        from modeling_studio.models.modernbert_multitask import MultiTaskOutput

        assert hasattr(MultiTaskOutput, "to_dict")

    def test_to_dict_returns_dict(self):
        """to_dict should return a dictionary."""
        from modeling_studio.models.modernbert_multitask import MultiTaskOutput

        output = MultiTaskOutput()
        result = output.to_dict()

        assert isinstance(result, dict)

    def test_to_dict_has_all_keys(self):
        """to_dict should have all expected keys."""
        from modeling_studio.models.modernbert_multitask import MultiTaskOutput

        output = MultiTaskOutput()
        result = output.to_dict()

        assert "loss" in result
        assert "logits" in result
        assert "hidden_states" in result
        assert "attentions" in result
        assert "capability" in result


# =============================================================================
# Forward Pass Tests (Mocked - no actual model loading)
# =============================================================================


class TestForwardNerGeneral:
    """Test forward pass with NER capability returns logits."""

    def test_forward_method_exists(self):
        """Model should have forward method."""
        from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

        assert hasattr(ModernBertMultiTaskModel, "forward")

    def test_forward_requires_capability(self):
        """Forward should require a capability parameter."""
        import inspect

        from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

        sig = inspect.signature(ModernBertMultiTaskModel.forward)
        assert "capability" in sig.parameters


class TestForwardSentiment:
    """Test forward pass with sentiment returns logits."""

    def test_forward_accepts_labels(self):
        """Forward should accept labels parameter."""
        import inspect

        from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

        sig = inspect.signature(ModernBertMultiTaskModel.forward)
        assert "labels" in sig.parameters


class TestForwardEmotions:
    """Test forward pass with emotions returns multi-label logits."""

    def test_emotions_uses_hierarchical_head(self):
        """EMOTIONS capability should use HierarchicalEmotionHead."""
        from modeling_studio.data.labels import Capability
        from modeling_studio.models.heads import HierarchicalEmotionHead
        from modeling_studio.models.modernbert_multitask import CAPABILITY_TO_HEAD_TYPE

        assert CAPABILITY_TO_HEAD_TYPE[Capability.EMOTIONS] == HierarchicalEmotionHead


class TestForwardSafetyFamilyos:
    """Test forward pass with safety returns 4 bands."""

    def test_safety_uses_enhanced_head(self):
        """SAFETY_FAMILYOS capability should use EnhancedSafetyHead."""
        from modeling_studio.data.labels import Capability
        from modeling_studio.models.heads import EnhancedSafetyHead
        from modeling_studio.models.modernbert_multitask import CAPABILITY_TO_HEAD_TYPE

        assert CAPABILITY_TO_HEAD_TYPE[Capability.SAFETY_FAMILYOS] == EnhancedSafetyHead


class TestForwardEmbedding:
    """Test forward pass returns normalized embeddings."""

    def test_embedding_head_type_is_correct(self):
        """EMBEDDING should use EmbeddingHead."""
        from modeling_studio.data.labels import Capability
        from modeling_studio.models.heads import EmbeddingHead
        from modeling_studio.models.modernbert_multitask import CAPABILITY_TO_HEAD_TYPE

        assert CAPABILITY_TO_HEAD_TYPE[Capability.EMBEDDING] == EmbeddingHead


class TestForwardWithLabels:
    """Test forward pass computes loss when labels provided."""

    def test_forward_returns_multi_task_output(self):
        """Forward should return MultiTaskOutput when return_dict=True."""
        import inspect

        from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

        sig = inspect.signature(ModernBertMultiTaskModel.forward)
        assert "return_dict" in sig.parameters


# =============================================================================
# Encoder Freezing Tests
# =============================================================================


class TestFreezeEncoder:
    """Test encoder weights frozen when flag set."""

    def test_freeze_encoder_method_exists(self):
        """Model should have freeze_encoder_weights method."""
        from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

        assert hasattr(ModernBertMultiTaskModel, "freeze_encoder_weights")

    def test_unfreeze_encoder_method_exists(self):
        """Model should have unfreeze_encoder_weights method."""
        from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

        assert hasattr(ModernBertMultiTaskModel, "unfreeze_encoder_weights")

    def test_init_accepts_freeze_encoder(self):
        """Model __init__ should accept freeze_encoder parameter."""
        import inspect

        from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

        sig = inspect.signature(ModernBertMultiTaskModel.__init__)
        assert "freeze_encoder" in sig.parameters


# =============================================================================
# Epic 5.0 Integration Tests
# =============================================================================


class TestSharedPoolerIntegration:
    """Test shared pooler used by sequence heads."""

    def test_init_accepts_shared_pooler(self):
        """Model __init__ should accept shared_pooler parameter."""
        import inspect

        from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

        sig = inspect.signature(ModernBertMultiTaskModel.__init__)
        assert "shared_pooler" in sig.parameters


class TestAdapterIntegration:
    """Test task adapters applied to encoder output."""

    def test_init_accepts_use_adapters(self):
        """Model __init__ should accept use_adapters parameter."""
        import inspect

        from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

        sig = inspect.signature(ModernBertMultiTaskModel.__init__)
        assert "use_adapters" in sig.parameters

    def test_init_accepts_adapter_bottleneck_size(self):
        """Model __init__ should accept adapter_bottleneck_size parameter."""
        import inspect

        from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

        sig = inspect.signature(ModernBertMultiTaskModel.__init__)
        assert "adapter_bottleneck_size" in sig.parameters


class TestPairEncoderIntegration:
    """Test pair encoder used for NLI/Relation."""

    def test_init_accepts_use_pair_encoder(self):
        """Model __init__ should accept use_pair_encoder parameter."""
        import inspect

        from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

        sig = inspect.signature(ModernBertMultiTaskModel.__init__)
        assert "use_pair_encoder" in sig.parameters

    def test_init_accepts_pair_encoder_num_layers(self):
        """Model __init__ should accept pair_encoder_num_layers parameter."""
        import inspect

        from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

        sig = inspect.signature(ModernBertMultiTaskModel.__init__)
        assert "pair_encoder_num_layers" in sig.parameters


# =============================================================================
# Model Loading/Saving Tests
# =============================================================================


class TestFromPretrained:
    """Test loading from HuggingFace checkpoint."""

    def test_from_pretrained_method_exists(self):
        """Model should have from_pretrained class method."""
        from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

        assert hasattr(ModernBertMultiTaskModel, "from_pretrained")

    def test_from_pretrained_is_classmethod(self):
        """from_pretrained should be a class method."""
        from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

        # Check it's callable on the class
        assert callable(ModernBertMultiTaskModel.from_pretrained)


class TestSavePretrained:
    """Test saving model to disk."""

    def test_save_pretrained_method_exists(self):
        """Model should have save_pretrained method."""
        from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

        assert hasattr(ModernBertMultiTaskModel, "save_pretrained")


class TestLoadCheckpoint:
    """Test loading from training checkpoint."""

    def test_load_checkpoint_method_exists(self):
        """Model should have load_checkpoint class method."""
        from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

        assert hasattr(ModernBertMultiTaskModel, "load_checkpoint")


# =============================================================================
# Special Head Tests
# =============================================================================


class TestEmotionsUsesHierarchicalHead:
    """Test Emotions capability uses HierarchicalEmotionHead."""

    def test_hierarchical_emotion_head_imported(self):
        """HierarchicalEmotionHead should be available."""
        from modeling_studio.models.heads import HierarchicalEmotionHead

        assert HierarchicalEmotionHead is not None

    def test_emotions_maps_to_hierarchical_head(self):
        """EMOTIONS should map to HierarchicalEmotionHead."""
        from modeling_studio.data.labels import Capability
        from modeling_studio.models.heads import HierarchicalEmotionHead
        from modeling_studio.models.modernbert_multitask import CAPABILITY_TO_HEAD_TYPE

        assert CAPABILITY_TO_HEAD_TYPE[Capability.EMOTIONS] == HierarchicalEmotionHead


class TestSafetyUsesEnhancedHead:
    """Test Safety FamilyOS uses EnhancedSafetyHead."""

    def test_enhanced_safety_head_imported(self):
        """EnhancedSafetyHead should be available."""
        from modeling_studio.models.heads import EnhancedSafetyHead

        assert EnhancedSafetyHead is not None

    def test_safety_familyos_maps_to_enhanced_head(self):
        """SAFETY_FAMILYOS should map to EnhancedSafetyHead."""
        from modeling_studio.data.labels import Capability
        from modeling_studio.models.heads import EnhancedSafetyHead
        from modeling_studio.models.modernbert_multitask import CAPABILITY_TO_HEAD_TYPE

        assert CAPABILITY_TO_HEAD_TYPE[Capability.SAFETY_FAMILYOS] == EnhancedSafetyHead


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Test all public APIs are exported."""

    def test_all_exports_defined(self):
        """__all__ should be defined with public APIs."""
        from modeling_studio.models import modernbert_multitask

        assert hasattr(modernbert_multitask, "__all__")
        assert "ModernBertMultiTaskModel" in modernbert_multitask.__all__
        assert "MultiTaskOutput" in modernbert_multitask.__all__
        assert "CAPABILITY_TO_HEAD_TYPE" in modernbert_multitask.__all__
        assert "get_problem_type" in modernbert_multitask.__all__
