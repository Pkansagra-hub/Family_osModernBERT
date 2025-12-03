"""
Milestone 2: Data Pipeline Tests
Issue 2.1.1: data/__init__.py exports

Tests for:
- LabelSchema class exported
- Capability enum exported
- get_labels_for_capability function exported
- All 12 capabilities available (6 generic + 6 FamilyOS)
- All label schemas exported correctly
- get_num_labels function exported
"""

import pytest


class TestDataModuleLabelExports:
    """Test that label-related exports are available from data module."""

    def test_label_schema_exported(self):
        """Issue 2.1.1: LabelSchema class should be exported."""
        from modeling_studio.data import LabelSchema

        assert LabelSchema is not None
        # LabelSchema is a dataclass, instance attributes are in __dataclass_fields__
        assert "label2id" in LabelSchema.__dataclass_fields__
        assert hasattr(LabelSchema, "encode")  # method on class
        assert hasattr(LabelSchema, "decode")  # method on class

    def test_capability_enum_exported(self):
        """Issue 2.1.1: Capability enum should be exported."""
        from modeling_studio.data import Capability

        assert Capability is not None
        # Verify it's an enum
        from enum import Enum

        assert issubclass(Capability, Enum)

    def test_get_labels_for_capability_exported(self):
        """Issue 2.1.1: get_labels_for_capability function should be exported."""
        from modeling_studio.data import get_labels_for_capability

        assert callable(get_labels_for_capability)

    def test_get_num_labels_exported(self):
        """Issue 2.1.1: get_num_labels function should be exported."""
        from modeling_studio.data import get_num_labels

        assert callable(get_num_labels)


class TestDataModuleCapabilityEnumCompleteness:
    """Test that all 12 capabilities are defined in the Capability enum."""

    def test_all_12_capabilities_in_enum(self):
        """Issue 2.1.1: All 12 capabilities should be in Capability enum."""
        from modeling_studio.data import Capability

        expected_capabilities = {
            # Generic (7, but EMBEDDING counts as a capability)
            "NER_GENERAL",
            "SENTIMENT",
            "EMOTIONS",
            "SAFETY_GENERIC",
            "NLI",
            "EMBEDDING",
            "TEMPORAL",
            # FamilyOS (5)
            "NER_FAMILY",
            "INGRESS",
            "SAFETY_FAMILYOS",
            "RELATION",
            "INTENT",
        }

        actual_capabilities = {c.name for c in Capability}
        assert expected_capabilities == actual_capabilities

    def test_capability_values_are_strings(self):
        """All Capability values should be lowercase string identifiers."""
        from modeling_studio.data import Capability

        for cap in Capability:
            assert isinstance(cap.value, str)
            assert cap.value == cap.name.lower()


class TestDataModuleGenericLabelExports:
    """Test that generic (Stage A) label schemas are exported."""

    def test_ner_general_labels_exported(self):
        """NER_GENERAL_LABELS should be exported."""
        from modeling_studio.data import NER_GENERAL_LABELS, LabelSchema

        assert NER_GENERAL_LABELS is not None
        assert isinstance(NER_GENERAL_LABELS, LabelSchema)

    def test_sentiment_labels_exported(self):
        """SENTIMENT_LABELS should be exported."""
        from modeling_studio.data import SENTIMENT_LABELS, LabelSchema

        assert SENTIMENT_LABELS is not None
        assert isinstance(SENTIMENT_LABELS, LabelSchema)

    def test_emotions_labels_exported(self):
        """EMOTIONS_LABELS should be exported."""
        from modeling_studio.data import EMOTIONS_LABELS, LabelSchema

        assert EMOTIONS_LABELS is not None
        assert isinstance(EMOTIONS_LABELS, LabelSchema)

    def test_emotions_reduced_labels_exported(self):
        """EMOTIONS_REDUCED_LABELS should be exported."""
        from modeling_studio.data import EMOTIONS_REDUCED_LABELS, LabelSchema

        assert EMOTIONS_REDUCED_LABELS is not None
        assert isinstance(EMOTIONS_REDUCED_LABELS, LabelSchema)

    def test_safety_generic_labels_exported(self):
        """SAFETY_GENERIC_LABELS should be exported."""
        from modeling_studio.data import SAFETY_GENERIC_LABELS, LabelSchema

        assert SAFETY_GENERIC_LABELS is not None
        assert isinstance(SAFETY_GENERIC_LABELS, LabelSchema)

    def test_nli_labels_exported(self):
        """NLI_LABELS should be exported."""
        from modeling_studio.data import NLI_LABELS, LabelSchema

        assert NLI_LABELS is not None
        assert isinstance(NLI_LABELS, LabelSchema)

    def test_temporal_labels_exported(self):
        """TEMPORAL_LABELS should be exported."""
        from modeling_studio.data import TEMPORAL_LABELS, LabelSchema

        assert TEMPORAL_LABELS is not None
        assert isinstance(TEMPORAL_LABELS, LabelSchema)


class TestDataModuleFamilyOSLabelExports:
    """Test that FamilyOS (Stage B) label schemas are exported."""

    def test_ner_family_labels_exported(self):
        """NER_FAMILY_LABELS should be exported."""
        from modeling_studio.data import NER_FAMILY_LABELS, LabelSchema

        assert NER_FAMILY_LABELS is not None
        assert isinstance(NER_FAMILY_LABELS, LabelSchema)

    def test_ingress_labels_exported(self):
        """INGRESS_LABELS should be exported."""
        from modeling_studio.data import INGRESS_LABELS, LabelSchema

        assert INGRESS_LABELS is not None
        assert isinstance(INGRESS_LABELS, LabelSchema)

    def test_safety_familyos_labels_exported(self):
        """SAFETY_FAMILYOS_LABELS should be exported."""
        from modeling_studio.data import SAFETY_FAMILYOS_LABELS, LabelSchema

        assert SAFETY_FAMILYOS_LABELS is not None
        assert isinstance(SAFETY_FAMILYOS_LABELS, LabelSchema)

    def test_safety_subcategories_exported(self):
        """SAFETY_SUBCATEGORIES should be exported (Issue 3.6.8)."""
        from modeling_studio.data import SAFETY_SUBCATEGORIES, LabelSchema

        assert SAFETY_SUBCATEGORIES is not None
        assert isinstance(SAFETY_SUBCATEGORIES, LabelSchema)

    def test_subcategory_to_band_id_exported(self):
        """SUBCATEGORY_TO_BAND_ID mapping should be exported (Issue 3.6.8)."""
        from modeling_studio.data import SUBCATEGORY_TO_BAND_ID

        assert SUBCATEGORY_TO_BAND_ID is not None
        assert isinstance(SUBCATEGORY_TO_BAND_ID, dict)

    def test_band_to_subcategory_ids_exported(self):
        """BAND_TO_SUBCATEGORY_IDS mapping should be exported (Issue 3.6.8)."""
        from modeling_studio.data import BAND_TO_SUBCATEGORY_IDS

        assert BAND_TO_SUBCATEGORY_IDS is not None
        assert isinstance(BAND_TO_SUBCATEGORY_IDS, dict)

    def test_relation_labels_exported(self):
        """RELATION_LABELS should be exported."""
        from modeling_studio.data import RELATION_LABELS, LabelSchema

        assert RELATION_LABELS is not None
        assert isinstance(RELATION_LABELS, LabelSchema)

    def test_intent_labels_exported(self):
        """INTENT_LABELS should be exported."""
        from modeling_studio.data import INTENT_LABELS, LabelSchema

        assert INTENT_LABELS is not None
        assert isinstance(INTENT_LABELS, LabelSchema)


class TestDataModuleMappingExports:
    """Test that mapping dictionaries are exported."""

    def test_capability_to_labels_exported(self):
        """CAPABILITY_TO_LABELS mapping should be exported."""
        from modeling_studio.data import CAPABILITY_TO_LABELS

        assert CAPABILITY_TO_LABELS is not None
        assert isinstance(CAPABILITY_TO_LABELS, dict)

    def test_all_label_schemas_exported(self):
        """ALL_LABEL_SCHEMAS mapping should be exported."""
        from modeling_studio.data import ALL_LABEL_SCHEMAS

        assert ALL_LABEL_SCHEMAS is not None
        assert isinstance(ALL_LABEL_SCHEMAS, dict)


class TestGetLabelsForCapability:
    """Test the get_labels_for_capability function."""

    def test_get_labels_with_capability_enum(self):
        """get_labels_for_capability should work with Capability enum."""
        from modeling_studio.data import (
            Capability,
            get_labels_for_capability,
            LabelSchema,
        )

        labels = get_labels_for_capability(Capability.NER_GENERAL)
        assert labels is not None
        assert isinstance(labels, LabelSchema)

    def test_get_labels_with_string(self):
        """get_labels_for_capability should work with string."""
        from modeling_studio.data import (
            get_labels_for_capability,
            LabelSchema,
        )

        labels = get_labels_for_capability("ner_general")
        assert labels is not None
        assert isinstance(labels, LabelSchema)

    def test_get_labels_embedding_returns_none(self):
        """Embedding capability should return None (no labels)."""
        from modeling_studio.data import Capability, get_labels_for_capability

        labels = get_labels_for_capability(Capability.EMBEDDING)
        assert labels is None

    def test_get_labels_all_capabilities(self):
        """All capabilities should return appropriate labels or None."""
        from modeling_studio.data import (
            Capability,
            get_labels_for_capability,
            LabelSchema,
        )

        for cap in Capability:
            labels = get_labels_for_capability(cap)
            if cap == Capability.EMBEDDING:
                assert labels is None
            else:
                assert labels is not None
                assert isinstance(labels, LabelSchema)


class TestGetNumLabels:
    """Test the get_num_labels function."""

    def test_get_num_labels_with_enum(self):
        """get_num_labels should work with Capability enum."""
        from modeling_studio.data import Capability, get_num_labels

        num = get_num_labels(Capability.NER_GENERAL)
        assert num > 0

    def test_get_num_labels_with_string(self):
        """get_num_labels should work with string."""
        from modeling_studio.data import get_num_labels

        num = get_num_labels("ner_general")
        assert num > 0

    def test_get_num_labels_embedding_returns_zero(self):
        """Embedding capability should return 0."""
        from modeling_studio.data import Capability, get_num_labels

        num = get_num_labels(Capability.EMBEDDING)
        assert num == 0


class TestDataModuleLoaderExports:
    """Test that loader functions are exported."""

    def test_load_ner_dataset_exported(self):
        """load_ner_dataset should be exported."""
        from modeling_studio.data import load_ner_dataset

        assert callable(load_ner_dataset)

    def test_load_classification_dataset_exported(self):
        """load_classification_dataset should be exported."""
        from modeling_studio.data import load_classification_dataset

        assert callable(load_classification_dataset)

    def test_load_multilabel_dataset_exported(self):
        """load_multilabel_dataset should be exported."""
        from modeling_studio.data import load_multilabel_dataset

        assert callable(load_multilabel_dataset)

    def test_load_nli_dataset_exported(self):
        """load_nli_dataset should be exported."""
        from modeling_studio.data import load_nli_dataset

        assert callable(load_nli_dataset)

    def test_load_embedding_dataset_exported(self):
        """load_embedding_dataset should be exported."""
        from modeling_studio.data import load_embedding_dataset

        assert callable(load_embedding_dataset)

    def test_load_familyos_ner_exported(self):
        """load_familyos_ner should be exported."""
        from modeling_studio.data import load_familyos_ner

        assert callable(load_familyos_ner)

    def test_load_familyos_ingress_exported(self):
        """load_familyos_ingress should be exported."""
        from modeling_studio.data import load_familyos_ingress

        assert callable(load_familyos_ingress)

    def test_load_familyos_safety_exported(self):
        """load_familyos_safety should be exported."""
        from modeling_studio.data import load_familyos_safety

        assert callable(load_familyos_safety)

    def test_load_familyos_relations_exported(self):
        """load_familyos_relations should be exported."""
        from modeling_studio.data import load_familyos_relations

        assert callable(load_familyos_relations)

    def test_load_familyos_intents_exported(self):
        """load_familyos_intents should be exported."""
        from modeling_studio.data import load_familyos_intents

        assert callable(load_familyos_intents)

    def test_load_familyos_temporal_exported(self):
        """load_familyos_temporal should be exported."""
        from modeling_studio.data import load_familyos_temporal

        assert callable(load_familyos_temporal)

    def test_load_from_config_exported(self):
        """load_from_config should be exported."""
        from modeling_studio.data import load_from_config

        assert callable(load_from_config)

    def test_load_stage_a_datasets_exported(self):
        """load_stage_a_datasets should be exported."""
        from modeling_studio.data import load_stage_a_datasets

        assert callable(load_stage_a_datasets)

    def test_load_stage_b_datasets_exported(self):
        """load_stage_b_datasets should be exported."""
        from modeling_studio.data import load_stage_b_datasets

        assert callable(load_stage_b_datasets)


class TestDataModuleMultiTaskDatasetExports:
    """Test that multi-task dataset classes are exported."""

    def test_task_dataset_exported(self):
        """TaskDataset should be exported."""
        from modeling_studio.data import TaskDataset

        assert TaskDataset is not None

    def test_multi_task_dataset_exported(self):
        """MultiTaskDataset should be exported."""
        from modeling_studio.data import MultiTaskDataset

        assert MultiTaskDataset is not None

    def test_create_multitask_dataset_exported(self):
        """create_multitask_dataset should be exported."""
        from modeling_studio.data import create_multitask_dataset

        assert callable(create_multitask_dataset)


class TestDataModuleTokenizationExports:
    """Test that tokenization utilities are exported."""

    def test_load_tokenizer_exported(self):
        """load_tokenizer should be exported."""
        from modeling_studio.data import load_tokenizer

        assert callable(load_tokenizer)

    def test_tokenize_for_classification_exported(self):
        """tokenize_for_classification should be exported."""
        from modeling_studio.data import tokenize_for_classification

        assert callable(tokenize_for_classification)

    def test_tokenize_for_token_classification_exported(self):
        """tokenize_for_token_classification should be exported."""
        from modeling_studio.data import tokenize_for_token_classification

        assert callable(tokenize_for_token_classification)

    def test_tokenize_for_nli_exported(self):
        """tokenize_for_nli should be exported."""
        from modeling_studio.data import tokenize_for_nli

        assert callable(tokenize_for_nli)

    def test_tokenize_for_embedding_exported(self):
        """tokenize_for_embedding should be exported."""
        from modeling_studio.data import tokenize_for_embedding

        assert callable(tokenize_for_embedding)

    def test_get_tokenize_function_exported(self):
        """get_tokenize_function should be exported."""
        from modeling_studio.data import get_tokenize_function

        assert callable(get_tokenize_function)


class TestDataModuleIndianEnglishExports:
    """Test that Indian English support (Issue 3.6.7) is exported."""

    def test_indian_english_mappings_exported(self):
        """INDIAN_ENGLISH_MAPPINGS should be exported."""
        from modeling_studio.data import INDIAN_ENGLISH_MAPPINGS

        assert INDIAN_ENGLISH_MAPPINGS is not None
        assert isinstance(INDIAN_ENGLISH_MAPPINGS, dict)

    def test_indian_venting_patterns_exported(self):
        """INDIAN_VENTING_PATTERNS should be exported."""
        from modeling_studio.data import INDIAN_VENTING_PATTERNS

        assert INDIAN_VENTING_PATTERNS is not None

    def test_kinship_variants_exported(self):
        """KINSHIP_VARIANTS should be exported."""
        from modeling_studio.data import KINSHIP_VARIANTS

        assert KINSHIP_VARIANTS is not None

    def test_family_structure_types_exported(self):
        """FAMILY_STRUCTURE_TYPES should be exported."""
        from modeling_studio.data import FAMILY_STRUCTURE_TYPES

        assert FAMILY_STRUCTURE_TYPES is not None

    def test_family_structure_type_dataclass_exported(self):
        """FamilyStructureType dataclass should be exported."""
        from dataclasses import is_dataclass

        from modeling_studio.data import FamilyStructureType

        assert FamilyStructureType is not None
        assert is_dataclass(FamilyStructureType)

    def test_indian_english_normalizer_exported(self):
        """IndianEnglishNormalizer class should be exported."""
        from modeling_studio.data import IndianEnglishNormalizer

        assert IndianEnglishNormalizer is not None

    def test_normalize_indian_english_exported(self):
        """normalize_indian_english function should be exported."""
        from modeling_studio.data import normalize_indian_english

        assert callable(normalize_indian_english)

    def test_is_venting_exported(self):
        """is_venting function should be exported."""
        from modeling_studio.data import is_venting

        assert callable(is_venting)

    def test_get_kinship_variants_exported(self):
        """get_kinship_variants function should be exported."""
        from modeling_studio.data import get_kinship_variants

        assert callable(get_kinship_variants)
