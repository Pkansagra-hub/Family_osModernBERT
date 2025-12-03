"""
Milestone 2: Data Pipeline Tests
Issue 2.1.2: data/labels.py LabelSchema and 12 capabilities

Tests for:
- LabelSchema encode/decode
- LabelSchema properties
- Exact label counts for all 12 capabilities
- Label mappings consistency
- Capability enum behavior
"""

import pytest


class TestLabelSchemaBasics:
    """Test LabelSchema dataclass basic functionality."""

    def test_label_schema_name_attribute(self):
        """LabelSchema should have name attribute."""
        from modeling_studio.data.labels import NER_GENERAL_LABELS

        assert hasattr(NER_GENERAL_LABELS, "name")
        assert NER_GENERAL_LABELS.name == "ner_general"

    def test_label_schema_label2id_attribute(self):
        """LabelSchema should have label2id attribute."""
        from modeling_studio.data.labels import NER_GENERAL_LABELS

        assert hasattr(NER_GENERAL_LABELS, "label2id")
        assert isinstance(NER_GENERAL_LABELS.label2id, dict)

    def test_label_schema_id2label_property(self):
        """LabelSchema should have id2label property."""
        from modeling_studio.data.labels import NER_GENERAL_LABELS

        assert hasattr(NER_GENERAL_LABELS, "id2label")
        assert isinstance(NER_GENERAL_LABELS.id2label, dict)

    def test_label_schema_num_labels_property(self):
        """LabelSchema should have num_labels property."""
        from modeling_studio.data.labels import NER_GENERAL_LABELS

        assert hasattr(NER_GENERAL_LABELS, "num_labels")
        assert isinstance(NER_GENERAL_LABELS.num_labels, int)

    def test_label_schema_problem_type_attribute(self):
        """LabelSchema should have problem_type attribute."""
        from modeling_studio.data.labels import NER_GENERAL_LABELS

        assert hasattr(NER_GENERAL_LABELS, "problem_type")
        assert NER_GENERAL_LABELS.problem_type == "token_classification"

    def test_label_schema_description_attribute(self):
        """LabelSchema should have description attribute."""
        from modeling_studio.data.labels import NER_GENERAL_LABELS

        assert hasattr(NER_GENERAL_LABELS, "description")
        assert isinstance(NER_GENERAL_LABELS.description, str)


class TestLabelSchemaEncode:
    """Test LabelSchema encode method."""

    def test_encode_valid_label(self):
        """encode should convert valid label string to ID."""
        from modeling_studio.data.labels import NER_GENERAL_LABELS

        label_id = NER_GENERAL_LABELS.encode("O")
        assert label_id == 0

    def test_encode_b_per_label(self):
        """encode should convert B-PER to correct ID."""
        from modeling_studio.data.labels import NER_GENERAL_LABELS

        label_id = NER_GENERAL_LABELS.encode("B-PER")
        assert label_id == 1

    def test_encode_invalid_label_raises(self):
        """encode should raise KeyError for invalid label."""
        from modeling_studio.data.labels import NER_GENERAL_LABELS

        with pytest.raises(KeyError):
            NER_GENERAL_LABELS.encode("INVALID_LABEL")

    def test_encode_case_sensitive(self):
        """encode should be case sensitive."""
        from modeling_studio.data.labels import NER_GENERAL_LABELS

        with pytest.raises(KeyError):
            NER_GENERAL_LABELS.encode("o")  # lowercase 'o' should fail


class TestLabelSchemaDecode:
    """Test LabelSchema decode method."""

    def test_decode_valid_id(self):
        """decode should convert valid ID to label string."""
        from modeling_studio.data.labels import NER_GENERAL_LABELS

        label = NER_GENERAL_LABELS.decode(0)
        assert label == "O"

    def test_decode_id_1(self):
        """decode should convert ID 1 to B-PER."""
        from modeling_studio.data.labels import NER_GENERAL_LABELS

        label = NER_GENERAL_LABELS.decode(1)
        assert label == "B-PER"

    def test_decode_invalid_id_raises(self):
        """decode should raise KeyError for invalid ID."""
        from modeling_studio.data.labels import NER_GENERAL_LABELS

        with pytest.raises(KeyError):
            NER_GENERAL_LABELS.decode(9999)


class TestLabelSchemaRoundTrip:
    """Test LabelSchema encode/decode round trip."""

    def test_encode_decode_round_trip(self):
        """encode then decode should return original label."""
        from modeling_studio.data.labels import NER_GENERAL_LABELS

        original = "B-PER"
        label_id = NER_GENERAL_LABELS.encode(original)
        decoded = NER_GENERAL_LABELS.decode(label_id)
        assert decoded == original

    def test_all_labels_round_trip(self):
        """All labels should survive encode/decode round trip."""
        from modeling_studio.data.labels import NER_GENERAL_LABELS

        for label in NER_GENERAL_LABELS.label2id:
            label_id = NER_GENERAL_LABELS.encode(label)
            decoded = NER_GENERAL_LABELS.decode(label_id)
            assert decoded == label


class TestLabelSchemaSerialization:
    """Test LabelSchema to_dict and from_dict."""

    def test_to_dict(self):
        """to_dict should return dictionary with all fields."""
        from modeling_studio.data.labels import NER_GENERAL_LABELS

        data = NER_GENERAL_LABELS.to_dict()
        assert isinstance(data, dict)
        assert "name" in data
        assert "label2id" in data
        assert "id2label" in data
        assert "num_labels" in data
        assert "problem_type" in data
        assert "description" in data

    def test_from_dict(self):
        """from_dict should reconstruct LabelSchema."""
        from modeling_studio.data.labels import LabelSchema, NER_GENERAL_LABELS

        data = NER_GENERAL_LABELS.to_dict()
        reconstructed = LabelSchema.from_dict(data)

        assert reconstructed.name == NER_GENERAL_LABELS.name
        assert reconstructed.label2id == NER_GENERAL_LABELS.label2id
        assert reconstructed.num_labels == NER_GENERAL_LABELS.num_labels

    def test_to_dict_from_dict_round_trip(self):
        """to_dict then from_dict should preserve data."""
        from modeling_studio.data.labels import LabelSchema, NER_GENERAL_LABELS

        data = NER_GENERAL_LABELS.to_dict()
        reconstructed = LabelSchema.from_dict(data)

        # Encode/decode should work on reconstructed schema
        assert reconstructed.encode("B-PER") == 1
        assert reconstructed.decode(1) == "B-PER"


class TestLabelSchemaConsistency:
    """Test LabelSchema internal consistency."""

    def test_label2id_id2label_inverse(self):
        """label2id and id2label should be inverses."""
        from modeling_studio.data.labels import NER_GENERAL_LABELS

        for label, idx in NER_GENERAL_LABELS.label2id.items():
            assert NER_GENERAL_LABELS.id2label[idx] == label

    def test_num_labels_matches_label2id_length(self):
        """num_labels should match length of label2id."""
        from modeling_studio.data.labels import NER_GENERAL_LABELS

        assert NER_GENERAL_LABELS.num_labels == len(NER_GENERAL_LABELS.label2id)

    def test_ids_are_contiguous(self):
        """IDs should be contiguous from 0 to num_labels-1."""
        from modeling_studio.data.labels import NER_GENERAL_LABELS

        ids = sorted(NER_GENERAL_LABELS.label2id.values())
        expected = list(range(NER_GENERAL_LABELS.num_labels))
        assert ids == expected


# =============================================================================
# Issue 2.1.2: Exact Label Counts for All 12 Capabilities
# =============================================================================


class TestNERGeneralLabelCount:
    """Test NER_GENERAL has exactly 17 BIO tags."""

    def test_ner_general_17_labels(self):
        """Issue 2.1.2: NER_GENERAL should have exactly 17 BIO tags."""
        from modeling_studio.data.labels import NER_GENERAL_LABELS

        assert NER_GENERAL_LABELS.num_labels == 17

    def test_ner_general_has_o_tag(self):
        """NER_GENERAL should have O tag at position 0."""
        from modeling_studio.data.labels import NER_GENERAL_LABELS

        assert "O" in NER_GENERAL_LABELS.label2id
        assert NER_GENERAL_LABELS.label2id["O"] == 0

    def test_ner_general_has_bio_pairs(self):
        """NER_GENERAL should have B- and I- for each entity type."""
        from modeling_studio.data.labels import NER_GENERAL_LABELS

        entity_types = ["PER", "ORG", "LOC", "MISC", "DATE", "TIME", "EVENT", "PRODUCT"]
        for ent in entity_types:
            assert f"B-{ent}" in NER_GENERAL_LABELS.label2id
            assert f"I-{ent}" in NER_GENERAL_LABELS.label2id


class TestNERFamilyLabelCount:
    """Test NER_FAMILY has exactly 21 BIO tags."""

    def test_ner_family_21_labels(self):
        """Issue 2.1.2: NER_FAMILY should have exactly 21 BIO tags."""
        from modeling_studio.data.labels import NER_FAMILY_LABELS

        assert NER_FAMILY_LABELS.num_labels == 21

    def test_ner_family_has_o_tag(self):
        """NER_FAMILY should have O tag at position 0."""
        from modeling_studio.data.labels import NER_FAMILY_LABELS

        assert "O" in NER_FAMILY_LABELS.label2id
        assert NER_FAMILY_LABELS.label2id["O"] == 0

    def test_ner_family_has_family_entities(self):
        """NER_FAMILY should have family-specific entity types."""
        from modeling_studio.data.labels import NER_FAMILY_LABELS

        family_entities = [
            "PERSON",
            "KINSHIP",
            "NICKNAME",
            "PET",
            "HOME_LOC",
            "FAMILY_EVENT",
            "ROUTINE",
            "TRADITION",
            "MILESTONE",
            "HEIRLOOM",
        ]
        for ent in family_entities:
            assert f"B-{ent}" in NER_FAMILY_LABELS.label2id
            assert f"I-{ent}" in NER_FAMILY_LABELS.label2id


class TestSentimentLabelCount:
    """Test SENTIMENT has exactly 5 classes."""

    def test_sentiment_5_labels(self):
        """Issue 2.1.2: SENTIMENT should have exactly 5 classes."""
        from modeling_studio.data.labels import SENTIMENT_LABELS

        assert SENTIMENT_LABELS.num_labels == 5

    def test_sentiment_labels_present(self):
        """SENTIMENT should have all 5 sentiment classes."""
        from modeling_studio.data.labels import SENTIMENT_LABELS

        expected = ["very_negative", "negative", "neutral", "positive", "very_positive"]
        for label in expected:
            assert label in SENTIMENT_LABELS.label2id


class TestEmotionsFamilyOSLabelCount:
    """Test EMOTIONS_FAMILYOS has exactly 44 classes."""

    def test_emotions_familyos_44_labels(self):
        """Issue 2.1.2: EMOTIONS_FAMILYOS should have exactly 44 classes."""
        from modeling_studio.data.labels import EMOTIONS_FAMILYOS_LABELS

        assert EMOTIONS_FAMILYOS_LABELS.num_labels == 44

    def test_emotions_familyos_problem_type(self):
        """EMOTIONS_FAMILYOS should be multi-label classification."""
        from modeling_studio.data.labels import EMOTIONS_FAMILYOS_LABELS

        assert EMOTIONS_FAMILYOS_LABELS.problem_type == "multi_label_classification"

    def test_emotions_familyos_has_core_emotions(self):
        """EMOTIONS_FAMILYOS should have core emotions."""
        from modeling_studio.data.labels import EMOTIONS_FAMILYOS_LABELS

        core = ["neutral", "joy", "sadness", "anger", "fear", "surprise", "love", "disgust"]
        for emotion in core:
            assert emotion in EMOTIONS_FAMILYOS_LABELS.label2id

    def test_emotions_familyos_has_family_specific(self):
        """EMOTIONS_FAMILYOS should have family-specific emotions."""
        from modeling_studio.data.labels import EMOTIONS_FAMILYOS_LABELS

        family_emotions = [
            "nostalgia",
            "protectiveness",
            "togetherness",
            "longing",
            "warmth",
            "playfulness",
            "celebration",
            "belonging",
            "parental_pride",
            "parental_guilt",
            "patience",
            "worry",
            "bittersweet",
            "homesickness",
        ]
        for emotion in family_emotions:
            assert emotion in EMOTIONS_FAMILYOS_LABELS.label2id


class TestSafetyGenericLabelCount:
    """Test SAFETY_GENERIC has exactly 8 types."""

    def test_safety_generic_8_labels(self):
        """Issue 2.1.2: SAFETY_GENERIC should have exactly 8 types."""
        from modeling_studio.data.labels import SAFETY_GENERIC_LABELS

        assert SAFETY_GENERIC_LABELS.num_labels == 8

    def test_safety_generic_has_jigsaw_types(self):
        """SAFETY_GENERIC should have Jigsaw toxicity types."""
        from modeling_studio.data.labels import SAFETY_GENERIC_LABELS

        jigsaw = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        for label in jigsaw:
            assert label in SAFETY_GENERIC_LABELS.label2id

    def test_safety_generic_has_additional_types(self):
        """SAFETY_GENERIC should have self_harm and dangerous_advice."""
        from modeling_studio.data.labels import SAFETY_GENERIC_LABELS

        assert "self_harm" in SAFETY_GENERIC_LABELS.label2id
        assert "dangerous_advice" in SAFETY_GENERIC_LABELS.label2id


class TestSafetyFamilyOSLabelCount:
    """Test SAFETY_FAMILYOS has exactly 4 bands."""

    def test_safety_familyos_4_labels(self):
        """Issue 2.1.2: SAFETY_FAMILYOS should have exactly 4 bands."""
        from modeling_studio.data.labels import SAFETY_FAMILYOS_LABELS

        assert SAFETY_FAMILYOS_LABELS.num_labels == 4

    def test_safety_familyos_bands_present(self):
        """SAFETY_FAMILYOS should have GREEN, AMBER, RED, CRISIS."""
        from modeling_studio.data.labels import SAFETY_FAMILYOS_LABELS

        expected = ["GREEN", "AMBER", "RED", "CRISIS"]
        for band in expected:
            assert band in SAFETY_FAMILYOS_LABELS.label2id

    def test_safety_familyos_band_order(self):
        """SAFETY_FAMILYOS bands should be in severity order."""
        from modeling_studio.data.labels import SAFETY_FAMILYOS_LABELS

        assert SAFETY_FAMILYOS_LABELS.label2id["GREEN"] == 0
        assert SAFETY_FAMILYOS_LABELS.label2id["AMBER"] == 1
        assert SAFETY_FAMILYOS_LABELS.label2id["RED"] == 2
        assert SAFETY_FAMILYOS_LABELS.label2id["CRISIS"] == 3


class TestNLILabelCount:
    """Test NLI has exactly 3 classes."""

    def test_nli_3_labels(self):
        """Issue 2.1.2: NLI should have exactly 3 classes."""
        from modeling_studio.data.labels import NLI_LABELS

        assert NLI_LABELS.num_labels == 3

    def test_nli_labels_present(self):
        """NLI should have entailment, neutral, contradiction."""
        from modeling_studio.data.labels import NLI_LABELS

        expected = ["entailment", "neutral", "contradiction"]
        for label in expected:
            assert label in NLI_LABELS.label2id


class TestIngressLabelCount:
    """Test INGRESS has exactly 12 domains."""

    def test_ingress_12_labels(self):
        """Issue 2.1.2: INGRESS should have exactly 12 domains."""
        from modeling_studio.data.labels import INGRESS_LABELS

        assert INGRESS_LABELS.num_labels == 12

    def test_ingress_original_domains(self):
        """INGRESS should have original 7 domains."""
        from modeling_studio.data.labels import INGRESS_LABELS

        original = ["DIARY", "TASK", "HEALTH", "FINANCE", "RELATIONSHIP", "WORK", "META"]
        for domain in original:
            assert domain in INGRESS_LABELS.label2id

    def test_ingress_extended_domains(self):
        """INGRESS should have 5 extended domains."""
        from modeling_studio.data.labels import INGRESS_LABELS

        extended = ["MEMORY", "PLANNING", "CELEBRATION", "CONCERN", "GRATITUDE"]
        for domain in extended:
            assert domain in INGRESS_LABELS.label2id


class TestTemporalLabelCount:
    """Test TEMPORAL has exactly 13 BIO tags."""

    def test_temporal_13_labels(self):
        """Issue 2.1.2: TEMPORAL should have exactly 13 BIO tags."""
        from modeling_studio.data.labels import TEMPORAL_LABELS

        assert TEMPORAL_LABELS.num_labels == 13

    def test_temporal_has_o_tag(self):
        """TEMPORAL should have O tag at position 0."""
        from modeling_studio.data.labels import TEMPORAL_LABELS

        assert "O" in TEMPORAL_LABELS.label2id
        assert TEMPORAL_LABELS.label2id["O"] == 0

    def test_temporal_has_temporal_types(self):
        """TEMPORAL should have temporal expression types."""
        from modeling_studio.data.labels import TEMPORAL_LABELS

        temporal_types = ["DATE_ABS", "DATE_REL", "TIME", "DURATION", "FREQUENCY", "AGE"]
        for ent in temporal_types:
            assert f"B-{ent}" in TEMPORAL_LABELS.label2id
            assert f"I-{ent}" in TEMPORAL_LABELS.label2id


class TestRelationLabelCount:
    """Test RELATION has exactly 15 relations."""

    def test_relation_15_labels(self):
        """Issue 2.1.2: RELATION should have exactly 15 relations."""
        from modeling_studio.data.labels import RELATION_LABELS

        assert RELATION_LABELS.num_labels == 15

    def test_relation_has_no_relation(self):
        """RELATION should have no_relation at position 0."""
        from modeling_studio.data.labels import RELATION_LABELS

        assert "no_relation" in RELATION_LABELS.label2id
        assert RELATION_LABELS.label2id["no_relation"] == 0

    def test_relation_has_family_relations(self):
        """RELATION should have family relation types."""
        from modeling_studio.data.labels import RELATION_LABELS

        family_relations = [
            "parent_of",
            "child_of",
            "spouse_of",
            "sibling_of",
            "grandparent_of",
            "grandchild_of",
            "aunt_uncle_of",
            "niece_nephew_of",
            "cousin_of",
            "pet_of",
        ]
        for rel in family_relations:
            assert rel in RELATION_LABELS.label2id

    def test_relation_has_non_family_relations(self):
        """RELATION should have non-family relation types."""
        from modeling_studio.data.labels import RELATION_LABELS

        non_family = ["friend_of", "colleague_of", "lives_at", "owns"]
        for rel in non_family:
            assert rel in RELATION_LABELS.label2id


class TestIntentLabelCount:
    """Test INTENT has exactly 8 intents."""

    def test_intent_8_labels(self):
        """Issue 2.1.2: INTENT should have exactly 8 intents."""
        from modeling_studio.data.labels import INTENT_LABELS

        assert INTENT_LABELS.num_labels == 8

    def test_intent_labels_present(self):
        """INTENT should have all 8 intent types."""
        from modeling_studio.data.labels import INTENT_LABELS

        expected = [
            "log_memory",
            "query_memory",
            "set_reminder",
            "express_feeling",
            "seek_advice",
            "share_news",
            "reflect",
            "other",
        ]
        for intent in expected:
            assert intent in INTENT_LABELS.label2id


# =============================================================================
# Capability Enum Tests
# =============================================================================


class TestCapabilityEnum:
    """Test Capability enum behavior."""

    def test_capability_is_string_enum(self):
        """Capability should be a string enum."""
        from modeling_studio.data.labels import Capability

        assert issubclass(Capability, str)

    def test_capability_str_returns_value(self):
        """str(capability) should return the value."""
        from modeling_studio.data.labels import Capability

        assert str(Capability.NER_GENERAL) == "ner_general"

    def test_capability_values_lowercase(self):
        """All capability values should be lowercase."""
        from modeling_studio.data.labels import Capability

        for cap in Capability:
            assert cap.value == cap.value.lower()

    def test_capability_from_string(self):
        """Capability should be constructible from string value."""
        from modeling_studio.data.labels import Capability

        cap = Capability("ner_general")
        assert cap == Capability.NER_GENERAL


# =============================================================================
# CAPABILITY_TO_LABELS Mapping Tests
# =============================================================================


class TestCapabilityToLabelsMapping:
    """Test CAPABILITY_TO_LABELS mapping."""

    def test_capability_to_labels_has_all_capabilities(self):
        """CAPABILITY_TO_LABELS should have entry for each Capability."""
        from modeling_studio.data.labels import Capability, CAPABILITY_TO_LABELS

        for cap in Capability:
            assert cap in CAPABILITY_TO_LABELS

    def test_capability_to_labels_embedding_is_none(self):
        """EMBEDDING capability should map to None."""
        from modeling_studio.data.labels import Capability, CAPABILITY_TO_LABELS

        assert CAPABILITY_TO_LABELS[Capability.EMBEDDING] is None

    def test_capability_to_labels_others_not_none(self):
        """Non-embedding capabilities should map to LabelSchema."""
        from modeling_studio.data.labels import (
            Capability,
            CAPABILITY_TO_LABELS,
            LabelSchema,
        )

        for cap in Capability:
            if cap != Capability.EMBEDDING:
                assert CAPABILITY_TO_LABELS[cap] is not None
                assert isinstance(CAPABILITY_TO_LABELS[cap], LabelSchema)


# =============================================================================
# ALL_LABEL_SCHEMAS Mapping Tests
# =============================================================================


class TestAllLabelSchemasMapping:
    """Test ALL_LABEL_SCHEMAS mapping."""

    def test_all_label_schemas_has_expected_keys(self):
        """ALL_LABEL_SCHEMAS should have all expected schema keys."""
        from modeling_studio.data.labels import ALL_LABEL_SCHEMAS

        expected_keys = {
            "ner_general",
            "sentiment",
            "emotions",
            "emotions_legacy",
            "emotions_reduced",
            "emotions_familyos",
            "safety_generic",
            "nli",
            "temporal",
            "ner_family",
            "ingress",
            "safety_familyos",
            "safety_subcategories",
            "relation",
            "intent",
        }
        for key in expected_keys:
            assert key in ALL_LABEL_SCHEMAS

    def test_all_label_schemas_values_are_label_schemas(self):
        """ALL_LABEL_SCHEMAS values should all be LabelSchema instances."""
        from modeling_studio.data.labels import ALL_LABEL_SCHEMAS, LabelSchema

        for key, schema in ALL_LABEL_SCHEMAS.items():
            assert isinstance(schema, LabelSchema), f"{key} is not a LabelSchema"


# =============================================================================
# Safety Subcategories Tests (Issue 3.6.8)
# =============================================================================


class TestSafetySubcategories:
    """Test SAFETY_SUBCATEGORIES label schema."""

    def test_safety_subcategories_13_labels(self):
        """SAFETY_SUBCATEGORIES should have 13 labels."""
        from modeling_studio.data.labels import SAFETY_SUBCATEGORIES

        assert SAFETY_SUBCATEGORIES.num_labels == 13

    def test_safety_subcategories_has_none(self):
        """SAFETY_SUBCATEGORIES should have 'none' at position 0."""
        from modeling_studio.data.labels import SAFETY_SUBCATEGORIES

        assert "none" in SAFETY_SUBCATEGORIES.label2id
        assert SAFETY_SUBCATEGORIES.label2id["none"] == 0


class TestSubcategoryToBandMapping:
    """Test SUBCATEGORY_TO_BAND_ID mapping."""

    def test_subcategory_to_band_has_all_subcategories(self):
        """SUBCATEGORY_TO_BAND_ID should have all 13 subcategories."""
        from modeling_studio.data.labels import SUBCATEGORY_TO_BAND_ID

        assert len(SUBCATEGORY_TO_BAND_ID) == 13

    def test_subcategory_to_band_values_are_band_ids(self):
        """SUBCATEGORY_TO_BAND_ID values should be 0-3."""
        from modeling_studio.data.labels import SUBCATEGORY_TO_BAND_ID

        for subcat_id, band_id in SUBCATEGORY_TO_BAND_ID.items():
            assert 0 <= band_id <= 3


class TestBandToSubcategoryMapping:
    """Test BAND_TO_SUBCATEGORY_IDS mapping."""

    def test_band_to_subcategory_has_all_bands(self):
        """BAND_TO_SUBCATEGORY_IDS should have all 4 bands."""
        from modeling_studio.data.labels import BAND_TO_SUBCATEGORY_IDS

        assert len(BAND_TO_SUBCATEGORY_IDS) == 4
        assert 0 in BAND_TO_SUBCATEGORY_IDS  # GREEN
        assert 1 in BAND_TO_SUBCATEGORY_IDS  # AMBER
        assert 2 in BAND_TO_SUBCATEGORY_IDS  # RED
        assert 3 in BAND_TO_SUBCATEGORY_IDS  # CRISIS

    def test_band_to_subcategory_green_has_only_none(self):
        """GREEN band should only have 'none' subcategory."""
        from modeling_studio.data.labels import BAND_TO_SUBCATEGORY_IDS

        assert BAND_TO_SUBCATEGORY_IDS[0] == [0]

    def test_band_to_subcategory_amber_has_4(self):
        """AMBER band should have 4 subcategories."""
        from modeling_studio.data.labels import BAND_TO_SUBCATEGORY_IDS

        assert len(BAND_TO_SUBCATEGORY_IDS[1]) == 4

    def test_band_to_subcategory_red_has_4(self):
        """RED band should have 4 subcategories."""
        from modeling_studio.data.labels import BAND_TO_SUBCATEGORY_IDS

        assert len(BAND_TO_SUBCATEGORY_IDS[2]) == 4

    def test_band_to_subcategory_crisis_has_4(self):
        """CRISIS band should have 4 subcategories."""
        from modeling_studio.data.labels import BAND_TO_SUBCATEGORY_IDS

        assert len(BAND_TO_SUBCATEGORY_IDS[3]) == 4


class TestBandSubcategoryConsistency:
    """Test consistency between band/subcategory mappings."""

    def test_mappings_are_inverses(self):
        """SUBCATEGORY_TO_BAND_ID and BAND_TO_SUBCATEGORY_IDS should be consistent."""
        from modeling_studio.data.labels import (
            SUBCATEGORY_TO_BAND_ID,
            BAND_TO_SUBCATEGORY_IDS,
        )

        # For each subcategory -> band mapping
        for subcat_id, band_id in SUBCATEGORY_TO_BAND_ID.items():
            # The subcategory should be in the band's list
            assert subcat_id in BAND_TO_SUBCATEGORY_IDS[band_id]

        # For each band -> subcategory list
        for band_id, subcat_ids in BAND_TO_SUBCATEGORY_IDS.items():
            for subcat_id in subcat_ids:
                # The subcategory should map back to this band
                assert SUBCATEGORY_TO_BAND_ID[subcat_id] == band_id


# =============================================================================
# Legacy Emotions Labels Tests
# =============================================================================


class TestEmotionsLegacy:
    """Test legacy EMOTIONS_LABELS (32 classes)."""

    def test_emotions_legacy_32_labels(self):
        """EMOTIONS_LABELS (legacy) should have 32 classes."""
        from modeling_studio.data.labels import EMOTIONS_LABELS

        assert EMOTIONS_LABELS.num_labels == 32


class TestEmotionsReduced:
    """Test EMOTIONS_REDUCED_LABELS (12 classes)."""

    def test_emotions_reduced_12_labels(self):
        """EMOTIONS_REDUCED_LABELS should have 12 classes."""
        from modeling_studio.data.labels import EMOTIONS_REDUCED_LABELS

        assert EMOTIONS_REDUCED_LABELS.num_labels == 12
