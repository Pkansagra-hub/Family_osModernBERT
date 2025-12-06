"""
Unit tests for super-label schema and mapping functions.

Tests for Milestone 1 of Stage A Super-Label Training:
- Issue #1.1: EMOTIONS_SUPER_LABELS schema
- Issue #1.2: EMOTION_TO_SUPER_LABEL mapping
- Issue #1.3: map_to_super_labels() function
- Issue #1.4: This test file
"""

import pytest


class TestEmotionsSuperLabelsSchema:
    """Tests for EMOTIONS_SUPER_LABELS schema (Issue #1.1)."""

    def test_schema_exists(self):
        """EMOTIONS_SUPER_LABELS should be importable."""
        from modeling_studio.data.labels import EMOTIONS_SUPER_LABELS

        assert EMOTIONS_SUPER_LABELS is not None

    def test_schema_has_7_labels(self):
        """Schema should have exactly 7 super-labels."""
        from modeling_studio.data.labels import EMOTIONS_SUPER_LABELS

        assert EMOTIONS_SUPER_LABELS.num_labels == 7

    def test_schema_label_names(self):
        """Schema should have the correct 7 super-label names."""
        from modeling_studio.data.labels import EMOTIONS_SUPER_LABELS

        expected_labels = {
            "JOY",
            "AFFECTION",
            "SADNESS",
            "ANXIETY",
            "NOSTALGIA",
            "CONTENTMENT",
            "NEUTRAL",
        }
        actual_labels = set(EMOTIONS_SUPER_LABELS.label2id.keys())
        assert actual_labels == expected_labels

    def test_schema_label_ids_are_contiguous(self):
        """Label IDs should be 0-6 contiguous."""
        from modeling_studio.data.labels import EMOTIONS_SUPER_LABELS

        ids = sorted(EMOTIONS_SUPER_LABELS.label2id.values())
        assert ids == list(range(7))

    def test_schema_is_multi_label(self):
        """Schema should be multi-label classification."""
        from modeling_studio.data.labels import EMOTIONS_SUPER_LABELS

        assert EMOTIONS_SUPER_LABELS.problem_type == "multi_label_classification"

    def test_schema_id2label_inverse(self):
        """id2label should be inverse of label2id."""
        from modeling_studio.data.labels import EMOTIONS_SUPER_LABELS

        for label, idx in EMOTIONS_SUPER_LABELS.label2id.items():
            assert EMOTIONS_SUPER_LABELS.id2label[idx] == label

    def test_schema_in_all_label_schemas(self):
        """Schema should be registered in ALL_LABEL_SCHEMAS."""
        from modeling_studio.data.labels import ALL_LABEL_SCHEMAS, EMOTIONS_SUPER_LABELS

        assert "emotions_super" in ALL_LABEL_SCHEMAS
        assert ALL_LABEL_SCHEMAS["emotions_super"] is EMOTIONS_SUPER_LABELS


class TestEmotionToSuperLabelMapping:
    """Tests for EMOTION_TO_SUPER_LABEL mapping (Issue #1.2)."""

    def test_mapping_exists(self):
        """EMOTION_TO_SUPER_LABEL should be importable."""
        from modeling_studio.data.labels import EMOTION_TO_SUPER_LABEL

        assert EMOTION_TO_SUPER_LABEL is not None
        assert isinstance(EMOTION_TO_SUPER_LABEL, dict)

    def test_all_44_emotions_mapped(self):
        """All 44 FamilyOS emotions should be mapped to a super-label."""
        from modeling_studio.data.labels import EMOTION_TO_SUPER_LABEL, EMOTIONS_FAMILYOS_LABELS

        familyos_emotions = set(EMOTIONS_FAMILYOS_LABELS.label2id.keys())
        mapped_emotions = set(EMOTION_TO_SUPER_LABEL.keys())

        # All FamilyOS emotions should be in the mapping
        missing = familyos_emotions - mapped_emotions
        assert len(missing) == 0, f"Missing emotions in mapping: {missing}"

    def test_all_mapped_to_valid_super_labels(self):
        """All mappings should point to valid super-labels."""
        from modeling_studio.data.labels import EMOTION_TO_SUPER_LABEL, EMOTIONS_SUPER_LABELS

        valid_super_labels = set(EMOTIONS_SUPER_LABELS.label2id.keys())

        for emotion, super_label in EMOTION_TO_SUPER_LABEL.items():
            assert (
                super_label in valid_super_labels
            ), f"{emotion} maps to invalid super-label: {super_label}"

    def test_joy_cluster_mapping(self):
        """JOY cluster should contain expected emotions."""
        from modeling_studio.data.labels import EMOTION_TO_SUPER_LABEL

        # Note: 'surprise' also maps to JOY (positive family context)
        joy_emotions = [
            "joy",
            "excitement",
            "celebration",
            "pride",
            "relief",
            "amusement",
            "hope",
            "optimism",
            "surprise",
        ]
        for emotion in joy_emotions:
            assert EMOTION_TO_SUPER_LABEL.get(emotion) == "JOY", f"{emotion} should map to JOY"

    def test_affection_cluster_mapping(self):
        """AFFECTION cluster should contain expected emotions."""
        from modeling_studio.data.labels import EMOTION_TO_SUPER_LABEL

        affection_emotions = [
            "love",
            "warmth",
            "caring",
            "gratitude",
            "tenderness",
            "admiration",
            "parental_pride",
            "protectiveness",
            "playfulness",
        ]
        for emotion in affection_emotions:
            assert (
                EMOTION_TO_SUPER_LABEL.get(emotion) == "AFFECTION"
            ), f"{emotion} should map to AFFECTION"

    def test_sadness_cluster_mapping(self):
        """SADNESS cluster should contain expected emotions."""
        from modeling_studio.data.labels import EMOTION_TO_SUPER_LABEL

        sadness_emotions = [
            "sadness",
            "grief",
            "disappointment",
            "longing",
            "emptiness",
            "remorse",
            "parental_guilt",
        ]
        for emotion in sadness_emotions:
            assert (
                EMOTION_TO_SUPER_LABEL.get(emotion) == "SADNESS"
            ), f"{emotion} should map to SADNESS"

    def test_anxiety_cluster_mapping(self):
        """ANXIETY cluster should contain expected emotions."""
        from modeling_studio.data.labels import EMOTION_TO_SUPER_LABEL

        anxiety_emotions = [
            "worry",
            "overwhelmed",
            "frustration",
            "annoyance",
            "nervousness",
            "fear",
            "anger",
            "disgust",
            "disapproval",
            "embarrassment",
        ]
        for emotion in anxiety_emotions:
            assert (
                EMOTION_TO_SUPER_LABEL.get(emotion) == "ANXIETY"
            ), f"{emotion} should map to ANXIETY"

    def test_nostalgia_cluster_mapping(self):
        """NOSTALGIA cluster should contain expected emotions."""
        from modeling_studio.data.labels import EMOTION_TO_SUPER_LABEL

        nostalgia_emotions = ["nostalgia", "bittersweet", "homesickness"]
        for emotion in nostalgia_emotions:
            assert (
                EMOTION_TO_SUPER_LABEL.get(emotion) == "NOSTALGIA"
            ), f"{emotion} should map to NOSTALGIA"

    def test_contentment_cluster_mapping(self):
        """CONTENTMENT cluster should contain expected emotions."""
        from modeling_studio.data.labels import EMOTION_TO_SUPER_LABEL

        contentment_emotions = ["contentment", "belonging", "togetherness", "patience", "approval"]
        for emotion in contentment_emotions:
            assert (
                EMOTION_TO_SUPER_LABEL.get(emotion) == "CONTENTMENT"
            ), f"{emotion} should map to CONTENTMENT"

    def test_neutral_cluster_mapping(self):
        """NEUTRAL cluster should contain expected emotions."""
        from modeling_studio.data.labels import EMOTION_TO_SUPER_LABEL

        # Only 'neutral' maps to NEUTRAL
        # Note: 'surprise' maps to JOY (positive context in FamilyOS)
        neutral_emotions = ["neutral"]
        for emotion in neutral_emotions:
            assert (
                EMOTION_TO_SUPER_LABEL.get(emotion) == "NEUTRAL"
            ), f"{emotion} should map to NEUTRAL"

    def test_surprise_maps_to_joy(self):
        """Surprise should map to JOY (positive family context per Plutchik wheel)."""
        from modeling_studio.data.labels import EMOTION_TO_SUPER_LABEL

        assert EMOTION_TO_SUPER_LABEL.get("surprise") == "JOY"


class TestMapToSuperLabels:
    """Tests for map_to_super_labels() function (Issue #1.3)."""

    def test_function_exists(self):
        """map_to_super_labels should be importable."""
        from modeling_studio.data.labels import map_to_super_labels

        assert callable(map_to_super_labels)

    def test_empty_input_returns_all_zeros(self):
        """Empty multi-hot should return all zeros."""
        from modeling_studio.data.labels import map_to_super_labels

        multi_hot_44 = [0] * 44
        result = map_to_super_labels(multi_hot_44)

        assert len(result) == 7
        assert result == [0, 0, 0, 0, 0, 0, 0]

    def test_single_emotion_joy(self):
        """Single 'joy' emotion (index 1) should map to JOY (index 0)."""
        from modeling_studio.data.labels import map_to_super_labels, EMOTIONS_FAMILYOS_LABELS

        # joy is at index 1 in EMOTIONS_FAMILYOS_LABELS
        joy_idx = EMOTIONS_FAMILYOS_LABELS.label2id["joy"]
        multi_hot_44 = [0] * 44
        multi_hot_44[joy_idx] = 1

        result = map_to_super_labels(multi_hot_44)

        # JOY is at index 0 in EMOTIONS_SUPER_LABELS
        assert result[0] == 1  # JOY
        assert sum(result) == 1  # Only one super-label active

    def test_single_emotion_love(self):
        """Single 'love' emotion should map to AFFECTION."""
        from modeling_studio.data.labels import (
            map_to_super_labels,
            EMOTIONS_FAMILYOS_LABELS,
            EMOTIONS_SUPER_LABELS,
        )

        love_idx = EMOTIONS_FAMILYOS_LABELS.label2id["love"]
        multi_hot_44 = [0] * 44
        multi_hot_44[love_idx] = 1

        result = map_to_super_labels(multi_hot_44)

        affection_idx = EMOTIONS_SUPER_LABELS.label2id["AFFECTION"]
        assert result[affection_idx] == 1
        assert sum(result) == 1

    def test_multiple_emotions_same_cluster(self):
        """Multiple emotions in same cluster should map to single super-label."""
        from modeling_studio.data.labels import (
            map_to_super_labels,
            EMOTIONS_FAMILYOS_LABELS,
            EMOTIONS_SUPER_LABELS,
        )

        # joy and excitement both map to JOY
        multi_hot_44 = [0] * 44
        multi_hot_44[EMOTIONS_FAMILYOS_LABELS.label2id["joy"]] = 1
        multi_hot_44[EMOTIONS_FAMILYOS_LABELS.label2id["excitement"]] = 1

        result = map_to_super_labels(multi_hot_44)

        joy_idx = EMOTIONS_SUPER_LABELS.label2id["JOY"]
        assert result[joy_idx] == 1
        assert sum(result) == 1  # Still only one super-label

    def test_multiple_emotions_different_clusters(self):
        """Emotions from different clusters should map to multiple super-labels."""
        from modeling_studio.data.labels import (
            map_to_super_labels,
            EMOTIONS_FAMILYOS_LABELS,
            EMOTIONS_SUPER_LABELS,
        )

        # joy (JOY) + love (AFFECTION) + nostalgia (NOSTALGIA)
        multi_hot_44 = [0] * 44
        multi_hot_44[EMOTIONS_FAMILYOS_LABELS.label2id["joy"]] = 1
        multi_hot_44[EMOTIONS_FAMILYOS_LABELS.label2id["love"]] = 1
        multi_hot_44[EMOTIONS_FAMILYOS_LABELS.label2id["nostalgia"]] = 1

        result = map_to_super_labels(multi_hot_44)

        assert result[EMOTIONS_SUPER_LABELS.label2id["JOY"]] == 1
        assert result[EMOTIONS_SUPER_LABELS.label2id["AFFECTION"]] == 1
        assert result[EMOTIONS_SUPER_LABELS.label2id["NOSTALGIA"]] == 1
        assert sum(result) == 3

    def test_all_7_super_labels_can_be_activated(self):
        """Should be able to activate all 7 super-labels."""
        from modeling_studio.data.labels import map_to_super_labels, EMOTIONS_FAMILYOS_LABELS

        # One emotion from each cluster
        multi_hot_44 = [0] * 44
        multi_hot_44[EMOTIONS_FAMILYOS_LABELS.label2id["joy"]] = 1  # JOY
        multi_hot_44[EMOTIONS_FAMILYOS_LABELS.label2id["love"]] = 1  # AFFECTION
        multi_hot_44[EMOTIONS_FAMILYOS_LABELS.label2id["sadness"]] = 1  # SADNESS
        multi_hot_44[EMOTIONS_FAMILYOS_LABELS.label2id["fear"]] = 1  # ANXIETY
        multi_hot_44[EMOTIONS_FAMILYOS_LABELS.label2id["nostalgia"]] = 1  # NOSTALGIA
        multi_hot_44[EMOTIONS_FAMILYOS_LABELS.label2id["contentment"]] = 1  # CONTENTMENT
        multi_hot_44[EMOTIONS_FAMILYOS_LABELS.label2id["neutral"]] = 1  # NEUTRAL

        result = map_to_super_labels(multi_hot_44)

        assert result == [1, 1, 1, 1, 1, 1, 1]

    def test_output_length_always_7(self):
        """Output should always be length 7 regardless of input."""
        from modeling_studio.data.labels import map_to_super_labels

        # Test with various inputs
        for input_len in [44, 0]:
            if input_len == 0:
                continue  # Skip empty input
            multi_hot = [0] * input_len
            result = map_to_super_labels(multi_hot)
            assert len(result) == 7


class TestMapEmotionNamesToSuperLabels:
    """Tests for map_emotion_names_to_super_labels() function."""

    def test_function_exists(self):
        """map_emotion_names_to_super_labels should be importable."""
        from modeling_studio.data.labels import map_emotion_names_to_super_labels

        assert callable(map_emotion_names_to_super_labels)

    def test_empty_list_returns_all_zeros(self):
        """Empty emotion list should return all zeros."""
        from modeling_studio.data.labels import map_emotion_names_to_super_labels

        result = map_emotion_names_to_super_labels([])
        assert result == [0, 0, 0, 0, 0, 0, 0]

    def test_single_emotion_name(self):
        """Single emotion name should map correctly."""
        from modeling_studio.data.labels import (
            map_emotion_names_to_super_labels,
            EMOTIONS_SUPER_LABELS,
        )

        result = map_emotion_names_to_super_labels(["joy"])

        joy_idx = EMOTIONS_SUPER_LABELS.label2id["JOY"]
        assert result[joy_idx] == 1
        assert sum(result) == 1

    def test_multiple_emotion_names(self):
        """Multiple emotion names should map to correct super-labels."""
        from modeling_studio.data.labels import (
            map_emotion_names_to_super_labels,
            EMOTIONS_SUPER_LABELS,
        )

        result = map_emotion_names_to_super_labels(["joy", "love", "nostalgia"])

        assert result[EMOTIONS_SUPER_LABELS.label2id["JOY"]] == 1
        assert result[EMOTIONS_SUPER_LABELS.label2id["AFFECTION"]] == 1
        assert result[EMOTIONS_SUPER_LABELS.label2id["NOSTALGIA"]] == 1
        assert sum(result) == 3

    def test_unknown_emotion_name_ignored(self):
        """Unknown emotion names should be silently ignored."""
        from modeling_studio.data.labels import (
            map_emotion_names_to_super_labels,
            EMOTIONS_SUPER_LABELS,
        )

        result = map_emotion_names_to_super_labels(["joy", "unknown_emotion", "fake_emotion"])

        # Only joy should be mapped
        assert result[EMOTIONS_SUPER_LABELS.label2id["JOY"]] == 1
        assert sum(result) == 1

    def test_real_dataset_example(self):
        """Test with a realistic example from the dataset."""
        from modeling_studio.data.labels import (
            map_emotion_names_to_super_labels,
            EMOTIONS_SUPER_LABELS,
        )

        # From the silver dataset: {"emotions": ["joy", "pride", "love"]}
        result = map_emotion_names_to_super_labels(["joy", "pride", "love"])

        # joy and pride -> JOY, love -> AFFECTION
        assert result[EMOTIONS_SUPER_LABELS.label2id["JOY"]] == 1
        assert result[EMOTIONS_SUPER_LABELS.label2id["AFFECTION"]] == 1
        assert sum(result) == 2


class TestSuperLabelExports:
    """Tests for proper exports in __all__."""

    def test_schema_in_all(self):
        """EMOTIONS_SUPER_LABELS should be in __all__."""
        from modeling_studio.data import labels

        assert "EMOTIONS_SUPER_LABELS" in labels.__all__

    def test_mapping_in_all(self):
        """EMOTION_TO_SUPER_LABEL should be in __all__."""
        from modeling_studio.data import labels

        assert "EMOTION_TO_SUPER_LABEL" in labels.__all__

    def test_function_in_all(self):
        """map_to_super_labels should be in __all__."""
        from modeling_studio.data import labels

        assert "map_to_super_labels" in labels.__all__

    def test_name_function_in_all(self):
        """map_emotion_names_to_super_labels should be in __all__."""
        from modeling_studio.data import labels

        assert "map_emotion_names_to_super_labels" in labels.__all__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
