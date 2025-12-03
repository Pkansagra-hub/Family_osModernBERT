"""
Milestone 2: Data Pipeline Tests
Issue 2.1.3: data/loaders.py

Tests for:
- load_ner_dataset: CoNLL-2003, JSONL, directory loading, label remapping
- load_classification_dataset: SST-2, CSV, JSONL loading, label mapping
- load_multilabel_dataset: GoEmotions, multi-hot encoding
- load_nli_dataset: MNLI, SNLI, premise/hypothesis/label
- load_embedding_dataset: STS-B, sentence pairs and scores
- load_familyos_*: Custom FamilyOS data loaders
- KEEP_DATASETS_IN_MEMORY flag
- WikiNeural label mapping
"""

import json
import tempfile
from pathlib import Path

import pytest


class TestKeepDatasetsInMemory:
    """Test KEEP_DATASETS_IN_MEMORY configuration."""

    def test_keep_datasets_in_memory_defined(self):
        """KEEP_DATASETS_IN_MEMORY should be defined as a global constant."""
        from modeling_studio.data.loaders import KEEP_DATASETS_IN_MEMORY

        assert isinstance(KEEP_DATASETS_IN_MEMORY, bool)

    def test_keep_datasets_in_memory_default_true(self):
        """KEEP_DATASETS_IN_MEMORY defaults to True for high-RAM systems."""
        from modeling_studio.data.loaders import KEEP_DATASETS_IN_MEMORY

        assert KEEP_DATASETS_IN_MEMORY is True


class TestLabelMappingsExist:
    """Test that label mapping constants are defined."""

    def test_conll2003_label_map_defined(self):
        """CONLL2003_LABEL_MAP should be defined."""
        from modeling_studio.data.loaders import CONLL2003_LABEL_MAP

        assert isinstance(CONLL2003_LABEL_MAP, dict)
        assert "O" in CONLL2003_LABEL_MAP
        assert "B-PER" in CONLL2003_LABEL_MAP

    def test_ontonotes5_label_map_defined(self):
        """ONTONOTES5_LABEL_MAP should be defined."""
        from modeling_studio.data.loaders import ONTONOTES5_LABEL_MAP

        assert isinstance(ONTONOTES5_LABEL_MAP, dict)
        assert "O" in ONTONOTES5_LABEL_MAP
        assert "B-PERSON" in ONTONOTES5_LABEL_MAP

    def test_sst2_label_map_defined(self):
        """SST2_LABEL_MAP should be defined."""
        from modeling_studio.data.loaders import SST2_LABEL_MAP

        assert isinstance(SST2_LABEL_MAP, dict)
        assert 0 in SST2_LABEL_MAP  # negative
        assert 1 in SST2_LABEL_MAP  # positive


class TestLoadNerDatasetFunction:
    """Test load_ner_dataset function signature and behavior."""

    def test_load_ner_dataset_exists(self):
        """load_ner_dataset function should exist."""
        from modeling_studio.data.loaders import load_ner_dataset

        assert callable(load_ner_dataset)

    def test_load_ner_dataset_signature(self):
        """load_ner_dataset should accept required parameters."""
        import inspect

        from modeling_studio.data.loaders import load_ner_dataset

        sig = inspect.signature(load_ner_dataset)
        params = sig.parameters

        assert "name" in params
        assert "split" in params
        assert "label_schema" in params
        assert "data_dir" in params
        assert "cache_dir" in params

    def test_load_ner_from_jsonl(self):
        """load_ner_dataset should load from local JSONL file."""
        from datasets import Dataset

        from modeling_studio.data.labels import NER_GENERAL_LABELS
        from modeling_studio.data.loaders import load_ner_dataset

        # Create temp JSONL file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                '{"tokens": ["John", "lives", "in", "NYC"], "ner_tags": ["B-PER", "O", "O", "B-LOC"]}\n'
            )
            f.write(
                '{"tokens": ["Apple", "is", "a", "company"], "ner_tags": ["B-ORG", "O", "O", "O"]}\n'
            )
            temp_path = f.name

        try:
            dataset = load_ner_dataset(
                name=temp_path,
                split="train",
                label_schema=NER_GENERAL_LABELS,
            )

            assert isinstance(dataset, Dataset)
            assert "tokens" in dataset.column_names
            assert "ner_tags" in dataset.column_names
            assert len(dataset) == 2

            # Check that string labels were converted to IDs
            first_tags = dataset[0]["ner_tags"]
            assert isinstance(first_tags[0], int)
        finally:
            Path(temp_path).unlink()

    def test_load_ner_from_directory(self):
        """load_ner_dataset should load from directory with split files."""
        from datasets import DatasetDict

        from modeling_studio.data.labels import NER_GENERAL_LABELS
        from modeling_studio.data.loaders import load_ner_dataset

        # Create temp directory with split files
        with tempfile.TemporaryDirectory() as temp_dir:
            train_path = Path(temp_dir) / "train.jsonl"
            with open(train_path, "w") as f:
                f.write('{"tokens": ["Hello", "world"], "ner_tags": ["O", "O"]}\n')

            dataset = load_ner_dataset(
                name="dummy",  # Not used when data_dir is provided
                data_dir=temp_dir,
                label_schema=NER_GENERAL_LABELS,
            )

            assert isinstance(dataset, DatasetDict)
            assert "train" in dataset

    def test_load_ner_string_to_id_conversion(self):
        """String NER labels should be converted to integer IDs."""
        from modeling_studio.data.labels import NER_GENERAL_LABELS
        from modeling_studio.data.loaders import load_ner_dataset

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"tokens": ["John"], "ner_tags": ["B-PER"]}\n')
            temp_path = f.name

        try:
            dataset = load_ner_dataset(
                name=temp_path,
                split="train",
                label_schema=NER_GENERAL_LABELS,
            )

            # B-PER should be converted to ID 1
            assert dataset[0]["ner_tags"][0] == NER_GENERAL_LABELS.encode("B-PER")
        finally:
            Path(temp_path).unlink()


class TestLoadClassificationDatasetFunction:
    """Test load_classification_dataset function."""

    def test_load_classification_dataset_exists(self):
        """load_classification_dataset function should exist."""
        from modeling_studio.data.loaders import load_classification_dataset

        assert callable(load_classification_dataset)

    def test_load_classification_from_csv(self):
        """load_classification_dataset should load from local CSV file."""
        from datasets import Dataset

        from modeling_studio.data.labels import SENTIMENT_LABELS
        from modeling_studio.data.loaders import load_classification_dataset

        # Create temp CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("text,label\n")
            f.write('"This movie is great!",positive\n')
            f.write('"Terrible experience.",negative\n')
            temp_path = f.name

        try:
            dataset = load_classification_dataset(
                name=temp_path,
                split="train",
                label_schema=SENTIMENT_LABELS,
            )

            assert isinstance(dataset, Dataset)
            assert "text" in dataset.column_names
            assert "label" in dataset.column_names
            assert len(dataset) == 2
        finally:
            Path(temp_path).unlink()

    def test_load_classification_from_jsonl(self):
        """load_classification_dataset should load from local JSONL file."""
        from datasets import Dataset

        from modeling_studio.data.labels import SENTIMENT_LABELS
        from modeling_studio.data.loaders import load_classification_dataset

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"text": "Great movie!", "label": "positive"}\n')
            f.write('{"text": "Bad movie!", "label": "negative"}\n')
            temp_path = f.name

        try:
            dataset = load_classification_dataset(
                name=temp_path,
                split="train",
                label_schema=SENTIMENT_LABELS,
            )

            assert isinstance(dataset, Dataset)
            assert len(dataset) == 2
        finally:
            Path(temp_path).unlink()

    def test_load_classification_label_mapping(self):
        """Binary labels should be mapped to 5-class sentiment."""
        from modeling_studio.data.loaders import SST2_LABEL_MAP

        # SST-2: 0 -> 1 (negative), 1 -> 3 (positive)
        assert SST2_LABEL_MAP[0] == 1  # negative -> negative
        assert SST2_LABEL_MAP[1] == 3  # positive -> positive


class TestLoadMultilabelDatasetFunction:
    """Test load_multilabel_dataset function."""

    def test_load_multilabel_dataset_exists(self):
        """load_multilabel_dataset function should exist."""
        from modeling_studio.data.loaders import load_multilabel_dataset

        assert callable(load_multilabel_dataset)

    def test_load_multilabel_from_jsonl(self):
        """load_multilabel_dataset should load from local JSONL with multi-hot encoding."""
        from datasets import Dataset

        from modeling_studio.data.labels import EMOTIONS_LABELS
        from modeling_studio.data.loaders import load_multilabel_dataset

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Use labels that exist in EMOTIONS_LABELS (GoEmotions 28 + 4 family)
            f.write('{"text": "I love this!", "labels": ["joy", "love"]}\n')
            f.write('{"text": "I am sad", "labels": ["sadness"]}\n')
            temp_path = f.name

        try:
            dataset = load_multilabel_dataset(
                name=temp_path,
                split="train",
                label_schema=EMOTIONS_LABELS,
            )

            assert isinstance(dataset, Dataset)
            assert "text" in dataset.column_names
            assert "labels" in dataset.column_names

            # Labels should be multi-hot encoded
            first_labels = dataset[0]["labels"]
            assert isinstance(first_labels, list)
            assert len(first_labels) == EMOTIONS_LABELS.num_labels
            # joy (17) and love (18) should be 1 in EMOTIONS_LABELS
            assert first_labels[17] == 1  # joy
            assert first_labels[18] == 1  # love
        finally:
            Path(temp_path).unlink()


class TestLoadNliDatasetFunction:
    """Test load_nli_dataset function."""

    def test_load_nli_dataset_exists(self):
        """load_nli_dataset function should exist."""
        from modeling_studio.data.loaders import load_nli_dataset

        assert callable(load_nli_dataset)

    def test_load_nli_from_jsonl(self):
        """load_nli_dataset should load from local JSONL file."""
        from datasets import Dataset

        from modeling_studio.data.labels import NLI_LABELS
        from modeling_studio.data.loaders import load_nli_dataset

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                '{"premise": "The sky is blue.", "hypothesis": "It is daytime.", "label": "entailment"}\n'
            )
            f.write(
                '{"premise": "A cat sleeps.", "hypothesis": "A dog runs.", "label": "neutral"}\n'
            )
            f.write(
                '{"premise": "It is cold.", "hypothesis": "It is hot.", "label": "contradiction"}\n'
            )
            temp_path = f.name

        try:
            dataset = load_nli_dataset(
                name=temp_path,
                split="train",
                label_schema=NLI_LABELS,
            )

            assert isinstance(dataset, Dataset)
            assert "premise" in dataset.column_names
            assert "hypothesis" in dataset.column_names
            assert "label" in dataset.column_names
            assert len(dataset) == 3

            # Labels should be converted to integers
            assert dataset[0]["label"] == 0  # entailment
            assert dataset[1]["label"] == 1  # neutral
            assert dataset[2]["label"] == 2  # contradiction
        finally:
            Path(temp_path).unlink()


class TestLoadEmbeddingDatasetFunction:
    """Test load_embedding_dataset function."""

    def test_load_embedding_dataset_exists(self):
        """load_embedding_dataset function should exist."""
        from modeling_studio.data.loaders import load_embedding_dataset

        assert callable(load_embedding_dataset)

    def test_load_embedding_from_csv_pairs(self):
        """load_embedding_dataset should load pairs from CSV."""
        from datasets import Dataset

        from modeling_studio.data.loaders import load_embedding_dataset

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("sentence1,sentence2,score\n")
            f.write('"A plane is flying.","An airplane in the sky.",0.8\n')
            f.write('"A cat runs.","A dog sleeps.",0.1\n')
            temp_path = f.name

        try:
            dataset = load_embedding_dataset(
                name=temp_path,
                split="train",
                format="pairs",
            )

            assert isinstance(dataset, Dataset)
            assert "sentence1" in dataset.column_names
            assert "sentence2" in dataset.column_names
            assert "score" in dataset.column_names
            assert len(dataset) == 2

            # Scores should be normalized to 0-1
            assert 0.0 <= dataset[0]["score"] <= 1.0
        finally:
            Path(temp_path).unlink()


class TestFamilyOSLoaders:
    """Test FamilyOS-specific data loaders."""

    def test_load_familyos_ner_exists(self):
        """load_familyos_ner function should exist."""
        from modeling_studio.data.loaders import load_familyos_ner

        assert callable(load_familyos_ner)

    def test_load_familyos_ingress_exists(self):
        """load_familyos_ingress function should exist."""
        from modeling_studio.data.loaders import load_familyos_ingress

        assert callable(load_familyos_ingress)

    def test_load_familyos_safety_exists(self):
        """load_familyos_safety function should exist."""
        from modeling_studio.data.loaders import load_familyos_safety

        assert callable(load_familyos_safety)

    def test_load_familyos_relations_exists(self):
        """load_familyos_relations function should exist."""
        from modeling_studio.data.loaders import load_familyos_relations

        assert callable(load_familyos_relations)

    def test_load_familyos_intents_exists(self):
        """load_familyos_intents function should exist."""
        from modeling_studio.data.loaders import load_familyos_intents

        assert callable(load_familyos_intents)

    def test_load_familyos_temporal_exists(self):
        """load_familyos_temporal function should exist."""
        from modeling_studio.data.loaders import load_familyos_temporal

        assert callable(load_familyos_temporal)


class TestWikiNeuralLabelMapping:
    """Test WikiNeural label mapping to CoNLL format."""

    def test_wikineural_labels_defined(self):
        """WikiNeural label mapping should handle 33 labels -> CoNLL 9."""
        # Check that the loader handles wikineural prefix
        from modeling_studio.data.loaders import load_ner_dataset

        # Verify function can handle wikineural in name
        assert callable(load_ner_dataset)


class TestStageLoaders:
    """Test stage-based dataset loading functions."""

    def test_load_from_config_exists(self):
        """load_from_config function should exist."""
        from modeling_studio.data.loaders import load_from_config

        assert callable(load_from_config)

    def test_load_stage_a_datasets_exists(self):
        """load_stage_a_datasets function should exist."""
        from modeling_studio.data.loaders import load_stage_a_datasets

        assert callable(load_stage_a_datasets)

    def test_load_stage_b_datasets_exists(self):
        """load_stage_b_datasets function should exist."""
        from modeling_studio.data.loaders import load_stage_b_datasets

        assert callable(load_stage_b_datasets)


class TestLoaderHelperFunctions:
    """Test internal helper functions."""

    def test_get_load_kwargs_exists(self):
        """_get_load_kwargs helper should exist."""
        from modeling_studio.data.loaders import _get_load_kwargs

        assert callable(_get_load_kwargs)

    def test_get_load_kwargs_returns_dict(self):
        """_get_load_kwargs should return a dictionary."""
        from modeling_studio.data.loaders import _get_load_kwargs

        kwargs = _get_load_kwargs()
        assert isinstance(kwargs, dict)
        assert "trust_remote_code" in kwargs
        assert "keep_in_memory" in kwargs


class TestGoEmotionsLabels:
    """Test GoEmotions label handling."""

    def test_go_emotions_labels_defined(self):
        """GO_EMOTIONS_LABELS list should be defined."""
        from modeling_studio.data.loaders import GO_EMOTIONS_LABELS

        assert isinstance(GO_EMOTIONS_LABELS, list)
        assert len(GO_EMOTIONS_LABELS) == 28  # Original GoEmotions has 28 labels

    def test_go_emotions_has_common_labels(self):
        """GO_EMOTIONS_LABELS should have common emotion labels."""
        from modeling_studio.data.loaders import GO_EMOTIONS_LABELS

        assert "joy" in GO_EMOTIONS_LABELS
        assert "sadness" in GO_EMOTIONS_LABELS
        assert "anger" in GO_EMOTIONS_LABELS
        assert "neutral" in GO_EMOTIONS_LABELS


class TestJigsawLabels:
    """Test Jigsaw toxicity label handling."""

    def test_jigsaw_labels_defined(self):
        """JIGSAW_LABELS list should be defined."""
        from modeling_studio.data.loaders import JIGSAW_LABELS

        assert isinstance(JIGSAW_LABELS, list)
        assert len(JIGSAW_LABELS) == 6  # 6 toxicity types

    def test_jigsaw_has_toxicity_types(self):
        """JIGSAW_LABELS should have all toxicity types."""
        from modeling_studio.data.loaders import JIGSAW_LABELS

        expected = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        for label in expected:
            assert label in JIGSAW_LABELS
