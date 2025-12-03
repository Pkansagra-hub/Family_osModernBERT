"""
Milestone 2: Data Pipeline Tests
Issue 2.1.4: data/tokenization.py

Tests for:
- load_tokenizer: Load ModernBERT tokenizer
- tokenize_for_classification: Returns input_ids and attention_mask
- tokenize_for_multilabel: Returns multi-hot encoded labels
- tokenize_for_token_classification: Returns aligned labels with word_ids
- tokenize_for_nli: Encodes premise-hypothesis pairs
- tokenize_for_embedding: Returns embeddings-ready input
- Subword alignment: first_only and all_tokens modes
- IGNORE_INDEX constant
- Batch tokenization
"""

import pytest


class TestIgnoreIndexConstant:
    """Test IGNORE_INDEX constant."""

    def test_ignore_index_defined(self):
        """IGNORE_INDEX should be defined."""
        from modeling_studio.data.tokenization import IGNORE_INDEX

        assert IGNORE_INDEX is not None

    def test_ignore_index_is_minus_100(self):
        """IGNORE_INDEX should be -100 (standard for PyTorch cross-entropy)."""
        from modeling_studio.data.tokenization import IGNORE_INDEX

        assert IGNORE_INDEX == -100


class TestLoadTokenizerFunction:
    """Test load_tokenizer function."""

    def test_load_tokenizer_exists(self):
        """load_tokenizer function should exist."""
        from modeling_studio.data.tokenization import load_tokenizer

        assert callable(load_tokenizer)

    def test_load_tokenizer_default_model(self):
        """load_tokenizer should load ModernBERT by default."""
        from modeling_studio.data.tokenization import load_tokenizer

        tokenizer = load_tokenizer()
        assert tokenizer is not None
        # ModernBERT tokenizer should have vocab_size
        assert hasattr(tokenizer, "vocab_size")
        assert tokenizer.vocab_size > 0

    def test_load_tokenizer_vocab_size(self):
        """Tokenizer should have expected vocab size for ModernBERT."""
        from modeling_studio.data.tokenization import load_tokenizer

        tokenizer = load_tokenizer("answerdotai/ModernBERT-base")
        # ModernBERT uses a vocab size around 50k
        assert tokenizer.vocab_size > 30000

    def test_load_tokenizer_has_special_tokens(self):
        """Tokenizer should have standard special tokens."""
        from modeling_studio.data.tokenization import load_tokenizer

        tokenizer = load_tokenizer()
        # Most tokenizers have pad, cls/bos, sep/eos tokens
        assert tokenizer.pad_token is not None or tokenizer.pad_token_id is not None


class TestTokenizeForClassification:
    """Test tokenize_for_classification function."""

    def test_tokenize_for_classification_exists(self):
        """tokenize_for_classification function should exist."""
        from modeling_studio.data.tokenization import tokenize_for_classification

        assert callable(tokenize_for_classification)

    def test_tokenize_for_classification_returns_input_ids(self):
        """tokenize_for_classification should return input_ids."""
        from modeling_studio.data.tokenization import (
            load_tokenizer,
            tokenize_for_classification,
        )

        tokenizer = load_tokenizer()
        result = tokenize_for_classification(tokenizer, "Hello world", max_length=128)

        assert "input_ids" in result
        assert len(result["input_ids"]) > 0

    def test_tokenize_for_classification_returns_attention_mask(self):
        """tokenize_for_classification should return attention_mask."""
        from modeling_studio.data.tokenization import (
            load_tokenizer,
            tokenize_for_classification,
        )

        tokenizer = load_tokenizer()
        result = tokenize_for_classification(tokenizer, "Hello world", max_length=128)

        assert "attention_mask" in result
        assert len(result["attention_mask"]) == len(result["input_ids"])

    def test_tokenize_for_classification_truncation(self):
        """tokenize_for_classification should truncate to max_length."""
        from modeling_studio.data.tokenization import (
            load_tokenizer,
            tokenize_for_classification,
        )

        tokenizer = load_tokenizer()
        long_text = "This is a very long text. " * 100
        result = tokenize_for_classification(tokenizer, long_text, max_length=32)

        assert len(result["input_ids"]) <= 32


class TestTokenizeForMultilabel:
    """Test tokenize_for_multilabel function."""

    def test_tokenize_for_multilabel_exists(self):
        """tokenize_for_multilabel function should exist."""
        from modeling_studio.data.tokenization import tokenize_for_multilabel

        assert callable(tokenize_for_multilabel)

    def test_tokenize_for_multilabel_returns_labels(self):
        """tokenize_for_multilabel should return multi-hot labels when provided."""
        from modeling_studio.data.tokenization import (
            load_tokenizer,
            tokenize_for_multilabel,
        )

        tokenizer = load_tokenizer()
        result = tokenize_for_multilabel(
            tokenizer,
            "I am happy and excited!",
            labels=[0, 5],
            num_labels=28,
            max_length=128,
        )

        assert "labels" in result
        assert len(result["labels"]) == 28
        assert result["labels"][0] == 1.0
        assert result["labels"][5] == 1.0
        assert result["labels"][1] == 0.0  # Not in labels list

    def test_tokenize_for_multilabel_num_labels_required(self):
        """tokenize_for_multilabel should raise error if num_labels missing when labels provided."""
        from modeling_studio.data.tokenization import (
            load_tokenizer,
            tokenize_for_multilabel,
        )

        tokenizer = load_tokenizer()

        with pytest.raises(ValueError, match="num_labels must be provided"):
            tokenize_for_multilabel(
                tokenizer,
                "I am happy!",
                labels=[0, 5],
                num_labels=None,  # This should raise
            )


class TestTokenizeForTokenClassification:
    """Test tokenize_for_token_classification function."""

    def test_tokenize_for_token_classification_exists(self):
        """tokenize_for_token_classification function should exist."""
        from modeling_studio.data.tokenization import tokenize_for_token_classification

        assert callable(tokenize_for_token_classification)

    def test_tokenize_for_token_classification_returns_labels(self):
        """tokenize_for_token_classification should return aligned labels."""
        from modeling_studio.data.tokenization import (
            load_tokenizer,
            tokenize_for_token_classification,
        )

        tokenizer = load_tokenizer()
        tokens = ["John", "lives", "in", "NYC"]
        ner_tags = [1, 0, 0, 5]  # B-PER, O, O, B-LOC

        result = tokenize_for_token_classification(
            tokenizer,
            tokens=tokens,
            ner_tags=ner_tags,
            max_length=128,
        )

        assert "labels" in result
        # Labels length should match input_ids length
        assert len(result["labels"]) == len(result["input_ids"])

    def test_tokenize_for_token_classification_word_ids(self):
        """tokenize_for_token_classification should return word_ids."""
        from modeling_studio.data.tokenization import (
            load_tokenizer,
            tokenize_for_token_classification,
        )

        tokenizer = load_tokenizer()
        tokens = ["Hello", "World"]

        result = tokenize_for_token_classification(
            tokenizer,
            tokens=tokens,
            max_length=128,
        )

        assert "word_ids" in result

    def test_subword_alignment_first_only(self):
        """By default, only first subword gets label (continuation gets -100)."""
        from modeling_studio.data.tokenization import (
            IGNORE_INDEX,
            load_tokenizer,
            tokenize_for_token_classification,
        )

        tokenizer = load_tokenizer()
        # A word that likely gets split into subwords
        tokens = ["internationalization"]
        ner_tags = [1]  # Some label

        result = tokenize_for_token_classification(
            tokenizer,
            tokens=tokens,
            ner_tags=ner_tags,
            max_length=128,
            label_all_tokens=False,  # Default
        )

        labels = result["labels"]
        # First subword should have label 1
        # Find first non-special token label
        non_special_labels = [l for l in labels if l != IGNORE_INDEX]
        if non_special_labels:
            assert non_special_labels[0] == 1

    def test_subword_alignment_all_tokens(self):
        """When label_all_tokens=True, all subwords get the word's label."""
        from modeling_studio.data.tokenization import (
            IGNORE_INDEX,
            load_tokenizer,
            tokenize_for_token_classification,
        )

        tokenizer = load_tokenizer()
        tokens = ["internationalization"]
        ner_tags = [1]

        result = tokenize_for_token_classification(
            tokenizer,
            tokens=tokens,
            ner_tags=ner_tags,
            max_length=128,
            label_all_tokens=True,
        )

        labels = result["labels"]
        non_special_labels = [l for l in labels if l != IGNORE_INDEX]
        # All non-special labels should be 1
        for label in non_special_labels:
            assert label == 1


class TestTokenizeForNli:
    """Test tokenize_for_nli function."""

    def test_tokenize_for_nli_exists(self):
        """tokenize_for_nli function should exist."""
        from modeling_studio.data.tokenization import tokenize_for_nli

        assert callable(tokenize_for_nli)

    def test_tokenize_for_nli_encodes_pairs(self):
        """tokenize_for_nli should encode premise-hypothesis pairs."""
        from modeling_studio.data.tokenization import load_tokenizer, tokenize_for_nli

        tokenizer = load_tokenizer()
        result = tokenize_for_nli(
            tokenizer,
            premise="The sky is blue.",
            hypothesis="It is daytime.",
            max_length=128,
        )

        assert "input_ids" in result
        assert "attention_mask" in result

    def test_tokenize_for_nli_separator(self):
        """tokenize_for_nli should have separator token between sentences."""
        from modeling_studio.data.tokenization import load_tokenizer, tokenize_for_nli

        tokenizer = load_tokenizer()
        result = tokenize_for_nli(
            tokenizer,
            premise="Hello",
            hypothesis="World",
            max_length=128,
        )

        # The separator token should be in the input_ids
        input_ids = result["input_ids"]
        sep_token_id = tokenizer.sep_token_id or tokenizer.eos_token_id

        if sep_token_id is not None:
            assert sep_token_id in input_ids


class TestTokenizeForEmbedding:
    """Test tokenize_for_embedding function."""

    def test_tokenize_for_embedding_exists(self):
        """tokenize_for_embedding function should exist."""
        from modeling_studio.data.tokenization import tokenize_for_embedding

        assert callable(tokenize_for_embedding)

    def test_tokenize_for_embedding_single_text(self):
        """tokenize_for_embedding should handle single text."""
        from modeling_studio.data.tokenization import (
            load_tokenizer,
            tokenize_for_embedding,
        )

        tokenizer = load_tokenizer()
        result = tokenize_for_embedding(
            tokenizer,
            text="Family dinner was great",
            max_length=128,
        )

        assert "input_ids" in result
        assert "attention_mask" in result

    def test_tokenize_for_embedding_batch(self):
        """tokenize_for_embedding should handle batch of texts."""
        from modeling_studio.data.tokenization import (
            load_tokenizer,
            tokenize_for_embedding,
        )

        tokenizer = load_tokenizer()
        result = tokenize_for_embedding(
            tokenizer,
            text=["Hello world", "Goodbye world"],
            max_length=128,
            padding=True,
        )

        assert "input_ids" in result
        # Should have 2 items in batch
        assert len(result["input_ids"]) == 2


class TestAlignLabelsWithTokens:
    """Test align_labels_with_tokens function."""

    def test_align_labels_with_tokens_exists(self):
        """align_labels_with_tokens function should exist."""
        from modeling_studio.data.tokenization import align_labels_with_tokens

        assert callable(align_labels_with_tokens)

    def test_align_labels_with_tokens_special_tokens(self):
        """Special tokens (None word_id) should get IGNORE_INDEX."""
        from modeling_studio.data.tokenization import (
            IGNORE_INDEX,
            align_labels_with_tokens,
        )

        # Simulated word_ids: [CLS]=None, word0, word0, word1, [SEP]=None
        word_ids = [None, 0, 0, 1, None]
        labels = [1, 2]  # Two words

        aligned = align_labels_with_tokens(word_ids, labels, label_all_tokens=False)

        assert aligned[0] == IGNORE_INDEX  # [CLS]
        assert aligned[4] == IGNORE_INDEX  # [SEP]

    def test_align_labels_first_subword_only(self):
        """With label_all_tokens=False, only first subword gets label."""
        from modeling_studio.data.tokenization import (
            IGNORE_INDEX,
            align_labels_with_tokens,
        )

        # word_ids: None, 0, 0, 1, None (word0 has 2 subwords)
        word_ids = [None, 0, 0, 1, None]
        labels = [1, 2]

        aligned = align_labels_with_tokens(word_ids, labels, label_all_tokens=False)

        assert aligned[1] == 1  # First subword of word0 gets label
        assert aligned[2] == IGNORE_INDEX  # Second subword gets -100
        assert aligned[3] == 2  # First subword of word1

    def test_align_labels_all_subwords(self):
        """With label_all_tokens=True, all subwords get the word's label."""
        from modeling_studio.data.tokenization import (
            IGNORE_INDEX,
            align_labels_with_tokens,
        )

        # word_ids: None, 0, 0, 1, None
        word_ids = [None, 0, 0, 1, None]
        labels = [1, 2]

        aligned = align_labels_with_tokens(word_ids, labels, label_all_tokens=True)

        assert aligned[1] == 1  # First subword
        assert aligned[2] == 1  # Second subword also gets label
        assert aligned[3] == 2


class TestGetTokenizeFunction:
    """Test get_tokenize_function factory."""

    def test_get_tokenize_function_exists(self):
        """get_tokenize_function should exist."""
        from modeling_studio.data.tokenization import get_tokenize_function

        assert callable(get_tokenize_function)

    def test_get_tokenize_function_classification(self):
        """get_tokenize_function should return classification tokenizer."""
        from modeling_studio.data.tokenization import (
            get_tokenize_function,
            load_tokenizer,
        )

        tokenizer = load_tokenizer()
        tokenize_fn = get_tokenize_function(tokenizer, task="classification", max_length=128)

        result = tokenize_fn("Hello world")
        assert "input_ids" in result

    def test_get_tokenize_function_token_classification(self):
        """get_tokenize_function should return token classification tokenizer."""
        from modeling_studio.data.tokenization import (
            get_tokenize_function,
            load_tokenizer,
        )

        tokenizer = load_tokenizer()
        tokenize_fn = get_tokenize_function(tokenizer, task="token_classification", max_length=128)

        result = tokenize_fn(tokens=["Hello", "world"])
        assert "input_ids" in result
        assert "word_ids" in result

    def test_get_tokenize_function_nli(self):
        """get_tokenize_function should return NLI tokenizer."""
        from modeling_studio.data.tokenization import (
            get_tokenize_function,
            load_tokenizer,
        )

        tokenizer = load_tokenizer()
        tokenize_fn = get_tokenize_function(tokenizer, task="nli", max_length=128)

        result = tokenize_fn(premise="Hello", hypothesis="World")
        assert "input_ids" in result

    def test_get_tokenize_function_embedding(self):
        """get_tokenize_function should return embedding tokenizer."""
        from modeling_studio.data.tokenization import (
            get_tokenize_function,
            load_tokenizer,
        )

        tokenizer = load_tokenizer()
        tokenize_fn = get_tokenize_function(tokenizer, task="embedding", max_length=128)

        result = tokenize_fn("Hello world")
        assert "input_ids" in result

    def test_get_tokenize_function_invalid_task(self):
        """get_tokenize_function should raise on invalid task."""
        from modeling_studio.data.tokenization import (
            get_tokenize_function,
            load_tokenizer,
        )

        tokenizer = load_tokenizer()

        with pytest.raises(ValueError, match="Unknown task"):
            get_tokenize_function(tokenizer, task="invalid_task", max_length=128)


class TestTokenizeForRelation:
    """Test tokenize_for_relation function."""

    def test_tokenize_for_relation_exists(self):
        """tokenize_for_relation function should exist."""
        from modeling_studio.data.tokenization import tokenize_for_relation

        assert callable(tokenize_for_relation)

    def test_tokenize_for_relation_marks_entities(self):
        """tokenize_for_relation should mark entities in text."""
        from modeling_studio.data.tokenization import (
            load_tokenizer,
            tokenize_for_relation,
        )

        tokenizer = load_tokenizer()
        result = tokenize_for_relation(
            tokenizer,
            text="Mom took Panda to the park",
            entity1="Mom",
            entity2="Panda",
            max_length=128,
        )

        assert "input_ids" in result
        assert "entity1_mask" in result
        assert "entity2_mask" in result


class TestModuleExports:
    """Test that all public APIs are exported."""

    def test_all_exports_defined(self):
        """__all__ should be defined with public APIs."""
        from modeling_studio.data import tokenization

        assert hasattr(tokenization, "__all__")
        assert "load_tokenizer" in tokenization.__all__
        assert "tokenize_for_classification" in tokenization.__all__
        assert "IGNORE_INDEX" in tokenization.__all__
