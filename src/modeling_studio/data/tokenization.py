"""
Tokenization Utilities

This module provides tokenization functions and utilities for
converting preprocessed text to model inputs.

Features:
    - Unified tokenization interface across tasks
    - Subword alignment for token classification
    - Pair encoding for NLI
    - Truncation strategies
    - Special token handling

Tokenizer Configuration:
    - max_length: Maximum sequence length
    - truncation: Truncation strategy
    - padding: Padding strategy (usually done in collator)
    - return_offsets: For NER alignment

Subword Alignment:
    For NER tasks, we need to align word-level labels with
    subword tokens. Strategies:
    - first: Only label first subword
    - all: Label all subwords
    - none: Ignore continuation tokens (-100)

Usage:
    tokenizer = load_tokenizer("answerdotai/ModernBERT-base")

    # Classification
    result = tokenize_for_classification(tokenizer, "Hello world", max_length=128)

    # Token classification (NER)
    result = tokenize_for_token_classification(
        tokenizer,
        tokens=["John", "lives", "in", "NYC"],
        ner_tags=[1, 0, 0, 5],
        max_length=128
    )

    # NLI
    result = tokenize_for_nli(tokenizer, "The sky is blue", "It's daytime", max_length=128)
"""

from __future__ import annotations

import logging
from typing import Literal

from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

# Label to ignore in loss computation (special tokens, padding)
IGNORE_INDEX = -100


def load_tokenizer(
    model_name_or_path: str = "answerdotai/ModernBERT-base",
    use_fast: bool = True,
    add_prefix_space: bool = False,
    **kwargs,
) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    """
    Load a tokenizer from HuggingFace.

    Args:
        model_name_or_path: HuggingFace model name or local path.
            Default: "answerdotai/ModernBERT-base"
        use_fast: Whether to use the fast tokenizer implementation.
            Default: True (recommended for performance)
        add_prefix_space: Whether to add a space prefix (for GPT-style tokenizers).
            Default: False
        **kwargs: Additional arguments passed to AutoTokenizer.from_pretrained()

    Returns:
        PreTrainedTokenizer or PreTrainedTokenizerFast instance.

    Example:
        >>> tokenizer = load_tokenizer("answerdotai/ModernBERT-base")
        >>> tokenizer("Hello world")
        {'input_ids': [...], 'attention_mask': [...]}
    """
    logger.info(f"Loading tokenizer: {model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=use_fast,
        add_prefix_space=add_prefix_space,
        **kwargs,
    )

    logger.info(
        f"Loaded tokenizer: vocab_size={tokenizer.vocab_size}, "
        f"model_max_length={tokenizer.model_max_length}"
    )

    return tokenizer


def tokenize_for_classification(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    text: str,
    max_length: int = 512,
    truncation: bool = True,
    padding: bool | str = False,
    return_tensors: str | None = None,
) -> BatchEncoding:
    """
    Tokenize text for sequence classification tasks.

    Suitable for: sentiment, emotions, safety, ingress, intent classification.

    Args:
        tokenizer: The tokenizer to use.
        text: Input text to tokenize.
        max_length: Maximum sequence length. Default: 512
        truncation: Whether to truncate to max_length. Default: True
        padding: Padding strategy. Default: False (pad in collator)
        return_tensors: Return type ("pt", "np", None). Default: None

    Returns:
        Dictionary with:
            - input_ids: Token IDs
            - attention_mask: Attention mask (1 for real tokens, 0 for padding)

    Example:
        >>> result = tokenize_for_classification(tokenizer, "I love this!", max_length=128)
        >>> assert "input_ids" in result
        >>> assert "attention_mask" in result
    """
    return tokenizer(
        text,
        max_length=max_length,
        truncation=truncation,
        padding=padding,
        return_tensors=return_tensors,
    )


def tokenize_for_multilabel(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    text: str,
    labels: list[int] | None = None,
    num_labels: int | None = None,
    max_length: int = 512,
    truncation: bool = True,
    padding: bool | str = False,
    return_tensors: str | None = None,
) -> BatchEncoding:
    """
    Tokenize text for multi-label classification tasks.

    Suitable for: emotions (GoEmotions), multi-label intent detection.
    Similar to classification but handles multi-hot label encoding.

    Args:
        tokenizer: The tokenizer to use.
        text: Input text to tokenize.
        labels: List of active label indices (e.g., [0, 3, 5] for emotions).
            If None, no labels are returned (inference mode).
        num_labels: Total number of possible labels (for creating multi-hot vector).
            Required if labels is provided.
        max_length: Maximum sequence length. Default: 512
        truncation: Whether to truncate to max_length. Default: True
        padding: Padding strategy. Default: False (pad in collator)
        return_tensors: Return type ("pt", "np", None). Default: None

    Returns:
        Dictionary with:
            - input_ids: Token IDs
            - attention_mask: Attention mask
            - labels: Multi-hot encoded labels (if labels provided)

    Example:
        >>> result = tokenize_for_multilabel(
        ...     tokenizer,
        ...     "I'm so happy and excited!",
        ...     labels=[0, 5],  # joy, excitement
        ...     num_labels=28,
        ...     max_length=128
        ... )
        >>> assert "input_ids" in result
        >>> assert sum(result["labels"]) == 2  # Two active labels
    """
    result = tokenizer(
        text,
        max_length=max_length,
        truncation=truncation,
        padding=padding,
        return_tensors=return_tensors,
    )

    # Convert label indices to multi-hot encoding if labels provided
    if labels is not None:
        if num_labels is None:
            raise ValueError("num_labels must be provided when labels is not None")
        # Create multi-hot vector
        multi_hot = [0.0] * num_labels
        for label_idx in labels:
            if 0 <= label_idx < num_labels:
                multi_hot[label_idx] = 1.0
        result["labels"] = multi_hot

    return result


def tokenize_for_token_classification(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    tokens: list[str],
    ner_tags: list[int] | None = None,
    max_length: int = 512,
    truncation: bool = True,
    padding: bool | str = False,
    label_all_tokens: bool = False,
    return_tensors: str | None = None,
) -> BatchEncoding:
    """
    Tokenize pre-tokenized text for token classification (NER, temporal extraction).

    Handles subword tokenization by aligning word-level labels with subword tokens.
    By default, only the first subword of each word gets the label, continuation
    subwords get IGNORE_INDEX (-100).

    Args:
        tokenizer: The tokenizer to use.
        tokens: List of words (pre-tokenized text).
        ner_tags: List of label IDs for each word. Same length as tokens.
            If None, no labels are returned (inference mode).
        max_length: Maximum sequence length. Default: 512
        truncation: Whether to truncate to max_length. Default: True
        padding: Padding strategy. Default: False (pad in collator)
        label_all_tokens: If True, label all subwords with the word's label.
            If False, only first subword gets label, rest get -100.
            Default: False
        return_tensors: Return type ("pt", "np", None). Default: None

    Returns:
        Dictionary with:
            - input_ids: Token IDs
            - attention_mask: Attention mask
            - labels: Aligned label IDs (if ner_tags provided)
            - word_ids: Word index for each token (useful for decoding)

    Example:
        >>> result = tokenize_for_token_classification(
        ...     tokenizer,
        ...     tokens=["John", "lives", "in", "New", "York"],
        ...     ner_tags=[1, 0, 0, 5, 6],  # B-PER, O, O, B-LOC, I-LOC
        ...     max_length=128
        ... )
        >>> assert len(result["labels"]) == len(result["input_ids"])
    """
    # Tokenize with is_split_into_words=True for pre-tokenized input
    tokenized = tokenizer(
        tokens,
        is_split_into_words=True,
        max_length=max_length,
        truncation=truncation,
        padding=padding,
        return_tensors=return_tensors,
    )

    # Get word IDs for alignment (which word each token belongs to)
    word_ids = tokenized.word_ids()

    # Store word_ids in output (useful for decoding predictions back to words)
    tokenized["word_ids"] = word_ids

    # Align labels with subword tokens if labels are provided
    if ner_tags is not None:
        aligned_labels = align_labels_with_tokens(
            word_ids=word_ids,
            labels=ner_tags,
            label_all_tokens=label_all_tokens,
        )
        tokenized["labels"] = aligned_labels

    return tokenized


def tokenize_for_nli(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    premise: str,
    hypothesis: str,
    max_length: int = 512,
    truncation: bool | str = True,
    padding: bool | str = False,
    return_tensors: str | None = None,
) -> BatchEncoding:
    """
    Tokenize premise-hypothesis pairs for NLI (Natural Language Inference).

    Encodes both sentences with a separator token between them:
    [CLS] premise [SEP] hypothesis [SEP]

    Args:
        tokenizer: The tokenizer to use.
        premise: The premise text.
        hypothesis: The hypothesis text.
        max_length: Maximum total sequence length. Default: 512
        truncation: Truncation strategy. Default: True (truncate to max_length)
            Can also be "only_first", "only_second", or "longest_first"
        padding: Padding strategy. Default: False (pad in collator)
        return_tensors: Return type ("pt", "np", None). Default: None

    Returns:
        Dictionary with:
            - input_ids: Token IDs for [CLS] premise [SEP] hypothesis [SEP]
            - attention_mask: Attention mask
            - token_type_ids: Segment IDs (0 for premise, 1 for hypothesis)
                Note: ModernBERT may not use token_type_ids

    Example:
        >>> result = tokenize_for_nli(
        ...     tokenizer,
        ...     premise="The sky is blue",
        ...     hypothesis="It is daytime",
        ...     max_length=128
        ... )
        >>> assert "input_ids" in result
    """
    return tokenizer(
        premise,
        hypothesis,
        max_length=max_length,
        truncation=truncation,
        padding=padding,
        return_tensors=return_tensors,
    )


def tokenize_for_embedding(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    text: str | list[str],
    max_length: int = 512,
    truncation: bool = True,
    padding: bool | str = False,
    return_tensors: str | None = None,
) -> BatchEncoding:
    """
    Tokenize text for embedding generation.

    Similar to classification tokenization but optimized for embedding tasks.
    Can handle single texts or batches.

    Args:
        tokenizer: The tokenizer to use.
        text: Single text string or list of texts.
        max_length: Maximum sequence length. Default: 512
        truncation: Whether to truncate to max_length. Default: True
        padding: Padding strategy. Default: False
        return_tensors: Return type ("pt", "np", None). Default: None

    Returns:
        Dictionary with:
            - input_ids: Token IDs
            - attention_mask: Attention mask

    Example:
        >>> result = tokenize_for_embedding(tokenizer, "Family dinner was great")
        >>> assert "input_ids" in result
    """
    return tokenizer(
        text,
        max_length=max_length,
        truncation=truncation,
        padding=padding,
        return_tensors=return_tensors,
    )


def tokenize_for_relation(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    text: str,
    entity1: str,
    entity2: str,
    max_length: int = 512,
    truncation: bool = True,
    padding: bool | str = False,
    return_tensors: str | None = None,
    mark_entities: bool = True,
) -> BatchEncoding:
    """
    Tokenize text for relation extraction between two entities.

    Optionally marks entity spans with special markers:
    "Mom took Panda to the park" → "[E1] Mom [/E1] took [E2] Panda [/E2] to the park"

    Args:
        tokenizer: The tokenizer to use.
        text: Input text containing both entities.
        entity1: First entity (subject).
        entity2: Second entity (object).
        max_length: Maximum sequence length. Default: 512
        truncation: Whether to truncate to max_length. Default: True
        padding: Padding strategy. Default: False
        return_tensors: Return type ("pt", "np", None). Default: None
        mark_entities: Whether to add entity markers. Default: True

    Returns:
        Dictionary with:
            - input_ids: Token IDs
            - attention_mask: Attention mask
            - entity1_mask: Mask for entity1 tokens (optional)
            - entity2_mask: Mask for entity2 tokens (optional)

    Example:
        >>> result = tokenize_for_relation(
        ...     tokenizer,
        ...     text="Mom took Panda to the park",
        ...     entity1="Mom",
        ...     entity2="Panda",
        ...     max_length=128
        ... )
        >>> assert "input_ids" in result
    """
    if mark_entities:
        # Mark entities in text (simple string replacement)
        # Note: This is a simple approach; production might need span-based marking
        marked_text = text
        if entity1 in marked_text:
            marked_text = marked_text.replace(entity1, f"[E1] {entity1} [/E1]", 1)
        if entity2 in marked_text:
            marked_text = marked_text.replace(entity2, f"[E2] {entity2} [/E2]", 1)
        text = marked_text

    return tokenizer(
        text,
        max_length=max_length,
        truncation=truncation,
        padding=padding,
        return_tensors=return_tensors,
    )


def align_labels_with_tokens(
    word_ids: list[int | None],
    labels: list[int],
    label_all_tokens: bool = False,
) -> list[int]:
    """
    Align word-level labels with subword tokens.

    When a word is split into multiple subwords, we need to decide how to
    label each subword:
    - label_all_tokens=False: Only first subword gets the label, rest get -100
    - label_all_tokens=True: All subwords get the same label as the word

    Special tokens (CLS, SEP, PAD) get IGNORE_INDEX (-100).

    Args:
        word_ids: List mapping each token to its word index (None for special tokens).
            Obtained from tokenizer.word_ids() after tokenizing with is_split_into_words=True.
        labels: Word-level labels (one per original word).
        label_all_tokens: Whether to label all subwords or just the first.

    Returns:
        List of token-level labels aligned with input_ids.

    Example:
        >>> # "John" tokenized as ["Jo", "##hn"], "York" as ["York"]
        >>> word_ids = [None, 0, 0, 1, 2, 3, 4, None]  # [CLS] Jo ##hn lives in New York [SEP]
        >>> labels = [1, 0, 0, 5, 6]  # B-PER, O, O, B-LOC, I-LOC
        >>> aligned = align_labels_with_tokens(word_ids, labels, label_all_tokens=False)
        >>> # Result: [-100, 1, -100, 0, 0, 5, 6, -100]
    """
    aligned_labels = []
    previous_word_id = None

    for word_id in word_ids:
        if word_id is None:
            # Special token ([CLS], [SEP], [PAD])
            aligned_labels.append(IGNORE_INDEX)
        elif word_id != previous_word_id:
            # First subword of a new word - use the word's label
            if word_id < len(labels):
                aligned_labels.append(labels[word_id])
            else:
                # Word was truncated
                aligned_labels.append(IGNORE_INDEX)
        else:
            # Continuation subword of the same word
            if label_all_tokens:
                # Label all subwords with word's label
                if word_id < len(labels):
                    aligned_labels.append(labels[word_id])
                else:
                    aligned_labels.append(IGNORE_INDEX)
            else:
                # Only first subword gets label, rest are ignored
                aligned_labels.append(IGNORE_INDEX)

        previous_word_id = word_id

    return aligned_labels


def get_tokenize_function(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    task: Literal[
        "classification",
        "token_classification",
        "nli",
        "embedding",
        "relation",
    ],
    max_length: int = 512,
    label_all_tokens: bool = False,
    **kwargs,
):
    """
    Get the appropriate tokenization function for a task.

    This is a convenience function that returns a callable tokenization function
    configured for the specified task.

    Args:
        tokenizer: The tokenizer to use.
        task: Task type. One of:
            - "classification": Sequence classification (sentiment, emotions, etc.)
            - "token_classification": Token classification (NER, temporal)
            - "nli": Natural Language Inference
            - "embedding": Embedding generation
            - "relation": Relation extraction
        max_length: Maximum sequence length. Default: 512
        label_all_tokens: For token_classification, whether to label all subwords.
        **kwargs: Additional arguments passed to the tokenization function.

    Returns:
        Callable tokenization function.

    Example:
        >>> tokenize_fn = get_tokenize_function(tokenizer, task="classification", max_length=128)
        >>> result = tokenize_fn("Hello world")
    """

    def classification_fn(text: str) -> BatchEncoding:
        return tokenize_for_classification(tokenizer, text, max_length=max_length, **kwargs)

    def multilabel_fn(
        text: str, labels: list[int] | None = None, num_labels: int | None = None
    ) -> BatchEncoding:
        return tokenize_for_multilabel(
            tokenizer, text, labels=labels, num_labels=num_labels, max_length=max_length, **kwargs
        )

    def token_classification_fn(
        tokens: list[str], ner_tags: list[int] | None = None
    ) -> BatchEncoding:
        return tokenize_for_token_classification(
            tokenizer,
            tokens,
            ner_tags=ner_tags,
            max_length=max_length,
            label_all_tokens=label_all_tokens,
            **kwargs,
        )

    def nli_fn(premise: str, hypothesis: str) -> BatchEncoding:
        return tokenize_for_nli(tokenizer, premise, hypothesis, max_length=max_length, **kwargs)

    def embedding_fn(text: str | list[str]) -> BatchEncoding:
        return tokenize_for_embedding(tokenizer, text, max_length=max_length, **kwargs)

    def relation_fn(text: str, entity1: str, entity2: str) -> BatchEncoding:
        return tokenize_for_relation(
            tokenizer, text, entity1, entity2, max_length=max_length, **kwargs
        )

    task_to_fn = {
        "classification": classification_fn,
        "multilabel": multilabel_fn,
        "token_classification": token_classification_fn,
        "nli": nli_fn,
        "embedding": embedding_fn,
        "relation": relation_fn,
    }

    if task not in task_to_fn:
        raise ValueError(f"Unknown task: {task}. Valid tasks: {list(task_to_fn.keys())}")

    return task_to_fn[task]


# Export public API
__all__ = [
    "load_tokenizer",
    "tokenize_for_classification",
    "tokenize_for_multilabel",
    "tokenize_for_token_classification",
    "tokenize_for_nli",
    "tokenize_for_embedding",
    "tokenize_for_relation",
    "align_labels_with_tokens",
    "get_tokenize_function",
    "IGNORE_INDEX",
]
