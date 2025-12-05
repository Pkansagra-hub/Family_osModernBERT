"""
ModernBERT v3 Hub Token Initialization.

This module provides semantic centroid initialization for hub token embeddings.
Instead of random initialization, hub tokens are initialized as the centroid
(mean) of semantically related word embeddings from the v2 model.

This gives hub tokens a "semantic head start" by placing them in the correct
neighborhood of the embedding space, reducing training time and improving
convergence.

Algorithm:
    For each hub token ([EMO], [MEM], [REL], [TASK]):
    1. Retrieve seed words (e.g., ["happy", "sad", "angry", ...] for [EMO])
    2. Tokenize each seed word using v2 tokenizer (may produce subwords)
    3. For multi-subword tokens, average the subword embeddings
    4. Compute centroid as mean across all word embeddings
    5. Assign centroid as the hub token's initial embedding

Example:
    >>> from transformers import AutoTokenizer, AutoModel
    >>> v2_tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    >>> v2_model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")
    >>> v2_embeddings = v2_model.embeddings.word_embeddings.weight
    >>>
    >>> # Initialize hub tokens in v3 model
    >>> initialize_hub_tokens_semantic(v3_model, v2_tokenizer, v2_embeddings)
    >>>
    >>> # Verify initialization quality
    >>> verification = verify_hub_token_initialization(v3_model, v2_tokenizer, v2_embeddings)
    >>> print(verification)
    {'[EMO]': 0.9945, '[MEM]': 0.9982, '[REL]': 0.9976, '[TASK]': 0.9933}

Classes:
    None

Functions:
    resize_token_embeddings_aligned: Resize embeddings to hardware-aligned size
    get_aligned_vocab_size: Calculate next aligned vocab size
    verify_padding_tokens_unreachable: Verify padding tokens are never tokenized
    compute_semantic_centroid: Compute centroid of word embeddings
    initialize_hub_tokens_semantic: Initialize hub token embeddings from v2
    verify_hub_token_initialization: Verify initialization quality

Constants:
    None

Deployment Note:
    ModernBERT-base has vocab_size=50265. After add_special_tokens() adds
    4 hub tokens, the tokenizer vocab becomes 50269. However, for GPU/TPU
    efficiency, embeddings should be resized to a multiple of 128 (e.g., 50368
    or 50432). Use resize_token_embeddings_aligned() to handle this alignment.

    Recommended initialization sequence:
    1. Load v2 model and tokenizer
    2. Add hub tokens via tokenizer.add_special_tokens() → vocab_size=50269
    3. Resize model embeddings to aligned size (50368 or 50432)
    4. Initialize hub tokens with semantic centroids
    5. Proceed with Phase 0.5 healing and training

Issue: 1.2.3 - Semantic Centroid Initialization
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

from modeling_studio.models.hub_tokens import HUB_TOKEN_IDS, get_semantic_seeds

logger = logging.getLogger(__name__)


def resize_token_embeddings_aligned(
    model: nn.Module,
    new_vocab_size: int,
    alignment: int = 128,
) -> None:
    """
    Resize token embeddings to align with hardware efficiency requirements.

    ModernBERT-base has vocab_size=50265. After adding 4 hub tokens via
    add_special_tokens(), the tokenizer will have vocab_size=50269. However,
    for GPU/TPU efficiency, we need vocab_size to be a multiple of 128.

    This function resizes the embedding matrix to the target vocab_size
    (e.g., 50368 = 128 * 393) by padding with random initialization.

    Args:
        model: Model with embeddings.word_embeddings attribute
        new_vocab_size: Target vocabulary size (must be >= current size)
        alignment: Alignment factor (default: 128 for GPU efficiency)

    Raises:
        ValueError: If new_vocab_size is smaller than current size
        ValueError: If new_vocab_size is not aligned

    Example:
        >>> # After tokenizer.add_special_tokens() increases vocab to 50269
        >>> resize_token_embeddings_aligned(model, new_vocab_size=50368)
        >>> # Embedding matrix now has shape (50368, 768)

    Note:
        - Newly added embeddings (beyond original tokens + hub tokens) are
          randomly initialized and will never be used during inference
        - This is purely for hardware efficiency (no semantic meaning)
        - Call this AFTER add_special_tokens() but BEFORE semantic init

    Safety:
        Padding tokens (e.g., IDs 50269-50431 when resizing to 50432) are
        NEVER produced by the tokenizer because:
        1. Base tokenizer knows vocab IDs 0-50264 only
        2. Hub tokens explicitly added get IDs 50265-50268
        3. HubTokenizer only uses base_tokenizer + 4 hub token IDs
        4. No tokenization path can produce IDs >= 50269
        Therefore, padding embeddings receive no gradient flow and exist
        purely for hardware alignment.
    """
    if new_vocab_size % alignment != 0:
        raise ValueError(f"new_vocab_size {new_vocab_size} must be divisible by {alignment}")

    # Get current embedding layer
    if not hasattr(model, "embeddings") or not hasattr(model.embeddings, "word_embeddings"):
        raise AttributeError("Model must have model.embeddings.word_embeddings structure")

    embedding_layer = model.embeddings.word_embeddings  # type: ignore
    current_vocab_size = embedding_layer.weight.shape[0]  # type: ignore
    hidden_dim = embedding_layer.weight.shape[1]  # type: ignore

    if new_vocab_size < current_vocab_size:
        raise ValueError(
            f"new_vocab_size {new_vocab_size} < current vocab_size {current_vocab_size}"
        )

    if new_vocab_size == current_vocab_size:
        logger.info(f"Vocab size already aligned at {current_vocab_size}")
        return

    # Create new embedding layer with larger vocab
    new_embeddings = nn.Embedding(new_vocab_size, int(hidden_dim))  # type: ignore

    # Copy existing embeddings
    with torch.no_grad():
        new_embeddings.weight[:current_vocab_size] = embedding_layer.weight  # type: ignore

    # Replace embedding layer
    model.embeddings.word_embeddings = new_embeddings  # type: ignore

    padding_tokens = new_vocab_size - current_vocab_size
    logger.info(
        f"Resized embeddings: {current_vocab_size} → {new_vocab_size} "
        f"(+{padding_tokens} padding tokens, IDs {current_vocab_size}-{new_vocab_size-1})"
    )
    logger.debug(
        f"Padding token IDs {current_vocab_size}-{new_vocab_size-1} are unreachable "
        f"by tokenizer and exist only for hardware alignment"
    )


def get_aligned_vocab_size(base_size: int, alignment: int = 128) -> int:
    """
    Calculate the next aligned vocabulary size.

    Args:
        base_size: Current vocabulary size
        alignment: Alignment factor (default: 128)

    Returns:
        Next multiple of alignment >= base_size

    Example:
        >>> get_aligned_vocab_size(50269, alignment=128)
        50304  # 128 * 393
        >>> get_aligned_vocab_size(50269, alignment=256)
        50432  # 256 * 197 (used in config)
    """
    import math

    return math.ceil(base_size / alignment) * alignment


def verify_padding_tokens_unreachable(
    tokenizer,
    model_vocab_size: int,
) -> dict[str, bool]:
    """
    Verify that padding tokens are unreachable by the tokenizer.

    This function confirms that the tokenizer can never produce token IDs
    in the padding range, ensuring those embeddings receive no gradient flow.

    Args:
        tokenizer: HubTokenizer instance
        model_vocab_size: Size of model's embedding matrix (e.g., 50432)

    Returns:
        Dictionary with safety checks:
        - 'tokenizer_vocab_in_bounds': Tokenizer vocab < model vocab
        - 'hub_tokens_in_bounds': Hub token IDs < model vocab
        - 'padding_range_unreachable': No tokenization produces padding IDs

    Example:
        >>> safety = verify_padding_tokens_unreachable(tokenizer, 50432)
        >>> assert all(safety.values()), "Padding tokens not safe!"
        >>> print(f"Padding range: {tokenizer.vocab_size}-{model_vocab_size-1}")
        Padding range: 50269-50431
    """
    checks = {}

    # Check 1: Tokenizer vocab size < model vocab size
    tokenizer_vocab = tokenizer.base_tokenizer.vocab_size
    checks["tokenizer_vocab_in_bounds"] = tokenizer_vocab < model_vocab_size

    # Check 2: All hub token IDs < model vocab size
    hub_ids = list(HUB_TOKEN_IDS.values())
    checks["hub_tokens_in_bounds"] = all(hid < model_vocab_size for hid in hub_ids)

    # Check 3: Max possible token ID < model vocab size
    max_tokenizer_id = max(tokenizer.base_tokenizer.vocab_size - 1, max(HUB_TOKEN_IDS.values()))
    checks["padding_range_unreachable"] = max_tokenizer_id < model_vocab_size

    # Log results
    padding_start = max_tokenizer_id + 1
    padding_end = model_vocab_size - 1
    padding_count = model_vocab_size - (max_tokenizer_id + 1)

    if all(checks.values()):
        logger.info(
            f"✓ Padding tokens {padding_start}-{padding_end} "
            f"({padding_count} tokens) are unreachable by tokenizer"
        )
    else:
        logger.warning(f"⚠ Padding token safety check failed: {checks}")

    return checks


def compute_semantic_centroid(
    word_list: list[str],
    tokenizer: PreTrainedTokenizer,
    embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the semantic centroid of a list of words.

    The centroid is computed as the mean of word embeddings. For words that
    tokenize into multiple subwords, the subword embeddings are first averaged
    to get a single word embedding.

    Args:
        word_list: List of seed words (e.g., ["happy", "sad", "angry"])
        tokenizer: Tokenizer to encode words (typically v2 tokenizer)
        embeddings: Word embedding matrix (vocab_size, hidden_dim)

    Returns:
        Centroid tensor of shape (hidden_dim,)

    Raises:
        ValueError: If no valid words can be tokenized

    Example:
        >>> seed_words = ["happy", "sad", "angry", "fear", "joy"]
        >>> centroid = compute_semantic_centroid(seed_words, tokenizer, embeddings)
        >>> centroid.shape
        torch.Size([768])
    """
    if not word_list:
        raise ValueError("word_list cannot be empty")

    word_embeddings = []

    for word in word_list:
        try:
            # Tokenize word (may produce multiple subword tokens)
            token_ids = tokenizer.encode(word, add_special_tokens=False)

            if not token_ids:
                logger.warning(f"Word '{word}' produced no tokens, skipping")
                continue

            # Get embeddings for all subword tokens
            subword_embeds = embeddings[token_ids]  # (num_subwords, hidden_dim)

            # Average across subwords to get single word embedding
            word_embed = subword_embeds.mean(dim=0)  # (hidden_dim,)
            word_embeddings.append(word_embed)

        except Exception as e:
            logger.warning(f"Failed to process word '{word}': {e}, skipping")
            continue

    if not word_embeddings:
        raise ValueError(f"No valid words could be tokenized from {word_list}")

    # Stack all word embeddings and compute centroid
    word_embeddings_tensor = torch.stack(word_embeddings)  # (num_words, hidden_dim)
    centroid = word_embeddings_tensor.mean(dim=0)  # (hidden_dim,)

    return centroid


def initialize_hub_tokens_semantic(
    model: nn.Module,
    v2_tokenizer: PreTrainedTokenizer,
    v2_embeddings: torch.Tensor,
) -> None:
    """
    Initialize hub token embeddings using semantic centroids from v2.

    This function updates the hub token embeddings in-place in the v3 model's
    embedding layer. Each hub token is initialized as the centroid of its
    semantic seed words from the v2 embedding space.

    Args:
        model: v3 model with hub tokens (expects model.embeddings.word_embeddings)
        v2_tokenizer: Tokenizer from v2 model
        v2_embeddings: Word embedding matrix from v2 model (vocab_size, 768)

    Raises:
        AttributeError: If model doesn't have expected embedding structure
        ValueError: If hub token IDs are out of bounds

    Example:
        >>> # Load v2 tokenizer and embeddings
        >>> v2_tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        >>> v2_model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")
        >>> v2_embeddings = v2_model.embeddings.word_embeddings.weight
        >>>
        >>> # Initialize hub tokens in v3 model
        >>> initialize_hub_tokens_semantic(v3_model, v2_tokenizer, v2_embeddings)
        >>> # Hub tokens now have semantic initialization instead of random

    Note:
        - This should be called AFTER model weight transfer but BEFORE training
        - Hub token positions must match HUB_TOKEN_IDS (50265-50268)
        - Operates in no_grad context for efficiency
    """
    # Validate model structure
    if not hasattr(model, "embeddings") or not hasattr(model.embeddings, "word_embeddings"):
        raise AttributeError("Model must have model.embeddings.word_embeddings structure")

    # Get embedding layer
    embedding_layer = model.embeddings.word_embeddings  # type: ignore
    vocab_size = embedding_layer.weight.shape[0]  # type: ignore
    hidden_dim = embedding_layer.weight.shape[1]  # type: ignore

    logger.info(
        f"Initializing hub tokens with semantic centroids "
        f"(vocab_size={vocab_size}, hidden_dim={hidden_dim})"
    )

    with torch.no_grad():
        for hub_token, hub_id in HUB_TOKEN_IDS.items():
            # Validate hub token ID is within vocab
            if hub_id >= vocab_size:
                raise ValueError(
                    f"Hub token {hub_token} has ID {hub_id} >= vocab_size {vocab_size}"
                )

            # Get semantic seed words for this hub token
            seed_words = get_semantic_seeds(hub_token)
            logger.debug(f"Initializing {hub_token} (ID={hub_id}) from {seed_words}")

            # Compute semantic centroid
            centroid = compute_semantic_centroid(seed_words, v2_tokenizer, v2_embeddings)

            # Update hub token embedding in-place
            embedding_layer.weight[hub_id] = centroid  # type: ignore

            logger.info(
                f"✓ {hub_token} initialized with centroid from {len(seed_words)} seed words"
            )

    logger.info("Hub token semantic initialization complete")


def verify_hub_token_initialization(
    model: nn.Module,
    v2_tokenizer: PreTrainedTokenizer,
    v2_embeddings: torch.Tensor,
) -> dict[str, float]:
    """
    Verify hub token initialization quality using cosine similarity.

    Computes the cosine similarity between each hub token's embedding and its
    expected semantic centroid. High similarity (>0.99) indicates successful
    initialization.

    Args:
        model: v3 model with initialized hub tokens
        v2_tokenizer: Tokenizer from v2 model
        v2_embeddings: Word embedding matrix from v2 model

    Returns:
        Dictionary mapping hub token names to cosine similarity scores

    Example:
        >>> verification = verify_hub_token_initialization(v3_model, v2_tokenizer, v2_embeddings)
        >>> print(verification)
        {'[EMO]': 0.9945, '[MEM]': 0.9982, '[REL]': 0.9976, '[TASK]': 0.9933}
        >>> assert all(sim > 0.99 for sim in verification.values())

    Note:
        - Similarity should be very close to 1.0 (typically >0.99)
        - Lower similarity indicates potential initialization issues
        - This is a diagnostic tool, not part of the training pipeline
    """
    # Get embedding layer
    if not hasattr(model, "embeddings") or not hasattr(model.embeddings, "word_embeddings"):
        raise AttributeError("Model must have model.embeddings.word_embeddings structure")

    embedding_layer = model.embeddings.word_embeddings
    similarities = {}

    with torch.no_grad():
        for hub_token, hub_id in HUB_TOKEN_IDS.items():
            # Get current hub token embedding
            hub_embedding = embedding_layer.weight[hub_id]  # type: ignore

            # Compute expected centroid
            seed_words = get_semantic_seeds(hub_token)
            expected_centroid = compute_semantic_centroid(seed_words, v2_tokenizer, v2_embeddings)

            # Compute cosine similarity
            cosine_sim = torch.nn.functional.cosine_similarity(
                hub_embedding.unsqueeze(0), expected_centroid.unsqueeze(0)
            ).item()

            similarities[hub_token] = cosine_sim
            logger.debug(
                f"{hub_token} centroid similarity: {cosine_sim:.4f} "
                f"({'✓' if cosine_sim > 0.99 else '✗'})"
            )

    return similarities
