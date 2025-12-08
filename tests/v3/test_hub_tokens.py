"""
Tests for Hub Token Registry (ModernBERT v3.3 Ultra).

Tests the hub token system including registry, capability mappings,
and semantic seed definitions.
"""

import pytest
import torch

from modeling_studio.models.hub_tokens import (
    HUB_TOKEN_IDS,
    HUB_TOKEN_REGISTRY,
    TOKEN_LEVEL_CAPABILITIES,
    HubToken,
    HubTokenSpec,
    get_all_hub_tokens,
    get_capabilities_for_hub,
    get_global_attention_positions,
    get_hub_for_capability,
    get_hub_positions,
    get_hub_token_id,
    get_semantic_seeds,
    print_hub_token_registry,
)


def test_hub_token_registry():
    """Test that hub token registry has all 4 hub tokens defined."""
    assert len(HUB_TOKEN_REGISTRY) == 4
    assert "[EMO]" in HUB_TOKEN_REGISTRY
    assert "[MEM]" in HUB_TOKEN_REGISTRY
    assert "[REL]" in HUB_TOKEN_REGISTRY
    assert "[TASK]" in HUB_TOKEN_REGISTRY


def test_hub_token_positions():
    """Test that hub tokens have correct positions (1-4)."""
    assert HUB_TOKEN_REGISTRY["[EMO]"].position == 1
    assert HUB_TOKEN_REGISTRY["[MEM]"].position == 2
    assert HUB_TOKEN_REGISTRY["[REL]"].position == 3
    assert HUB_TOKEN_REGISTRY["[TASK]"].position == 4


def test_hub_token_ids():
    """Test that hub token IDs are correctly defined."""
    assert HUB_TOKEN_IDS["[EMO]"] == 50368
    assert HUB_TOKEN_IDS["[MEM]"] == 50369
    assert HUB_TOKEN_IDS["[REL]"] == 50370
    assert HUB_TOKEN_IDS["[TASK]"] == 50371


def test_emo_hub_capabilities():
    """Test that [EMO] hub maps to correct capabilities."""
    emo_caps = HUB_TOKEN_REGISTRY["[EMO]"].capabilities
    assert "emotions" in emo_caps
    assert "sentiment" in emo_caps
    assert "safety_generic" in emo_caps
    assert "safety_familyos" in emo_caps
    assert len(emo_caps) == 4


def test_mem_hub_capabilities():
    """Test that [MEM] hub maps to embedding capability."""
    mem_caps = HUB_TOKEN_REGISTRY["[MEM]"].capabilities
    assert "embedding" in mem_caps
    assert len(mem_caps) == 1


def test_rel_hub_capabilities():
    """Test that [REL] hub maps to relational capabilities."""
    rel_caps = HUB_TOKEN_REGISTRY["[REL]"].capabilities
    assert "nli" in rel_caps
    assert "relation" in rel_caps
    assert len(rel_caps) == 2


def test_task_hub_capabilities():
    """Test that [TASK] hub maps to intent/ingress capabilities."""
    task_caps = HUB_TOKEN_REGISTRY["[TASK]"].capabilities
    assert "intent" in task_caps
    assert "ingress" in task_caps
    assert len(task_caps) == 2


def test_semantic_seeds_emo():
    """Test that [EMO] hub has correct semantic seeds."""
    seeds = HUB_TOKEN_REGISTRY["[EMO]"].semantic_seeds
    expected_words = ["happy", "sad", "angry", "fear", "joy", "anxious", "love", "feeling"]
    assert seeds == expected_words


def test_semantic_seeds_mem():
    """Test that [MEM] hub has correct semantic seeds."""
    seeds = HUB_TOKEN_REGISTRY["[MEM]"].semantic_seeds
    expected_words = ["remember", "memory", "past", "history", "recall", "yesterday"]
    assert seeds == expected_words


def test_semantic_seeds_rel():
    """Test that [REL] hub has correct semantic seeds."""
    seeds = HUB_TOKEN_REGISTRY["[REL]"].semantic_seeds
    expected_words = ["family", "mother", "father", "sister", "brother", "parent", "child"]
    assert seeds == expected_words


def test_semantic_seeds_task():
    """Test that [TASK] hub has correct semantic seeds."""
    seeds = HUB_TOKEN_REGISTRY["[TASK]"].semantic_seeds
    expected_words = ["action", "do", "want", "need", "help", "schedule", "plan"]
    assert seeds == expected_words


def test_token_level_capabilities():
    """Test that token-level capabilities are correctly defined."""
    assert len(TOKEN_LEVEL_CAPABILITIES) == 3
    assert "ner_general" in TOKEN_LEVEL_CAPABILITIES
    assert "ner_family" in TOKEN_LEVEL_CAPABILITIES
    assert "temporal" in TOKEN_LEVEL_CAPABILITIES


def test_get_hub_for_capability_emotions():
    """Test hub routing for emotions capability."""
    assert get_hub_for_capability("emotions") == "[EMO]"


def test_get_hub_for_capability_sentiment():
    """Test hub routing for sentiment capability."""
    assert get_hub_for_capability("sentiment") == "[EMO]"


def test_get_hub_for_capability_safety():
    """Test hub routing for safety capabilities."""
    assert get_hub_for_capability("safety_generic") == "[EMO]"
    assert get_hub_for_capability("safety_familyos") == "[EMO]"


def test_get_hub_for_capability_embedding():
    """Test hub routing for embedding capability."""
    assert get_hub_for_capability("embedding") == "[MEM]"


def test_get_hub_for_capability_nli():
    """Test hub routing for NLI capability."""
    assert get_hub_for_capability("nli") == "[REL]"


def test_get_hub_for_capability_relation():
    """Test hub routing for relation capability."""
    assert get_hub_for_capability("relation") == "[REL]"


def test_get_hub_for_capability_intent():
    """Test hub routing for intent capability."""
    assert get_hub_for_capability("intent") == "[TASK]"


def test_get_hub_for_capability_ingress():
    """Test hub routing for ingress capability."""
    assert get_hub_for_capability("ingress") == "[TASK]"


def test_get_hub_for_capability_ner_general():
    """Test that ner_general uses CLS (token-level)."""
    assert get_hub_for_capability("ner_general") == "[CLS]"


def test_get_hub_for_capability_ner_family():
    """Test that ner_family uses CLS (token-level)."""
    assert get_hub_for_capability("ner_family") == "[CLS]"


def test_get_hub_for_capability_temporal():
    """Test that temporal uses CLS (token-level)."""
    assert get_hub_for_capability("temporal") == "[CLS]"


def test_get_hub_for_capability_unknown():
    """Test that unknown capabilities fallback to CLS."""
    assert get_hub_for_capability("unknown_capability") == "[CLS]"


def test_get_capabilities_for_hub_emo():
    """Test getting capabilities for [EMO] hub."""
    caps = get_capabilities_for_hub("[EMO]")
    assert "emotions" in caps
    assert "sentiment" in caps
    assert "safety_generic" in caps
    assert "safety_familyos" in caps


def test_get_capabilities_for_hub_mem():
    """Test getting capabilities for [MEM] hub."""
    caps = get_capabilities_for_hub("[MEM]")
    assert "embedding" in caps


def test_get_capabilities_for_hub_rel():
    """Test getting capabilities for [REL] hub."""
    caps = get_capabilities_for_hub("[REL]")
    assert "nli" in caps
    assert "relation" in caps


def test_get_capabilities_for_hub_task():
    """Test getting capabilities for [TASK] hub."""
    caps = get_capabilities_for_hub("[TASK]")
    assert "intent" in caps
    assert "ingress" in caps


def test_get_capabilities_for_hub_invalid():
    """Test getting capabilities for invalid hub token."""
    caps = get_capabilities_for_hub("[INVALID]")
    assert caps == []


def test_get_hub_positions():
    """Test that hub positions include CLS and all hub tokens."""
    positions = get_hub_positions()
    assert positions["[CLS]"] == 0
    assert positions["[EMO]"] == 1
    assert positions["[MEM]"] == 2
    assert positions["[REL]"] == 3
    assert positions["[TASK]"] == 4
    assert len(positions) == 5


def test_get_global_attention_positions():
    """Test that global attention positions are correct."""
    positions = get_global_attention_positions()
    assert positions == [0, 1, 2, 3, 4]
    assert len(positions) == 5


def test_get_semantic_seeds_emo():
    """Test getting semantic seeds for [EMO]."""
    seeds = get_semantic_seeds("[EMO]")
    assert "happy" in seeds
    assert "sad" in seeds
    assert "angry" in seeds
    assert len(seeds) == 8


def test_get_semantic_seeds_mem():
    """Test getting semantic seeds for [MEM]."""
    seeds = get_semantic_seeds("[MEM]")
    assert "remember" in seeds
    assert "memory" in seeds
    assert len(seeds) == 6


def test_get_semantic_seeds_rel():
    """Test getting semantic seeds for [REL]."""
    seeds = get_semantic_seeds("[REL]")
    assert "family" in seeds
    assert "mother" in seeds
    assert len(seeds) == 7


def test_get_semantic_seeds_task():
    """Test getting semantic seeds for [TASK]."""
    seeds = get_semantic_seeds("[TASK]")
    assert "action" in seeds
    assert "do" in seeds
    assert len(seeds) == 7


def test_get_semantic_seeds_invalid():
    """Test getting semantic seeds for invalid hub token."""
    seeds = get_semantic_seeds("[INVALID]")
    assert seeds == []


def test_get_hub_token_id_emo():
    """Test getting token ID for [EMO]."""
    assert get_hub_token_id("[EMO]") == 50368


def test_get_hub_token_id_mem():
    """Test getting token ID for [MEM]."""
    assert get_hub_token_id("[MEM]") == 50369


def test_get_hub_token_id_rel():
    """Test getting token ID for [REL]."""
    assert get_hub_token_id("[REL]") == 50370


def test_get_hub_token_id_task():
    """Test getting token ID for [TASK]."""
    assert get_hub_token_id("[TASK]") == 50371


def test_get_hub_token_id_invalid():
    """Test that invalid hub token raises KeyError."""
    with pytest.raises(KeyError, match="not found in registry"):
        get_hub_token_id("[INVALID]")


def test_get_all_hub_tokens():
    """Test getting all hub tokens."""
    tokens = get_all_hub_tokens()
    assert len(tokens) == 4
    assert "[EMO]" in tokens
    assert "[MEM]" in tokens
    assert "[REL]" in tokens
    assert "[TASK]" in tokens


# ============================================================================
# Issue 1.2.3: Semantic Centroid Initialization Tests
# ============================================================================


def test_compute_semantic_centroid_single_word(mock_tokenizer, mock_embeddings):
    """Test centroid computation with single-word tokens."""
    from modeling_studio.models.hub_initialization_v3 import compute_semantic_centroid

    # Mock tokenizer returns single token ID per word
    mock_tokenizer.encode.side_effect = lambda word, **kwargs: [100] if word == "happy" else [101]

    word_list = ["happy", "sad"]
    centroid = compute_semantic_centroid(word_list, mock_tokenizer, mock_embeddings)

    # Centroid should be average of embeddings at indices 100 and 101
    expected = (mock_embeddings[100] + mock_embeddings[101]) / 2
    assert torch.allclose(centroid, expected, atol=1e-6)


def test_compute_semantic_centroid_multi_subword(mock_tokenizer, mock_embeddings):
    """Test centroid computation with multi-subword tokens."""
    from modeling_studio.models.hub_initialization_v3 import compute_semantic_centroid

    # Mock tokenizer returns multiple token IDs for "happiness"
    def encode_side_effect(word, **kwargs):
        if word == "happiness":
            return [100, 101]  # Multi-subword token
        return [102]  # Single token

    mock_tokenizer.encode.side_effect = encode_side_effect

    word_list = ["happiness", "joy"]
    centroid = compute_semantic_centroid(word_list, mock_tokenizer, mock_embeddings)

    # "happiness" should be averaged across subwords first
    happiness_embed = (mock_embeddings[100] + mock_embeddings[101]) / 2
    joy_embed = mock_embeddings[102]
    expected = (happiness_embed + joy_embed) / 2

    assert torch.allclose(centroid, expected, atol=1e-6)


def test_compute_semantic_centroid_empty_list(mock_tokenizer, mock_embeddings):
    """Test that empty word list raises ValueError."""
    from modeling_studio.models.hub_initialization_v3 import compute_semantic_centroid

    with pytest.raises(ValueError, match="word_list cannot be empty"):
        compute_semantic_centroid([], mock_tokenizer, mock_embeddings)


def test_compute_semantic_centroid_all_oov(mock_tokenizer, mock_embeddings):
    """Test that all OOV words raises ValueError."""
    from modeling_studio.models.hub_initialization_v3 import compute_semantic_centroid

    # Mock tokenizer returns empty list for all words
    mock_tokenizer.encode.return_value = []

    word_list = ["oov1", "oov2", "oov3"]

    with pytest.raises(ValueError, match="No valid words could be tokenized"):
        compute_semantic_centroid(word_list, mock_tokenizer, mock_embeddings)


def test_compute_semantic_centroid_partial_oov(mock_tokenizer, mock_embeddings):
    """Test that partial OOV words are skipped gracefully."""
    from modeling_studio.models.hub_initialization_v3 import compute_semantic_centroid

    # Mock tokenizer returns empty for "oov" words, valid for others
    def encode_side_effect(word, **kwargs):
        if "oov" in word:
            return []  # OOV word
        return [100]  # Valid word

    mock_tokenizer.encode.side_effect = encode_side_effect

    word_list = ["happy", "oov_word", "sad"]
    centroid = compute_semantic_centroid(word_list, mock_tokenizer, mock_embeddings)

    # Should only use "happy" and "sad", skip "oov_word"
    expected = (mock_embeddings[100] + mock_embeddings[100]) / 2
    assert torch.allclose(centroid, expected, atol=1e-6)


def test_initialize_hub_tokens_semantic(mock_model, mock_tokenizer, mock_embeddings):
    """Test hub token initialization updates embeddings correctly."""
    from modeling_studio.models.hub_initialization_v3 import initialize_hub_tokens_semantic

    # Mock tokenizer encode for semantic seeds
    mock_tokenizer.encode.side_effect = lambda word, **kwargs: [hash(word) % 1000]

    # Initialize hub tokens
    initialize_hub_tokens_semantic(mock_model, mock_tokenizer, mock_embeddings)

    # Check that hub token embeddings were updated
    for _hub_token, hub_id in [
        ("[EMO]", 50368),
        ("[MEM]", 50369),
        ("[REL]", 50370),
        ("[TASK]", 50371),
    ]:
        # Embedding should no longer be the default value (all zeros)
        hub_embedding = mock_model.embeddings.word_embeddings.weight[hub_id]
        assert not torch.allclose(hub_embedding, torch.zeros_like(hub_embedding))


def test_initialize_hub_tokens_semantic_invalid_model(mock_tokenizer, mock_embeddings):
    """Test that invalid model structure raises AttributeError."""
    from modeling_studio.models.hub_initialization_v3 import initialize_hub_tokens_semantic

    # Model without embeddings attribute
    class InvalidModel:
        pass

    invalid_model = InvalidModel()  # type: ignore

    with pytest.raises(AttributeError, match="model.embeddings.word_embeddings"):
        initialize_hub_tokens_semantic(invalid_model, mock_tokenizer, mock_embeddings)  # type: ignore


def test_initialize_hub_tokens_semantic_out_of_bounds(mock_tokenizer, mock_embeddings):
    """Test that hub token ID out of bounds raises ValueError."""
    import torch.nn as nn

    from modeling_studio.models.hub_initialization_v3 import initialize_hub_tokens_semantic

    # Create model with small vocab (hub tokens won't fit)
    class SmallVocabModel:
        class Embeddings:
            def __init__(self):
                self.word_embeddings = nn.Embedding(1000, 768)  # Vocab too small

        def __init__(self):
            self.embeddings = self.Embeddings()

    small_model = SmallVocabModel()  # type: ignore

    with pytest.raises(ValueError, match="ID.*>= vocab_size"):
        initialize_hub_tokens_semantic(small_model, mock_tokenizer, mock_embeddings)  # type: ignore


def test_verify_hub_token_initialization(mock_model, mock_tokenizer, mock_embeddings):
    """Test verification of hub token initialization."""
    from modeling_studio.models.hub_initialization_v3 import (
        initialize_hub_tokens_semantic,
        verify_hub_token_initialization,
    )

    # Mock tokenizer encode
    mock_tokenizer.encode.side_effect = lambda word, **kwargs: [hash(word) % 1000]

    # Initialize hub tokens
    initialize_hub_tokens_semantic(mock_model, mock_tokenizer, mock_embeddings)

    # Verify initialization
    similarities = verify_hub_token_initialization(mock_model, mock_tokenizer, mock_embeddings)

    # Check all hub tokens have high similarity (>0.99)
    assert "[EMO]" in similarities
    assert "[MEM]" in similarities
    assert "[REL]" in similarities
    assert "[TASK]" in similarities

    # All similarities should be very close to 1.0 (exact match after initialization)
    for hub_token, sim in similarities.items():
        assert sim > 0.99, f"{hub_token} similarity {sim} < 0.99"


def test_verify_hub_token_initialization_invalid_model(mock_tokenizer, mock_embeddings):
    """Test that verification with invalid model raises AttributeError."""
    from modeling_studio.models.hub_initialization_v3 import verify_hub_token_initialization

    # Model without embeddings attribute
    class InvalidModel:
        pass

    invalid_model = InvalidModel()  # type: ignore

    with pytest.raises(AttributeError, match="model.embeddings.word_embeddings"):
        verify_hub_token_initialization(invalid_model, mock_tokenizer, mock_embeddings)  # type: ignore


def test_compute_semantic_centroid_shape(mock_tokenizer, mock_embeddings):
    """Test that centroid has correct shape (hidden_dim,)."""
    from modeling_studio.models.hub_initialization_v3 import compute_semantic_centroid

    mock_tokenizer.encode.side_effect = lambda word, **kwargs: [100]

    word_list = ["happy", "sad", "angry"]
    centroid = compute_semantic_centroid(word_list, mock_tokenizer, mock_embeddings)

    assert centroid.shape == (768,)
    assert centroid.dtype == mock_embeddings.dtype


def test_initialize_hub_tokens_semantic_no_grad(mock_model, mock_tokenizer, mock_embeddings):
    """Test that initialization operates in no_grad context."""
    from modeling_studio.models.hub_initialization_v3 import initialize_hub_tokens_semantic

    mock_tokenizer.encode.side_effect = lambda word, **kwargs: [100]

    # Enable gradient tracking
    mock_model.embeddings.word_embeddings.weight.requires_grad = True

    # Initialize hub tokens
    initialize_hub_tokens_semantic(mock_model, mock_tokenizer, mock_embeddings)

    # Verify that gradients are not tracked during initialization
    # (we can't directly check if no_grad was used, but we can verify the result)
    assert (
        mock_model.embeddings.word_embeddings.weight.requires_grad is True
    )  # Still requires grad after


def test_resize_token_embeddings_aligned():
    """Test resizing embeddings to hardware-aligned size."""
    import torch.nn as nn

    from modeling_studio.models.hub_initialization_v3 import resize_token_embeddings_aligned

    # Create mock model with embeddings
    class MockModel:
        class Embeddings:
            def __init__(self):
                self.word_embeddings = nn.Embedding(50372, 768)  # After add_special_tokens

        def __init__(self):
            self.embeddings = self.Embeddings()

    model = MockModel()  # type: ignore
    original_weights = model.embeddings.word_embeddings.weight.clone()

    # Resize to 50432 (next multiple of 256 after 50372)
    resize_token_embeddings_aligned(model, new_vocab_size=50432, alignment=256)  # type: ignore

    # Verify new size
    assert model.embeddings.word_embeddings.weight.shape[0] == 50432
    assert model.embeddings.word_embeddings.weight.shape[1] == 768

    # Verify original embeddings preserved
    assert torch.allclose(
        model.embeddings.word_embeddings.weight[:50372], original_weights, atol=1e-6
    )


def test_resize_token_embeddings_aligned_already_aligned():
    """Test resizing when vocab is already aligned."""
    import torch.nn as nn

    from modeling_studio.models.hub_initialization_v3 import resize_token_embeddings_aligned

    class MockModel:
        class Embeddings:
            def __init__(self):
                self.word_embeddings = nn.Embedding(50432, 768)  # Already aligned

        def __init__(self):
            self.embeddings = self.Embeddings()

    model = MockModel()  # type: ignore

    # Should not resize (no-op)
    resize_token_embeddings_aligned(model, new_vocab_size=50432, alignment=256)  # type: ignore

    # Verify size unchanged
    assert model.embeddings.word_embeddings.weight.shape[0] == 50432


def test_resize_token_embeddings_aligned_invalid_alignment():
    """Test that non-aligned vocab size raises error."""
    import torch.nn as nn

    from modeling_studio.models.hub_initialization_v3 import resize_token_embeddings_aligned

    class MockModel:
        class Embeddings:
            def __init__(self):
                self.word_embeddings = nn.Embedding(50372, 768)

        def __init__(self):
            self.embeddings = self.Embeddings()

    model = MockModel()  # type: ignore

    # Should raise ValueError for non-aligned size
    with pytest.raises(ValueError, match="must be divisible by"):
        resize_token_embeddings_aligned(model, new_vocab_size=50372)  # type: ignore


def test_resize_token_embeddings_aligned_shrink():
    """Test that shrinking vocab size raises error."""
    import torch.nn as nn

    from modeling_studio.models.hub_initialization_v3 import resize_token_embeddings_aligned

    class MockModel:
        class Embeddings:
            def __init__(self):
                self.word_embeddings = nn.Embedding(50432, 768)

        def __init__(self):
            self.embeddings = self.Embeddings()

    model = MockModel()  # type: ignore

    # Should raise ValueError when trying to shrink (alignment checked first)
    with pytest.raises(ValueError, match="must be divisible by|< current vocab_size"):
        resize_token_embeddings_aligned(model, new_vocab_size=50304)  # type: ignore


def test_get_aligned_vocab_size():
    """Test calculating aligned vocab size."""
    from modeling_studio.models.hub_initialization_v3 import get_aligned_vocab_size

    # ModernBERT v2 (50368) + 4 hub tokens = 50372
    # Next multiple of 128 is 50432 (128 * 394)
    assert get_aligned_vocab_size(50372, alignment=128) == 50432

    # For 256 alignment (config uses this for better efficiency)
    assert get_aligned_vocab_size(50372, alignment=256) == 50432

    # Already aligned
    assert get_aligned_vocab_size(50432, alignment=256) == 50432

    # Smaller alignment
    assert get_aligned_vocab_size(50372, alignment=64) == 50432

    # Edge case: exact multiple
    assert get_aligned_vocab_size(50432, alignment=128) == 50432


def test_get_aligned_vocab_size_config_value():
    """Test that config uses 256-alignment for 50432."""
    from modeling_studio.models.hub_initialization_v3 import get_aligned_vocab_size

    # Config uses 50432 which is 256-aligned (better than 128 for some accelerators)
    assert 50432 % 256 == 0  # Is 256-aligned
    assert 50432 % 128 == 0  # Also 128-aligned
    assert get_aligned_vocab_size(50372, alignment=256) == 50432


def test_verify_padding_tokens_unreachable():
    """Test that padding tokens are verified as unreachable by tokenizer."""
    from modeling_studio.models.hub_initialization_v3 import verify_padding_tokens_unreachable
    from modeling_studio.models.tokenization_v3 import HubTokenizer

    # Create tokenizer (adds 4 hub tokens to base vocab)
    tokenizer = HubTokenizer()

    # Verify padding tokens are unreachable
    safety_checks = verify_padding_tokens_unreachable(tokenizer, model_vocab_size=50432)

    # All checks should pass
    assert safety_checks["tokenizer_vocab_in_bounds"] is True
    assert safety_checks["hub_tokens_in_bounds"] is True
    assert safety_checks["padding_range_unreachable"] is True

    # Verify tokenizer vocab < model vocab (padding exists)
    assert tokenizer.base_tokenizer.vocab_size < 50432
    # All hub token IDs should be within model vocab
    assert all(hid < 50432 for hid in tokenizer.hub_token_ids.values())


def test_verify_padding_tokens_unreachable_smaller_model():
    """Test padding verification with smaller model vocab."""
    from modeling_studio.models.hub_initialization_v3 import verify_padding_tokens_unreachable
    from modeling_studio.models.tokenization_v3 import HubTokenizer

    tokenizer = HubTokenizer()

    # Model with 50432 vocab (128-aligned)
    safety_checks = verify_padding_tokens_unreachable(tokenizer, model_vocab_size=50432)

    assert all(safety_checks.values())  # Should still be safe
    # Padding range: 50372-50431 (60 tokens)


# ============================================================================
# Fixtures for Semantic Initialization Tests
# ============================================================================


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    from unittest.mock import MagicMock

    tokenizer = MagicMock()
    tokenizer.encode.return_value = [100, 101, 102]
    return tokenizer


@pytest.fixture
def mock_embeddings():
    """Mock embedding matrix for testing."""
    import torch

    # Create embedding matrix (vocab_size=50400, hidden_dim=768)
    torch.manual_seed(42)
    embeddings = torch.randn(50400, 768)
    return embeddings


@pytest.fixture
def mock_model():
    """Mock v3 model for testing."""
    import torch
    import torch.nn as nn

    class MockEmbeddings:
        def __init__(self):
            self.word_embeddings = nn.Embedding(50400, 768)
            # Initialize hub token embeddings to zero for testing
            with torch.no_grad():
                self.word_embeddings.weight[50368:50372] = 0.0

    class MockModel:
        def __init__(self):
            self.embeddings = MockEmbeddings()

    return MockModel()
    """Test getting all hub token strings."""
    tokens = get_all_hub_tokens()
    assert "[EMO]" in tokens
    assert "[MEM]" in tokens
    assert "[REL]" in tokens
    assert "[TASK]" in tokens
    assert len(tokens) == 4


def test_all_12_capabilities_covered():
    """Test that all 12 FamilyOS capabilities are covered."""
    # All 12 capabilities from FamilyOS Unified Encoder
    all_caps = [
        "emotions",
        "sentiment",
        "safety_generic",
        "safety_familyos",
        "embedding",
        "nli",
        "relation",
        "intent",
        "ingress",
        "ner_general",
        "ner_family",
        "temporal",
    ]

    for cap in all_caps:
        hub = get_hub_for_capability(cap)
        assert hub in ["[CLS]", "[EMO]", "[MEM]", "[REL]", "[TASK]"]


def test_hub_token_enum():
    """Test HubToken enum values."""
    assert HubToken.CLS.value == "[CLS]"
    assert HubToken.EMO.value == "[EMO]"
    assert HubToken.MEM.value == "[MEM]"
    assert HubToken.REL.value == "[REL]"
    assert HubToken.TASK.value == "[TASK]"


def test_hub_token_spec_dataclass():
    """Test HubTokenSpec dataclass structure."""
    spec = HUB_TOKEN_REGISTRY["[EMO]"]
    assert isinstance(spec, HubTokenSpec)
    assert hasattr(spec, "token")
    assert hasattr(spec, "position")
    assert hasattr(spec, "capabilities")
    assert hasattr(spec, "semantic_seeds")
    assert hasattr(spec, "description")


def test_print_hub_token_registry():
    """Test that print_hub_token_registry runs without errors."""
    import io
    import sys

    # Capture stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    try:
        print_hub_token_registry()
        output = captured_output.getvalue()

        # Verify output contains expected information
        assert "ModernBERT v3.3 Ultra - Hub Token Registry" in output
        assert "[EMO]" in output
        assert "[MEM]" in output
        assert "[REL]" in output
        assert "[TASK]" in output
        assert "Token-Level Capabilities" in output
        assert "Capability → Hub Token Mapping" in output
    finally:
        sys.stdout = sys.__stdout__


def test_hub_tokenizer_initialization():
    """Test that HubTokenizer initializes correctly."""
    from modeling_studio.models.tokenization_v3 import HubTokenizer

    tokenizer = HubTokenizer()

    # Check hub tokens were added
    assert tokenizer.num_hub_tokens == 4
    assert len(tokenizer.hub_sequence) == 4
    assert "[EMO]" in tokenizer.hub_token_ids
    assert "[MEM]" in tokenizer.hub_token_ids
    assert "[REL]" in tokenizer.hub_token_ids
    assert "[TASK]" in tokenizer.hub_token_ids


def test_hub_tokenizer_single_text():
    """Test tokenization of a single text."""
    from modeling_studio.models.tokenization_v3 import HubTokenizer

    tokenizer = HubTokenizer()
    result = tokenizer("Mom is happy today", max_length=20, padding="max_length")

    # Check result structure
    assert "input_ids" in result
    assert "attention_mask" in result
    assert "hub_token_mask" in result

    # Check shapes
    assert result["input_ids"].shape[0] == 1  # Batch size 1
    assert result["input_ids"].shape[1] == 20  # Max length
    assert result["attention_mask"].shape == result["input_ids"].shape
    assert result["hub_token_mask"].shape == result["input_ids"].shape

    # Check first 5 positions are CLS + 4 hub tokens
    input_ids = result["input_ids"][0].tolist()
    assert input_ids[0] == tokenizer.cls_token_id
    assert input_ids[1] == tokenizer.hub_token_ids["[EMO]"]
    assert input_ids[2] == tokenizer.hub_token_ids["[MEM]"]
    assert input_ids[3] == tokenizer.hub_token_ids["[REL]"]
    assert input_ids[4] == tokenizer.hub_token_ids["[TASK]"]

    # Text starts at position 5
    assert input_ids[5] != tokenizer.pad_token_id  # First text token


def test_hub_tokenizer_batch_text():
    """Test tokenization of multiple texts."""
    from modeling_studio.models.tokenization_v3 import HubTokenizer

    tokenizer = HubTokenizer()
    texts = ["Mom is happy", "Dad is cooking", "Sister is playing"]
    result = tokenizer(texts, max_length=15, padding="max_length")

    # Check batch size
    assert result["input_ids"].shape[0] == 3
    assert result["attention_mask"].shape[0] == 3
    assert result["hub_token_mask"].shape[0] == 3

    # Check all have hub tokens at positions 1-4
    for i in range(3):
        input_ids = result["input_ids"][i].tolist()
        assert input_ids[0] == tokenizer.cls_token_id
        assert input_ids[1] == tokenizer.hub_token_ids["[EMO]"]
        assert input_ids[2] == tokenizer.hub_token_ids["[MEM]"]
        assert input_ids[3] == tokenizer.hub_token_ids["[REL]"]
        assert input_ids[4] == tokenizer.hub_token_ids["[TASK]"]


def test_hub_token_mask():
    """Test that hub_token_mask correctly identifies hub positions."""
    from modeling_studio.models.tokenization_v3 import HubTokenizer

    tokenizer = HubTokenizer()
    result = tokenizer("Hello world", max_length=20, padding="max_length")

    hub_mask = result["hub_token_mask"][0].tolist()

    # Position 0 (CLS) should be 0 (not a hub)
    assert hub_mask[0] == 0

    # Positions 1-4 should be 1 (hub tokens)
    assert hub_mask[1] == 1
    assert hub_mask[2] == 1
    assert hub_mask[3] == 1
    assert hub_mask[4] == 1

    # Position 5+ should be 0 (text or padding)
    assert all(m == 0 for m in hub_mask[5:])


def test_hub_tokenizer_positions():
    """Test get_hub_token_positions method."""
    from modeling_studio.models.tokenization_v3 import HubTokenizer

    tokenizer = HubTokenizer()
    positions = tokenizer.get_hub_token_positions()

    assert positions["[CLS]"] == 0
    assert positions["[EMO]"] == 1
    assert positions["[MEM]"] == 2
    assert positions["[REL]"] == 3
    assert positions["[TASK]"] == 4


def test_text_start_position():
    """Test that text start position is 5."""
    from modeling_studio.models.tokenization_v3 import HubTokenizer

    tokenizer = HubTokenizer()
    assert tokenizer.get_text_start_position() == 5


def test_hub_tokenizer_decode():
    """Test decoding token IDs back to text."""
    from modeling_studio.models.tokenization_v3 import HubTokenizer

    tokenizer = HubTokenizer()
    original_text = "Mom is happy today"

    # Encode
    result = tokenizer(original_text, max_length=30, padding="max_length")

    # Decode (skipping special tokens)
    decoded = tokenizer.decode(result["input_ids"][0], skip_special_tokens=True)

    # Should get back original text (roughly)
    assert "Mom" in decoded or "mom" in decoded.lower()
    assert "happy" in decoded.lower()


def test_hub_tokenizer_batch_decode():
    """Test batch decoding."""
    from modeling_studio.models.tokenization_v3 import HubTokenizer

    tokenizer = HubTokenizer()
    texts = ["Mom is happy", "Dad is cooking"]

    # Encode
    result = tokenizer(texts, max_length=20, padding="max_length")

    # Batch decode
    decoded = tokenizer.batch_decode(result["input_ids"], skip_special_tokens=True)

    assert len(decoded) == 2
    assert "Mom" in decoded[0] or "mom" in decoded[0].lower()
    assert "Dad" in decoded[1] or "dad" in decoded[1].lower()


def test_hub_tokenizer_truncation():
    """Test that truncation works with hub token overhead."""
    from modeling_studio.models.tokenization_v3 import HubTokenizer

    tokenizer = HubTokenizer()

    # Long text that should be truncated
    long_text = " ".join(["word"] * 100)
    result = tokenizer(long_text, max_length=20, truncation=True, padding="max_length")

    # Should be exactly max_length
    assert result["input_ids"].shape[1] == 20

    # First 5 positions should still be CLS + hubs
    input_ids = result["input_ids"][0].tolist()
    assert input_ids[0] == tokenizer.cls_token_id
    assert input_ids[1] == tokenizer.hub_token_ids["[EMO]"]


def test_hub_tokenizer_padding():
    """Test that padding works correctly."""
    from modeling_studio.models.tokenization_v3 import HubTokenizer

    tokenizer = HubTokenizer()

    # Short text
    result = tokenizer("Hi", max_length=20, padding="max_length")

    input_ids = result["input_ids"][0].tolist()
    attention_mask = result["attention_mask"][0].tolist()

    # Count padding tokens
    num_pad = input_ids.count(tokenizer.pad_token_id)
    assert num_pad > 0  # Should have padding

    # Attention mask should be 0 for padding
    for i, token_id in enumerate(input_ids):
        if token_id == tokenizer.pad_token_id:
            assert attention_mask[i] == 0


def test_hub_tokenizer_vocab_size():
    """Test vocab_size property."""
    from modeling_studio.models.tokenization_v3 import HubTokenizer

    tokenizer = HubTokenizer()
    assert tokenizer.vocab_size > 50000  # ModernBERT-base vocab size


def test_hub_tokenizer_repr():
    """Test string representation."""
    from modeling_studio.models.tokenization_v3 import HubTokenizer

    tokenizer = HubTokenizer()
    repr_str = repr(tokenizer)

    assert "HubTokenizer" in repr_str
    assert "hub_tokens=4" in repr_str


def test_hub_tokenizer_save_load(tmp_path):
    """Test saving and loading tokenizer."""
    from modeling_studio.models.tokenization_v3 import HubTokenizer

    # Create and save
    tokenizer = HubTokenizer()
    save_path = tmp_path / "tokenizer"
    tokenizer.save_pretrained(str(save_path))

    # Load
    loaded_tokenizer = HubTokenizer.from_pretrained(str(save_path))

    # Verify loaded tokenizer works
    assert loaded_tokenizer.num_hub_tokens == 4
    assert loaded_tokenizer.hub_token_ids["[EMO]"] == tokenizer.hub_token_ids["[EMO]"]

    # Test encoding with loaded tokenizer
    result = loaded_tokenizer("Test text", max_length=20, padding="max_length")
    assert "input_ids" in result
    assert "hub_token_mask" in result


###############################################################################
# Hub Token Pooler Tests (Issue 1.2.4)
###############################################################################


def test_hub_token_pooler_basic():
    """Test basic hub token pooler extraction."""
    from modeling_studio.models.poolers_v3 import HubTokenPooler

    pooler = HubTokenPooler(hidden_size=768, add_projection=False)

    # Create dummy hidden states [batch, seq_len, hidden]
    batch_size = 2
    seq_len = 128
    hidden_size = 768
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    # Pool hub tokens
    pooled = pooler(hidden_states)

    # Check all hub tokens are extracted
    assert "[CLS]" in pooled
    assert "[EMO]" in pooled
    assert "[MEM]" in pooled
    assert "[REL]" in pooled
    assert "[TASK]" in pooled

    # Check shapes are correct
    assert pooled["[CLS]"].shape == (batch_size, hidden_size)
    assert pooled["[EMO]"].shape == (batch_size, hidden_size)
    assert pooled["[MEM]"].shape == (batch_size, hidden_size)
    assert pooled["[REL]"].shape == (batch_size, hidden_size)
    assert pooled["[TASK]"].shape == (batch_size, hidden_size)


def test_hub_token_pooler_extracts_correct_positions():
    """Test that pooler extracts representations at correct positions (0-4)."""
    from modeling_studio.models.poolers_v3 import HubTokenPooler

    pooler = HubTokenPooler(hidden_size=768)

    # Create hidden states with unique values at each position
    batch_size = 1
    seq_len = 128
    hidden_size = 768
    hidden_states = torch.zeros(batch_size, seq_len, hidden_size)

    # Set unique values at positions 0-4
    hidden_states[:, 0, :] = 1.0  # [CLS]
    hidden_states[:, 1, :] = 2.0  # [EMO]
    hidden_states[:, 2, :] = 3.0  # [MEM]
    hidden_states[:, 3, :] = 4.0  # [REL]
    hidden_states[:, 4, :] = 5.0  # [TASK]

    # Pool hub tokens
    pooled = pooler(hidden_states)

    # Verify correct positions extracted
    assert torch.allclose(pooled["[CLS]"], torch.ones(1, 768) * 1.0)
    assert torch.allclose(pooled["[EMO]"], torch.ones(1, 768) * 2.0)
    assert torch.allclose(pooled["[MEM]"], torch.ones(1, 768) * 3.0)
    assert torch.allclose(pooled["[REL]"], torch.ones(1, 768) * 4.0)
    assert torch.allclose(pooled["[TASK]"], torch.ones(1, 768) * 5.0)


def test_hub_token_pooler_with_projection():
    """Test hub token pooler with projection layers."""
    from modeling_studio.models.poolers_v3 import HubTokenPooler

    pooler = HubTokenPooler(hidden_size=768, add_projection=True)

    # Check projections are created
    assert hasattr(pooler, "projections")
    assert "EMO" in pooler.projections
    assert "MEM" in pooler.projections
    assert "REL" in pooler.projections
    assert "TASK" in pooler.projections

    # Create dummy hidden states
    batch_size = 2
    seq_len = 128
    hidden_size = 768
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    # Pool hub tokens
    pooled = pooler(hidden_states)

    # Check shapes are still correct after projection
    assert pooled["[EMO]"].shape == (batch_size, hidden_size)
    assert pooled["[MEM]"].shape == (batch_size, hidden_size)

    # Projections should change the values (not identity)
    pooled_no_proj = HubTokenPooler(hidden_size=768, add_projection=False)(hidden_states)
    assert not torch.allclose(pooled["[EMO]"], pooled_no_proj["[EMO]"])


def test_hub_token_pooler_get_pooled_for_capability():
    """Test get_pooled_for_capability() returns correct hub for each capability."""
    from modeling_studio.models.poolers_v3 import HubTokenPooler

    pooler = HubTokenPooler(hidden_size=768)

    # Create hidden states with unique values at each position
    batch_size = 1
    seq_len = 128
    hidden_size = 768
    hidden_states = torch.zeros(batch_size, seq_len, hidden_size)

    # Set unique values at positions 0-4
    hidden_states[:, 0, :] = 1.0  # [CLS]
    hidden_states[:, 1, :] = 2.0  # [EMO]
    hidden_states[:, 2, :] = 3.0  # [MEM]
    hidden_states[:, 3, :] = 4.0  # [REL]
    hidden_states[:, 4, :] = 5.0  # [TASK]

    # Test EMO hub capabilities
    assert torch.allclose(
        pooler.get_pooled_for_capability(hidden_states, "emotions"),
        torch.ones(1, 768) * 2.0,
    )
    assert torch.allclose(
        pooler.get_pooled_for_capability(hidden_states, "sentiment"),
        torch.ones(1, 768) * 2.0,
    )
    assert torch.allclose(
        pooler.get_pooled_for_capability(hidden_states, "safety_generic"),
        torch.ones(1, 768) * 2.0,
    )

    # Test MEM hub capabilities
    assert torch.allclose(
        pooler.get_pooled_for_capability(hidden_states, "embedding"),
        torch.ones(1, 768) * 3.0,
    )

    # Test REL hub capabilities
    assert torch.allclose(
        pooler.get_pooled_for_capability(hidden_states, "nli"),
        torch.ones(1, 768) * 4.0,
    )
    assert torch.allclose(
        pooler.get_pooled_for_capability(hidden_states, "relation"),
        torch.ones(1, 768) * 4.0,
    )

    # Test TASK hub capabilities
    assert torch.allclose(
        pooler.get_pooled_for_capability(hidden_states, "intent"),
        torch.ones(1, 768) * 5.0,
    )
    assert torch.allclose(
        pooler.get_pooled_for_capability(hidden_states, "ingress"),
        torch.ones(1, 768) * 5.0,
    )

    # Test token-level capabilities (should use CLS)
    assert torch.allclose(
        pooler.get_pooled_for_capability(hidden_states, "ner_general"),
        torch.ones(1, 768) * 1.0,
    )
    assert torch.allclose(
        pooler.get_pooled_for_capability(hidden_states, "ner_family"),
        torch.ones(1, 768) * 1.0,
    )
    assert torch.allclose(
        pooler.get_pooled_for_capability(hidden_states, "temporal"),
        torch.ones(1, 768) * 1.0,
    )


def test_hub_token_pooler_variable_sequence_length():
    """Test that pooler handles variable sequence lengths correctly."""
    from modeling_studio.models.poolers_v3 import HubTokenPooler

    pooler = HubTokenPooler(hidden_size=768)

    # Test with different sequence lengths
    for seq_len in [32, 64, 128, 256, 512]:
        hidden_states = torch.randn(2, seq_len, 768)
        pooled = pooler(hidden_states)

        # Should always extract positions 0-4 regardless of seq_len
        assert pooled["[CLS]"].shape == (2, 768)
        assert pooled["[EMO]"].shape == (2, 768)
        assert pooled["[MEM]"].shape == (2, 768)
        assert pooled["[REL]"].shape == (2, 768)
        assert pooled["[TASK]"].shape == (2, 768)


def test_combined_pooler_basic():
    """Test basic CombinedPooler functionality."""
    from modeling_studio.models.poolers_v3 import CombinedPooler

    pooler = CombinedPooler(hidden_size=768)

    # Create dummy hidden states
    batch_size = 2
    seq_len = 128
    hidden_size = 768
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)

    # Pool all representations
    pooled = pooler(hidden_states, attention_mask)

    # Check all pooled representations are present
    assert "[CLS]" in pooled
    assert "[CLS]_projected" in pooled
    assert "[EMO]" in pooled
    assert "[MEM]" in pooled
    assert "[REL]" in pooled
    assert "[TASK]" in pooled
    assert "mean" in pooled

    # Check shapes
    assert pooled["[CLS]"].shape == (batch_size, hidden_size)
    assert pooled["[CLS]_projected"].shape == (batch_size, hidden_size)
    assert pooled["mean"].shape == (batch_size, hidden_size)


def test_combined_pooler_mean_excludes_cls_and_hub():
    """Test that mean pooling excludes CLS and hub tokens at positions 0-4."""
    from modeling_studio.models.poolers_v3 import CombinedPooler

    pooler = CombinedPooler(hidden_size=768)

    # Create hidden states where:
    # - Positions 0-4: value = 999.0 (should be excluded)
    # - Positions 5+: value = 1.0 (should be included)
    batch_size = 1
    seq_len = 128
    hidden_size = 768
    hidden_states = torch.ones(batch_size, seq_len, hidden_size)

    # Set positions 0-4 to high values
    hidden_states[:, :5, :] = 999.0

    # All positions are valid (no padding)
    attention_mask = torch.ones(batch_size, seq_len)

    # Pool
    pooled = pooler(hidden_states, attention_mask)

    # Mean should be ~1.0 (excluding positions 0-4)
    # Expected: sum of (seq_len - 5) positions with value 1.0, divided by (seq_len - 5)
    expected_mean = torch.ones(batch_size, hidden_size) * 1.0
    assert torch.allclose(pooled["mean"], expected_mean, atol=1e-5)


def test_combined_pooler_mean_with_padding():
    """Test that mean pooling correctly handles padding tokens."""
    from modeling_studio.models.poolers_v3 import CombinedPooler

    pooler = CombinedPooler(hidden_size=768)

    # Create hidden states
    batch_size = 1
    seq_len = 20
    hidden_size = 768
    hidden_states = torch.ones(batch_size, seq_len, hidden_size)

    # Set positions 0-4 to 999.0 (should be excluded)
    hidden_states[:, :5, :] = 999.0

    # Attention mask: positions 0-9 are valid, 10+ are padding
    attention_mask = torch.zeros(batch_size, seq_len)
    attention_mask[:, :10] = 1.0

    # Pool
    pooled = pooler(hidden_states, attention_mask)

    # Mean should only include positions 5-9 (5 tokens)
    # All with value 1.0
    expected_mean = torch.ones(batch_size, hidden_size) * 1.0
    assert torch.allclose(pooled["mean"], expected_mean, atol=1e-5)


def test_combined_pooler_mean_without_attention_mask():
    """Test mean pooling when attention_mask is None."""
    from modeling_studio.models.poolers_v3 import CombinedPooler

    pooler = CombinedPooler(hidden_size=768)

    # Create hidden states
    batch_size = 1
    seq_len = 15
    hidden_size = 768
    hidden_states = torch.ones(batch_size, seq_len, hidden_size)

    # Set positions 0-4 to 999.0
    hidden_states[:, :5, :] = 999.0

    # Pool without attention mask
    pooled = pooler(hidden_states, attention_mask=None)

    # Mean should only include positions 5+ (simple mean from position 5 onwards)
    # Positions 5-14 (10 tokens) all have value 1.0
    expected_mean = torch.ones(batch_size, hidden_size) * 1.0
    assert torch.allclose(pooled["mean"], expected_mean, atol=1e-5)


def test_combined_pooler_cls_projection():
    """Test that CLS projection applies tanh activation."""
    from modeling_studio.models.poolers_v3 import CombinedPooler

    pooler = CombinedPooler(hidden_size=768)

    # Create hidden states
    batch_size = 2
    seq_len = 128
    hidden_size = 768
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)

    # Pool
    pooled = pooler(hidden_states, attention_mask)

    # CLS_projected should be different from raw CLS (due to Linear + Tanh)
    assert not torch.allclose(pooled["[CLS]"], pooled["[CLS]_projected"])

    # CLS_projected should be in range [-1, 1] (tanh output)
    assert (pooled["[CLS]_projected"] >= -1.0).all()
    assert (pooled["[CLS]_projected"] <= 1.0).all()


###############################################################################
# Hub Routing Tests (Issue 1.2.5)
###############################################################################


def test_hub_router_routing_table():
    """Test that HubRouter has correct routing table with all 12 capabilities."""
    from modeling_studio.models.routing_v3 import HubRouter

    router = HubRouter()

    # Check all 12 capabilities are mapped
    assert len(HubRouter.ROUTING_TABLE) == 12

    # Check EMO hub capabilities (4)
    assert HubRouter.ROUTING_TABLE["emotions"] == ("hub", "[EMO]")
    assert HubRouter.ROUTING_TABLE["sentiment"] == ("hub", "[EMO]")
    assert HubRouter.ROUTING_TABLE["safety_generic"] == ("hub", "[EMO]")
    assert HubRouter.ROUTING_TABLE["safety_familyos"] == ("hub", "[EMO]")

    # Check MEM hub capabilities (1)
    assert HubRouter.ROUTING_TABLE["embedding"] == ("hub", "[MEM]")

    # Check REL hub capabilities (2)
    assert HubRouter.ROUTING_TABLE["nli"] == ("hub", "[REL]")
    assert HubRouter.ROUTING_TABLE["relation"] == ("hub", "[REL]")

    # Check TASK hub capabilities (2)
    assert HubRouter.ROUTING_TABLE["intent"] == ("hub", "[TASK]")
    assert HubRouter.ROUTING_TABLE["ingress"] == ("hub", "[TASK]")

    # Check token-level capabilities (3)
    assert HubRouter.ROUTING_TABLE["ner_general"] == ("token", None)
    assert HubRouter.ROUTING_TABLE["ner_family"] == ("token", None)
    assert HubRouter.ROUTING_TABLE["temporal"] == ("token", None)


def test_hub_router_get_representation_for_capability_hub():
    """Test get_representation_for_capability returns correct hub for hub capabilities."""
    from modeling_studio.models.routing_v3 import HubRouter

    router = HubRouter()

    # Create dummy data
    batch_size = 2
    seq_len = 128
    hidden_size = 768
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    # Create pooled outputs with unique values
    pooled_outputs = {
        "[CLS]": torch.ones(batch_size, hidden_size) * 1.0,
        "[EMO]": torch.ones(batch_size, hidden_size) * 2.0,
        "[MEM]": torch.ones(batch_size, hidden_size) * 3.0,
        "[REL]": torch.ones(batch_size, hidden_size) * 4.0,
        "[TASK]": torch.ones(batch_size, hidden_size) * 5.0,
    }

    # Test EMO hub capabilities
    repr_emo, pool_type = router.get_representation_for_capability(
        hidden_states, pooled_outputs, "emotions"
    )
    assert pool_type == "hub"
    assert torch.allclose(repr_emo, torch.ones(batch_size, hidden_size) * 2.0)

    # Test MEM hub capability
    repr_mem, pool_type = router.get_representation_for_capability(
        hidden_states, pooled_outputs, "embedding"
    )
    assert pool_type == "hub"
    assert torch.allclose(repr_mem, torch.ones(batch_size, hidden_size) * 3.0)

    # Test REL hub capabilities
    repr_rel, pool_type = router.get_representation_for_capability(
        hidden_states, pooled_outputs, "nli"
    )
    assert pool_type == "hub"
    assert torch.allclose(repr_rel, torch.ones(batch_size, hidden_size) * 4.0)

    # Test TASK hub capabilities
    repr_task, pool_type = router.get_representation_for_capability(
        hidden_states, pooled_outputs, "intent"
    )
    assert pool_type == "hub"
    assert torch.allclose(repr_task, torch.ones(batch_size, hidden_size) * 5.0)


def test_hub_router_get_representation_for_capability_token():
    """Test get_representation_for_capability returns full sequence for token-level capabilities."""
    from modeling_studio.models.routing_v3 import HubRouter

    router = HubRouter()

    # Create dummy data
    batch_size = 2
    seq_len = 128
    hidden_size = 768
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    pooled_outputs = {
        "[CLS]": torch.randn(batch_size, hidden_size),
        "[EMO]": torch.randn(batch_size, hidden_size),
        "[MEM]": torch.randn(batch_size, hidden_size),
        "[REL]": torch.randn(batch_size, hidden_size),
        "[TASK]": torch.randn(batch_size, hidden_size),
    }

    # Test token-level capabilities
    for capability in ["ner_general", "ner_family", "temporal"]:
        repr_token, pool_type = router.get_representation_for_capability(
            hidden_states, pooled_outputs, capability
        )
        assert pool_type == "token"
        assert repr_token.shape == (batch_size, seq_len, hidden_size)
        # Should return the same hidden_states object
        assert torch.equal(repr_token, hidden_states)


def test_hub_router_get_hub_gradient_mask():
    """Test gradient mask creation for selective hub training."""
    from modeling_studio.models.routing_v3 import HubRouter

    router = HubRouter()
    batch_size = 4
    device = torch.device("cpu")

    # Test with EMO capabilities active
    active_caps = ["emotions", "sentiment"]
    masks = router.get_hub_gradient_mask(active_caps, batch_size, device)

    # EMO should be trained (emotions, sentiment active)
    assert torch.allclose(masks["[EMO]"], torch.ones(batch_size))

    # Other hubs should be frozen
    assert torch.allclose(masks["[MEM]"], torch.zeros(batch_size))
    assert torch.allclose(masks["[REL]"], torch.zeros(batch_size))
    assert torch.allclose(masks["[TASK]"], torch.zeros(batch_size))

    # Test with multiple hubs active
    active_caps = ["emotions", "embedding", "intent"]
    masks = router.get_hub_gradient_mask(active_caps, batch_size, device)

    assert torch.allclose(masks["[EMO]"], torch.ones(batch_size))  # emotions
    assert torch.allclose(masks["[MEM]"], torch.ones(batch_size))  # embedding
    assert torch.allclose(masks["[REL]"], torch.zeros(batch_size))  # not active
    assert torch.allclose(masks["[TASK]"], torch.ones(batch_size))  # intent

    # Test with no capabilities active
    active_caps = []
    masks = router.get_hub_gradient_mask(active_caps, batch_size, device)

    # All hubs should be frozen
    assert torch.allclose(masks["[EMO]"], torch.zeros(batch_size))
    assert torch.allclose(masks["[MEM]"], torch.zeros(batch_size))
    assert torch.allclose(masks["[REL]"], torch.zeros(batch_size))
    assert torch.allclose(masks["[TASK]"], torch.zeros(batch_size))


def test_capability_head_hub_routing():
    """Test CapabilityHead wrapper correctly routes hub capabilities."""
    from modeling_studio.models.routing_v3 import CapabilityHead

    # Create a simple mock head
    class MockHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(768, 44)

        def forward(self, x):
            return self.linear(x)

    head = MockHead()
    wrapped = CapabilityHead("emotions", head, hidden_size=768)

    # Check routing properties
    assert wrapped.capability == "emotions"
    assert wrapped.pool_type == "hub"
    assert wrapped.hub_token == "[EMO]"

    # Create dummy data
    batch_size = 2
    seq_len = 128
    hidden_size = 768
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    pooled_outputs = {
        "[CLS]": torch.randn(batch_size, hidden_size),
        "[EMO]": torch.randn(batch_size, hidden_size),
        "[MEM]": torch.randn(batch_size, hidden_size),
        "[REL]": torch.randn(batch_size, hidden_size),
        "[TASK]": torch.randn(batch_size, hidden_size),
    }

    # Forward pass
    logits = wrapped(hidden_states, pooled_outputs)

    # Should receive hub token representation, not full sequence
    assert logits.shape == (batch_size, 44)


def test_capability_head_token_routing():
    """Test CapabilityHead wrapper correctly routes token-level capabilities."""
    from modeling_studio.models.routing_v3 import CapabilityHead

    # Create a simple mock token-level head
    class MockTokenHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(768, 9)

        def forward(self, x):
            return self.linear(x)

    head = MockTokenHead()
    wrapped = CapabilityHead("ner_general", head, hidden_size=768)

    # Check routing properties
    assert wrapped.capability == "ner_general"
    assert wrapped.pool_type == "token"
    assert wrapped.hub_token is None

    # Create dummy data
    batch_size = 2
    seq_len = 128
    hidden_size = 768
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    pooled_outputs = {
        "[CLS]": torch.randn(batch_size, hidden_size),
        "[EMO]": torch.randn(batch_size, hidden_size),
    }

    # Forward pass
    logits = wrapped(hidden_states, pooled_outputs)

    # Should receive full sequence, not hub token
    assert logits.shape == (batch_size, seq_len, 9)


def test_create_hub_routing_info():
    """Test create_hub_routing_info returns correct routing information."""
    from modeling_studio.models.routing_v3 import create_hub_routing_info

    # Test hub capability
    info = create_hub_routing_info("emotions")
    assert info["capability"] == "emotions"
    assert info["pool_type"] == "hub"
    assert info["hub_token"] == "[EMO]"
    assert "hub_description" in info
    assert "Affective understanding" in info["hub_description"]

    # Test token-level capability
    info = create_hub_routing_info("ner_general")
    assert info["capability"] == "ner_general"
    assert info["pool_type"] == "token"
    assert info["hub_token"] is None
    assert "hub_description" not in info  # No hub for token-level

    # Test all 12 capabilities
    all_capabilities = [
        "emotions",
        "sentiment",
        "safety_generic",
        "safety_familyos",
        "embedding",
        "nli",
        "relation",
        "intent",
        "ingress",
        "ner_general",
        "ner_family",
        "temporal",
    ]

    for cap in all_capabilities:
        info = create_hub_routing_info(cap)
        assert info["capability"] == cap
        assert info["pool_type"] in ["hub", "token"]


def test_hub_router_all_capabilities_mapped():
    """Test that all 12 FamilyOS capabilities are correctly mapped."""
    from modeling_studio.models.routing_v3 import HubRouter

    # All 12 capabilities
    all_capabilities = [
        "emotions",
        "sentiment",
        "safety_generic",
        "safety_familyos",
        "embedding",
        "nli",
        "relation",
        "intent",
        "ingress",
        "ner_general",
        "ner_family",
        "temporal",
    ]

    # Check all are in routing table
    for cap in all_capabilities:
        assert cap in HubRouter.ROUTING_TABLE

    # Check counts by hub
    emo_caps = [cap for cap, (pool_type, hub) in HubRouter.ROUTING_TABLE.items() if hub == "[EMO]"]
    mem_caps = [cap for cap, (pool_type, hub) in HubRouter.ROUTING_TABLE.items() if hub == "[MEM]"]
    rel_caps = [cap for cap, (pool_type, hub) in HubRouter.ROUTING_TABLE.items() if hub == "[REL]"]
    task_caps = [
        cap for cap, (pool_type, hub) in HubRouter.ROUTING_TABLE.items() if hub == "[TASK]"
    ]
    token_caps = [
        cap for cap, (pool_type, hub) in HubRouter.ROUTING_TABLE.items() if pool_type == "token"
    ]

    # Verify counts match design
    assert len(emo_caps) == 4  # emotions, sentiment, safety_generic, safety_familyos
    assert len(mem_caps) == 1  # embedding
    assert len(rel_caps) == 2  # nli, relation
    assert len(task_caps) == 2  # intent, ingress
    assert len(token_caps) == 3  # ner_general, ner_family, temporal

    # Total should be 12
    assert len(emo_caps) + len(mem_caps) + len(rel_caps) + len(task_caps) + len(token_caps) == 12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
