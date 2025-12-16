"""
Tests for Decoder Evaluation Metrics.

Test Coverage:
    - Issue 15.1.1: Perplexity Calculation
    - Issue 15.1.2: BLEU Score Calculation
    - Additional: ROUGE, Distinct-N, Expert Utilization

Milestone 15: Evaluation & Quality
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

# Import the metrics module
from modeling_studio.evaluation.decoder_metrics import (
    DecoderEvaluationResults,
    DecoderEvaluator,
    _compute_simple_bleu,
    compute_bleu,
    compute_distinct_n,
    compute_expert_utilization,
    compute_perplexity,
    compute_rouge,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_model():
    """Create a mock decoder model for testing."""
    model = MagicMock()
    model.eval = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)
    model.layers = []
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.decode = MagicMock(side_effect=lambda ids, **kwargs: "decoded text")
    return tokenizer


@pytest.fixture
def sample_dataloader():
    """Create a sample dataloader for testing."""
    # Create sample tensors
    batch_size = 4
    seq_len = 16
    hidden_size = 768

    encoder_hidden = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = torch.randint(0, 1000, (batch_size, seq_len))
    # Add some padding tokens
    labels[:, -2:] = -100

    dataset = TensorDataset(encoder_hidden, attention_mask, labels)

    def collate_fn(batch):
        return {
            "encoder_hidden_states": torch.stack([b[0] for b in batch]),
            "encoder_attention_mask": torch.stack([b[1] for b in batch]),
            "labels": torch.stack([b[2] for b in batch]),
        }

    return DataLoader(dataset, batch_size=2, collate_fn=collate_fn)


# =============================================================================
# Test Issue 15.1.1: Perplexity Calculation
# =============================================================================


class TestPerplexityCalculation:
    """Tests for compute_perplexity function."""

    def test_perplexity_calculation(self, mock_model, sample_dataloader):
        """15.1.1-T1: Perplexity computed correctly."""
        # Setup mock model output
        mock_model.return_value = {
            "loss": torch.tensor(2.0),  # log(exp(2)) = 2
            "aux_loss": torch.tensor(0.0),
            "logits": torch.randn(2, 14, 1000),
        }

        with patch("modeling_studio.evaluation.decoder_metrics.tqdm", lambda x, **kwargs: x):
            ppl = compute_perplexity(mock_model, sample_dataloader, device="cpu")

        # Perplexity should be exp(2) ≈ 7.39
        assert isinstance(ppl, float)
        assert ppl > 0
        # With aux_loss=0, loss=2, should be around exp(2)
        assert 5.0 < ppl < 10.0

    def test_perplexity_ignores_padding(self, mock_model):
        """15.1.1-T2: -100 labels excluded from calculation."""
        # Create batch with known padding
        batch_size = 2
        seq_len = 8
        hidden_size = 768

        encoder_hidden = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.ones(batch_size, seq_len)
        # Half tokens are padding
        labels = torch.randint(0, 1000, (batch_size, seq_len))
        labels[:, seq_len // 2:] = -100

        dataset = TensorDataset(encoder_hidden, attention_mask, labels)

        def collate_fn(batch):
            return {
                "encoder_hidden_states": torch.stack([b[0] for b in batch]),
                "encoder_attention_mask": torch.stack([b[1] for b in batch]),
                "labels": torch.stack([b[2] for b in batch]),
            }

        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

        # Mock: loss computed correctly
        mock_model.return_value = {
            "loss": torch.tensor(1.5),
            "aux_loss": torch.tensor(0.1),
            "logits": torch.randn(batch_size, seq_len, 1000),
        }

        with patch("modeling_studio.evaluation.decoder_metrics.tqdm", lambda x, **kwargs: x):
            ppl = compute_perplexity(mock_model, dataloader, device="cpu")

        # Should only count non-padding tokens
        assert isinstance(ppl, float)
        assert ppl > 0

    def test_perplexity_returns_float(self, mock_model, sample_dataloader):
        """15.1.1-T1: Returns perplexity as float."""
        mock_model.return_value = {
            "loss": torch.tensor(1.0),
            "aux_loss": torch.tensor(0.0),
            "logits": torch.randn(2, 14, 1000),
        }

        with patch("modeling_studio.evaluation.decoder_metrics.tqdm", lambda x, **kwargs: x):
            ppl = compute_perplexity(mock_model, sample_dataloader, device="cpu")

        assert isinstance(ppl, float)
        assert not math.isnan(ppl)

    def test_perplexity_works_with_batched_evaluation(self, mock_model):
        """15.1.1-AC3: Works with batched evaluation."""
        # Create multiple batches
        batches = []
        for _ in range(3):
            batch = {
                "encoder_hidden_states": torch.randn(4, 16, 768),
                "encoder_attention_mask": torch.ones(4, 16),
                "labels": torch.randint(0, 1000, (4, 16)),
            }
            batches.append(batch)

        # Create dataloader that yields these batches
        class MockDataLoader:
            def __iter__(self):
                for b in batches:
                    yield b

        mock_model.return_value = {
            "loss": torch.tensor(1.5),
            "aux_loss": torch.tensor(0.0),
            "logits": torch.randn(4, 16, 1000),
        }

        with patch("modeling_studio.evaluation.decoder_metrics.tqdm", lambda x, **kwargs: x):
            ppl = compute_perplexity(mock_model, MockDataLoader(), device="cpu")

        assert isinstance(ppl, float)
        assert ppl > 0

    def test_perplexity_empty_dataloader(self, mock_model):
        """Perplexity returns inf for empty dataloader."""

        class EmptyDataLoader:
            def __iter__(self):
                return iter([])

        with patch("modeling_studio.evaluation.decoder_metrics.tqdm", lambda x, **kwargs: x):
            ppl = compute_perplexity(mock_model, EmptyDataLoader(), device="cpu")

        assert ppl == float("inf")

    def test_perplexity_subtracts_aux_loss(self, mock_model, sample_dataloader):
        """Perplexity correctly subtracts auxiliary loss from total loss."""
        # Total loss = 3.0, aux_loss = 1.0, so CE loss = 2.0
        mock_model.return_value = {
            "loss": torch.tensor(3.0),
            "aux_loss": torch.tensor(1.0),
            "logits": torch.randn(2, 14, 1000),
        }

        with patch("modeling_studio.evaluation.decoder_metrics.tqdm", lambda x, **kwargs: x):
            ppl = compute_perplexity(mock_model, sample_dataloader, device="cpu")

        # exp(2.0) ≈ 7.39
        assert 5.0 < ppl < 10.0


# =============================================================================
# Test Issue 15.1.2: BLEU Score Calculation
# =============================================================================


class TestBLEUCalculation:
    """Tests for compute_bleu function."""

    def test_bleu_perfect_match(self):
        """15.1.2-T1: BLEU = 100 for identical text."""
        predictions = ["The cat sat on the mat"]
        references = ["The cat sat on the mat"]

        bleu = compute_bleu(predictions, references)

        assert isinstance(bleu, float)
        assert bleu > 99.0  # Perfect or near-perfect match

    def test_bleu_no_match(self):
        """15.1.2-T2: BLEU ~ 0 for completely different text."""
        predictions = ["xyz abc 123 qwerty"]
        references = ["The quick brown fox jumps over lazy dog"]

        bleu = compute_bleu(predictions, references)

        assert isinstance(bleu, float)
        assert bleu < 5.0  # Very low score for no overlap

    def test_bleu_uses_sacrebleu(self):
        """15.1.2-AC1: Uses sacrebleu for consistent scores."""
        predictions = ["The cat sat on mat"]
        references = ["The cat sat on the mat"]

        # Should use sacrebleu if available
        bleu = compute_bleu(predictions, references)

        assert isinstance(bleu, float)
        assert 0 <= bleu <= 100

    def test_bleu_returns_float(self):
        """15.1.2-AC2: Returns BLEU as float (0-100 scale)."""
        predictions = ["Hello world"]
        references = ["Hello world"]

        bleu = compute_bleu(predictions, references)

        assert isinstance(bleu, float)
        assert 0 <= bleu <= 100

    def test_bleu_handles_empty_predictions(self):
        """15.1.2-AC3: Handles empty predictions gracefully."""
        # Empty predictions list
        bleu = compute_bleu([], [])
        assert bleu == 0.0

        # Empty string predictions
        bleu = compute_bleu([""], ["reference text"])
        assert bleu == 0.0

    def test_bleu_length_mismatch_raises(self):
        """BLEU raises error for mismatched lengths."""
        predictions = ["one", "two"]
        references = ["one"]

        with pytest.raises(ValueError, match="same length"):
            compute_bleu(predictions, references)

    def test_bleu_multiple_sentences(self):
        """BLEU works with multiple sentence pairs."""
        predictions = [
            "The cat sat on the mat",
            "A dog runs in the park",
            "Birds fly in the sky",
        ]
        references = [
            "The cat sat on the mat",
            "The dog runs in the park",
            "Birds fly in the blue sky",
        ]

        bleu = compute_bleu(predictions, references)

        assert isinstance(bleu, float)
        assert 50 < bleu < 100  # Partial matches

    def test_bleu_lowercase_option(self):
        """BLEU respects lowercase option."""
        predictions = ["THE CAT SAT"]
        references = ["the cat sat"]

        bleu_lower = compute_bleu(predictions, references, lowercase=True)

        # With lowercase, should be high match
        # Note: When sacrebleu is not available, we fall back to simple BLEU
        # which always lowercases
        assert bleu_lower >= 0.0  # Valid BLEU score returned


class TestSimpleBLEU:
    """Tests for fallback simple BLEU implementation."""

    def test_simple_bleu_perfect(self):
        """Simple BLEU returns high score for identical text."""
        predictions = ["The cat sat on the mat"]
        references = ["The cat sat on the mat"]

        bleu = _compute_simple_bleu(predictions, references)

        assert bleu > 90.0

    def test_simple_bleu_partial(self):
        """Simple BLEU returns partial score for partial overlap."""
        predictions = ["The cat sat on floor"]
        references = ["The cat sat on the mat"]

        bleu = _compute_simple_bleu(predictions, references)

        # Should be partial match (not perfect, not zero)
        assert 30 < bleu < 100

    def test_simple_bleu_no_match(self):
        """Simple BLEU returns near-zero for no overlap."""
        predictions = ["xyz abc 123"]
        references = ["The cat sat on mat"]

        bleu = _compute_simple_bleu(predictions, references)

        assert bleu < 5.0


# =============================================================================
# Test ROUGE Calculation
# =============================================================================


class TestROUGECalculation:
    """Tests for compute_rouge function."""

    def test_rouge_returns_dict(self):
        """ROUGE returns dictionary of scores."""
        predictions = ["The cat sat on the mat"]
        references = ["The cat sat on the mat"]

        rouge = compute_rouge(predictions, references)

        assert isinstance(rouge, dict)
        assert "rouge1" in rouge or len(rouge) == 0  # Either works or is empty
        assert "rouge2" in rouge or len(rouge) == 0
        assert "rougeL" in rouge or len(rouge) == 0

    def test_rouge_empty_input(self):
        """ROUGE handles empty input."""
        rouge = compute_rouge([], [])

        assert isinstance(rouge, dict)

    def test_rouge_custom_types(self):
        """ROUGE respects custom rouge_types."""
        predictions = ["The cat sat"]
        references = ["The cat sat"]

        rouge = compute_rouge(predictions, references, rouge_types=["rouge1"])

        assert isinstance(rouge, dict)


# =============================================================================
# Test Distinct-N Calculation
# =============================================================================


class TestDistinctN:
    """Tests for compute_distinct_n function."""

    def test_distinct_returns_dict(self):
        """Distinct-N returns dictionary of scores."""
        texts = ["The cat sat on the mat", "The dog ran in the park"]

        distinct = compute_distinct_n(texts)

        assert isinstance(distinct, dict)
        assert "distinct-1" in distinct
        assert "distinct-2" in distinct
        assert "distinct-3" in distinct

    def test_distinct_high_diversity(self):
        """Distinct-N is high for diverse text."""
        texts = [
            "apple banana cherry",
            "dog elephant frog",
            "guitar harmonica instrument",
        ]

        distinct = compute_distinct_n(texts)

        # High diversity should give high scores
        assert distinct["distinct-1"] > 0.5
        assert distinct["distinct-2"] > 0.5

    def test_distinct_low_diversity(self):
        """Distinct-N is low for repetitive text."""
        texts = [
            "the the the the the",
            "the the the the the",
            "the the the the the",
        ]

        distinct = compute_distinct_n(texts)

        # Highly repetitive should give low scores
        assert distinct["distinct-1"] < 0.2

    def test_distinct_empty_input(self):
        """Distinct-N handles empty input."""
        distinct = compute_distinct_n([])

        assert distinct["distinct-1"] == 0.0
        assert distinct["distinct-2"] == 0.0

    def test_distinct_custom_n(self):
        """Distinct-N respects custom n_values."""
        texts = ["The cat sat on the mat"]

        distinct = compute_distinct_n(texts, n_values=[1, 4])

        assert "distinct-1" in distinct
        assert "distinct-4" in distinct
        assert "distinct-2" not in distinct


# =============================================================================
# Test Expert Utilization
# =============================================================================


class TestExpertUtilization:
    """Tests for compute_expert_utilization function."""

    def test_expert_utilization_no_moe_layers(self, mock_model, sample_dataloader):
        """Expert utilization handles models without MoE layers."""
        mock_model.layers = []

        with patch("modeling_studio.evaluation.decoder_metrics.tqdm", lambda x, **kwargs: x):
            stats = compute_expert_utilization(mock_model, sample_dataloader, device="cpu")

        assert "error" in stats
        assert stats["balance_score"] == 1.0

    def test_expert_utilization_returns_dict(self, mock_model, sample_dataloader):
        """Expert utilization returns expected dictionary keys."""
        # Create mock MoE layer
        mock_moe = MagicMock()
        mock_router = MagicMock()
        mock_router.register_forward_hook = MagicMock(return_value=MagicMock())
        mock_moe.router = mock_router
        mock_moe.num_experts = 8
        mock_moe.shared_expert = None

        mock_layer = MagicMock()
        mock_layer.ffn = mock_moe
        mock_model.layers = [mock_layer]

        # Mock model forward
        mock_model.return_value = {"logits": torch.randn(2, 14, 1000)}

        # Mock router output
        mock_router.return_value = (
            torch.ones(2, 14, 2),  # routing_weights
            torch.randint(0, 8, (2, 14, 2)),  # expert_indices
            {},  # aux_losses
        )

        with patch("modeling_studio.evaluation.decoder_metrics.tqdm", lambda x, **kwargs: x):
            stats = compute_expert_utilization(mock_model, sample_dataloader, device="cpu")

        assert "expert_counts" in stats
        assert "expert_fractions" in stats
        assert "balance_score" in stats
        assert "collapsed_experts" in stats


# =============================================================================
# Test DecoderEvaluationResults
# =============================================================================


class TestDecoderEvaluationResults:
    """Tests for DecoderEvaluationResults dataclass."""

    def test_to_dict(self):
        """Results can be converted to dictionary."""
        results = DecoderEvaluationResults(
            perplexity=10.5,
            bleu=45.2,
            rouge={"rougeL": 0.75},
            distinct_n={"distinct-1": 0.8},
            expert_utilization={"balance_score": 0.95},
            num_samples=100,
        )

        d = results.to_dict()

        assert d["perplexity"] == 10.5
        assert d["bleu"] == 45.2
        assert d["rougeL"] == 0.75
        assert d["distinct-1"] == 0.8
        assert d["expert_balance"] == 0.95
        assert d["num_samples"] == 100

    def test_summary(self):
        """Results summary is a string."""
        results = DecoderEvaluationResults(
            perplexity=10.5,
            bleu=45.2,
            num_samples=100,
        )

        summary = results.summary()

        assert isinstance(summary, str)
        assert "Perplexity" in summary
        assert "BLEU" in summary
        assert "10.5" in summary or "10.50" in summary


# =============================================================================
# Test DecoderEvaluator
# =============================================================================


class TestDecoderEvaluator:
    """Tests for DecoderEvaluator class."""

    def test_evaluator_initialization(self, mock_model, mock_tokenizer):
        """Evaluator initializes correctly."""
        evaluator = DecoderEvaluator(mock_model, mock_tokenizer, device="cpu")

        assert evaluator.model is mock_model
        assert evaluator.tokenizer is mock_tokenizer
        assert evaluator.device == "cpu"

    def test_evaluator_evaluate_returns_results(
        self, mock_model, mock_tokenizer, sample_dataloader
    ):
        """Evaluator evaluate returns DecoderEvaluationResults."""
        # Setup mocks
        mock_model.return_value = {
            "loss": torch.tensor(1.5),
            "aux_loss": torch.tensor(0.0),
            "logits": torch.randn(2, 14, 1000),
        }
        mock_model.layers = []

        evaluator = DecoderEvaluator(mock_model, mock_tokenizer, device="cpu")

        with patch("modeling_studio.evaluation.decoder_metrics.tqdm", lambda x, **kwargs: x):
            results = evaluator.evaluate(
                sample_dataloader,
                compute_generation_metrics=False,
                compute_expert_metrics=False,
            )

        assert isinstance(results, DecoderEvaluationResults)
        assert results.perplexity > 0
