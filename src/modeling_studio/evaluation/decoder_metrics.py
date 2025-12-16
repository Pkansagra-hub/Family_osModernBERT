"""
Decoder Evaluation Metrics for Counterfactual Generation.

This module provides metrics for evaluating the counterfactual decoder head:
    - Perplexity: Language modeling quality
    - BLEU: N-gram overlap with references
    - ROUGE: Recall-oriented summary evaluation (optional)
    - Distinct-N: Generation diversity

Key Metrics:
    - Perplexity: Lower is better (good models: 5-20)
    - BLEU: Higher is better (0-100 scale)
    - ROUGE-L: Higher is better (0-1 scale)

Usage:
    from modeling_studio.evaluation.decoder_metrics import (
        compute_perplexity,
        compute_bleu,
        compute_rouge,
        compute_distinct_n,
        DecoderEvaluator,
    )

    # Perplexity on validation set
    ppl = compute_perplexity(model, val_dataloader, device='cuda')

    # BLEU score for generated text
    bleu = compute_bleu(predictions, references)

    # Full evaluation
    evaluator = DecoderEvaluator(model, tokenizer)
    results = evaluator.evaluate(val_dataset, references)
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

    from modeling_studio.models.decoder_moe import CounterfactualDecoderHead

logger = logging.getLogger(__name__)


# =============================================================================
# Perplexity (Issue 15.1.1)
# =============================================================================


def compute_perplexity(
    model: "CounterfactualDecoderHead",
    dataloader: DataLoader,
    device: str = "cuda",
) -> float:
    """
    Compute perplexity on validation set.

    Perplexity measures how well the model predicts the next token.
    Lower values indicate better language modeling performance.

    Formula:
        PPL = exp(average cross-entropy loss)

    Args:
        model: Counterfactual decoder head model.
        dataloader: DataLoader yielding batches with:
            - encoder_hidden_states or hidden_states
            - attention_mask or encoder_attention_mask
            - labels: Target token IDs (-100 for padding)
        device: Device to run evaluation on.

    Returns:
        Perplexity as float. Typical range: 1.0 to 100+
        Lower is better. Good models: 5-20.

    Example:
        >>> model = CounterfactualDecoderHead(config)
        >>> val_loader = DataLoader(val_dataset, batch_size=8)
        >>> ppl = compute_perplexity(model, val_loader, device='cuda')
        >>> print(f"Perplexity: {ppl:.2f}")
    """
    model.eval()
    model.to(device)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing perplexity"):
            # Move batch to device
            batch = _move_batch_to_device(batch, device)

            # Forward pass
            outputs = model(**batch)

            # Get loss (already computed with ignore_index=-100)
            if "loss" not in outputs:
                logger.warning("Model output missing 'loss' key, skipping batch")
                continue

            # Count valid tokens (not -100)
            labels = batch.get("labels")
            if labels is None:
                logger.warning("Batch missing 'labels', skipping")
                continue

            # Number of valid tokens in this batch
            valid_tokens = labels.ne(-100).sum().item()

            if valid_tokens == 0:
                continue

            # Accumulate
            # Note: outputs['loss'] includes aux_loss, we need to extract CE loss
            # For perplexity, we want only the LM loss
            loss_value = outputs["loss"].item()

            # Subtract auxiliary loss if present
            if "aux_loss" in outputs:
                aux_loss = outputs["aux_loss"]
                if isinstance(aux_loss, torch.Tensor):
                    aux_loss = aux_loss.item()
                loss_value = loss_value - aux_loss

            total_loss += loss_value * valid_tokens
            total_tokens += valid_tokens

    if total_tokens == 0:
        logger.warning("No valid tokens found, returning inf perplexity")
        return float("inf")

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    logger.info(f"Perplexity: {perplexity:.4f} (avg loss: {avg_loss:.4f}, tokens: {total_tokens:,})")

    return perplexity


def _move_batch_to_device(batch: dict, device: str) -> dict:
    """Move all tensors in batch to specified device."""
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


# =============================================================================
# BLEU Score (Issue 15.1.2)
# =============================================================================


def compute_bleu(
    predictions: list[str],
    references: list[str],
    lowercase: bool = True,
) -> float:
    """
    Compute corpus BLEU score using sacrebleu.

    BLEU measures n-gram overlap between predictions and references.
    Uses sacrebleu for consistent, reproducible scores.

    Args:
        predictions: List of generated texts.
        references: List of reference texts (same length as predictions).
        lowercase: Whether to lowercase texts before comparison.

    Returns:
        BLEU score as float (0-100 scale). Higher is better.
        100 = perfect match, 0 = no overlap.

    Example:
        >>> predictions = ["The cat sat on mat"]
        >>> references = ["The cat sat on the mat"]
        >>> bleu = compute_bleu(predictions, references)
        >>> print(f"BLEU: {bleu:.2f}")
    """
    # Handle empty inputs gracefully
    if not predictions or not references:
        logger.warning("Empty predictions or references, returning 0.0 BLEU")
        return 0.0

    if len(predictions) != len(references):
        raise ValueError(
            f"Predictions ({len(predictions)}) and references ({len(references)}) "
            "must have same length"
        )

    # Filter out empty strings
    valid_pairs = [
        (pred, ref) for pred, ref in zip(predictions, references)
        if pred.strip() and ref.strip()
    ]

    if not valid_pairs:
        logger.warning("No valid prediction-reference pairs, returning 0.0 BLEU")
        return 0.0

    predictions_filtered = [p for p, _ in valid_pairs]
    references_filtered = [r for _, r in valid_pairs]

    try:
        from sacrebleu import corpus_bleu

        # sacrebleu expects references as list of lists (multiple refs per hypothesis)
        bleu_result = corpus_bleu(
            predictions_filtered,
            [references_filtered],
            lowercase=lowercase,
        )
        return bleu_result.score

    except ImportError:
        logger.warning("sacrebleu not installed, falling back to simple BLEU")
        return _compute_simple_bleu(predictions_filtered, references_filtered)


def _compute_simple_bleu(
    predictions: list[str],
    references: list[str],
    max_n: int = 4,
) -> float:
    """
    Simple BLEU implementation as fallback when sacrebleu unavailable.

    This is a basic implementation for testing. Use sacrebleu for production.
    """
    from collections import Counter

    def get_ngrams(text: str, n: int) -> Counter:
        tokens = text.lower().split()
        if len(tokens) < n:
            return Counter()
        return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

    total_matches = [0] * max_n
    total_counts = [0] * max_n

    for pred, ref in zip(predictions, references):
        for n in range(1, max_n + 1):
            pred_ngrams = get_ngrams(pred, n)
            ref_ngrams = get_ngrams(ref, n)

            # Clipped count
            for ngram, count in pred_ngrams.items():
                total_matches[n - 1] += min(count, ref_ngrams.get(ngram, 0))
            total_counts[n - 1] += sum(pred_ngrams.values())

    # Compute precision for each n
    precisions = []
    for n in range(max_n):
        if total_counts[n] > 0:
            precisions.append(total_matches[n] / total_counts[n])
        # Only include n-grams that have counts (skip if no data for this n)

    # Geometric mean with smoothing
    # Only compute over n-grams where we have data
    if precisions and all(p > 0 for p in precisions):
        log_prec = sum(math.log(p) for p in precisions) / len(precisions)
        bleu = math.exp(log_prec) * 100
    elif precisions:
        # If some precisions are 0, use smoothed version
        smoothed = [max(p, 1e-10) for p in precisions]
        log_prec = sum(math.log(p) for p in smoothed) / len(smoothed)
        bleu = math.exp(log_prec) * 100
    else:
        bleu = 0.0

    return bleu


# =============================================================================
# ROUGE Score (Optional Extension)
# =============================================================================


def compute_rouge(
    predictions: list[str],
    references: list[str],
    rouge_types: list[str] | None = None,
) -> dict[str, float]:
    """
    Compute ROUGE scores for generated text.

    ROUGE measures recall-oriented overlap between predictions and references.
    Useful for summarization and generation tasks.

    Args:
        predictions: List of generated texts.
        references: List of reference texts.
        rouge_types: ROUGE variants to compute. Default: ["rouge1", "rouge2", "rougeL"]

    Returns:
        Dictionary mapping ROUGE type to F1 score (0-1 scale).

    Example:
        >>> rouge = compute_rouge(predictions, references)
        >>> print(f"ROUGE-L: {rouge['rougeL']:.4f}")
    """
    if rouge_types is None:
        rouge_types = ["rouge1", "rouge2", "rougeL"]

    if not predictions or not references:
        return {rt: 0.0 for rt in rouge_types}

    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)

        scores = {rt: [] for rt in rouge_types}
        for pred, ref in zip(predictions, references):
            result = scorer.score(ref, pred)
            for rt in rouge_types:
                scores[rt].append(result[rt].fmeasure)

        return {rt: sum(s) / len(s) for rt, s in scores.items()}

    except ImportError:
        logger.warning("rouge_score not installed, returning zeros")
        return {rt: 0.0 for rt in rouge_types}


# =============================================================================
# Generation Diversity (Distinct-N)
# =============================================================================


def compute_distinct_n(
    texts: list[str],
    n_values: list[int] | None = None,
) -> dict[str, float]:
    """
    Compute Distinct-N scores measuring generation diversity.

    Distinct-N is the ratio of unique n-grams to total n-grams.
    Higher values indicate more diverse, less repetitive generation.

    Args:
        texts: List of generated texts.
        n_values: N-gram sizes to compute. Default: [1, 2, 3]

    Returns:
        Dictionary mapping "distinct-N" to score (0-1 scale).

    Example:
        >>> distinct = compute_distinct_n(generated_texts)
        >>> print(f"Distinct-2: {distinct['distinct-2']:.4f}")
    """
    if n_values is None:
        n_values = [1, 2, 3]

    if not texts:
        return {f"distinct-{n}": 0.0 for n in n_values}

    results = {}

    for n in n_values:
        all_ngrams = []
        for text in texts:
            tokens = text.lower().split()
            ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
            all_ngrams.extend(ngrams)

        if all_ngrams:
            unique_count = len(set(all_ngrams))
            total_count = len(all_ngrams)
            results[f"distinct-{n}"] = unique_count / total_count
        else:
            results[f"distinct-{n}"] = 0.0

    return results


# =============================================================================
# MoE Expert Utilization Metrics
# =============================================================================


def compute_expert_utilization(
    model: "CounterfactualDecoderHead",
    dataloader: DataLoader,
    device: str = "cuda",
) -> dict[str, Any]:
    """
    Compute MoE expert utilization statistics.

    Measures how evenly tokens are distributed across experts.
    Good load balancing should show ~12.5% per expert (for 8 experts).

    Args:
        model: Counterfactual decoder head with MoE layers.
        dataloader: DataLoader yielding evaluation batches.
        device: Device to run evaluation on.

    Returns:
        Dictionary containing:
            - expert_counts: Dict mapping expert_idx to token count
            - expert_fractions: Dict mapping expert_idx to fraction
            - balance_score: 1.0 = perfect balance, lower = imbalanced
            - collapsed_experts: Count of experts receiving 0 tokens
            - shared_expert_fraction: Fraction of tokens through shared expert

    Example:
        >>> stats = compute_expert_utilization(model, val_loader)
        >>> print(f"Balance: {stats['balance_score']:.4f}")
        >>> print(f"Collapsed: {stats['collapsed_experts']}")
    """
    model.eval()
    model.to(device)

    # Find MoE layers
    moe_layers = []
    for layer in model.layers:
        if hasattr(layer, "ffn") and hasattr(layer.ffn, "router"):
            moe_layers.append(layer.ffn)

    if not moe_layers:
        return {
            "expert_counts": {},
            "expert_fractions": {},
            "balance_score": 1.0,
            "collapsed_experts": 0,
            "shared_expert_fraction": 0.0,
            "error": "No MoE layers found",
        }

    # Hook to capture routing decisions
    routing_info = {
        "expert_counts": Counter(),
        "total_tokens": 0,
        "shared_tokens": 0,
    }

    def router_hook(module, input_tensors, output):
        routing_weights, expert_indices, _ = output
        # Count tokens per expert
        for idx in expert_indices.flatten().tolist():
            routing_info["expert_counts"][idx] += 1
        routing_info["total_tokens"] += expert_indices.numel()

    # Register hooks
    hooks = []
    for moe_layer in moe_layers:
        hook = moe_layer.router.register_forward_hook(router_hook)
        hooks.append(hook)

    try:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing expert utilization"):
                batch = _move_batch_to_device(batch, device)
                _ = model(**batch)

    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()

    # Compute statistics
    num_experts = moe_layers[0].num_experts
    expert_counts = dict(routing_info["expert_counts"])
    total = routing_info["total_tokens"]

    # Expert fractions
    expert_fractions = {}
    for i in range(num_experts):
        count = expert_counts.get(i, 0)
        expert_fractions[i] = count / total if total > 0 else 0.0

    # Balance score: 1 - std deviation normalized by mean
    if total > 0:
        fractions = list(expert_fractions.values())
        mean_frac = sum(fractions) / len(fractions)
        if mean_frac > 0:
            variance = sum((f - mean_frac) ** 2 for f in fractions) / len(fractions)
            std_dev = math.sqrt(variance)
            # Coefficient of variation (lower = more balanced)
            cv = std_dev / mean_frac
            # Convert to balance score (1 = perfect, 0 = very imbalanced)
            balance_score = max(0.0, 1.0 - cv)
        else:
            balance_score = 0.0
    else:
        balance_score = 0.0

    # Count collapsed experts
    collapsed = sum(1 for i in range(num_experts) if expert_counts.get(i, 0) == 0)

    # Shared expert always processes all tokens if enabled
    has_shared = any(moe.shared_expert is not None for moe in moe_layers)
    shared_fraction = 1.0 if has_shared else 0.0

    return {
        "expert_counts": expert_counts,
        "expert_fractions": expert_fractions,
        "balance_score": balance_score,
        "collapsed_experts": collapsed,
        "shared_expert_fraction": shared_fraction,
        "num_experts": num_experts,
        "total_tokens": total,
    }


# =============================================================================
# Decoder Evaluator (Full Pipeline)
# =============================================================================


@dataclass
class DecoderEvaluationResults:
    """Results from decoder evaluation."""

    perplexity: float
    bleu: float
    rouge: dict[str, float] = field(default_factory=dict)
    distinct_n: dict[str, float] = field(default_factory=dict)
    expert_utilization: dict[str, Any] = field(default_factory=dict)
    num_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "perplexity": self.perplexity,
            "bleu": self.bleu,
            **self.rouge,
            **self.distinct_n,
            "expert_balance": self.expert_utilization.get("balance_score", 0.0),
            "collapsed_experts": self.expert_utilization.get("collapsed_experts", 0),
            "num_samples": self.num_samples,
        }

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=== Decoder Evaluation Results ===",
            f"Perplexity: {self.perplexity:.4f}",
            f"BLEU: {self.bleu:.2f}",
        ]

        if self.rouge:
            for name, score in self.rouge.items():
                lines.append(f"{name}: {score:.4f}")

        if self.distinct_n:
            for name, score in self.distinct_n.items():
                lines.append(f"{name}: {score:.4f}")

        if self.expert_utilization:
            lines.append(f"Expert Balance: {self.expert_utilization.get('balance_score', 0):.4f}")
            lines.append(f"Collapsed Experts: {self.expert_utilization.get('collapsed_experts', 0)}")

        lines.append(f"Samples: {self.num_samples}")
        return "\n".join(lines)


class DecoderEvaluator:
    """
    Full evaluation pipeline for counterfactual decoder.

    Combines perplexity, BLEU, ROUGE, diversity, and expert utilization
    into a single evaluation run.

    Args:
        model: Counterfactual decoder head.
        tokenizer: Tokenizer for decoding generated tokens.
        device: Device to run evaluation on.

    Example:
        >>> evaluator = DecoderEvaluator(model, tokenizer)
        >>> results = evaluator.evaluate(
        ...     dataloader=val_loader,
        ...     references=ref_texts,
        ...     max_new_tokens=64,
        ... )
        >>> print(results.summary())
    """

    def __init__(
        self,
        model: "CounterfactualDecoderHead",
        tokenizer: "PreTrainedTokenizer | PreTrainedTokenizerFast",
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def evaluate(
        self,
        dataloader: DataLoader,
        references: list[str] | None = None,
        compute_generation_metrics: bool = True,
        compute_expert_metrics: bool = True,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
    ) -> DecoderEvaluationResults:
        """
        Run full evaluation pipeline.

        Args:
            dataloader: Validation dataloader.
            references: Reference texts for BLEU/ROUGE (optional).
            compute_generation_metrics: Whether to generate and compute BLEU/ROUGE.
            compute_expert_metrics: Whether to compute expert utilization.
            max_new_tokens: Maximum tokens to generate per sample.
            temperature: Sampling temperature for generation.

        Returns:
            DecoderEvaluationResults with all metrics.
        """
        self.model.to(self.device)
        self.model.eval()

        # 1. Perplexity
        logger.info("Computing perplexity...")
        perplexity = compute_perplexity(self.model, dataloader, self.device)

        # 2. Generation metrics (BLEU, ROUGE, Distinct-N)
        bleu = 0.0
        rouge = {}
        distinct = {}
        predictions = []

        if compute_generation_metrics and references is not None:
            logger.info("Generating predictions...")
            predictions = self._generate_predictions(
                dataloader, max_new_tokens, temperature
            )

            if predictions:
                logger.info("Computing BLEU...")
                bleu = compute_bleu(predictions, references[:len(predictions)])

                logger.info("Computing ROUGE...")
                rouge = compute_rouge(predictions, references[:len(predictions)])

                logger.info("Computing Distinct-N...")
                distinct = compute_distinct_n(predictions)

        # 3. Expert utilization
        expert_stats = {}
        if compute_expert_metrics:
            logger.info("Computing expert utilization...")
            expert_stats = compute_expert_utilization(
                self.model, dataloader, self.device
            )

        return DecoderEvaluationResults(
            perplexity=perplexity,
            bleu=bleu,
            rouge=rouge,
            distinct_n=distinct,
            expert_utilization=expert_stats,
            num_samples=len(predictions) if predictions else 0,
        )

    def _generate_predictions(
        self,
        dataloader: DataLoader,
        max_new_tokens: int,
        temperature: float,
    ) -> list[str]:
        """Generate predictions for all samples in dataloader."""
        predictions = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating"):
                batch = _move_batch_to_device(batch, self.device)

                # Get encoder outputs
                encoder_hidden = batch.get("hidden_states") or batch.get("encoder_hidden_states")
                encoder_mask = batch.get("attention_mask") or batch.get("encoder_attention_mask")

                if encoder_hidden is None:
                    logger.warning("Missing encoder hidden states, skipping batch")
                    continue

                # Generate
                generated_ids = self.model.generate(
                    encoder_hidden_states=encoder_hidden,
                    encoder_attention_mask=encoder_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )

                # Decode
                for ids in generated_ids:
                    text = self.tokenizer.decode(ids, skip_special_tokens=True)
                    predictions.append(text)

        return predictions


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "compute_perplexity",
    "compute_bleu",
    "compute_rouge",
    "compute_distinct_n",
    "compute_expert_utilization",
    "DecoderEvaluator",
    "DecoderEvaluationResults",
]
