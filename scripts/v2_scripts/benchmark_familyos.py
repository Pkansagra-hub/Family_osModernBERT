#!/usr/bin/env python3
"""
FamilyOS Benchmarking Script

Comprehensive benchmarking following the benchmarking_plan.md specifications.
Evaluates model performance using FamilyOS-appropriate metrics rather than
strict academic benchmarks.

Usage:
    # Benchmark trained checkpoint
    python scripts/benchmark_familyos.py --checkpoint checkpoints/modernbert-multitask-v0-stage-a-fast/checkpoint-best
    python scripts/benchmark_familyos.py --checkpoint outputs/stage_a --tasks sentiment emotions
    python scripts/benchmark_familyos.py --checkpoint outputs/stage_a --all-tasks

    # Compute random baselines for comparison
    python scripts/benchmark_familyos.py --baseline-only

    # Benchmark with HuggingFace base models (zero-shot/pre-trained)
    python scripts/benchmark_familyos.py --base-model --tasks sentiment nli

Features:
    - Sentiment: Strict 5-class, Grouped 3-class, Binary direction
    - Emotions: Strict Micro-F1, Top-K Recall, Primary emotion accuracy, Jaccard
    - Safety: Per-type recall, Cultural robustness, Keyword override tests
    - NER: Token-level F1, Entity-level F1
    - Temporal: Per-type F1
    - NLI: Binary conflict detection
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
# benchmark_familyos.py is at scripts/v2_scripts/, so need .parent.parent.parent for project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from datasets import Dataset, load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from transformers import AutoTokenizer

from modeling_studio.data.labels import (
    EMOTIONS_FAMILYOS_LABELS,
    NER_GENERAL_LABELS,
    NLI_LABELS,
    SAFETY_FAMILYOS_LABELS,
    SAFETY_GENERIC_LABELS,
    SENTIMENT_LABELS,
    TEMPORAL_LABELS,
)
from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel
from modeling_studio.trainers.collators import MultiTaskCollator


# =============================================================================
# Data Classes for Results
# =============================================================================


@dataclass
class SentimentResults:
    """Sentiment benchmarking results."""

    strict_5class_accuracy: float = 0.0
    grouped_3class_accuracy: float = 0.0
    binary_direction_accuracy: float = 0.0
    adjacent_tolerance_accuracy: float = 0.0
    per_class_f1: dict[str, float] = field(default_factory=dict)
    confusion_matrix_5x5: list[list[int]] = field(default_factory=list)
    confusion_matrix_3x3: list[list[int]] = field(default_factory=list)
    negative_recall: float = 0.0
    positive_precision: float = 0.0
    neutral_f1: float = 0.0


@dataclass
class EmotionsResults:
    """Emotions benchmarking results."""

    strict_micro_f1: float = 0.0
    strict_macro_f1: float = 0.0
    top_1_recall: float = 0.0
    top_2_recall: float = 0.0
    top_3_recall: float = 0.0
    primary_emotion_accuracy: float = 0.0
    partial_match_jaccard: float = 0.0
    at_least_one_correct: float = 0.0
    per_family_recall: dict[str, float] = field(default_factory=dict)
    avg_predictions_per_sample: float = 0.0
    avg_ground_truth_per_sample: float = 0.0


@dataclass
class SafetyResults:
    """Safety benchmarking results."""

    # Generic (8-type multi-label)
    any_toxic_recall: float = 0.0
    per_type_recall: dict[str, float] = field(default_factory=dict)
    micro_f1: float = 0.0
    macro_f1: float = 0.0
    precision_at_95_recall: float = 0.0
    # FamilyOS (4-band)
    crisis_recall: float = 0.0
    red_recall: float = 0.0
    amber_recall: float = 0.0
    green_precision: float = 0.0
    cultural_fp_rate: float = 0.0
    keyword_override_accuracy: float = 0.0
    band_confusion_matrix: list[list[int]] = field(default_factory=list)


@dataclass
class NERResults:
    """NER benchmarking results."""

    token_level_f1: float = 0.0
    entity_level_f1: float = 0.0
    per_entity_f1: dict[str, float] = field(default_factory=dict)
    partial_span_match: float = 0.0


@dataclass
class TemporalResults:
    """Temporal benchmarking results."""

    overall_f1: float = 0.0
    per_type_f1: dict[str, float] = field(default_factory=dict)


@dataclass
class NLIResults:
    """NLI benchmarking results."""

    three_class_accuracy: float = 0.0
    binary_conflict_accuracy: float = 0.0
    contradiction_recall: float = 0.0
    entailment_precision: float = 0.0
    confusion_matrix: list[list[int]] = field(default_factory=list)


@dataclass
class BenchmarkResults:
    """Complete benchmark results."""

    checkpoint: str = ""
    timestamp: str = ""
    sentiment: SentimentResults | None = None
    emotions: EmotionsResults | None = None
    safety_generic: SafetyResults | None = None
    safety_familyos: SafetyResults | None = None
    ner: NERResults | None = None
    temporal: TemporalResults | None = None
    nli: NLIResults | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {"checkpoint": self.checkpoint, "timestamp": self.timestamp}
        for name in [
            "sentiment",
            "emotions",
            "safety_generic",
            "safety_familyos",
            "ner",
            "temporal",
            "nli",
        ]:
            val = getattr(self, name)
            if val is not None:
                result[name] = vars(val)
        return result

    def save(self, path: str | Path) -> None:
        """Save results to JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 60,
            "📊 FamilyOS Benchmark Results",
            "=" * 60,
            f"Checkpoint: {self.checkpoint}",
            "",
        ]

        if self.sentiment:
            lines.extend(
                [
                    "📝 SENTIMENT",
                    f"   Strict 5-class accuracy:    {self.sentiment.strict_5class_accuracy:.1%}",
                    f"   Grouped 3-class accuracy:   {self.sentiment.grouped_3class_accuracy:.1%}",
                    f"   Binary direction accuracy:  {self.sentiment.binary_direction_accuracy:.1%}",
                    f"   Adjacent tolerance:         {self.sentiment.adjacent_tolerance_accuracy:.1%}",
                    "",
                ]
            )

        if self.emotions:
            lines.extend(
                [
                    "😊 EMOTIONS",
                    f"   Strict Micro-F1:            {self.emotions.strict_micro_f1:.1%}",
                    f"   Top-1 Recall:               {self.emotions.top_1_recall:.1%}",
                    f"   Top-2 Recall:               {self.emotions.top_2_recall:.1%}",
                    f"   Primary Emotion Accuracy:   {self.emotions.primary_emotion_accuracy:.1%}",
                    f"   Jaccard (Partial Match):    {self.emotions.partial_match_jaccard:.1%}",
                    f"   At-Least-One Correct:       {self.emotions.at_least_one_correct:.1%}",
                    "",
                ]
            )

        if self.safety_generic:
            lines.extend(
                [
                    "🛡️ SAFETY GENERIC (8 types)",
                    f"   Any-Toxic Recall:           {self.safety_generic.any_toxic_recall:.1%}",
                    f"   Micro-F1:                   {self.safety_generic.micro_f1:.1%}",
                    f"   Macro-F1:                   {self.safety_generic.macro_f1:.1%}",
                    "",
                ]
            )

        if self.safety_familyos:
            lines.extend(
                [
                    "🚨 SAFETY FAMILYOS (4 bands)",
                    f"   CRISIS Recall:              {self.safety_familyos.crisis_recall:.1%}",
                    f"   RED Recall:                 {self.safety_familyos.red_recall:.1%}",
                    f"   AMBER Recall:               {self.safety_familyos.amber_recall:.1%}",
                    f"   GREEN Precision:            {self.safety_familyos.green_precision:.1%}",
                    f"   Cultural FP Rate:           {self.safety_familyos.cultural_fp_rate:.1%}",
                    f"   Keyword Override Accuracy:  {self.safety_familyos.keyword_override_accuracy:.1%}",
                    "",
                ]
            )

        if self.ner:
            lines.extend(
                [
                    "👤 NER",
                    f"   Token-level F1:             {self.ner.token_level_f1:.1%}",
                    f"   Entity-level F1:            {self.ner.entity_level_f1:.1%}",
                    "",
                ]
            )

        if self.temporal:
            lines.extend(
                [
                    "🕐 TEMPORAL",
                    f"   Overall F1:                 {self.temporal.overall_f1:.1%}",
                    "",
                ]
            )

        if self.nli:
            lines.extend(
                [
                    "🔗 NLI",
                    f"   3-class Accuracy:           {self.nli.three_class_accuracy:.1%}",
                    f"   Binary Conflict Accuracy:   {self.nli.binary_conflict_accuracy:.1%}",
                    f"   Contradiction Recall:       {self.nli.contradiction_recall:.1%}",
                    "",
                ]
            )

        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# Utility Functions
# =============================================================================


def to_3class_sentiment(label: int) -> int:
    """Map 5-class sentiment to 3-class (negative=0, neutral=1, positive=2)."""
    if label in (0, 1):  # very_negative, negative
        return 0
    elif label == 2:  # neutral
        return 1
    else:  # positive, very_positive (3, 4)
        return 2


def is_adjacent(pred: int, label: int) -> bool:
    """Check if prediction is within ±1 of true label."""
    return abs(pred - label) <= 1


def top_k_recall(predictions: set, ground_truth: set, k: int) -> bool:
    """Check if at least K ground truth items were predicted."""
    if not ground_truth:
        return True
    correct = len(predictions & ground_truth)
    return correct >= min(k, len(ground_truth))


def jaccard_similarity(predictions: set, ground_truth: set) -> float:
    """Compute Jaccard similarity (intersection / union)."""
    if not predictions and not ground_truth:
        return 1.0
    intersection = len(predictions & ground_truth)
    union = len(predictions | ground_truth)
    return intersection / union if union > 0 else 0.0


def find_best_checkpoint(output_dir: Path) -> Path:
    """Find the best or latest checkpoint."""
    best = output_dir / "checkpoint-best"
    if best.exists():
        return best

    checkpoints = [c for c in output_dir.glob("checkpoint-*") if c.name != "checkpoint-best"]
    if checkpoints:
        return max(checkpoints, key=lambda x: int(x.name.split("-")[1]))

    return output_dir


# =============================================================================
# Random Baseline Computation
# =============================================================================


def compute_random_baselines() -> dict:
    """
    Compute random baseline metrics for comparison.

    This shows what performance to expect from random guessing,
    which helps contextualize our model's actual performance.
    """
    print("\n🎲 Computing RANDOM BASELINES...")

    baselines = {}

    # Sentiment (5-class)
    # Random: 1/5 = 20% for exact match
    # Grouped 3-class: Random is ~33% (uneven class distribution)
    baselines["sentiment"] = {
        "strict_5class_accuracy": 0.20,  # 1/5
        "grouped_3class_accuracy": 0.33,  # ~1/3
        "binary_direction_accuracy": 0.50,  # 1/2
        "adjacent_tolerance": 0.52,  # P(|pred - true| <= 1) for uniform
    }
    print(f"   Sentiment 5-class random baseline: 20%")
    print(f"   Sentiment 3-class random baseline: 33%")

    # Emotions (44-class multi-label, ~3 active per sample)
    # Random prediction of 3 emotions: P(any correct) ≈ 3/44 * 3 ≈ 20%
    # Top-2 with 3 predictions and 3 labels: depends on overlap
    baselines["emotions"] = {
        "strict_micro_f1": 0.07,  # Very low for multi-label
        "top_1_recall": 0.20,  # ~20% chance of hitting 1 correct
        "top_2_recall": 0.05,  # Much lower for 2 correct
        "primary_emotion_accuracy": 0.023,  # 1/44
        "at_least_one_correct": 0.20,
    }
    print(f"   Emotions random baseline (micro-F1): ~7%")
    print(f"   Emotions random baseline (top-1): ~20%")

    # Safety Generic (8-type multi-label)
    # If predicting all 1s: 100% recall, ~12% precision (if 12% are actually toxic)
    baselines["safety_generic"] = {
        "any_toxic_recall": 1.0,  # If predicting all toxic
        "micro_f1": 0.125,  # 1/8 random
        "precision": 0.125,
    }
    print(f"   Safety generic random baseline: ~12.5%")

    # Safety FamilyOS (4 bands)
    baselines["safety_familyos"] = {
        "accuracy": 0.25,  # 1/4
        "crisis_recall": 0.25,  # If predicting randomly
        "cultural_fp_rate": 0.50,  # Random would flag ~50% as RED/CRISIS
    }
    print(f"   Safety FamilyOS random baseline: 25%")

    # NLI (3-class)
    baselines["nli"] = {
        "three_class_accuracy": 0.33,  # 1/3
        "binary_conflict_accuracy": 0.50,  # 1/2 for conflict/no-conflict
        "contradiction_recall": 0.33,
    }
    print(f"   NLI random baseline: 33%")

    return baselines


def print_baseline_comparison(results: "BenchmarkResults", baselines: dict) -> str:
    """Print comparison of results vs random baselines."""
    lines = [
        "",
        "=" * 60,
        "📊 COMPARISON vs RANDOM BASELINE",
        "=" * 60,
    ]

    if results.sentiment and "sentiment" in baselines:
        b = baselines["sentiment"]
        r = results.sentiment
        lines.extend(
            [
                "",
                "📝 SENTIMENT",
                f"   5-class:  {r.strict_5class_accuracy:.1%} vs {b['strict_5class_accuracy']:.1%} baseline "
                f"({r.strict_5class_accuracy/b['strict_5class_accuracy']:.1f}x)",
                f"   3-class:  {r.grouped_3class_accuracy:.1%} vs {b['grouped_3class_accuracy']:.1%} baseline "
                f"({r.grouped_3class_accuracy/b['grouped_3class_accuracy']:.1f}x)",
            ]
        )

    if results.emotions and "emotions" in baselines:
        b = baselines["emotions"]
        r = results.emotions
        lines.extend(
            [
                "",
                "😊 EMOTIONS",
                (
                    f"   Micro-F1: {r.strict_micro_f1:.1%} vs {b['strict_micro_f1']:.1%} baseline "
                    f"({r.strict_micro_f1/b['strict_micro_f1']:.1f}x)"
                    if b["strict_micro_f1"] > 0
                    else ""
                ),
                (
                    f"   Top-1:    {r.top_1_recall:.1%} vs {b['top_1_recall']:.1%} baseline "
                    f"({r.top_1_recall/b['top_1_recall']:.1f}x)"
                    if b["top_1_recall"] > 0
                    else ""
                ),
            ]
        )

    if results.nli and "nli" in baselines:
        b = baselines["nli"]
        r = results.nli
        lines.extend(
            [
                "",
                "🔗 NLI",
                f"   3-class:  {r.three_class_accuracy:.1%} vs {b['three_class_accuracy']:.1%} baseline "
                f"({r.three_class_accuracy/b['three_class_accuracy']:.1f}x)",
            ]
        )

    lines.append("=" * 60)
    return "\n".join(lines)


# =============================================================================
# Base Model Benchmarking (HuggingFace pre-trained)
# =============================================================================


def benchmark_with_base_models(tasks: list[str], device: torch.device) -> dict:
    """
    Benchmark using pre-trained HuggingFace models as comparison.

    This uses models that were NOT fine-tuned on our specific tasks,
    showing what off-the-shelf models can achieve.
    """
    from transformers import pipeline

    print("\n🏠 Benchmarking with BASE MODELS (pre-trained)...")

    results = {}

    if "sentiment" in tasks:
        print("\n   Loading sentiment model (distilbert-sst2)...")
        try:
            sentiment_pipe = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if device.type == "cuda" else -1,
            )

            # Load SST-2 validation
            ds = load_dataset("glue", "sst2", split="validation")
            ds = ds.select(range(min(500, len(ds))))

            correct = 0
            for sample in tqdm(ds, desc="   Sentiment (base)"):
                pred = sentiment_pipe(sample["sentence"])[0]
                pred_label = 1 if pred["label"] == "POSITIVE" else 0
                if pred_label == sample["label"]:
                    correct += 1

            results["sentiment_base"] = {
                "model": "distilbert-base-uncased-finetuned-sst-2-english",
                "binary_accuracy": correct / len(ds),
            }
            print(f"   ✅ Base sentiment binary accuracy: {correct / len(ds):.1%}")

        except Exception as e:
            print(f"   ⚠️ Could not benchmark base sentiment: {e}")

    if "nli" in tasks:
        print("\n   Loading NLI model (roberta-large-mnli)...")
        try:
            nli_pipe = pipeline(
                "text-classification",
                model="roberta-large-mnli",
                device=0 if device.type == "cuda" else -1,
            )

            # Load SNLI validation
            ds = load_dataset("snli", split="validation")
            ds = ds.filter(lambda x: x["label"] != -1)
            ds = ds.select(range(min(500, len(ds))))

            # MNLI label mapping
            label_map = {"ENTAILMENT": 0, "NEUTRAL": 1, "CONTRADICTION": 2}

            correct = 0
            for sample in tqdm(ds, desc="   NLI (base)"):
                text = f"{sample['premise']} </s></s> {sample['hypothesis']}"
                pred = nli_pipe(text)[0]
                pred_label = label_map.get(pred["label"], -1)
                if pred_label == sample["label"]:
                    correct += 1

            results["nli_base"] = {
                "model": "roberta-large-mnli",
                "three_class_accuracy": correct / len(ds),
            }
            print(f"   ✅ Base NLI accuracy: {correct / len(ds):.1%}")

        except Exception as e:
            print(f"   ⚠️ Could not benchmark base NLI: {e}")

    if "emotions" in tasks:
        print("\n   Loading emotions model (roberta-base-go_emotions)...")
        try:
            emotions_pipe = pipeline(
                "text-classification",
                model="SamLowe/roberta-base-go_emotions",
                device=0 if device.type == "cuda" else -1,
                top_k=5,  # Get top 5 predictions
            )

            # Load GoEmotions test
            ds = load_dataset("google-research-datasets/go_emotions", "simplified", split="test")
            ds = ds.select(range(min(500, len(ds))))

            # GoEmotions has 28 labels
            at_least_one = 0
            total_with_labels = 0

            for sample in tqdm(ds, desc="   Emotions (base)"):
                if not sample["labels"]:
                    continue
                total_with_labels += 1

                preds = emotions_pipe(sample["text"])
                pred_labels = {p["label"].lower() for p in preds}

                # GoEmotions label names
                go_emotions_labels = [
                    "admiration",
                    "amusement",
                    "anger",
                    "annoyance",
                    "approval",
                    "caring",
                    "confusion",
                    "curiosity",
                    "desire",
                    "disappointment",
                    "disapproval",
                    "disgust",
                    "embarrassment",
                    "excitement",
                    "fear",
                    "gratitude",
                    "grief",
                    "joy",
                    "love",
                    "nervousness",
                    "optimism",
                    "pride",
                    "realization",
                    "relief",
                    "remorse",
                    "sadness",
                    "surprise",
                    "neutral",
                ]
                true_labels = {go_emotions_labels[i] for i in sample["labels"]}

                if pred_labels & true_labels:
                    at_least_one += 1

            results["emotions_base"] = {
                "model": "SamLowe/roberta-base-go_emotions",
                "at_least_one_correct": (
                    at_least_one / total_with_labels if total_with_labels > 0 else 0
                ),
            }
            print(f"   ✅ Base emotions at-least-one: {at_least_one / total_with_labels:.1%}")

        except Exception as e:
            print(f"   ⚠️ Could not benchmark base emotions: {e}")

    return results


def benchmark_raw_modernbert(tasks: list[str], device: torch.device) -> dict:
    """
    Benchmark raw ModernBERT-base (answerdotai/ModernBERT-base) without any finetuning.

    This is the 2T token pretrained model straight from AnswerAI.
    Since it has no classification heads, we use simple probing approaches:
    - Sentiment: Linear probe on [CLS] embedding
    - NLI: Cosine similarity between premise/hypothesis embeddings
    - Emotions: Can't do without training
    - Safety: Can't do without training

    This gives a fair comparison of what ModernBERT can do out-of-box vs our finetuned model.
    """
    from transformers import AutoModel, AutoTokenizer

    print("\n" + "=" * 70)
    print("🔧 Benchmarking RAW ModernBERT-base (answerdotai/ModernBERT-base)")
    print("   2T tokens pretrained, NO finetuning, NO classification heads")
    print("=" * 70)

    results = {}

    # Load raw ModernBERT
    print("\n   Loading answerdotai/ModernBERT-base...")
    try:
        base_model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")
        base_tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        base_model = base_model.to(device)
        base_model.eval()

        params = sum(p.numel() for p in base_model.parameters())
        print(f"   ✅ Loaded: {params/1e6:.1f}M parameters")
        print(f"   Hidden size: {base_model.config.hidden_size}")
        print(f"   Layers: {base_model.config.num_hidden_layers}")
        print(f"   Max position: {base_model.config.max_position_embeddings}")
    except Exception as e:
        print(f"   ❌ Could not load ModernBERT-base: {e}")
        return results

    # Sentiment - use embedding similarity to sentiment anchors
    if "sentiment" in tasks:
        print("\n   📝 Testing SENTIMENT (embedding similarity approach)...")
        try:
            # Define sentiment anchor phrases
            sentiment_anchors = {
                0: "terrible awful horrible negative bad",  # very negative
                1: "bad disappointing poor negative",  # negative
                2: "okay neutral average fine",  # neutral
                3: "good nice positive pleasant",  # positive
                4: "amazing wonderful excellent fantastic",  # very positive
            }

            # Get anchor embeddings
            anchor_embeddings = {}
            with torch.no_grad():
                for label, text in sentiment_anchors.items():
                    inputs = base_tokenizer(
                        text, return_tensors="pt", truncation=True, max_length=512
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = base_model(**inputs)
                    # Use mean pooling
                    emb = outputs.last_hidden_state.mean(dim=1)
                    anchor_embeddings[label] = emb / emb.norm(dim=-1, keepdim=True)

            # Load SST-2 validation (same as checkpoint benchmark uses)
            ds = load_dataset("glue", "sst2", split="validation")
            ds = ds.select(range(min(500, len(ds))))

            correct_5class = 0
            correct_3class = 0
            correct_binary = 0
            total = 0
            total_binary = 0

            with torch.no_grad():
                for sample in tqdm(ds, desc="   Sentiment (raw)"):
                    text = sample["sentence"]
                    # SST-2 is binary: 0=negative, 1=positive
                    # Map to 5-class: 0→1 (negative), 1→3 (positive)
                    true_binary = sample["label"]
                    true_5class = 1 if true_binary == 0 else 3

                    inputs = base_tokenizer(
                        text, return_tensors="pt", truncation=True, max_length=512
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = base_model(**inputs)
                    emb = outputs.last_hidden_state.mean(dim=1)
                    emb = emb / emb.norm(dim=-1, keepdim=True)

                    # Find most similar anchor
                    similarities = {
                        label: (emb @ anchor.T).item()
                        for label, anchor in anchor_embeddings.items()
                    }
                    pred_5class = max(similarities, key=similarities.get)

                    if pred_5class == true_5class:
                        correct_5class += 1

                    # 3-class grouping: 0,1 -> negative, 2 -> neutral, 3,4 -> positive
                    true_3class = 0 if true_5class < 2 else (1 if true_5class == 2 else 2)
                    pred_3class = 0 if pred_5class < 2 else (1 if pred_5class == 2 else 2)
                    if true_3class == pred_3class:
                        correct_3class += 1

                    # Binary: negative (0,1) vs positive (3,4)
                    pred_binary = 0 if pred_5class < 2 else 1
                    if pred_binary == true_binary:
                        correct_binary += 1
                    total_binary += 1

                    total += 1

            results["sentiment_raw"] = {
                "5class_accuracy": correct_5class / total if total > 0 else 0,
                "3class_accuracy": correct_3class / total if total > 0 else 0,
                "binary_accuracy": correct_binary / total_binary if total_binary > 0 else 0,
                "method": "embedding_similarity_to_anchors",
            }
            print(f"   ✅ Raw ModernBERT sentiment 5-class: {correct_5class / total:.1%}")
            print(f"   ✅ Raw ModernBERT sentiment 3-class: {correct_3class / total:.1%}")
            print(f"   ✅ Raw ModernBERT sentiment binary:  {correct_binary / total_binary:.1%}")

        except Exception as e:
            print(f"   ⚠️ Could not benchmark raw sentiment: {e}")
            import traceback

            traceback.print_exc()

    # NLI - use embedding similarity between premise and hypothesis
    if "nli" in tasks:
        print("\n   🔗 Testing NLI (embedding similarity approach)...")
        try:
            ds = load_dataset("snli", split="validation")
            ds = ds.filter(lambda x: x["label"] != -1)
            ds = ds.select(range(min(500, len(ds))))

            correct = 0
            total = 0

            with torch.no_grad():
                for sample in tqdm(ds, desc="   NLI (raw)"):
                    premise = sample["premise"]
                    hypothesis = sample["hypothesis"]
                    true_label = sample["label"]  # 0=entailment, 1=neutral, 2=contradiction

                    # Get embeddings for premise and hypothesis separately
                    p_inputs = base_tokenizer(
                        premise, return_tensors="pt", truncation=True, max_length=256
                    )
                    p_inputs = {k: v.to(device) for k, v in p_inputs.items()}
                    p_outputs = base_model(**p_inputs)
                    p_emb = p_outputs.last_hidden_state.mean(dim=1)
                    p_emb = p_emb / p_emb.norm(dim=-1, keepdim=True)

                    h_inputs = base_tokenizer(
                        hypothesis, return_tensors="pt", truncation=True, max_length=256
                    )
                    h_inputs = {k: v.to(device) for k, v in h_inputs.items()}
                    h_outputs = base_model(**h_inputs)
                    h_emb = h_outputs.last_hidden_state.mean(dim=1)
                    h_emb = h_emb / h_emb.norm(dim=-1, keepdim=True)

                    # Cosine similarity
                    similarity = (p_emb @ h_emb.T).item()

                    # Heuristic: high similarity = entailment, low = contradiction, middle = neutral
                    if similarity > 0.85:
                        pred_label = 0  # entailment
                    elif similarity < 0.70:
                        pred_label = 2  # contradiction
                    else:
                        pred_label = 1  # neutral

                    if pred_label == true_label:
                        correct += 1
                    total += 1

            results["nli_raw"] = {
                "3class_accuracy": correct / total if total > 0 else 0,
                "method": "embedding_cosine_similarity_heuristic",
            }
            print(f"   ✅ Raw ModernBERT NLI 3-class: {correct / total:.1%}")

        except Exception as e:
            print(f"   ⚠️ Could not benchmark raw NLI: {e}")
            import traceback

            traceback.print_exc()

    # Emotions - can't really benchmark without training, but can try zero-shot
    if "emotions" in tasks:
        print("\n   😊 EMOTIONS: Skipped (requires trained classification head)")
        results["emotions_raw"] = {
            "note": "Cannot benchmark without classification head - raw model has no emotion labels",
            "method": "N/A",
        }

    # Safety - same issue
    if "safety_generic" in tasks or "safety_familyos" in tasks:
        print("\n   🛡️ SAFETY: Skipped (requires trained classification head)")
        results["safety_raw"] = {
            "note": "Cannot benchmark without classification head - raw model has no safety labels",
            "method": "N/A",
        }

    # Clean up
    del base_model
    torch.cuda.empty_cache()

    return results


def print_modernbert_comparison(checkpoint_results: "BenchmarkResults", raw_results: dict) -> str:
    """Print side-by-side comparison of checkpoint vs raw ModernBERT."""
    lines = [
        "",
        "=" * 70,
        "📊 COMPARISON: Your Checkpoint vs Raw ModernBERT-base (2T pretrained)",
        "=" * 70,
        "",
        f"{'Task':<20} {'Your Model':<20} {'Raw ModernBERT':<20} {'Improvement':<15}",
        "-" * 75,
    ]

    # Sentiment comparison
    if checkpoint_results.sentiment and "sentiment_raw" in raw_results:
        ours = checkpoint_results.sentiment.strict_5class_accuracy
        raw = raw_results["sentiment_raw"]["5class_accuracy"]
        diff = ours - raw
        mult = ours / raw if raw > 0 else float("inf")
        lines.append(
            f"{'Sentiment 5-class':<20} {ours:<20.1%} {raw:<20.1%} {'+' if diff > 0 else ''}{diff:.1%} ({mult:.1f}x)"
        )

        ours = checkpoint_results.sentiment.grouped_3class_accuracy
        raw = raw_results["sentiment_raw"]["3class_accuracy"]
        diff = ours - raw
        mult = ours / raw if raw > 0 else float("inf")
        lines.append(
            f"{'Sentiment 3-class':<20} {ours:<20.1%} {raw:<20.1%} {'+' if diff > 0 else ''}{diff:.1%} ({mult:.1f}x)"
        )

    # NLI comparison
    if checkpoint_results.nli and "nli_raw" in raw_results:
        ours = checkpoint_results.nli.three_class_accuracy
        raw = raw_results["nli_raw"]["3class_accuracy"]
        diff = ours - raw
        mult = ours / raw if raw > 0 else float("inf")
        lines.append(
            f"{'NLI 3-class':<20} {ours:<20.1%} {raw:<20.1%} {'+' if diff > 0 else ''}{diff:.1%} ({mult:.1f}x)"
        )

    # Emotions - our model vs N/A
    if checkpoint_results.emotions:
        ours = checkpoint_results.emotions.strict_micro_f1
        lines.append(f"{'Emotions Micro-F1':<20} {ours:<20.1%} {'N/A (no head)':<20} {'N/A':<15}")

    # Safety - our model vs N/A
    if checkpoint_results.safety_generic:
        ours = checkpoint_results.safety_generic.micro_f1
        lines.append(f"{'Safety Micro-F1':<20} {ours:<20.1%} {'N/A (no head)':<20} {'N/A':<15}")

    lines.extend(
        [
            "-" * 75,
            "",
            "Note: Raw ModernBERT has NO classification heads. Sentiment/NLI use embedding",
            "      similarity heuristics. Emotions/Safety require trained heads.",
            "=" * 70,
        ]
    )

    return "\n".join(lines)


# =============================================================================
# Data Loading
# =============================================================================


def load_familyos_emotions_validation(tokenizer) -> Dataset:
    """Load FamilyOS emotions validation set."""
    data_path = project_root / "data" / "familyos" / "emotions" / "gold" / "validation.jsonl"

    if not data_path.exists():
        raise FileNotFoundError(f"Emotions validation data not found: {data_path}")

    # Load JSONL
    samples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    # Convert to dataset format
    texts = [s["text"] for s in samples]
    emotion_lists = [s["emotions"] for s in samples]
    primary_emotions = [
        s.get("primary_emotion", s["emotions"][0] if s["emotions"] else None) for s in samples
    ]

    # Convert to multi-hot using FamilyOS emotion schema
    label2id = EMOTIONS_FAMILYOS_LABELS.label2id
    num_labels = EMOTIONS_FAMILYOS_LABELS.num_labels

    multi_hot_labels = []
    for emotions in emotion_lists:
        vec = [0] * num_labels
        for em in emotions:
            if em in label2id:
                vec[label2id[em]] = 1
        multi_hot_labels.append(vec)

    # Create dataset
    ds = Dataset.from_dict(
        {
            "text": texts,
            "labels": multi_hot_labels,
            "primary_emotion": primary_emotions,
            "emotion_list": emotion_lists,
        }
    )

    # Tokenize
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding=False)

    ds = ds.map(tokenize, batched=True)
    return ds


def load_familyos_safety_validation(tokenizer, data_type: str = "gold") -> Dataset:
    """Load FamilyOS safety validation set."""
    data_path = project_root / "data" / "familyos" / "safety" / data_type / "validation.jsonl"

    if not data_path.exists():
        raise FileNotFoundError(f"Safety validation data not found: {data_path}")

    samples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    texts = [s["text"] for s in samples]
    labels = [s["label"] for s in samples]

    ds = Dataset.from_dict({"text": texts, "labels": labels})

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding=False)

    ds = ds.map(tokenize, batched=True)
    return ds


# =============================================================================
# Cultural Robustness Test Data
# =============================================================================

INDIAN_VENTING_PATTERNS = [
    "I'll die of embarrassment",
    "This is killing me",
    "I could die",
    "My head is bursting",
    "I'm going mad",
    "I want to kill the mood",
    "My head is exploding with tension",
    "This exam will be the death of me",
    "Mummy will kill me if I fail",
    "Papa will have a heart attack when he sees the bill",
    "The traffic is killing me today",
    "I could die from this heat",
    "These kids are driving me crazy",
    "My boss is killing me with deadlines",
    "I'm dying of boredom in this meeting",
]

CRISIS_KEYWORDS = [
    "I want to kill myself",
    "I'm going to end my life",
    "I don't want to live anymore",
    "suicide seems like the only option",
    "I want to die",
    "I've been thinking about hurting myself",
    "I can't go on living like this",
    "ending it all would be better",
]


# =============================================================================
# Benchmark Functions
# =============================================================================


def benchmark_sentiment(
    model: ModernBertMultiTaskModel,
    tokenizer,
    device: torch.device,
) -> SentimentResults:
    """Run sentiment benchmarking with FamilyOS-appropriate metrics."""
    print("\n📝 Benchmarking SENTIMENT...")

    results = SentimentResults()

    # Load SST-2 validation (binary → mapped to 5-class)
    try:
        ds = load_dataset("glue", "sst2", split="validation")
    except Exception as e:
        print(f"   ⚠️ Could not load SST-2: {e}")
        return results

    # Map SST-2 binary to 5-class: 0→1 (negative), 1→3 (positive)
    def map_labels(ex):
        ex["label_5class"] = 1 if ex["label"] == 0 else 3
        ex["label_binary"] = ex["label"]
        return ex

    ds = ds.map(map_labels)
    ds = ds.rename_column("sentence", "text")
    ds = ds.select(range(min(872, len(ds))))  # Full validation set

    # Tokenize
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding=False)

    ds = ds.map(tokenize, batched=True)

    # Create dataloader
    collator = MultiTaskCollator(tokenizer=tokenizer)

    def add_task(ex):
        ex["task"] = "sentiment"
        ex["labels"] = ex["label_5class"]
        return ex

    ds = ds.map(add_task)
    loader = DataLoader(ds, batch_size=32, collate_fn=collator)

    # Collect predictions
    all_preds = []
    all_labels_5class = []
    all_labels_binary = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="   Sentiment"):
            inputs = {
                k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]
            }
            outputs = model(**inputs, capability="sentiment")
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels_5class.extend(batch["labels"].numpy())

    all_preds = np.array(all_preds)
    all_labels_5class = np.array(all_labels_5class)

    # Binary labels from 5-class: 0,1,2 → negative(0), 3,4 → positive(1)
    all_labels_binary = np.where(all_labels_5class >= 3, 1, 0)
    all_preds_binary = np.where(all_preds >= 3, 1, 0)

    # 1. Strict 5-class accuracy
    results.strict_5class_accuracy = accuracy_score(all_labels_5class, all_preds)

    # 2. Grouped 3-class accuracy
    labels_3class = np.array([to_3class_sentiment(l) for l in all_labels_5class])
    preds_3class = np.array([to_3class_sentiment(p) for p in all_preds])
    results.grouped_3class_accuracy = accuracy_score(labels_3class, preds_3class)

    # 3. Binary direction accuracy
    results.binary_direction_accuracy = accuracy_score(all_labels_binary, all_preds_binary)

    # 4. Adjacent tolerance (±1 class)
    adjacent_correct = sum(is_adjacent(p, l) for p, l in zip(all_preds, all_labels_5class))
    results.adjacent_tolerance_accuracy = adjacent_correct / len(all_preds)

    # 5. Per-class F1
    class_names = ["very_negative", "negative", "neutral", "positive", "very_positive"]
    report = classification_report(all_labels_5class, all_preds, output_dict=True, zero_division=0)
    for i, name in enumerate(class_names):
        if str(i) in report:
            results.per_class_f1[name] = report[str(i)]["f1-score"]

    # 6. Confusion matrices
    results.confusion_matrix_5x5 = confusion_matrix(all_labels_5class, all_preds).tolist()
    results.confusion_matrix_3x3 = confusion_matrix(labels_3class, preds_3class).tolist()

    # 7. Negative recall, Positive precision, Neutral F1
    # For 3-class
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels_3class, preds_3class, labels=[0, 1, 2], zero_division=0
    )
    results.negative_recall = rec[0]
    results.positive_precision = prec[2]
    results.neutral_f1 = f1[1]

    print(f"   ✅ Strict 5-class: {results.strict_5class_accuracy:.1%}")
    print(f"   ✅ Grouped 3-class: {results.grouped_3class_accuracy:.1%}")
    print(f"   ✅ Binary direction: {results.binary_direction_accuracy:.1%}")

    return results


def benchmark_emotions(
    model: ModernBertMultiTaskModel,
    tokenizer,
    device: torch.device,
) -> EmotionsResults:
    """Run emotions benchmarking with FamilyOS-appropriate metrics."""
    print("\n😊 Benchmarking EMOTIONS...")

    results = EmotionsResults()

    try:
        ds = load_familyos_emotions_validation(tokenizer)
    except FileNotFoundError as e:
        print(f"   ⚠️ {e}")
        return results

    # Create dataloader
    collator = MultiTaskCollator(tokenizer=tokenizer)

    def add_task(ex):
        ex["task"] = "emotions"
        return ex

    ds = ds.map(add_task)
    loader = DataLoader(ds, batch_size=32, collate_fn=collator)

    id2label = EMOTIONS_FAMILYOS_LABELS.id2label
    label2id = EMOTIONS_FAMILYOS_LABELS.label2id

    # Collect predictions
    all_preds_binary = []
    all_labels_binary = []
    all_preds_indices = []
    all_labels_indices = []
    all_primary_emotions = ds["primary_emotion"]
    all_primary_correct = []

    # Check for dynamic thresholds in emotions head
    emotions_head = model.heads["emotions"] if "emotions" in model.heads else None
    use_dynamic_thresholds = False
    learned_thresholds = None

    if emotions_head is not None and hasattr(emotions_head, "use_dynamic_thresholds"):
        use_dynamic_thresholds = getattr(emotions_head, "use_dynamic_thresholds", False)
        if use_dynamic_thresholds and hasattr(emotions_head, "raw_thresholds"):
            raw_thresh = emotions_head.raw_thresholds.detach()
            learned_thresholds = torch.sigmoid(raw_thresh).cpu().numpy()
            print(
                f"   📊 Using LEARNED dynamic thresholds (mean={learned_thresholds.mean():.3f}, min={learned_thresholds.min():.3f}, max={learned_thresholds.max():.3f})"
            )
        else:
            print(f"   ⚠️ Dynamic thresholds enabled but raw_thresholds not found")

    if not use_dynamic_thresholds:
        print(f"   ℹ️ Using fixed threshold: 0.3")
        learned_thresholds = np.full(len(id2label), 0.3)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="   Emotions"):
            inputs = {
                k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]
            }
            outputs = model(**inputs, capability="emotions")
            logits = outputs.logits.cpu()

            # Apply sigmoid and use dynamic or fixed thresholds
            probs = torch.sigmoid(logits)
            # Apply per-emotion thresholds (broadcast thresholds across batch)
            threshold_tensor = torch.tensor(learned_thresholds, dtype=probs.dtype)
            preds_binary = (probs > threshold_tensor).int().numpy()

            all_preds_binary.extend(preds_binary)
            all_labels_binary.extend(batch["labels"].numpy())

            # Get indices for Top-K and Jaccard calculations
            for i in range(len(preds_binary)):
                pred_indices = set(np.where(preds_binary[i] == 1)[0])
                label_indices = set(np.where(batch["labels"][i].numpy() == 1)[0])
                all_preds_indices.append(pred_indices)
                all_labels_indices.append(label_indices)

                # Primary emotion accuracy
                pred_primary = logits[i].argmax().item()
                all_primary_correct.append(pred_primary)

    all_preds_binary = np.array(all_preds_binary)
    all_labels_binary = np.array(all_labels_binary)

    # 1. Strict Micro-F1 and Macro-F1
    results.strict_micro_f1 = f1_score(
        all_labels_binary, all_preds_binary, average="micro", zero_division=0
    )
    results.strict_macro_f1 = f1_score(
        all_labels_binary, all_preds_binary, average="macro", zero_division=0
    )

    # 2. Top-K Recall
    top1_correct = sum(
        top_k_recall(p, l, k=1) for p, l in zip(all_preds_indices, all_labels_indices)
    )
    top2_correct = sum(
        top_k_recall(p, l, k=2) for p, l in zip(all_preds_indices, all_labels_indices)
    )
    top3_correct = sum(
        top_k_recall(p, l, k=3) for p, l in zip(all_preds_indices, all_labels_indices)
    )
    n_samples = len(all_preds_indices)
    results.top_1_recall = top1_correct / n_samples
    results.top_2_recall = top2_correct / n_samples
    results.top_3_recall = top3_correct / n_samples

    # 3. Primary emotion accuracy
    primary_correct = 0
    for i, primary_em in enumerate(all_primary_emotions):
        if primary_em and primary_em in label2id:
            if all_primary_correct[i] == label2id[primary_em]:
                primary_correct += 1
    results.primary_emotion_accuracy = primary_correct / n_samples if n_samples > 0 else 0.0

    # 4. Partial match (Jaccard)
    jaccard_scores = [
        jaccard_similarity(p, l) for p, l in zip(all_preds_indices, all_labels_indices)
    ]
    results.partial_match_jaccard = np.mean(jaccard_scores)

    # 5. At-least-one correct
    at_least_one = sum(
        len(p & l) >= 1
        for p, l in zip(all_preds_indices, all_labels_indices)
        if l  # Only count samples with labels
    )
    samples_with_labels = sum(1 for l in all_labels_indices if l)
    results.at_least_one_correct = (
        at_least_one / samples_with_labels if samples_with_labels > 0 else 0.0
    )

    # 6. Statistics
    results.avg_predictions_per_sample = np.mean([len(p) for p in all_preds_indices])
    results.avg_ground_truth_per_sample = np.mean([len(l) for l in all_labels_indices])

    print(f"   ✅ Strict Micro-F1: {results.strict_micro_f1:.1%}")
    print(f"   ✅ Top-2 Recall: {results.top_2_recall:.1%}")
    print(f"   ✅ At-Least-One: {results.at_least_one_correct:.1%}")

    # === THRESHOLD SWEEP: Test multiple thresholds to find optimal ===
    print("\n   " + "=" * 56)
    print("   🔍 THRESHOLD SWEEP: Testing optimal threshold")
    print("   " + "=" * 56)

    thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    best_threshold = 0.3
    best_f1 = results.strict_micro_f1

    # Collect all probs for threshold testing (need to recompute from logits)
    all_probs = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="   Threshold sweep"):
            inputs = {
                k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]
            }
            outputs = model(**inputs, capability="emotions")
            probs = torch.sigmoid(outputs.logits.cpu())
            all_probs.extend(probs.numpy())

    all_probs = np.array(all_probs)

    print(
        f"\n   {'Threshold':<12} {'Micro-F1':<12} {'Precision':<12} {'Recall':<12} {'Avg Pred':<10}"
    )
    print("   " + "-" * 58)

    for thresh in thresholds_to_test:
        preds_at_thresh = (all_probs > thresh).astype(int)

        micro_f1 = f1_score(all_labels_binary, preds_at_thresh, average="micro", zero_division=0)

        total_pred = preds_at_thresh.sum()
        total_true = all_labels_binary.sum()
        tp = ((preds_at_thresh == 1) & (all_labels_binary == 1)).sum()

        prec = tp / total_pred if total_pred > 0 else 0
        rec = tp / total_true if total_true > 0 else 0
        avg_pred = preds_at_thresh.sum(axis=1).mean()

        marker = " 👈 CURRENT" if thresh == 0.3 else ""
        if micro_f1 > best_f1:
            best_f1 = micro_f1
            best_threshold = thresh
            marker = " ⭐ BEST" if thresh != 0.3 else " 👈 CURRENT + BEST"

        print(
            f"   {thresh:<12.2f} {micro_f1:<12.1%} {prec:<12.1%} {rec:<12.1%} {avg_pred:<10.1f}{marker}"
        )

    print("\n   " + "-" * 58)
    print(f"   💡 OPTIMAL THRESHOLD: {best_threshold:.2f} (Micro-F1: {best_f1:.1%})")

    if best_threshold != 0.3:
        print(f"   ⚠️  RECOMMENDATION: Change threshold from 0.3 to {best_threshold:.2f}")
        improvement = (best_f1 - results.strict_micro_f1) / results.strict_micro_f1 * 100
        print(f"   📈 Expected improvement: +{improvement:.1f}%")

    # Check for degenerate emotions head (all thresholds at 0.5 = not trained)
    if learned_thresholds is not None:
        thresh_std = np.std(learned_thresholds)
        if thresh_std < 0.01:
            print(
                "\n   ⛔ CRITICAL: Dynamic thresholds NOT LEARNED (std={:.4f})".format(thresh_std)
            )
            print("      The thresholds are all at initialization value (~0.5).")
            print("      This indicates the emotions head may need re-training.")

    # Check for head collapse (same predictions for all samples)
    pred_variance = np.var(all_probs.mean(axis=0))
    if pred_variance < 0.02:
        print("\n   ⛔ CRITICAL: Possible head collapse detected!")
        print(f"      Per-emotion prediction variance: {pred_variance:.4f}")
        print("      The model predicts similar emotions for ALL inputs.")
        print("      Recommended fixes:")
        print("      1. Re-train with reduced pos_weight (remove or set to 1.0)")
        print("      2. Reduce asl_gamma_neg from 4.0 to 2.0")
        print("      3. Use configs/training/multitask/stage_a_a100_fast_v2_1.yaml")

    print("   " + "=" * 56)

    # === DIAGNOSTIC: Show sample predictions to debug low precision ===
    print("\n   " + "=" * 56)
    print("   📋 EMOTIONS DIAGNOSTIC: Sample Predictions")
    print("   " + "=" * 56)
    print(f"   Avg predictions/sample: {results.avg_predictions_per_sample:.1f}")
    print(f"   Avg ground truth/sample: {results.avg_ground_truth_per_sample:.1f}")

    # Calculate precision and recall separately
    total_pred_positive = all_preds_binary.sum()
    total_true_positive = all_labels_binary.sum()
    true_positives = ((all_preds_binary == 1) & (all_labels_binary == 1)).sum()

    precision = true_positives / total_pred_positive if total_pred_positive > 0 else 0
    recall = true_positives / total_true_positive if total_true_positive > 0 else 0

    print(f"\n   Precision: {precision:.1%} (TP={true_positives}, Pred+={total_pred_positive})")
    print(f"   Recall: {recall:.1%} (TP={true_positives}, True+={total_true_positive})")

    # Show 10 sample predictions
    print("\n   --- Sample Predictions (first 10) ---")
    texts = ds["text"][:10] if "text" in ds.features else ["[text not available]"] * 10

    for i in range(min(10, len(all_preds_indices))):
        pred_emotions = {id2label[idx] for idx in all_preds_indices[i]}
        true_emotions = {id2label[idx] for idx in all_labels_indices[i]}

        # Overlap analysis
        correct = pred_emotions & true_emotions
        missed = true_emotions - pred_emotions
        extra = pred_emotions - true_emotions

        text_preview = texts[i][:60] + "..." if len(texts[i]) > 60 else texts[i]

        print(f'\n   [{i+1}] "{text_preview}"')
        print(f"       TRUE:  {sorted(true_emotions)}")
        print(f"       PRED:  {sorted(pred_emotions)}")
        print(f"       ✓ Correct: {sorted(correct) if correct else 'none'}")
        if missed:
            print(f"       ✗ Missed:  {sorted(missed)}")
        if extra:
            print(f"       ✗ Extra:   {sorted(extra)}")

    # Show emotion prediction frequency
    print("\n   --- Top 10 Most Predicted Emotions ---")
    pred_counts = all_preds_binary.sum(axis=0)
    true_counts = all_labels_binary.sum(axis=0)

    top_pred_indices = np.argsort(pred_counts)[::-1][:10]
    for idx in top_pred_indices:
        emotion = id2label[idx]
        pred_count = int(pred_counts[idx])
        true_count = int(true_counts[idx])
        print(f"       {emotion:20s} Pred: {pred_count:3d}, True: {true_count:3d}")

    print("   " + "=" * 56)

    return results


def benchmark_safety_familyos(
    model: ModernBertMultiTaskModel,
    tokenizer,
    device: torch.device,
) -> SafetyResults:
    """Run safety FamilyOS benchmarking (4 policy bands)."""
    print("\n🚨 Benchmarking SAFETY FAMILYOS...")

    results = SafetyResults()

    try:
        ds = load_familyos_safety_validation(tokenizer, data_type="gold")
    except FileNotFoundError as e:
        print(f"   ⚠️ {e}")
        return results

    # Create dataloader
    collator = MultiTaskCollator(tokenizer=tokenizer)

    def add_task(ex):
        ex["task"] = "safety_familyos"
        return ex

    ds = ds.map(add_task)
    loader = DataLoader(ds, batch_size=32, collate_fn=collator)

    # Collect predictions
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="   Safety FamilyOS"):
            inputs = {
                k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]
            }
            try:
                outputs = model(**inputs, capability="safety_familyos")
                preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            except Exception:
                # If safety_familyos not available, try safety_generic
                print("   ⚠️ safety_familyos not available, skipping...")
                return results

            all_preds.extend(preds)
            all_labels.extend(batch["labels"].numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Band indices: GREEN=0, AMBER=1, RED=2, CRISIS=3
    band_names = ["GREEN", "AMBER", "RED", "CRISIS"]

    # Per-band recall
    for i, band_name in enumerate(band_names):
        mask = all_labels == i
        if mask.sum() > 0:
            recall = (all_preds[mask] == i).mean()
            results.per_type_recall[band_name] = float(recall)

    results.crisis_recall = results.per_type_recall.get("CRISIS", 0.0)
    results.red_recall = results.per_type_recall.get("RED", 0.0)
    results.amber_recall = results.per_type_recall.get("AMBER", 0.0)

    # GREEN precision
    green_pred_mask = all_preds == 0
    if green_pred_mask.sum() > 0:
        results.green_precision = (all_labels[green_pred_mask] == 0).mean()

    # Confusion matrix
    results.band_confusion_matrix = confusion_matrix(
        all_labels, all_preds, labels=[0, 1, 2, 3]
    ).tolist()

    # Cultural robustness testing
    print("   Testing cultural robustness...")
    cultural_fps = 0

    for text in INDIAN_VENTING_PATTERNS:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            try:
                outputs = model(**inputs, capability="safety_familyos")
                pred = outputs.logits.argmax(dim=-1).item()
                if pred >= 2:  # RED or CRISIS
                    cultural_fps += 1
            except Exception:
                pass

    results.cultural_fp_rate = cultural_fps / len(INDIAN_VENTING_PATTERNS)

    # Keyword override testing
    print("   Testing keyword override...")
    keyword_correct = 0

    for text in CRISIS_KEYWORDS:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            try:
                outputs = model(**inputs, capability="safety_familyos")
                pred = outputs.logits.argmax(dim=-1).item()
                if pred == 3:  # CRISIS
                    keyword_correct += 1
            except Exception:
                pass

    results.keyword_override_accuracy = keyword_correct / len(CRISIS_KEYWORDS)

    print(f"   ✅ CRISIS Recall: {results.crisis_recall:.1%}")
    print(f"   ✅ Cultural FP Rate: {results.cultural_fp_rate:.1%}")
    print(f"   ✅ Keyword Override: {results.keyword_override_accuracy:.1%}")

    return results


def benchmark_safety_generic(
    model: ModernBertMultiTaskModel,
    tokenizer,
    device: torch.device,
) -> SafetyResults:
    """Run safety generic benchmarking (8-type multi-label)."""
    print("\n🛡️ Benchmarking SAFETY GENERIC...")

    results = SafetyResults()

    # Try to load civil comments curated validation
    data_path = project_root / "data" / "public" / "civil_comments_curated" / "validation.jsonl"

    if not data_path.exists():
        print(f"   ⚠️ Civil comments validation not found: {data_path}")
        return results

    # Load data
    samples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    texts = [s["text"] for s in samples]
    labels = [s["labels"] for s in samples]

    ds = Dataset.from_dict({"text": texts, "labels": labels})

    # Tokenize
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding=False)

    ds = ds.map(tokenize, batched=True)

    # Create dataloader
    collator = MultiTaskCollator(tokenizer=tokenizer)

    def add_task(ex):
        ex["task"] = "safety_generic"
        return ex

    ds = ds.map(add_task)
    ds = ds.select(range(min(1000, len(ds))))  # Limit for speed
    loader = DataLoader(ds, batch_size=32, collate_fn=collator)

    # Collect predictions
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="   Safety Generic"):
            inputs = {
                k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]
            }
            outputs = model(**inputs, capability="safety_generic")
            logits = outputs.logits.cpu()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.3).int().numpy()

            all_preds.extend(preds)
            all_labels.extend(batch["labels"].numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Micro/Macro F1
    results.micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    results.macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # Any-toxic recall
    any_toxic_labels = all_labels.max(axis=1)
    any_toxic_preds = all_preds.max(axis=1)
    toxic_mask = any_toxic_labels == 1
    if toxic_mask.sum() > 0:
        results.any_toxic_recall = (any_toxic_preds[toxic_mask] == 1).mean()

    # Per-type recall
    type_names = list(SAFETY_GENERIC_LABELS.label2id.keys())
    for i, type_name in enumerate(type_names):
        type_mask = all_labels[:, i] == 1
        if type_mask.sum() > 0:
            results.per_type_recall[type_name] = float((all_preds[type_mask, i] == 1).mean())

    print(f"   ✅ Any-Toxic Recall: {results.any_toxic_recall:.1%}")
    print(f"   ✅ Micro-F1: {results.micro_f1:.1%}")

    # === THRESHOLD SWEEP for Safety ===
    print("\n   " + "=" * 56)
    print("   🔍 SAFETY THRESHOLD SWEEP")
    print("   " + "=" * 56)

    # Get all probs for threshold testing
    all_probs_safety = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="   Safety threshold sweep"):
            inputs = {
                k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]
            }
            outputs = model(**inputs, capability="safety_generic")
            probs = torch.sigmoid(outputs.logits.cpu())
            all_probs_safety.extend(probs.numpy())

    all_probs_safety = np.array(all_probs_safety)

    thresholds_safety = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    best_thresh_safety = 0.3
    best_f1_safety = results.micro_f1

    print(
        f"\n   {'Threshold':<12} {'Micro-F1':<12} {'Precision':<12} {'Recall':<12} {'Avg Pred':<10}"
    )
    print("   " + "-" * 58)

    for thresh in thresholds_safety:
        preds_at_thresh = (all_probs_safety > thresh).astype(int)

        micro_f1 = f1_score(all_labels, preds_at_thresh, average="micro", zero_division=0)

        total_pred = preds_at_thresh.sum()
        total_true = all_labels.sum()
        tp = ((preds_at_thresh == 1) & (all_labels == 1)).sum()

        prec = tp / total_pred if total_pred > 0 else 0
        rec = tp / total_true if total_true > 0 else 0
        avg_pred = preds_at_thresh.sum(axis=1).mean()

        marker = " 👈 CURRENT" if thresh == 0.3 else ""
        if micro_f1 > best_f1_safety:
            best_f1_safety = micro_f1
            best_thresh_safety = thresh
            marker = " ⭐ BEST"

        print(
            f"   {thresh:<12.2f} {micro_f1:<12.1%} {prec:<12.1%} {rec:<12.1%} {avg_pred:<10.1f}{marker}"
        )

    if best_thresh_safety != 0.3:
        print(f"\n   💡 OPTIMAL: {best_thresh_safety:.2f} (Micro-F1: {best_f1_safety:.1%})")

    # Check for head collapse
    pred_variance_safety = np.var(all_probs_safety.mean(axis=0))
    avg_preds_per_sample = all_preds.sum(axis=1).mean()

    if avg_preds_per_sample > 5:  # Predicting too many types
        print(f"\n   ⚠️ WARNING: Avg {avg_preds_per_sample:.1f} predictions/sample (expected ~1-2)")
        print("      Model may be over-predicting safety types.")

    print("   " + "=" * 56)

    return results


def benchmark_nli(
    model: ModernBertMultiTaskModel,
    tokenizer,
    device: torch.device,
) -> NLIResults:
    """Run NLI benchmarking."""
    print("\n🔗 Benchmarking NLI...")

    results = NLIResults()

    try:
        ds = load_dataset("snli", split="validation")
        ds = ds.filter(lambda x: x["label"] != -1)
        ds = ds.select(range(min(1000, len(ds))))
    except Exception as e:
        print(f"   ⚠️ Could not load SNLI: {e}")
        return results

    # Tokenize
    def tokenize(examples):
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation=True,
            max_length=512,
            padding=False,
        )

    ds = ds.map(tokenize, batched=True)
    ds = ds.rename_column("label", "labels")

    # Create dataloader
    collator = MultiTaskCollator(tokenizer=tokenizer)

    def add_task(ex):
        ex["task"] = "nli"
        return ex

    ds = ds.map(add_task, remove_columns=["premise", "hypothesis"])
    loader = DataLoader(ds, batch_size=32, collate_fn=collator)

    # Collect predictions
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="   NLI"):
            inputs = {
                k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]
            }
            outputs = model(**inputs, capability="nli")
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(batch["labels"].numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 3-class accuracy
    results.three_class_accuracy = accuracy_score(all_labels, all_preds)

    # Binary: conflict (contradiction=2) vs no-conflict (entailment=0, neutral=1)
    labels_binary = (all_labels == 2).astype(int)
    preds_binary = (all_preds == 2).astype(int)
    results.binary_conflict_accuracy = accuracy_score(labels_binary, preds_binary)

    # Contradiction recall
    contradiction_mask = all_labels == 2
    if contradiction_mask.sum() > 0:
        results.contradiction_recall = (all_preds[contradiction_mask] == 2).mean()

    # Entailment precision
    entailment_pred_mask = all_preds == 0
    if entailment_pred_mask.sum() > 0:
        results.entailment_precision = (all_labels[entailment_pred_mask] == 0).mean()

    # Confusion matrix
    results.confusion_matrix = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2]).tolist()

    print(f"   ✅ 3-class Accuracy: {results.three_class_accuracy:.1%}")
    print(f"   ✅ Binary Conflict: {results.binary_conflict_accuracy:.1%}")

    return results


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="FamilyOS Benchmarking Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Benchmark trained checkpoint
    python scripts/benchmark_familyos.py --checkpoint outputs/stage_a/checkpoint-best
    python scripts/benchmark_familyos.py --checkpoint outputs/stage_a --tasks sentiment emotions
    python scripts/benchmark_familyos.py --checkpoint outputs/stage_a --all-tasks

    # Compute random baselines only
    python scripts/benchmark_familyos.py --baseline-only

    # Benchmark with pre-trained HuggingFace models
    python scripts/benchmark_familyos.py --base-model --tasks sentiment nli

    # Full comparison (checkpoint + baselines)
    python scripts/benchmark_familyos.py --checkpoint outputs/stage_a --compare-baseline
        """,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["sentiment", "emotions", "safety_familyos"],
        help="Tasks to benchmark",
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Run all available benchmarks",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Only compute random baselines (no model needed)",
    )
    parser.add_argument(
        "--base-model",
        action="store_true",
        help="Benchmark with pre-trained HuggingFace models instead of checkpoint",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also show comparison vs random baseline",
    )
    parser.add_argument(
        "--compare-modernbert",
        action="store_true",
        help="Compare checkpoint vs raw ModernBERT-base (answerdotai/ModernBERT-base, 2T tokens, no finetuning)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for results JSON",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")

    # Determine tasks to run
    if args.all_tasks:
        tasks = ["sentiment", "emotions", "safety_generic", "safety_familyos", "nli"]
    else:
        tasks = args.tasks

    # Option 1: Random baselines only
    if args.baseline_only:
        baselines = compute_random_baselines()
        print("\n" + "=" * 60)
        print("📊 Random Baseline Summary")
        print("=" * 60)
        for task, metrics in baselines.items():
            print(f"\n{task}:")
            for metric, value in metrics.items():
                print(f"   {metric}: {value:.1%}")
        return

    # Option 2: Base model benchmarking
    if args.base_model:
        base_results = benchmark_with_base_models(tasks, device)
        print("\n" + "=" * 60)
        print("📊 Base Model (Pre-trained) Results")
        print("=" * 60)
        for task, metrics in base_results.items():
            print(f"\n{task}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"   {metric}: {value:.1%}")
                else:
                    print(f"   {metric}: {value}")
        return

    # Option 3: Benchmark trained checkpoint
    if not args.checkpoint:
        print("❌ Error: --checkpoint is required unless using --baseline-only or --base-model")
        parser.print_help()
        return

    # Find checkpoint
    checkpoint_path = find_best_checkpoint(Path(args.checkpoint))
    print(f"\n📁 Using checkpoint: {checkpoint_path}")

    # Load model
    print("\n🔄 Loading model...")
    model = ModernBertMultiTaskModel.load_checkpoint(str(checkpoint_path), device=str(device))
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
    model.eval()

    print(f"   Model capabilities: {model.capabilities}")

    # Run benchmarks
    import datetime

    results = BenchmarkResults(
        checkpoint=str(checkpoint_path),
        timestamp=datetime.datetime.now().isoformat(),
    )

    # Convert capabilities to strings for comparison
    capability_strs = [str(c.value) if hasattr(c, "value") else str(c) for c in model.capabilities]

    if "sentiment" in tasks and "sentiment" in capability_strs:
        results.sentiment = benchmark_sentiment(model, tokenizer, device)

    if "emotions" in tasks and "emotions" in capability_strs:
        results.emotions = benchmark_emotions(model, tokenizer, device)

    if "safety_generic" in tasks and "safety_generic" in capability_strs:
        results.safety_generic = benchmark_safety_generic(model, tokenizer, device)

    if "safety_familyos" in tasks and "safety_familyos" in capability_strs:
        results.safety_familyos = benchmark_safety_familyos(model, tokenizer, device)

    if "nli" in tasks and "nli" in capability_strs:
        results.nli = benchmark_nli(model, tokenizer, device)

    # Print summary
    print("\n" + results.summary())

    # Compare to baseline if requested
    if args.compare_baseline:
        baselines = compute_random_baselines()
        print(print_baseline_comparison(results, baselines))

    # Compare to raw ModernBERT-base if requested
    if args.compare_modernbert:
        raw_results = benchmark_raw_modernbert(tasks, device)
        print(print_modernbert_comparison(results, raw_results))

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = checkpoint_path / "benchmark_results.json"

    results.save(output_path)
    print(f"\n💾 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
