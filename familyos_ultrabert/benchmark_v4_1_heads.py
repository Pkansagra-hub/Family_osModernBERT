"""
FamilyOS UltraBERT v4.0.1 - Comprehensive Head Benchmarks
Tests all GlobalPointer heads + new LabelDescriptionHeads (Intent/Ingress)

Metrics:
- F1, Precision, Recall for NER heads
- Accuracy, Hit Rate for classification heads
- Latency (P50, P95, P99)
- Throughput (inferences/sec)
"""

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Test data for each head
NER_GENERAL_TEST_CASES = [
    {
        "text": "John Smith works at Microsoft in Seattle.",
        "expected": [
            {"text": "John Smith", "label": "PER"},
            {"text": "Microsoft", "label": "ORG"},
            {"text": "Seattle", "label": "LOC"},
        ],
    },
    {
        "text": "Apple announced new products at WWDC in San Francisco.",
        "expected": [
            {"text": "Apple", "label": "ORG"},
            {"text": "WWDC", "label": "MISC"},
            {"text": "San Francisco", "label": "LOC"},
        ],
    },
    {
        "text": "Dr. Sarah Johnson from Harvard Medical School presented research.",
        "expected": [
            {"text": "Dr. Sarah Johnson", "label": "PER"},
            {"text": "Harvard Medical School", "label": "ORG"},
        ],
    },
    {
        "text": "The United Nations headquarters is in New York City.",
        "expected": [
            {"text": "United Nations", "label": "ORG"},
            {"text": "New York City", "label": "LOC"},
        ],
    },
    {
        "text": "Elon Musk tweeted about Tesla stock prices.",
        "expected": [
            {"text": "Elon Musk", "label": "PER"},
            {"text": "Tesla", "label": "ORG"},
        ],
    },
]

NER_FAMILY_TEST_CASES = [
    {
        "text": "My grandmother Martha and grandfather Bob celebrated their 50th anniversary.",
        "expected": [
            {"text": "grandmother", "label": "KINSHIP"},
            {"text": "Martha", "label": "PERSON"},
            {"text": "grandfather", "label": "KINSHIP"},
            {"text": "Bob", "label": "PERSON"},
            {"text": "50th anniversary", "label": "MILESTONE"},
        ],
    },
    {
        "text": "Our dog Max loves playing with the kids at our house on Elm Street.",
        "expected": [
            {"text": "dog", "label": "KINSHIP"},
            {"text": "Max", "label": "PET"},
            {"text": "kids", "label": "KINSHIP"},
            {"text": "Elm Street", "label": "HOME_LOC"},
        ],
    },
    {
        "text": "Dad's old watch from 1965 is a precious heirloom from great-grandpa.",
        "expected": [
            {"text": "Dad", "label": "KINSHIP"},
            {"text": "old watch from 1965", "label": "HEIRLOOM"},
            {"text": "great-grandpa", "label": "KINSHIP"},
        ],
    },
    {
        "text": "Little Emma started kindergarten today - such a big milestone!",
        "expected": [
            {"text": "Emma", "label": "PERSON"},
            {"text": "started kindergarten", "label": "MILESTONE"},
        ],
    },
    {
        "text": "Every Sunday we have family dinner at grandma's - it's our tradition.",
        "expected": [
            {"text": "Sunday", "label": "ROUTINE"},
            {"text": "family dinner", "label": "FAMILY_EVENT"},
            {"text": "grandma", "label": "KINSHIP"},
            {"text": "tradition", "label": "TRADITION"},
        ],
    },
]

TEMPORAL_TEST_CASES = [
    {
        "text": "The meeting is scheduled for January 15th, 2026 at 3pm.",
        "expected": [
            {"text": "January 15th, 2026", "label": "DATE_ABS"},
            {"text": "3pm", "label": "TIME"},
        ],
    },
    {
        "text": "I'll call you back in 30 minutes, maybe around next Tuesday.",
        "expected": [
            {"text": "in 30 minutes", "label": "DURATION"},
            {"text": "next Tuesday", "label": "DATE_REL"},
        ],
    },
    {
        "text": "She exercises twice a week and has been doing it for 3 years.",
        "expected": [
            {"text": "twice a week", "label": "FREQUENCY"},
            {"text": "3 years", "label": "DURATION"},
        ],
    },
    {
        "text": "My son is 5 years old and starts school next September.",
        "expected": [
            {"text": "5 years old", "label": "AGE"},
            {"text": "next September", "label": "DATE_REL"},
        ],
    },
    {
        "text": "Yesterday was my birthday, I turned 30!",
        "expected": [
            {"text": "Yesterday", "label": "DATE_REL"},
            {"text": "30", "label": "AGE"},
        ],
    },
]

INTENT_TEST_CASES = [
    {"text": "Remember that dad's birthday is next week", "expected": "log_memory"},
    {"text": "When was mom's last doctor appointment?", "expected": "query_memory"},
    {"text": "Remind me to call grandma at 5pm", "expected": "set_reminder"},
    {"text": "I'm feeling so happy today!", "expected": "express_feeling"},
    {"text": "What should I do about the kids' screen time?", "expected": "seek_advice"},
    {"text": "Guess what! Sarah got promoted today!", "expected": "share_news"},
    {"text": "Looking back, last year was really challenging", "expected": "reflect"},
    {"text": "Hello, how are you?", "expected": "other"},
    {"text": "Save this: we moved to the new house on March 1st", "expected": "log_memory"},
    {"text": "What did we do for dad's 60th birthday?", "expected": "query_memory"},
]

INGRESS_TEST_CASES = [
    {"text": "Dear diary, today was an amazing day with the family", "expected": "DIARY"},
    {"text": "Need to pick up groceries and do laundry tomorrow", "expected": "TASK"},
    {"text": "My back has been hurting all week, need to see a doctor", "expected": "HEALTH"},
    {"text": "We need to budget better, spending too much on eating out", "expected": "FINANCE"},
    {"text": "Had a wonderful date night with my husband", "expected": "RELATIONSHIP"},
    {"text": "The project deadline at work is stressing me out", "expected": "WORK"},
    {"text": "How do I use this app?", "expected": "META"},
    {"text": "Remember when we went to Disneyland last summer?", "expected": "MEMORY"},
    {"text": "Planning the kids' birthday party for next month", "expected": "PLANNING"},
    {"text": "So proud of Emma for her graduation!", "expected": "CELEBRATION"},
    {"text": "Worried about mom's health lately", "expected": "CONCERN"},
    {"text": "So thankful for my supportive family", "expected": "GRATITUDE"},
]

SENTIMENT_TEST_CASES = [
    {"text": "This is the best day ever! I'm so happy!", "expected": "very_positive"},
    {"text": "Had a nice time with friends today.", "expected": "positive"},
    {"text": "The weather is okay I guess.", "expected": "neutral"},
    {"text": "Feeling a bit down today.", "expected": "negative"},
    {"text": "This is absolutely terrible. I hate everything.", "expected": "very_negative"},
    {"text": "Just had an amazing surprise party!", "expected": "very_positive"},
    {"text": "Work was fine, nothing special.", "expected": "neutral"},
    {"text": "Really disappointed with how things turned out.", "expected": "negative"},
]

SAFETY_TEST_CASES = [
    {"text": "Had a great family dinner tonight!", "expected": "GREEN"},
    {"text": "Feeling a bit stressed about work deadlines.", "expected": "AMBER"},
    {"text": "I've been feeling really isolated and hopeless lately.", "expected": "RED"},
    {"text": "I don't want to live anymore.", "expected": "CRISIS"},
    {"text": "The kids loved the birthday cake!", "expected": "GREEN"},
    {"text": "So frustrated with everything right now!", "expected": "AMBER"},
    {"text": "Nothing brings me joy anymore, everything feels empty.", "expected": "RED"},
    {"text": "I want to hurt myself.", "expected": "CRISIS"},
]


@dataclass
class LatencyStats:
    """Latency statistics."""

    times: List[float] = field(default_factory=list)

    def add(self, t: float) -> None:
        self.times.append(t)

    @property
    def p50(self) -> float:
        return np.percentile(self.times, 50) if self.times else 0.0

    @property
    def p95(self) -> float:
        return np.percentile(self.times, 95) if self.times else 0.0

    @property
    def p99(self) -> float:
        return np.percentile(self.times, 99) if self.times else 0.0

    @property
    def avg(self) -> float:
        return np.mean(self.times) if self.times else 0.0

    @property
    def min(self) -> float:
        return np.min(self.times) if self.times else 0.0

    @property
    def max(self) -> float:
        return np.max(self.times) if self.times else 0.0


def compute_ner_metrics(
    predictions: List[List[Dict[str, Any]]],
    expected: List[List[Dict[str, Any]]],
) -> Dict[str, float]:
    """Compute NER F1, Precision, Recall with partial matching."""
    tp = 0
    fp = 0
    fn = 0

    for pred_entities, exp_entities in zip(predictions, expected):
        pred_set = {(e["text"].lower(), e["label"]) for e in pred_entities}
        exp_set = {(e["text"].lower(), e["label"]) for e in exp_entities}

        # Exact matches
        exact_matches = pred_set & exp_set
        tp += len(exact_matches)

        # For remaining, check partial overlap
        pred_remaining = pred_set - exact_matches
        exp_remaining = exp_set - exact_matches

        for pred_text, pred_label in pred_remaining:
            found_partial = False
            for exp_text, exp_label in exp_remaining:
                if pred_label == exp_label and (pred_text in exp_text or exp_text in pred_text):
                    tp += 0.5  # Partial credit
                    found_partial = True
                    break
            if not found_partial:
                fp += 1

        for exp_text, exp_label in exp_remaining:
            found_partial = False
            for pred_text, pred_label in pred_remaining:
                if pred_label == exp_label and (pred_text in exp_text or exp_text in pred_text):
                    found_partial = True
                    break
            if not found_partial:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def compute_classification_metrics(
    predictions: List[str],
    expected: List[str],
) -> Dict[str, float]:
    """Compute classification accuracy and per-class metrics."""
    correct = sum(1 for p, e in zip(predictions, expected) if p == e)
    accuracy = correct / len(expected) if expected else 0.0

    # Per-class metrics
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for pred, exp in zip(predictions, expected):
        class_total[exp] += 1
        if pred == exp:
            class_correct[exp] += 1

    per_class_accuracy = {
        cls: class_correct[cls] / class_total[cls]
        for cls in class_total
    }

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(expected),
        "per_class_accuracy": per_class_accuracy,
    }


def run_benchmark() -> Dict[str, Any]:
    """Run comprehensive benchmark on all heads."""

    print("=" * 80)
    print("FamilyOS UltraBERT v4.0.1 - Comprehensive Head Benchmark")
    print("=" * 80)
    print()

    # Import and initialize
    print("Loading model...")
    start = time.time()

    from familyos_ultrabert.client import Client
    client = Client()

    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s")
    print()

    results = {
        "version": "4.0.1",
        "load_time_s": load_time,
        "heads": {},
    }

    # Warmup
    print("Warming up...")
    for _ in range(5):
        client.analyze("Warmup text for the model.")
    print()

    # ==========================================================================
    # 1. NER General Head (GlobalPointer)
    # ==========================================================================
    print("-" * 80)
    print("1. NER GENERAL HEAD (GlobalPointer)")
    print("-" * 80)

    latency = LatencyStats()
    predictions = []
    expected_all = []

    for case in NER_GENERAL_TEST_CASES:
        t0 = time.perf_counter()
        result = client.analyze(case["text"])
        latency.add((time.perf_counter() - t0) * 1000)

        # Extract NER entities - ClientResult has .general_entities for ner_general
        entities = result.general_entities if result.general_entities else []
        # Convert to dict format for comparison
        entities = [{"text": e.get("text", "").strip(), "label": e.get("label", "")} for e in entities]
        predictions.append(entities)
        expected_all.append(case["expected"])

    ner_general_metrics = compute_ner_metrics(predictions, expected_all)

    print(f"  Precision: {ner_general_metrics['precision']:.1%}")
    print(f"  Recall:    {ner_general_metrics['recall']:.1%}")
    print(f"  F1:        {ner_general_metrics['f1']:.1%}")
    print(f"  Latency P50: {latency.p50:.2f}ms | P95: {latency.p95:.2f}ms | P99: {latency.p99:.2f}ms")
    print()

    results["heads"]["ner_general"] = {
        "type": "GlobalPointer",
        "metrics": ner_general_metrics,
        "latency_ms": {"p50": latency.p50, "p95": latency.p95, "p99": latency.p99, "avg": latency.avg},
    }

    # ==========================================================================
    # 2. NER Family Head (GlobalPointer)
    # ==========================================================================
    print("-" * 80)
    print("2. NER FAMILY HEAD (GlobalPointer)")
    print("-" * 80)

    latency = LatencyStats()
    predictions = []
    expected_all = []

    for case in NER_FAMILY_TEST_CASES:
        t0 = time.perf_counter()
        result = client.analyze(case["text"])
        latency.add((time.perf_counter() - t0) * 1000)

        # Family entities are in result.entities (not general_entities)
        all_entities = result.entities if result.entities else []
        entities = [{"text": e.get("text", "").strip(), "label": e.get("label", "")}
                   for e in all_entities]

        predictions.append(entities)
        expected_all.append(case["expected"])

    ner_family_metrics = compute_ner_metrics(predictions, expected_all)

    print(f"  Precision: {ner_family_metrics['precision']:.1%}")
    print(f"  Recall:    {ner_family_metrics['recall']:.1%}")
    print(f"  F1:        {ner_family_metrics['f1']:.1%}")
    print(f"  Latency P50: {latency.p50:.2f}ms | P95: {latency.p95:.2f}ms | P99: {latency.p99:.2f}ms")
    print()

    results["heads"]["ner_family"] = {
        "type": "GlobalPointer",
        "metrics": ner_family_metrics,
        "latency_ms": {"p50": latency.p50, "p95": latency.p95, "p99": latency.p99, "avg": latency.avg},
    }

    # ==========================================================================
    # 3. Temporal Head (GlobalPointer)
    # ==========================================================================
    print("-" * 80)
    print("3. TEMPORAL HEAD (GlobalPointer)")
    print("-" * 80)

    latency = LatencyStats()
    predictions = []
    expected_all = []

    for case in TEMPORAL_TEST_CASES:
        t0 = time.perf_counter()
        result = client.analyze(case["text"])
        latency.add((time.perf_counter() - t0) * 1000)

        temporal_ents = result.temporal if result.temporal else []
        entities = [{"text": e.get("text", "").strip(), "label": e.get("label", "")} for e in temporal_ents]
        predictions.append(entities)
        expected_all.append(case["expected"])

    temporal_metrics = compute_ner_metrics(predictions, expected_all)

    print(f"  Precision: {temporal_metrics['precision']:.1%}")
    print(f"  Recall:    {temporal_metrics['recall']:.1%}")
    print(f"  F1:        {temporal_metrics['f1']:.1%}")
    print(f"  Latency P50: {latency.p50:.2f}ms | P95: {latency.p95:.2f}ms | P99: {latency.p99:.2f}ms")
    print()

    results["heads"]["temporal"] = {
        "type": "GlobalPointer",
        "metrics": temporal_metrics,
        "latency_ms": {"p50": latency.p50, "p95": latency.p95, "p99": latency.p99, "avg": latency.avg},
    }

    # ==========================================================================
    # 4. Intent Head (LabelDescriptionHead - NEW in v4.0.1)
    # ==========================================================================
    print("-" * 80)
    print("4. INTENT HEAD (LabelDescriptionHead - v4.0.1)")
    print("-" * 80)

    latency = LatencyStats()
    predictions = []
    expected = []

    for case in INTENT_TEST_CASES:
        t0 = time.perf_counter()
        result = client.analyze(case["text"])
        latency.add((time.perf_counter() - t0) * 1000)

        intent = result.intent if result.intent else "other"
        predictions.append(intent)
        expected.append(case["expected"])

    intent_metrics = compute_classification_metrics(predictions, expected)

    print(f"  Accuracy:  {intent_metrics['accuracy']:.1%}")
    print(f"  Correct:   {intent_metrics['correct']}/{intent_metrics['total']}")
    print(f"  Latency P50: {latency.p50:.2f}ms | P95: {latency.p95:.2f}ms | P99: {latency.p99:.2f}ms")
    print("  Per-class:")
    for cls, acc in intent_metrics["per_class_accuracy"].items():
        print(f"    {cls}: {acc:.1%}")
    print()

    results["heads"]["intent"] = {
        "type": "LabelDescriptionHead",
        "metrics": {k: v for k, v in intent_metrics.items() if k != "per_class_accuracy"},
        "per_class_accuracy": intent_metrics["per_class_accuracy"],
        "latency_ms": {"p50": latency.p50, "p95": latency.p95, "p99": latency.p99, "avg": latency.avg},
    }

    # ==========================================================================
    # 5. Ingress Head (LabelDescriptionHead - NEW in v4.0.1)
    # ==========================================================================
    print("-" * 80)
    print("5. INGRESS HEAD (LabelDescriptionHead - v4.0.1)")
    print("-" * 80)

    latency = LatencyStats()
    predictions = []
    expected = []

    for case in INGRESS_TEST_CASES:
        t0 = time.perf_counter()
        result = client.analyze(case["text"])
        latency.add((time.perf_counter() - t0) * 1000)

        ingress = result.ingress if result.ingress else "DIARY"
        predictions.append(ingress)
        expected.append(case["expected"])

    ingress_metrics = compute_classification_metrics(predictions, expected)

    print(f"  Accuracy:  {ingress_metrics['accuracy']:.1%}")
    print(f"  Correct:   {ingress_metrics['correct']}/{ingress_metrics['total']}")
    print(f"  Latency P50: {latency.p50:.2f}ms | P95: {latency.p95:.2f}ms | P99: {latency.p99:.2f}ms")
    print("  Per-class:")
    for cls, acc in ingress_metrics["per_class_accuracy"].items():
        print(f"    {cls}: {acc:.1%}")
    print()

    results["heads"]["ingress"] = {
        "type": "LabelDescriptionHead",
        "metrics": {k: v for k, v in ingress_metrics.items() if k != "per_class_accuracy"},
        "per_class_accuracy": ingress_metrics["per_class_accuracy"],
        "latency_ms": {"p50": latency.p50, "p95": latency.p95, "p99": latency.p99, "avg": latency.avg},
    }

    # ==========================================================================
    # 6. Sentiment Head
    # ==========================================================================
    print("-" * 80)
    print("6. SENTIMENT HEAD")
    print("-" * 80)

    latency = LatencyStats()
    predictions = []
    expected = []

    for case in SENTIMENT_TEST_CASES:
        t0 = time.perf_counter()
        result = client.analyze(case["text"])
        latency.add((time.perf_counter() - t0) * 1000)

        sentiment = result.sentiment if result.sentiment else "neutral"
        predictions.append(sentiment)
        expected.append(case["expected"])

    sentiment_metrics = compute_classification_metrics(predictions, expected)

    # Direction accuracy (positive vs negative)
    direction_correct = 0
    direction_total = 0
    pos_labels = {"very_positive", "positive"}
    neg_labels = {"very_negative", "negative"}

    for pred, exp in zip(predictions, expected):
        if exp in pos_labels or exp in neg_labels:
            direction_total += 1
            pred_pos = pred in pos_labels
            exp_pos = exp in pos_labels
            if pred_pos == exp_pos:
                direction_correct += 1

    direction_accuracy = direction_correct / direction_total if direction_total > 0 else 0.0

    print(f"  5-class Accuracy:   {sentiment_metrics['accuracy']:.1%}")
    print(f"  Direction Accuracy: {direction_accuracy:.1%}")
    print(f"  Latency P50: {latency.p50:.2f}ms | P95: {latency.p95:.2f}ms | P99: {latency.p99:.2f}ms")
    print()

    results["heads"]["sentiment"] = {
        "type": "ClassificationHead",
        "metrics": {
            "accuracy_5class": sentiment_metrics["accuracy"],
            "direction_accuracy": direction_accuracy,
        },
        "latency_ms": {"p50": latency.p50, "p95": latency.p95, "p99": latency.p99, "avg": latency.avg},
    }

    # ==========================================================================
    # 7. Safety Head
    # ==========================================================================
    print("-" * 80)
    print("7. SAFETY HEAD (FamilyOS 4-band)")
    print("-" * 80)

    latency = LatencyStats()
    predictions = []
    expected = []

    for case in SAFETY_TEST_CASES:
        t0 = time.perf_counter()
        result = client.analyze(case["text"])
        latency.add((time.perf_counter() - t0) * 1000)

        safety = result.safety if result.safety else "GREEN"
        predictions.append(safety)
        expected.append(case["expected"])

    safety_metrics = compute_classification_metrics(predictions, expected)

    # CRISIS recall (critical metric)
    crisis_expected = [e for e in expected if e == "CRISIS"]
    crisis_predicted = [predictions[i] for i, e in enumerate(expected) if e == "CRISIS"]
    crisis_recall = sum(1 for p in crisis_predicted if p == "CRISIS") / len(crisis_expected) if crisis_expected else 1.0

    print(f"  Band Accuracy: {safety_metrics['accuracy']:.1%}")
    print(f"  CRISIS Recall: {crisis_recall:.1%} {'[CRITICAL]' if crisis_recall < 1.0 else '[PASS]'}")
    print(f"  Latency P50: {latency.p50:.2f}ms | P95: {latency.p95:.2f}ms | P99: {latency.p99:.2f}ms")
    print()

    results["heads"]["safety_familyos"] = {
        "type": "HierarchicalClassificationHead",
        "metrics": {
            "band_accuracy": safety_metrics["accuracy"],
            "crisis_recall": crisis_recall,
        },
        "latency_ms": {"p50": latency.p50, "p95": latency.p95, "p99": latency.p99, "avg": latency.avg},
    }

    # ==========================================================================
    # Throughput Test
    # ==========================================================================
    print("-" * 80)
    print("THROUGHPUT TEST (100 inferences)")
    print("-" * 80)

    test_texts = [
        "Had a great day with the family!",
        "Feeling stressed about work.",
        "Remember to call mom tomorrow.",
        "The kids loved the birthday party!",
        "Need to finish the project by Friday.",
    ] * 20  # 100 texts

    t0 = time.perf_counter()
    for text in test_texts:
        client.analyze(text)
    total_time = time.perf_counter() - t0

    throughput = len(test_texts) / total_time
    avg_latency = (total_time / len(test_texts)) * 1000

    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Throughput: {throughput:.1f} inferences/sec")
    print(f"  Avg Latency: {avg_latency:.2f}ms")
    print()

    results["throughput"] = {
        "total_inferences": len(test_texts),
        "total_time_s": total_time,
        "throughput_per_sec": throughput,
        "avg_latency_ms": avg_latency,
    }

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("| Head | Type | Primary Metric | Score | Latency P95 |")
    print("|------|------|----------------|-------|-------------|")

    summary_rows = [
        ("ner_general", "GlobalPointer", "F1", f"{ner_general_metrics['f1']:.1%}", f"{results['heads']['ner_general']['latency_ms']['p95']:.1f}ms"),
        ("ner_family", "GlobalPointer", "F1", f"{ner_family_metrics['f1']:.1%}", f"{results['heads']['ner_family']['latency_ms']['p95']:.1f}ms"),
        ("temporal", "GlobalPointer", "F1", f"{temporal_metrics['f1']:.1%}", f"{results['heads']['temporal']['latency_ms']['p95']:.1f}ms"),
        ("intent", "LabelDescriptionHead", "Accuracy", f"{intent_metrics['accuracy']:.1%}", f"{results['heads']['intent']['latency_ms']['p95']:.1f}ms"),
        ("ingress", "LabelDescriptionHead", "Accuracy", f"{ingress_metrics['accuracy']:.1%}", f"{results['heads']['ingress']['latency_ms']['p95']:.1f}ms"),
        ("sentiment", "Classification", "Direction Acc", f"{direction_accuracy:.1%}", f"{results['heads']['sentiment']['latency_ms']['p95']:.1f}ms"),
        ("safety", "Hierarchical", "CRISIS Recall", f"{crisis_recall:.1%}", f"{results['heads']['safety_familyos']['latency_ms']['p95']:.1f}ms"),
    ]

    for row in summary_rows:
        print(f"| {row[0]} | {row[1]} | {row[2]} | **{row[3]}** | {row[4]} |")

    print()
    print(f"Throughput: **{throughput:.1f}** inferences/sec")
    print()

    # SLO/SLI Assessment
    print("=" * 80)
    print("SLO/SLI ASSESSMENT")
    print("=" * 80)
    print()

    slo_checks = [
        ("CRISIS Recall >= 99%", crisis_recall >= 0.99, f"{crisis_recall:.1%}"),
        ("Safety Band Accuracy >= 95%", safety_metrics["accuracy"] >= 0.95, f"{safety_metrics['accuracy']:.1%}"),
        ("Intent Accuracy >= 90%", intent_metrics["accuracy"] >= 0.90, f"{intent_metrics['accuracy']:.1%}"),
        ("Ingress Accuracy >= 90%", ingress_metrics["accuracy"] >= 0.90, f"{ingress_metrics['accuracy']:.1%}"),
        ("NER General F1 >= 70%", ner_general_metrics["f1"] >= 0.70, f"{ner_general_metrics['f1']:.1%}"),
        ("NER Family F1 >= 75%", ner_family_metrics["f1"] >= 0.75, f"{ner_family_metrics['f1']:.1%}"),
        ("Throughput >= 50/sec", throughput >= 50, f"{throughput:.1f}/sec"),
        ("P95 Latency <= 50ms", results["heads"]["intent"]["latency_ms"]["p95"] <= 50, f"{results['heads']['intent']['latency_ms']['p95']:.1f}ms"),
    ]

    print("| SLO | Target | Actual | Status |")
    print("|-----|--------|--------|--------|")

    all_passed = True
    for name, passed, actual in slo_checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"| {name} | - | {actual} | **{status}** |")

    print()
    if all_passed:
        print("All SLOs PASSED - Ready for production!")
    else:
        print("Some SLOs FAILED - Review required.")

    # Save results
    results["slo_assessment"] = {
        "all_passed": all_passed,
        "checks": [{"name": n, "passed": p, "actual": a} for n, p, a in slo_checks],
    }

    with open("benchmark_v4_1_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print()
    print(f"Results saved to benchmark_v4_1_results.json")

    return results


if __name__ == "__main__":
    run_benchmark()
