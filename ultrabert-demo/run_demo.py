"""
UltraBERT v4 Classifier Demo with Real Model Predictions.

This demo:
1. Loads the actual UltraBERT model
2. Warms up the model for accurate latency measurement
3. Runs 100+ real test cases
4. Generates confusion matrices for intent and ingress
5. Tracks SLO/SLI metrics (latency, accuracy)
6. Routes based on complexity classification
"""

import sys
import time
import json
import statistics
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from familyos_ultrabert.client import Client
from complexity_classifier import (
    classify_complexity,
    ComplexityTier,
    ComplexityResult,
    get_routing_recommendation,
)


# =============================================================================
# Test Case Data - 100+ Real Examples
# =============================================================================

TEST_CASES = [
    # =========================================================================
    # LOW COMPLEXITY - Tool Execution (log_memory, set_reminder, query_memory)
    # =========================================================================

    # log_memory + DIARY/MEMORY/TASK (25 examples)
    {"text": "Remember to buy milk tomorrow", "expected_intent": "log_memory", "expected_ingress": "TASK", "expected_complexity": "LOW"},
    {"text": "Save this: meeting at 3pm", "expected_intent": "log_memory", "expected_ingress": "TASK", "expected_complexity": "LOW"},
    {"text": "Note: dentist appointment next week", "expected_intent": "log_memory", "expected_ingress": "TASK", "expected_complexity": "LOW"},
    {"text": "Remember that Mom's birthday is March 15", "expected_intent": "log_memory", "expected_ingress": "MEMORY", "expected_complexity": "LOW"},
    {"text": "Save this recipe for later", "expected_intent": "log_memory", "expected_ingress": "MEMORY", "expected_complexity": "LOW"},
    {"text": "Note down: kids school pickup at 3:30", "expected_intent": "log_memory", "expected_ingress": "TASK", "expected_complexity": "LOW"},
    {"text": "Remember to call the plumber", "expected_intent": "log_memory", "expected_ingress": "TASK", "expected_complexity": "LOW"},
    {"text": "Save: grocery list - eggs, bread, cheese", "expected_intent": "log_memory", "expected_ingress": "TASK", "expected_complexity": "LOW"},
    {"text": "Note that dad prefers decaf coffee", "expected_intent": "log_memory", "expected_ingress": "MEMORY", "expected_complexity": "LOW"},
    {"text": "Remember the wifi password is sunshine123", "expected_intent": "log_memory", "expected_ingress": "MEMORY", "expected_complexity": "LOW"},
    {"text": "Today I went to the park with the kids", "expected_intent": "log_memory", "expected_ingress": "DIARY", "expected_complexity": "LOW"},
    {"text": "Had lunch with Sarah at the cafe", "expected_intent": "log_memory", "expected_ingress": "DIARY", "expected_complexity": "LOW"},
    {"text": "Finished reading that book finally", "expected_intent": "log_memory", "expected_ingress": "DIARY", "expected_complexity": "LOW"},

    # set_reminder + TASK/PLANNING (15 examples)
    {"text": "Remind me to call John at 2pm", "expected_intent": "set_reminder", "expected_ingress": "TASK", "expected_complexity": "LOW"},
    {"text": "Set a reminder for tomorrow morning", "expected_intent": "set_reminder", "expected_ingress": "TASK", "expected_complexity": "LOW"},
    {"text": "Remind me about the meeting in 30 minutes", "expected_intent": "set_reminder", "expected_ingress": "TASK", "expected_complexity": "LOW"},
    {"text": "Set reminder: pick up dry cleaning Friday", "expected_intent": "set_reminder", "expected_ingress": "TASK", "expected_complexity": "LOW"},
    {"text": "Don't let me forget to water the plants", "expected_intent": "set_reminder", "expected_ingress": "TASK", "expected_complexity": "LOW"},
    {"text": "Remind me to take my medication at 8pm", "expected_intent": "set_reminder", "expected_ingress": "HEALTH", "expected_complexity": "LOW"},
    {"text": "Set alarm for 6am workout", "expected_intent": "set_reminder", "expected_ingress": "TASK", "expected_complexity": "LOW"},
    {"text": "Remind me to submit the report by EOD", "expected_intent": "set_reminder", "expected_ingress": "WORK", "expected_complexity": "LOW"},
    {"text": "Alert me when it's time to leave for airport", "expected_intent": "set_reminder", "expected_ingress": "PLANNING", "expected_complexity": "LOW"},
    {"text": "Reminder: anniversary dinner reservation", "expected_intent": "set_reminder", "expected_ingress": "PLANNING", "expected_complexity": "LOW"},

    # query_memory + MEMORY/DIARY (10 examples)
    {"text": "What's Mom's favorite restaurant?", "expected_intent": "query_memory", "expected_ingress": "MEMORY", "expected_complexity": "LOW"},
    {"text": "When is Dad's doctor appointment?", "expected_intent": "query_memory", "expected_ingress": "MEMORY", "expected_complexity": "LOW"},
    {"text": "What did I do last Tuesday?", "expected_intent": "query_memory", "expected_ingress": "DIARY", "expected_complexity": "LOW"},
    {"text": "Where did we go for vacation last year?", "expected_intent": "query_memory", "expected_ingress": "MEMORY", "expected_complexity": "LOW"},
    {"text": "What's the name of Sarah's husband?", "expected_intent": "query_memory", "expected_ingress": "MEMORY", "expected_complexity": "LOW"},

    # =========================================================================
    # MEDIUM COMPLEXITY - Small LLM (express_feeling, share_news)
    # =========================================================================

    # express_feeling + CELEBRATION/GRATITUDE/DIARY (20 examples)
    {"text": "I'm so happy today!", "expected_intent": "express_feeling", "expected_ingress": "CELEBRATION", "expected_complexity": "MEDIUM"},
    {"text": "Feeling grateful for my family", "expected_intent": "express_feeling", "expected_ingress": "GRATITUDE", "expected_complexity": "MEDIUM"},
    {"text": "I love spending time with my kids", "expected_intent": "express_feeling", "expected_ingress": "CELEBRATION", "expected_complexity": "MEDIUM"},
    {"text": "Today was a good day", "expected_intent": "express_feeling", "expected_ingress": "DIARY", "expected_complexity": "MEDIUM"},
    {"text": "I'm excited about the weekend", "expected_intent": "express_feeling", "expected_ingress": "CELEBRATION", "expected_complexity": "MEDIUM"},
    {"text": "Feeling blessed to have such great friends", "expected_intent": "express_feeling", "expected_ingress": "GRATITUDE", "expected_complexity": "MEDIUM"},
    {"text": "I'm proud of my daughter's achievement", "expected_intent": "express_feeling", "expected_ingress": "CELEBRATION", "expected_complexity": "MEDIUM"},
    {"text": "Missing my parents today", "expected_intent": "express_feeling", "expected_ingress": "RELATIONSHIP", "expected_complexity": "MEDIUM"},
    {"text": "Feeling a bit tired but content", "expected_intent": "express_feeling", "expected_ingress": "DIARY", "expected_complexity": "MEDIUM"},
    {"text": "I appreciate all the support from my spouse", "expected_intent": "express_feeling", "expected_ingress": "GRATITUDE", "expected_complexity": "MEDIUM"},
    {"text": "Feeling nostalgic about our old house", "expected_intent": "express_feeling", "expected_ingress": "MEMORY", "expected_complexity": "MEDIUM"},
    {"text": "I'm really enjoying this quiet evening", "expected_intent": "express_feeling", "expected_ingress": "DIARY", "expected_complexity": "MEDIUM"},
    {"text": "So thankful for my health", "expected_intent": "express_feeling", "expected_ingress": "GRATITUDE", "expected_complexity": "MEDIUM"},

    # share_news + CELEBRATION/RELATIONSHIP (15 examples)
    {"text": "My daughter got an A on her test!", "expected_intent": "share_news", "expected_ingress": "CELEBRATION", "expected_complexity": "MEDIUM"},
    {"text": "We're having a baby!", "expected_intent": "share_news", "expected_ingress": "CELEBRATION", "expected_complexity": "MEDIUM"},
    {"text": "I got promoted at work today", "expected_intent": "share_news", "expected_ingress": "CELEBRATION", "expected_complexity": "MEDIUM"},
    {"text": "My son learned to ride a bike", "expected_intent": "share_news", "expected_ingress": "CELEBRATION", "expected_complexity": "MEDIUM"},
    {"text": "Mom is coming to visit next month", "expected_intent": "share_news", "expected_ingress": "RELATIONSHIP", "expected_complexity": "MEDIUM"},
    {"text": "We adopted a new puppy!", "expected_intent": "share_news", "expected_ingress": "CELEBRATION", "expected_complexity": "MEDIUM"},
    {"text": "The kids finished their school year", "expected_intent": "share_news", "expected_ingress": "CELEBRATION", "expected_complexity": "MEDIUM"},
    {"text": "My brother is getting married", "expected_intent": "share_news", "expected_ingress": "CELEBRATION", "expected_complexity": "MEDIUM"},
    {"text": "We bought our first house!", "expected_intent": "share_news", "expected_ingress": "CELEBRATION", "expected_complexity": "MEDIUM"},
    {"text": "Grandma recovered from her surgery", "expected_intent": "share_news", "expected_ingress": "HEALTH", "expected_complexity": "MEDIUM"},

    # =========================================================================
    # HIGH COMPLEXITY - Full LLM (seek_advice, reflect, express_feeling+CONCERN)
    # =========================================================================

    # seek_advice + HEALTH/FINANCE/WORK/RELATIONSHIP (25 examples)
    {"text": "Should I invest in stocks or bonds?", "expected_intent": "seek_advice", "expected_ingress": "FINANCE", "expected_complexity": "HIGH"},
    {"text": "How do I deal with my difficult boss?", "expected_intent": "seek_advice", "expected_ingress": "WORK", "expected_complexity": "HIGH"},
    {"text": "What should I do about my back pain?", "expected_intent": "seek_advice", "expected_ingress": "HEALTH", "expected_complexity": "HIGH"},
    {"text": "How can I improve my marriage?", "expected_intent": "seek_advice", "expected_ingress": "RELATIONSHIP", "expected_complexity": "HIGH"},
    {"text": "Should I change careers at 45?", "expected_intent": "seek_advice", "expected_ingress": "WORK", "expected_complexity": "HIGH"},
    {"text": "How do I save for retirement?", "expected_intent": "seek_advice", "expected_ingress": "FINANCE", "expected_complexity": "HIGH"},
    {"text": "What's the best diet for diabetes?", "expected_intent": "seek_advice", "expected_ingress": "HEALTH", "expected_complexity": "HIGH"},
    {"text": "How should I handle my teenager's rebellion?", "expected_intent": "seek_advice", "expected_ingress": "RELATIONSHIP", "expected_complexity": "HIGH"},
    {"text": "Should I quit my job and start a business?", "expected_intent": "seek_advice", "expected_ingress": "WORK", "expected_complexity": "HIGH"},
    {"text": "How do I talk to my kids about divorce?", "expected_intent": "seek_advice", "expected_ingress": "RELATIONSHIP", "expected_complexity": "HIGH"},
    {"text": "What insurance should I get for my family?", "expected_intent": "seek_advice", "expected_ingress": "FINANCE", "expected_complexity": "HIGH"},
    {"text": "How can I manage my anxiety better?", "expected_intent": "seek_advice", "expected_ingress": "HEALTH", "expected_complexity": "HIGH"},
    {"text": "Should I move closer to my aging parents?", "expected_intent": "seek_advice", "expected_ingress": "RELATIONSHIP", "expected_complexity": "HIGH"},
    {"text": "How do I balance work and family life?", "expected_intent": "seek_advice", "expected_ingress": "WORK", "expected_complexity": "HIGH"},
    {"text": "What's the best school for my child?", "expected_intent": "seek_advice", "expected_ingress": "RELATIONSHIP", "expected_complexity": "HIGH"},

    # reflect + META/RELATIONSHIP/CONCERN (15 examples)
    {"text": "I've been thinking about the meaning of life", "expected_intent": "reflect", "expected_ingress": "META", "expected_complexity": "HIGH"},
    {"text": "Why do I always feel like I'm not good enough?", "expected_intent": "reflect", "expected_ingress": "CONCERN", "expected_complexity": "HIGH"},
    {"text": "What does it mean to be a good parent?", "expected_intent": "reflect", "expected_ingress": "META", "expected_complexity": "HIGH"},
    {"text": "I wonder if I made the right choices in life", "expected_intent": "reflect", "expected_ingress": "META", "expected_complexity": "HIGH"},
    {"text": "Why is it so hard to let go of the past?", "expected_intent": "reflect", "expected_ingress": "META", "expected_complexity": "HIGH"},
    {"text": "What kind of legacy do I want to leave?", "expected_intent": "reflect", "expected_ingress": "META", "expected_complexity": "HIGH"},
    {"text": "Why do relationships require so much work?", "expected_intent": "reflect", "expected_ingress": "RELATIONSHIP", "expected_complexity": "HIGH"},
    {"text": "I keep questioning my career path", "expected_intent": "reflect", "expected_ingress": "WORK", "expected_complexity": "HIGH"},
    {"text": "What makes a family truly happy?", "expected_intent": "reflect", "expected_ingress": "META", "expected_complexity": "HIGH"},
    {"text": "Why am I always worried about the future?", "expected_intent": "reflect", "expected_ingress": "CONCERN", "expected_complexity": "HIGH"},

    # express_feeling + CONCERN/HEALTH (HIGH complexity due to concern) (10 examples)
    {"text": "I'm really worried about my health lately", "expected_intent": "express_feeling", "expected_ingress": "HEALTH", "expected_complexity": "HIGH"},
    {"text": "I feel so overwhelmed with everything", "expected_intent": "express_feeling", "expected_ingress": "CONCERN", "expected_complexity": "HIGH"},
    {"text": "I'm scared about my upcoming surgery", "expected_intent": "express_feeling", "expected_ingress": "HEALTH", "expected_complexity": "HIGH"},
    {"text": "I feel lost and don't know what to do", "expected_intent": "express_feeling", "expected_ingress": "CONCERN", "expected_complexity": "HIGH"},
    {"text": "I'm anxious about my relationship", "expected_intent": "express_feeling", "expected_ingress": "RELATIONSHIP", "expected_complexity": "HIGH"},
    {"text": "I feel like I'm failing as a parent", "expected_intent": "express_feeling", "expected_ingress": "CONCERN", "expected_complexity": "HIGH"},
    {"text": "I'm stressed about money all the time", "expected_intent": "express_feeling", "expected_ingress": "FINANCE", "expected_complexity": "HIGH"},
    {"text": "I feel disconnected from my family", "expected_intent": "express_feeling", "expected_ingress": "RELATIONSHIP", "expected_complexity": "HIGH"},
    {"text": "I'm exhausted and can't seem to recover", "expected_intent": "express_feeling", "expected_ingress": "HEALTH", "expected_complexity": "HIGH"},
    {"text": "I feel trapped in my current situation", "expected_intent": "express_feeling", "expected_ingress": "CONCERN", "expected_complexity": "HIGH"},

    # =========================================================================
    # CRISIS - Safety Override (must detect and route immediately)
    # =========================================================================
    {"text": "I want to hurt myself", "expected_intent": "express_feeling", "expected_ingress": "CONCERN", "expected_complexity": "CRISIS"},
    {"text": "I don't want to live anymore", "expected_intent": "express_feeling", "expected_ingress": "CONCERN", "expected_complexity": "CRISIS"},
    {"text": "I'm thinking about ending it all", "expected_intent": "express_feeling", "expected_ingress": "CONCERN", "expected_complexity": "CRISIS"},
    {"text": "I feel like giving up on life", "expected_intent": "express_feeling", "expected_ingress": "CONCERN", "expected_complexity": "CRISIS"},
    {"text": "Nobody would miss me if I was gone", "expected_intent": "express_feeling", "expected_ingress": "CONCERN", "expected_complexity": "CRISIS"},

    # =========================================================================
    # Edge Cases / Ambiguous (test model discrimination)
    # =========================================================================
    {"text": "Remember to be happy", "expected_intent": "log_memory", "expected_ingress": "DIARY", "expected_complexity": "LOW"},
    {"text": "I need to think about things", "expected_intent": "reflect", "expected_ingress": "META", "expected_complexity": "HIGH"},
    {"text": "Just wanted to say thanks", "expected_intent": "express_feeling", "expected_ingress": "GRATITUDE", "expected_complexity": "MEDIUM"},
    {"text": "Life is complicated", "expected_intent": "reflect", "expected_ingress": "META", "expected_complexity": "HIGH"},
    {"text": "Quick note about the meeting", "expected_intent": "log_memory", "expected_ingress": "TASK", "expected_complexity": "LOW"},
    {"text": "I have mixed feelings about this", "expected_intent": "express_feeling", "expected_ingress": "DIARY", "expected_complexity": "MEDIUM"},
    {"text": "Help me understand my options", "expected_intent": "seek_advice", "expected_ingress": "META", "expected_complexity": "HIGH"},
    {"text": "That's interesting", "expected_intent": "other", "expected_ingress": "DIARY", "expected_complexity": "MEDIUM"},
]


# =============================================================================
# SLO/SLI Metrics Tracker
# =============================================================================

@dataclass
class SLOMetrics:
    """Service Level Objective metrics."""

    # Latency SLOs
    latency_p50_target_ms: float = 50.0
    latency_p95_target_ms: float = 100.0
    latency_p99_target_ms: float = 200.0

    # Accuracy SLOs
    intent_accuracy_target: float = 0.85
    ingress_accuracy_target: float = 0.80
    complexity_accuracy_target: float = 0.75
    crisis_recall_target: float = 1.0  # Must be 100%

    # Tracked metrics
    latencies: List[float] = field(default_factory=list)
    intent_correct: int = 0
    intent_total: int = 0
    ingress_correct: int = 0
    ingress_total: int = 0
    complexity_correct: int = 0
    complexity_total: int = 0
    crisis_detected: int = 0
    crisis_total: int = 0

    def record_latency(self, latency_ms: float):
        self.latencies.append(latency_ms)

    def record_intent(self, predicted: str, expected: str):
        self.intent_total += 1
        if predicted == expected:
            self.intent_correct += 1

    def record_ingress(self, predicted: str, expected: str):
        self.ingress_total += 1
        if predicted == expected:
            self.ingress_correct += 1

    def record_complexity(self, predicted: str, expected: str):
        self.complexity_total += 1
        if predicted == expected:
            self.complexity_correct += 1

    def record_crisis(self, detected: bool, is_crisis: bool):
        if is_crisis:
            self.crisis_total += 1
            if detected:
                self.crisis_detected += 1

    def get_latency_percentile(self, p: float) -> float:
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * p)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    def get_report(self) -> Dict:
        p50 = self.get_latency_percentile(0.50)
        p95 = self.get_latency_percentile(0.95)
        p99 = self.get_latency_percentile(0.99)

        intent_acc = self.intent_correct / self.intent_total if self.intent_total else 0
        ingress_acc = self.ingress_correct / self.ingress_total if self.ingress_total else 0
        complexity_acc = self.complexity_correct / self.complexity_total if self.complexity_total else 0
        crisis_recall = self.crisis_detected / self.crisis_total if self.crisis_total else 1.0

        return {
            "latency": {
                "p50_ms": round(p50, 2),
                "p95_ms": round(p95, 2),
                "p99_ms": round(p99, 2),
                "mean_ms": round(statistics.mean(self.latencies), 2) if self.latencies else 0,
                "min_ms": round(min(self.latencies), 2) if self.latencies else 0,
                "max_ms": round(max(self.latencies), 2) if self.latencies else 0,
                "p50_slo_met": p50 <= self.latency_p50_target_ms,
                "p95_slo_met": p95 <= self.latency_p95_target_ms,
                "p99_slo_met": p99 <= self.latency_p99_target_ms,
            },
            "accuracy": {
                "intent": round(intent_acc * 100, 1),
                "intent_slo_met": intent_acc >= self.intent_accuracy_target,
                "ingress": round(ingress_acc * 100, 1),
                "ingress_slo_met": ingress_acc >= self.ingress_accuracy_target,
                "complexity": round(complexity_acc * 100, 1),
                "complexity_slo_met": complexity_acc >= self.complexity_accuracy_target,
                "crisis_recall": round(crisis_recall * 100, 1),
                "crisis_slo_met": crisis_recall >= self.crisis_recall_target,
            },
            "counts": {
                "total_tests": self.intent_total,
                "intent_correct": self.intent_correct,
                "ingress_correct": self.ingress_correct,
                "complexity_correct": self.complexity_correct,
                "crisis_cases": self.crisis_total,
                "crisis_detected": self.crisis_detected,
            }
        }


# =============================================================================
# Confusion Matrix Builder
# =============================================================================

class ConfusionMatrix:
    """Builds confusion matrix for classification."""

    def __init__(self, labels: List[str]):
        self.labels = labels
        self.label_to_idx = {l: i for i, l in enumerate(labels)}
        self.matrix = [[0] * len(labels) for _ in range(len(labels))]

    def add(self, true_label: str, pred_label: str):
        if true_label in self.label_to_idx and pred_label in self.label_to_idx:
            true_idx = self.label_to_idx[true_label]
            pred_idx = self.label_to_idx[pred_label]
            self.matrix[true_idx][pred_idx] += 1

    def get_accuracy(self) -> float:
        correct = sum(self.matrix[i][i] for i in range(len(self.labels)))
        total = sum(sum(row) for row in self.matrix)
        return correct / total if total > 0 else 0.0

    def get_per_class_metrics(self) -> Dict[str, Dict]:
        metrics = {}
        for i, label in enumerate(self.labels):
            tp = self.matrix[i][i]
            fp = sum(self.matrix[j][i] for j in range(len(self.labels))) - tp
            fn = sum(self.matrix[i]) - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics[label] = {
                "precision": round(precision * 100, 1),
                "recall": round(recall * 100, 1),
                "f1": round(f1 * 100, 1),
                "support": sum(self.matrix[i]),
            }
        return metrics

    def print_matrix(self, title: str = "Confusion Matrix"):
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")

        # Header
        max_label_len = max(len(l) for l in self.labels)
        header = " " * (max_label_len + 2) + "  ".join(f"{l[:8]:>8}" for l in self.labels)
        print(f"  Predicted ->")
        print(f"  {header}")
        print(f"  {'-' * len(header)}")

        # Rows
        for i, label in enumerate(self.labels):
            row_str = "  ".join(f"{v:>8}" for v in self.matrix[i])
            print(f"  {label:<{max_label_len}} | {row_str}")

        print(f"\n  Overall Accuracy: {self.get_accuracy()*100:.1f}%")

        # Per-class metrics
        metrics = self.get_per_class_metrics()
        print(f"\n  Per-Class Metrics:")
        print(f"  {'Label':<15} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
        print(f"  {'-'*47}")
        for label, m in metrics.items():
            print(f"  {label:<15} {m['precision']:>7.1f}% {m['recall']:>7.1f}% {m['f1']:>7.1f}% {m['support']:>8}")


# =============================================================================
# Main Demo Runner
# =============================================================================

def warmup_model(client: Client, rounds: int = 5) -> List[float]:
    """Warm up the model and return warmup latencies."""
    print(f"\nWarming up model ({rounds} rounds)...")
    warmup_texts = [
        "Remember to buy groceries",
        "I'm feeling happy today",
        "How should I handle this situation?",
        "My daughter got an A!",
        "Set a reminder for 3pm",
    ]

    latencies = []
    for i in range(rounds):
        text = warmup_texts[i % len(warmup_texts)]
        start = time.perf_counter()
        _ = client.analyze(text)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)
        print(f"  Warmup {i+1}: {elapsed:.1f}ms")

    print(f"  Average warmup latency: {statistics.mean(latencies):.1f}ms")
    return latencies


def run_single_test(client: Client, test_case: Dict, verbose: bool = False) -> Dict:
    """Run a single test case and return results."""
    text = test_case["text"]
    expected_intent = test_case["expected_intent"]
    expected_ingress = test_case["expected_ingress"]
    expected_complexity = test_case["expected_complexity"]

    # Run inference
    start = time.perf_counter()
    result = client.analyze(text)
    latency_ms = (time.perf_counter() - start) * 1000

    # Extract predictions
    pred_intent = result.intent
    pred_ingress = result.ingress
    pred_safety = result.safety
    pred_emotions = result.emotions
    intent_conf = result.intent_confidence

    # Get ingress confidence (approximate from result)
    ingress_conf = 0.8  # Default, actual would come from model

    # Classify complexity
    complexity_result = classify_complexity(
        text=text,
        intent=pred_intent,
        ingress=pred_ingress,
        safety=pred_safety,
        emotions=pred_emotions,
        intent_confidence=intent_conf,
        ingress_confidence=ingress_conf,
    )

    # Check if crisis was expected and detected
    is_crisis = expected_complexity == "CRISIS"
    crisis_detected = pred_safety == "CRISIS" or complexity_result.tier == ComplexityTier.CRISIS

    # Build result
    test_result = {
        "text": text,
        "expected": {
            "intent": expected_intent,
            "ingress": expected_ingress,
            "complexity": expected_complexity,
        },
        "predicted": {
            "intent": pred_intent,
            "ingress": pred_ingress,
            "safety": pred_safety,
            "emotions": pred_emotions,
            "complexity": complexity_result.tier.value,
        },
        "correct": {
            "intent": pred_intent == expected_intent,
            "ingress": pred_ingress == expected_ingress,
            "complexity": complexity_result.tier.value == expected_complexity,
        },
        "confidence": {
            "intent": intent_conf,
        },
        "complexity_reason": complexity_result.reason,
        "latency_ms": latency_ms,
        "is_crisis": is_crisis,
        "crisis_detected": crisis_detected,
    }

    if verbose:
        print_test_result(test_result)

    return test_result


def print_test_result(result: Dict):
    """Print formatted test result."""
    print(f"\n{'─'*70}")
    print(f"Test: \"{result['text'][:60]}{'...' if len(result['text']) > 60 else ''}\"")
    print()

    # Predictions
    print(f"UltraBERT Predictions:")
    intent_mark = "correct" if result['correct']['intent'] else "WRONG"
    ingress_mark = "correct" if result['correct']['ingress'] else "WRONG"
    print(f"  Intent:  {result['predicted']['intent']} ({result['confidence']['intent']*100:.0f}%) [{intent_mark}]")
    print(f"  Ingress: {result['predicted']['ingress']} [{ingress_mark}]")
    print(f"  Safety:  {result['predicted']['safety']}")
    print(f"  Emotions: {result['predicted']['emotions'][:3]}...")

    # Complexity
    print()
    print(f"Complexity Classification:")
    comp_mark = "correct" if result['correct']['complexity'] else "WRONG"
    print(f"  Decision: {result['predicted']['complexity']} [{comp_mark}]")
    print(f"  Expected: {result['expected']['complexity']}")
    print(f"  Reason:   {result['complexity_reason'][:60]}...")

    # Latency
    print(f"\n  Latency: {result['latency_ms']:.1f}ms")


def print_slo_report(metrics: SLOMetrics):
    """Print SLO/SLI report."""
    report = metrics.get_report()

    print(f"\n{'='*70}")
    print(f"  SLO/SLI METRICS REPORT")
    print(f"{'='*70}")

    # Latency SLOs
    lat = report["latency"]
    print(f"\n  LATENCY SLOs:")
    print(f"  {'Metric':<20} {'Value':>10} {'Target':>10} {'Status':>10}")
    print(f"  {'-'*50}")
    print(f"  {'P50 Latency':<20} {lat['p50_ms']:>9.1f}ms {metrics.latency_p50_target_ms:>9.1f}ms {'PASS' if lat['p50_slo_met'] else 'FAIL':>10}")
    print(f"  {'P95 Latency':<20} {lat['p95_ms']:>9.1f}ms {metrics.latency_p95_target_ms:>9.1f}ms {'PASS' if lat['p95_slo_met'] else 'FAIL':>10}")
    print(f"  {'P99 Latency':<20} {lat['p99_ms']:>9.1f}ms {metrics.latency_p99_target_ms:>9.1f}ms {'PASS' if lat['p99_slo_met'] else 'FAIL':>10}")
    print(f"  {'Mean Latency':<20} {lat['mean_ms']:>9.1f}ms")
    print(f"  {'Min Latency':<20} {lat['min_ms']:>9.1f}ms")
    print(f"  {'Max Latency':<20} {lat['max_ms']:>9.1f}ms")

    # Accuracy SLOs
    acc = report["accuracy"]
    print(f"\n  ACCURACY SLOs:")
    print(f"  {'Metric':<20} {'Value':>10} {'Target':>10} {'Status':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Intent Accuracy':<20} {acc['intent']:>9.1f}% {metrics.intent_accuracy_target*100:>9.1f}% {'PASS' if acc['intent_slo_met'] else 'FAIL':>10}")
    print(f"  {'Ingress Accuracy':<20} {acc['ingress']:>9.1f}% {metrics.ingress_accuracy_target*100:>9.1f}% {'PASS' if acc['ingress_slo_met'] else 'FAIL':>10}")
    print(f"  {'Complexity Accuracy':<20} {acc['complexity']:>9.1f}% {metrics.complexity_accuracy_target*100:>9.1f}% {'PASS' if acc['complexity_slo_met'] else 'FAIL':>10}")
    print(f"  {'Crisis Recall':<20} {acc['crisis_recall']:>9.1f}% {metrics.crisis_recall_target*100:>9.1f}% {'PASS' if acc['crisis_slo_met'] else 'FAIL':>10}")

    # Counts
    counts = report["counts"]
    print(f"\n  COUNTS:")
    print(f"  Total Tests:        {counts['total_tests']}")
    print(f"  Intent Correct:     {counts['intent_correct']}/{counts['total_tests']}")
    print(f"  Ingress Correct:    {counts['ingress_correct']}/{counts['total_tests']}")
    print(f"  Complexity Correct: {counts['complexity_correct']}/{counts['total_tests']}")
    print(f"  Crisis Cases:       {counts['crisis_detected']}/{counts['crisis_cases']} detected")


def main():
    """Run the full demo."""
    print("="*70)
    print("  UltraBERT v4 Classifier Demo")
    print("  Real Model Predictions + Complexity Routing")
    print("="*70)

    # Initialize client
    print("\nLoading UltraBERT model...")
    client = Client(verbose=False, warmup=False)  # We'll do our own warmup
    print(f"  Backend: {client.backend}")
    print(f"  Capabilities: {len(client.capabilities)}")

    # Warmup
    warmup_latencies = warmup_model(client, rounds=5)

    # Initialize metrics
    metrics = SLOMetrics()

    # Intent labels
    intent_labels = ["log_memory", "query_memory", "set_reminder", "express_feeling",
                     "seek_advice", "share_news", "reflect", "other"]
    intent_cm = ConfusionMatrix(intent_labels)

    # Ingress labels
    ingress_labels = ["DIARY", "TASK", "HEALTH", "FINANCE", "RELATIONSHIP", "WORK",
                      "META", "MEMORY", "PLANNING", "CELEBRATION", "CONCERN", "GRATITUDE"]
    ingress_cm = ConfusionMatrix(ingress_labels)

    # Complexity labels
    complexity_labels = ["LOW", "MEDIUM", "HIGH", "CRISIS"]
    complexity_cm = ConfusionMatrix(complexity_labels)

    # Run all tests
    print(f"\nRunning {len(TEST_CASES)} test cases...")
    print("-"*70)

    results = []
    errors_intent = []
    errors_ingress = []
    errors_complexity = []

    for i, test_case in enumerate(TEST_CASES):
        # Show progress every 20 tests
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(TEST_CASES)} tests completed...")

        try:
            result = run_single_test(client, test_case, verbose=False)
            results.append(result)

            # Record metrics
            metrics.record_latency(result["latency_ms"])
            metrics.record_intent(result["predicted"]["intent"], result["expected"]["intent"])
            metrics.record_ingress(result["predicted"]["ingress"], result["expected"]["ingress"])
            metrics.record_complexity(result["predicted"]["complexity"], result["expected"]["complexity"])
            metrics.record_crisis(result["crisis_detected"], result["is_crisis"])

            # Update confusion matrices
            intent_cm.add(result["expected"]["intent"], result["predicted"]["intent"])
            ingress_cm.add(result["expected"]["ingress"], result["predicted"]["ingress"])
            complexity_cm.add(result["expected"]["complexity"], result["predicted"]["complexity"])

            # Track errors for analysis
            if not result["correct"]["intent"]:
                errors_intent.append(result)
            if not result["correct"]["ingress"]:
                errors_ingress.append(result)
            if not result["correct"]["complexity"]:
                errors_complexity.append(result)

        except Exception as e:
            print(f"  ERROR on test {i+1}: {e}")

    print(f"\nCompleted {len(results)} tests successfully")

    # Print confusion matrices
    intent_cm.print_matrix("INTENT Confusion Matrix")
    ingress_cm.print_matrix("INGRESS Confusion Matrix")
    complexity_cm.print_matrix("COMPLEXITY Confusion Matrix")

    # Print SLO report
    print_slo_report(metrics)

    # Print sample errors
    print(f"\n{'='*70}")
    print(f"  ERROR ANALYSIS")
    print(f"{'='*70}")

    print(f"\n  Intent Errors: {len(errors_intent)}/{len(results)}")
    for err in errors_intent[:5]:
        print(f"    - \"{err['text'][:40]}...\"")
        print(f"      Expected: {err['expected']['intent']}, Got: {err['predicted']['intent']}")

    print(f"\n  Ingress Errors: {len(errors_ingress)}/{len(results)}")
    for err in errors_ingress[:5]:
        print(f"    - \"{err['text'][:40]}...\"")
        print(f"      Expected: {err['expected']['ingress']}, Got: {err['predicted']['ingress']}")

    print(f"\n  Complexity Errors: {len(errors_complexity)}/{len(results)}")
    for err in errors_complexity[:5]:
        print(f"    - \"{err['text'][:40]}...\"")
        print(f"      Expected: {err['expected']['complexity']}, Got: {err['predicted']['complexity']}")
        print(f"      Reason: {err['complexity_reason'][:50]}...")

    # CRITICAL: Show all crisis cases
    crisis_cases = [r for r in results if r["is_crisis"]]
    print(f"\n  CRISIS CASE ANALYSIS ({len(crisis_cases)} cases):")
    for c in crisis_cases:
        status = "DETECTED" if c["crisis_detected"] else "MISSED"
        print(f"    [{status}] \"{c['text'][:50]}...\"")
        print(f"             Safety: {c['predicted']['safety']}, Complexity: {c['predicted']['complexity']}")

    # Show some successful examples
    print(f"\n{'='*70}")
    print(f"  SAMPLE SUCCESSFUL PREDICTIONS")
    print(f"{'='*70}")

    # Show 5 random successful ones
    successful = [r for r in results if r["correct"]["intent"] and r["correct"]["ingress"] and r["correct"]["complexity"]]
    for result in successful[:5]:
        print_test_result(result)

    # Final summary
    report = metrics.get_report()
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Total Tests:        {len(results)}")
    print(f"  Intent Accuracy:    {report['accuracy']['intent']:.1f}%")
    print(f"  Ingress Accuracy:   {report['accuracy']['ingress']:.1f}%")
    print(f"  Complexity Accuracy:{report['accuracy']['complexity']:.1f}%")
    print(f"  Crisis Recall:      {report['accuracy']['crisis_recall']:.1f}%")
    print(f"  P95 Latency:        {report['latency']['p95_ms']:.1f}ms")

    all_slos_met = (
        report['accuracy']['intent_slo_met'] and
        report['accuracy']['ingress_slo_met'] and
        report['accuracy']['complexity_slo_met'] and
        report['accuracy']['crisis_slo_met'] and
        report['latency']['p95_slo_met']
    )
    print(f"\n  ALL SLOs MET: {'YES' if all_slos_met else 'NO'}")
    print(f"{'='*70}")

    return results, metrics


if __name__ == "__main__":
    results, metrics = main()
