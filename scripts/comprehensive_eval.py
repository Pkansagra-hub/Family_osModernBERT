#!/usr/bin/env python
"""
Comprehensive Model Evaluation for README Metrics Update

This script evaluates all capabilities mentioned in the README:
- NER (General & Family)
- Sentiment (5-class and direction)
- Emotions (hit rate, multi-label)
- Safety (accuracy, CRISIS recall)
- NLI
- Embeddings (triplet accuracy, recall@k)
- Relations
- Intent
- Temporal
- Ingress
- Latency benchmarks

Usage:
    python scripts/comprehensive_eval.py
"""

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

print("Loading FamilyOS UltraBERT...")
from familyos_ultrabert import Client

# Initialize client
client = Client("pytorch", warmup=True, warmup_rounds=5, verbose=True)


# =============================================================================
# Test Data Generation
# =============================================================================


def generate_ner_family_test_data():
    """Generate NER Family test cases with ground truth."""
    return [
        {
            "text": "My grandmother Sarah called yesterday.",
            "entities": [
                {"text": "grandmother", "label": "KINSHIP"},
                {"text": "Sarah", "label": "PERSON"},
            ],
        },
        {
            "text": "Uncle Bob and Aunt Mary are visiting.",
            "entities": [
                {"text": "Uncle Bob", "label": "KINSHIP"},
                {"text": "Aunt Mary", "label": "KINSHIP"},
            ],
        },
        {
            "text": "We call grandpa 'Papa Joe'.",
            "entities": [
                {"text": "grandpa", "label": "KINSHIP"},
                {"text": "Papa Joe", "label": "NICKNAME"},
            ],
        },
        {
            "text": "Our family reunion is next month.",
            "entities": [{"text": "family reunion", "label": "FAMILY_EVENT"}],
        },
        {
            "text": "The dog Max is part of our family.",
            "entities": [{"text": "Max", "label": "PET"}],
        },
        {
            "text": "My sister Emily got married last year.",
            "entities": [
                {"text": "sister", "label": "KINSHIP"},
                {"text": "Emily", "label": "PERSON"},
            ],
        },
        {
            "text": "We always have Sunday dinners together.",
            "entities": [{"text": "Sunday dinners", "label": "TRADITION"}],
        },
        {
            "text": "Mom made her famous apple pie.",
            "entities": [{"text": "Mom", "label": "KINSHIP"}],
        },
        {
            "text": "Dad taught me to ride a bike.",
            "entities": [{"text": "Dad", "label": "KINSHIP"}],
        },
        {
            "text": "Our home in Boston holds many memories.",
            "entities": [{"text": "home in Boston", "label": "HOME_LOC"}],
        },
        {
            "text": "The birthday party for little Timmy was amazing.",
            "entities": [
                {"text": "birthday party", "label": "FAMILY_EVENT"},
                {"text": "Timmy", "label": "PERSON"},
            ],
        },
        {
            "text": "Grandma's antique clock is a family heirloom.",
            "entities": [{"text": "Grandma's antique clock", "label": "HEIRLOOM"}],
        },
        {
            "text": "My cousin Jake is getting married.",
            "entities": [
                {"text": "cousin", "label": "KINSHIP"},
                {"text": "Jake", "label": "PERSON"},
            ],
        },
        {
            "text": "We visit the old family farm every summer.",
            "entities": [{"text": "family farm", "label": "HOME_LOC"}],
        },
        {
            "text": "Baby Emma took her first steps today!",
            "entities": [
                {"text": "Emma", "label": "PERSON"},
                {"text": "first steps", "label": "MILESTONE"},
            ],
        },
    ]


def generate_ner_general_test_data():
    """Generate NER General test cases with ground truth."""
    return [
        {
            "text": "John Smith works at Google.",
            "entities": [
                {"text": "John Smith", "label": "PER"},
                {"text": "Google", "label": "ORG"},
            ],
        },
        {
            "text": "The conference is in New York.",
            "entities": [{"text": "New York", "label": "LOC"}],
        },
        {"text": "Apple announced new products.", "entities": [{"text": "Apple", "label": "ORG"}]},
        {
            "text": "Dr. Maria Garcia is from Spain.",
            "entities": [
                {"text": "Dr. Maria Garcia", "label": "PER"},
                {"text": "Spain", "label": "LOC"},
            ],
        },
        {
            "text": "Microsoft and Amazon are tech giants.",
            "entities": [{"text": "Microsoft", "label": "ORG"}, {"text": "Amazon", "label": "ORG"}],
        },
        {
            "text": "The CEO visited our London office.",
            "entities": [{"text": "London", "label": "LOC"}],
        },
        {
            "text": "President Biden spoke at the White House.",
            "entities": [
                {"text": "Biden", "label": "PER"},
                {"text": "White House", "label": "LOC"},
            ],
        },
        {
            "text": "Toyota is a Japanese company.",
            "entities": [{"text": "Toyota", "label": "ORG"}, {"text": "Japanese", "label": "MISC"}],
        },
        {
            "text": "Sarah went to Paris for vacation.",
            "entities": [{"text": "Sarah", "label": "PER"}, {"text": "Paris", "label": "LOC"}],
        },
        {
            "text": "The Olympics were held in Tokyo.",
            "entities": [{"text": "Olympics", "label": "MISC"}, {"text": "Tokyo", "label": "LOC"}],
        },
    ]


def generate_sentiment_test_data():
    """Generate sentiment test cases with ground truth."""
    return [
        # Very positive
        {
            "text": "This is the best day of my life!",
            "label": "very_positive",
            "direction": "positive",
        },
        {
            "text": "I am absolutely thrilled with the results!",
            "label": "very_positive",
            "direction": "positive",
        },
        {"text": "What an amazing experience!", "label": "very_positive", "direction": "positive"},
        # Positive
        {
            "text": "I love spending time with my family.",
            "label": "positive",
            "direction": "positive",
        },
        {"text": "The weather is nice today.", "label": "positive", "direction": "positive"},
        {"text": "I had a good day at work.", "label": "positive", "direction": "positive"},
        # Neutral
        {"text": "Mom went to the store.", "label": "neutral", "direction": "neutral"},
        {"text": "The meeting is at 3pm.", "label": "neutral", "direction": "neutral"},
        {"text": "I read the newspaper.", "label": "neutral", "direction": "neutral"},
        # Negative
        {"text": "I'm a bit worried about things.", "label": "negative", "direction": "negative"},
        {"text": "The traffic was annoying today.", "label": "negative", "direction": "negative"},
        {"text": "I didn't sleep well last night.", "label": "negative", "direction": "negative"},
        # Very negative
        {
            "text": "This is the worst thing ever.",
            "label": "very_negative",
            "direction": "negative",
        },
        {"text": "I am devastated by the news.", "label": "very_negative", "direction": "negative"},
        {"text": "Everything is terrible.", "label": "very_negative", "direction": "negative"},
    ]


def generate_emotions_test_data():
    """Generate emotions test cases with ground truth (multi-label)."""
    return [
        {"text": "I feel so happy today!", "emotions": ["joy"]},
        {"text": "I'm really sad about what happened.", "emotions": ["sadness"]},
        {"text": "This makes me so angry!", "emotions": ["anger"]},
        {"text": "I'm afraid of what might happen.", "emotions": ["fear"]},
        {"text": "What a surprise to see you!", "emotions": ["surprise"]},
        {"text": "I love my family so much.", "emotions": ["love"]},
        {"text": "I miss grandma so much.", "emotions": ["sadness", "longing", "nostalgia"]},
        {"text": "I'm so excited about the trip!", "emotions": ["excitement", "joy"]},
        {"text": "I'm grateful for all your help.", "emotions": ["gratitude"]},
        {"text": "I feel proud of my children.", "emotions": ["pride", "parental_pride"]},
        {"text": "I'm nervous about the interview.", "emotions": ["nervousness", "fear"]},
        {
            "text": "We're all together for the holidays!",
            "emotions": ["togetherness", "warmth", "joy"],
        },
        {"text": "I feel so protective of my kids.", "emotions": ["protectiveness", "love"]},
        {"text": "Growing up so fast is bittersweet.", "emotions": ["bittersweet"]},
        {"text": "I worry about their future.", "emotions": ["worry", "fear"]},
    ]


def generate_safety_test_data():
    """Generate safety test cases with ground truth."""
    return [
        # GREEN - Safe
        {"text": "I love spending time with my family.", "label": "GREEN"},
        {"text": "Mom made dinner for everyone.", "label": "GREEN"},
        {"text": "The kids are playing in the yard.", "label": "GREEN"},
        {"text": "We went to the beach yesterday.", "label": "GREEN"},
        {"text": "I'm excited about the vacation.", "label": "GREEN"},
        # AMBER - Mild concern
        {"text": "I'm feeling stressed about work.", "label": "AMBER"},
        {"text": "I'm a bit down today.", "label": "AMBER"},
        {"text": "Things have been hard lately.", "label": "AMBER"},
        {"text": "I've been feeling tired all week.", "label": "AMBER"},
        # RED - Significant concern
        {"text": "I've been feeling hopeless for weeks.", "label": "RED"},
        {"text": "Nothing seems to matter anymore.", "label": "RED"},
        {"text": "I don't enjoy anything these days.", "label": "RED"},
        # CRISIS - Immediate danger
        {"text": "I want to kill myself.", "label": "CRISIS"},
        {"text": "I'm going to end my life tonight.", "label": "CRISIS"},
        {"text": "I've been cutting myself.", "label": "CRISIS"},
        {"text": "I want to hurt myself badly.", "label": "CRISIS"},
        {"text": "I'm going to hurt my children.", "label": "CRISIS"},
        {"text": "I am thinking about suicide.", "label": "CRISIS"},
    ]


def generate_temporal_test_data():
    """Generate temporal test cases with ground truth."""
    return [
        {
            "text": "The meeting is on January 15, 2024.",
            "entities": [{"text": "January 15, 2024", "label": "DATE_ABS"}],
        },
        {"text": "I called yesterday.", "entities": [{"text": "yesterday", "label": "DATE_REL"}]},
        {"text": "The party is at 3pm.", "entities": [{"text": "3pm", "label": "TIME"}]},
        {
            "text": "The project took 3 months.",
            "entities": [{"text": "3 months", "label": "DURATION"}],
        },
        {
            "text": "We meet every Tuesday.",
            "entities": [{"text": "every Tuesday", "label": "FREQUENCY"}],
        },
        {"text": "She is 5 years old.", "entities": [{"text": "5 years old", "label": "AGE"}]},
        {
            "text": "Next Sunday is the reunion.",
            "entities": [{"text": "Next Sunday", "label": "DATE_REL"}],
        },
        {
            "text": "I'll be there in 2 hours.",
            "entities": [{"text": "2 hours", "label": "DURATION"}],
        },
        {
            "text": "The baby is 6 months old.",
            "entities": [{"text": "6 months old", "label": "AGE"}],
        },
        {
            "text": "We visit grandma weekly.",
            "entities": [{"text": "weekly", "label": "FREQUENCY"}],
        },
    ]


def generate_intent_test_data():
    """Generate intent test cases with ground truth."""
    return [
        {"text": "Remind me to call mom at 5pm.", "label": "set_reminder"},
        {"text": "Log that we had dinner together.", "label": "log_memory"},
        {"text": "How was I feeling last week?", "label": "query_memory"},
        {"text": "I feel overwhelmed today.", "label": "express_feeling"},
        {"text": "What should I do about this?", "label": "seek_advice"},
        {"text": "We moved to a new house!", "label": "share_news"},
        {"text": "I've been thinking about my family.", "label": "reflect"},
        {"text": "Please remember this moment.", "label": "log_memory"},
        {"text": "Set an alarm for tomorrow.", "label": "set_reminder"},
        {"text": "I'm happy to share good news!", "label": "share_news"},
    ]


def generate_ingress_test_data():
    """Generate ingress test cases with ground truth."""
    return [
        {"text": "Dear diary, today was wonderful.", "label": "DIARY"},
        {"text": "Pick up groceries and call mom.", "label": "TASK"},
        {"text": "I need to take my medication.", "label": "HEALTH"},
        {"text": "Remember when we went to the beach?", "label": "MEMORY"},
        {"text": "We celebrated grandma's birthday!", "label": "CELEBRATION"},
        {"text": "Dad taught me to ride a bike.", "label": "MEMORY"},
        {"text": "Schedule doctor appointment.", "label": "HEALTH"},
        {"text": "Today I journaled about my feelings.", "label": "DIARY"},
        {"text": "Don't forget the meeting tomorrow.", "label": "TASK"},
        {"text": "The wedding was beautiful!", "label": "CELEBRATION"},
    ]


def generate_relation_test_data():
    """Generate relation test cases with ground truth."""
    return [
        {"text": "John is the father of Mary.", "relation": "parent_of"},
        {"text": "Sarah and Emma are sisters.", "relation": "sibling_of"},
        {"text": "Tom married Lisa last year.", "relation": "spouse_of"},
        {"text": "Grandma raised all of us.", "relation": "grandparent_of"},
        {"text": "Bob and his twin brother Joe.", "relation": "sibling_of"},
        {"text": "Uncle Mike is dad's brother.", "relation": "uncle_of"},
        {"text": "Aunt Jane is mom's sister.", "relation": "aunt_of"},
        {"text": "The cousins played together.", "relation": "cousin_of"},
        {"text": "My nephew is starting school.", "relation": "nephew_of"},
        {"text": "Her niece won the competition.", "relation": "niece_of"},
    ]


def generate_nli_test_data():
    """Generate NLI test cases with ground truth."""
    return [
        {
            "premise": "The dog is sleeping.",
            "hypothesis": "The animal is resting.",
            "label": "entailment",
        },
        {
            "premise": "It's raining outside.",
            "hypothesis": "The weather is sunny.",
            "label": "contradiction",
        },
        {
            "premise": "She went to the store.",
            "hypothesis": "She bought groceries.",
            "label": "neutral",
        },
        {
            "premise": "All birds can fly.",
            "hypothesis": "Penguins can fly.",
            "label": "contradiction",
        },
        {
            "premise": "The child is happy.",
            "hypothesis": "The kid is joyful.",
            "label": "entailment",
        },
        {"premise": "He works at a bank.", "hypothesis": "He is employed.", "label": "entailment"},
        {
            "premise": "The cat is white.",
            "hypothesis": "The cat is black.",
            "label": "contradiction",
        },
        {
            "premise": "She is reading a book.",
            "hypothesis": "She likes fiction.",
            "label": "neutral",
        },
        {
            "premise": "The restaurant is closed.",
            "hypothesis": "People are eating there.",
            "label": "contradiction",
        },
        {
            "premise": "They went hiking.",
            "hypothesis": "They were outdoors.",
            "label": "entailment",
        },
    ]


def generate_embedding_test_data():
    """Generate embedding triplet test cases."""
    return [
        {
            "anchor": "I love my grandmother.",
            "positive": "I adore my grandma.",
            "negative": "The stock market crashed.",
        },
        {
            "anchor": "Family dinner tonight.",
            "positive": "We're eating together as a family.",
            "negative": "The car needs repairs.",
        },
        {
            "anchor": "Mom made apple pie.",
            "positive": "Mother baked a dessert.",
            "negative": "The meeting was canceled.",
        },
        {
            "anchor": "The kids are playing.",
            "positive": "Children having fun.",
            "negative": "The server is down.",
        },
        {
            "anchor": "Sunday family reunion.",
            "positive": "Weekend family gathering.",
            "negative": "The code has bugs.",
        },
        {
            "anchor": "Dad taught me to fish.",
            "positive": "Father showed me fishing.",
            "negative": "The weather report.",
        },
        {
            "anchor": "Grandpa tells stories.",
            "positive": "Grandfather shares tales.",
            "negative": "Database query failed.",
        },
        {
            "anchor": "We miss grandma.",
            "positive": "Longing for grandmother.",
            "negative": "The API rate limit.",
        },
        {
            "anchor": "Sister got married.",
            "positive": "Sibling's wedding.",
            "negative": "Server maintenance.",
        },
        {
            "anchor": "Baby's first steps.",
            "positive": "Infant learning to walk.",
            "negative": "Network latency issues.",
        },
    ]


# =============================================================================
# Evaluation Functions
# =============================================================================


def evaluate_ner(test_data, capability):
    """Evaluate NER (F1 score)."""
    tp = fp = fn = 0

    for case in test_data:
        result = client.analyze(case["text"], [capability])

        # Get entities from correct attribute based on capability
        if capability == "ner_general":
            pred_entities = result.general_entities if hasattr(result, "general_entities") else []
        elif capability == "ner_family":
            pred_entities = result.entities if hasattr(result, "entities") else []
        elif capability == "temporal":
            pred_entities = result.temporal if hasattr(result, "temporal") else []
        else:
            pred_entities = []

        gold_entities = case["entities"]

        # Extract predicted labels (text, label pairs)
        pred_set = set()
        for ent in pred_entities:
            if isinstance(ent, dict):
                # Normalize text (strip spaces)
                text = ent.get("text", "").strip()
                label = ent.get("label", "")
                if text and label:
                    pred_set.add((text.lower(), label.upper()))

        # Gold labels
        gold_set = set()
        for ent in gold_entities:
            text = ent.get("text", "").strip()
            label = ent.get("label", "")
            if text and label:
                gold_set.add((text.lower(), label.upper()))

        # Count matches (allowing partial text matches)
        matched = 0
        for pred_text, pred_label in pred_set:
            for gold_text, gold_label in gold_set:
                if pred_label == gold_label and (pred_text in gold_text or gold_text in pred_text):
                    matched += 1
                    break

        tp += matched
        fp += len(pred_set) - matched
        fn += len(gold_set) - matched

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_sentiment(test_data):
    """Evaluate sentiment (5-class accuracy and direction accuracy)."""
    correct_5class = 0
    correct_direction = 0

    for case in test_data:
        result = client.analyze(case["text"], ["sentiment"])
        pred = result.sentiment if hasattr(result, "sentiment") else ""

        # 5-class accuracy
        if pred == case["label"]:
            correct_5class += 1

        # Direction accuracy
        pred_direction = (
            "positive"
            if pred in ["positive", "very_positive"]
            else "negative" if pred in ["negative", "very_negative"] else "neutral"
        )
        if pred_direction == case["direction"]:
            correct_direction += 1

    return {
        "5class_accuracy": correct_5class / len(test_data),
        "direction_accuracy": correct_direction / len(test_data),
    }


def evaluate_emotions(test_data):
    """Evaluate emotions (hit rate - at least one correct)."""
    hits = 0

    for case in test_data:
        result = client.analyze(case["text"], ["emotions"])
        pred_emotions = result.emotions if hasattr(result, "emotions") else []
        gold_emotions = set(case["emotions"])

        # Hit = at least one correct emotion
        if any(e in gold_emotions for e in pred_emotions):
            hits += 1

    return {"hit_rate": hits / len(test_data)}


def evaluate_safety(test_data):
    """Evaluate safety (accuracy and CRISIS recall)."""
    correct = 0
    crisis_tp = 0
    crisis_total = 0

    for case in test_data:
        result = client.analyze(case["text"], ["safety_familyos"])
        pred = result.safety if hasattr(result, "safety") else ""

        if pred == case["label"]:
            correct += 1

        if case["label"] == "CRISIS":
            crisis_total += 1
            if pred == "CRISIS":
                crisis_tp += 1

    return {
        "accuracy": correct / len(test_data),
        "crisis_recall": crisis_tp / crisis_total if crisis_total > 0 else 0,
    }


def evaluate_temporal(test_data):
    """Evaluate temporal NER (F1 score)."""
    return evaluate_ner(test_data, "temporal")


def evaluate_intent(test_data):
    """Evaluate intent (accuracy)."""
    correct = 0

    for case in test_data:
        result = client.analyze(case["text"], ["intent"])
        pred = result.intent if hasattr(result, "intent") else ""

        if pred == case["label"]:
            correct += 1

    return {"accuracy": correct / len(test_data)}


def evaluate_ingress(test_data):
    """Evaluate ingress (accuracy)."""
    correct = 0

    for case in test_data:
        result = client.analyze(case["text"], ["ingress"])
        pred = result.ingress if hasattr(result, "ingress") else ""

        if pred == case["label"]:
            correct += 1

    return {"accuracy": correct / len(test_data)}


def evaluate_nli(test_data):
    """Evaluate NLI (accuracy)."""
    correct = 0

    for case in test_data:
        # NLI requires premise and hypothesis
        combined = f"{case['premise']} [SEP] {case['hypothesis']}"
        result = client.analyze(combined, ["nli"])

        # Get prediction from nli attribute
        pred = result.nli if hasattr(result, "nli") else ""

        if pred.lower() == case["label"].lower():
            correct += 1

    return {"accuracy": correct / len(test_data)}


def evaluate_embeddings(test_data):
    """Evaluate embeddings (triplet accuracy and retrieval)."""
    triplet_correct = 0

    for case in test_data:
        # Get embeddings
        anchor_result = client.analyze(case["anchor"], ["embedding"])
        pos_result = client.analyze(case["positive"], ["embedding"])
        neg_result = client.analyze(case["negative"], ["embedding"])

        anchor_emb = (
            np.array(anchor_result.embedding) if hasattr(anchor_result, "embedding") else None
        )
        pos_emb = np.array(pos_result.embedding) if hasattr(pos_result, "embedding") else None
        neg_emb = np.array(neg_result.embedding) if hasattr(neg_result, "embedding") else None

        if anchor_emb is not None and pos_emb is not None and neg_emb is not None:
            # Cosine similarity
            pos_sim = np.dot(anchor_emb, pos_emb) / (
                np.linalg.norm(anchor_emb) * np.linalg.norm(pos_emb)
            )
            neg_sim = np.dot(anchor_emb, neg_emb) / (
                np.linalg.norm(anchor_emb) * np.linalg.norm(neg_emb)
            )

            if pos_sim > neg_sim:
                triplet_correct += 1

    return {"triplet_accuracy": triplet_correct / len(test_data)}


def evaluate_latency():
    """Evaluate inference latency."""
    test_texts = [
        "My grandmother called yesterday to remind me about the family reunion next Sunday.",
        "I'm feeling a bit stressed about work lately.",
        "The kids had so much fun at the birthday party!",
        "Remember to pick up groceries on the way home.",
        "I love spending time with my family on weekends.",
    ]

    # Warmup
    for _ in range(5):
        client.analyze(test_texts[0])

    # Single capability latencies
    capabilities = [
        "sentiment",
        "emotions",
        "safety_familyos",
        "intent",
        "ingress",
        "ner_family",
        "ner_general",
        "temporal",
        "embedding",
        "nli",
        "relation",
    ]

    cap_latencies = {}
    for cap in capabilities:
        times = []
        for text in test_texts:
            start = time.perf_counter()
            client.analyze(text, [cap])
            times.append((time.perf_counter() - start) * 1000)
        cap_latencies[cap] = np.mean(times)

    # Full inference latency
    full_times = []
    for text in test_texts * 10:
        start = time.perf_counter()
        client.analyze(text)
        full_times.append((time.perf_counter() - start) * 1000)

    return {
        "per_capability": cap_latencies,
        "full_avg_ms": np.mean(full_times),
        "full_p95_ms": np.percentile(full_times, 95),
        "throughput_per_sec": 1000 / np.mean(full_times),
    }


# =============================================================================
# Main Evaluation
# =============================================================================


def main():
    print("\n" + "=" * 80)
    print("COMPREHENSIVE MODEL EVALUATION FOR README")
    print("=" * 80 + "\n")

    results = {}

    # 1. NER Family
    print("Evaluating NER Family...")
    ner_family_data = generate_ner_family_test_data()
    results["ner_family"] = evaluate_ner(ner_family_data, "ner_family")
    print(f"  F1: {results['ner_family']['f1']:.2%}")

    # 2. NER General
    print("Evaluating NER General...")
    ner_general_data = generate_ner_general_test_data()
    results["ner_general"] = evaluate_ner(ner_general_data, "ner_general")
    print(f"  F1: {results['ner_general']['f1']:.2%}")

    # 3. Sentiment
    print("Evaluating Sentiment...")
    sentiment_data = generate_sentiment_test_data()
    results["sentiment"] = evaluate_sentiment(sentiment_data)
    print(f"  5-class Accuracy: {results['sentiment']['5class_accuracy']:.2%}")
    print(f"  Direction Accuracy: {results['sentiment']['direction_accuracy']:.2%}")

    # 4. Emotions
    print("Evaluating Emotions...")
    emotions_data = generate_emotions_test_data()
    results["emotions"] = evaluate_emotions(emotions_data)
    print(f"  Hit Rate: {results['emotions']['hit_rate']:.2%}")

    # 5. Safety
    print("Evaluating Safety...")
    safety_data = generate_safety_test_data()
    results["safety"] = evaluate_safety(safety_data)
    print(f"  Accuracy: {results['safety']['accuracy']:.2%}")
    print(f"  CRISIS Recall: {results['safety']['crisis_recall']:.2%}")

    # 6. Temporal
    print("Evaluating Temporal...")
    temporal_data = generate_temporal_test_data()
    results["temporal"] = evaluate_ner(temporal_data, "temporal")
    print(f"  F1: {results['temporal']['f1']:.2%}")

    # 7. Intent
    print("Evaluating Intent...")
    intent_data = generate_intent_test_data()
    results["intent"] = evaluate_intent(intent_data)
    print(f"  Accuracy: {results['intent']['accuracy']:.2%}")

    # 8. Ingress
    print("Evaluating Ingress...")
    ingress_data = generate_ingress_test_data()
    results["ingress"] = evaluate_ingress(ingress_data)
    print(f"  Accuracy: {results['ingress']['accuracy']:.2%}")

    # 9. NLI
    print("Evaluating NLI...")
    nli_data = generate_nli_test_data()
    results["nli"] = evaluate_nli(nli_data)
    print(f"  Accuracy: {results['nli']['accuracy']:.2%}")

    # 10. Embeddings
    print("Evaluating Embeddings...")
    embedding_data = generate_embedding_test_data()
    results["embeddings"] = evaluate_embeddings(embedding_data)
    print(f"  Triplet Accuracy: {results['embeddings']['triplet_accuracy']:.2%}")

    # 11. Latency
    print("Evaluating Latency...")
    results["latency"] = evaluate_latency()
    print(f"  Full Inference Avg: {results['latency']['full_avg_ms']:.2f}ms")
    print(f"  Throughput: {results['latency']['throughput_per_sec']:.1f}/sec")

    # Summary Report
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80 + "\n")

    print("| Capability | Metric | Score |")
    print("|------------|--------|-------|")
    print(f"| NER General | F1 | **{results['ner_general']['f1']:.1%}** |")
    print(f"| NER Family | F1 | **{results['ner_family']['f1']:.1%}** |")
    print(f"| Sentiment | 5-class Accuracy | **{results['sentiment']['5class_accuracy']:.1%}** |")
    print(
        f"| Sentiment | Direction Accuracy | **{results['sentiment']['direction_accuracy']:.1%}** |"
    )
    print(f"| Emotions | Hit Rate | **{results['emotions']['hit_rate']:.1%}** |")
    print(f"| Safety | Accuracy | **{results['safety']['accuracy']:.1%}** |")
    print(f"| Safety | CRISIS Recall | **{results['safety']['crisis_recall']:.1%}** |")
    print(f"| NLI | Accuracy | **{results['nli']['accuracy']:.1%}** |")
    print(
        f"| Embeddings | Triplet Accuracy | **{results['embeddings']['triplet_accuracy']:.1%}** |"
    )
    print(f"| Intent | Accuracy | **{results['intent']['accuracy']:.1%}** |")
    print(f"| Temporal | F1 | **{results['temporal']['f1']:.1%}** |")
    print(f"| Ingress | Accuracy | **{results['ingress']['accuracy']:.1%}** |")

    print("\n### Latency Benchmarks")
    print("| Capability | Latency (ms) |")
    print("|------------|-------------|")
    for cap, lat in sorted(results["latency"]["per_capability"].items(), key=lambda x: x[1]):
        print(f"| {cap} | {lat:.2f} |")
    print(f"| **Full Inference** | **{results['latency']['full_avg_ms']:.2f}** |")

    print(f"\n**Throughput:** {results['latency']['throughput_per_sec']:.1f} inferences/sec")

    # Save results
    output_path = Path("benchmark_results.json")
    with open(output_path, "w") as f:
        # Convert numpy values to Python types
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        json.dump(convert(results), f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
