"""
CHECKPOINT COMPARISON: 14000 vs 16500
=====================================
Compare actual outputs (not just pass/fail) to find subtle differences.
"""

import sys
sys.path.insert(0, "D:\\Modeling_studio\\familyos_ultrabert")

import numpy as np
from familyos_ultrabert import UltraBERT

print("=" * 80)
print("  CHECKPOINT COMPARISON: 14000 vs 16500")
print("=" * 80)

# Test inputs covering all 12 heads
TEST_CASES = [
    # Sentiment edge cases
    "I'm so happy my daughter won the competition!",
    "This is the worst day of my life.",
    "Well, isn't that just great. Thanks for nothing.",  # Sarcasm
    "The weather is nice today.",  # Neutral

    # Emotion edge cases
    "I miss my grandmother so much since she passed.",
    "My son is driving me crazy but I love him.",
    "I feel so alone even when surrounded by family.",

    # Safety edge cases
    "I don't want to live anymore.",  # Crisis
    "My husband hit me again last night.",  # DV
    "Just a normal day with the kids.",  # Green

    # NER edge cases
    "My step-dad's ex-wife's daughter came to visit.",  # Blended family
    "Aunt Mary, Uncle Bob, and cousin Jake arrived.",
    "We went to Target, Walmart, and Costco.",  # Retail
    "The service at St. Mary's Church was beautiful.",  # Religious

    # Temporal
    "The appointment is next Tuesday at 3:30pm.",
    "We've been married for 25 years now.",

    # Intent
    "Remind me to call Mom tomorrow.",
    "I just need someone to talk to about my problems.",
]

NLI_CASES = [
    ("My daughter is 10 years old.", "My child is a minor."),
    ("It is raining.", "The ground is dry."),
]

CHECKPOINT_PATHS = {
    "14000": "D:\\Modeling_studio\\outputs\\modernbert-v2-for-v3-transfer\\checkpoint-14000",
    "16500": "D:\\Modeling_studio\\outputs\\modernbert-v2-for-v3-transfer\\checkpoint-16500",
}

def get_all_outputs(model, text):
    """Get all outputs from a model for comparison."""
    result = model.analyze(text)
    caps = result.capabilities

    return {
        "sentiment": caps.get("sentiment", {}).get("label"),
        "sentiment_conf": caps.get("sentiment", {}).get("confidence"),
        "emotions": caps.get("emotions", {}).get("labels", []),
        "safety": caps.get("safety_familyos", {}).get("label"),
        "safety_conf": caps.get("safety_familyos", {}).get("confidence"),
        "intent": caps.get("intent", {}).get("label"),
        "intent_conf": caps.get("intent", {}).get("confidence"),
        "ingress": caps.get("ingress", {}).get("label"),
        "ner_family": [e.get("text") for e in caps.get("ner_family", {}).get("entities", [])],
        "ner_general": [e.get("text") for e in caps.get("ner_general", {}).get("entities", [])],
        "temporal": [e.get("text") for e in caps.get("temporal", {}).get("entities", [])],
        "relations": caps.get("relation", {}).get("labels", []),
    }

def compare_outputs(out1, out2):
    """Compare two outputs and report differences."""
    diffs = []

    for key in out1:
        v1, v2 = out1[key], out2[key]

        if key.endswith("_conf"):
            # Numeric comparison
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                diff = abs(v1 - v2)
                if diff > 0.01:  # 1% threshold
                    diffs.append(f"  {key}: {v1:.4f} -> {v2:.4f} (diff: {diff:.4f})")
        else:
            # Categorical comparison
            if v1 != v2:
                diffs.append(f"  {key}: '{v1}' -> '{v2}'")

    return diffs

# Load models
print("\nLoading checkpoint-14000...")
model_14000 = UltraBERT.load(
    model_path=CHECKPOINT_PATHS["14000"],
    backend="pytorch",
    device="cuda"
)

print("Loading checkpoint-16500...")
model_16500 = UltraBERT.load(
    model_path=CHECKPOINT_PATHS["16500"],
    backend="pytorch",
    device="cuda"
)

print("\n" + "=" * 80)
print("  DETAILED COMPARISON")
print("=" * 80)

total_diffs = 0
diff_details = []

for i, test in enumerate(TEST_CASES):
    out1 = get_all_outputs(model_14000, test)
    out2 = get_all_outputs(model_16500, test)
    text_display = test[:60] + "..." if len(test) > 60 else test

    diffs = compare_outputs(out1, out2)

    if diffs:
        total_diffs += 1
        print(f"\n[{i+1}] DIFF: {text_display}")
        for d in diffs:
            print(d)
        diff_details.append((test, diffs))
    else:
        print(f"[{i+1}] SAME: {text_display}")

# NLI cases
for i, (premise, hypothesis) in enumerate(NLI_CASES):
    text = f"{premise} [SEP] {hypothesis}"
    out1 = get_all_outputs(model_14000, text)
    out2 = get_all_outputs(model_16500, text)

    # Get NLI specifically
    r1 = model_14000.analyze(text)
    r2 = model_16500.analyze(text)
    nli1 = r1.capabilities.get("nli", {}).get("label")
    nli2 = r2.capabilities.get("nli", {}).get("label")

    diffs = []
    if nli1 != nli2:
        diffs.append(f"  nli: '{nli1}' -> '{nli2}'")

    text_display = f"{premise[:25]}... | {hypothesis[:15]}..."

    if diffs:
        total_diffs += 1
        print(f"\n[NLI-{i+1}] DIFF: {text_display}")
        for d in diffs:
            print(d)
    else:
        print(f"[NLI-{i+1}] SAME: {text_display}")

print("\n" + "=" * 80)
print("  CONFIDENCE SCORE ANALYSIS")
print("=" * 80)

# Collect confidence scores for statistical analysis
conf_14000 = {"sentiment": [], "safety": [], "intent": []}
conf_16500 = {"sentiment": [], "safety": [], "intent": []}

for test in TEST_CASES:
    out1 = get_all_outputs(model_14000, test)
    out2 = get_all_outputs(model_16500, test)

    if out1["sentiment_conf"]:
        conf_14000["sentiment"].append(out1["sentiment_conf"])
        conf_16500["sentiment"].append(out2["sentiment_conf"])
    if out1["safety_conf"]:
        conf_14000["safety"].append(out1["safety_conf"])
        conf_16500["safety"].append(out2["safety_conf"])
    if out1["intent_conf"]:
        conf_14000["intent"].append(out1["intent_conf"])
        conf_16500["intent"].append(out2["intent_conf"])

print("\n  Average Confidence Scores:")
print("  " + "-" * 50)
for head in ["sentiment", "safety", "intent"]:
    avg1 = np.mean(conf_14000[head]) if conf_14000[head] else 0
    avg2 = np.mean(conf_16500[head]) if conf_16500[head] else 0
    diff = avg2 - avg1
    arrow = "+" if diff > 0 else "" if diff < 0 else "="
    print(f"  {head:12}: 14000={avg1:.4f}  16500={avg2:.4f}  ({arrow}{diff:.4f})")

print("\n" + "=" * 80)
print("  EMBEDDING DRIFT ANALYSIS")
print("=" * 80)

# Compare embedding similarity across checkpoints
embedding_sims = []
for test in TEST_CASES[:10]:
    result1 = model_14000.analyze(test)
    result2 = model_16500.analyze(test)

    emb1 = result1.capabilities.get("embedding", {}).get("vector", [])
    emb2 = result2.capabilities.get("embedding", {}).get("vector", [])

    if emb1 and emb2:
        e1, e2 = np.array(emb1), np.array(emb2)
        cos_sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)
        embedding_sims.append(cos_sim)

if embedding_sims:
    print(f"\n  Embedding Cosine Similarity (14000 vs 16500):")
    print(f"    Min:  {min(embedding_sims):.6f}")
    print(f"    Max:  {max(embedding_sims):.6f}")
    print(f"    Mean: {np.mean(embedding_sims):.6f}")
    print(f"    Std:  {np.std(embedding_sims):.6f}")

    if np.mean(embedding_sims) > 0.99:
        print("\n  -> Embeddings are nearly IDENTICAL (>99% similarity)")
    elif np.mean(embedding_sims) > 0.95:
        print("\n  -> Embeddings have MINOR drift (95-99% similarity)")
    else:
        print("\n  -> Embeddings have SIGNIFICANT drift (<95% similarity)")

print("\n" + "=" * 80)
print("  SUMMARY")
print("=" * 80)
print(f"\n  Total test cases: {len(TEST_CASES) + len(NLI_CASES)}")
print(f"  Cases with differences: {total_diffs}")
print(f"  Cases identical: {len(TEST_CASES) + len(NLI_CASES) - total_diffs}")

if total_diffs == 0:
    print("\n  >> CHECKPOINTS ARE FUNCTIONALLY IDENTICAL <<")
elif total_diffs < 3:
    print("\n  >> MINIMAL DIFFERENCES - Checkpoints are very similar <<")
else:
    print("\n  >> NOTABLE DIFFERENCES FOUND - Review above <<")

print("\n" + "=" * 80)
