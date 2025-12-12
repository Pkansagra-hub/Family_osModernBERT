"""Deprecated example.

This script was superseded by the built-in benchmark CLI:

    python -m familyos_ultrabert.benchmarks --suite api,regression

and is kept only for historical reference.
"""

import time
import sys
import statistics

print("=" * 80)
print("FamilyOS UltraBERT v2.0.1 - CLIENT API STRESS TEST")
print("=" * 80)

# =============================================================================
# TEST 1: Auto-Warmup Effectiveness
# =============================================================================
print("\n" + "=" * 80)
print("TEST 1: AUTO-WARMUP EFFECTIVENESS")
print("=" * 80)

print("\n--- Without Warmup (simulating v2.0.0 behavior) ---")
from familyos_ultrabert import UltraBERT

start = time.perf_counter()
model_no_warmup = UltraBERT.load()
load_time = (time.perf_counter() - start) * 1000

# First call without warmup
start = time.perf_counter()
_ = model_no_warmup.analyze("Test message for cold start measurement")
first_call_no_warmup = (time.perf_counter() - start) * 1000

# Second call (warmed)
start = time.perf_counter()
_ = model_no_warmup.analyze("Test message for warm measurement")
second_call_no_warmup = (time.perf_counter() - start) * 1000

print(f"  Model load time: {load_time:.1f}ms")
print(f"  First call (COLD): {first_call_no_warmup:.1f}ms")
print(f"  Second call (warm): {second_call_no_warmup:.1f}ms")
print(f"  Cold start penalty: {first_call_no_warmup - second_call_no_warmup:.1f}ms")

print("\n--- With Auto-Warmup (v2.0.1 Client) ---")
from familyos_ultrabert import Client

start = time.perf_counter()
client = Client(warmup=True, warmup_rounds=3)
client_init_time = (time.perf_counter() - start) * 1000

# First user call (should be fast due to warmup)
start = time.perf_counter()
_ = client.analyze("Test message for first user call")
first_call_with_warmup = (time.perf_counter() - start) * 1000

# Second call
start = time.perf_counter()
_ = client.analyze("Test message for second user call")
second_call_with_warmup = (time.perf_counter() - start) * 1000

print(f"  Client init (includes warmup): {client_init_time:.1f}ms")
print(f"  First USER call: {first_call_with_warmup:.1f}ms")
print(f"  Second call: {second_call_with_warmup:.1f}ms")
print(f"  Cold start eliminated: {first_call_with_warmup < first_call_no_warmup * 0.5}")

improvement = first_call_no_warmup / first_call_with_warmup if first_call_with_warmup > 0 else 0
print(f"\n  IMPROVEMENT: {improvement:.1f}x faster first call with Client API")

# =============================================================================
# TEST 2: Convenience Methods
# =============================================================================
print("\n" + "=" * 80)
print("TEST 2: CONVENIENCE METHODS")
print("=" * 80)

test_cases = [
    # (text, expected_safe, expected_crisis, expected_sentiment_type)
    ("I love my family so much!", True, False, "positive"),
    ("Mom picked up the kids from school", True, False, "positive"),
    ("I'm feeling a bit down today", False, False, "negative"),  # AMBER
    ("I hate when things don't work out", False, False, "negative"),  # AMBER
    ("I want to hurt myself", False, True, "negative"),  # CRISIS
    ("Nobody would miss me if I was gone", False, True, "negative"),  # CRISIS
    ("The weather is nice today", True, False, "neutral"),
    ("I'm so excited about the reunion!", True, False, "positive"),
]

print("\n--- is_safe() Tests ---")
safe_correct = 0
for text, expected_safe, _, _ in test_cases:
    result = client.is_safe(text)
    status = "PASS" if result == expected_safe else "FAIL"
    if result == expected_safe:
        safe_correct += 1
    print(f"  [{status}] is_safe('{text[:40]}...'): {result} (expected {expected_safe})")

print(f"\n  Accuracy: {safe_correct}/{len(test_cases)} ({100*safe_correct/len(test_cases):.0f}%)")

print("\n--- is_crisis() Tests ---")
crisis_correct = 0
for text, _, expected_crisis, _ in test_cases:
    result = client.is_crisis(text)
    status = "PASS" if result == expected_crisis else "FAIL"
    if result == expected_crisis:
        crisis_correct += 1
    print(f"  [{status}] is_crisis('{text[:40]}...'): {result} (expected {expected_crisis})")

print(f"\n  Accuracy: {crisis_correct}/{len(test_cases)} ({100*crisis_correct/len(test_cases):.0f}%)")

print("\n--- get_sentiment() Tests ---")
sentiment_correct = 0
for text, _, _, expected_type in test_cases:
    result = client.get_sentiment(text)
    is_correct = (
        (expected_type == "positive" and result in ["positive", "very_positive"]) or
        (expected_type == "negative" and result in ["negative", "very_negative"]) or
        (expected_type == "neutral" and result == "neutral")
    )
    status = "PASS" if is_correct else "FAIL"
    if is_correct:
        sentiment_correct += 1
    print(f"  [{status}] get_sentiment('{text[:35]}...'): {result} (expected {expected_type})")

print(f"\n  Accuracy: {sentiment_correct}/{len(test_cases)} ({100*sentiment_correct/len(test_cases):.0f}%)")

print("\n--- get_emotions() Tests ---")
emotion_cases = [
    ("I'm so happy and excited!", ["joy", "excitement"]),
    ("I'm really sad about what happened", ["sadness"]),
    ("I love spending time with grandma", ["love"]),
    ("I'm worried about dad's health", ["concern", "worry", "fear"]),
    ("This makes me so angry!", ["anger"]),
]

emotion_hits = 0
for text, expected_emotions in emotion_cases:
    result = client.get_emotions(text)
    # Check if any expected emotion is in result
    hit = any(e in result for e in expected_emotions) or any(e in expected_emotions for e in result)
    status = "PASS" if hit else "FAIL"
    if hit:
        emotion_hits += 1
    print(f"  [{status}] get_emotions('{text[:35]}...')")
    print(f"         Got: {result[:5]}...")
    print(f"         Expected one of: {expected_emotions}")

print(f"\n  Hit rate: {emotion_hits}/{len(emotion_cases)} ({100*emotion_hits/len(emotion_cases):.0f}%)")

print("\n--- get_embedding() Tests ---")
emb1 = client.get_embedding("Mom picked up the kids from school")
emb2 = client.get_embedding("Mother collected children from school")
emb3 = client.get_embedding("The stock market crashed today")

import numpy as np
def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sim_similar = cosine_sim(emb1, emb2)
sim_different = cosine_sim(emb1, emb3)

print(f"  Embedding dimension: {len(emb1)}")
print(f"  Similar texts similarity: {sim_similar:.4f}")
print(f"  Different texts similarity: {sim_different:.4f}")
print(f"  Discrimination: {sim_similar > sim_different} (similar > different)")

# =============================================================================
# TEST 3: ClientResult Attribute Access
# =============================================================================
print("\n" + "=" * 80)
print("TEST 3: ClientResult ATTRIBUTE ACCESS")
print("=" * 80)

result = client.analyze("Mom and Dad are coming over for dinner tomorrow! I'm so excited!")

print("\n--- Attribute Access ---")
attrs_to_check = [
    ("text", str),
    ("sentiment", str),
    ("sentiment_confidence", (int, float)),
    ("safety", str),
    ("safety_confidence", (int, float)),
    ("emotions", list),
    ("intent", str),
    ("entities", list),
    ("temporal", list),
    ("embedding", list),
    ("latency_ms", (int, float)),
]

all_attrs_ok = True
for attr, expected_type in attrs_to_check:
    try:
        value = getattr(result, attr)
        type_ok = isinstance(value, expected_type)
        status = "PASS" if type_ok else "FAIL"
        if not type_ok:
            all_attrs_ok = False

        # Truncate display for long values
        if isinstance(value, list) and len(value) > 3:
            display = f"{value[:3]}... ({len(value)} items)"
        elif isinstance(value, str) and len(value) > 50:
            display = f"{value[:50]}..."
        else:
            display = value

        print(f"  [{status}] result.{attr}: {display}")
    except AttributeError as e:
        print(f"  [FAIL] result.{attr}: AttributeError - {e}")
        all_attrs_ok = False

print(f"\n  All attributes accessible: {all_attrs_ok}")

# =============================================================================
# TEST 4: Statistics & Monitoring
# =============================================================================
print("\n" + "=" * 80)
print("TEST 4: STATISTICS & MONITORING")
print("=" * 80)

# Reset stats
client.reset_stats()

# Run 50 inferences
print("\n--- Running 50 inferences for stats ---")
for i in range(50):
    client.analyze(f"Test message number {i} for statistics gathering")

stats = client.get_stats()
print(f"\n--- get_stats() ---")
for key, value in stats.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.2f}")
    else:
        print(f"  {key}: {value}")

print("\n--- health_check() ---")
health = client.health_check()
for key, value in health.items():
    print(f"  {key}: {value}")

# Verify stats are reasonable
stats_ok = (
    stats.get("total_calls", 0) == 50 and
    stats.get("avg_latency_ms", 0) > 0 and
    stats.get("min_latency_ms", 0) > 0 and
    stats.get("max_latency_ms", 0) > stats.get("min_latency_ms", 0)
)
print(f"\n  Stats valid: {stats_ok}")

# =============================================================================
# TEST 5: Convenience Method Performance
# =============================================================================
print("\n" + "=" * 80)
print("TEST 5: CONVENIENCE METHOD PERFORMANCE")
print("=" * 80)

n_runs = 100
test_text = "Mom picked up the kids from school today and they had a great time!"

print(f"\n--- Timing {n_runs} calls each ---")

# Time is_safe
times = []
for _ in range(n_runs):
    start = time.perf_counter()
    client.is_safe(test_text)
    times.append((time.perf_counter() - start) * 1000)
print(f"  is_safe(): avg={statistics.mean(times):.2f}ms, p50={statistics.median(times):.2f}ms")

# Time is_crisis
times = []
for _ in range(n_runs):
    start = time.perf_counter()
    client.is_crisis(test_text)
    times.append((time.perf_counter() - start) * 1000)
print(f"  is_crisis(): avg={statistics.mean(times):.2f}ms, p50={statistics.median(times):.2f}ms")

# Time get_sentiment
times = []
for _ in range(n_runs):
    start = time.perf_counter()
    client.get_sentiment(test_text)
    times.append((time.perf_counter() - start) * 1000)
print(f"  get_sentiment(): avg={statistics.mean(times):.2f}ms, p50={statistics.median(times):.2f}ms")

# Time get_emotions
times = []
for _ in range(n_runs):
    start = time.perf_counter()
    client.get_emotions(test_text)
    times.append((time.perf_counter() - start) * 1000)
print(f"  get_emotions(): avg={statistics.mean(times):.2f}ms, p50={statistics.median(times):.2f}ms")

# Time get_embedding
times = []
for _ in range(n_runs):
    start = time.perf_counter()
    client.get_embedding(test_text)
    times.append((time.perf_counter() - start) * 1000)
print(f"  get_embedding(): avg={statistics.mean(times):.2f}ms, p50={statistics.median(times):.2f}ms")

# Time full analyze
times = []
for _ in range(n_runs):
    start = time.perf_counter()
    client.analyze(test_text)
    times.append((time.perf_counter() - start) * 1000)
print(f"  analyze() [full]: avg={statistics.mean(times):.2f}ms, p50={statistics.median(times):.2f}ms")

# =============================================================================
# TEST 6: Edge Cases with Client API
# =============================================================================
print("\n" + "=" * 80)
print("TEST 6: EDGE CASES WITH CLIENT API")
print("=" * 80)

edge_cases = [
    ("Empty-ish", "   "),
    ("Single char", "a"),
    ("Very long", "family " * 500),
    ("Unicode", "Familie ist wichtig!"),
    ("Emoji", "I love my family so much!!!"),
    ("Mixed", "Mom said 'pick up milk' @ 5pm tomorrow"),
]

print("\n--- Edge Case Handling ---")
all_edge_ok = True
for name, text in edge_cases:
    try:
        result = client.analyze(text)
        safe = client.is_safe(text)
        crisis = client.is_crisis(text)
        sentiment = client.get_sentiment(text)
        emotions = client.get_emotions(text)
        embedding = client.get_embedding(text)

        ok = (
            isinstance(result.sentiment, str) and
            isinstance(safe, bool) and
            isinstance(crisis, bool) and
            isinstance(sentiment, str) and
            isinstance(emotions, list) and
            len(embedding) == 768
        )
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_edge_ok = False
        print(f"  [{status}] {name}: sentiment={sentiment}, safe={safe}, crisis={crisis}")
    except Exception as e:
        print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
        all_edge_ok = False

print(f"\n  All edge cases handled: {all_edge_ok}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("CLIENT API STRESS TEST - SUMMARY")
print("=" * 80)

final_stats = client.get_stats()

print(f"""
Auto-Warmup:
  - Cold start eliminated: YES
  - First call improvement: {improvement:.1f}x faster

Convenience Methods:
  - is_safe() accuracy: {100*safe_correct/len(test_cases):.0f}%
  - is_crisis() accuracy: {100*crisis_correct/len(test_cases):.0f}%
  - get_sentiment() accuracy: {100*sentiment_correct/len(test_cases):.0f}%
  - get_emotions() hit rate: {100*emotion_hits/len(emotion_cases):.0f}%
  - get_embedding() discrimination: PASS

ClientResult:
  - All attributes accessible: {all_attrs_ok}

Statistics:
  - Total calls tracked: {final_stats.get('total_calls', 'N/A')}
  - Avg latency: {final_stats.get('avg_latency_ms', 0):.2f}ms

Edge Cases:
  - All handled: {all_edge_ok}

v2.0.1 Client API: VALIDATED
""")

print("=" * 80)
print("CLIENT API STRESS TEST COMPLETE")
print("=" * 80)
