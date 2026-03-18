"""End-to-end inference test for familyos_ultrabert release package."""

from familyos_ultrabert import Client
import time

WEIGHTS = "D:/Modeling_studio/familyos_ultrabert/weights/pytorch"
print(f"=== Loading Client (PyTorch) from {WEIGHTS} ===")
t0 = time.time()
client = Client(backend="pytorch", model_path=WEIGHTS)
print(f"Loaded in {time.time()-t0:.1f}s")
print()

# --- 1. Basic text analysis ---
print("=== Test 1: Basic analyze() ===")
t0 = time.time()
r = client.analyze("Mom picked up Anya from school at 3pm today.")
ms = (time.time() - t0) * 1000
print(f"  Latency: {ms:.0f}ms")
print(f"  Sentiment: {r.sentiment}")
print(f"  Emotions:  {r.emotions[:3]}")
print(f"  Intent:    {r.intent}")
print(f"  Safety:    {r.safety}")
if r.entities:
    print(f"  Entities:  {r.entities[:5]}")
if r.temporal:
    print(f"  Temporal:  {r.temporal[:3]}")
print()

# --- 2. Relevance scoring (MGRH) ---
print("=== Test 2: score_relevance() ===")
query = "Who picked up the kids from school?"
pairs = [
    ("GOOD", "Mom picked up Anya and Rohan from Lincoln Elementary at 3pm."),
    ("BAD", "Dad went to the grocery store to buy milk."),
    ("WRONG_PERSON", "Uncle picked up the neighbor kids from school at 3pm."),
    ("WRONG_TIME", "Mom picked up Anya from school last Tuesday morning."),
]

for label, doc in pairs:
    t0 = time.time()
    result = client.score_relevance(query, doc)
    ms = (time.time() - t0) * 1000
    score = result["score"]
    nli = result.get("nli_label", "?")
    print(f"  [{label:13s}] score={score:.4f}  nli={nli:14s}  latency={ms:.0f}ms")
    print(f"                 doc: {doc}")
print()

# --- 3. Reranking ---
print("=== Test 3: rerank() ===")
query2 = "What did we do for grandma birthday?"
docs = [
    "We threw a surprise party for grandma with the whole family.",
    "Dad went to work early and came back late.",
    "It rained all day so we stayed inside.",
    "Grandma loved the homemade cake Anya baked for her.",
    "The dog chewed up the newspaper again.",
]
t0 = time.time()
ranked = client.rerank(query2, docs, top_k=3)
ms = (time.time() - t0) * 1000
print(f"  Rerank latency: {ms:.0f}ms")
for i, item in enumerate(ranked):
    score = item["score"]
    text = item["text"]
    print(f"  #{i+1} score={score:.4f} | {text}")
print()

# --- 4. Embedding ---
print("=== Test 4: get_embedding() ===")
t0 = time.time()
emb = client.get_embedding("Mom picked up the kids from school")
ms = (time.time() - t0) * 1000
print(f"  Embedding dim: {len(emb)}")
print(f"  First 5 values: {[round(v, 4) for v in emb[:5]]}")
print(f"  Latency: {ms:.0f}ms")
print()

# --- 5. Sanity checks ---
print("=== Sanity Checks ===")
scores = {}
for label, doc in pairs:
    result = client.score_relevance(query, doc)
    scores[label] = result["score"]

# Ordering check: GOOD > BAD, and GOOD > WRONG_PERSON
assert scores["GOOD"] > scores["BAD"], (
    f"GOOD should rank above BAD: {scores['GOOD']:.4f} vs {scores['BAD']:.4f}"
)
assert scores["GOOD"] > scores["WRONG_PERSON"], (
    f"GOOD should rank above WRONG_PERSON: {scores['GOOD']:.4f} vs {scores['WRONG_PERSON']:.4f}"
)
assert scores["GOOD"] > scores["WRONG_TIME"], (
    f"GOOD should rank above WRONG_TIME: {scores['GOOD']:.4f} vs {scores['WRONG_TIME']:.4f}"
)
print(f"  Pointwise ordering: GOOD({scores['GOOD']:.4f}) > WRONG_TIME({scores['WRONG_TIME']:.4f}) "
      f"> WRONG_PERSON({scores['WRONG_PERSON']:.4f}) > BAD({scores['BAD']:.4f})")

# Rerank: top doc should be about grandma/party
top_doc = ranked[0]["text"]
assert "grandma" in top_doc.lower() or "party" in top_doc.lower(), f"Top doc wrong: {top_doc}"
assert ranked[0]["score"] > ranked[1]["score"], "Rerank ordering broken"
print(f"  Rerank top doc: {ranked[0]['score']:.4f} - mentions grandma/party: OK")

# Rerank margin: top should be well-separated from #2
margin = ranked[0]["score"] - ranked[1]["score"]
assert margin > 0.1, f"Rerank margin too small: {margin:.4f}"
print(f"  Rerank margin (top vs #2): {margin:.4f}")

# Embedding sanity: should be 768-dim unit vector
import math
norm = math.sqrt(sum(v * v for v in emb))
assert len(emb) == 768, f"Wrong dim: {len(emb)}"
assert 0.95 < norm < 1.05, f"Not unit vector: norm={norm}"
print(f"  Embedding: 768-dim, norm={norm:.4f}")
print()

print("=== ALL TESTS PASSED ===")
