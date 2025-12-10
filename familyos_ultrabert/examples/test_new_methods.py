"""Test all new convenience methods in v2.0.2."""

from familyos_ultrabert import Client
import familyos_ultrabert

print("=" * 60)
print(f"FamilyOS UltraBERT v{familyos_ultrabert.__version__}")
print("Testing NEW convenience methods")
print("=" * 60)

c = Client(warmup=True, warmup_rounds=2)

# Test text
text = "Mom picked up the kids from school yesterday!"

print("\n--- Client Methods ---")
print(f"get_intent: {c.get_intent(text)}")
print(f"get_ingress: {c.get_ingress(text)}")
print(f"get_entities: {c.get_entities(text)}")
print(f"get_temporal: {c.get_temporal(text)}")
print(f"needs_attention: {c.needs_attention(text)}")
print(f"is_positive: {c.is_positive(text)}")
print(f"is_negative: {c.is_negative(text)}")

# Similarity test
t1 = "I love my family"
t2 = "My family is wonderful"
t3 = "The weather is nice today"
print(f'\nsimilarity("{t1}", "{t2}"): {c.similarity(t1, t2):.4f}')
print(f'similarity("{t1}", "{t3}"): {c.similarity(t1, t3):.4f}')

# Find similar
corpus = ["I love spending time with my family", "Work is stressful today", "My kids are amazing"]
print(f'\nfind_similar("Family time is great", corpus):')
for r in c.find_similar("Family time is great", corpus, top_k=3):
    print(f'  {r["similarity"]:.4f} - {r["text"]}')

# embed_batch
print(f"\nembed_batch([3 texts]): {len(c.embed_batch([t1, t2, t3]))} embeddings")

# classify_batch
print(f'classify_batch([3 texts], "sentiment"): {c.classify_batch([t1, t2, t3], "sentiment")}')

# get_all_entities
print(f"\nget_all_entities: {c.get_all_entities(text)}")

print("\n--- ClientResult Properties ---")
result = c.analyze(text)
print(f"needs_attention: {result.needs_attention}")
print(f"top_emotion: {result.top_emotion}")
print(f"sentiment_direction: {result.sentiment_direction}")
print(f"has_entities: {result.has_entities}")
print(f"entity_texts: {result.entity_texts}")
print(f"summary: {result.summary}")
print(f"to_json() length: {len(result.to_json())} chars")

print("\n" + "=" * 60)
print("ALL NEW METHODS WORKING!")
print("=" * 60)
