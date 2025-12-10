Here's the current state and what we can add:

## Current Convenience Methods

### Client Methods (8 existing)
| Method | Description |
|--------|-------------|
| `get_sentiment(text)` | Quick sentiment: very_negative → very_positive |
| `get_emotions(text)` | Quick emotion list |
| `get_safety(text)` | Quick safety band: GREEN/AMBER/RED/CRISIS |
| `get_embedding(text)` | Get 768-dim vector |
| `is_safe(text)` | Returns True if GREEN |
| `is_crisis(text)` | Returns True if CRISIS |
| `health_check()` | Returns health status dict |
| `analyze_batch(texts)` | Analyze multiple texts |

### ClientResult Properties (20+ existing)
- `sentiment`, `sentiment_confidence`, `sentiment_scores`
- `emotions`, `emotion_scores`
- `safety`, `safety_confidence`, `safety_scores`, `is_safe`, `is_crisis`
- `entities`, `general_entities`, `temporal`
- `intent`, `intent_confidence`, `ingress`
- `relations`, `nli`, `embedding`, `embedding_dim`
- `to_dict()`, `latency_ms`

---

## Potential New Convenience Methods

### Client Methods (suggested additions)

| Method | Description | Priority |
|--------|-------------|----------|
| `get_intent(text)` | Quick intent classification | High |
| `get_ingress(text)` | Quick routing category | High |
| `get_entities(text)` | Quick family entity extraction | High |
| `get_temporal(text)` | Quick temporal expressions | Medium |
| `get_all_entities(text)` | Both family + general entities | Medium |
| `needs_attention(text)` | True if AMBER, RED, or CRISIS | High |
| `is_positive(text)` | True if positive/very_positive sentiment | Medium |
| `is_negative(text)` | True if negative/very_negative sentiment | Medium |
| `similarity(text1, text2)` | Cosine similarity between embeddings | High |
| `find_similar(query, corpus)` | Find most similar texts in corpus | High |
| `classify_batch(texts, capability)` | Batch single-capability | Medium |
| `embed_batch(texts)` | Batch embeddings (efficient) | High |
| `stream_analyze(texts)` | Generator for memory efficiency | Low |
| `export_embeddings(texts, path)` | Save embeddings to file | Low |

### ClientResult Properties (suggested additions)

| Property | Description | Priority |
|----------|-------------|----------|
| `needs_attention` | True if not GREEN | High |
| `top_emotion` | Highest confidence emotion | Medium |
| `sentiment_direction` | "positive"/"negative"/"neutral" | Medium |
| `has_entities` | True if any entities found | Medium |
| `entity_texts` | Just the text spans of entities | Medium |
| `to_json()` | JSON string output | Medium |
| `summary` | One-line summary string | Low |

Would you like me to implement any of these? I'd suggest starting with the **high priority** ones:
1. `get_intent()`, `get_ingress()`, `get_entities()`
2. `needs_attention()`, `is_positive()`, `is_negative()`
3. `similarity()`, `embed_batch()`, `find_similar()`
