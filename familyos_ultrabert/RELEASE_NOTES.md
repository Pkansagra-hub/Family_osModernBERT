# FamilyOS UltraBERT v2.0.2

## Production-Ready Family NLP with Extended Convenience API

High-performance multi-task NLP for family communication analysis. Now with 20+ convenience methods and embedding utilities.

---

## What's New in v2.0.2

### Extended Convenience Methods

**New Client Methods (14 added):**

| Method | Description |
|--------|-------------|
| `get_intent(text)` | Quick intent classification |
| `get_ingress(text)` | Quick routing category |
| `get_entities(text)` | Family entity extraction |
| `get_temporal(text)` | Temporal expressions |
| `get_all_entities(text)` | Both family + general entities |
| `needs_attention(text)` | True if AMBER/RED/CRISIS |
| `is_positive(text)` | True if positive sentiment |
| `is_negative(text)` | True if negative sentiment |
| `similarity(text1, text2)` | Cosine similarity (0-1) |
| `find_similar(query, corpus)` | Find most similar texts |
| `embed_batch(texts)` | Batch embeddings |
| `classify_batch(texts, capability)` | Batch classification |
| `stream_analyze(texts)` | Generator for memory efficiency |
| `export_embeddings(texts, path)` | Export to JSONL/CSV |

**New ClientResult Properties (7 added):**

| Property | Description |
|----------|-------------|
| `needs_attention` | True if not GREEN |
| `top_emotion` | Highest confidence emotion |
| `sentiment_direction` | "positive"/"negative"/"neutral" |
| `has_entities` | True if any entities found |
| `entity_texts` | Just the text spans |
| `to_json()` | JSON string output |
| `summary` | One-line summary |

### Example Usage

```python
from familyos_ultrabert import Client

client = Client()

# Semantic similarity
sim = client.similarity("I love my family", "My family is great")
print(sim)  # 0.92

# Find similar in corpus
corpus = ["Family dinner was fun", "Work meeting tomorrow", "Kids played outside"]
matches = client.find_similar("Great family day", corpus, top_k=2)
# [{"text": "Family dinner was fun", "similarity": 0.91}, ...]

# Batch operations
sentiments = client.classify_batch(texts, "sentiment")
embeddings = client.embed_batch(texts)

# Result properties
result = client.analyze("Mom picked up the kids!")
print(result.summary)           # "safety=GREEN | sentiment=positive | emotions=['joy']"
print(result.top_emotion)       # "joy"
print(result.sentiment_direction)  # "positive"
```

---

## What's in v2.0.1

### Client API with Auto-Warmup

- **Zero cold-start latency**: First user call is fast (~17ms instead of 285ms)
- **Convenience methods**: `is_safe()`, `is_crisis()`, `get_sentiment()`, `get_emotions()`
- **Built-in latency tracking**: `client.stats` provides real-time metrics
- **Health check endpoint**: `client.health_check()` for production monitoring
- **Clean result wrapper**: `ClientResult` with easy attribute access

```python
from familyos_ultrabert import Client

client = Client()  # Auto warmup happens here
result = client.analyze("Mom picked up the kids!")

print(result.sentiment)      # "very_positive"
print(result.safety)         # "GREEN"
print(result.emotions)       # ["joy", "love", "excitement"]
print(result.latency_ms)     # 7.5

# Convenience methods
client.is_safe("I love my family")       # True
client.is_crisis("I want to hurt myself") # True
```

### Performance Improvements in v2.0.1

| Scenario | v2.0.0 | v2.0.1 | Improvement |
|----------|--------|--------|-------------|
| First user call | ~285ms | **~17ms** | **16.8x faster** |
| Convenience methods | N/A | **~12ms** | New feature |
| Production monitoring | Manual | **Built-in** | Zero config |
| Warmup required | Manual | **Automatic** | Better UX |

### Version Comparison

| Feature | v2.0.0 | v2.0.1 |
|---------|--------|--------|
| Model accuracy | 89.60% | 89.60% |
| First call latency | ~285ms | **~17ms** |
| Convenience methods | No | Yes |
| Auto-warmup | No | Yes |
| Health monitoring | No | Yes |
| Latency tracking | No | Yes |
| API style | Dict-based | Object-based |
| Backward compatible | - | Yes |

### Backward Compatibility

- **Both APIs work**: `UltraBERT.load()` and `Client()` are both supported
- **Same model weights**: All v2.0.0 benchmarks apply to v2.0.1
- **Drop-in replacement**: Update from v2.0.0 to v2.0.1 with no code changes
- **Legacy API supported**: Dictionary-style access continues to work

---

## Installation

```bash
# Download wheel and install with PyTorch (GPU)
pip install familyos_ultrabert-2.0.1-py3-none-any.whl torch

# Or with ONNX for CPU
pip install familyos_ultrabert-2.0.1-py3-none-any.whl onnxruntime
```

---

## Quick Start

### New Client API (v2.0.1 - Recommended)

```python
from familyos_ultrabert import Client

client = Client()  # Auto-warmup happens here
result = client.analyze('Mom picked up Panda from school!')
print(result.sentiment)      # very_positive (attribute access)
print(result.safety)         # GREEN
print(result.latency_ms)     # 12.5
```

### Legacy API (v2.0.0 style, still works)

```python
from familyos_ultrabert import UltraBERT

model = UltraBERT.load()
result = model.analyze('Mom picked up Panda from school!')
print(result['sentiment'])   # dictionary style
print(result['emotions'])    # ['joy']
print(result['safety_familyos'])  # GREEN
```

---

## Migrating from v2.0.0 to v2.0.1

### Option 1: Minimal changes (backward compatible)

```python
# Old code works exactly the same
from familyos_ultrabert import UltraBERT
model = UltraBERT.load()
result = model.analyze(text)
```

### Option 2: Upgrade to new Client API (recommended)

```python
# Before (v2.0.0):
from familyos_ultrabert import UltraBERT
model = UltraBERT.load()
result = model.analyze(text)
is_safe = result['safety_familyos'] == 'GREEN'

# After (v2.0.1):
from familyos_ultrabert import Client
client = Client()  # Auto-warms up
result = client.analyze(text)
is_safe = client.is_safe(text)  # Cleaner!
```

---

## Production Deployment

### Web Server Example (FastAPI)

```python
from fastapi import FastAPI
from familyos_ultrabert import Client

app = FastAPI()
client = Client()  # Starts warmup on server startup

@app.post("/analyze")
def analyze_text(text: str):
    result = client.analyze(text)
    return {
        "sentiment": result.sentiment,
        "safety": result.safety,
        "latency_ms": result.latency_ms
    }

@app.get("/health")
def health_check():
    return client.health_check()
```

---

## Task Performance

| Task | Metric | Score |
|------|--------|-------|
| safety_familyos | Accuracy | **96.20%** |
| intent | Actionable Rate | **96.58%** |
| emotions | Hit Rate | 88.30% |
| sentiment | Direction Accuracy | 88.10% |
| ner_family | F1 | 87.71% |
| temporal | F1 | 87.17% |
| ingress | Accuracy | 84.60% |
| relation | Micro-F1 | 84.83% |
| **Weighted Average** | | **89.60%** |

---

## Embedding Quality Benchmarks

### Triplet Accuracy

| Metric | Value | Assessment |
|--------|-------|------------|
| **Triplet Accuracy** (pos vs neg) | **98.53%** | Excellent |
| Mean Positive Similarity | 0.9159 | High cohesion |
| Mean Negative Similarity | 0.8258 | Good separation |
| Mean Margin | 0.0901 | Healthy gap |

### Retrieval Benchmarks (Search Quality)

| Benchmark | Metric | Score |
|-----------|--------|-------|
| **10 distractors** | Recall@1 | **79.60%** |
| **100 distractors** | Recall@1 | 33.70% |
| **100 distractors** | Recall@5 | 78.73% |
| **100 distractors** | Recall@10 | **92.10%** |

**Interpretation:** 92% Recall@10 with 100 candidates is excellent for memory search UI.

---

## Inference Latency Benchmarks (RTX 4090)

### Full Multi-Task Inference (12 Capabilities)

| Metric | Value |
|--------|-------|
| **Average** | **29.60 ms** |
| **P50** | 27.53 ms |
| **P95** | 44.29 ms |
| **Min** | 21.62 ms |
| **Throughput** | **33.8 inferences/sec** |

### Per-Capability Latency

| Capability | Latency |
|------------|---------|
| ner_family | 15.5 ms |
| safety_familyos | 16.0 ms |
| temporal | 16.1 ms |
| intent | 16.7 ms |
| nli | 17.0 ms |
| relation | 17.1 ms |
| ingress | 17.2 ms |
| embedding | 17.2 ms |
| safety_generic | 18.6 ms |
| sentiment | 19.2 ms |
| ner_general | 21.0 ms |
| emotions | 25.3 ms |

---

## Embedding Query Performance

### Corpus Indexing

| Metric | Value |
|--------|-------|
| **Embedding Throughput** | **987 docs/sec** |
| **Embedding Dimension** | 768 |

### Query Latency (1000 doc corpus)

| Metric | Value |
|--------|-------|
| **Average (embed + search)** | **12.67 ms** |
| **P50** | 12.09 ms |
| **P95** | 17.34 ms |

### Latency Breakdown

| Component | Time | % |
|-----------|------|---|
| Query Embedding | 11.77 ms | 97% |
| Search (1000 docs) | 0.32 ms | 3% |

### Search Scaling (Pre-computed Embeddings)

| Corpus Size | Search Time |
|-------------|-------------|
| 100 docs | 0.09 ms |
| 500 docs | 0.09 ms |
| 1000 docs | 0.13 ms |

---

## Sample Inference Output

```json
{
  "text": "My grandmother called yesterday to remind me about the family reunion next Sunday. I am so excited!",
  "emotions": ["joy", "excitement", "togetherness"],
  "sentiment": "very_positive",
  "safety": "GREEN",
  "intent": "share_news",
  "ingress": "CELEBRATION",
  "entities": [
    {"text": "grandmother", "label": "KINSHIP"},
    {"text": "family reunion", "label": "FAMILY_EVENT"}
  ],
  "temporal": [
    {"text": "yesterday", "label": "DATE_REL"},
    {"text": "next Sunday", "label": "DATE_REL"}
  ],
  "embedding_dim": 768,
  "inference_time_ms": 13.64
}
```

---

## Features

- 12 NLP capabilities in one unified model
- PyTorch (GPU) and ONNX (CPU) inference backends
- 155M parameters, 15% magnitude pruned
- Single encoder pass for multi-capability inference

## Capabilities

| Capability | Type | Description |
|------------|------|-------------|
| sentiment | Classification | 5-class sentiment |
| emotions | Multi-label | 44 emotion labels |
| safety_familyos | Classification | GREEN/AMBER/RED/CRISIS |
| safety_generic | Multi-label | Toxicity detection |
| ner_family | Token | Family member NER |
| ner_general | Token | General NER |
| temporal | Token | Date/time extraction |
| intent | Classification | User intent |
| ingress | Classification | Message routing |
| relation | Multi-label | Relationship types |
| nli | Classification | Natural language inference |
| embedding | Vector | 768-dim embeddings |

---

## License

Proprietary - All Rights Reserved
