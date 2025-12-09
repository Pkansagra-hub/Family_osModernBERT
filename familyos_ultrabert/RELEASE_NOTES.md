# FamilyOS UltraBERT v2.0.0

High-performance multi-task NLP for family communication analysis.

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
| **Triplet Accuracy** (pos vs neg) | **98.80%** | Excellent |
| Mean Positive Similarity | 0.9179 | High cohesion |
| Mean Negative Similarity | 0.8247 | Good separation |
| Mean Margin | 0.0932 | Healthy gap |

### Retrieval Benchmarks (Search Quality)

| Benchmark | Metric | Score |
|-----------|--------|-------|
| **10 distractors** | Recall@1 | **74%** |
| **10 distractors** | Recall@5 | **99%** |
| **10 distractors** | Recall@10 | **100%** |
| **100 distractors** | Recall@1 | 36% |
| **100 distractors** | Recall@5 | 80% |
| **100 distractors** | Recall@10 | **89%** |

**Interpretation:** 89% Recall@10 with 100 candidates is excellent for memory search UI.

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

## Installation

```bash
# Download wheel and install with PyTorch
pip install familyos_ultrabert-2.0.0-py3-none-any.whl torch

# Or with ONNX for CPU
pip install familyos_ultrabert-2.0.0-py3-none-any.whl onnxruntime
```

## Quick Start

```python
from familyos_ultrabert import UltraBERT

model = UltraBERT.load()
result = model.analyze('Mom picked up Panda from school!')
print(result['sentiment'])  # very_positive
print(result['emotions'])   # ['joy']
print(result['safety_familyos'])  # GREEN
```

## License

Proprietary - All Rights Reserved
