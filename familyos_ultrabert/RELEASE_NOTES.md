## FamilyOS UltraBERT v2.0.0

High-performance multi-task NLP for family communication analysis.

### Model Performance

| Task | Metric | Score |
|------|--------|-------|
| emotions | Hit Rate | 88.30% |
| sentiment | Direction Accuracy | 88.10% |
| ner_family | F1 | 87.71% |
| safety_familyos | Accuracy | 96.20% |
| intent | Actionable Rate | 96.58% |
| ingress | Accuracy | 84.60% |
| relation | Micro-F1 | 84.83% |
| temporal | F1 | 87.17% |
| **Weighted Average** | | **89.60%** |

### Latency Benchmarks (RTX 4090)

**Single Capability:**

| Capability | Latency | P95 |
|------------|---------|-----|
| sentiment | 9.2 ms | 16.1 ms |
| emotions | 12.2 ms | 13.1 ms |
| safety_familyos | 8.6 ms | 9.4 ms |
| ner_family | 8.2 ms | 9.1 ms |
| intent | 7.9 ms | 8.6 ms |
| embedding | 7.9 ms | 8.6 ms |

**Multi-Capability (shared encoder):**

| Capabilities | Latency | P95 |
|--------------|---------|-----|
| 1 | 8.2 ms | 9.2 ms |
| 3 | 13.2 ms | 13.9 ms |
| 6 | 13.7 ms | 14.4 ms |
| **12 (all)** | **13.9 ms** | 15.0 ms |

**Embedding Performance:**

- Single query: 12.7 ms (embed + search 1K docs)
- Batch throughput: 1,921 embeddings/sec (batch=32)
- Corpus indexing: 1,020 docs/sec

**Batch Inference (12 caps):**

| Batch | Latency | Throughput |
|-------|---------|------------|
| 1 | 14 ms | 71 samples/sec |
| 8 | 50 ms | 161 samples/sec |
| 16 | 98 ms | 163 samples/sec |

### Features

- 12 NLP capabilities in one unified model
- PyTorch (GPU) and ONNX (CPU) inference backends
- 155M parameters, 15% magnitude pruned
- Single encoder pass for multi-capability inference

### Capabilities

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

### Installation

```bash
# Download wheel and install with PyTorch
pip install familyos_ultrabert-2.0.0-py3-none-any.whl torch

# Or with ONNX for CPU
pip install familyos_ultrabert-2.0.0-py3-none-any.whl onnxruntime
```

### Quick Start

```python
from familyos_ultrabert import UltraBERT

model = UltraBERT.load()
result = model.analyze('Mom picked up Panda from school!')
print(result['sentiment'])  # very_positive
print(result['emotions'])   # ['joy']
print(result['safety_familyos'])  # GREEN
```

### License

Proprietary - All Rights Reserved
