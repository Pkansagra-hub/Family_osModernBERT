# FamilyOS UltraBERT v2

**High-performance multi-task NLP model for family communication analysis.**

Built on ModernBERT architecture, UltraBERT delivers 12 NLP capabilities in a single unified model - extract sentiment, emotions, safety signals, entities, and more from text with state-of-the-art accuracy.

## Features

- **12 NLP Capabilities** in one unified model
- **PyTorch & ONNX** backends for flexible deployment
- **< 15ms latency** for 6 capabilities on GPU
- **Single encoder pass** for multi-capability inference
- **768-dim sentence embeddings** for semantic search
- **155M parameters** - Optimized ModernBERT architecture

## Installation

### From GitHub (recommended)

```bash
# Clone the full repository
git clone https://github.com/Pkansagra-hub/Family_osModernBERT.git
cd Family_osModernBERT

# Install the package with PyTorch backend
pip install ./familyos_ultrabert[pytorch]

# Or with ONNX backend
pip install ./familyos_ultrabert[onnx]

# Or both
pip install ./familyos_ultrabert[all]
```

### From PyPI (coming soon)

```bash
# Basic installation
pip install familyos-ultrabert

# With PyTorch backend (recommended for GPU)
pip install familyos-ultrabert[pytorch]

# With ONNX backend (recommended for CPU)
pip install familyos-ultrabert[onnx]

# Full installation (both backends)
pip install familyos-ultrabert[all]
```

> **Note**: The PyTorch backend requires the full repository for model architecture code.
> The ONNX backend is fully standalone.

## Quick Start

```python
from familyos_ultrabert import UltraBERT

# Load model (auto-selects best backend)
model = UltraBERT.load()

# Analyze text with multiple capabilities
result = model.analyze(
    "Mom picked up Panda from school today!",
    capabilities=["sentiment", "ner_family", "safety_familyos", "emotions"]
)

# Access results
print(result["sentiment"])
# {'prediction': 'positive', 'confidence': 0.89, 'scores': {...}}

print(result["ner_family"]["entities"])
# [{'text': 'Mom', 'label': 'KINSHIP'}, {'text': 'Panda', 'label': 'NICKNAME'}]

print(result["safety_familyos"])
# {'band': 'GREEN', 'confidence': 0.98, 'probabilities': {...}}

print(result["emotions"]["predictions"])
# ['joy', 'caring', 'togetherness']
```

## Capabilities

| Capability | Type | Description |
|------------|------|-------------|
| `sentiment` | Classification | 5-class sentiment (very_negative to very_positive) |
| `emotions` | Multi-label | 44 emotions including family-specific feelings |
| `safety_familyos` | Classification | Safety bands: GREEN, AMBER, RED, CRISIS |
| `safety_generic` | Multi-label | 8 toxicity types (Jigsaw-style) |
| `intent` | Classification | 8 user intents (log_memory, query_memory, etc.) |
| `ingress` | Classification | 12 domain categories (DIARY, TASK, HEALTH, etc.) |
| `ner_family` | Token | Family entities (KINSHIP, NICKNAME, PET, etc.) |
| `ner_general` | Token | General NER (PER, ORG, LOC, DATE, etc.) |
| `temporal` | Token | Temporal expressions (DATE_ABS, DATE_REL, DURATION) |
| `relation` | Multi-label | 15 relationship types (parent_of, spouse_of, etc.) |
| `nli` | Classification | Natural language inference |
| `embedding` | Vector | 768-dim sentence embeddings |

## Convenience Methods

```python
# Sentiment
sentiment = model.get_sentiment("I love this!")
print(sentiment["prediction"])  # "very_positive"

# Emotions
emotions = model.get_emotions("So excited for the trip!")
print(emotions)  # ["excitement", "joy", "anticipation"]

# Safety check
band = model.get_safety_band("Having a great day!")
print(band)  # "GREEN"

# Entity extraction
entities = model.get_entities("Mom and Dad took the kids to grandma's house")
print(entities)
# [{'text': 'Mom', 'label': 'KINSHIP'}, {'text': 'Dad', 'label': 'KINSHIP'}, ...]

# Embeddings
embedding = model.get_embedding("Sample text for embedding")
print(len(embedding))  # 768
```

## Backend Selection

```python
# Auto-detect (uses GPU if available)
model = UltraBERT.load()

# Force PyTorch on GPU (best for multi-capability)
model = UltraBERT.load(backend="pytorch", device="cuda")

# Force ONNX on CPU (best for single-capability, deployment)
model = UltraBERT.load(backend="onnx", device="cpu")

# Custom model path
model = UltraBERT.load(model_path="/path/to/weights")
```

## Model Architecture

| Component | Details |
|-----------|---------|
| Base Model | ModernBERT-base (22 layers, 768 hidden) |
| Parameters | 155M total |
| Encoder | Shared transformer backbone |
| Heads | 12 specialized task heads |
| Optimization | 15% magnitude pruning |
| Quantization | Dynamic INT8 (ONNX) |

## Benchmarks (RTX 4090)

### Running the built-in benchmark suites

Benchmarks ship with the package and can be run either as a module:

```bash
python -m familyos_ultrabert.benchmarks --suite api,regression
```

or via the console script (when installed):

```bash
ultrabert-benchmark --suite api,regression
```

Reporting formats:

```bash
python -m familyos_ultrabert.benchmarks --suite api,regression --format text
python -m familyos_ultrabert.benchmarks --suite api,regression --format json --output benchmark_report.json
python -m familyos_ultrabert.benchmarks --suite api,regression --format markdown --output benchmark_report.md
```

For a faster smoke run:

```bash
python -m familyos_ultrabert.benchmarks --quick
```

Standard profiles (recommended for CI):

```bash
python -m familyos_ultrabert.benchmarks --profile smoke
python -m familyos_ultrabert.benchmarks --profile full
```

Baseline drift tracking (compare against last-known-good per environment key and update baseline):

```bash
python -m familyos_ultrabert.benchmarks --profile smoke --baseline --format json --output benchmark_report.json
```

### Task Performance

| Task | Metric | Score |
|------|--------|-------|
| safety_familyos | Accuracy | 96.20% |
| intent | Actionable Rate | 96.58% |
| emotions | Hit Rate | 88.30% |
| sentiment | Direction Accuracy | 88.10% |
| ner_family | F1 | 87.71% |
| temporal | F1 | 87.17% |
| **Weighted Average** | | **89.60%** |

### Latency

| Scenario | Latency | Throughput |
|----------|---------|------------|
| 1 capability | 8 ms | - |
| 6 capabilities | 14 ms | - |
| 12 capabilities | 14 ms | 71 samples/sec |
| Embedding query | 13 ms | 1,921 emb/sec |

## License

Proprietary - All Rights Reserved. See [LICENSE](LICENSE) for details.
