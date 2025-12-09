# FamilyOS NLP

High-performance multi-task NLP model for family communication analysis. Extract sentiment, emotions, safety signals, entities, and more from text with a single model.

## Features

- **12 NLP Capabilities** in one unified model
- **PyTorch & ONNX** backends for flexible deployment
- **< 15ms latency** for 6 capabilities on GPU
- **Single encoder pass** for multi-capability inference
- **768-dim sentence embeddings** for semantic search

## Installation

### From GitHub (recommended)

```bash
# Clone the full repository
git clone https://github.com/your-org/Modeling_studio.git
cd Modeling_studio

# Install the package with PyTorch backend
pip install ./familyos_nlp[pytorch]

# Or with ONNX backend
pip install ./familyos_nlp[onnx]

# Or both
pip install ./familyos_nlp[all]
```

### From PyPI (coming soon)

```bash
# Basic installation (requires PyTorch or ONNX Runtime)
pip install familyos-nlp

# With PyTorch backend (recommended for GPU)
pip install familyos-nlp[pytorch]

# With ONNX backend (recommended for CPU)
pip install familyos-nlp[onnx]

# Full installation (both backends)
pip install familyos-nlp[all]
```

> **Note**: The PyTorch backend requires the full repository for model architecture code.
> The ONNX backend is fully standalone.

## Quick Start

```python
from familyos_nlp import FamilyOSModel

# Load model (auto-selects best backend)
model = FamilyOSModel.load()

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
model = FamilyOSModel.load()

# Force PyTorch on GPU (best for multi-capability)
model = FamilyOSModel.load(backend="pytorch", device="cuda")

# Force ONNX on CPU (best for single-capability, deployment)
model = FamilyOSModel.load(backend="onnx", device="cpu")

# Custom model path
model = FamilyOSModel.load(model_path="/path/to/weights")
```

## Performance

| Scenario | Backend | Latency |
|----------|---------|---------|
| 6 capabilities, GPU | PyTorch | ~15ms |
| 6 capabilities, CPU | ONNX | ~150ms |
| 1 capability, CPU | ONNX (quantized) | ~25ms |
| 12 capabilities, GPU | PyTorch | ~20ms |

## License

Apache 2.0
