# FamilyOS UltraBERT API Reference

Complete developer documentation for the FamilyOS UltraBERT Python SDK.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Client API (Recommended)](#client-api-recommended)
  - [Client Class](#client-class)
  - [ClientResult Class](#clientresult-class)
  - [Convenience Methods](#convenience-methods)
  - [Statistics & Monitoring](#statistics--monitoring)
- [Legacy API](#legacy-api)
  - [UltraBERT Class](#ultrabert-class)
  - [AnalysisOutput Class](#analysisoutput-class)
- [Capabilities Reference](#capabilities-reference)
- [Label Reference](#label-reference)
- [Error Handling](#error-handling)
- [Performance Tips](#performance-tips)
- [Examples](#examples)

---

## Installation

```bash
# GPU (PyTorch + CUDA)
pip install familyos_ultrabert-2.0.1-py3-none-any.whl torch

# CPU only (ONNX Runtime)
pip install familyos_ultrabert-2.0.1-py3-none-any.whl onnxruntime
```

---

## Quick Start

```python
from familyos_ultrabert import Client

# Initialize with auto-warmup
client = Client()

# Analyze text
result = client.analyze("Mom picked up Panda from school!")

# Access results
print(result.sentiment)      # "very_positive"
print(result.safety)         # "GREEN"
print(result.emotions)       # ["joy"]
print(result.latency_ms)     # 12.5
```

---

## Client API (Recommended)

### Client Class

The `Client` class is the recommended way to use UltraBERT. It provides auto-warmup, convenience methods, and built-in telemetry.

#### Constructor

```python
Client(
    backend: str = "auto",
    warmup: bool = True,
    warmup_rounds: int = 3
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `str` | `"auto"` | Inference backend: `"auto"`, `"pytorch"`, or `"onnx"` |
| `warmup` | `bool` | `True` | Whether to run warmup on initialization |
| `warmup_rounds` | `int` | `3` | Number of warmup iterations |

**Backend Selection:**

- `"auto"`: Uses PyTorch if CUDA available, otherwise ONNX
- `"pytorch"`: Force PyTorch backend (requires GPU for best performance)
- `"onnx"`: Force ONNX backend (optimized for CPU)

#### Example

```python
from familyos_ultrabert import Client

# Auto-select backend with warmup (recommended)
client = Client()

# Force ONNX for CPU deployment
client = Client(backend="onnx")

# Skip warmup (not recommended for production)
client = Client(warmup=False)

# Custom warmup rounds
client = Client(warmup_rounds=5)
```

---

### Client.analyze()

Analyze text and return all capabilities.

```python
Client.analyze(
    text: str,
    capabilities: list[str] | None = None
) -> ClientResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | Required | Text to analyze (max ~512 tokens) |
| `capabilities` | `list[str]` | `None` | Specific capabilities to run. If `None`, runs all. |

#### Returns

`ClientResult` object with all analysis results.

#### Example

```python
# Analyze with all capabilities
result = client.analyze("I love spending time with my grandmother!")

# Analyze with specific capabilities only (faster)
result = client.analyze(
    "I love spending time with my grandmother!",
    capabilities=["sentiment", "safety_familyos", "emotions"]
)
```

---

### ClientResult Class

Result object returned by `Client.analyze()`.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Original input text |
| `sentiment` | `str` | Sentiment label |
| `sentiment_score` | `float` | Sentiment confidence (0-1) |
| `emotions` | `list[str]` | Detected emotions |
| `emotion_scores` | `dict[str, float]` | Emotion label -> confidence |
| `safety` | `str` | Safety level: GREEN, AMBER, RED, or CRISIS |
| `safety_score` | `float` | Safety confidence (0-1) |
| `intent` | `str` | Detected user intent |
| `intent_score` | `float` | Intent confidence (0-1) |
| `ingress` | `str` | Message routing category |
| `ingress_score` | `float` | Ingress confidence (0-1) |
| `entities` | `list[dict]` | Family NER entities |
| `general_entities` | `list[dict]` | General NER entities |
| `temporal` | `list[dict]` | Temporal expressions |
| `relations` | `list[str]` | Relationship types |
| `relation_scores` | `dict[str, float]` | Relation label -> confidence |
| `embedding` | `list[float]` | 768-dim embedding vector |
| `latency_ms` | `float` | Inference time in milliseconds |

#### Entity Format

```python
{
    "text": "grandmother",
    "label": "KINSHIP",
    "start": 35,
    "end": 46
}
```

#### Example

```python
result = client.analyze("Mom and Dad are coming over tomorrow!")

# Sentiment
print(result.sentiment)         # "very_positive"
print(result.sentiment_score)   # 0.92

# Emotions
print(result.emotions)          # ["joy", "anticipation"]
print(result.emotion_scores)    # {"joy": 0.85, "anticipation": 0.72, ...}

# Safety
print(result.safety)            # "GREEN"
print(result.safety_score)      # 0.99

# Named entities
for entity in result.entities:
    print(f"{entity['text']} -> {entity['label']}")
# Mom -> KINSHIP
# Dad -> KINSHIP

# Temporal
for temp in result.temporal:
    print(f"{temp['text']} -> {temp['label']}")
# tomorrow -> DATE_REL

# Embedding (for similarity/search)
print(len(result.embedding))    # 768
print(result.embedding[:5])     # [0.023, -0.145, ...]

# Latency
print(result.latency_ms)        # 12.5
```

---

### Convenience Methods

Quick boolean/value checks without full analysis overhead.

#### Client.is_safe()

Check if text is safe (GREEN level).

```python
Client.is_safe(text: str) -> bool
```

```python
client.is_safe("I love my family")           # True
client.is_safe("I hate everyone")            # False (AMBER)
client.is_safe("I want to hurt myself")      # False (CRISIS)
```

#### Client.is_crisis()

Check if text indicates crisis.

```python
Client.is_crisis(text: str) -> bool
```

```python
client.is_crisis("Having a great day!")      # False
client.is_crisis("I can't go on anymore")    # True
```

#### Client.get_sentiment()

Get sentiment label only.

```python
Client.get_sentiment(text: str) -> str
```

```python
client.get_sentiment("This is amazing!")     # "very_positive"
client.get_sentiment("It's okay")            # "neutral"
client.get_sentiment("I'm disappointed")     # "negative"
```

#### Client.get_emotions()

Get list of detected emotions.

```python
Client.get_emotions(text: str) -> list[str]
```

```python
client.get_emotions("I'm so happy!")         # ["joy", "excitement"]
client.get_emotions("I'm worried about him") # ["concern", "anxiety"]
```

#### Client.get_embedding()

Get embedding vector for similarity search.

```python
Client.get_embedding(text: str) -> list[float]
```

```python
embedding = client.get_embedding("Family dinner tonight")
print(len(embedding))  # 768

# Use for similarity search
import numpy as np
query_emb = np.array(client.get_embedding("dinner plans"))
doc_emb = np.array(client.get_embedding("Family dinner tonight"))
similarity = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
print(similarity)  # 0.89
```

---

### Statistics & Monitoring

#### Client.health_check()

Verify model is working correctly.

```python
Client.health_check() -> dict
```

```python
health = client.health_check()
print(health)
# {
#     "status": "healthy",
#     "backend": "pytorch",
#     "device": "cuda",
#     "model_loaded": True,
#     "latency_ms": 12.5
# }
```

#### Client.get_stats()

Get latency statistics.

```python
Client.get_stats() -> dict
```

```python
stats = client.get_stats()
print(stats)
# {
#     "total_calls": 150,
#     "avg_latency_ms": 12.3,
#     "min_latency_ms": 10.1,
#     "max_latency_ms": 18.5,
#     "p50_latency_ms": 11.8,
#     "p95_latency_ms": 15.2,
#     "p99_latency_ms": 17.1
# }
```

#### Client.reset_stats()

Reset latency statistics.

```python
Client.reset_stats() -> None
```

```python
client.reset_stats()
print(client.get_stats()["total_calls"])  # 0
```

---

## Legacy API

For backward compatibility with v2.0.0.

### UltraBERT Class

#### UltraBERT.load()

Load the model.

```python
UltraBERT.load(backend: str = "auto") -> UltraBERT
```

```python
from familyos_ultrabert import UltraBERT

model = UltraBERT.load()
model = UltraBERT.load(backend="onnx")
```

#### UltraBERT.analyze()

Analyze text.

```python
UltraBERT.analyze(
    text: str,
    capabilities: list[str] | None = None
) -> AnalysisOutput
```

```python
result = model.analyze("Hello family!")

# Dictionary-style access
print(result['sentiment'])       # "positive"
print(result['safety_familyos']) # "GREEN"
print(result['emotions'])        # ["joy"]
```

### AnalysisOutput Class

Result object with dictionary-style access.

```python
result = model.analyze("Test message")

# Access by capability
result['sentiment']           # Sentiment result dict
result['emotions']            # Emotions result dict
result['safety_familyos']     # Safety result dict

# Get specific values
result.get_label('sentiment')           # "positive"
result.get_confidence('sentiment')      # 0.85
result.get_labels('emotions')           # ["joy", "love"]
result.get_entities('ner_family')       # [{"text": "Mom", "label": "KINSHIP"}]
```

---

## Capabilities Reference

| Capability | Type | Output | Description |
|------------|------|--------|-------------|
| `sentiment` | Classification | 5 classes | very_negative, negative, neutral, positive, very_positive |
| `emotions` | Multi-label | 44 labels | Plutchik + compound emotions |
| `safety_familyos` | Classification | 4 classes | GREEN, AMBER, RED, CRISIS |
| `safety_generic` | Multi-label | 6 labels | Toxicity categories |
| `intent` | Classification | 15 classes | User intent detection |
| `ingress` | Classification | 8 classes | Message routing |
| `ner_family` | Token | BIO tags | Family member NER |
| `ner_general` | Token | BIO tags | General NER |
| `temporal` | Token | BIO tags | Date/time extraction |
| `relation` | Multi-label | 12 labels | Relationship types |
| `nli` | Classification | 3 classes | Entailment, neutral, contradiction |
| `embedding` | Vector | 768-dim | Sentence embeddings |

---

## Label Reference

### Sentiment Labels

| Label | Description |
|-------|-------------|
| `very_negative` | Strong negative sentiment |
| `negative` | Negative sentiment |
| `neutral` | Neutral/objective |
| `positive` | Positive sentiment |
| `very_positive` | Strong positive sentiment |

### Safety Labels

| Label | Description | Action |
|-------|-------------|--------|
| `GREEN` | Safe content | None required |
| `AMBER` | Potentially concerning | Monitor |
| `RED` | Harmful content | Review |
| `CRISIS` | Immediate danger | Escalate immediately |

### Intent Labels

| Label | Description |
|-------|-------------|
| `share_news` | Sharing information |
| `ask_question` | Asking a question |
| `make_request` | Making a request |
| `express_emotion` | Expressing feelings |
| `schedule` | Scheduling/planning |
| `remind` | Reminder |
| `thank` | Expressing gratitude |
| `apologize` | Apologizing |
| `greet` | Greeting |
| `farewell` | Saying goodbye |
| `affirm` | Agreement/confirmation |
| `deny` | Disagreement/denial |
| `complain` | Complaining |
| `praise` | Praising |
| `other` | Other intent |

### Ingress Labels

| Label | Description |
|-------|-------------|
| `CELEBRATION` | Celebrations, good news |
| `COORDINATION` | Logistics, scheduling |
| `SUPPORT` | Emotional support needed |
| `CONFLICT` | Conflict, disagreement |
| `HEALTH` | Health-related |
| `FINANCE` | Financial matters |
| `EDUCATION` | Education-related |
| `GENERAL` | General conversation |

### Family NER Labels

| Label | Description |
|-------|-------------|
| `KINSHIP` | Family relationship terms (mom, dad, sister) |
| `FAMILY_MEMBER` | Named family members |
| `FAMILY_EVENT` | Family events (reunion, birthday) |
| `FAMILY_ROLE` | Family roles (caregiver, breadwinner) |

### Emotion Labels (Top 20)

| Label | Description |
|-------|-------------|
| `joy` | Happiness, delight |
| `love` | Love, affection |
| `gratitude` | Thankfulness |
| `excitement` | Enthusiasm |
| `pride` | Pride, accomplishment |
| `contentment` | Satisfaction |
| `hope` | Optimism |
| `amusement` | Fun, humor |
| `sadness` | Sadness, sorrow |
| `anger` | Anger, frustration |
| `fear` | Fear, anxiety |
| `disgust` | Disgust, revulsion |
| `surprise` | Surprise |
| `confusion` | Confusion |
| `disappointment` | Disappointment |
| `guilt` | Guilt, remorse |
| `shame` | Shame |
| `jealousy` | Jealousy, envy |
| `loneliness` | Loneliness |
| `nostalgia` | Nostalgia |

---

## Error Handling

```python
from familyos_ultrabert import Client
from familyos_ultrabert.exceptions import (
    ModelNotLoadedError,
    InvalidInputError,
    InferenceError
)

client = Client()

try:
    result = client.analyze(text)
except InvalidInputError as e:
    # Empty or invalid input
    print(f"Invalid input: {e}")
except InferenceError as e:
    # Model inference failed
    print(f"Inference error: {e}")
except Exception as e:
    # Unexpected error
    print(f"Unexpected error: {e}")
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `InvalidInputError` | Empty or None text | Validate input before calling |
| `InferenceError` | Model failure | Check GPU memory, restart |
| `ModelNotLoadedError` | Model not initialized | Call `Client()` first |
| `CUDA out of memory` | GPU memory exhausted | Reduce batch size, use ONNX |

---

## Performance Tips

### 1. Use Auto-Warmup (Default)

```python
# Good - warmup eliminates cold start
client = Client()  # 3 warmup rounds

# Bad - first call will be slow
client = Client(warmup=False)
```

### 2. Reuse Client Instance

```python
# Good - single instance
client = Client()
for text in texts:
    result = client.analyze(text)

# Bad - recreating client
for text in texts:
    client = Client()  # Slow!
    result = client.analyze(text)
```

### 3. Use Specific Capabilities

```python
# Good - only run what you need
result = client.analyze(text, capabilities=["safety_familyos"])

# Slower - runs all 12 capabilities
result = client.analyze(text)
```

### 4. Use Convenience Methods for Single Checks

```python
# Good - optimized for single capability
is_safe = client.is_safe(text)

# Slower - runs all capabilities
result = client.analyze(text)
is_safe = result.safety == "GREEN"
```

### 5. Batch Processing (Future)

```python
# Coming in v2.1.0
results = client.analyze_batch(texts)
```

### 6. Choose Right Backend

```python
# GPU available - use PyTorch (faster)
client = Client(backend="pytorch")

# CPU only - use ONNX (optimized)
client = Client(backend="onnx")
```

---

## Examples

### Content Moderation

```python
from familyos_ultrabert import Client

client = Client()

def moderate_message(text: str) -> dict:
    """Check if message is safe to post."""
    result = client.analyze(text, capabilities=["safety_familyos", "safety_generic"])

    return {
        "allowed": result.safety == "GREEN",
        "safety_level": result.safety,
        "reason": "Content flagged" if result.safety != "GREEN" else None
    }

# Usage
check = moderate_message("I love my family!")
print(check)  # {"allowed": True, "safety_level": "GREEN", "reason": None}

check = moderate_message("I hate everyone")
print(check)  # {"allowed": False, "safety_level": "AMBER", "reason": "Content flagged"}
```

### Sentiment Dashboard

```python
from familyos_ultrabert import Client
from collections import Counter

client = Client()

def analyze_sentiment_distribution(messages: list[str]) -> dict:
    """Analyze sentiment distribution across messages."""
    sentiments = []
    for msg in messages:
        sentiments.append(client.get_sentiment(msg))

    counts = Counter(sentiments)
    total = len(messages)

    return {
        label: {
            "count": count,
            "percentage": round(count / total * 100, 1)
        }
        for label, count in counts.items()
    }

# Usage
messages = [
    "Great day with the family!",
    "Feeling okay today",
    "Stressed about work",
    "So happy to see grandma!"
]
print(analyze_sentiment_distribution(messages))
```

### Crisis Detection Pipeline

```python
from familyos_ultrabert import Client

client = Client()

def crisis_pipeline(text: str) -> dict:
    """Detect and triage crisis messages."""
    result = client.analyze(text, capabilities=["safety_familyos", "emotions", "intent"])

    response = {
        "is_crisis": result.safety == "CRISIS",
        "safety_level": result.safety,
        "emotions": result.emotions,
        "intent": result.intent,
        "action": None
    }

    if result.safety == "CRISIS":
        response["action"] = "ESCALATE_IMMEDIATELY"
    elif result.safety == "RED":
        response["action"] = "REVIEW_REQUIRED"
    elif result.safety == "AMBER":
        response["action"] = "MONITOR"
    else:
        response["action"] = "NONE"

    return response

# Usage
result = crisis_pipeline("I can't take this anymore")
print(result)
# {
#     "is_crisis": True,
#     "safety_level": "CRISIS",
#     "emotions": ["despair", "hopelessness"],
#     "intent": "express_emotion",
#     "action": "ESCALATE_IMMEDIATELY"
# }
```

### Semantic Search

```python
from familyos_ultrabert import Client
import numpy as np

client = Client()

class MemorySearch:
    def __init__(self):
        self.memories = []
        self.embeddings = []

    def add(self, text: str):
        """Add memory to index."""
        embedding = client.get_embedding(text)
        self.memories.append(text)
        self.embeddings.append(embedding)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Search for similar memories."""
        query_emb = np.array(client.get_embedding(query))

        scores = []
        for emb in self.embeddings:
            emb = np.array(emb)
            similarity = np.dot(query_emb, emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(emb)
            )
            scores.append(similarity)

        # Sort by similarity
        ranked = sorted(
            zip(self.memories, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:top_k]

# Usage
search = MemorySearch()
search.add("Family dinner last Sunday was wonderful")
search.add("Mom's birthday party next week")
search.add("Dad fixed the car yesterday")
search.add("Sister graduated from college")

results = search.search("celebration with mom")
for text, score in results:
    print(f"{score:.3f}: {text}")
# 0.891: Mom's birthday party next week
# 0.823: Family dinner last Sunday was wonderful
# ...
```

### FastAPI Production Server

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from familyos_ultrabert import Client

app = FastAPI(title="UltraBERT API", version="2.0.1")
client = Client()

class AnalyzeRequest(BaseModel):
    text: str
    capabilities: list[str] | None = None

class AnalyzeResponse(BaseModel):
    sentiment: str
    safety: str
    emotions: list[str]
    latency_ms: float

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    if not request.text.strip():
        raise HTTPException(400, "Text cannot be empty")

    result = client.analyze(request.text, request.capabilities)

    return AnalyzeResponse(
        sentiment=result.sentiment,
        safety=result.safety,
        emotions=result.emotions,
        latency_ms=result.latency_ms
    )

@app.get("/health")
def health():
    return client.health_check()

@app.get("/stats")
def stats():
    return client.get_stats()

# Run with: uvicorn api:app --host 0.0.0.0 --port 8000
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0.1 | 2024-12 | Client API, auto-warmup, convenience methods |
| 2.0.0 | 2024-12 | Initial release, UltraBERT API |

---

## Support

- GitHub Issues: [Report bugs](https://github.com/Pkansagra-hub/Family_osModernBERT/issues)
- Documentation: This file

---

## License

Proprietary - All Rights Reserved

Copyright 2024 Princeton BPL / FamilyOS
