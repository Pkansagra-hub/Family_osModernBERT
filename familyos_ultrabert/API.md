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
pip install familyos_ultrabert-2.1.0-py3-none-any.whl torch

# CPU only (ONNX Runtime)
pip install familyos_ultrabert-2.1.0-py3-none-any.whl onnxruntime
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

#### Overview Table

**Client Methods (22 total):**

| Method | Description |
|--------|-------------|
| `get_sentiment(text)` | Quick sentiment: very_negative -> very_positive |
| `get_emotions(text)` | Quick emotion list |
| `get_safety(text)` | Quick safety band: GREEN/AMBER/RED/CRISIS |
| `get_embedding(text)` | Get 768-dim vector |
| `get_intent(text)` | Quick intent classification |
| `get_ingress(text)` | Quick routing category |
| `get_entities(text)` | Quick family entity extraction |
| `get_temporal(text)` | Quick temporal expressions |
| `get_all_entities(text)` | Both family + general entities |
| `is_safe(text)` | Returns True if GREEN |
| `is_crisis(text)` | Returns True if CRISIS |
| `needs_attention(text)` | True if AMBER, RED, or CRISIS |
| `is_positive(text)` | True if positive/very_positive sentiment |
| `is_negative(text)` | True if negative/very_negative sentiment |
| `similarity(text1, text2)` | Cosine similarity between embeddings |
| `find_similar(query, corpus)` | Find most similar texts in corpus |
| `embed_batch(texts)` | Batch embeddings (efficient) |
| `classify_batch(texts, capability)` | Batch single-capability |
| `stream_analyze(texts)` | Generator for memory efficiency |
| `export_embeddings(texts, path)` | Save embeddings to file |
| `health_check()` | Returns health status dict |
| `analyze_batch(texts)` | Analyze multiple texts |

**ClientResult Properties (27 total):**

| Property | Description |
|----------|-------------|
| `sentiment` | Sentiment label |
| `sentiment_confidence` | Confidence score |
| `sentiment_scores` | All class scores |
| `emotions` | List of detected emotions |
| `emotion_scores` | All emotion scores |
| `safety` | Safety band |
| `safety_confidence` | Confidence score |
| `safety_scores` | All safety scores |
| `is_safe` | True if GREEN |
| `is_crisis` | True if CRISIS |
| `needs_attention` | True if not GREEN |
| `entities` | Family entities |
| `general_entities` | General entities |
| `temporal` | Temporal expressions |
| `intent` | Intent classification |
| `intent_confidence` | Intent confidence |
| `ingress` | Routing category |
| `relations` | Relationship predictions |
| `nli` | NLI result |
| `embedding` | 768-dim vector |
| `embedding_dim` | Dimension (768) |
| `top_emotion` | Highest confidence emotion |
| `sentiment_direction` | "positive"/"negative"/"neutral" |
| `has_entities` | True if any entities found |
| `entity_texts` | Just the text spans of entities |
| `to_dict()` | Dictionary output |
| `to_json()` | JSON string output |
| `summary` | One-line summary string |
| `latency_ms` | Inference latency |

---

#### Safety Methods

##### Client.is_safe()

Check if text is safe (GREEN level).

```python
Client.is_safe(text: str) -> bool
```

```python
client.is_safe("I love my family")           # True
client.is_safe("I hate everyone")            # False (AMBER)
client.is_safe("I want to hurt myself")      # False (CRISIS)
```

##### Client.is_crisis()

Check if text indicates crisis.

```python
Client.is_crisis(text: str) -> bool
```

```python
client.is_crisis("Having a great day!")      # False
client.is_crisis("I can't go on anymore")    # True
```

##### Client.needs_attention()

Check if text needs attention (AMBER, RED, or CRISIS).

```python
Client.needs_attention(text: str) -> bool
```

```python
client.needs_attention("I love my family")    # False (GREEN)
client.needs_attention("I'm feeling stressed") # True (AMBER)
client.needs_attention("I want to hurt myself") # True (CRISIS)
```

---

#### Sentiment Methods

##### Client.get_sentiment()

Get sentiment label only.

```python
Client.get_sentiment(text: str) -> str
```

```python
client.get_sentiment("This is amazing!")     # "very_positive"
client.get_sentiment("It's okay")            # "neutral"
client.get_sentiment("I'm disappointed")     # "negative"
```

##### Client.is_positive()

Check if sentiment is positive or very_positive.

```python
Client.is_positive(text: str) -> bool
```

```python
client.is_positive("I love this!")           # True
client.is_positive("It's okay")              # False
```

##### Client.is_negative()

Check if sentiment is negative or very_negative.

```python
Client.is_negative(text: str) -> bool
```

```python
client.is_negative("I hate this!")           # True
client.is_negative("It's okay")              # False
```

---

#### Emotion Methods

##### Client.get_emotions()

Get list of detected emotions.

```python
Client.get_emotions(text: str) -> list[str]
```

```python
client.get_emotions("I'm so happy!")         # ["joy", "excitement"]
client.get_emotions("I'm worried about him") # ["concern", "anxiety"]
```

---

#### Classification Methods

##### Client.get_intent()

Get user intent classification.

```python
Client.get_intent(text: str) -> str
```

```python
client.get_intent("Remember mom's birthday")  # "set_reminder"
client.get_intent("What did we do last year?") # "query_memory"
client.get_intent("Today was great")          # "log_memory"
```

##### Client.get_ingress()

Get routing category for message handling.

```python
Client.get_ingress(text: str) -> str
```

```python
client.get_ingress("Tell me a joke")          # "entertainment"
client.get_ingress("I'm feeling sad")         # "emotional_support"
```

---

#### Entity Methods

##### Client.get_entities()

Get family entities (family members, pets, etc.).

```python
Client.get_entities(text: str) -> list[dict]
```

```python
entities = client.get_entities("Mom picked up Panda from school")
# [{"text": "Mom", "label": "FAMILY_MEMBER"}, {"text": "Panda", "label": "PET"}]
```

##### Client.get_temporal()

Get temporal expressions.

```python
Client.get_temporal(text: str) -> list[dict]
```

```python
temporal = client.get_temporal("Meeting at 3pm tomorrow")
# [{"text": "3pm tomorrow", "label": "DATETIME"}]
```

##### Client.get_all_entities()

Get both family and general entities.

```python
Client.get_all_entities(text: str) -> dict
```

```python
all_ents = client.get_all_entities("Mom went to Apple Store in NYC")
# {
#     "family": [{"text": "Mom", "label": "FAMILY_MEMBER"}],
#     "general": [{"text": "Apple Store", "label": "ORG"}, {"text": "NYC", "label": "LOC"}]
# }
```

---

#### Embedding Methods

##### Client.get_embedding()

Get embedding vector for similarity search.

```python
Client.get_embedding(text: str) -> list[float]
```

```python
embedding = client.get_embedding("Family dinner tonight")
print(len(embedding))  # 768
```

##### Client.similarity()

Calculate cosine similarity between two texts.

```python
Client.similarity(text1: str, text2: str) -> float
```

```python
sim = client.similarity("I love my family", "My family is great")
print(sim)  # 0.92
```

##### Client.find_similar()

Find most similar texts from a corpus.

```python
Client.find_similar(query: str, corpus: list[str], top_k: int = 5) -> list[dict]
```

```python
corpus = ["Family dinner was fun", "Work meeting tomorrow", "Kids played outside"]
matches = client.find_similar("Great family day", corpus, top_k=2)
# [{"text": "Family dinner was fun", "similarity": 0.91, "index": 0}, ...]
```

##### Client.embed_batch()

Get embeddings for multiple texts efficiently.

```python
Client.embed_batch(texts: list[str]) -> list[list[float]]
```

```python
embeddings = client.embed_batch(["Hello", "World"])
print(len(embeddings))     # 2
print(len(embeddings[0]))  # 768
```

##### Client.export_embeddings()

Export embeddings to file (JSONL or CSV).

```python
Client.export_embeddings(texts: list[str], path: str, format: str = "jsonl") -> None
```

```python
texts = ["Hello world", "Family dinner"]
client.export_embeddings(texts, "embeddings.jsonl")
client.export_embeddings(texts, "embeddings.csv", format="csv")
```

---

#### Batch Methods

##### Client.analyze_batch()

Analyze multiple texts.

```python
Client.analyze_batch(texts: list[str]) -> list[ClientResult]
```

```python
results = client.analyze_batch(["I love you", "I hate this"])
for r in results:
    print(r.sentiment)
```

##### Client.classify_batch()

Classify multiple texts with a single capability.

```python
Client.classify_batch(texts: list[str], capability: str) -> list[str]
```

```python
sentiments = client.classify_batch(["Happy day", "Sad news"], "sentiment")
# ["positive", "negative"]
```

##### Client.stream_analyze()

Generator for memory-efficient batch processing.

```python
Client.stream_analyze(texts: list[str]) -> Generator[ClientResult]
```

```python
for result in client.stream_analyze(large_text_list):
    print(result.sentiment)  # Process one at a time
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
| 2.1.0 | 2025-06 | Self-contained wheel (PyTorch backend bundled), no external dependencies |
| 2.0.3 | 2025-06 | Extended API (22 client methods, 27 result properties), critical safety fix |
| 2.0.1 | 2024-12 | Client API, auto-warmup, basic convenience methods |
| 2.0.0 | 2024-12 | Initial release, UltraBERT API |

---

## Support

- GitHub Issues: [Report bugs](https://github.com/Pkansagra-hub/Family_osModernBERT/issues)
- Documentation: This file

---

## License

Proprietary - All Rights Reserved

Copyright 2024 Princeton BPL / FamilyOS
