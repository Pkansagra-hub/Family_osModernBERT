# FamilyOS UltraBERT - Complete Model Output Reference

> **Version:** v3.0.2
> **Architecture:** ModernBERT-base (22 layers, 768-dim, 155M params) + GPT-2 Decoder (24 layers, 355M params)
> **Total Parameters:** ~510M (encoder: 155M, decoder: 355M)

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [All 13 Capabilities](#all-13-capabilities)
3. [Complete Label Reference](#complete-label-reference)
4. [Raw Output Structure](#raw-output-structure)
5. [Client API Reference](#client-api-reference)
6. [Example Outputs](#example-outputs)

---

## Quick Start

```python
from familyos_ultrabert import Client

client = Client()
result = client.analyze("Mom took my dog Panda to the park yesterday, feeling so happy today!")

# Quick access
print(result.sentiment)      # "very_positive"
print(result.safety)         # "GREEN"
print(result.emotions)       # ["joy", "love", "caring"]
print(result.entities)       # [{"text": "Mom", "label": "KINSHIP"}, ...]
print(result.latency_ms)     # 7.5
```

---

## All 13 Capabilities

| # | Capability | Type | Problem Type | Output | Head Location |
|---|------------|------|--------------|--------|---------------|
| 1 | `sentiment` | Sequence | single_label_classification | 5 classes | Encoder |
| 2 | `emotions` | Sequence | multi_label_classification | 44 emotions | Encoder |
| 3 | `safety_familyos` | Sequence | single_label_classification | 4 bands | Encoder |
| 4 | `safety_generic` | Sequence | multi_label_classification | 8 toxicity types | Encoder |
| 5 | `intent` | Sequence | single_label_classification | 8 classes | Encoder |
| 6 | `ingress` | Sequence | single_label_classification | 12 domains | Encoder |
| 7 | `ner_family` | Token | token_classification | 21 BIO tags | Encoder |
| 8 | `ner_general` | Token | token_classification | 17 BIO tags | Encoder |
| 9 | `temporal` | Token | token_classification | 13 BIO tags | Encoder |
| 10 | `relation` | Pair | multi_label_classification | 15 types | Encoder |
| 11 | `nli` | Pair | single_label_classification | 3 classes | Encoder |
| 12 | `embedding` | Vector | n/a | 768-dim float vector | Encoder |
| **13** | **`counterfactual`** | **Generation** | **generation** | **Text** | **GPT-2 Decoder** |

---

## Complete Label Reference

### 1. Sentiment (5 labels)

Single-label classification: predicts overall sentiment of text.

| ID | Label | Description |
|----|-------|-------------|
| 0 | `very_negative` | Strong negative sentiment |
| 1 | `negative` | Negative sentiment |
| 2 | `neutral` | Neutral/no sentiment |
| 3 | `positive` | Positive sentiment |
| 4 | `very_positive` | Strong positive sentiment |

**Output structure:**
```json
{
  "prediction": "neutral",
  "confidence": 0.949,
  "scores": {
    "very_negative": 0.0041,
    "negative": 0.0165,
    "neutral": 0.949,
    "positive": 0.0103,
    "very_positive": 0.0202
  }
}
```

---

### 2. Emotions (44 labels)

Multi-label classification: multiple emotions can be detected simultaneously.

#### Core Emotions (8)
| ID | Label |
|----|-------|
| 0 | `neutral` |
| 1 | `joy` |
| 2 | `sadness` |
| 3 | `anger` |
| 4 | `fear` |
| 5 | `surprise` |
| 6 | `love` |
| 7 | `disgust` |

#### Positive Emotions (12)
| ID | Label |
|----|-------|
| 8 | `admiration` |
| 9 | `amusement` |
| 10 | `approval` |
| 11 | `caring` |
| 12 | `excitement` |
| 13 | `gratitude` |
| 14 | `optimism` |
| 15 | `pride` |
| 16 | `relief` |
| 17 | `contentment` |
| 18 | `hope` |
| 19 | `tenderness` |

#### Negative Emotions (10)
| ID | Label |
|----|-------|
| 20 | `annoyance` |
| 21 | `disappointment` |
| 22 | `disapproval` |
| 23 | `embarrassment` |
| 24 | `grief` |
| 25 | `nervousness` |
| 26 | `remorse` |
| 27 | `frustration` |
| 28 | `overwhelmed` |
| 29 | `emptiness` |

#### Family-Specific Emotions (14)
| ID | Label | Description |
|----|-------|-------------|
| 30 | `nostalgia` | Longing for the past |
| 31 | `protectiveness` | Desire to protect family members |
| 32 | `togetherness` | Feeling of family unity |
| 33 | `longing` | Missing someone/something |
| 34 | `warmth` | Warm family feelings |
| 35 | `playfulness` | Fun, playful moments |
| 36 | `celebration` | Celebratory feelings |
| 37 | `belonging` | Sense of belonging |
| 38 | `parental_pride` | Pride in children |
| 39 | `parental_guilt` | Guilt about parenting |
| 40 | `patience` | Patient understanding |
| 41 | `worry` | Concern for family |
| 42 | `bittersweet` | Mixed happy/sad feelings |
| 43 | `homesickness` | Missing home |

**Output structure:**
```json
{
  "predictions": ["love", "caring", "relief", "worry"],
  "scores": {
    "neutral": 0.0007,
    "joy": 0.0947,
    "love": 0.6268,
    "caring": 0.7351,
    "worry": 0.811,
    ...all 44 emotions with scores...
  }
}
```

---

### 3. Safety FamilyOS (4 labels)

Single-label classification: hierarchical safety bands for family AI.

| ID | Label | Description | Action |
|----|-------|-------------|--------|
| 0 | `GREEN` | Safe content | Normal processing |
| 1 | `AMBER` | Caution needed | Monitor, soft intervention |
| 2 | `RED` | Unsafe content | Block or escalate |
| 3 | `CRISIS` | Emergency (self-harm, violence) | Immediate intervention |

**Output structure:**
```json
{
  "band": "AMBER",
  "confidence": 1.0,
  "probabilities": {
    "GREEN": 0.0,
    "AMBER": 1.0,
    "RED": 0.0,
    "CRISIS": 0.0
  }
}
```

---

### 4. Safety Generic (8 labels)

Multi-label classification: standard toxicity detection.

| ID | Label | Description |
|----|-------|-------------|
| 0 | `toxic` | General toxic content |
| 1 | `severe_toxic` | Highly toxic content |
| 2 | `obscene` | Obscene language |
| 3 | `threat` | Threatening content |
| 4 | `insult` | Insulting content |
| 5 | `identity_hate` | Identity-based hate |
| 6 | `self_harm` | Self-harm related |
| 7 | `dangerous_advice` | Dangerous suggestions |

**Output structure:**
```json
{
  "predictions": ["severe_toxic", "obscene", "self_harm", "dangerous_advice"],
  "scores": {
    "toxic": 0.1404,
    "severe_toxic": 0.5366,
    "obscene": 0.4002,
    "threat": 0.0433,
    "insult": 0.2531,
    "identity_hate": 0.2694,
    "self_harm": 0.7185,
    "dangerous_advice": 0.4322
  }
}
```

---

### 5. Intent (8 labels)

Single-label classification: user's communicative intent.

| ID | Label | Description |
|----|-------|-------------|
| 0 | `log_memory` | Recording a memory/event |
| 1 | `query_memory` | Asking about past memories |
| 2 | `set_reminder` | Setting a reminder/task |
| 3 | `express_feeling` | Expressing emotions |
| 4 | `seek_advice` | Asking for advice |
| 5 | `share_news` | Sharing news/updates |
| 6 | `reflect` | Reflecting on experiences |
| 7 | `other` | Other intents |

**Output structure:**
```json
{
  "prediction": "express_feeling",
  "confidence": 0.6292,
  "scores": {
    "log_memory": 0.1678,
    "query_memory": 0.0001,
    "set_reminder": 0.0005,
    "express_feeling": 0.6292,
    "seek_advice": 0.0018,
    "share_news": 0.1894,
    "reflect": 0.0105,
    "other": 0.0008
  }
}
```

---

### 6. Ingress (12 labels)

Single-label classification: message routing/domain classification.

| ID | Label | Description |
|----|-------|-------------|
| 0 | `DIARY` | Personal diary entries |
| 1 | `TASK` | Task management |
| 2 | `HEALTH` | Health-related |
| 3 | `FINANCE` | Financial topics |
| 4 | `RELATIONSHIP` | Relationship discussions |
| 5 | `WORK` | Work-related |
| 6 | `META` | Meta/system queries |
| 7 | `MEMORY` | Memory queries |
| 8 | `PLANNING` | Planning/scheduling |
| 9 | `CELEBRATION` | Celebrations/events |
| 10 | `CONCERN` | Concerns/worries |
| 11 | `GRATITUDE` | Expressions of gratitude |

**Output structure:**
```json
{
  "prediction": "HEALTH",
  "confidence": 0.7537,
  "scores": {
    "DIARY": 0.0148,
    "TASK": 0.0002,
    "HEALTH": 0.7537,
    "FINANCE": 0.0,
    "RELATIONSHIP": 0.0269,
    "WORK": 0.0,
    "META": 0.0,
    "MEMORY": 0.0004,
    "PLANNING": 0.0,
    "CELEBRATION": 0.0003,
    "CONCERN": 0.2027,
    "GRATITUDE": 0.0008
  }
}
```

---

### 7. NER Family (21 BIO tags, 10 entity types)

Token classification: family-specific named entity recognition.

| Tag | Entity Type | Description |
|-----|-------------|-------------|
| `O` | Outside | Not an entity |
| `B-PERSON` / `I-PERSON` | Person | Person names |
| `B-KINSHIP` / `I-KINSHIP` | Kinship | Family relations (mom, dad, sister) |
| `B-NICKNAME` / `I-NICKNAME` | Nickname | Pet names, nicknames |
| `B-PET` / `I-PET` | Pet | Pet names/references |
| `B-HOME_LOC` / `I-HOME_LOC` | Home Location | Home, rooms, addresses |
| `B-FAMILY_EVENT` / `I-FAMILY_EVENT` | Family Event | Birthdays, reunions |
| `B-ROUTINE` / `I-ROUTINE` | Routine | Daily routines |
| `B-TRADITION` / `I-TRADITION` | Tradition | Family traditions |
| `B-MILESTONE` / `I-MILESTONE` | Milestone | First steps, graduations |
| `B-HEIRLOOM` / `I-HEIRLOOM` | Heirloom | Family heirlooms |

**Output structure:**
```json
{
  "entities": [
    {
      "text": "Mom",
      "label": "KINSHIP",
      "start_token": 1,
      "end_token": 1
    },
    {
      "text": "Panda",
      "label": "PET",
      "start_token": 5,
      "end_token": 6
    },
    {
      "text": "dad's",
      "label": "KINSHIP",
      "start_token": 21,
      "end_token": 22
    }
  ]
}
```

---

### 8. NER General (17 BIO tags, 8 entity types)

Token classification: standard named entity recognition.

| Tag | Entity Type | Description |
|-----|-------------|-------------|
| `O` | Outside | Not an entity |
| `B-PER` / `I-PER` | Person | Person names |
| `B-ORG` / `I-ORG` | Organization | Company/org names |
| `B-LOC` / `I-LOC` | Location | Geographic locations |
| `B-MISC` / `I-MISC` | Miscellaneous | Other entities |
| `B-DATE` / `I-DATE` | Date | Date expressions |
| `B-TIME` / `I-TIME` | Time | Time expressions |
| `B-EVENT` / `I-EVENT` | Event | Named events |
| `B-PRODUCT` / `I-PRODUCT` | Product | Product names |

**Output structure:**
```json
{
  "entities": [
    {
      "text": "Panda",
      "label": "PER",
      "start_token": 5,
      "end_token": 6
    }
  ]
}
```

---

### 9. Temporal (13 BIO tags, 6 entity types)

Token classification: temporal expression extraction.

| Tag | Entity Type | Description |
|-----|-------------|-------------|
| `O` | Outside | Not temporal |
| `B-DATE_ABS` / `I-DATE_ABS` | Absolute Date | "January 15, 2024" |
| `B-DATE_REL` / `I-DATE_REL` | Relative Date | "yesterday", "next week" |
| `B-TIME` / `I-TIME` | Time | "3:00 PM", "morning" |
| `B-DURATION` / `I-DURATION` | Duration | "2 hours", "all day" |
| `B-FREQUENCY` / `I-FREQUENCY` | Frequency | "every day", "weekly" |
| `B-AGE` / `I-AGE` | Age | "5 years old" |

**Output structure:**
```json
{
  "entities": [
    {
      "text": "yesterday,",
      "label": "DATE_REL",
      "start_token": 10,
      "end_token": 11
    }
  ]
}
```

---

### 10. Relation (15 labels)

Multi-label classification: relationship type detection.

| ID | Label | Description |
|----|-------|-------------|
| 0 | `no_relation` | No relationship detected |
| 1 | `parent_of` | Parent relationship |
| 2 | `child_of` | Child relationship |
| 3 | `spouse_of` | Spouse/partner |
| 4 | `sibling_of` | Sibling relationship |
| 5 | `grandparent_of` | Grandparent |
| 6 | `grandchild_of` | Grandchild |
| 7 | `aunt_uncle_of` | Aunt/Uncle |
| 8 | `niece_nephew_of` | Niece/Nephew |
| 9 | `cousin_of` | Cousin |
| 10 | `pet_of` | Pet ownership |
| 11 | `friend_of` | Friendship |
| 12 | `colleague_of` | Work colleague |
| 13 | `lives_at` | Residence |
| 14 | `owns` | Ownership |

**Output structure:**
```json
{
  "predictions": ["child_of", "pet_of"],
  "scores": {
    "no_relation": 0.0008,
    "parent_of": 0.181,
    "child_of": 0.4514,
    "spouse_of": 0.054,
    "sibling_of": 0.0004,
    "grandparent_of": 0.0051,
    "grandchild_of": 0.001,
    "aunt_uncle_of": 0.0001,
    "niece_nephew_of": 0.0001,
    "cousin_of": 0.0002,
    "pet_of": 0.4397,
    "friend_of": 0.0018,
    "colleague_of": 0.0001,
    "lives_at": 0.0017,
    "owns": 0.0026
  }
}
```

---

### 11. NLI (3 labels)

Single-label classification: natural language inference.

| ID | Label | Description |
|----|-------|-------------|
| 0 | `entailment` | Premise implies hypothesis |
| 1 | `neutral` | No inference relation |
| 2 | `contradiction` | Premise contradicts hypothesis |

**Output structure:**
```json
{
  "prediction": "contradiction",
  "confidence": 0.5615,
  "scores": {
    "entailment": 0.0054,
    "neutral": 0.4331,
    "contradiction": 0.5615
  }
}
```

---

### 12. Embedding (768-dim vector)

Dense vector representation for semantic similarity and retrieval.

**Output structure:**
```json
{
  "embedding": [-0.00219, 0.00113, 0.00730, ..., 0.04638]
}
```

- **Dimensionality:** 768
- **Use cases:** Semantic search, similarity, clustering, RAG

---

### 13. Counterfactual (Text Generation)

GPT-2 decoder for generating alternative reframes.

**Input:** Negative family situation
**Output:** Constructive alternative perspective

```python
# Example usage
suggestion = client.suggest_alternative("I yelled at my kids this morning")
# Output: "Instead of yelling, I could have taken a deep breath and calmly explained..."
```

---

## Raw Output Structure

### AnalysisOutput (from model.analyze())

```python
@dataclass
class AnalysisOutput:
    text: str                           # Input text
    capabilities: Dict[str, Dict]       # Results per capability
    latency_ms: float                   # Inference time
    backend: str                        # "pytorch" or "onnx"
```

### ClientResult (from client.analyze())

```python
class ClientResult:
    # Properties with direct access
    text: str                    # Original input
    sentiment: str               # "very_positive", "positive", etc.
    sentiment_confidence: float  # 0.0-1.0
    sentiment_scores: Dict       # All class scores

    emotions: List[str]          # ["joy", "love", ...]
    emotion_scores: Dict         # All 44 emotion scores

    safety: str                  # "GREEN", "AMBER", "RED", "CRISIS"
    safety_confidence: float     # 0.0-1.0
    safety_scores: Dict          # All band probabilities

    entities: List[Dict]         # Family NER entities
    general_entities: List[Dict] # General NER entities
    temporal: List[Dict]         # Temporal expressions

    intent: str                  # User intent
    intent_confidence: float     # 0.0-1.0

    ingress: str                 # Routing category
    relations: List[str]         # Detected relations
    nli: str                     # NLI prediction

    embedding: List[float]       # 768-dim vector
    embedding_dim: int           # 768

    latency_ms: float            # Inference time

    # Convenience properties
    is_safe: bool                # safety == "GREEN"
    is_crisis: bool              # safety == "CRISIS"
    needs_attention: bool        # safety in ("AMBER", "RED", "CRISIS")
    top_emotion: str             # Highest scoring emotion
    sentiment_direction: str     # "positive", "negative", "neutral"
    has_entities: bool           # len(entities) > 0
    entity_texts: List[str]      # ["Mom", "Panda", ...]
    summary: str                 # One-line summary
```

---

## Client API Reference

### Initialization

```python
from familyos_ultrabert import Client

client = Client(
    backend="auto",       # "auto", "pytorch", "onnx"
    device="auto",        # "auto", "cpu", "cuda", "npu"
    warmup=True,          # Warmup on init for consistent latency
    warmup_rounds=3,      # Number of warmup passes
    verbose=False,        # Print loading info
    load_decoder=False,   # Load decoder for generation
)
```

### Core Methods

```python
# Full analysis (all 12 capabilities)
result = client.analyze(text)
result = client.analyze(text, capabilities=["sentiment", "emotions", "safety_familyos"])

# Quick single-capability methods
sentiment = client.get_sentiment(text)      # str
emotions = client.get_emotions(text)        # List[str]
safety = client.get_safety(text)            # str
intent = client.get_intent(text)            # str
ingress = client.get_ingress(text)          # str
embedding = client.get_embedding(text)      # List[float] (768-dim)
entities = client.get_entities(text)        # List[Dict]
temporal = client.get_temporal(text)        # List[Dict]

# Boolean checks
is_safe = client.is_safe(text)              # True if GREEN
is_crisis = client.is_crisis(text)          # True if CRISIS
needs_attention = client.needs_attention(text)  # True if not GREEN
is_positive = client.is_positive(text)      # True if positive sentiment
is_negative = client.is_negative(text)      # True if negative sentiment
```

### Batch & Utility Methods

```python
# Batch processing
results = client.analyze_batch(texts)
embeddings = client.embed_batch(texts)

# Similarity
similarity = client.similarity(text1, text2)  # 0.0-1.0
similar = client.find_similar(query, corpus, top_k=5)

# Stats & Health
stats = client.get_stats()
health = client.health_check()
```

### Decoder Methods (v3)

```python
# Simple generation
suggestion = client.suggest_alternative("I yelled at my kids")

# With decoder session (more efficient for batches)
with client.create_decoder_session() as decoder:
    encoder_output = client.encode(text)
    suggestion = decoder.generate(encoder_output)
```

---

## Example Outputs

### Full Analysis Example

**Input:**
```
"Mom took my dog Panda to the park yesterday, feeling so happy today but I'm worried about dad's health"
```

**Complete Raw Output:**

```json
{
  "sentiment": {
    "prediction": "neutral",
    "confidence": 0.949,
    "scores": {
      "very_negative": 0.0041,
      "negative": 0.0165,
      "neutral": 0.949,
      "positive": 0.0103,
      "very_positive": 0.0202
    }
  },
  "emotions": {
    "predictions": ["love", "caring", "relief", "worry"],
    "scores": {
      "neutral": 0.0007,
      "joy": 0.0947,
      "sadness": 0.2127,
      "anger": 0.001,
      "fear": 0.0667,
      "surprise": 0.0081,
      "love": 0.6268,
      "disgust": 0.0027,
      "admiration": 0.0111,
      "amusement": 0.0186,
      "approval": 0.0102,
      "caring": 0.7351,
      "excitement": 0.0081,
      "gratitude": 0.0671,
      "optimism": 0.0366,
      "pride": 0.0067,
      "relief": 0.389,
      "contentment": 0.0121,
      "hope": 0.2291,
      "tenderness": 0.2132,
      "annoyance": 0.0044,
      "disappointment": 0.0102,
      "disapproval": 0.0004,
      "embarrassment": 0.0025,
      "grief": 0.0607,
      "nervousness": 0.0375,
      "remorse": 0.0049,
      "frustration": 0.0135,
      "overwhelmed": 0.0092,
      "emptiness": 0.0034,
      "nostalgia": 0.0035,
      "protectiveness": 0.1693,
      "togetherness": 0.0808,
      "longing": 0.0119,
      "warmth": 0.1209,
      "playfulness": 0.0187,
      "celebration": 0.0087,
      "belonging": 0.0027,
      "parental_pride": 0.028,
      "parental_guilt": 0.0222,
      "patience": 0.0157,
      "worry": 0.811,
      "bittersweet": 0.1151,
      "homesickness": 0.0028
    }
  },
  "safety_familyos": {
    "band": "AMBER",
    "confidence": 1.0,
    "probabilities": {
      "GREEN": 0.0,
      "AMBER": 1.0,
      "RED": 0.0,
      "CRISIS": 0.0
    }
  },
  "safety_generic": {
    "predictions": ["severe_toxic", "obscene", "self_harm", "dangerous_advice"],
    "scores": {
      "toxic": 0.1404,
      "severe_toxic": 0.5366,
      "obscene": 0.4002,
      "threat": 0.0433,
      "insult": 0.2531,
      "identity_hate": 0.2694,
      "self_harm": 0.7185,
      "dangerous_advice": 0.4322
    }
  },
  "intent": {
    "prediction": "express_feeling",
    "confidence": 0.6292,
    "scores": {
      "log_memory": 0.1678,
      "query_memory": 0.0001,
      "set_reminder": 0.0005,
      "express_feeling": 0.6292,
      "seek_advice": 0.0018,
      "share_news": 0.1894,
      "reflect": 0.0105,
      "other": 0.0008
    }
  },
  "ingress": {
    "prediction": "HEALTH",
    "confidence": 0.7537,
    "scores": {
      "DIARY": 0.0148,
      "TASK": 0.0002,
      "HEALTH": 0.7537,
      "FINANCE": 0.0,
      "RELATIONSHIP": 0.0269,
      "WORK": 0.0,
      "META": 0.0,
      "MEMORY": 0.0004,
      "PLANNING": 0.0,
      "CELEBRATION": 0.0003,
      "CONCERN": 0.2027,
      "GRATITUDE": 0.0008
    }
  },
  "ner_family": {
    "entities": [
      {"text": "Mom", "label": "KINSHIP", "start_token": 1, "end_token": 1},
      {"text": "dog", "label": "PET", "start_token": 4, "end_token": 4},
      {"text": "Panda", "label": "PET", "start_token": 5, "end_token": 6},
      {"text": "dad's", "label": "KINSHIP", "start_token": 21, "end_token": 22}
    ]
  },
  "ner_general": {
    "entities": [
      {"text": "Panda", "label": "PER", "start_token": 5, "end_token": 6}
    ]
  },
  "temporal": {
    "entities": [
      {"text": "yesterday,", "label": "DATE_REL", "start_token": 10, "end_token": 11}
    ]
  },
  "relation": {
    "predictions": ["child_of", "pet_of"],
    "scores": {
      "no_relation": 0.0008,
      "parent_of": 0.181,
      "child_of": 0.4514,
      "spouse_of": 0.054,
      "sibling_of": 0.0004,
      "grandparent_of": 0.0051,
      "grandchild_of": 0.001,
      "aunt_uncle_of": 0.0001,
      "niece_nephew_of": 0.0001,
      "cousin_of": 0.0002,
      "pet_of": 0.4397,
      "friend_of": 0.0018,
      "colleague_of": 0.0001,
      "lives_at": 0.0017,
      "owns": 0.0026
    }
  },
  "nli": {
    "prediction": "contradiction",
    "confidence": 0.5615,
    "scores": {
      "entailment": 0.0054,
      "neutral": 0.4331,
      "contradiction": 0.5615
    }
  },
  "embedding": {
    "embedding": [-0.00219, 0.00113, 0.00730, ..., 0.04638],
    "embedding_dim": 768
  }
}
```

---

## Label Count Summary

| Capability | # Labels | Type |
|------------|----------|------|
| sentiment | 5 | single-label |
| emotions | 44 | multi-label |
| safety_familyos | 4 | single-label |
| safety_generic | 8 | multi-label |
| intent | 8 | single-label |
| ingress | 12 | single-label |
| ner_family | 21 (10 types) | BIO token |
| ner_general | 17 (8 types) | BIO token |
| temporal | 13 (6 types) | BIO token |
| relation | 15 | multi-label |
| nli | 3 | single-label |
| embedding | 768 | vector |
| **TOTAL** | **170+ discrete labels** | |

---

## Performance

| Metric | Value |
|--------|-------|
| Model load time | ~6.7s (cold) |
| Warmup (3 rounds) | ~200ms |
| Inference (all 12 heads) | ~110ms |
| Inference (single head) | ~7-15ms |
| Memory (CPU) | ~1.2GB |
| Memory (GPU) | ~1.5GB |

---

*Generated from FamilyOS UltraBERT v3.0.2 on January 17, 2026*
