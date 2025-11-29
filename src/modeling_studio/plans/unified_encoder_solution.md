# FamilyOS Unified Encoder Solution - Enhanced v2

> **Problem:** 9 separate models consuming ~4350MB with redundant architectures, slow load times (~62s), and multiple forward passes per envelope.
>
> **Solution:** Single multi-task ModernBERT encoder with **12 capability heads**, ~500MB memory, single forward pass.
>
> **Enhanced v2:** Based on latest research in multi-task learning, family NLP, emotion AI, and safety classification (2023-2025).

---

## 1. Current State Analysis

### 1.1 Model Zoo Inventory

| # | Model | HuggingFace ID | Memory | Architecture | Module |
|---|-------|----------------|--------|--------------|--------|
| 1 | `spacy_nlp` | en_core_web_sm | 100MB | Spacy | M02 fallback |
| 2 | `vader_analyzer` | vaderSentiment | 50MB | Lexicon | M04 Tier-0 |
| 3 | `ner_transformer` | dslim/bert-base-NER | 500MB | BERT | M02 |
| 4 | `sentence_transformer` | all-MiniLM-L6-v2 | 500MB | MiniLM | P08 |
| 5 | `sentiment_transformer` | distilbert-sst2 | 400MB | DistilBERT | M04 |
| 6 | `go_emotions` | roberta-base-go_emotions | 500MB | RoBERTa | M04 |
| 7 | `clinical_safety` | distilbert-sst2 | 300MB | DistilBERT | M04 |
| 8 | `zero_shot_classifier` | bart-large-mnli | 1500MB | BART | M10 |
| 9 | `ner_family` | dslim/bert-base-NER | 0MB* | BERT | M02 |

**Total: ~4350MB, 5 different architectures**

### 1.2 Critical Problems

| Problem | Impact | Cost |
|---------|--------|------|
| **Redundant Models** | `ner_transformer` = `ner_family` (identical) | 500MB wasted |
| **Redundant Base** | `sentiment_transformer` ≈ `clinical_safety` (same distilbert) | 300MB wasted |
| **5 Architectures** | BERT, RoBERTa, DistilBERT, BART, MiniLM | Maintenance nightmare |
| **Multiple Passes** | 6+ forward passes per envelope | ~150ms latency |
| **Zero-Shot Overhead** | BART-large for activity classification | 1500MB for one task |
| **No Domain Adaptation** | Models don't understand "Panda", "mummy", family context | Poor accuracy (51% NER) |

### 1.3 Module-to-Model Current Mapping

```
M02 (hippocampus.semantic_project)
├── ner_transformer (dslim/bert-base-NER) ──► WHO/WHERE/WHAT entities
└── spacy_nlp (fallback) ──► DATE/TIME extraction

M04 (affect.analyze)
├── vader_analyzer ──► Tier-0 lexicon sentiment
├── go_emotions ──► 28-class emotion detection
├── sentiment_transformer ──► positive/negative polarity
└── clinical_safety ──► mental health risk score

M07 (social.social_context_classifier)
└── [Rule-based only] ──► social context from participants

M10 (context.ingress_classify)
└── zero_shot_classifier ──► activity type (meal, social, work...)

P08 (embedding pipeline)
└── sentence_transformer ──► 384-dim vectors for similarity
```

---

## 2. Proposed Solution: ModernBERT Unified Encoder

### 2.1 Why ModernBERT?

| Feature | ModernBERT | BERT | RoBERTa | DeBERTa |
|---------|------------|------|---------|---------|
| **Training Data** | 2T tokens | 16GB | 160GB | 78GB |
| **Context Length** | 8192 tokens | 512 | 512 | 512 |
| **Architecture** | Modern (RoPE, Flash Attn) | 2018 | 2019 | 2020 |
| **Speed** | 2x faster | Baseline | Similar | Slower |
| **License** | Apache 2.0 ✓ | Apache 2.0 | MIT | MIT |
| **NLI Performance** | SOTA | Good | Better | Best |

**Winner:** `answerdotai/ModernBERT-base` - modern architecture, fast, permissive license

### 2.2 Unified Architecture (Enhanced v2)

```
                              ┌─────────────────────┐
                              │     Input Text      │
                              │ "Had dinner with    │
                              │  mom at Olive Garden│
                              │  for Emma's bday"   │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │   ModernBERT-base   │
                              │   Shared Encoder    │
                              │   768-dim hidden    │
                              │   22 layers         │
                              └──────────┬──────────┘
                                         │
           ┌─────────────┬───────────────┼───────────────┬─────────────┐
           │             │               │               │             │
    ┌──────▼──────┐ ┌────▼────┐ ┌────────▼────────┐ ┌────▼────┐ ┌──────▼──────┐
    │ Token Heads │ │ Emotion │ │ Classification  │ │Embedding│ │  Pair Heads │
    │             │ │  Head   │ │     Heads       │ │  Head   │ │             │
    ├─────────────┤ │         │ ├─────────────────┤ │         │ ├─────────────┤
    │ner_general  │ │32-cls   │ │sentiment (5)    │ │mean pool│ │nli (3)      │
    │ (17 BIO)    │ │multi-   │ │safety_generic(8)│ │768→768  │ │relation (15)│
    │ner_family   │ │label +  │ │safety_fos (4)   │ │L2 norm  │ │             │
    │ (21 BIO)    │ │family   │ │ingress (12)     │ │         │ │             │
    │temporal     │ │emotions │ │intent (8)       │ │         │ │             │
    │ (13 BIO)    │ │         │ │                 │ │         │ │             │
    └──────┬──────┘ └────┬────┘ └────────┬────────┘ └────┬────┘ └──────┬──────┘
           │             │               │               │             │
           ▼             ▼               ▼               ▼             ▼
       Entities      Emotions    Classifications    Embeddings    Relations
       + Temporal
```

### 2.3 Capability Mapping (Old → New) - Enhanced v2

| Old Model | Old Task | New Capability | Head Type | Labels |
|-----------|----------|----------------|-----------|--------|
| `ner_transformer` | Entity extraction | `ner_general` | TokenClassification | 17 BIO tags |
| `ner_family` | Family entities | `ner_family` | TokenClassification | 21 BIO tags |
| `go_emotions` | 28 emotions | `emotions` | SequenceClassification (multi-label) | 32 classes |
| `sentiment_transformer` | Pos/neg/neu | `sentiment` | SequenceClassification | 5 classes |
| `clinical_safety` | Risk detection | `safety_generic` | SequenceClassification (multi-label) | 8 types |
| [NEW] | Policy bands | `safety_familyos` | SafetyHead | 4 bands |
| `zero_shot_classifier` | Activity type | `ingress` | SequenceClassification | 12 domains |
| `sentence_transformer` | Embeddings | `embedding` | EmbeddingHead (mean pool) | 768-dim |
| [NEW] | Entailment | `nli` | NLIHead | 3 classes |
| **[NEW v2]** | Temporal expressions | `temporal` | TemporalHead | 13 BIO tags |
| **[NEW v2]** | Family relationships | `relation` | RelationHead | 15 relations |
| **[NEW v2]** | User intent | `intent` | IntentHead | 8 intents |

### 2.4 Label Schemas (Enhanced v2 - Implemented in `labels.py`)

#### Generic Labels (Stage A - Public Datasets)

| Capability | Labels | Type |
|------------|--------|------|
| `ner_general` | O, B/I-PER, B/I-ORG, B/I-LOC, B/I-MISC, B/I-DATE, B/I-TIME, B/I-EVENT, B/I-PRODUCT | BIO (17 tags) |
| `sentiment` | very_negative, negative, neutral, positive, very_positive | Single-label (5) |
| `emotions` | GoEmotions (28) + nostalgia, protectiveness, togetherness, longing | Multi-label (32) |
| `safety_generic` | toxic, severe_toxic, obscene, threat, insult, identity_hate, self_harm, dangerous_advice | Multi-label (8) |
| `nli` | entailment, neutral, contradiction | Single-label (3) |
| `temporal` | O, B/I-DATE_ABS, B/I-DATE_REL, B/I-TIME, B/I-DURATION, B/I-FREQUENCY, B/I-AGE | BIO (13 tags) |

#### FamilyOS Labels (Stage B - Domain Adaptation)

| Capability | Labels | Type |
|------------|--------|------|
| `ner_family` | O, B/I-PERSON, B/I-KINSHIP, B/I-NICKNAME, B/I-PET, B/I-HOME_LOC, B/I-FAMILY_EVENT, B/I-ROUTINE, B/I-TRADITION, B/I-MILESTONE, B/I-HEIRLOOM | BIO (21 tags) |
| `ingress` | DIARY, TASK, HEALTH, FINANCE, RELATIONSHIP, WORK, META, MEMORY, PLANNING, CELEBRATION, CONCERN, GRATITUDE | Single-label (12) |
| `safety_familyos` | GREEN, AMBER, RED, CRISIS | Single-label (4) |
| `relation` | no_relation, parent_of, child_of, spouse_of, sibling_of, grandparent_of, grandchild_of, aunt_uncle_of, niece_nephew_of, cousin_of, pet_of, friend_of, colleague_of, lives_at, owns | Single-label (15) |
| `intent` | log_memory, query_memory, set_reminder, express_feeling, seek_advice, share_news, reflect, other | Single-label (8) |

---

## 3. Module Integration Plan

### 3.1 New K0 Architecture

```python
# k0/runtime/model_registry.py

MODEL_REGISTRY = {
    "familyos_unified_v2": {
        "backend": "modernbert_multitask",
        "checkpoint": "checkpoints/modernbert-unified-v2",
        "tier": "transformer_medium",
        "memory_mb": 550,
        "capabilities": [
            # Generic (Stage A)
            "ner_general",
            "sentiment",
            "emotions",
            "safety_generic",
            "nli",
            "embedding",
            "temporal",      # NEW v2
            # FamilyOS (Stage B)
            "ner_family",
            "safety_familyos",
            "ingress",
            "relation",      # NEW v2
            "intent",        # NEW v2
        ]
    }
}

def resolve_capability(capability: str) -> tuple[str, str]:
    """Route capability to model and head."""
    return ("familyos_unified_v1", capability)
```

### 3.2 Module Updates

#### M02: `hippocampus.semantic_project`

```python
# Before (OLD)
ner_model = load_model("ner_transformer")
entities = ner_model(text)

# After (NEW)
from k0.runtime.model_registry import get_unified_model

model = get_unified_model()
output = model.infer(text, capabilities=["ner_general", "ner_family", "temporal"])
entities = output.ner_general + output.ner_family
temporal_expressions = output.temporal  # "last Sunday", "3pm", etc.
```

#### M04: `affect.analyze`

```python
# Before (OLD) - 4 separate models
vader_scores = vader_analyzer(text)
emotions = go_emotions(text)
sentiment = sentiment_transformer(text)
safety = clinical_safety(text)

# After (NEW) - Single call
output = model.infer(
    text,
    capabilities=["sentiment", "emotions", "safety_generic", "safety_familyos", "intent"]
)
emotions = output.emotions  # 32 emotions including family-specific
sentiment = output.sentiment  # 5-point scale
safety_band = output.safety_familyos  # GREEN/AMBER/RED/CRISIS
user_intent = output.intent  # log_memory/query_memory/express_feeling/...
```

#### M10: `context.ingress_classify`

```python
# Before (OLD) - Zero-shot with 1.5GB BART
result = zero_shot_classifier(text, candidate_labels=ACTIVITY_TYPES)

# After (NEW) - Direct classification
output = model.infer(text, capabilities=["ingress"])
activity_type = output.ingress  # DIARY/TASK/HEALTH/FINANCE/RELATIONSHIP/WORK/META
```

#### P08: Embedding Pipeline

```python
# Before (OLD)
embedding = sentence_transformer.encode(text)  # 384-dim

# After (NEW)
output = model.infer(text, capabilities=["embedding"])
embedding = output.embedding  # 768-dim (or 384 with projection)
```

### 3.3 Unified Inference API

```python
# k0/syscalls/nlp.py

@dataclass
class UnifiedNLPOutput:
    """Single call output for all NLP tasks (Enhanced v2 - 12 capabilities)."""

    # Token-level outputs
    ner_general: list[Entity]  # [{text, label, start, end, confidence}] - 17 BIO tags
    ner_family: list[Entity]   # Family-specific entities - 21 BIO tags
    temporal: list[Entity]     # Temporal expressions - 13 BIO tags (NEW v2)

    # Emotions (multi-label)
    emotions: dict[str, float]  # {joy: 0.85, gratitude: 0.78, nostalgia: 0.5, ...} - 32 classes
    primary_emotion: str

    # Sentiment (5-point scale)
    sentiment: str  # very_negative/negative/neutral/positive/very_positive
    valence: float  # 0.0-1.0

    # Safety
    safety_generic: dict[str, float]  # {toxic: 0.1, self_harm: 0.05, ...} - 8 types
    safety_familyos: str  # GREEN/AMBER/RED/CRISIS
    safety_score: float

    # Ingress (activity domain) - 12 domains
    ingress: str  # DIARY/TASK/HEALTH/FINANCE/RELATIONSHIP/WORK/META/MEMORY/PLANNING/CELEBRATION/CONCERN/GRATITUDE
    ingress_confidence: float

    # Intent (NEW v2) - 8 intents
    intent: str  # log_memory/query_memory/set_reminder/express_feeling/seek_advice/share_news/reflect/other
    intent_confidence: float

    # Relation (NEW v2) - 15 relations
    relations: list[dict]  # [{subject, relation, object, confidence}]

    # Embeddings
    embedding: list[float]  # 768-dim vector

    # NLI (if premise-hypothesis provided)
    nli_label: str | None  # entailment/neutral/contradiction


def sys_nlp_infer(
    texts: list[str],
    capabilities: list[str],
    pairs: list[tuple[str, str]] | None = None,  # For NLI/Relation
    entity_pairs: list[tuple[tuple[int, int], tuple[int, int]]] | None = None,  # For Relation
) -> list[UnifiedNLPOutput]:
    """
    Unified NLP inference syscall (Enhanced v2).

    Example:
        outputs = sys_nlp_infer(
            texts=["Had dinner with mom at Olive Garden last Sunday"],
            capabilities=["ner_family", "sentiment", "emotions", "safety_familyos", "temporal", "intent"]
        )
    """
    model = get_unified_model()
    return model.batch_infer(texts, capabilities, pairs, entity_pairs)
```

---

## 4. Performance Comparison

### 4.1 Memory Footprint

| Component | Current | Unified | Savings |
|-----------|---------|---------|---------|
| NER (BERT) | 500MB | - | - |
| Emotions (RoBERTa) | 500MB | - | - |
| Sentiment (DistilBERT) | 400MB | - | - |
| Safety (DistilBERT) | 300MB | - | - |
| Zero-Shot (BART) | 1500MB | - | - |
| Embeddings (MiniLM) | 500MB | - | - |
| Spacy | 100MB | 100MB | Keep |
| VADER | 50MB | 50MB | Keep |
| **Unified ModernBERT** | - | 500MB | - |
| **Total** | **4350MB** | **650MB** | **85% reduction** |

### 4.2 Latency

| Operation | Current | Unified | Improvement |
|-----------|---------|---------|-------------|
| Load time | ~62s | ~8s | 7.7x faster |
| Per-envelope (all tasks) | ~150ms | ~35ms | 4.3x faster |
| NER only | ~25ms | ~15ms | 1.7x faster |
| Emotions only | ~30ms | ~15ms | 2x faster |
| Embeddings only | ~20ms | ~10ms | 2x faster |

### 4.3 Accuracy Targets

| Task | Current | Target | Notes |
|------|---------|--------|-------|
| NER (general) | 51%* | 88%+ | CoNLL-2003 benchmark |
| NER (family) | N/A | 85%+ | Custom family entities |
| Sentiment | 85% | 92%+ | SST-2 benchmark |
| Emotions | 70% | 75%+ | GoEmotions macro F1 |
| Safety (bands) | Rule-based | 80%+ | New ML-based |
| Ingress | ~60%* | 90%+ | Replace zero-shot |

*Current accuracy from benchmark

---

## 5. Training Strategy

### 5.1 Stage A: Generic Multi-Task (Public Data)

**Objective:** Create `modernbert-multitask-v0` with strong generic NLU

| Task | Dataset | Size | Source |
|------|---------|------|--------|
| NER | CoNLL-2003 | 20K | HuggingFace |
| NER | OntoNotes 5.0 | 77K | HuggingFace |
| Sentiment | SST-2 + Amazon Reviews | 100K | HuggingFace |
| Emotions | GoEmotions | 58K | Google |
| Safety | Jigsaw Toxicity + Self-harm | 180K | Kaggle + Custom |
| NLI | MNLI + SNLI | 570K | HuggingFace |
| Embeddings | STS-B + AllNLI | 150K | HuggingFace |
| Temporal | TempEval-3 + TimeBank | 30K | HuggingFace |

**Training Config:** `configs/training/multitask/stage_a_generic.yaml`

```yaml
model:
  name: answerdotai/ModernBERT-base

training:
  num_epochs: 10-12
  batch_size: 32
  gradient_accumulation: 8  # Effective batch = 256
  warmup_ratio: 0.1

  # Head-wise Learning Rates (NEW - from 2024-2025 best practices)
  encoder_lr: 2e-5         # Lower for pretrained backbone
  head_lr: 1e-4            # Higher for classification heads
  token_head_lr: 5e-5      # Token heads need finer updates

  # EMA Model (NEW - +0.8-1.5 pt consistent improvement)
  use_ema: true
  ema_decay: 0.999

  # Hard-negative mining for embedding head
  embedding_hard_negatives: 15  # Mine from same batch

tasks:
  ner_general: {weight: 1.0, head: token_classification, num_labels: 17}
  sentiment: {weight: 1.0, head: sequence_classification, num_labels: 5}
  emotions: {weight: 1.5, head: sequence_classification, multi_label: true, num_labels: 32}
  safety_generic: {weight: 2.0, head: sequence_classification, multi_label: true, num_labels: 8}
  nli: {weight: 1.0, head: nli, num_labels: 3}
  embedding: {weight: 0.5, head: embedding}
  temporal: {weight: 1.0, head: temporal, num_labels: 13}  # NEW v2
```

### 5.2 Stage B: FamilyOS Domain Adaptation

**Objective:** Add `ner_family`, `ingress`, `safety_familyos` heads + LoRA fine-tuning

| Task | Dataset | Size | Source |
|------|---------|------|--------|
| NER (family) | Custom | 3-5K | Annotate from logs |
| Ingress | Custom | 5-7K | Label existing logs |
| Safety (bands) | Custom | 3-4K | Label by policy band |
| Embeddings | Custom clusters | 2K | Family memory clusters |
| Relation | Custom | 2-3K | Annotate family relationships |
| Intent | Custom | 4-5K | Label user intents |

**Training Config:** `configs/training/multitask/stage_b_familyos.yaml`

```yaml
base_model: checkpoints/modernbert-multitask-v0-ema  # Use EMA checkpoint from Stage A

peft:
  method: lora
  r: 32              # INCREASED from 16 (family domain needs more capacity)
  alpha: 64          # INCREASED proportionally
  target_modules: [q_proj, v_proj, k_proj, o_proj]  # Consider adding ff if GPU allows
  dropout: 0.05      # REDUCED for stability

training:
  num_epochs: 5-8
  learning_rate: 1e-4  # Higher for LoRA

  # Safety-critical oversampling (NEW - non-negotiable for CRISIS recall)
  safety_oversampling:
    CRISIS: 20x       # Duplicate CRISIS samples 20x
    RED: 5x           # Duplicate RED samples 5x
  safety_loss_weight: 10-20x  # Heavy safety loss weighting

tasks:
  # New FamilyOS heads
  ner_family: {weight: 2.0, head: token_classification, num_labels: 21}
  ingress: {weight: 1.5, head: sequence_classification, num_labels: 12}
  safety_familyos: {weight: 15.0, head: safety, num_labels: 4}  # INCREASED from 3.0
  relation: {weight: 2.0, head: relation, num_labels: 15}  # NEW v2
  intent: {weight: 1.5, head: intent, num_labels: 8}      # NEW v2

  # Keep generic tasks (prevent forgetting) - with replay
  ner_general: {weight: 0.5, freeze_head: false, replay_ratio: 0.1}
  sentiment: {weight: 0.3, freeze_head: false, replay_ratio: 0.1}
  emotions: {weight: 0.3, freeze_head: false, replay_ratio: 0.1}
  nli: {weight: 0.3, freeze_head: false, replay_ratio: 0.1}
```

---

## 6. Data Annotation Plan

### 6.1 Family NER Annotation (Enhanced v2: 15 → 21 BIO tags)

**Entity Types to Annotate:**

| Entity | Examples | BIO Tags |
|--------|----------|----------|
| PERSON | "Emma", "John" | B-PERSON, I-PERSON |
| KINSHIP | "mom", "dad", "uncle", "nana", "bhai", "didi" | B-KINSHIP, I-KINSHIP |
| NICKNAME | "Panda", "Bunny", "Sweetie" | B-NICKNAME, I-NICKNAME |
| PET | "Max", "Whiskers" | B-PET, I-PET |
| HOME_LOC | "kitchen", "backyard", "Emma's room" | B-HOME_LOC, I-HOME_LOC |
| FAMILY_EVENT | "birthday", "anniversary", "graduation" | B-FAMILY_EVENT, I-FAMILY_EVENT |
| ROUTINE | "school run", "dinner time", "bedtime" | B-ROUTINE, I-ROUTINE |
| **TRADITION** | "Sunday brunch", "movie night", "Diwali" | B-TRADITION, I-TRADITION | **NEW v2** |
| **MILESTONE** | "first steps", "lost tooth", "got married" | B-MILESTONE, I-MILESTONE | **NEW v2** |
| **HEIRLOOM** | "grandma's ring", "dad's watch", "family album" | B-HEIRLOOM, I-HEIRLOOM | **NEW v2** |

**Annotation Format (JSONL):**

```json
{
  "text": "Picked up Panda from school, mom made dinner",
  "tokens": ["Picked", "up", "Panda", "from", "school", ",", "mom", "made", "dinner"],
  "ner_tags": [0, 0, 5, 0, 0, 0, 3, 0, 0]
}
```

Where: 0=O, 3=B-KINSHIP, 5=B-NICKNAME

### 6.2 Ingress Annotation (Enhanced v2: 7 → 12 domains)

**Domain Labels:**

| Label | Description | Examples |
|-------|-------------|----------|
| DIARY | Personal reflections, journaling | "Today was a good day", "Feeling grateful" |
| TASK | To-dos, reminders, action items | "Need to buy groceries", "Don't forget to call mom" |
| HEALTH | Medical, wellness, fitness | "Doctor appointment tomorrow", "Went to gym" |
| FINANCE | Money, bills, budgets | "Paid the electricity bill", "Budget for vacation" |
| RELATIONSHIP | Family dynamics, social | "Had a fight with mom", "Quality time with kids" |
| WORK | Job, career, professional | "Meeting went well", "Project deadline" |
| META | System commands, queries about FamilyOS | "What did I do yesterday?", "Show my memories" |
| **MEMORY** | Recalling past events | "Remember when we went to Goa?", "That time when..." | **NEW v2** |
| **PLANNING** | Future events, scheduling | "Next week we should visit grandma" | **NEW v2** |
| **CELEBRATION** | Achievements, milestones | "Emma got an A!", "Happy anniversary" | **NEW v2** |
| **CONCERN** | Worries, anxieties | "I'm worried about dad's health" | **NEW v2** |
| **GRATITUDE** | Appreciation expressions | "So thankful for my family" | **NEW v2** |

**Annotation Format:**

```json
{"text": "Need to pick up Emma from school at 3pm", "label": "TASK"}
{"text": "Had a wonderful dinner with the family", "label": "DIARY"}
{"text": "Mum's blood pressure was high today", "label": "HEALTH"}
```

### 6.3 Safety (FamilyOS Bands) Annotation

**Policy Band Definitions:**

| Band | Description | Action | Examples |
|------|-------------|--------|----------|
| GREEN | Safe, routine content | Normal processing | "Had lunch with mom" |
| AMBER | Needs attention, mild concern | Flag for review | "Feeling a bit down today" |
| RED | Serious concern | Escalate to K1 | "I've been crying every day this week" |
| CRISIS | Immediate intervention | Emergency protocol | "I don't want to be here anymore" |

**Annotation Guidelines:**

- Consider cultural context (Indian expressions, venting style)
- When in doubt, escalate (AMBER over GREEN, RED over AMBER)
- CRISIS requires explicit self-harm/harm indicators

**Annotation Format:**

```json
{"text": "Everything feels hopeless", "label": "RED", "indicators": ["hopelessness"]}
{"text": "Stressed about work deadlines", "label": "AMBER", "indicators": ["stress"]}
{"text": "Great day at the park with kids!", "label": "GREEN", "indicators": []}
```

---

## 7. Safety Calibration

### 7.1 Threshold Calibration Process

1. **Collect validation set** (~500 samples per band)
2. **Run inference** with unified model
3. **Compute optimal thresholds** for each band transition
4. **Apply temperature scaling** for confidence calibration
5. **Validate CRISIS recall** ≥ 98% (RAISED from 95%)

### 7.2 Calibration Output

**File:** `configs/calibration/safety_thresholds.yaml`

```yaml
safety_familyos:
  temperature: 1.15  # Calibrated temperature
  thresholds:
    GREEN_AMBER: 0.35  # If P(AMBER|RED|CRISIS) > 0.35, escalate to AMBER
    AMBER_RED: 0.45    # If P(RED|CRISIS) > 0.45, escalate to RED
    RED_CRISIS: 0.60   # If P(CRISIS) > 0.60, escalate to CRISIS

  # Override rules (always trigger regardless of score)
  crisis_keywords:
    - "kill myself"
    - "end my life"
    - "don't want to live"
    - "suicide"
```

### 7.3 Safety Evaluation Metrics (UPDATED)

| Metric | Target | Priority | Notes |
|--------|--------|----------|-------|
| CRISIS Recall | **≥ 98%** | P0 (Must not miss) | **RAISED** - non-negotiable |
| RED Recall | ≥ 90% | P0 | Serious concern |
| AMBER Recall | ≥ 85% | P1 | Monitoring required |
| Overall Accuracy | ≥ 80% | P1 | Balanced performance |
| GREEN Precision | ≥ 90% | P2 | Avoid false calm |
| **Cultural FP Rate** | **≤ 2%** | **P0** | **NEW** - Indian hyperbole robustness |

### 7.4 Cultural Robustness Tests (NEW - from 2024-2025 best practices)

Test cases that must NOT trigger CRISIS:

```python
INDIAN_VENTING_PATTERNS = [
    "I'll die of embarrassment",
    "This is killing me",
    "I could die",
    "My head is bursting",
    "I'm going mad",
    "I want to kill the mood",
    "My head is exploding with tension",
]

# All must return GREEN or AMBER, never RED/CRISIS
for text in INDIAN_VENTING_PATTERNS:
    result = model.infer(text, capabilities=["safety_familyos"])
    assert result.safety_familyos in ["GREEN", "AMBER"], f"False CRISIS on: {text}"
```

### 7.5 Catastrophic Forgetting Gates (NEW)

After Stage B training, re-evaluate on Stage A benchmarks:

| Benchmark | Max Allowed Drop | Action if Exceeded |
|-----------|------------------|-------------------|
| CoNLL-2003 (NER) | ≤ 2% F1 | Reduce LoRA r, increase replay |
| SST-2 (Sentiment) | ≤ 2% Acc | Reduce LoRA r, increase replay |
| MNLI (NLI) | ≤ 2% Acc | Reduce LoRA r, increase replay |
| GoEmotions | ≤ 3% F1 | Reduce LoRA r, increase replay |

---

## 8. Rollout Plan

### 8.1 Phase 1: Shadow Mode (Week 1-2)

- Deploy unified model alongside existing zoo
- Log both outputs, compare divergences
- No user-facing changes

### 8.2 Phase 2: Gradual Migration (Week 3-4)

- Enable unified model for internal dev space
- Monitor error rates, latency, safety triggers
- Keep fallback to old zoo

### 8.3 Phase 3: Full Rollout (Week 5+)

- Enable for all spaces
- Deprecate old model zoo
- Monitor and iterate

### 8.4 Rollback Criteria

- CRISIS recall drops below 95%
- Cultural FP rate exceeds 5%
- Latency P95 > 100ms
- Error rate > 1%

---

## 9. Files & Implementation Reference

### 9.1 Already Implemented ✅

| File | Content |
|------|---------|
| `src/modeling_studio/data/labels.py` | All 12 capability label schemas (Enhanced v2) |
| `src/modeling_studio/models/heads.py` | Head implementations (Seq, Token, NLI, Embedding, Safety, Relation, Intent, Temporal) |
| `src/modeling_studio/models/modernbert_multitask.py` | Main multi-task model with 12 capabilities |
| `configs/model/encoder/modernbert_base.yaml` | Model config |
| `configs/training/multitask/stage_a_generic.yaml` | Stage A training config |
| `configs/training/multitask/stage_b_familyos.yaml` | Stage B training config |

### 9.2 To Implement (Priority Order)

| Priority | File | Purpose |
|----------|------|---------|
| P0 | `src/modeling_studio/data/loaders.py` | Dataset loaders |
| P0 | `src/modeling_studio/data/tokenization.py` | Tokenization utils |
| P0 | `src/modeling_studio/data/multitask_dataset.py` | Dataset wrapper |
| P1 | `src/modeling_studio/trainers/multitask_trainer.py` | Multi-task trainer |
| P1 | `src/modeling_studio/trainers/collators.py` | Data collators |
| P1 | `src/modeling_studio/trainers/task_sampler.py` | Task sampling |
| P1 | `src/modeling_studio/trainers/ema.py` | **NEW** - EMA model |
| P1 | `src/modeling_studio/trainers/optimizer.py` | **NEW** - Head-wise LR |
| P2 | `src/modeling_studio/evaluation/metrics.py` | Metric functions |
| P2 | `src/modeling_studio/evaluation/evaluator.py` | Evaluation runner |
| P2 | `src/modeling_studio/evaluation/safety_eval.py` | Safety evaluation |
| P2 | `src/modeling_studio/evaluation/forgetting_eval.py` | **NEW** - Catastrophic forgetting checks |
| P2 | `src/modeling_studio/evaluation/cultural_robustness.py` | **NEW** - Indian hyperbole FP tests |
| P3 | `scripts/train_stage_a.py` | Stage A training |
| P3 | `scripts/train_stage_b.py` | Stage B training |
| P3 | `scripts/calibrate_safety.py` | Safety calibration |

---

## 10. Success Metrics (Updated November 2025)

### 10.1 Technical Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Memory | 4350MB | 650MB | `nvidia-smi` / `psutil` |
| Load time | 62s | 8s | Startup timing |
| Latency (all tasks) | 150ms | 35ms | P95 inference time |
| Model count | 9 | 1 (+spacy, vader) | Registry count |
| Capabilities | 9 | **12** | Enhanced v2 |

### 10.2 Quality Metrics (Updated with Safety Focus)

| Task | Current | Target | Dataset | Priority |
|------|---------|--------|---------|----------|
| NER (general) F1 | 51% | 88% | CoNLL-2003 test | P1 |
| NER (family) F1 | N/A | 85% | Custom test set | P1 |
| Sentiment Acc | 85% | 92% | SST-2 test (5-class) | P1 |
| Emotions macro F1 | 70% | 75% | GoEmotions test (32-class) | P1 |
| Ingress Acc | 60% | 90% | Custom test set (12-class) | P1 |
| **Safety (CRISIS recall)** | Rule-based | **98%** | Custom test set | **P0** |
| **Cultural FP Rate** | N/A | **≤2%** | Indian hyperbole test set | **P0** |
| Temporal F1 | N/A | 82% | TempEval-3 test | P1 |
| Relation F1 | N/A | 80% | Custom test set | P1 |
| Intent Acc | N/A | 88% | Custom test set | P1 |

### 10.3 Catastrophic Forgetting Gates (NEW)

After Stage B, must pass these gates:

| Benchmark | Max Allowed Drop | Status |
|-----------|------------------|--------|
| CoNLL-2003 (NER) | ≤ 2% F1 | Gate |
| SST-2 (Sentiment) | ≤ 2% Acc | Gate |
| MNLI (NLI) | ≤ 2% Acc | Gate |
| GoEmotions | ≤ 3% F1 | Gate |

### 10.4 Business Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| Inference cost | -80% | Fewer GPU cycles |
| Maintenance | -70% | Single model to update |
| Time-to-deploy | -50% | One model vs nine |

---

## Appendix A: Quick Reference Commands

```bash
# Stage A Training (with EMA and head-wise LR)
python scripts/train_stage_a.py \
    --config configs/training/multitask/stage_a_generic.yaml \
    --use_ema \
    --ema_decay 0.999

# Stage B Training (with safety oversampling)
python scripts/train_stage_b.py \
    --config configs/training/multitask/stage_b_familyos.yaml \
    --base_model checkpoints/modernbert-multitask-v0-ema \
    --crisis_oversample 20 \
    --red_oversample 5

# Evaluation
python scripts/evaluate.py \
    --model checkpoints/modernbert-unified-v2 \
    --tasks all

# Safety Calibration
python scripts/calibrate_safety.py \
    --model checkpoints/modernbert-unified-v2 \
    --data data/familyos/safety/calibration.jsonl

# Inference (Enhanced v2 - 12 capabilities)
python scripts/infer.py \
    --model checkpoints/modernbert-unified-v2 \
    --text "Had dinner with mom last Sunday" \
    --capabilities ner_family,sentiment,safety_familyos,temporal,intent
```

---

## Appendix B: Capability → Head Mapping (Enhanced v2)

```python
CAPABILITY_TO_HEAD_TYPE = {
    # Token classification heads
    "ner_general": TokenClassificationHead,   # 17 BIO tags
    "ner_family": TokenClassificationHead,    # 21 BIO tags
    "temporal": TemporalHead,                 # 13 BIO tags (NEW v2)

    # Sequence classification (single-label)
    "sentiment": SequenceClassificationHead,  # 5 classes
    "ingress": SequenceClassificationHead,    # 12 classes
    "intent": IntentHead,                     # 8 classes (NEW v2)

    # Sequence classification (multi-label)
    "emotions": SequenceClassificationHead,   # 32 emotions
    "safety_generic": SequenceClassificationHead,  # 8 toxicity types

    # Safety head (with calibration)
    "safety_familyos": SafetyHead,            # 4 bands

    # Pair/Relation heads
    "nli": NLIHead,                           # 3 classes
    "relation": RelationHead,                 # 15 relations (NEW v2)

    # Embedding head
    "embedding": EmbeddingHead,               # 768-dim mean pooling
}
```

---

**Document Version:** 2.0 (Enhanced v2)
**Last Updated:** November 2025
**Author:** FamilyOS ML Team
**Changes:** 9 → 12 capabilities (added temporal, relation, intent)
