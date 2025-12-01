# FamilyOS Unified Encoder - Enhanced Design v2

> **Based on:** Latest research in multi-task learning, family NLP, emotion AI, and safety classification (2023-2025)

---

## 1. Architecture Enhancements

### 1.1 Upgraded Base Model Options

| Model | Params | Context | Key Advantage | Recommendation |
|-------|--------|---------|---------------|----------------|
| `answerdotai/ModernBERT-base` | 149M | 8192 | Flash Attention, RoPE, 2T tokens | **Primary Choice** |
| `answerdotai/ModernBERT-large` | 395M | 8192 | Higher capacity for complex tasks | If GPU available |
| `microsoft/deberta-v3-base` | 184M | 512 | Best NLI/understanding | Backup option |
| `BAAI/bge-base-en-v1.5` | 109M | 512 | SOTA embeddings | Embedding-only fallback |

**Enhancement:** Consider **ModernBERT-large** for production if you have GPU, as larger models show significant gains on nuanced tasks like safety and emotion detection.

### 1.2 Advanced Multi-Task Architecture

```
                              ┌─────────────────────┐
                              │     Input Text      │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │   ModernBERT-base   │
                              │   + Adapter Layers  │◄── Task-specific adapters
                              │   768-dim, 22 layers│
                              └──────────┬──────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
           ┌────────▼────────┐  ┌───────▼───────┐  ┌────────▼────────┐
           │  Shared Pooler  │  │ Token Output  │  │ Pair Encoder    │
           │  (CLS + Mean)   │  │ (All tokens)  │  │ (Cross-attn)    │
           └────────┬────────┘  └───────┬───────┘  └────────┬────────┘
                    │                   │                   │
    ┌───────┬───────┼───────┬───────┐   │         ┌────────┴────────┐
    │       │       │       │       │   │         │                 │
┌───▼──┐┌───▼──┐┌───▼──┐┌───▼──┐┌───▼──┐│    ┌────▼────┐     ┌─────▼─────┐
│Sent. ││Emot. ││Safety││Ingr. ││Embed.││    │   NLI   │     │ Relation  │
│Head ││ Head ││ Head ││ Head ││ Head ││    │  Head   │     │   Head    │
└──────┘└──────┘└──────┘└──────┘└──────┘│    └─────────┘     └───────────┘
                                        │
                           ┌────────────┴────────────┐
                           │                         │
                      ┌────▼────┐              ┌─────▼─────┐
                      │NER Gen. │              │NER Family │
                      │  Head   │              │   Head    │
                      └─────────┘              └───────────┘
```

**New Components:**

1. **Task-Specific Adapters** - Lightweight adapters per task group (reduces interference)
2. **Shared Pooler** - Combines CLS and mean pooling for richer representations
3. **Cross-Attention Pair Encoder** - Better for NLI and relation tasks
4. **Relation Head** - NEW: For family relationship extraction

---

## 2. Enhanced Capability Set (9 → 12 Capabilities)

### 2.1 Capability Overview

| # | Capability | Type | Labels | New/Enhanced |
|---|------------|------|--------|--------------|
| 1 | `ner_general` | Token | 9 BIO tags | Enhanced |
| 2 | `ner_family` | Token | 21 BIO tags | **Enhanced** |
| 3 | `sentiment` | Sequence | 5 classes | **Enhanced** |
| 4 | `emotions` | Sequence (multi) | 44 emotions | **Enhanced** |
| 5 | `safety_generic` | Sequence (multi) | 8 types | **Enhanced** |
| 6 | `safety_familyos` | Sequence | 4 bands | Same |
| 7 | `ingress` | Sequence | 12 domains | **Enhanced** |
| 8 | `embedding` | Embedding | 768-dim | Same |
| 9 | `nli` | Pair | 3 classes | Same |
| 10 | `relation` | Pair | 15 relations | **NEW** |
| 11 | `intent` | Sequence | 8 intents | **NEW** |
| 12 | `temporal` | Token | 12 BIO tags | **NEW** |

---

## 3. Enhanced Label Schemas

### 3.1 NER General (Enhanced: 9 → 17 BIO tags)

```python
NER_GENERAL_LABELS_V2 = LabelSchema(
    name="ner_general",
    label2id={
        "O": 0,
        # Person
        "B-PER": 1, "I-PER": 2,
        # Organization
        "B-ORG": 3, "I-ORG": 4,
        # Location/Place
        "B-LOC": 5, "I-LOC": 6,
        # Miscellaneous
        "B-MISC": 7, "I-MISC": 8,
        # Date (explicit dates)
        "B-DATE": 9, "I-DATE": 10,
        # Time (explicit times)
        "B-TIME": 11, "I-TIME": 12,
        # Event (named events)
        "B-EVENT": 13, "I-EVENT": 14,
        # Product/Food/Item
        "B-PRODUCT": 15, "I-PRODUCT": 16,
    },
    problem_type="token_classification",
    description="Extended general NER with temporal and product entities",
)
```

### 3.2 NER Family (Enhanced: 15 → 21 BIO tags)

```python
NER_FAMILY_LABELS_V2 = LabelSchema(
    name="ner_family",
    label2id={
        "O": 0,
        # Person (full names)
        "B-PERSON": 1, "I-PERSON": 2,
        # Kinship terms (mom, dad, uncle, nana, bhai, didi)
        "B-KINSHIP": 3, "I-KINSHIP": 4,
        # Nicknames (Panda, Bunny, Sweetie, Baby)
        "B-NICKNAME": 5, "I-NICKNAME": 6,
        # Pets (Max, Whiskers, our dog)
        "B-PET": 7, "I-PET": 8,
        # Home locations (kitchen, Emma's room, backyard)
        "B-HOME_LOC": 9, "I-HOME_LOC": 10,
        # Family events (birthday, anniversary, graduation, wedding)
        "B-FAMILY_EVENT": 11, "I-FAMILY_EVENT": 12,
        # Routines (school run, dinner time, bedtime story)
        "B-ROUTINE": 13, "I-ROUTINE": 14,
        # NEW: Family traditions (Sunday brunch, movie night, game night)
        "B-TRADITION": 15, "I-TRADITION": 16,
        # NEW: Milestone (first steps, first word, lost tooth)
        "B-MILESTONE": 17, "I-MILESTONE": 18,
        # NEW: Heirloom/Special item (grandma's ring, dad's watch)
        "B-HEIRLOOM": 19, "I-HEIRLOOM": 20,
    },
    problem_type="token_classification",
    description="Family-specific NER with traditions, milestones, and heirlooms",
)
```

**New Entity Types:**

| Entity | Examples | Why Important |
|--------|----------|---------------|
| TRADITION | "Sunday brunch", "movie night", "Diwali celebration" | Recurring family rituals |
| MILESTONE | "first steps", "graduation day", "got married" | Life events to remember |
| HEIRLOOM | "grandma's necklace", "dad's old car", "family photo album" | Sentimental objects |

### 3.3 Sentiment (Enhanced: 3 → 5 classes)

```python
SENTIMENT_LABELS_V2 = LabelSchema(
    name="sentiment",
    label2id={
        "very_negative": 0,  # Strong negative (angry, frustrated, devastated)
        "negative": 1,        # Mild negative (disappointed, sad)
        "neutral": 2,         # Neutral/factual
        "positive": 3,        # Mild positive (happy, content)
        "very_positive": 4,   # Strong positive (ecstatic, overjoyed)
    },
    problem_type="single_label_classification",
    description="5-point sentiment scale for nuanced analysis",
)
```

**Why 5 classes?** Research shows 5-point scales capture intensity better, important for:

- Tracking emotional trends over time
- Detecting mood shifts (amber signals)
- Understanding celebration vs routine positive moments

### 3.4 Emotions (Enhanced: 28 → 44 emotions)

```python
EMOTIONS_LABELS_V2 = LabelSchema(
    name="emotions",
    label2id={
        # Core Emotions (8)
        "neutral": 0,
        "joy": 1,
        "sadness": 2,
        "anger": 3,
        "fear": 4,
        "surprise": 5,
        "love": 6,
        "disgust": 7,
        # Positive Emotions (12)
        "admiration": 8,
        "amusement": 9,
        "approval": 10,
        "caring": 11,
        "excitement": 12,
        "gratitude": 13,
        "optimism": 14,
        "pride": 15,
        "relief": 16,
        "contentment": 17,
        "hope": 18,
        "tenderness": 19,
        # Negative Emotions (10)
        "annoyance": 20,
        "disappointment": 21,
        "disapproval": 22,
        "embarrassment": 23,
        "grief": 24,
        "nervousness": 25,
        "remorse": 26,
        "frustration": 27,
        "overwhelmed": 28,
        "emptiness": 29,
        # Family-Specific Emotions (14)
        "nostalgia": 30,
        "protectiveness": 31,
        "togetherness": 32,
        "longing": 33,
        "warmth": 34,
        "playfulness": 35,
        "celebration": 36,
        "belonging": 37,
        "parental_pride": 38,
        "parental_guilt": 39,
        "patience": 40,
        "worry": 41,
        "bittersweet": 42,
        "homesickness": 43,
    },
    problem_type="multi_label_classification",
    description="FamilyOS 44-emotion schema with family-specific expansions",
)
```

**Family-Specific Emotion Highlights:**

| Emotion | Description | Trigger Examples |
|---------|-------------|------------------|
| nostalgia | Warm memories of the past | "Remember when...", anniversaries, photo albums |
| protectiveness | Parental/family concern | Checking on kids, shielding loved ones |
| togetherness | Feeling of family unity | Game nights, shared meals, "we" statements |
| parental_pride | Pride tied to caregiver role | Report cards, first steps, milestones |
| bittersweet | Mixed joy + sadness | Kids growing up, graduation, farewells |
| homesickness | Missing home/family | Travel, relocation, long deployments |

### 3.5 Safety Generic (Enhanced: 6 → 8 types)

```python
SAFETY_GENERIC_LABELS_V2 = LabelSchema(
    name="safety_generic",
    label2id={
        # Jigsaw toxicity base
        "toxic": 0,
        "severe_toxic": 1,
        "obscene": 2,
        "threat": 3,
        "insult": 4,
        "identity_hate": 5,

        # NEW: Additional safety dimensions
        "self_harm": 6,       # Self-harm ideation
        "dangerous_advice": 7, # Harmful recommendations
    },
    problem_type="multi_label_classification",
    description="Toxicity detection with self-harm and dangerous advice",
)
```

### 3.6 Ingress (Enhanced: 7 → 12 domains)

```python
INGRESS_LABELS_V2 = LabelSchema(
    name="ingress",
    label2id={
        # Original domains
        "DIARY": 0,         # Personal reflections
        "TASK": 1,          # To-dos, reminders
        "HEALTH": 2,        # Medical, wellness
        "FINANCE": 3,       # Money, bills
        "RELATIONSHIP": 4,  # Family dynamics
        "WORK": 5,          # Job, career
        "META": 6,          # System queries

        # NEW: Extended domains
        "MEMORY": 7,        # Recalling past events ("Remember when...")
        "PLANNING": 8,      # Future events ("Next week we should...")
        "CELEBRATION": 9,   # Birthdays, achievements, milestones
        "CONCERN": 10,      # Worries, anxieties (feeds into safety)
        "GRATITUDE": 11,    # Appreciation expressions
    },
    problem_type="single_label_classification",
    description="Extended domain classification for family context",
)
```

**New Domains:**

| Domain | Examples | Why Important |
|--------|----------|---------------|
| MEMORY | "Remember our trip to Goa?", "That time when..." | Memory retrieval, nostalgia tracking |
| PLANNING | "Let's go to the park tomorrow", "We should visit grandma" | Future event detection |
| CELEBRATION | "Emma got an A!", "Happy anniversary to us" | Milestone detection |
| CONCERN | "I'm worried about dad's health", "Kids are struggling" | Early amber signal |
| GRATITUDE | "So thankful for my family", "Blessed to have you" | Positive sentiment booster |

---

## 4. New Capabilities

### 4.1 Relation Extraction (NEW)

**Purpose:** Extract relationships between entities in text

```python
RELATION_LABELS = LabelSchema(
    name="relation",
    label2id={
        "no_relation": 0,
        # Family relations
        "parent_of": 1,       # X is parent of Y
        "child_of": 2,        # X is child of Y
        "spouse_of": 3,       # X is married to Y
        "sibling_of": 4,      # X is sibling of Y
        "grandparent_of": 5,  # X is grandparent of Y
        "grandchild_of": 6,   # X is grandchild of Y
        "aunt_uncle_of": 7,   # X is aunt/uncle of Y
        "niece_nephew_of": 8, # X is niece/nephew of Y
        "cousin_of": 9,       # X is cousin of Y
        "pet_of": 10,         # X is pet of Y
        # Non-family relations
        "friend_of": 11,      # X is friend of Y
        "colleague_of": 12,   # X works with Y
        "lives_at": 13,       # X lives at Y (location)
        "owns": 14,           # X owns Y (heirloom)
    },
    problem_type="single_label_classification",
    description="Relationship extraction between entities",
)
```

**Example:**

```text
Input: "Mom took Panda to the park"
Output: [
  {"subject": "Mom", "relation": "parent_of", "object": "Panda"},
  {"subject": "Mom", "action": "took", "object": "park"}
]
```

### 4.2 Intent Classification (NEW)

**Purpose:** Understand user's intent when interacting with FamilyOS

```python
INTENT_LABELS = LabelSchema(
    name="intent",
    label2id={
        "log_memory": 0,      # "Had dinner with family" (store this)
        "query_memory": 1,    # "What did we do last Sunday?"
        "set_reminder": 2,    # "Remind me to call mom"
        "express_feeling": 3, # "Feeling grateful today"
        "seek_advice": 4,     # "What should I do about..."
        "share_news": 5,      # "Guess what happened!"
        "reflect": 6,         # "Thinking about the past..."
        "other": 7,           # Catch-all
    },
    problem_type="single_label_classification",
    description="User intent classification for FamilyOS interactions",
)
```

### 4.3 Temporal Expression Extraction (NEW)

**Purpose:** Extract and normalize time expressions

```python
TEMPORAL_LABELS = LabelSchema(
    name="temporal",
    label2id={
        "O": 0,
        # Absolute dates
        "B-DATE_ABS": 1, "I-DATE_ABS": 2,     # "January 15, 2024"
        # Relative dates
        "B-DATE_REL": 3, "I-DATE_REL": 4,     # "yesterday", "last week"
        # Times
        "B-TIME": 5, "I-TIME": 6,             # "3pm", "morning"
        # Durations
        "B-DURATION": 7, "I-DURATION": 8,     # "for 2 hours", "all day"
        # Frequency
        "B-FREQUENCY": 9, "I-FREQUENCY": 10,  # "every Sunday", "weekly"
        # Age/Period
        "B-AGE": 11, "I-AGE": 12,             # "when she was 5", "in my 20s"
    },
    problem_type="token_classification",
    description="Temporal expression extraction for timeline construction",
)
```

**Example:**

```text
Input: "Last Sunday we went to grandma's for her 80th birthday"
Output: [
  {"text": "Last Sunday", "label": "DATE_REL", "normalized": "2025-11-23"},
  {"text": "80th birthday", "label": "AGE", "normalized": "80 years"}
]
```

---

## 5. Training Enhancements

### 5.1 Advanced Multi-Task Learning Strategies

#### 5.1.1 Task Grouping with Shared Adapters

```yaml
# configs/training/multitask/stage_a_enhanced.yaml

task_groups:
  # Group 1: Token-level tasks (share NER adapter)
  token_tasks:
    adapter: ner_adapter
    tasks: [ner_general, ner_family, temporal]

  # Group 2: Sequence classification (share classifier adapter)
  sequence_tasks:
    adapter: clf_adapter
    tasks: [sentiment, emotions, safety_generic, safety_familyos, ingress, intent]

  # Group 3: Pair tasks (share pair encoder adapter)
  pair_tasks:
    adapter: pair_adapter
    tasks: [nli, relation]

  # Group 4: Embedding (dedicated)
  embedding_tasks:
    adapter: embed_adapter
    tasks: [embedding]
```

#### 5.1.2 Curriculum Learning

```yaml
curriculum:
  # Stage 1: Easy tasks first (weeks 1-2)
  stage1:
    tasks: [sentiment, ner_general]
    epochs: 3

  # Stage 2: Add medium tasks (weeks 3-4)
  stage2:
    tasks: [sentiment, ner_general, emotions, nli]
    epochs: 3

  # Stage 3: Add hard tasks (weeks 5-6)
  stage3:
    tasks: [all]
    epochs: 4
```

#### 5.1.3 Dynamic Task Weighting (Uncertainty Weighting)

```python
# src/modeling_studio/trainers/task_weighting.py

class UncertaintyWeighting(nn.Module):
    """
    Learns task weights automatically based on homoscedastic uncertainty.
    Paper: "Multi-Task Learning Using Uncertainty to Weigh Losses" (Kendall et al.)
    """
    def __init__(self, num_tasks: int):
        super().__init__()
        # Learnable log variances (one per task)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        """
        Compute weighted sum of losses.

        weight_i = 1 / (2 * sigma_i^2)
        regularization = log(sigma_i)
        """
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        return total_loss
```

#### 5.1.4 EMA Model Checkpointing (NEW - from 2024-2025 best practices)

```python
# src/modeling_studio/trainers/ema.py

class EMAModel:
    """
    Exponential Moving Average of model weights.

    Benefits:
    - +0.8-1.5 pt consistent improvement across all tasks
    - Smoother training dynamics
    - More robust final checkpoint

    Paper: "Mean teachers are better role models" (Tarvainen & Valpola)
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        """Update EMA weights after each training step."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1 - self.decay) * param.data
                )

    def apply_shadow(self, model: nn.Module):
        """Apply EMA weights for evaluation/checkpointing."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model: nn.Module):
        """Restore original weights after evaluation."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
```

#### 5.1.5 Head-wise Learning Rates (NEW - from 2024-2025 best practices)

```yaml
# configs/training/multitask/stage_a_enhanced.yaml

optimizer:
  # Different learning rates for encoder vs heads
  encoder_lr: 2e-5      # Lower for pretrained backbone
  head_lr: 1e-4         # Higher for randomly initialized heads
  token_head_lr: 5e-5   # Token heads need finer updates

  # Optional: Layer-wise LR decay (0.95 per layer from top)
  layer_decay: 0.95     # Bottom layers learn slower
```

```python
# src/modeling_studio/trainers/optimizer.py

def create_optimizer_with_head_lr(
    model: nn.Module,
    encoder_lr: float = 2e-5,
    head_lr: float = 1e-4,
    token_head_lr: float = 5e-5,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    """
    Create optimizer with different LRs for encoder and heads.

    This prevents heads from overfitting while allowing
    the encoder to adapt more carefully.
    """
    encoder_params = []
    classification_head_params = []
    token_head_params = []

    for name, param in model.named_parameters():
        if 'encoder' in name or 'embeddings' in name:
            encoder_params.append(param)
        elif 'token' in name or 'ner' in name or 'temporal' in name:
            token_head_params.append(param)
        else:
            classification_head_params.append(param)

    return torch.optim.AdamW([
        {'params': encoder_params, 'lr': encoder_lr},
        {'params': classification_head_params, 'lr': head_lr},
        {'params': token_head_params, 'lr': token_head_lr},
    ], weight_decay=weight_decay)
```

### 5.2 Data Augmentation Strategies

#### 5.2.1 Family-Specific Augmentation

```python
# src/modeling_studio/data/augmentation.py

class FamilyAugmenter:
    """Augmentation strategies for family text data."""

    # Kinship term variations (Indian + Western)
    KINSHIP_VARIANTS = {
        "mom": ["mum", "mother", "mummy", "ma", "amma", "aai"],
        "dad": ["father", "daddy", "papa", "baba", "appa"],
        "grandma": ["grandmother", "nani", "dadi", "granny", "nana"],
        "grandpa": ["grandfather", "nana", "dada", "grandad"],
        "sister": ["sis", "didi", "akka", "chechi"],
        "brother": ["bro", "bhai", "anna", "chettan"],
        "aunt": ["auntie", "masi", "chachi", "bua", "athai"],
        "uncle": ["chacha", "mama", "kaka", "periappa"],
    }

    # Nickname patterns
    NICKNAME_PATTERNS = [
        "{name}y",      # Emma → Emmy
        "little {name}",
        "baby {name}",
        "{name} bear",
        "{name} bug",
    ]

    def augment_kinship(self, text: str) -> list[str]:
        """Replace kinship terms with variants."""
        augmented = []
        for term, variants in self.KINSHIP_VARIANTS.items():
            if term in text.lower():
                for variant in variants:
                    augmented.append(text.replace(term, variant))
        return augmented
```

#### 5.2.2 Back-Translation Augmentation

```python
def back_translate(text: str, languages: list[str] = ["hi", "es", "fr"]) -> list[str]:
    """
    Translate to another language and back for paraphrase.
    Especially useful for Indian English variations.
    """
    augmented = []
    for lang in languages:
        # English → Language → English
        translated = translate(text, src="en", tgt=lang)
        back = translate(translated, src=lang, tgt="en")
        if back != text:
            augmented.append(back)
    return augmented
```

### 5.3 Contrastive Learning for Embeddings

```python
# src/modeling_studio/models/losses.py

class FamilyContrastiveLoss(nn.Module):
    """
    Contrastive loss optimized for family memory retrieval.

    Hard negatives:
    - Same person, different event
    - Same event type, different family
    - Temporal neighbors (before/after)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchor: torch.Tensor,    # (batch, dim)
        positive: torch.Tensor,  # (batch, dim)
        negatives: torch.Tensor, # (batch, num_neg, dim)
    ) -> torch.Tensor:
        # Normalize
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)

        # Positive similarity
        pos_sim = (anchor * positive).sum(dim=-1) / self.temperature

        # Negative similarities
        neg_sim = torch.bmm(negatives, anchor.unsqueeze(-1)).squeeze(-1) / self.temperature

        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
        labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)

        return F.cross_entropy(logits, labels)
```

---

## 6. Safety Enhancements

### 6.1 Hierarchical Safety Classification

```python
SAFETY_HIERARCHY = {
    "GREEN": {
        "description": "Safe, routine content",
        "action": "PROCESS_NORMAL",
        "confidence_threshold": 0.7,
        "examples": [
            "Had lunch with mom",
            "Kids played in the park",
            "Good day at work",
        ]
    },
    "AMBER": {
        "description": "Mild concern, needs monitoring",
        "action": "FLAG_FOR_REVIEW",
        "confidence_threshold": 0.5,
        "subcategories": {
            "stress": "Work/life stress",
            "mild_sadness": "Temporary low mood",
            "frustration": "Family conflicts",
            "health_mention": "Non-urgent health",
        },
        "examples": [
            "Feeling stressed about deadlines",
            "Had a fight with spouse",
            "Kids are being difficult",
        ]
    },
    "RED": {
        "description": "Serious concern, escalate",
        "action": "ESCALATE_K1",
        "confidence_threshold": 0.4,
        "subcategories": {
            "persistent_sadness": "Ongoing depression indicators",
            "isolation": "Social withdrawal",
            "hopelessness": "Negative outlook",
            "substance": "Alcohol/drug mentions",
        },
        "examples": [
            "Nothing matters anymore",
            "I've been crying every day",
            "Can't get out of bed",
        ]
    },
    "CRISIS": {
        "description": "Immediate intervention needed",
        "action": "EMERGENCY_PROTOCOL",
        "confidence_threshold": 0.3,  # Lower threshold = more sensitive
        "subcategories": {
            "self_harm_ideation": "Thoughts of self-harm",
            "suicide_ideation": "Thoughts of suicide",
            "harm_to_others": "Violence indicators",
            "abuse_disclosure": "Abuse reports",
        },
        "examples": [
            "I want to end it all",
            "Don't want to be here anymore",
            "Life isn't worth living",
        ],
        "keyword_overrides": [
            "kill myself",
            "end my life",
            "suicide",
            "want to die",
            "hurt myself",
        ]
    }
}
```

### 6.2 Multi-Signal Safety Detection

```python
class EnhancedSafetyHead(nn.Module):
    """
    Multi-signal safety detection combining:
    1. ML classification
    2. Keyword matching
    3. Pattern detection
    4. Temporal context (mood trends)
    """

    def __init__(self, hidden_size: int = 768):
        super().__init__()

        # Primary classifier
        self.classifier = SequenceClassificationHead(
            hidden_size=hidden_size,
            num_labels=4,  # GREEN, AMBER, RED, CRISIS
        )

        # Subcategory classifier
        self.subcategory = SequenceClassificationHead(
            hidden_size=hidden_size,
            num_labels=12,  # All subcategories
        )

        # Confidence calibration
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        text: str = None,
    ) -> dict:
        # ML prediction
        logits = self.classifier(hidden_states, attention_mask)
        probs = F.softmax(logits / self.temperature, dim=-1)

        # Subcategory prediction
        sub_logits = self.subcategory(hidden_states, attention_mask)

        output = {
            "band": probs.argmax(dim=-1),
            "band_probs": probs,
            "subcategory_logits": sub_logits,
        }

        # Keyword override (if text provided)
        if text:
            for keyword in CRISIS_KEYWORDS:
                if keyword.lower() in text.lower():
                    output["band"] = 3  # Force CRISIS
                    output["keyword_triggered"] = keyword
                    break

        return output
```

### 6.3 Temporal Safety Monitoring

```python
class TemporalSafetyMonitor:
    """
    Track safety signals over time to detect trends.

    A single RED might be okay, but:
    - 3 AMBERs in a week → escalate to RED
    - RED followed by isolation → escalate to CRISIS
    """

    def __init__(self, window_days: int = 7):
        self.window_days = window_days
        self.history: list[SafetySignal] = []

    def add_signal(self, signal: SafetySignal) -> SafetyEscalation | None:
        self.history.append(signal)
        return self.check_escalation()

    def check_escalation(self) -> SafetyEscalation | None:
        recent = self.get_recent_signals()

        # Rule 1: Multiple AMBERs → RED
        amber_count = sum(1 for s in recent if s.band == "AMBER")
        if amber_count >= 3:
            return SafetyEscalation(
                from_band="AMBER",
                to_band="RED",
                reason=f"{amber_count} AMBER signals in {self.window_days} days",
            )

        # Rule 2: RED + isolation keywords → CRISIS
        has_red = any(s.band == "RED" for s in recent)
        has_isolation = any("alone" in s.text or "nobody" in s.text for s in recent)
        if has_red and has_isolation:
            return SafetyEscalation(
                from_band="RED",
                to_band="CRISIS",
                reason="RED signal combined with isolation indicators",
            )

        return None
```

---

## 7. Cultural Adaptations

### 7.1 Indian English Variations

```python
INDIAN_ENGLISH_MAPPINGS = {
    # Kinship (covered above)

    # Common expressions
    "doing the needful": "doing what's needed",
    "prepone": "move earlier",
    "revert back": "reply",
    "passed out": "graduated",  # NOT unconscious!
    "expired": "died",          # Common in India

    # Emotional expressions
    "tension": "stress/worry",
    "bore": "bored/boring",
    "irritated": "annoyed",     # Higher intensity in Indian English

    # Food/Events
    "tiffin": "lunch box/snack",
    "function": "celebration/event",
    "puja": "religious ceremony",
}

# Safety considerations for Indian expressions
INDIAN_VENTING_PATTERNS = [
    # These are normal venting, NOT crisis
    "I'll die of embarrassment",
    "This is killing me",
    "I could die",
    "My head is bursting",
    "I'm going mad",
    # Context: Indian English uses hyperbole more freely
]
```

### 7.2 Multi-Cultural Family Structures

```python
FAMILY_STRUCTURE_TYPES = {
    "nuclear": {
        "members": ["parents", "children"],
        "kinship_terms": ["mom", "dad", "son", "daughter"],
    },
    "joint_family": {
        "members": ["grandparents", "parents", "children", "aunts", "uncles", "cousins"],
        "kinship_terms": ["dadi", "dada", "nani", "nana", "chacha", "chachi", "bua", "mama", "masi"],
        "common_in": ["India", "Middle East", "Latin America"],
    },
    "blended": {
        "members": ["step-parents", "step-siblings", "half-siblings"],
        "kinship_terms": ["step-mom", "step-dad", "step-brother"],
    },
    "single_parent": {
        "members": ["parent", "children"],
    },
    "multi_generational": {
        "members": ["great-grandparents", "grandparents", "parents", "children"],
    },
}
```

---

## 8. Updated Implementation Files

### 8.1 Enhanced Labels File

```python
# src/modeling_studio/data/labels_v2.py

# Add all new label schemas:
# - NER_GENERAL_LABELS_V2 (17 tags)
# - NER_FAMILY_LABELS_V2 (21 tags)
# - SENTIMENT_LABELS_V2 (5 classes)
# - EMOTIONS_LABELS_V2 (44 emotions)
# - SAFETY_GENERIC_LABELS_V2 (8 types)
# - INGRESS_LABELS_V2 (12 domains)
# - RELATION_LABELS (15 relations) [NEW]
# - INTENT_LABELS (8 intents) [NEW]
# - TEMPORAL_LABELS (12 tags) [NEW]

# Updated capability enum
class Capability(str, Enum):
    # Original
    NER_GENERAL = "ner_general"
    NER_FAMILY = "ner_family"
    SENTIMENT = "sentiment"
    EMOTIONS = "emotions"
    SAFETY_GENERIC = "safety_generic"
    SAFETY_FAMILYOS = "safety_familyos"
    INGRESS = "ingress"
    EMBEDDING = "embedding"
    NLI = "nli"

    # NEW
    RELATION = "relation"
    INTENT = "intent"
    TEMPORAL = "temporal"
```

### 8.2 Enhanced Heads File

```python
# src/modeling_studio/models/heads_v2.py

# Add:
# - RelationHead (for entity pair classification)
# - IntentHead (with slot filling option)
# - TemporalHead (token classification + normalization)
# - EnhancedSafetyHead (with keyword override)
# - HierarchicalEmotionHead (primary + secondary emotions)
```

### 8.3 Enhanced Model File

```python
# src/modeling_studio/models/modernbert_multitask_v2.py

# Add:
# - Task-specific adapter layers
# - Cross-attention pair encoder
# - Shared pooler (CLS + mean)
# - Uncertainty-based task weighting
# - Curriculum learning hooks
```

---

## 9. Quality Targets (Updated)

| Capability | Metric | Target v1 | Target v2 | Notes |
|------------|--------|-----------|-----------|-------|
| ner_general | F1 | 88% | 91% | More entity types |
| ner_family | F1 | 85% | 88% | Traditions, milestones |
| sentiment | Accuracy | 92% | 94% | 5-class scale |
| emotions | Macro F1 | 75% | 78% | 44 emotions |
| safety_familyos | CRISIS Recall | 95% | **98%** | **Raised priority - non-negotiable** |
| safety_familyos | Cultural FP | - | **≤2%** | **Indian hyperbole robustness** |
| ingress | Accuracy | 90% | 92% | 12 domains |
| relation | F1 | - | 82% | New capability |
| intent | Accuracy | - | 90% | New capability |
| temporal | F1 | - | 85% | New capability |

### 9.1 Catastrophic Forgetting Gates (NEW)

After Stage B training, re-evaluate on Stage A benchmarks:

| Benchmark | Max Allowed Drop | Action if Exceeded |
|-----------|------------------|--------------------|
| CoNLL-2003 (NER) | ≤ 2% F1 | Reduce LoRA r, increase replay |
| SST-2 (Sentiment) | ≤ 2% Acc | Reduce LoRA r, increase replay |
| MNLI (NLI) | ≤ 2% Acc | Reduce LoRA r, increase replay |
| FamilyOS Emotions | ≤ 3% F1 | Reduce LoRA r, increase replay |

---

## 10. Summary: v1 → v2 Changes

| Aspect | v1 | v2 | Improvement |
|--------|----|----|-------------|
| Capabilities | 9 | 12 | +relation, intent, temporal |
| NER Family tags | 15 | 21 | +TRADITION, MILESTONE, HEIRLOOM |
| Emotions | 28 | 44 | +family-specific cluster (nostalgia, protectiveness, parental_pride, etc.) |
| Sentiment classes | 3 | 5 | Intensity scale |
| Ingress domains | 7 | 12 | +MEMORY, PLANNING, CELEBRATION, CONCERN, GRATITUDE |
| Safety subcategories | 0 | 12 | Hierarchical classification |
| Task weighting | Manual | Uncertainty | Auto-balancing |
| Cultural support | Basic | Enhanced | Indian English, joint families |
| **EMA Model** | No | Yes | +0.8-1.5 pt consistent |
| **Head-wise LR** | No | Yes | +1-3 pt, prevents head overfitting |
| **Safety Oversampling** | No | CRISIS 20×, RED 5× | CRISIS recall ≥98% |
| **Forgetting Gates** | No | ≤2% drop checks | Prevents Stage A degradation |

---

## 11. Final Training Schedule (Updated November 2025)

Based on expert review of 2024-2025 SOTA practices (Med-PaLM, BloombergGPT, Phi-4, Nemotron-4):

```yaml
# DO NOT SKIP THESE PHASES

Phase A – Generic Multi-Task (7 heads)
  base: answerdotai/ModernBERT-base
  epochs: 10-12
  learning_rate:
    encoder: 2e-5
    heads: 1e-4
    token_heads: 5e-5
  batch_size: 256 (via gradient accumulation)
  techniques:
    - EMA model (decay=0.999)
    - Hard-negative mining for embeddings (15 per batch)
    - Uncertainty weighting with log-var regularization
  output: modernbert-multitask-v0-ema

Phase B – FamilyOS Domain (5 new heads + LoRA)
  base: Phase A EMA checkpoint
  epochs: 5-8
  peft:
    method: lora
    r: 32
    alpha: 64
    target: [q, k, v, o]
  safety:
    CRISIS_oversampling: 20x
    RED_oversampling: 5x
    loss_weight: 15x
  replay:
    ratio: 0.1  # 10% Stage A data to prevent forgetting
  output: modernbert-unified-v2-lora

Phase C – Calibration (NO training, just calibration)
  input: Phase B checkpoint
  steps:
    - Temperature scaling per head
    - Threshold optimization for safety bands
    - Platt scaling for confidence
  output: modernbert-unified-v2-calibrated

Evaluation Gates (MUST PASS before deployment):
  - CRISIS recall ≥ 98% @ 99% precision
  - Cultural hyperbole false-CRISIS ≤ 2%
  - CoNLL-2003 F1 drop ≤ 2% from Phase A
  - SST-2 accuracy drop ≤ 2% from Phase A
  - MNLI accuracy drop ≤ 2% from Phase A

Final – LoRA Merge + Export
  - Merge LoRA weights into base model
  - Export 530MB unified model
  - Upload to HuggingFace / internal registry
```

---

## 12. What We Did NOT Do (And Why)

| Skipped | Reason |
|---------|--------|
| 750B-1T token Continued Pre-Training | Don't have enough family data at that scale |
| T5-style span corruption | ModernBERT is encoder-only, not compatible |
| Adding 500 vocab tokens | Risky - can destroy pretrained embeddings |
| PCGrad/GradVac gradient surgery | Complex, add later if head conflicts observed |
| Sleep-style replay | Only needed for 100k+ step training |
| Separate Safety Phase C fine-tuning | Risky - can destroy other heads |

---

**Document Version:** 2.1
**Last Updated:** November 2025
**Based On:** Latest multi-task learning research (2023-2025) + Expert Review
