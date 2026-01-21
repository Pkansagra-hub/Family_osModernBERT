# Training Pipeline Dependency Trace - Source of Truth

**Purpose**: Authoritative reference for training pipeline architecture. Verified against actual code.

**Last Verified**: January 2026

---

# STAGE A: Generic Multi-Task Training

**Entry Point**: `scripts/train_stage_a.py`
**Output**: `modernbert-multitask-v0`

---

## Level 0: Entry Point

### `scripts/train_stage_a.py`

**Actual imports from code (lines 66-71)**:

```python
from modeling_studio.data.labels import Capability, get_num_labels
from modeling_studio.data.loaders import load_stage_a_datasets
from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel
from modeling_studio.trainers.collators import MultiTaskCollator
from modeling_studio.trainers.ema import EMAModel
from modeling_studio.trainers.multitask_trainer import MultiTaskTrainer
```

**Tasks trained (from docstring)**:

- ner_general: CoNLL-2003, WikiNeural
- sentiment: SST-2, DynaSent, Yelp
- emotions: 7 super-labels (single-label classification)
- safety_generic: Jigsaw toxicity (8 multi-label classes)
- nli: MNLI, SNLI
- embedding: STS-B, NLI pairs
- temporal: Time expressions (13 BIO tags)

---

## Level 1: Configuration Files

### `configs/training/multitask/stage_a_a100_fast.yaml`

**Verified settings**:

- **Model**: `answerdotai/ModernBERT-base`, bfloat16, Flash Attention 2
- **Epochs**: 4 (not 10 - optimized for plain BCE)
- **Batch size**: 128 per device, gradient_accumulation=2 (effective 256)
- **EMA**: enabled, decay=0.999
- **Uncertainty weighting**: DISABLED

**Head configurations (ACTUAL from YAML)**:

| Head | Type | num_labels | Notes |
|------|------|------------|-------|
| ner_general | token_classification | **9** | O + 4 entity types x B/I |
| sentiment | sequence_classification | 5 | 5-point scale |
| emotions | sequence_classification | **7** | Super-labels (single-label) |
| safety_generic | sequence_classification | 8 | Multi-label with ASL |
| nli | sequence_classification | 3 | Entailment/neutral/contradiction |
| embedding | embedding | 768-dim | Matryoshka [768,512,256,128] |
| temporal | token_classification | 13 | BIO time expressions |

**Task weights (ACTUAL)**:

```yaml
ner_general: 1.0
sentiment: 1.0
emotions: 1.5      # Emphasized
safety_generic: 2.0  # Emphasized
nli: 1.0
embedding: 0.5     # Reduced
temporal: 1.0
```

**Optimizer (ACTUAL)**:

```yaml
encoder_lr: 2e-5
head_lr: 1e-4
token_head_lr: 5e-5
layer_decay: 0.95
```

### `configs/model/encoder/modernbert_base.yaml`

**Verified architecture**:

- hidden_size: 768
- num_hidden_layers: 22
- num_attention_heads: 12
- intermediate_size: 1152
- max_position_embeddings: 8192
- vocab_size: 50368
- rope_theta: 160000.0
- global_attn_every_n_layers: 3
- local_attention: 128

### `configs/data/multitask/stage_a_datasets.yaml`

**NER datasets (ACTUAL)**:

- ner_conll2003: 9 BIO labels (O, B/I-PER, B/I-ORG, B/I-LOC, B/I-MISC)
- ner_wikineural: tner/wikineural, en config, same 9 labels
- ner_fewnerd: **DISABLED** (not BIO format)
- ner_ontonotes: **DISABLED** (requires script loading)

**Emotion datasets (ACTUAL)**:

- emotions_goemotions: **DISABLED** for Stage A
- emotions_super: 7 super-labels, single-label classification
- emotions_familyos_gold: **DISABLED** for Stage A

**Sentiment datasets (ACTUAL)**:

- sentiment_sst2: Binary mapped to 5-class (1=neg, 3=pos)
- sentiment_dynasent: 3-class mapped (neutral support)
- sentiment_yelp_full: 100K sampled, 5-star → 5-class

---

## Level 2: Core Modules

### `src/modeling_studio/data/labels.py`

**CRITICAL DISCREPANCY FOUND**:

| Label Schema | In labels.py | In YAML Config | Used in Training |
|--------------|--------------|----------------|------------------|
| NER_GENERAL_LABELS | **17 BIO tags** | **9 BIO tags** | **9** (YAML overrides) |
| NER_FAMILY_LABELS | **21 BIO tags** | 21 BIO tags | 21 |

**NER_GENERAL_LABELS in labels.py (17 tags)**:

```python
"O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4,
"B-LOC": 5, "I-LOC": 6, "B-MISC": 7, "I-MISC": 8,
"B-DATE": 9, "I-DATE": 10, "B-TIME": 11, "I-TIME": 12,
"B-EVENT": 13, "I-EVENT": 14, "B-PRODUCT": 15, "I-PRODUCT": 16
```

**BUT stage_a_a100_fast.yaml overrides to 9**:

```yaml
ner_general:
  num_labels: 9  # O, B/I-PER, B/I-ORG, B/I-LOC, B/I-MISC
```

**Implication**: Stage A training only uses 9 labels (CoNLL format), not the extended 17.

### `src/modeling_studio/data/loaders.py`

**Key functions**:

- `load_stage_a_datasets()`: Main loader for Stage A
- `KEEP_DATASETS_IN_MEMORY = True`: Caches datasets in RAM
- Uses HuggingFace `load_dataset()` with label mapping

### `src/modeling_studio/models/modernbert_multitask.py`

**CAPABILITY_TO_HEAD_TYPE mapping (ACTUAL from code)**:

```python
Capability.NER_GENERAL: TokenClassificationHead
Capability.NER_FAMILY: TokenClassificationHead
Capability.TEMPORAL: TemporalHead
Capability.SENTIMENT: SequenceClassificationHead
Capability.EMOTIONS: HierarchicalEmotionHead  # 44 emotions capable
Capability.SAFETY_GENERIC: SequenceClassificationHead  # Multi-label ASL
Capability.SAFETY_FAMILYOS: SafetyHead  # 4 bands + 13 subcats
Capability.INGRESS: SequenceClassificationHead
Capability.INTENT: IntentHead
Capability.NLI: NLIHead
Capability.RELATION: RelationHead
Capability.EMBEDDING: EmbeddingHead
Capability.COUNTERFACTUAL: GPT2DecoderHead
```

**Task groups for adapters**:

```python
"token_tasks": [NER_GENERAL, NER_FAMILY, TEMPORAL]
"sequence_tasks": [SENTIMENT, EMOTIONS, SAFETY_*, INGRESS, INTENT]
"pair_tasks": [NLI, RELATION]
"embedding_tasks": [EMBEDDING]
```

### `src/modeling_studio/trainers/collators.py`

**Collator routing (ACTUAL)**:

```python
"ner_general": TokenClassificationCollator
"ner_family": TokenClassificationCollator
"temporal": TokenClassificationCollator
"sentiment": SequenceClassificationCollator
"emotions": SequenceClassificationCollator
"safety_generic": SequenceClassificationCollator
"safety_familyos": SequenceClassificationCollator
"nli": PairCollator
"relation": RelationCollator
"embedding": TripletCollator
```

### `src/modeling_studio/trainers/ema.py`

- **EMAModel**: Maintains exponential moving average of model weights
- decay=0.999 (default)

### `src/modeling_studio/trainers/multitask_trainer.py`

**Key imports (ACTUAL)**:

```python
from modeling_studio.models.losses import FGM, PGD, EmbeddingMixup, RDropLoss
from modeling_studio.trainers.collators import MultiTaskCollator
from modeling_studio.trainers.task_sampler import TaskSampler, create_sampler
from modeling_studio.trainers.task_weighting import UncertaintyWeighting
```

---

## Level 3: Head Architectures (ACTUAL from heads.py)

### `src/modeling_studio/models/heads.py`

**11 Head classes defined** (line numbers):

1. **BaseHead** (line 48) - Abstract base
2. **SequenceClassificationHead** (line 330) - Base for classification
3. **TokenClassificationHead** (line 494) - For NER
4. **EmbeddingHead** (line 583) - Dense vectors
5. **NLIHead** (line 692) - Extends SequenceClassificationHead
6. **SafetyHead** (line 810) - 4 bands + 13 subcategories
7. **EnhancedSafetyHead** (line 1139) - Multi-label toxicity
8. **RelationHead** (line 1592) - Entity pairs
9. **IntentHead** (line 1797) - Extends SequenceClassificationHead
10. **TemporalHead** (line 1869) - Extends TokenClassificationHead
11. **HierarchicalEmotionHead** (line 1969) - 44 emotions

**TokenClassificationHead (ACTUAL architecture)**:

```python
def __init__(self, hidden_size, num_labels=9, dropout=0.1):
    self.classifier = nn.Linear(hidden_size, num_labels)

def forward(self, hidden_states, ...):
    x = self.dropout(hidden_states)
    logits = self.classifier(x)  # (B, L, num_labels)
```

**SequenceClassificationHead (ACTUAL architecture)**:

```python
def __init__(self, hidden_size, num_labels, dropout, pooling="cls", ...):
    self.dense = nn.Linear(hidden_size, hidden_size)
    self.classifier = nn.Linear(hidden_size, num_labels)

def forward(self, hidden_states, ...):
    pooled = self.pool(hidden_states)  # (B, hidden_size)
    x = self.dropout(pooled)
    x = self.dense(x)
    x = torch.tanh(x)
    x = self.dropout(x)
    logits = self.classifier(x)  # (B, num_labels)
```

**SafetyHead (ACTUAL - 4 bands + 13 subcategories)**:

```python
BAND_NAMES = ["GREEN", "AMBER", "RED", "CRISIS"]
SUBCATEGORY_NAMES = [
    "none",  # GREEN
    "stress", "mild_sadness", "frustration", "health_mention",  # AMBER
    "persistent_sadness", "isolation", "hopelessness", "substance",  # RED
    "self_harm_ideation", "suicide_ideation", "harm_to_others", "abuse_disclosure"  # CRISIS
]
```

### `src/modeling_studio/models/decoder_gpt2.py`

- **GPT2DecoderHead**: Pre-trained GPT-2 Medium with prefix injection
- MoE decoder **deprecated** (Chinchilla scaling failure)

---

## Level 4: Training Infrastructure

### `src/modeling_studio/trainers/task_sampler.py`

- **TaskSampler**: Samples tasks based on weights/temperature
- **create_sampler()**: Factory function
- Strategies: proportional, temperature, round_robin

### `src/modeling_studio/trainers/task_weighting.py`

- **UncertaintyWeighting**: Learnable per-task weights
- **DISABLED** in stage_a_a100_fast.yaml

### `src/modeling_studio/models/losses.py`

- **FGM**: Fast Gradient Method (adversarial)
- **PGD**: Projected Gradient Descent (adversarial)
- **EmbeddingMixup**: Interpolation augmentation
- **RDropLoss**: Regularized dropout consistency

---

## Level 5: Optional Epic 5.0 Components

### `src/modeling_studio/models/poolers.py`

- **CLSMeanPooler**: Concatenates [CLS] and mean pooling
- **AttentionPooler**: Learned attention pooling
- **get_pooler()**: Factory function

### `src/modeling_studio/models/pair_encoder.py`

- **CrossAttentionPairEncoder**: Cross-attention for NLI/Relation
- Used by: NLIHead, RelationHead (optional)

### `src/modeling_studio/models/adapters.py`

- **TaskGroupAdapter**: Per-group adapters
- **EPIC_5_AVAILABLE**: Feature flag for optional import

---

## STAGE B: FamilyOS Domain Adaptation

**Entry Point**: `scripts/train_stage_b.py`
**Output**: `modernbert-v2-for-v3-transfer`

**Goal**: Train v2 encoder layers 15-20 (will become v3 layers 23-28)

---

## Level 0: Entry Point

### `scripts/train_stage_b.py`

**Actual imports from code (lines 80-88)**:

```python
from modeling_studio.data import load_stage_b_datasets
from modeling_studio.data.labels import Capability
from modeling_studio.data.loaders import (
    load_embedding_triplets,
    load_familyos_unified,
    load_familyos_unified_for_training,
)
from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel
from modeling_studio.trainers.multitask_trainer import MultiTaskTrainer, MultiTaskTrainingArguments
from modeling_studio.evaluation.evaluator import Evaluator
```

**Two training modes**:

1. **LoRA adapters** (default) - preserves generic capabilities
2. **Full fine-tuning** (v3 prep) - trains encoder layers for transfer

**12 capabilities total**: 5 new FamilyOS + 7 Stage A replay

---

## Level 1: Configuration Files

### `configs/training/multitask/stage_b_for_v3_prep.yaml`

**VERIFIED - Critical settings**:

- **Model**: Load from `outputs/modernbert-multitask-v0-stage-a-fast`
- **PEFT**: `method: none` (NO LoRA - full fine-tuning)
- **Epochs**: 8
- **Batch**: 128 per device, gradient_accumulation=4 (effective 512)
- **Max length**: 96 tokens (FamilyOS data is short: avg=15, P99=35, max=58)

**Layer-wise LR strategy (ACTUAL from YAML)**:

| Layer Group | Learning Rate | Purpose |
|-------------|---------------|---------|
| layers_1_6 | 1e-5 | Preserve foundation |
| layers_7_14 | 2e-5 | Light context tuning |
| layers_15_20 | **5e-5** | CRITICAL - become v3 L23-28 |
| layers_21_22 | 4e-5 | Semantic band |
| head_lr | 1e-4 | Heads (discarded in v3) |

**Heads configuration (ACTUAL)**:

| Head | Type | num_labels | Enabled |
|------|------|------------|---------|
| ner_general | (inherited) | - | true |
| ner_family | token_classification | 21 | true |
| sentiment | (inherited) | - | true |
| emotions | (inherited) | - | true |
| safety_generic | (inherited) | - | true |
| safety_familyos | sequence_classification | 4 | true |
| nli | (inherited) | - | true |
| embedding | (inherited) | - | true |
| temporal | (inherited) | - | true |
| ingress | sequence_classification | 12 | true |
| intent | sequence_classification | 8 | true |
| relation | relation_classification | 15 | true |

**Task weights (ACTUAL)**:

```yaml
# Generic tasks (replay)
ner_general: 0.5
sentiment: 0.5
emotions: 1.0        # Raised from 0.5
safety_generic: 0.3
nli: 0.5
embedding: 0.5

# FamilyOS tasks (main focus)
ner_family: 2.0
ingress: 1.5
safety_familyos: 5.0  # Reduced from 15.0 (CRISIS already oversampled 20x)
intent: 1.5
relation: 2.0
temporal: 1.5
embedding_familyos: 2.0
```

**Safety oversampling (ACTUAL)**:

```yaml
CRISIS: 20  # 445 samples → 8,900 effective
RED: 5      # 18K → 90K effective
AMBER: 1
GREEN: 1
```

**Data configuration (ACTUAL)**:

```yaml
loader: familyos_unified
familyos_data_dirs:
  - data/familyos/unified/output_synthetic_healed
  - data/familyos/unified/output_healed

familyos_tasks:
  - emotions
  - sentiment
  - ner_family
  - safety_familyos
  - intent
  - ingress
  - temporal
  - relations

# Stage A Replay (15% to prevent forgetting)
replay:
  enabled: true
  ratio: 0.15
  datasets:
    - name: conll2003
      task: ner_general
    - name: stanfordnlp/sst2
      task: sentiment
    - name: multi_nli
      task: nli

# Embedding triplets (separate loader)
embedding_familyos:
  data_dir: data/familyos/embeddings/silver_synthetic
  format: triplets
  margin: 0.3
```

**Scheduler (ACTUAL)**:

```yaml
lr_scheduler_type: cosine_with_restarts
warmup_ratio: 0.05
lr_scheduler_kwargs:
  num_cycles: 2  # Restart at epoch 4
```

**Forgetting gates (ACTUAL)**:

```yaml
forgetting_evaluation:
  enabled: true
  after_epoch: [4, 8]
  benchmarks:
    - name: CoNLL-2003, metric: F1, max_drop: 0.03
    - name: SST-2, metric: accuracy, max_drop: 0.03
    - name: MNLI, metric: accuracy, max_drop: 0.03
  action_on_failure: reduce_lr_layers_1_14
```

**v3 transfer verification (ACTUAL)**:

```yaml
v3_transfer_verification:
  enabled: true
  steps:
    - name: "Verify layers 15-20 learned family semantics"
      method: "probe_family_entities"
      min_accuracy: 0.70
    - name: "Verify layers 1-14 preserved generic knowledge"
      method: "probe_conll_entities"
      min_accuracy: 0.85
    - name: "Export layer weights for v3 initialization"
      method: "export_layers_15_20"
      output: "checkpoints/v2_layers_15_20_for_v3.pt"
```

---

## Level 2 (Stage B): Core Modules

### `src/modeling_studio/data/__init__.py`

- Exports `load_stage_b_datasets`

### `src/modeling_studio/data/loaders.py`

**Key functions for Stage B**:

- `load_familyos_unified()`: Reads 420K unified FamilyOS samples
- `load_familyos_unified_for_training()`: With task-specific splits
- `load_embedding_triplets()`: Loads 200K+ anchor/pos/neg triplets

### `src/modeling_studio/evaluation/evaluator.py`

- `Evaluator`: Runs per-task evaluation
- Forgetting evaluation: Compares Stage A baseline metrics

---

## Level 3 (Stage B): All 12 Head Architectures

### `src/modeling_studio/models/heads.py`

**CRITICAL FILE**: Contains all head implementations

**Summary of 12 heads**:

### Token Classification Heads (3)

**1. TokenClassificationHead** (ner_general, ner_family)

```python
# Line 494
def __init__(self, hidden_size, num_labels, dropout=0.1):
    self.classifier = nn.Linear(hidden_size, num_labels)

def forward(self, hidden_states):
    x = self.dropout(hidden_states)  # (B, L, 768)
    logits = self.classifier(x)      # (B, L, num_labels)
    return logits
```

- **Stage A ner_general**: 9 labels (YAML override, not 17)
- **Stage B ner_family**: 21 labels
- **Issue**: No POS, no CRF, no span extraction

**2. TemporalHead** (extends TokenClassificationHead)

- Line 1869
- 13 BIO tags for time expressions
- Inherits all base class issues

### Sequence Classification Heads (6)

**3. SequenceClassificationHead** (sentiment, ingress)

```python
# Line 330
def __init__(self, hidden_size, num_labels, dropout, pooling="cls"):
    self.dense = nn.Linear(hidden_size, hidden_size)
    self.classifier = nn.Linear(hidden_size, num_labels)

def forward(self, hidden_states):
    pooled = self.pool(hidden_states)  # (B, 768)
    x = self.dropout(pooled)
    x = self.dense(x)
    x = torch.tanh(x)
    x = self.dropout(x)
    logits = self.classifier(x)        # (B, num_labels)
    return logits
```

- **sentiment**: 5 classes (5-point scale)
- **ingress**: 12 domain classes

**4. IntentHead** (extends SequenceClassificationHead)

- Line 1797
- 8 user intent classes
- Same architecture as base

**5. NLIHead** (extends SequenceClassificationHead)

- Line 692
- 3 classes: entailment, neutral, contradiction
- Optional: CrossAttentionPairEncoder integration

**6. HierarchicalEmotionHead** (emotions)

```python
# Line 1969 - MOST COMPLEX HEAD
def __init__(self, hidden_size, num_labels=44, ...):
    self.primary_head = nn.Linear(hidden_size, num_labels)
    self.intensity_head = nn.Linear(hidden_size, 1)
    # Optional: secondary, valence_arousal heads

def forward(self, hidden_states):
    pooled = self.pool(hidden_states)
    primary_logits = self.primary_head(pooled)    # (B, 44)
    intensity = self.intensity_head(pooled)        # (B, 1)
    return primary_logits, intensity
```

- **Stage A**: 7 super-labels (YAML override to simplify)
- **Stage B**: Full 44-class multi-label
- **Loss**: Plain BCE (ASL/Focal caused collapse)

**7. SafetyHead** (safety_familyos)

```python
# Line 810
BAND_NAMES = ["GREEN", "AMBER", "RED", "CRISIS"]  # 4 bands
SUBCATEGORY_NAMES = [
    "none",  # GREEN
    "stress", "mild_sadness", "frustration", "health_mention",  # AMBER
    "persistent_sadness", "isolation", "hopelessness", "substance",  # RED
    "self_harm_ideation", "suicide_ideation", "harm_to_others", "abuse_disclosure"  # CRISIS
]  # 13 subcategories

def __init__(self, hidden_size):
    self.band_classifier = nn.Linear(hidden_size, 4)
    self.subcat_classifier = nn.Linear(hidden_size, 13)
```

- Hierarchical: Band first, then subcategory
- Keyword override logic for explicit crisis detection

**8. EnhancedSafetyHead** (safety_generic)

- Line 1139
- Multi-label toxicity (8 classes)
- Uses ASL (Asymmetric Loss) for class imbalance

### Relation Classification Head (1)

**9. RelationHead**

```python
# Line 1592
def __init__(self, hidden_size, num_labels=15):
    self.classifier = nn.Linear(hidden_size * 2, num_labels)

def forward(self, hidden_states, entity1_mask, entity2_mask):
    e1_pooled = self.pool_entity(hidden_states, entity1_mask)  # (B, 768)
    e2_pooled = self.pool_entity(hidden_states, entity2_mask)  # (B, 768)
    concat = torch.cat([e1_pooled, e2_pooled], dim=-1)         # (B, 1536)
    logits = self.classifier(concat)                           # (B, 15)
    return logits
```

- 15 family relationship types
- Optional: CrossAttentionPairEncoder

### Embedding Head (1)

**10. EmbeddingHead**

```python
# Line 583
def __init__(self, hidden_size=768, pooling="mean"):
    # Optional projection layer

def forward(self, hidden_states):
    pooled = self.pool(hidden_states)  # (B, 768)
    normalized = F.normalize(pooled, p=2, dim=-1)
    return normalized
```

- 768-dim dense vectors
- Matryoshka support: [768, 512, 256, 128]
- Loss: TripletMarginLoss (margin=0.3)

---

## Level 4 (Stage B): Supporting Modules

### Pooling Strategies

**`src/modeling_studio/models/poolers.py`**

- **CLSMeanPooler**: Concatenates [CLS] and mean pooling
- **get_pooler()**: Factory function

### Pair Encoding

**`src/modeling_studio/models/pair_encoder.py`**

- **CrossAttentionPairEncoder**: Cross-attention for NLI/Relation
- Optional enhancement over simple concatenation

### Loss Functions

**`src/modeling_studio/models/losses.py`**

- FGM (Fast Gradient Method adversarial)
- PGD (Projected Gradient Descent)
- EmbeddingMixup
- RDropLoss

### Data Collation

**`src/modeling_studio/trainers/collators.py`**

- **MultiTaskCollator**: Routes to task-specific collators
- **TokenClassificationCollator**: Label alignment for NER
- **SequenceClassificationCollator**: Padding
- **PairCollator**: For NLI/relation
- **TripletCollator**: For embeddings
- **CounterfactualCollator**: For decoder (Stage C)

---

## Level 5 (Stage B): Training Infrastructure

### Task Sampling

**`src/modeling_studio/trainers/task_sampler.py`**

- **TaskSampler**: Samples tasks based on weights/temperature
- Strategies: proportional, temperature, uncertainty

### Task Weighting

**`src/modeling_studio/trainers/task_weighting.py`**

- **UncertaintyWeighting**: Dynamic loss balancing (DISABLED in Stage A/B)

### Layer-wise Optimization

**`src/modeling_studio/trainers/optimizer.py`**

- **create_layer_wise_optimizer()**: Assigns LRs per layer group
- Critical for v3 prep: L15-20 get highest LR (5e-5)

### EMA

**`src/modeling_studio/trainers/ema.py`**

- **EMAModel**: Exponential moving average (decay=0.999)

---

## Complete 12-Head Architecture Map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ModernBERT Encoder (22 layers)                       │
│                         Output: hidden_states (B, L, 768)                   │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         │                                                       │
         ▼                                                       ▼
┌─────────────────────┐                               ┌─────────────────────┐
│  TOKEN-LEVEL HEADS  │                               │  SEQUENCE-LEVEL    │
│     (3 heads)       │                               │      HEADS         │
└─────────────────────┘                               │    (7 heads)       │
                                                      └─────────────────────┘
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. ner_general (TokenClassificationHead)                                    │
│    Dropout → Linear(768 → 17)                                               │
│    Issues: ❌ No POS, ❌ No CRF, ❌ Partial entities                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. ner_family (TokenClassificationHead)                                     │
│    Dropout → Linear(768 → 21)                                               │
│    Issues: ❌ Tags verbs as MILESTONE, ❌ Tags "the" as PET                │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. temporal (TemporalHead extends TokenClassificationHead)                  │
│    Dropout → Linear(768 → 13)                                               │
│    Issues: Same as base TokenClassificationHead                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. sentiment (SequenceClassificationHead)                                   │
│    Pooler → Dropout → Linear(768 → 5)                                       │
│    5-point scale: very_negative → very_positive                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ 5. ingress (SequenceClassificationHead)                                     │
│    Pooler → Dropout → Linear(768 → 12)                                      │
│    12 domains: DIARY, TASK, HEALTH, MEMORY, etc.                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ 6. intent (IntentHead extends SequenceClassificationHead)                   │
│    Pooler → Dropout → Linear(768 → 8)                                       │
│    8 intents: log_memory, query_memory, set_reminder, etc.                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ 7. nli (NLIHead extends SequenceClassificationHead)                         │
│    Option A: Pooler → Dropout → Linear(768 → 3)                             │
│    Option B: CrossAttentionPairEncoder → Classifier                         │
│    3 classes: entailment, neutral, contradiction                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ 8. safety_familyos (SafetyHead)                                             │
│    Architecture:                                                            │
│      Pooler → band_classifier(768 → 4)  [GREEN/AMBER/RED/CRISIS]            │
│            → subcat_classifier(768 → 13) [stress, isolation, self_harm, ...]│
│    Hierarchical: Band first, then subcategory within band                   │
│    Keyword override: Explicit crisis detection bypasses model               │
├─────────────────────────────────────────────────────────────────────────────┤
│ 9. safety_generic (EnhancedSafetyHead - more complex)                       │
│    Multi-label toxicity (8 classes)                                         │
│    Uses ASL (Asymmetric Loss) for class imbalance                           │
│    Calibration support for threshold tuning                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ 10. emotions (HierarchicalEmotionHead - MOST COMPLEX)                       │
│     Architecture:                                                           │
│       Pooler → primary_head(768 → 44)  [Multi-label BCE]                    │
│            → secondary_head(768 → 3)   [Top-3 selection]                    │
│            → intensity_head(768 → 1)   [Strength prediction]                │
│            → valence_arousal(768 → 2)  [Optional V-A space]                 │
│     44 FamilyOS emotions (8 core + 12 positive + 10 negative + 14 family)   │
│     Issues: Prone to collapse with ASL/Focal loss (now uses plain BCE)      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 11. relation (RelationHead)                                                 │
│     Architecture:                                                           │
│       Entity1 pooling + Entity2 pooling → Concatenate(1536) → Classifier(15)│
│     Option: CrossAttentionPairEncoder for better entity interaction         │
│     15 relations: parent_of, sibling_of, spouse_of, pet_of, etc.           │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 12. embedding (EmbeddingHead)                                               │
│     Architecture:                                                           │
│       Pooling (mean/cls/max) → [Projection(768 → 768)] → L2 Normalize      │
│     Output: 768-dim dense vectors                                           │
│     Loss: TripletMarginLoss with hard negatives (margin=0.3)                │
│     Use case: Semantic similarity, retrieval, clustering                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Critical Path for v3 Transfer

```
train_stage_b.py
    ↓
stage_b_for_v3_prep.yaml
    ↓ (defines layer-wise LRs)
Layer-wise optimizer creation
    ↓
L1-6:   1e-5 (preserve foundation)
L7-14:  2e-5 (light tuning)
L15-20: 5e-5 ⭐ CRITICAL - strongly train for family context
L21-22: 4e-5 (semantic band)
Heads:  1e-4 (train heads but they'll be discarded)
    ↓
Training loop with FamilyOS data
    ↓
Encoder layers learn family-specific patterns
    ↓
Output: modernbert-v2-for-v3-transfer
    ↓
┌─────────────────────────────────────────────────────────────────┐
│  v3 Weight Transfer (happens in initialize_v3_from_v2.py)       │
│                                                                 │
│  v3 L1-22   ← Direct copy from v2 L1-22                         │
│  v3 L23-28  ← CLONE from v2 L15-20 (these trained layers!) ⭐  │
│  v3 heads   ← New random init (v2 heads discarded)              │
│  Hub tokens ← Semantic centroid init                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Head Complexity Ranking (Simple → Complex)

### Tier 1: Simplest (Single Linear Layer)

1. **TokenClassificationHead** (ner_general, ner_family, temporal)
   - `Dropout → Linear`
   - **Most problematic** due to simplicity

### Tier 2: Basic Classification

1. **SequenceClassificationHead** (sentiment, ingress)
   - `Pooler → Dropout → Linear`
2. **IntentHead** (extends SequenceClassificationHead)
   - Same as base

### Tier 3: Structured Prediction

1. **NLIHead** (with optional CrossAttentionPairEncoder)
   - Can use pair encoder for better representations
2. **RelationHead** (entity concatenation + optional pair encoder)
   - Entity-aware pooling

### Tier 4: Hierarchical/Multi-output

1. **SafetyHead** (4-band hierarchical + 13 subcategories)
   - Two-stage classification
   - Keyword override logic
2. **EnhancedSafetyHead** (multi-label with ASL)
   - Complex loss function

### Tier 5: Most Complex

1. **HierarchicalEmotionHead** (44-class multi-label + intensity + V-A)
   - Multiple output heads
   - Prone to training instability
2. **EmbeddingHead** (metric learning)
   - Triplet loss, hard negative mining

---

## Known Issues by Head

### TokenClassificationHead (ner_general, ner_family)

**Garbage Rate**: 66%+ before filtering

**Issues**:

- No POS tagging → Tags verbs as entities
- No character features → Misses capitalization patterns
- No CRF → Invalid BIO transitions
- No span extraction → Partial entities ("Lincoln" not "Lincoln School")

**Mitigation**: 10-stage post-processing filter pipeline

### HierarchicalEmotionHead (emotions)

**Issues**:

- Training collapse with ASL/Focal loss
- Current fix: Plain BCE loss (expert recommended)
- Multi-label thresholding needed

### SafetyHead (safety_familyos)

**Issues**:

- Indian English false positives ("dying of laughter" → CRISIS)
- Cultural awareness needed
- Current fix: Keyword override + calibration

### RelationHead (relation)

**Issues**:

- Simple concatenation may miss entity interactions
- CrossAttentionPairEncoder improves but adds complexity

---

## Data Flow (Full 12-Task Training)

```
1. YAML Config Loading
   ├─ stage_b_for_v3_prep.yaml: Training hyperparameters
   ├─ stage_b_datasets.yaml: Data sources (gold + silver)
   └─ modernbert_base.yaml: Model architecture

2. Data Loading (Unified Format)
   ├─ load_familyos_unified() reads 420K synthetic samples
   ├─ Extracts 11 tasks from unified JSONL
   ├─ load_embedding_triplets() loads 200K+ triplets separately
   └─ Stage A replay datasets (15%): CoNLL, SST-2, MNLI

3. Tokenization & Collation
   ├─ MultiTaskCollator routes to task-specific collators
   ├─ TokenClassificationCollator: Aligns NER labels with subwords
   ├─ SequenceClassificationCollator: Pads classification tasks
   ├─ PairCollator: Handles NLI/relation entity pairs
   └─ TripletCollator: Prepares anchor/pos/neg for embeddings

4. Model Initialization
   ├─ Load Stage A checkpoint: modernbert-multitask-v0-stage-a-fast
   ├─ Initialize 12 heads (or load existing)
   ├─ Create layer-wise optimizer groups:
   │   L1-6: 1e-5, L7-14: 2e-5, L15-20: 5e-5, L21-22: 4e-5, Heads: 1e-4
   └─ Setup EMA model (decay=0.999)

5. Training Loop (MultiTaskTrainer)
   ├─ Task sampling with temperature (temp=4 for speed)
   ├─ Safety oversampling: CRISIS 20x, RED 5x
   ├─ Forward pass through encoder + selected head
   ├─ Compute task-specific loss (weighted)
   ├─ Backward pass (gradient accumulation)
   └─ Update layer-wise optimizer groups

6. Per-Head Forward Pass Examples

   NER (token-level):
   encoder(input_ids) → hidden_states (B, L, 768)
                      → TokenClassificationHead
                      → logits (B, L, num_labels)
                      → CrossEntropyLoss with ignore_index=-100

   Sentiment (sequence-level):
   encoder(input_ids) → hidden_states (B, L, 768)
                      → SequenceClassificationHead.pooler(hidden_states)
                      → pooled (B, 768)
                      → Dropout → Linear → logits (B, 5)
                      → CrossEntropyLoss

   Emotions (multi-label):
   encoder(input_ids) → hidden_states (B, L, 768)
                      → HierarchicalEmotionHead.pooler
                      → primary_head → logits (B, 44)
                      → BCEWithLogitsLoss (multi-label)

   Relation (pairs):
   encoder(input_ids) → hidden_states (B, L, 768)
                      → RelationHead.pool_entities(e1, e2)
                      → concat([e1_pooled, e2_pooled]) (B, 1536)
                      → Linear(1536 → 15) → logits (B, 15)
                      → CrossEntropyLoss

   Embedding (triplets):
   encoder(anchor) → hidden_anchor (B, L, 768)
   encoder(pos)    → hidden_pos (B, L, 768)
   encoder(neg)    → hidden_neg (B, L, 768)
                   → EmbeddingHead.pool + normalize
                   → embeddings (B, 768) each
                   → TripletMarginLoss(anchor, pos, neg, margin=0.3)

7. Evaluation & Checkpointing
   ├─ Per-task metrics (F1, accuracy, etc.)
   ├─ Forgetting evaluation: Compare Stage A baseline metrics
   ├─ Save checkpoints every N steps
   └─ Save final model: modernbert-v2-for-v3-transfer
```

---

## Files to Modify for Head Improvements

### Primary Targets (All Heads)

**`src/modeling_studio/models/heads.py`** - All 12 head implementations

#### Token Classification Heads (ner_general, ner_family, temporal)

- Replace TokenClassificationHead with:
  - **ContextAwareNERHead**: Add POS features, char features
  - **NERHeadWithCRF**: Add CRF layer for valid transitions
  - **SpanNERHead**: Span-based extraction (no more partial entities)

#### Emotion Head (emotions)

- **HierarchicalEmotionHead** improvements:
  - Better threshold calibration
  - Curriculum learning (easier emotions first)
  - Label correlation loss (optional, currently disabled)

#### Safety Heads (safety_familyos, safety_generic)

- **SafetyHead** improvements:
  - Better cultural awareness features
  - Expanded keyword override dictionary
  - Confidence calibration

#### Relation Head (relation)

- **RelationHead** improvements:
  - Better entity representation (span pooling)
  - CrossAttentionPairEncoder as default
  - Multi-hop reasoning for complex relations

### Secondary Targets

**`src/modeling_studio/models/modernbert_multitask.py`**

- Update CAPABILITY_TO_HEAD_TYPE mapping
- Add new head type configurations

**`configs/training/multitask/stage_b_for_v3_prep.yaml`**

- Add head-specific hyperparameters
- Tune per-head dropout rates
- Add per-head learning rates

### Supporting Changes

**`src/modeling_studio/trainers/collators.py`**

- Add span-based NER collator
- Improve relation pair collation

**`src/modeling_studio/evaluation/metrics.py`**

- Add span-level F1 for NER
- Add per-band metrics for safety
- Add per-emotion-category metrics

---

## Stage B Capabilities Overview

### Stage A Replay (7 tasks - prevent forgetting)

1. **ner_general** (TokenClassificationHead) - 17 BIO tags
2. **sentiment** (SequenceClassificationHead) - 5 classes
3. **emotions** (HierarchicalEmotionHead) - 44 classes
4. **safety_generic** (EnhancedSafetyHead) - 8 toxicity types
5. **nli** (NLIHead) - 3 classes
6. **embedding** (EmbeddingHead) - 768-dim vectors
7. **temporal** (TemporalHead) - 13 BIO tags

### Stage B New (5 tasks - FamilyOS domain)

1. **ner_family** (TokenClassificationHead) - 21 BIO tags
2. **ingress** (SequenceClassificationHead) - 12 domains
3. **safety_familyos** (SafetyHead) - 4 bands + 13 subcategories
4. **intent** (IntentHead) - 8 user intents
5. **relation** (RelationHead) - 15 family relationship types

---

## Training Statistics

### Data Volumes

| Task | Gold Samples | Silver Samples | Total |
|------|-------------|----------------|-------|
| ner_family | 250 | 12,703 | 12,953 |
| ingress | 360 | 22,747 | 23,107 |
| safety_familyos | 200 | 14,333 | 14,533 |
| intent | 200 | 15,972 | 16,172 |
| relation | 40 | 21,340 | 21,380 |
| embedding | - | 200,000+ | 200,000+ |
| **FamilyOS Total** | - | - | **~288K** |
| **Stage A Replay** | - | - | **~25K** |
| **Grand Total** | - | - | **~313K** |

### Training Configuration

- **Epochs**: 8
- **Batch size**: 128 per device (A100 optimized)
- **Effective batch**: 512 (with grad accumulation)
- **Max sequence length**: 96 tokens (FamilyOS data is short)
- **Learning rates**:
  - L1-6: 1e-5 (foundation)
  - L7-14: 2e-5 (context)
  - L15-20: 5e-5 ⭐ (target for v3 transfer)
  - L21-22: 4e-5 (semantic)
  - Heads: 1e-4

---

## Next Steps for All-Head Improvements

### Phase 1: Token Classification (Highest Priority)

1. Implement **ContextAwareNERHead** with POS features
2. Implement **NERHeadWithCRF** for valid transitions
3. Test on CoNLL-2003 (Stage A) and FamilyOS NER (Stage B)
4. Measure garbage entity reduction

### Phase 2: Hierarchical Heads

1. Improve **HierarchicalEmotionHead** threshold calibration
2. Enhance **SafetyHead** cultural awareness
3. Add confidence calibration for both

### Phase 3: Relation & Pair Heads

1. Make **CrossAttentionPairEncoder** default for RelationHead
2. Improve **NLIHead** with better pair encoding
3. Add span-level entity pooling

### Phase 4: Embedding & Loss Functions

1. Experiment with **hard negative mining** for EmbeddingHead
2. Add **focal loss** alternatives for safety heads
3. Implement **label smoothing** for sequence heads

---

## Key Metrics to Track

### Before Enhancement (Current Baselines)

- **NER garbage rate**: 66%+ (requires 10-stage filtering)
- **Emotion collapse**: Fixed with plain BCE (was ASL/Focal)
- **Safety cultural FP**: ~2% (Indian English patterns)
- **Relation accuracy**: ~75% (simple concatenation)

### After Enhancement (Targets)

- **NER garbage rate**: <10% (reduce filtering to 2-3 stages)
- **Span-level NER F1**: +5-10% improvement
- **Emotion calibration**: Better thresholds, no collapse
- **Safety cultural FP**: <1% with expanded awareness
- **Relation accuracy**: +10% with CrossAttention

---

## References

- **Stage A Trace**: [TRAINING_DEPENDENCY_TRACE.md](./TRAINING_DEPENDENCY_TRACE.md)
- **NER Quality Issues**: See separate NER quality catalog
- **ADR K023**: Entity Filtering Consolidation
- **v3 Architecture**: [RELEASE_ARCHITECTURE_v3.md](./RELEASE_ARCHITECTURE_v3.md)
- **Training Logs**: `outputs/modernbert-v2-for-v3-transfer/`
- **Checkpoints**: `checkpoints/modernbert-v2-for-v3-transfer/`
