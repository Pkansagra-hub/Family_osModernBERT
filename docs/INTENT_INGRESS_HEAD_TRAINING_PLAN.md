# Intent & Ingress SOTA Head Training Plan

## Executive Summary

Train **future-proof** Intent and Ingress heads using **Label-Description Embedding** architecture. This enables adding new labels without retraining the encoder or breaking existing functionality.

**Key Insight:** We already have all the pieces - proven training infrastructure, consolidated data, and working checkpoints. We simply **adapt the existing GlobalPointer training flow** to train Intent/Ingress heads.

**Key Design Decision:** Both Intent and Ingress are **multi-label** because in K1 FamilyOS context, utterances naturally have multiple valid classifications:

- "Dad got promoted!" → [share_news, express_feeling] + [CELEBRATION, WORK]
- "I miss the old days" → [reflect, query_memory] + [MEMORY, RELATIONSHIP]

---

## 1. Why This Approach Works

### We Already Have Everything

| Component | Status | Location |
|-----------|--------|----------|
| Training Script | EXISTS | `scripts/training/train_globalpointer_unified.py` |
| Config Template | EXISTS | `configs/training/globalpointer_heads.yaml` |
| Base Checkpoint | EXISTS | `outputs/modernbert-v2-for-v3-transfer/checkpoint-18000` |
| Intent Data | READY | `data/processed/intent_unified/` (304K train, 34K val) |
| Ingress Data | READY | `data/processed/ingress_unified/` (380K train, 42K val) |
| Head Classes | TO ADD | `src/modeling_studio/models/heads.py` |

### The Pattern We Follow

When we trained GlobalPointer NER heads, we:

1. Loaded `checkpoint-18000` (12 capabilities, trained encoder)
2. Froze the encoder
3. Replaced 3 NER heads with GlobalPointer architecture
4. Trained only the new heads
5. Saved unified checkpoint with all 12 capabilities

**For Intent/Ingress, we do the same:**

1. Load `checkpoint-8000` (has trained GlobalPointer NER)
2. Freeze encoder + NER heads
3. Replace Intent/Ingress heads with LabelDescriptionHead architecture
4. Train only Intent/Ingress heads
5. Save unified checkpoint with all capabilities

---

## 2. Checkpoint Flow

```
checkpoint-18000 (base, 12 heads, no GlobalPointer)
      |
      | GlobalPointer training (freeze encoder, replace NER heads)
      v
checkpoint-8000 (12 heads, with trained GlobalPointer NER)
      |
      | Intent/Ingress training (freeze encoder+NER, replace Intent/Ingress)
      v
checkpoint-FINAL (12 heads, GlobalPointer NER + SOTA Intent/Ingress)
```

### Source Checkpoint: `checkpoints/checkpoint-8000`

| Property | Value |
|----------|-------|
| NER Type | **GlobalPointer** (SOTA span-based, TRAINED) |
| ner_general | 4 labels (PER, ORG, LOC, MISC) |
| ner_family | 10 labels (KINSHIP, MILESTONE, etc.) |
| temporal | 6 labels (DATE_ABS, DURATION, etc.) |
| Intent | `IntentHead` - 8 labels (old simple classifier, TO REPLACE) |
| Ingress | `SequenceClassificationHead` - 12 labels (old, TO REPLACE) |

---

## 3. Training Data (READY)

### 3.1 Data Sources Summary

| Source | Intent Records | Ingress Records | Format |
|--------|----------------|-----------------|--------|
| **Gold** (manual) | 320 | 288 | High quality |
| **Silver** (synthetic) | 15,972 | 22,747 | Balanced |
| **Unified** (multi-task) | 562,156 | 562,156 | Large scale |

### 3.2 Consolidated Datasets (CREATED)

Data consolidation script: `scripts/data/consolidate_intent_ingress_data.py`

#### Intent Unified (`data/processed/intent_unified/`)

| Split | Records | Shards (5000 each) |
|-------|---------|-------------------|
| Train | 304,120 | 61 shards |
| Val | 33,791 | 7 shards |

**Distribution (balanced):**

| Intent | Count | % |
|--------|-------|---|
| express_feeling | 45,012 | 14.8% |
| query_memory | 45,000 | 14.8% |
| seek_advice | 44,991 | 14.8% |
| set_reminder | 44,961 | 14.8% |
| share_news | 41,227 | 13.6% |
| log_memory | 40,945 | 13.5% |
| reflect | 33,374 | 11.0% |
| other | 8,610 | 2.8% |

#### Ingress Unified (`data/processed/ingress_unified/`)

| Split | Records | Shards (5000 each) |
|-------|---------|-------------------|
| Train | 379,890 | 76 shards |
| Val | 42,210 | 9 shards |

**Distribution (balanced):**

| Ingress | % |
|---------|---|
| RELATIONSHIP | 9.5% |
| PLANNING | 9.5% |
| WORK | 9.5% |
| TASK | 9.5% |
| MEMORY | 9.5% |
| HEALTH | 9.1% |
| CONCERN | 9.0% |
| CELEBRATION | 8.8% |
| DIARY | 8.3% |
| FINANCE | 7.8% |
| GRATITUDE | 5.5% |
| META | 4.1% |

### 3.3 Data Format

```json
{"text": "Remember to call grandma tomorrow", "intent": "set_reminder", "source": "unified"}
{"text": "I feel so happy today", "ingress": "DIARY", "source": "unified"}
```

---

## 4. Label Schemas

### 4.1 Intent Labels (8 classes)

| ID | Label | Description (for embedding init) |
|----|-------|----------------------------------|
| 0 | log_memory | User wants to record or save a memory, thought, or experience |
| 1 | query_memory | User wants to retrieve or search past memories |
| 2 | set_reminder | User wants to set a reminder, alarm, or scheduled task |
| 3 | express_feeling | User is sharing emotions or feelings |
| 4 | seek_advice | User is asking for guidance or recommendations |
| 5 | share_news | User is sharing news, updates, or events |
| 6 | reflect | User is reflecting on past experiences or contemplating |
| 7 | other | General conversation or unclear intent |

### 4.2 Ingress Labels (12 domains)

| ID | Label | Description (for embedding init) |
|----|-------|----------------------------------|
| 0 | DIARY | Personal journal entries, daily reflections |
| 1 | TASK | To-do items, reminders, action items |
| 2 | HEALTH | Medical, wellness, fitness, mental health |
| 3 | FINANCE | Money, bills, budgets, expenses |
| 4 | RELATIONSHIP | Family, friends, social connections |
| 5 | WORK | Professional, career, job-related |
| 6 | META | Questions about the app, system, features |
| 7 | MEMORY | Recalling past events, nostalgia |
| 8 | PLANNING | Future plans, goals, scheduling |
| 9 | CELEBRATION | Achievements, milestones, happy events |
| 10 | CONCERN | Worries, problems, issues |
| 11 | GRATITUDE | Thanks, appreciation, positive acknowledgment |

---

## 5. Architecture: SOTA vs Current

### 5.1 Current Intent/Ingress Heads (Simple)

```
[CLS] (768-dim)
    |
Linear(768 -> num_labels)
    |
Softmax -> single label
```

**Problems:**

- Fixed labels - adding new ones requires architecture change
- No semantic understanding of label meanings
- Single-label only

### 5.2 SOTA Label-Description Head

```
                    [CLS] (768-dim)
                         |
                    Query Projection (768 -> 768)
                         |
         +---------------+---------------+
         |               |               |
    Label Embed 0   Label Embed 1   ... Label Embed N
    "log_memory"    "query_memory"      (learnable)
         |               |               |
         +-------+-------+-------+-------+
                 |
         Cosine Similarity per Label
                 |
         Sigmoid (independent per label)
                 |
         Multi-Label Prediction
```

**Benefits:**

- **Future-proof**: Add new labels by just adding embeddings
- **Zero-shot capable**: Initialize from text descriptions
- **Multi-label**: Natural for FamilyOS where utterances have multiple intents

---

## 6. Implementation Plan (Simple)

### Files to Touch

| File | Action | Purpose |
|------|--------|---------|
| `src/modeling_studio/models/heads.py` | ADD | `LabelDescriptionHead`, `IntentHeadV2`, `IngressHeadV2` |
| `scripts/training/train_globalpointer_unified.py` | ADAPT | Copy and modify for intent/ingress |
| `configs/training/globalpointer_heads.yaml` | ADAPT | Copy and modify for intent/ingress |

### Step-by-Step

#### Step 1: Add Head Classes to `heads.py`

Add `LabelDescriptionHead` base class and `IntentHeadV2`, `IngressHeadV2` subclasses.

#### Step 2: Create Config

Copy `configs/training/globalpointer_heads.yaml` to `configs/training/intent_ingress_heads.yaml` and modify:

- Change `heads.enabled` from NER heads to `[intent, ingress]`
- Point to intent/ingress data paths
- Adjust training params for classification (vs span extraction)

#### Step 3: Adapt Training Script

Copy `scripts/training/train_globalpointer_unified.py` to `scripts/training/train_intent_ingress_unified.py` and modify:

- Replace GlobalPointer head creation with LabelDescriptionHead
- Replace span-label collator with classification collator
- Keep the freeze/load/save logic intact

#### Step 4: Run Training

```bash
python scripts/training/train_intent_ingress_unified.py \
    --config configs/training/intent_ingress_heads.yaml
```

---

## 7. Training Configuration

### `configs/training/intent_ingress_heads.yaml`

```yaml
# =============================================================================
# Intent/Ingress Head Training Configuration
# =============================================================================
# Follows same pattern as GlobalPointer training:
# Load checkpoint, freeze encoder, replace heads, train, save unified.

encoder:
  # Checkpoint with trained GlobalPointer NER
  checkpoint: checkpoints/checkpoint-8000
  freeze: true
  hidden_size: 768

heads:
  enabled:
    - intent
    - ingress

  # Which heads to keep frozen (already trained)
  frozen:
    - ner_general
    - ner_family
    - temporal
    - emotions
    - sentiment
    - safety_generic
    - safety_familyos
    - nli
    - embedding
    - relation

  architecture:
    type: label_description
    projection_dim: 768
    temperature: 0.07
    dropout: 0.1

  intent:
    num_labels: 8
    multi_label: false  # Start with single-label, can enable multi later
    threshold: 0.5

  ingress:
    num_labels: 12
    multi_label: false  # Start with single-label
    threshold: 0.5

training:
  learning_rate: 5.0e-4
  weight_decay: 0.01
  num_epochs: 3
  batch_size: 64
  warmup_steps: 500
  eval_steps: 500
  save_steps: 1000
  logging_steps: 100

data:
  intent:
    train: data/processed/intent_unified/train.jsonl
    val: data/processed/intent_unified/val.jsonl
  ingress:
    train: data/processed/ingress_unified/train.jsonl
    val: data/processed/ingress_unified/val.jsonl
  max_length: 256

output:
  dir: outputs/intent-ingress-unified-v1
  save_total_limit: 3
```

---

## 8. Expected Outcomes

### 8.1 Accuracy Targets

| Metric | Current | Target | Notes |
|--------|---------|--------|-------|
| Intent Accuracy | 81.8% | **90%+** | Primary intent |
| Ingress Accuracy | 69.7% | **85%+** | Primary domain |
| MEMORY Recall | 33.3% | **75%+** | Major improvement |

### 8.2 Future-Proofing

| Scenario | Old Architecture | New Architecture |
|----------|-----------------|------------------|
| Add "schedule_meeting" intent | Retrain from scratch | Add 1 embedding, fine-tune 1 epoch |
| Add "EDUCATION" domain | Change output dim, retrain | Add 1 embedding, fine-tune 1 epoch |
| Zero-shot new label | Impossible | Initialize from text description |

---

## 9. Timeline

| Task | Status | Notes |
|------|--------|-------|
| Data exploration | DONE | 570K+ records discovered |
| Data consolidation | DONE | Balanced shards created |
| Add head classes to `heads.py` | TODO | ~200 lines |
| Create training config | TODO | Copy + modify existing |
| Adapt training script | TODO | Copy + modify existing |
| Run training | TODO | ~2-4 hours |
| Validate results | TODO | Run demo tests |

---

## 10. Success Criteria

- [ ] Intent accuracy >= 90% on validation set
- [ ] Ingress accuracy >= 85% on validation set
- [ ] MEMORY recall >= 75%
- [ ] Zero log_memory/set_reminder confusion on clear cases
- [ ] Can add new label with < 1 hour fine-tune
- [ ] P95 latency unchanged (< 25ms)
- [ ] Unified checkpoint saved with all 12 capabilities

---

## Appendix A: LabelDescriptionHead Architecture

```python
class LabelDescriptionHead(nn.Module):
    """
    SOTA classification head using learnable label embeddings.

    Future-proof: Add new labels by just adding their embeddings.
    Supports both single-label (softmax) and multi-label (sigmoid).
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_labels: int = 8,
        projection_dim: int = 768,
        temperature: float = 0.07,
        multi_label: bool = False,
        dropout: float = 0.1,
        label_names: list[str] | None = None,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.multi_label = multi_label
        self.label_names = label_names or [f"label_{i}" for i in range(num_labels)]

        # Query projection
        self.query_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, projection_dim),
            nn.LayerNorm(projection_dim),
        )

        # Learnable label embeddings
        self.label_embeddings = nn.Parameter(
            torch.randn(num_labels, projection_dim) * 0.02
        )

        # Learnable temperature
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, hidden_states, attention_mask=None, labels=None):
        # Get [CLS] token
        cls_hidden = hidden_states[:, 0]  # (B, 768)

        # Project query
        query = self.query_proj(cls_hidden)  # (B, proj_dim)
        query = F.normalize(query, dim=-1)

        # Normalize label embeddings
        label_emb = F.normalize(self.label_embeddings, dim=-1)

        # Compute similarity logits
        logits = torch.matmul(query, label_emb.T) / self.temperature.clamp(min=0.01)

        # Compute loss
        loss = None
        if labels is not None:
            if self.multi_label:
                loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            else:
                loss = F.cross_entropy(logits, labels)

        return {"logits": logits, "loss": loss}
```

---

## Appendix B: Comparison with GlobalPointer Training

| Aspect | GlobalPointer Training | Intent/Ingress Training |
|--------|----------------------|------------------------|
| Source checkpoint | checkpoint-18000 | checkpoint-8000 |
| Heads to replace | ner_general, ner_family, temporal | intent, ingress |
| New head type | GlobalPointerNERHead | LabelDescriptionHead |
| Data format | Span labels (start, end, type) | Classification labels |
| Collator | GlobalPointerCollator | ClassificationCollator |
| Loss | GlobalPointer loss | Cross-entropy / BCE |
| Everything else | SAME | SAME |

The training infrastructure is identical - only the head architecture and data format differ.
