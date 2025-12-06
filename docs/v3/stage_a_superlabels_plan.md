# Stage A: Super-Label Training Implementation Plan

> **Epic**: Enable 7 Super-Label Curriculum Training for Stage A Foundation Model
> **Status**: In Progress
> **Created**: 2025-12-06
> **Updated**: 2025-12-06 (Post Staff Review)

---

## Executive Summary

The current codebase is configured to train on 44 granular FamilyOS emotions, but the `TRAINING_STRATEGY.md` specifies a curriculum learning approach where **Stage A trains on 7 super-labels** before Stage B fine-tunes on the full 44 labels.

This plan outlines the implementation work to enable super-label training.

### Staff Engineer Review Notes (2025-12-06)

1. **Milestone 2 DELETED** - Runtime label mapping is sub-optimal. Instead, pre-process data to `train_CONSOLIDATED.jsonl` with super-labels already applied.
2. **Head Transplant Awareness** - Stage A emotion head `[768, 7]` will be DISCARDED in Stage B. Stage B initializes fresh `[768, 44]` head. The encoder (layers 1-22) transfers knowledge.
3. **Surprise Mapping Fix** - Changed `surprise` from NEUTRAL to JOY per Plutchik wheel (high-energy, positive in FamilyOS context).

---

## Epic: Stage A Super-Label Training

### Goal

Train ModernBERT-base on 7 broad emotion super-labels to build a robust foundation before fine-tuning on 44 granular emotions in Stage B.

### Success Criteria

- [x] Model trains on 7 super-labels instead of 44
- [ ] Super-label F1 > 0.85 on validation set
- [ ] Clean separation between super-label clusters in embedding space
- [ ] Stage B can load Stage A checkpoint and fine-tune on 44 labels

---

## Milestones

### Milestone 1: Schema & Mapping Infrastructure ✅ COMPLETE

**Goal**: Define the 7 super-labels and create mapping from 44 emotions

| Issue | Title | Priority | Status |
|-------|-------|----------|--------|
| #1.1 | Create `EMOTIONS_SUPER_LABELS` schema in `labels.py` | P0 | ✅ Done |
| #1.2 | Create `EMOTION_TO_SUPER_LABEL` mapping dict | P0 | ✅ Done |
| #1.3 | Add `map_to_super_labels()` utility function | P0 | ✅ Done |
| #1.4 | Write unit tests for super-label mapping | P1 | ✅ Done (36 tests) |

**Note**: `surprise` maps to `JOY` (not NEUTRAL) per Plutchik wheel review.

---

### ~~Milestone 2: Data Loader Updates~~ ❌ DELETED

> **Rationale**: Runtime label mapping in Python loops slows GPU pipeline.
> **Better approach**: Pre-process data to `train_CONSOLIDATED.jsonl` with super-labels.
> See Milestone 2A below for the replacement task.

---

### Milestone 2A: Data Pre-Processing (Replaces Milestone 2)

**Goal**: Create consolidated training data with super-labels pre-applied

| Issue | Title | Priority | Estimate |
|-------|-------|----------|----------|
| #2A.1 | Create `consolidate_labels.py` script | P0 | 1h |
| #2A.2 | Generate `train_CONSOLIDATED.jsonl` from silver data | P0 | 30m |
| #2A.3 | Verify super-label distribution in consolidated file | P1 | 30m |

**Script should**:
- Read `data/familyos/emotions/silver/train.jsonl`
- Apply `map_emotion_names_to_super_labels()` to each row
- Output `data/familyos/emotions/silver/train_CONSOLIDATED.jsonl`

---

### Milestone 3: Configuration Updates ✅ COMPLETE

**Goal**: Update YAML configs to point to consolidated data and use 7 labels

| Issue | Title | Priority | Status |
|-------|-------|----------|--------|
| #3.1 | Update `stage_a_a100_fast.yaml` - set `num_labels: 7` for emotions | P0 | ✅ Done |
| #3.2 | Update `stage_a_datasets.yaml` - add `emotions_super` dataset | P0 | ✅ Done |
| #3.3 | Create `stage_a_superlabel.yaml` variant config (optional) | P2 | Skipped |

**Changes Made**:
- `stage_a_a100_fast.yaml`: `emotions.num_labels: 44` → `7` with comment
- `stage_a_datasets.yaml`: Added `emotions_super` dataset with `label_schema: emotions_super` and `splits.train: train_CONSOLIDATED`
- Disabled `emotions_familyos_silver` to avoid loading wrong schema

---

### Milestone 4: Training Script & Model Compatibility

**Goal**: Ensure training script and model work with 7-label emotions head

| Issue | Title | Priority | Estimate |
|-------|-------|----------|----------|
| #4.1 | Verify config accepts `num_labels: 7` for emotions head | P0 | 30m |
| #4.2 | Update evaluation metrics for 7-class emotions | P1 | 1h |
| #4.3 | Add super-label confusion matrix logging to W&B | P2 | 1h |

**Critical Note**: Stage B will DISCARD the Stage A emotion head and initialize a fresh 44-class head. Only the encoder weights transfer.

---

### Milestone 5: Validation & Testing

**Goal**: End-to-end validation of super-label training pipeline

| Issue | Title | Priority | Estimate |
|-------|-------|----------|----------|
| #5.1 | Dry-run training with `--debug` flag | P0 | 30m |
| #5.2 | Verify multi-hot conversion (44→7) is correct | P0 | 1h |
| #5.3 | Run full Stage A training and evaluate | P0 | 4h |
| #5.4 | Document super-label training results | P1 | 1h |

---

## Issue Details

### Issue #1.1: Create `EMOTIONS_SUPER_LABELS` schema in `labels.py`

**File**: `src/modeling_studio/data/labels.py`

**Description**:
Add a new `LabelSchema` for the 7 super-labels:

```python
EMOTIONS_SUPER_LABELS = LabelSchema(
    name="emotions_super",
    label2id={
        "JOY": 0,           # joy, excitement, celebration, pride, relief, amusement
        "AFFECTION": 1,     # love, warmth, caring, gratitude, tenderness, admiration
        "SADNESS": 2,       # sadness, grief, disappointment, longing, emptiness
        "ANXIETY": 3,       # worry, overwhelmed, frustration, annoyance, nervousness, fear, anger
        "NOSTALGIA": 4,     # nostalgia, bittersweet, homesickness
        "CONTENTMENT": 5,   # contentment, belonging, togetherness, patience
        "NEUTRAL": 6,       # neutral
    },
    problem_type="multi_label_classification",
    description="7 super-label emotion categories for Stage A curriculum learning",
)
```

**Acceptance Criteria**:

- [ ] Schema defined with 7 labels
- [ ] Includes docstring explaining curriculum learning purpose
- [ ] Exported in `__all__`

---

### Issue #1.2: Create `EMOTION_TO_SUPER_LABEL` mapping dict

**File**: `src/modeling_studio/data/labels.py`

**Description**:
Create a mapping from each of the 44 FamilyOS emotions to their super-label:

```python
EMOTION_TO_SUPER_LABEL: dict[str, str] = {
    # JOY cluster
    "joy": "JOY",
    "excitement": "JOY",
    "celebration": "JOY",
    "pride": "JOY",
    "relief": "JOY",
    "amusement": "JOY",
    "hope": "JOY",
    "optimism": "JOY",

    # AFFECTION cluster
    "love": "AFFECTION",
    "warmth": "AFFECTION",
    "caring": "AFFECTION",
    "gratitude": "AFFECTION",
    "tenderness": "AFFECTION",
    "admiration": "AFFECTION",
    "parental_pride": "AFFECTION",
    "protectiveness": "AFFECTION",
    "playfulness": "AFFECTION",

    # SADNESS cluster
    "sadness": "SADNESS",
    "grief": "SADNESS",
    "disappointment": "SADNESS",
    "longing": "SADNESS",
    "emptiness": "SADNESS",
    "remorse": "SADNESS",
    "parental_guilt": "SADNESS",

    # ANXIETY cluster
    "worry": "ANXIETY",
    "overwhelmed": "ANXIETY",
    "frustration": "ANXIETY",
    "annoyance": "ANXIETY",
    "nervousness": "ANXIETY",
    "fear": "ANXIETY",
    "anger": "ANXIETY",
    "disgust": "ANXIETY",
    "disapproval": "ANXIETY",
    "embarrassment": "ANXIETY",

    # NOSTALGIA cluster
    "nostalgia": "NOSTALGIA",
    "bittersweet": "NOSTALGIA",
    "homesickness": "NOSTALGIA",

    # CONTENTMENT cluster
    "contentment": "CONTENTMENT",
    "belonging": "CONTENTMENT",
    "togetherness": "CONTENTMENT",
    "patience": "CONTENTMENT",
    "approval": "CONTENTMENT",

    # NEUTRAL cluster
    "neutral": "NEUTRAL",
    # Note: 'surprise' maps to JOY (positive family context per Plutchik wheel)
    "surprise": "JOY",
}
```

**Acceptance Criteria**:

- [x] All 44 FamilyOS emotions mapped to one of 7 super-labels
- [x] Mapping aligns with `TRAINING_STRATEGY.md` taxonomy
- [x] No emotion left unmapped
- [x] `surprise` maps to JOY (staff review correction)

---

### Issue #1.3: Add `map_to_super_labels()` utility function

**File**: `src/modeling_studio/data/labels.py`

**Description**:
Create a function to convert 44-label multi-hot vector to 7-label:

```python
def map_to_super_labels(
    multi_hot_44: list[int],
    source_schema: LabelSchema = EMOTIONS_FAMILYOS_LABELS,
    target_schema: LabelSchema = EMOTIONS_SUPER_LABELS,
) -> list[int]:
    """
    Convert a 44-label multi-hot vector to 7 super-label multi-hot.

    Args:
        multi_hot_44: Multi-hot vector of length 44
        source_schema: Source label schema (44 labels)
        target_schema: Target label schema (7 super-labels)

    Returns:
        Multi-hot vector of length 7

    Example:
        >>> labels_44 = [0] * 44
        >>> labels_44[1] = 1  # joy
        >>> labels_44[6] = 1  # love
        >>> super_labels = map_to_super_labels(labels_44)
        >>> super_labels  # [1, 1, 0, 0, 0, 0, 0] -> JOY + AFFECTION
    """
    multi_hot_7 = [0] * target_schema.num_labels

    for idx, val in enumerate(multi_hot_44):
        if val == 1:
            emotion_name = source_schema.id2label[idx]
            super_label = EMOTION_TO_SUPER_LABEL.get(emotion_name)
            if super_label:
                super_idx = target_schema.label2id[super_label]
                multi_hot_7[super_idx] = 1

    return multi_hot_7
```

**Acceptance Criteria**:

- [ ] Function handles all 44 emotions
- [ ] Correctly produces 7-element multi-hot output
- [ ] Handles edge cases (empty labels, unknown labels)

---

### ~~Issue #2.1: Add `use_super_labels` parameter to data loaders~~ ❌ DELETED

> **Deleted per Staff Review**: Use pre-processed data instead of runtime mapping.

---

### Issue #2A.1: Create `consolidate_labels.py` script (NEW)

**File**: `scripts/consolidate_labels.py`

**Description**:
Create a script to pre-process the emotions data with super-labels:

```python
#!/usr/bin/env python
"""
Consolidate 44 FamilyOS emotions into 7 super-labels.

Reads: data/familyos/emotions/silver/train.jsonl
Writes: data/familyos/emotions/silver/train_CONSOLIDATED.jsonl

Each output row has:
    {"text": "...", "labels": [1, 0, 0, 0, 1, 0, 0]}  # Multi-hot 7 labels
"""
import json
from pathlib import Path
from modeling_studio.data.labels import map_emotion_names_to_super_labels, EMOTIONS_SUPER_LABELS

def consolidate(input_path: Path, output_path: Path):
    with open(input_path) as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            row = json.loads(line)
            emotions = row.get("emotions", [])
            super_labels = map_emotion_names_to_super_labels(emotions)
            f_out.write(json.dumps({
                "text": row["text"],
                "labels": super_labels,
            }) + "\n")
```

**Acceptance Criteria**:

- [x] Script reads silver train.jsonl
- [x] Applies `map_emotion_names_to_super_labels()`
- [x] Outputs train_CONSOLIDATED.jsonl with 7-element multi-hot labels

---

### Issue #3.1: Update `stage_a_a100_fast.yaml`

**File**: `configs/training/multitask/stage_a_a100_fast.yaml`

**Changes**:

```yaml
heads:
  emotions:
    enabled: true
    type: sequence_classification
    num_labels: 7  # CHANGED from 44 - Stage A uses super-labels
    problem_type: multi_label_classification
    dropout: 0.1
```

**Acceptance Criteria**:

- [ ] `num_labels` changed to 7
- [ ] Comment updated to explain Stage A uses super-labels

---

### Issue #3.2: Update `stage_a_datasets.yaml`

**File**: `configs/data/multitask/stage_a_datasets.yaml`

**Changes**:

```yaml
emotions_familyos_silver:
  task: emotions
  source: local
  name: data/familyos/emotions/silver/train_CONSOLIDATED.jsonl  # Pre-processed with super-labels
  label_schema: emotions_super
  format: jsonl
  splits:
    train: train
```

**Acceptance Criteria**:

- [ ] Points to `train_CONSOLIDATED.jsonl`
- [ ] `label_schema` set to `emotions_super`

---

## Implementation Order (REVISED)

```text
Day 1: ✅ Milestone 1 (Schema & Mapping) ─────────► COMPLETE
Day 2: Milestone 2A (Data Pre-Processing) ────────► Issues #2A.1, #2A.2, #2A.3
Day 3: Milestone 3 (Configs) ─────────────────────► Issues #3.1, #3.2
Day 4: Milestone 4 (Model Compatibility) ─────────► Issues #4.1, #4.2
Day 5: Milestone 5 (Validation) ──────────────────► Issues #5.1, #5.2, #5.3
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Super-label mapping loses granular info | Low | Medium | Multi-hot preserves co-occurrence (JOY+AFFECTION) |
| Class imbalance in super-labels | Medium | Low | Check distribution, apply class weights if needed |
| Stage B fine-tuning from Stage A fails | Low | High | Stage B discards emotion head, only encoder transfers |
| Training collapse on 7 labels | Low | High | Use plain BCE (already configured), monitor early |

---

## Architecture Note: Head Transplant

```text
Stage A Training:
├── Encoder (layers 1-22): Learns emotion representations
└── Emotion Head [768 → 7]: Disposable interface for super-labels

Stage B Training:
├── Encoder (layers 1-22): LOADED from Stage A checkpoint ✓
└── Emotion Head [768 → 44]: FRESH RANDOM initialization (head transplant)
```

The encoder learns rich emotion representations. The head is just a projection layer.

---

## Dependencies

- [x] FamilyOS emotions dataset in `data/familyos/emotions/silver/` (EXISTS)
- [x] `EMOTIONS_FAMILYOS_LABELS` in `labels.py` (EXISTS)
- [x] `EMOTIONS_SUPER_LABELS` in `labels.py` (CREATED)
- [x] `map_emotion_names_to_super_labels()` function (CREATED)
- [ ] `train_CONSOLIDATED.jsonl` (TO BE CREATED)
- [ ] Working training script `train_stage_a.py` (EXISTS)
- [ ] A100 GPU access for training (REQUIRED)

---

## Next Steps

1. ~~Start with Issue #1.1~~ ✅ COMPLETE
2. **Create `consolidate_labels.py` script** (Issue #2A.1)
3. **Generate `train_CONSOLIDATED.jsonl`** (Issue #2A.2)
4. **Update configs** (Milestone 3)
5. **Dry-run training** (Milestone 5)

---

## References

- [TRAINING_STRATEGY.md](../../TRAINING_STRATEGY.md) - Curriculum learning strategy
- [DATA_QUALITY_AUDIT.md](../DATA_QUALITY_AUDIT.md) - Super-label grouping proposal
- [labels.py](../../src/modeling_studio/data/labels.py) - Current label schemas
