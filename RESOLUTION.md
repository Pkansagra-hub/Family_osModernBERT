# Resolution Plan: Fix Broken Stage B Training Tasks

**Date:** December 6, 2025
**Priority:** CRITICAL - Blocking Stage B Training
**Estimated Effort:** 4-6 hours total

---

## Executive Summary

Two tasks are broken in the Stage B training pipeline:

| Task | Issue | Impact |
|------|-------|--------|
| `relations` | Loader has `pass` - no data extracted | 0% of relation data used |
| `embedding` | Separate triplet format not integrated | FamilyOS embeddings not trained |

Both must be fixed before Stage B training can properly train encoder layers 15-20 for v3 cloning.

---

## Milestone 1: Relations Task Integration ✅ COMPLETE

**Goal:** Enable sentence-level relation classification for Stage B training
**Status:** COMPLETE - 173,477 training samples, 19,356 eval samples

### Epic 1.1: Data Format Analysis

**Issue 1.1.1: Audit Current Relation Data Format**

- [x] Check data structure in unified files
- [x] Identify format: `{"subject": "user", "predicate": "aunt_uncle_of", "object": "David"}`
- [x] Problem: No entity span positions (start/end indices)
- [x] Decision: Use sentence-level multi-label classification (predicate types present)

**Issue 1.1.2: Verify RELATION_LABELS Schema**

- [x] Confirm 15 relation types exist in `labels.py`
- [x] Labels: `no_relation, parent_of, child_of, spouse_of, sibling_of, grandparent_of, grandchild_of, aunt_uncle_of, niece_nephew_of, cousin_of, pet_of, friend_of, colleague_of, lives_at, owns`

### Epic 1.2: Loader Implementation

**Issue 1.2.1: Implement `_extract_task_data` for relations**

- [x] COMPLETED - Relations extraction now works
- [x] 440 samples loaded from test data

File: `src/modeling_studio/data/loaders.py` (line ~4350)

Current code:

```python
elif task == "relations":
    # TODO: Implement relation extraction format
    pass
```

New implementation:

```python
elif task == "relations":
    # Sentence-level relation classification
    # Extract predicate types as multi-label (can have multiple relations)
    if not label_value or not isinstance(label_value, list):
        continue

    # Convert list of relations to multi-hot vector
    try:
        labels = _relations_to_multihot(label_value)
        if sum(labels) > 0:  # At least one relation
            task_data.append({"text": text, "labels": labels, "task": task})
    except Exception as e:
        logger.debug(f"Error processing relations: {e}")
```

**Issue 1.2.2: Add `_relations_to_multihot` helper function**

```python
def _relations_to_multihot(relations_list: list[dict]) -> list[int]:
    """Convert list of relation dicts to multi-hot vector."""
    from modeling_studio.data.labels import RELATION_LABELS

    multihot = [0] * RELATION_LABELS.num_labels
    for rel in relations_list:
        if not isinstance(rel, dict):
            continue
        predicate = rel.get("predicate", "")
        if predicate:
            try:
                idx = RELATION_LABELS.encode(predicate)
                multihot[idx] = 1
            except KeyError:
                logger.warning(f"Unknown relation predicate: {predicate}")
    return multihot
```

### Epic 1.3: Tokenization & Model Integration

**Issue 1.3.1: Add `relations` to TASK_TYPE_MAP**

File: `src/modeling_studio/data/loaders.py` (in `_apply_tokenization`)

```python
TASK_TYPE_MAP = {
    ...
    "relations": "multilabel",  # Sentence-level multi-label classification
}
```

**Issue 1.3.2: Verify RelationHead supports multi-label**

File: `src/modeling_studio/models/heads.py`

- Check if `RelationHead` can handle multi-label (no entity spans)
- If not, add fallback to CLS-based classification when no spans provided
- Current code already has fallback: `entity1_repr = hidden_states[:, 0, :]`

**Issue 1.3.3: Update config naming consistency**

File: `configs/training/multitask/stage_b_for_v3_prep.yaml`

- [x] Changed `relation` → `relations` in `familyos_tasks`
- [x] Changed `relation` → `relations` in `task_weights`
- [ ] Verify head name in model matches (may need mapping)

### Epic 1.4: Testing

**Issue 1.4.1: Test loader extracts relations**

```bash
python -c "
from modeling_studio.data.loaders import load_familyos_unified
ds = load_familyos_unified(
    data_dirs=['data/familyos/unified/output_synthetic_healed'],
    tasks=['relations'],
    max_samples=100,
)
print(f'Relations: {len(ds.get(\"relations\", []))} samples')
"
```

**Issue 1.4.2: Test end-to-end training loop**

```bash
python scripts/train_stage_b.py \
    --config configs/training/multitask/stage_b_for_v3_prep.yaml \
    --dry_run
```

---

## Milestone 2: Embedding Task Integration ✅ COMPLETE

**Goal:** Load FamilyOS synthetic embedding triplets for Stage B training
**Status:** COMPLETE - 185,085 train / 20,567 eval embedding triplets

### Epic 2.1: Data Format Analysis ✅

**Issue 2.1.1: Audit embedding triplet format** ✅

Location: `data/familyos/embeddings/silver_synthetic/`

Format verified:

```json
{
  "anchor": "Had a lovely dinner with my parents last Sunday",
  "positive": "Family meal on the weekend was wonderful",
  "negative": "The quarterly budget meeting was rescheduled",
  "cluster": "family_meals",
  "difficulty": "hard"
}
```

**Issue 2.1.2: Verify data exists and count**

```bash
python -c "
from pathlib import Path
import json
triplets = list(Path('data/familyos/embeddings/silver_synthetic').glob('*.jsonl'))
print(f'Triplet files: {len(triplets)}')
total = sum(1 for f in triplets for _ in open(f))
print(f'Total triplets: {total}')
"
```

### Epic 2.2: Loader Implementation

**Issue 2.2.1: Add `load_embedding_triplets` function**

File: `src/modeling_studio/data/loaders.py`

```python
def load_embedding_triplets(
    data_dir: str | Path,
    split: str = "train",
    validation_ratio: float = 0.1,
    seed: int = 42,
    max_samples: int | None = None,
) -> Dataset:
    """
    Load embedding triplets for contrastive learning.

    Returns Dataset with columns:
        - anchor: str
        - positive: str
        - negative: str
        - cluster: str (optional)
    """
    import random

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Embedding data not found: {data_dir}")

    # Load all triplets
    triplets = []
    for jsonl_file in sorted(data_dir.glob("*.jsonl")):
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        triplet = json.loads(line)
                        if all(k in triplet for k in ["anchor", "positive", "negative"]):
                            triplets.append(triplet)
                    except json.JSONDecodeError:
                        continue

    logger.info(f"Loaded {len(triplets)} embedding triplets from {data_dir}")

    # Apply max_samples
    if max_samples and len(triplets) > max_samples:
        random.seed(seed)
        triplets = random.sample(triplets, max_samples)

    # Split train/val
    random.seed(seed)
    random.shuffle(triplets)
    val_size = int(len(triplets) * validation_ratio)

    if split == "train":
        triplets = triplets[val_size:]
    elif split == "validation":
        triplets = triplets[:val_size]

    # Convert to Dataset
    return Dataset.from_list([
        {
            "anchor": t["anchor"],
            "positive": t["positive"],
            "negative": t["negative"],
            "cluster": t.get("cluster", ""),
            "task": "embedding",
        }
        for t in triplets
    ])
```

**Issue 2.2.2: Integrate into `load_familyos_unified_for_training`**

Add embedding loading to the unified training loader:

```python
def load_familyos_unified_for_training(...):
    ...

    # Load embedding triplets if configured
    embedding_config = config.get("embedding_familyos", {})
    if embedding_config.get("enabled", False):
        embedding_dir = embedding_config.get("data_dir")
        if embedding_dir:
            train_datasets["embedding"] = load_embedding_triplets(
                data_dir=embedding_dir,
                split="train",
                validation_ratio=validation_ratio,
                seed=seed,
            )
            val_datasets["embedding"] = load_embedding_triplets(
                data_dir=embedding_dir,
                split="validation",
                validation_ratio=validation_ratio,
                seed=seed,
            )
```

### Epic 2.3: Tokenization for Triplets

**Issue 2.3.1: Update `_apply_tokenization` for embedding triplets**

```python
elif mapped_task == "embedding":
    def tokenize_wrapper(example):
        # Tokenize anchor, positive, negative separately
        anchor_enc = tokenizer(
            example["anchor"],
            max_length=max_length,
            truncation=True,
            padding=False,
        )
        positive_enc = tokenizer(
            example["positive"],
            max_length=max_length,
            truncation=True,
            padding=False,
        )
        negative_enc = tokenizer(
            example["negative"],
            max_length=max_length,
            truncation=True,
            padding=False,
        )

        return {
            "anchor_input_ids": anchor_enc["input_ids"],
            "anchor_attention_mask": anchor_enc["attention_mask"],
            "positive_input_ids": positive_enc["input_ids"],
            "positive_attention_mask": positive_enc["attention_mask"],
            "negative_input_ids": negative_enc["input_ids"],
            "negative_attention_mask": negative_enc["attention_mask"],
            "task": "embedding",
        }
```

### Epic 2.4: Trainer Integration

**Issue 2.4.1: Verify `_compute_embedding_loss` handles triplets**

File: `src/modeling_studio/trainers/multitask_trainer.py`

Check that the trainer can handle the triplet format and compute TripletMarginLoss.

**Issue 2.4.2: Add triplet collation**

Ensure the data collator properly pads triplet batches.

### Epic 2.5: Testing

**Issue 2.5.1: Test triplet loading**

```bash
python -c "
from modeling_studio.data.loaders import load_embedding_triplets
ds = load_embedding_triplets('data/familyos/embeddings/silver_synthetic')
print(f'Loaded {len(ds)} triplets')
print(f'Sample: {ds[0]}')
"
```

**Issue 2.5.2: Test embedding in training loop**

```bash
python scripts/train_stage_b.py \
    --config configs/training/multitask/stage_b_for_v3_prep.yaml \
    --debug --dry_run
```

---

## Milestone 3: Data Healing (If Needed)

### Epic 3.1: Relations Data Healing

**Issue 3.1.1: Check if relations data needs healing**

Some records may have:

- Empty relations: `[]`
- Invalid predicates not in schema
- Missing required fields

**Issue 3.1.2: Create `scripts/heal_relations_data.py`**

Only if needed after testing reveals issues.

```python
#!/usr/bin/env python
"""
Heal relations data to ensure all predicates match RELATION_LABELS schema.
"""

VALID_PREDICATES = {
    "no_relation", "parent_of", "child_of", "spouse_of", "sibling_of",
    "grandparent_of", "grandchild_of", "aunt_uncle_of", "niece_nephew_of",
    "cousin_of", "pet_of", "friend_of", "colleague_of", "lives_at", "owns"
}

def heal_relations(input_dir, output_dir):
    # Read all shards
    # Validate predicates
    # Fix or skip invalid
    # Write healed output
    pass
```

### Epic 3.2: Embedding Data Healing

**Issue 3.2.1: Verify embedding triplets are valid**

Already have `scripts/agents/embedding_data_healer.py` - use it:

```bash
python scripts/agents/embedding_data_healer.py \
    --input-dir data/familyos/embeddings/silver_synthetic \
    --dry-run
```

---

## Implementation Order

| Step | Task | Effort | Blocker |
|------|------|--------|---------|
| 1 | Implement `_relations_to_multihot` | 15 min | None |
| 2 | Implement relations extraction in `_extract_task_data` | 15 min | Step 1 |
| 3 | Add `relations` to TASK_TYPE_MAP | 5 min | Step 2 |
| 4 | Test relations loading | 10 min | Step 3 |
| 5 | Implement `load_embedding_triplets` | 30 min | None |
| 6 | Integrate embedding into unified loader | 20 min | Step 5 |
| 7 | Update embedding tokenization | 20 min | Step 6 |
| 8 | Test embedding loading | 10 min | Step 7 |
| 9 | Run full dry-run test | 15 min | Steps 4, 8 |
| 10 | Fix any issues found | 30-60 min | Step 9 |

**Total: 3-4 hours**

---

## Success Criteria

1. **Relations**: `load_familyos_unified(tasks=['relations'])` returns non-zero samples
2. **Embeddings**: `load_embedding_triplets()` returns triplets with anchor/positive/negative
3. **Dry Run**: `train_stage_b.py --dry_run` completes without errors
4. **All 8 Tasks**: Loader returns data for all 8 FamilyOS tasks

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/modeling_studio/data/loaders.py` | Add relations extraction, add `load_embedding_triplets`, update `_apply_tokenization` |
| `src/modeling_studio/data/labels.py` | (Verify RELATION_LABELS - no changes expected) |
| `scripts/train_stage_b.py` | (May need to call embedding loader) |
| `configs/training/multitask/stage_b_for_v3_prep.yaml` | (Already updated) |

---

## Milestone 3: Embedding Score Normalization Fix ✅ COMPLETE

**Goal:** Fix Stage A embedding learning by normalizing STS scores to consistent [0, 1] range
**Status:** COMPLETE - All embedding datasets now properly normalized

### Issue Analysis

**Problem Identified:**
Stage A embedding training showed Spearman correlation stuck at ~0.32 despite training. Investigation revealed:

1. Different STS datasets have different score scales:
   - `sentence-transformers/stsb`: 0-1 (already normalized)
   - `sentence-transformers/all-nli`: 0-1 (already normalized)
   - `mteb/sickr-sts`: 1-5 (raw STS scores)
   - `mteb/sts12-sts`: 0-5 (raw STS scores)
   - `mteb/sts13-sts`: 0-5 (raw STS scores)
   - `mteb/sts14-sts`: 0-5 (raw STS scores)

2. The loss function had per-batch normalization:
   ```python
   if labels.max() > 1.0:
       labels = labels / 5.0
   ```
   This was incorrect because batches with only 0-1 scores wouldn't be normalized, while mixed batches would have incorrect normalization applied to already-normalized samples.

**Solution Applied:**

1. **Normalize during data loading** (in `_apply_tokenization`):
   ```python
   # In embedding tokenize_wrapper
   if score is not None:
       # Normalize if score > 1 (assumes 0-5 scale)
       if score > 1.0:
           score = score / 5.0
       # Clamp to [0, 1] range for safety
       score = max(0.0, min(1.0, score))
       result["labels"] = score
   ```

2. **Remove per-batch normalization** from `_compute_embedding_loss`:
   ```python
   # Before: normalized per-batch (incorrect)
   if labels.max() > 1.0:
       labels = labels / 5.0
   loss = F.mse_loss(cos_sim, labels)

   # After: data is pre-normalized during loading
   loss = F.mse_loss(cos_sim, labels)
   ```

**Verification:**
After fix, label distribution across 124K samples:
- 40.7% at 1.0 (perfect matches)
- 31.7% at 0.0-0.2 (low similarity)
- 27.1% at 0.4-0.6 (medium similarity)
- Mean: 0.5489, Range: [0.0, 1.0]

**Files Changed:**
- `src/modeling_studio/data/loaders.py`: Score normalization in `tokenize_wrapper`
- `src/modeling_studio/trainers/multitask_trainer.py`: Remove per-batch normalization

---

## Notes

- **Relations Simplification**: Using sentence-level multi-label instead of span-based extraction. v3 can upgrade to span-based using Pair Encoder later.
- **Embedding Triplets**: Generated separately by `synthetic_embedding_generator.py`, stored in different location from unified data.
- **No Data Regeneration Needed**: Existing data format is usable with loader fixes.
- **Score Normalization**: All STS datasets now normalized to [0, 1] during loading for consistent loss computation.
