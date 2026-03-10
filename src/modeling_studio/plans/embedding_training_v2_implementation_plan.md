# Embedding Training V2 - Full Implementation Plan

> Scope: Build unified data loading, slice-aware evaluation, and single-config training pipeline to train AgreementGatedHeadV2 on all 5 data folders in one run.
> Created: March 10, 2026
> Depends on: `sota_retrieval_architecture.md`, `synthetic_embedding_generator_v2_implementation_plan.md`
> Target scripts: `train_embedding_heads_bakeoff.py`, `train_embedding_head.py`
> Target config: `configs/training/embedding_heads_bakeoff.yaml`

---

## 0. Current State

### 0.1 What exists and works

| Component | Status | Location |
|---|---|---|
| AgreementGatedHeadV2 | Built, tested, registered | `src/modeling_studio/models/heads_embedding.py` |
| FamilyContrastiveLoss | Supports triplets AND pairs (in-batch negatives when `negatives=None`) | `src/modeling_studio/models/losses.py` |
| Bakeoff training script | Works for triplet-only data, 7 heads registered | `scripts/training/train_embedding_heads_bakeoff.py` |
| TripletDataset | Loads `*.jsonl` from directories, filter fixed to accept mined_v2 filenames | Both training scripts |
| Data: silver_synthetic | 261,805 triplets | `data/familyos/embeddings/silver_synthetic/` |
| Data: hard_negatives | 42,945 triplets | `data/familyos/embeddings/hard_negatives/` |
| Data: wrong_person | 4,437 triplets | `data/familyos/embeddings/mined_v2/wrong_person/` |
| Data: wrong_time | 5,460 triplets | `data/familyos/embeddings/mined_v2/wrong_time/` |
| Data: safety_emotion | 2,860 triplets | `data/familyos/embeddings/mined_v2/safety_emotion/` |
| Data: query_doc | 6,220 pairs (query+document, NO negative) | `data/familyos/embeddings/mined_v2/query_doc/` |

### 0.2 What is broken or missing

| Gap | Impact | Severity |
|---|---|---|
| `TripletDataset` silently drops query_doc pairs (no `negative` field) | 6,220 samples wasted, no asymmetric retrieval training | **Critical** |
| `TripletDataset` treats all data as one flat pool | No slice tracking, no balanced sampling, mined_v2 drowned by 261K silver | **High** |
| Config only lists 2 data paths (silver_synthetic, hard_negatives) | mined_v2 folders never loaded | **High** |
| No slice-aware evaluation | Cannot measure wrong_person accuracy vs wrong_time accuracy vs safety accuracy separately | **High** |
| No retrieval eval (Recall@k, MRR) for query_doc | Cannot validate asymmetric retrieval quality | **High** |
| `train_step` always passes explicit negatives | Pair-only samples cannot use in-batch negative path | **Medium** |
| No eval holdout per slice | Eval set is random 10% of training distribution (silver-dominated) | **Medium** |
| No sampling weights or slice balancing | 261K silver : 19K mined_v2 = 14:1 ratio drowns specialization | **Medium** |

---

## 1. Milestone Overview

```
Milestone 1: Unified Data Pipeline
    Epic 1.1: EmbeddingDataset (replaces TripletDataset)
    Epic 1.2: SliceBalancedSampler
    Epic 1.3: Config schema update

Milestone 2: Training Step Updates
    Epic 2.1: Mixed-batch train_step (triplet + pair)
    Epic 2.2: Per-slice loss routing

Milestone 3: Slice-Aware Evaluation
    Epic 3.1: SliceAwareEvaluator
    Epic 3.2: Retrieval eval for query_doc (Recall@k, MRR)
    Epic 3.3: Per-slice leaderboard reporting

Milestone 4: Config & Integration
    Epic 4.1: Updated bakeoff YAML with all 5 folders + V2 experiment
    Epic 4.2: End-to-end smoke test
    Epic 4.3: Full training run
```

---

## 2. Milestone 1: Unified Data Pipeline

### Epic 1.1: EmbeddingDataset

**Goal**: Replace `TripletDataset` with a unified loader that handles triplets AND pairs from all 5 data folders.

#### Issue 1.1.1: Create `EmbeddingDataset` class

**File**: `scripts/training/train_embedding_heads_bakeoff.py` (replace `TripletDataset`)

The new dataset must handle three record schemas:

**Schema A - Standard triplets** (silver_synthetic, hard_negatives):
```json
{"anchor": "...", "positive": "...", "negative": "...", "hard_negative_type": "entity_swap"}
```

**Schema B - Mined triplets** (wrong_person, wrong_time, safety_emotion):
```json
{"anchor": "...", "positive": "...", "negative": "...", "hard_negative_type": "entity_swap",
 "mismatch_features": [...], "difficulty": "hard", "slice_tags": [...]}
```

**Schema C - Query-document pairs** (query_doc):
```json
{"query": "...", "document": "...", "pair_type": "memory_match",
 "shared_features": [...], "difficulty": "easy"}
```

**Normalized output record** (what `__getitem__` returns):
```python
{
    "anchor": str,           # anchor text (or query for query_doc)
    "positive": str,         # positive text (or document for query_doc)
    "negative": str | None,  # negative text (None for pair-only)
    "has_negative": bool,    # False for query_doc, True for triplets
    "is_hard_negative": bool,
    "slice": str,            # "silver_synthetic" | "hard_negatives" | "wrong_person" | "wrong_time" | "safety_emotion" | "query_doc"
    "difficulty": str,       # "easy" | "medium" | "hard" | "unknown"
}
```

**Key behaviors**:
- Auto-detect schema from record keys (`query`/`document` vs `anchor`/`positive`/`negative`)
- Map `query` -> `anchor`, `document` -> `positive` for query_doc records
- Set `has_negative=False` for query_doc (signals pair-only mode to collator/train_step)
- Assign `slice` tag based on source directory name
- Track per-slice sample counts for logging

**Acceptance criteria**:
- [ ] Loads all 5 data folders without silent drops
- [ ] query_doc records produce `has_negative=False` with `negative=None`
- [ ] Per-slice counts logged at init
- [ ] Backward compatible: triplet-only folders produce identical output to old `TripletDataset`

---

#### Issue 1.1.2: Update `EmbeddingCollator` (replaces `TripletCollator`)

**File**: `scripts/training/train_embedding_heads_bakeoff.py`

The collator must handle mixed batches containing both triplets and pairs.

**Output batch dict**:
```python
{
    "anchor_input_ids": Tensor,       # [B, L]
    "anchor_attention_mask": Tensor,  # [B, L]
    "positive_input_ids": Tensor,     # [B, L]
    "positive_attention_mask": Tensor,# [B, L]
    "negative_input_ids": Tensor,     # [B_trip, L] (only triplet samples)
    "negative_attention_mask": Tensor,# [B_trip, L]
    "hard_negative_mask": Tensor,     # [B_trip]
    "has_negative": Tensor,           # [B] bool mask
    "slice_tags": list[str],          # [B] per-sample slice name
    "triplet_indices": Tensor,        # indices into batch where has_negative=True
    "pair_indices": Tensor,           # indices into batch where has_negative=False
}
```

**Key behaviors**:
- Tokenize anchor and positive for ALL samples
- Tokenize negative ONLY for triplet samples (where `has_negative=True`)
- Produce index tensors `triplet_indices` and `pair_indices` so train_step can split the batch
- Pad negative tensors to handle variable-count triplets in a batch

**Acceptance criteria**:
- [ ] Mixed batch of 50% triplets + 50% pairs produces correct shapes
- [ ] Pure triplet batch is identical to old `TripletCollator` output
- [ ] Pure pair batch has empty negative tensors and `pair_indices` covering all samples

---

#### Issue 1.1.3: Eval holdout per slice

**File**: `scripts/training/train_embedding_heads_bakeoff.py`

Instead of a single random `val_split` across the whole pool, hold out a fixed percentage **per slice** to guarantee eval coverage.

**Logic**:
```
For each slice:
    eval_samples = slice_samples[:ceil(len(slice) * eval_split)]
    train_samples = slice_samples[ceil(len(slice) * eval_split):]
```

**Config**:
```yaml
data:
  eval_split_per_slice: 0.15  # 15% of each slice held out
```

**Why**: A random 10% global split from 280K samples yields ~28K eval samples, but statistically only ~400 from wrong_person and ~280 from safety_emotion. Per-slice holdout guarantees at least ~660 wrong_person eval, ~819 wrong_time eval, ~429 safety_emotion eval, and ~933 query_doc eval.

**Acceptance criteria**:
- [ ] Every slice has at least `eval_split` fraction in eval set
- [ ] Eval set slice distribution logged
- [ ] Train set never contains eval samples from any slice
- [ ] Deterministic split (seeded shuffle per slice)

---

### Epic 1.2: SliceBalancedSampler

**Goal**: Prevent 261K silver_synthetic from drowning 19K mined_v2 during training.

#### Issue 1.2.1: Implement `SliceBalancedSampler`

**File**: `scripts/training/train_embedding_heads_bakeoff.py`

A custom `torch.utils.data.Sampler` that controls per-epoch sampling ratios.

**Config**:
```yaml
data:
  sampling:
    strategy: "balanced"  # or "proportional" or "custom"
    slice_weights:
      silver_synthetic: 1.0
      hard_negatives: 2.0       # upsample hard negs
      wrong_person: 4.0         # upsample role confusion
      wrong_time: 4.0           # upsample temporal confusion
      safety_emotion: 5.0       # upsample safety (smallest + most critical)
      query_doc: 3.0            # upsample asymmetric pairs
```

**Behavior**:
- Each epoch, sample from each slice proportional to `slice_weight * slice_count`
- This effectively upsamples small critical slices and gives them more gradient signal
- With these weights, effective per-epoch distribution shifts from ~92% silver to roughly:
  - silver_synthetic: ~55%
  - hard_negatives: ~18%
  - wrong_person: ~4%
  - wrong_time: ~5%
  - safety_emotion: ~3%
  - query_doc: ~15% (query_doc gets decent representation)
- Samples within each slice are shuffled each epoch (with replacement for upsampled slices)

**Acceptance criteria**:
- [ ] Effective sampling ratios match config weights within 5%
- [ ] Each epoch sees a different shuffle order
- [ ] Compatible with `DataLoader(sampler=...)` interface
- [ ] Logged: effective sample counts per slice per epoch

---

### Epic 1.3: Config Schema Update

#### Issue 1.3.1: Extend data config to list all 5 folders + sampling weights

**File**: `configs/training/embedding_heads_bakeoff.yaml`

Replace flat comma-separated path string with structured source list:

```yaml
data:
  root: data
  max_length: 128
  num_workers: 12
  eval_split_per_slice: 0.15

  sources:
    - path: familyos/embeddings/silver_synthetic
      slice: silver_synthetic
      format: triplet

    - path: familyos/embeddings/hard_negatives
      slice: hard_negatives
      format: triplet

    - path: familyos/embeddings/mined_v2/wrong_person
      slice: wrong_person
      format: triplet

    - path: familyos/embeddings/mined_v2/wrong_time
      slice: wrong_time
      format: triplet

    - path: familyos/embeddings/mined_v2/safety_emotion
      slice: safety_emotion
      format: triplet

    - path: familyos/embeddings/mined_v2/query_doc
      slice: query_doc
      format: pair  # signals query/document schema

  sampling:
    strategy: balanced
    slice_weights:
      silver_synthetic: 1.0
      hard_negatives: 2.0
      wrong_person: 4.0
      wrong_time: 4.0
      safety_emotion: 5.0
      query_doc: 3.0
```

**Backward compatibility**: If `data.sources` is absent, fall back to parsing `data.embedding.train` as comma-separated paths (old format).

**Acceptance criteria**:
- [ ] New config loads without error
- [ ] Old configs still work via fallback
- [ ] `get_embedding_data_paths()` updated to read new schema

---

## 3. Milestone 2: Training Step Updates

### Epic 2.1: Mixed-Batch Train Step

**Goal**: Handle batches containing both triplet and pair samples in a single forward/backward pass.

#### Issue 2.1.1: Split-batch training logic

**File**: `scripts/training/train_embedding_heads_bakeoff.py` (update `train_step` and joint training loop)

**Current behavior**:
```python
# Always passes explicit negatives
negative_emb = embedding_head(negative_hidden, negative_mask)
negatives = negative_emb.unsqueeze(1)
loss = loss_fn(anchor=anchor_emb, positive=positive_emb, negatives=negatives, ...)
```

**New behavior**:
```python
triplet_idx = batch["triplet_indices"]
pair_idx = batch["pair_indices"]

loss = 0.0
count = 0

# Triplet sub-batch: explicit negatives
if len(triplet_idx) > 0:
    a_trip = anchor_emb[triplet_idx]
    p_trip = positive_emb[triplet_idx]
    n_trip = embedding_head(negative_hidden[triplet_idx], negative_mask[triplet_idx])
    hn_mask = hard_negative_mask[triplet_idx].unsqueeze(1)
    loss_trip = loss_fn(anchor=a_trip, positive=p_trip,
                        negatives=n_trip.unsqueeze(1), hard_negative_mask=hn_mask)
    loss = loss + loss_trip
    count += 1

# Pair sub-batch: in-batch negatives only
if len(pair_idx) > 0:
    a_pair = anchor_emb[pair_idx]
    p_pair = positive_emb[pair_idx]
    loss_pair = loss_fn(anchor=a_pair, positive=p_pair, negatives=None)
    loss = loss + loss_pair
    count += 1

loss = loss / max(count, 1)
```

**Why split instead of one call**: Mixing explicit negatives and in-batch negatives in a single InfoNCE denominator is mathematically messy. The clean approach is two loss terms averaged.

**Acceptance criteria**:
- [ ] Pure triplet batches produce identical loss to current code
- [ ] Pure pair batches use in-batch negatives path
- [ ] Mixed batches compute both losses and average
- [ ] Metrics (pos_sim, neg_sim, margin) reported for triplet sub-batch only (pair sub-batch has no explicit neg_sim)

---

#### Issue 2.1.2: Encoder forward handles variable negative counts

**File**: `scripts/training/train_embedding_heads_bakeoff.py` (update `encode_triplet_batch`)

Current `encode_triplet_batch` always encodes negatives. Update to skip negative encoding when the batch has no triplet samples.

```python
# Only encode negatives if there are triplet samples
if batch["triplet_indices"].numel() > 0:
    negative_ids = batch["negative_input_ids"].to(device)
    negative_mask = batch["negative_attention_mask"].to(device)
    # ... encode negatives
else:
    negative_hidden = None
    negative_mask = None
```

**Acceptance criteria**:
- [ ] Pair-only batches skip negative encoding entirely (saves ~33% encoder forward cost)
- [ ] Return dict includes `negative_hidden: None` when no triplets present

---

### Epic 2.2: Per-Slice Loss Routing

#### Issue 2.2.1: Slice-tagged loss accumulation for logging

**Goal**: Track loss contributions per slice for training diagnostics, without changing the optimization.

**Not** implementing the full multi-objective loss from the spec ($L = \lambda_1 L_{retrieval} + ...$) yet. That's a future enhancement once we validate the data pipeline works. For now, all slices go through the same `FamilyContrastiveLoss` but we **log** per-slice loss separately.

**Implementation**: After computing per-sample losses, group by `slice_tags` and log averages:

```python
# In logging block (every logging_steps):
for slice_name in unique_slices:
    slice_mask = [s == slice_name for s in batch["slice_tags"]]
    # Log: slice_name -> avg loss for this slice
```

**Acceptance criteria**:
- [ ] Per-slice loss logged every `logging_steps`
- [ ] No change to optimization (same backward pass)
- [ ] Visible in training output: `wrong_person_loss=X.XX wrong_time_loss=X.XX ...`

---

## 4. Milestone 3: Slice-Aware Evaluation

### Epic 3.1: SliceAwareEvaluator

**Goal**: Evaluate each held-out slice separately and report per-slice metrics.

#### Issue 3.1.1: Implement `evaluate_by_slice()` function

**File**: `scripts/training/train_embedding_heads_bakeoff.py`

**Interface**:
```python
def evaluate_by_slice(
    model: ModernBertMultiTaskModel,
    val_loader: DataLoader,
    loss_fn: FamilyContrastiveLoss,
    device: torch.device,
    ...
) -> dict[str, dict[str, float]]:
    """
    Returns:
        {
            "silver_synthetic": {"val_loss": ..., "margin": ..., "accuracy": ..., "hard_neg_accuracy": ...},
            "wrong_person": {"val_loss": ..., "margin": ..., "accuracy": ..., "hard_neg_accuracy": ...},
            "wrong_time": {...},
            "safety_emotion": {...},
            "query_doc": {"val_loss": ..., "pair_accuracy": ...},  # different metrics for pairs
            "_aggregate": {"val_loss": ..., "margin": ..., ...},
        }
    ```

**Per-slice metrics for triplet slices**:
- `val_loss`: average loss
- `pos_sim`: average positive cosine similarity
- `neg_sim`: average negative cosine similarity
- `margin`: pos_sim - neg_sim
- `accuracy`: fraction where pos_sim > neg_sim
- `hard_neg_accuracy`: accuracy on hard negatives only
- `sample_count`: number of eval samples

**Per-slice metrics for query_doc (pair slice)**:
- `val_loss`: in-batch InfoNCE loss
- `pair_accuracy`: fraction where correct positive has highest similarity in batch
- `sample_count`: number of eval pairs

**Acceptance criteria**:
- [ ] Every slice with >0 eval samples gets its own metrics dict
- [ ] Aggregate metrics computed across all slices
- [ ] query_doc uses in-batch eval (no explicit negatives needed)

---

#### Issue 3.1.2: Per-slice leaderboard logging

**File**: `scripts/training/train_embedding_heads_bakeoff.py`

Extend the existing `log_head_leaderboard()` to also log a **slice breakdown** at each eval step:

```
==================================================
  SLICE EVAL @ STEP 500 (agreement_gated_v2)
==================================================
Slice               Margin    Acc       HardNeg   Loss      N
---------------------------------------------------------------
silver_synthetic    0.3412    0.8950    0.7120    0.4521    39270
hard_negatives      0.2103    0.7890    0.6234    0.6102    6441
wrong_person        0.1845    0.7320    0.7320    0.7012    665
wrong_time          0.2012    0.7560    0.7560    0.6523    819
safety_emotion      0.1523    0.6980    0.6980    0.7834    429
query_doc           --        0.6245    --        0.8912    933
_aggregate          0.2891    0.8412    0.6821    0.5234    48557
```

**Acceptance criteria**:
- [ ] Slice table logged at every eval step
- [ ] query_doc shows `--` for margin/hard_neg (not applicable)
- [ ] Aggregate row always present

---

### Epic 3.2: Retrieval Eval for Query-Doc

#### Issue 3.2.1: Implement `evaluate_retrieval()` for Recall@k and MRR

**File**: `scripts/training/train_embedding_heads_bakeoff.py`

**Goal**: Measure actual retrieval quality on query_doc eval set.

**Method**:
1. Embed all query_doc eval queries -> Q matrix [N_q, O]
2. Embed all query_doc eval documents -> D matrix [N_d, O]
3. Compute similarity matrix S = Q @ D.T [N_q, N_d]
4. For each query, the correct document is its paired document
5. Compute Recall@1, Recall@5, Recall@10, MRR

**Implementation**:
```python
@torch.no_grad()
def evaluate_retrieval(
    model: ModernBertMultiTaskModel,
    query_doc_eval_samples: list[dict],
    tokenizer: Any,
    device: torch.device,
    max_length: int = 128,
    batch_size: int = 256,
) -> dict[str, float]:
    """Returns: {"recall@1": ..., "recall@5": ..., "recall@10": ..., "mrr": ...}"""
```

**Acceptance criteria**:
- [ ] Recall@1, Recall@5, Recall@10, MRR computed correctly
- [ ] Logged at each eval step alongside slice metrics
- [ ] Handles up to ~1000 query-doc pairs efficiently

---

### Epic 3.3: Best-Model Selection Update

#### Issue 3.3.1: Composite score for model selection

**Current**: Model selected by best `margin` on aggregate eval.

**New**: Use a weighted composite score that penalizes slice regressions:

```python
composite_score = (
    0.35 * aggregate_margin
    + 0.15 * hard_neg_accuracy
    + 0.15 * wrong_person_accuracy
    + 0.10 * wrong_time_accuracy
    + 0.10 * safety_emotion_accuracy
    + 0.15 * query_doc_recall_at_5
)
```

Weights configurable in YAML:

```yaml
evaluation:
  selection_metric: composite
  composite_weights:
    aggregate_margin: 0.35
    hard_neg_accuracy: 0.15
    wrong_person_accuracy: 0.15
    wrong_time_accuracy: 0.10
    safety_emotion_accuracy: 0.10
    query_doc_recall_at_5: 0.15
```

**Acceptance criteria**:
- [ ] Composite score computed and logged at each eval
- [ ] Best checkpoint saved by composite score (not just margin)
- [ ] Configurable weights in YAML

---

## 5. Milestone 4: Config & Integration

### Epic 4.1: Updated Bakeoff YAML

#### Issue 4.1.1: Add V2 experiment + all data sources to config

**File**: `configs/training/embedding_heads_bakeoff.yaml`

Add AgreementGatedHeadV2 as E6 in experiments:

```yaml
experiments:
  heads:
    # ... existing E0-E5 ...

    # E6: AgreementGatedHeadV2 (evolution of E3)
    - head_type: agreement_gated_v2
      params:
        num_latents: 4
        num_attn_heads: 4
        gate_hidden: 128
        gate_rank: 4
        use_mode_prompts: true
        use_confidence_head: true
        dropout: 0.1
```

Add all 5 data sources (as defined in Issue 1.3.1).

**Acceptance criteria**:
- [ ] Config parses without error
- [ ] `agreement_gated_v2` appears in experiment list
- [ ] All 5 data folders listed in sources

---

### Epic 4.2: End-to-End Smoke Test

#### Issue 4.2.1: Debug-mode validation run

Run with `--debug --max_samples 200 --head_type agreement_gated_v2` to verify:

1. All 5 data folders load (check per-slice counts in log)
2. Mixed batches (triplet + pair) process without error
3. Slice-aware eval runs and reports per-slice metrics
4. Retrieval eval reports Recall@k on query_doc holdout
5. Checkpoint saves with correct metadata
6. Composite selection score computed

**Acceptance criteria**:
- [ ] No crashes or silent data drops
- [ ] All 6 slices appear in eval output
- [ ] query_doc Recall@k is non-zero
- [ ] Checkpoint `embedding_metadata.json` contains `head_type: agreement_gated_v2`

---

### Epic 4.3: Full Training Run

#### Issue 4.3.1: Single-head V2 training run

```bash
python scripts/training/train_embedding_heads_bakeoff.py \
    --config configs/training/embedding_heads_bakeoff.yaml \
    --head_type agreement_gated_v2
```

**Expected outcome**:
- ~323K total samples loaded (261K silver + 42K hard_neg + 4.4K wrong_person + 5.4K wrong_time + 2.8K safety_emotion + 6.2K query_doc)
- ~275K train / ~48K eval (15% per-slice holdout)
- Slice-balanced sampling ensures mined_v2 slices get adequate gradient signal
- Training for 7 epochs with cosine schedule
- Best checkpoint selected by composite score
- Output: `outputs/embedding-bakeoff/agreement_gated_v2/best/`

#### Issue 4.3.2: Joint bakeoff including V2

```bash
python scripts/training/train_embedding_heads_bakeoff.py \
    --config configs/training/embedding_heads_bakeoff.yaml \
    --run_all
```

Train all 7 heads (E0-E6) together with shared encoder pass. Compare V2 against V1 and other candidates on the composite score.

---

## 6. Implementation Order

Strict dependency chain:

```
Issue 1.1.1 (EmbeddingDataset)
    |
    +---> Issue 1.1.2 (EmbeddingCollator)
    |         |
    |         +---> Issue 2.1.1 (mixed-batch train_step)
    |         |         |
    |         |         +---> Issue 2.1.2 (encoder skip negatives)
    |         |
    |         +---> Issue 2.2.1 (per-slice loss logging)
    |
    +---> Issue 1.1.3 (eval holdout per slice)
              |
              +---> Issue 3.1.1 (evaluate_by_slice)
              |         |
              |         +---> Issue 3.1.2 (slice leaderboard)
              |
              +---> Issue 3.2.1 (retrieval eval)
                        |
                        +---> Issue 3.3.1 (composite score)

Issue 1.2.1 (SliceBalancedSampler)  -- parallel with above

Issue 1.3.1 (config schema)  -- parallel with above

All above ---> Issue 4.1.1 (updated YAML)
                    |
                    +---> Issue 4.2.1 (smoke test)
                              |
                              +---> Issue 4.3.1 (V2 training)
                              +---> Issue 4.3.2 (joint bakeoff)
```

**Recommended build order** (serial path, total 14 issues):

| Phase | Issues | What ships |
|---|---|---|
| Phase 1 | 1.1.1, 1.1.2, 1.1.3, 1.2.1, 1.3.1 | Data pipeline complete, config updated |
| Phase 2 | 2.1.1, 2.1.2, 2.2.1 | Training handles mixed batches |
| Phase 3 | 3.1.1, 3.1.2, 3.2.1, 3.3.1 | Slice-aware eval + retrieval metrics |
| Phase 4 | 4.1.1, 4.2.1 | Integration verified |
| Phase 5 | 4.3.1, 4.3.2 | Training runs |

---

## 7. Files Modified

| File | Changes |
|---|---|
| `scripts/training/train_embedding_heads_bakeoff.py` | Replace `TripletDataset` with `EmbeddingDataset`, replace `TripletCollator` with `EmbeddingCollator`, add `SliceBalancedSampler`, update `train_step` for mixed batches, add `evaluate_by_slice()`, add `evaluate_retrieval()`, update leaderboard logging, add composite score selection |
| `scripts/training/train_embedding_head.py` | Same dataset/collator changes (shared code or import) |
| `configs/training/embedding_heads_bakeoff.yaml` | New `data.sources` schema, sampling weights, eval config, V2 experiment |

---

## 8. Risk Assessment

| Risk | Mitigation |
|---|---|
| query_doc in-batch negatives are too easy at large batch sizes | Monitor pair_accuracy; if >95% trivially, reduce batch size for pair sub-batches or add hard document distractors |
| Slice upsampling causes overfitting on small slices | EMA + early stopping per slice; monitor per-slice val_loss divergence |
| Mixed-batch training is slower than pure-triplet | Pair sub-batch skips negative encoding (~33% encoder savings); net throughput should be similar |
| Composite score makes model selection opaque | Log all individual slice metrics alongside composite; allow override to single-metric selection |

---

## 9. Success Criteria

The full pipeline is considered working when:

1. All 323K samples load across 6 slices (no silent drops)
2. Per-slice eval reports metrics for all 6 slices at every eval step
3. query_doc Recall@1 > 0.30, Recall@5 > 0.60 (non-trivial retrieval)
4. wrong_person accuracy > 0.65 (better than random on entity-swap negatives)
5. safety_emotion accuracy > 0.60 (better than random on safety near-misses)
6. AgreementGatedHeadV2 composite score beats V1 and mean_baseline
7. Best checkpoint saves correctly with full metadata
