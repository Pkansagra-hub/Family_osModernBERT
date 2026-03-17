We are building a SOTA NLI re-ranking head (Multi-Granularity Relevance Head — MGRH) for UltraBERT v4.
This plan turns the research findings from `docs/NLI_RERANKING_GAP_AND_TRAINING_PLAN.md` into a concrete implementation roadmap that reuses the existing bakeoff pipeline.

Created 6 todos

The bi-encoder cosine path (`similarity()`) is confirmed BLOCKED for re-ranking (Spearman=0.2047, AUC=0.6919).
A cross-encoder relevance head is the only viable path to unblock episode retrieval re-ranking.

Completed: *Validate approach against existing codebase* (1/6)

---

## Short answer

The existing codebase has almost everything we need:

1. `NLIHead` in `heads.py:801` already supports `pair_encoder` + `text_a_hidden/text_b_hidden` inputs — but it was never invoked with one in production
2. `CrossAttentionPairEncoder` in `pair_encoder.py` is fully implemented (ESIM bidirectional, attention pooling, residual + FFN)
3. `AgreementGatedHeadV2` in `heads_embedding.py` already produces query/document asymmetric embeddings we can reuse as Signal 3
4. The bakeoff training script (`train_embedding_heads_bakeoff.py`, ~4700 lines) handles dataset loading, per-slice sampling, evaluation, and EMA — we reuse all of it
5. The bakeoff config (`embedding_heads_bakeoff.yaml`) already defines data sources under `data/familyos/embeddings/` — the same hard negatives we need

What is missing:

- The `MultiGranularityRelevanceHead` class itself (new head in `heads.py`)
- NLI general-domain pre-training datasets: ANLI (disabled), WANLI (absent), FEVER-NLI (absent)
- A relevance-specific training config section in the YAML
- Client API methods: `score_relevance()` and `rerank()`
- Human benchmark data formatted as listwise JSONL

The approach:
> **Register the MGRH head in `heads.py`. Stage datasets in `data/familyos/nli/`. Extend the existing YAML and training script. No new standalone scripts.**

---

## Epic: Multi-Granularity Relevance Head (MGRH) — SOTA NLI Re-Ranking for Episode Retrieval

Goal:

- register a new `MultiGranularityRelevanceHead` in `heads.py` following established patterns
- curate and stage NLI + relevance training data under `data/familyos/nli/`
- extend `embedding_heads_bakeoff.yaml` with MGRH-specific config (no rewrite)
- extend `train_embedding_heads_bakeoff.py` with MGRH training mode (no new scripts)
- train in 3 stages: general NLI pre-train → domain adaptation → relevance fine-tune
- wire through `client.py` as `score_relevance()` and `rerank()`
- target: Spearman > 0.70, AUC-ROC > 0.85, nDCG lift +10-16pp over bi-encoder baseline (upgraded from 0.50/0.80/+8-14pp with SOTA enhancements)

---

## Milestone 1 — Register MultiGranularityRelevanceHead in heads.py

This is the "design the head" milestone. Follows the exact pattern of NLIHead, HierarchicalEmotionHead, and other heads in `heads.py`.

### Why

The MGRH does not exist yet. All 14 existing heads are defined in `heads.py` and exported via `__all__`. No head registry — heads are discovered by direct import. A new head must follow this pattern exactly.

### Files

- `src/modeling_studio/models/heads.py`

### Issues

#### Issue 1.1 — Define MultiGranularityRelevanceHead class

Add the MGRH class after `NLIHead` (around line 920). It inherits from `BaseHead` (not `SequenceClassificationHead`) because it needs a custom forward that fuses three signals, not a simple classify-from-pooled path.

Constructor signature (follows BaseHead pattern):

```python
class MultiGranularityRelevanceHead(BaseHead):
    def __init__(
        self,
        hidden_size: int = 768,
        num_labels: int = 1,           # regression: single relevance score
        dropout: float = 0.1,
        problem_type: str = "regression",
        pair_encoder: nn.Module | None = None,  # CrossAttentionPairEncoder
    ):
        # ...
        # Two-head output design (avoids cold-start weight reset between stages):
        self.nli_head = nn.Linear(256, 3)       # Stage A only — 3-class NLI logits, discarded after Stage A
        self.relevance_head = nn.Linear(256, 1)  # Stage B+C — always trained, warm by Stage C
```

**Critical design decision — two-head output layer:**

The naive approach (single `Linear(256, num_labels)` swapped between stages) has a fatal flaw: if Stage A trains with `Linear(256, 3)` and Stage C replaces it with a fresh `Linear(256, 1)`, all of Stage B's learned representations in the 256-dim intermediate are thrown away by the random re-initialization.

The fix: instantiate **both** output heads at construction time. In Stage A, compute both losses but weight `relevance_head` loss at 0.1 (tiny auxiliary warmup). By Stage C, `relevance_head` already has warm representations from seeing every Stage A+B training step. The `nli_head` is simply ignored after Stage A.

```python
# Stage A forward:
nli_logits = self.nli_head(fused_256)       # primary loss: cross_entropy
rel_score = self.relevance_head(fused_256)   # auxiliary loss: 0.1 * BCE (warmup only)

# Stage B+C forward:
rel_score = self.relevance_head(fused_256)   # primary loss: margin / LambdaRank
# self.nli_head is frozen/ignored
```

Internal architecture (4-signal fusion):

- Signal 1: CLS token from joint cross-encoder sequence → `cls_proj(h[:, 0])` → [B, H]
- Signal 2: CrossAttentionPairEncoder output from separated query/episode streams (2-layer ESIM) → [B, H]
- Signal 3: Asymmetric embedding interaction: `[q_emb, d_emb, q_emb*d_emb, |q_emb-d_emb|]` → [B, 4H]
- Signal 4: ColBERT-style MaxSim token-level alignment (no learned params) → [B, 1] (z-score normalized per batch)
- Fusion: `[h_cls | h_cross | interaction | maxsim]` → **LayerNorm(4609)** → Linear(4609, 1024) → GELU → Dropout → Linear(1024, 256) → GELU → Dropout → `nli_head(256, 3)` + `relevance_head(256, 1)` (two-head output, see below)

Signal 4 implementation (ColBERT MaxSim — directly targets S7-1 keyword-overlap failure):

```python
# Inside MGRH forward, after obtaining text_a_hidden, text_b_hidden:
sim_matrix = torch.bmm(text_a_hidden, text_b_hidden.transpose(1, 2))  # [B, q_len, d_len]
if text_b_mask is not None:
    sim_matrix = sim_matrix.masked_fill(~text_b_mask.unsqueeze(1).bool(), -1e9)
maxsim_per_token = sim_matrix.max(dim=-1).values  # [B, q_len]
if text_a_mask is not None:
    maxsim = (maxsim_per_token * text_a_mask.float()).sum(1) / text_a_mask.float().sum(1).clamp(min=1)
else:
    maxsim = maxsim_per_token.mean(1)  # [B] scalar MaxSim score

# CRITICAL: z-score normalize MaxSim before concatenation with other signals.
# Raw MaxSim is an unbounded dot-product scalar with wildly different magnitude
# than sigmoid/L2-normed signals — without normalization the first Linear layer
# either ignores it (gradient too small) or it dominates.
maxsim = maxsim.unsqueeze(-1)  # [B, 1]
maxsim = (maxsim - maxsim.mean()) / (maxsim.std() + 1e-8)  # z-score per batch

# Additionally, LayerNorm(4609) before the fusion MLP handles cross-signal
# magnitude differences globally (see fusion_input_ln in constructor).
```

MaxSim catches the exact failure from S7-1: "Maya eating at a restaurant" vs "Maya watched a cooking show about restaurants" — per-token alignment reveals that "eating" has no strong match in the episode, while a pooled representation would blur this signal away.

Forward signature (follows established pattern from NLIHead):

```python
def forward(
    self,
    hidden_states: torch.Tensor,              # [B, seq_len, H] from cross-encoder
    attention_mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,       # [B] relevance grades 0-3 or 0.0-1.0
    pair_encoder: nn.Module | None = None,    # override
    text_a_hidden: torch.Tensor | None = None,
    text_b_hidden: torch.Tensor | None = None,
    text_a_mask: torch.Tensor | None = None,
    text_b_mask: torch.Tensor | None = None,
    query_embed: torch.Tensor | None = None,  # [B, H] from AgreementGatedHeadV2
    doc_embed: torch.Tensor | None = None,    # [B, H] from AgreementGatedHeadV2
) -> dict[str, torch.Tensor]:
    # Returns: {"logits": score, "loss": loss (if labels)}
```

Loss: `F.binary_cross_entropy(score, labels_normalized)` for pointwise training. LambdaRank and pairwise margin are applied externally in the training loop (same pattern as contrastive loss in the bakeoff script).

#### Issue 1.2 — Add relevance_score() accessor method

```python
def relevance_score(self, logits: torch.Tensor) -> torch.Tensor:
    """Logits are already sigmoid-activated — direct relevance score."""
    return logits
```

This follows the pattern from the NLI doc's Option B design.

#### Issue 1.3 — Add MGRH to **all** exports

Append `"MultiGranularityRelevanceHead"` to the `__all__` list at line 4469.

#### Issue 1.4 — Add LambdaRankLoss and CombinedRankingLoss

These loss classes go in the same file or a `losses.py` alongside `heads.py`. They are used by the training loop, not the head's internal `forward()`.

- `LambdaRankLoss`: listwise nDCG optimization for graded labels (0-3)
- `CombinedRankingLoss`: LambdaRank + pairwise margin on hard negatives
- Both are defined in the NLI doc section 6.6 with full implementations

Decision: place in a new `src/modeling_studio/models/losses_ranking.py` to keep `heads.py` clean. This is the only new file.

### Deliverable

A registered, importable `MultiGranularityRelevanceHead` that follows the same constructor/forward/loss patterns as NLIHead and can be instantiated from config.

---

## Milestone 2 — Curate and stage NLI + relevance training datasets

This is the "prepare all training data" milestone. Datasets are staged under `data/familyos/nli/` following the convention of `data/familyos/embeddings/`, `data/familyos/safety/`, etc.

### Why

MGRH training requires 3 stages of data:

- Stage A (general NLI): MNLI + SNLI + ANLI + WANLI + FEVER-NLI = ~854K pairs
- Stage B (domain adaptation): FamilyOS hard negatives from existing `data/familyos/embeddings/`
- Stage C (relevance fine-tune): human benchmark listwise + mined_v2 hard negatives

Stage A datasets are mostly available via HuggingFace but need to be downloaded, formatted to unified JSONL, and staged. Stage B data already exists. Stage C requires formatting the human benchmark.

### Files

- `data/familyos/nli/` (new directory, dataset staging area)
- `scripts/data/` (dataset preparation scripts if needed)

### Issues

#### Issue 2.1 — Stage general NLI datasets (Stage A)

Create `data/familyos/nli/general/` with unified JSONL format:

```jsonl
{"premise": "...", "hypothesis": "...", "label": 0}
```

Label mapping: `{entailment: 0, neutral: 1, contradiction: 2}`

Datasets to download and format:

- `nyu-mll/multi_nli` → `mnli_train.jsonl` (392K, CC-BY-3.0)
- `stanfordnlp/snli` → `snli_train.jsonl` (570K, CC-BY-SA-4.0)
- `facebook/anli` → `anli_r1.jsonl`, `anli_r2.jsonl`, `anli_r3.jsonl` (169K total, CC-BY-NC-4.0)
- `alisawuffles/WANLI` → `wanli_train.jsonl` (103K, CC-BY-4.0)
- FEVER-NLI preprocessed → `fever_nli_train.jsonl` (185K, CC-BY-SA-3.0)

Use `easonnie/combine-FEVER-NSMN` or equivalent preprocessed form for FEVER-NLI (avoids retrieval preprocessing from raw fever dataset).

**SHIP BLOCKER — ANLI License (CC-BY-NC-4.0):**

ANLI R1-R3 is CC-BY-NC-4.0. If FamilyOS has any commercial use path, ANLI cannot be used in Stage A training. This is a **blocking decision required before Issue B1 starts.**

Options if commercial:
- **Replace with DocNLI** (Apache 2.0, 942K pairs, document-level NLI — actually better domain fit for episode-length passages)
- **Replace with NLI-in-the-wild** (MIT license, smaller but adversarial)
- **Use only MNLI + WANLI + FEVER-NLI + SNLI** (all CC-BY compatible, drops ~169K ANLI pairs but retains 1.25M)

WANLI (CC-BY-4.0), FEVER-NLI (CC-BY-SA-3.0), MNLI (CC-BY-3.0), SNLI (CC-BY-SA-4.0) are all fine commercially.

**Decision deadline: before B1 execution.** Do not download ANLI until resolved.

#### Issue 2.2 — Stage domain NLI / relevance data (Stage B + C)

Create `data/familyos/nli/domain/` with symlinks or copies from existing embeddings data:

Already exists (reuse directly):

- `data/familyos/embeddings/hard_negatives/` → entity_swap, temporal_shift, same_topic_different_event types (~7K usable)
- `data/familyos/embeddings/mined_v2/query_doc/` → positive pairs (10K)
- `data/familyos/embeddings/mined_v2/wrong_time/` → temporal hard negatives (10K)
- `data/familyos/embeddings/mined_v2/wrong_person/` → entity hard negatives (12K)

Excluded:

- `data/familyos/embeddings/silver_synthetic/` → cross-cluster easy negatives, too easy for cross-encoder training

#### Issue 2.3 — Format human benchmark as listwise JSONL (Stage C)

Create `data/familyos/nli/relevance/human_benchmark_listwise.jsonl`:

```jsonl
{"query": "...", "episodes": [{"text": "...", "grade": 3}, {"text": "...", "grade": 1}, {"text": "...", "grade": 0}]}
```

Source: 50 queries x 88 episodes with grades 0-3 (~4,400 pairs total). This is the only listwise data for LambdaRank training.

#### Issue 2.4 — Create train/dev/holdout splits

Split all staged data:

- 80% train, 10% dev, 10% holdout
- Stratified by source (general NLI vs domain vs relevance)
- Dev set used for early stopping and checkpoint selection
- Holdout reserved for final evaluation only

### Deliverable

All NLI + relevance data staged under `data/familyos/nli/` in unified JSONL format, split into train/dev/holdout, ready for the training pipeline.

---

## Milestone 3 — Extend training config for MGRH

This is the "teach the YAML new words" milestone. Follows the exact pattern from the embedding_training_v2 plan (Milestone 1 there).

### Why

The current `embedding_heads_bakeoff.yaml` knows about embedding heads and contrastive loss. It does not know about:

- NLI classification heads
- Cross-encoder pair input
- Listwise ranking loss (LambdaRank)
- Multi-stage training (general NLI → domain → relevance)
- The MGRH head type

### Files

- `configs/training/embedding_heads_bakeoff.yaml`

### Issues

#### Issue 3.1 — Add mgrh_training section to config

```yaml
mgrh_training:
  enabled: false    # opt-in, does not affect existing bakeoff runs

  head:
    type: multi_granularity_relevance
    hidden_size: 768
    dropout: 0.1
    use_pair_encoder: true
    pair_encoder:
      num_heads: 8
      num_layers: 2             # SOTA upgrade: 2-layer ESIM for chain-of-alignment reasoning
      use_bidirectional: true
      pooling_strategy: attention
    use_asymmetric_embeddings: true   # Signal 3 from AgreementGatedHeadV2
    use_maxsim: true                  # Signal 4: ColBERT-style token-level alignment
    use_domain_saliency: true         # Use TemporalHead + NERHead as attention weights in pair encoder

  base_checkpoint: checkpoints/distil_stage_b_bestema

  freeze:
    encoder: true
    existing_heads: true
    trainable_modules:
      - relevance_head
      - pair_encoder
```

#### Issue 3.2 — Add stage-specific data config

```yaml
mgrh_training:
  stages:
    stage_a:
      name: general_nli_pretrain
      data_root: data/familyos/nli/general
      sources:
        - path: mnli_train.jsonl
          weight: 1.0
        - path: anli_r1.jsonl
          weight: 1.2
        - path: anli_r2.jsonl
          weight: 1.2
        - path: anli_r3.jsonl
          weight: 1.5    # R3 is hardest, most valuable
        - path: wanli_train.jsonl
          weight: 1.0
        - path: fever_nli_train.jsonl
          weight: 0.8
      loss:
        type: cross_entropy    # standard 3-class NLI
        label_smoothing: 0.0
        contrastive_nli:       # SOTA: SimCSE-style contrastive on NLI pairs alongside CE
          enabled: true
          weight: 0.3          # auxiliary alongside main CE objective
          temperature: 0.05
          # entailment = positive pair, contradiction = hard negative, in-batch = easy negatives
      epochs: 5
      lr_head: 2.0e-4
      batch_size: 64
      max_length: 256

    stage_b:
      name: domain_nli_adaptation
      data_root: data/familyos/embeddings
      sources:
        - path: hard_negatives
          include_types: [entity_swap, temporal_shift, same_topic_different_event, causality_flip]
          weight: 0.6
        - path: mined_v2/wrong_time
          weight: 0.8
          hard_negative_weight: 2.0
        - path: mined_v2/wrong_person
          weight: 0.8
          hard_negative_weight: 2.0
      loss:
        type: pairwise_margin
        margin: 0.2
      epochs: 5
      lr_head: 1.0e-4
      batch_size: 32
      max_length: 512    # query (~50 tok) + episode (~400 tok)

    bridge_bc:
      name: bridge_b_to_c
      # 500-step bridge prevents loss discontinuity when transitioning from
      # pure pairwise margin (Stage B) to graded relevance (Stage C).
      # Joint training with both loss types lets the pair encoder adapt
      # its representations smoothly instead of a cold loss switch.
      data_root: data/familyos/nli/relevance
      sources:
        - path: human_benchmark_listwise.jsonl
          format: listwise
          weight: 1.0
      loss:
        type: bridge_joint
        pairwise_margin_weight: 0.7   # dominant — preserving Stage B's discrimination
        relevance_bce_weight: 0.3     # introduce graded relevance signal gently
        margin: 0.2
      max_steps: 500                  # fixed 500 steps, not epoch-based
      lr_head: 5.0e-5                 # half the Stage B LR — gentle transition
      lr_pair_encoder: 2.5e-5
      batch_size: 32
      max_length: 512

    stage_c:
      name: relevance_finetune
      data_root: data/familyos/nli/relevance
      sources:
        - path: human_benchmark_listwise.jsonl
          format: listwise
          weight: 1.0
        - path: ../embeddings/mined_v2/query_doc
          format: positive_pair
          weight: 0.8
      loss:
        type: combined_ranking
        lambda_rank_weight: 1.0
        pairwise_margin_weight: 0.3
        margin: 0.2
        ndcg_at: 10
        r_drop:                # SOTA: R-Drop regularization (Liang et al., 2021)
          enabled: true
          alpha: 1.0           # KL divergence weight — paper default
      ance:                    # SOTA: ANCE dynamic hard negative refresh
        enabled: true
        refresh_every_n_epochs: 3
        mine_top_k: 20         # re-encode corpus, take top-20 non-relevant as new negatives
        max_refresh_negatives: 5000
      epochs: 10
      lr_head: 1.0e-4
      lr_pair_encoder: 5.0e-5
      batch_size: 32
      max_length: 512
      calibration:             # SOTA: post-training score calibration
        enabled: true
        method: temperature_scaling   # Platt scaling or temperature scaling on holdout
        max_iter: 50
```

#### Issue 3.3 — Add MGRH evaluation config

```yaml
mgrh_training:
  evaluation:
    selection_metric: spearman_correlation
    gate_metrics:
      spearman_correlation: 0.50     # minimum to ship
      auc_roc: 0.80                  # minimum to ship
    eval_steps: 200
    save_steps: 200
    ema:
      enabled: true
      decay: 0.995
    early_stopping:
      patience: 5
      metric: spearman_correlation
```

#### Issue 3.4 — Backward compatibility guard

If `mgrh_training.enabled` is `false` or the section is absent, the existing bakeoff/stage_b flows must work exactly as before. This follows the same pattern as `distillation.enabled: false` in the current config.

### Deliverable

A backward-compatible config extension. Old runs still work. MGRH training becomes available when `mgrh_training.enabled: true`.

---

## Milestone 4 — Extend training script for MGRH training mode

This is the "teach the script MGRH" milestone. Follows the same philosophy as Milestone 3 in the embedding_training_v2 plan: extend, do not rewrite.

### Why

The bakeoff script (`train_embedding_heads_bakeoff.py`, ~4700 lines) already handles:

- Config loading from YAML
- Dataset loading with `EmbeddingDataset` + `EmbeddingCollator`
- Per-slice balanced sampling via `SliceBalancedSampler`
- Model loading from checkpoint with `load_model_and_replace_embedding_head()`
- Encoder freezing with `freeze_model_except_embedding_head()`
- Training loop with AMP, gradient accumulation, EMA
- Evaluation: slice eval, retrieval eval, composite scoring
- Checkpoint saving with metadata

We reuse ALL of that. We add MGRH-specific paths gated behind `mgrh_training.enabled`.

### Files

- `scripts/training/train_embedding_heads_bakeoff.py`

### Issues

#### Issue 4.1 — Add NLI dataset loader for Stage A

Extend `EmbeddingDataset` or add a parallel `NLIDataset` class that loads premise/hypothesis/label JSONL records. The collator must produce:

- `input_ids`: tokenized `[CLS] premise [SEP] hypothesis [SEP]`
- `attention_mask`
- `labels`: 3-class NLI labels (0/1/2)
- Optional: `text_a_input_ids` / `text_b_input_ids` for separate encoding (pair encoder path)

Pattern: follows `EmbeddingCollator` which already handles mixed triplet+pair batches. NLI collator handles premise+hypothesis.

#### Issue 4.2 — Add relevance listwise dataset loader for Stage C

For human benchmark listwise data, the loader must produce per-query groups:

- `query_text`: the search query
- `episodes`: list of `(episode_text, grade)` tuples
- `is_hard_negative`: boolean mask for pairwise margin targeting

The collator tokenizes all `(query, episode)` pairs in the group as cross-encoder input: `[CLS] query [SEP] episode [SEP]`.

#### Issue 4.3 — Add MGRH model loading path

Add a function (or extend `load_model_and_replace_embedding_head`) that:

1. Loads base model from `checkpoints/distil_stage_b_bestema`
2. Instantiates `MultiGranularityRelevanceHead` with config params
3. Instantiates `CrossAttentionPairEncoder` with config params
4. Optionally loads AgreementGatedHeadV2 for Signal 3 embeddings (frozen, inference-only)
5. Freezes encoder + all existing heads, only MGRH + pair_encoder trainable

#### Issue 4.4 — Add MGRH forward path in train_step

Extend `train_step()` or add `mgrh_train_step()` that:

1. Encodes `[CLS] query [SEP] episode [SEP]` through the frozen encoder → `hidden_states`
2. Separately encodes query and episode for pair_encoder → `text_a_hidden`, `text_b_hidden`
3. Compute domain saliency weights from frozen TemporalHead + GlobalPointerNERHead (see Issue 4.7)
4. Gets query/doc embeddings from AgreementGatedHeadV2 (frozen forward) → `query_embed`, `doc_embed`
5. Computes ColBERT MaxSim from `text_a_hidden`, `text_b_hidden` → scalar Signal 4
6. Forwards all 4 signals through MGRH fusion MLP → `score`
7. Computes loss:
   - Stage A: `F.cross_entropy(logits, nli_labels)` + `contrastive_nli_loss` (weighted)
   - Stage B: pairwise margin loss on hard negatives
   - Stage C: `CombinedRankingLoss(scores, grades, is_hard_negative)` + `R-Drop KL loss`

#### Issue 4.4a — Add R-Drop regularization for Stage C

R-Drop (Liang et al., 2021) runs each sample through the model twice with different dropout masks and minimizes KL divergence between the two output distributions. This is especially valuable for Stage C where training data is small (~8.9K) and overfitting risk is high.

```python
# In mgrh_train_step, Stage C only:
if r_drop_enabled:
    score_1 = model.forward(...)  # first pass (dropout mask A)
    score_2 = model.forward(...)  # second pass (dropout mask B)
    task_loss = (criterion(score_1, labels) + criterion(score_2, labels)) / 2
    kl_loss = F.kl_div(
        F.log_softmax(score_1.unsqueeze(-1), dim=-1),
        F.softmax(score_2.unsqueeze(-1), dim=-1),
        reduction="batchmean"
    )
    kl_loss += F.kl_div(
        F.log_softmax(score_2.unsqueeze(-1), dim=-1),
        F.softmax(score_1.unsqueeze(-1), dim=-1),
        reduction="batchmean"
    )
    loss = task_loss + alpha * (kl_loss / 2)
```

#### Issue 4.5 — Add MGRH evaluation functions

Add `evaluate_mgrh()` that computes:

- Spearman correlation between predicted scores and true grades
- AUC-ROC for binary relevant/irrelevant classification
- nDCG@10 on listwise query groups
- Per-query-type breakdown (temporal, emotional, causal, entity, thematic, cross_episode)

Use existing `evaluate_by_slice()` pattern for per-type breakdown.

#### Issue 4.4b — Add Stage B→C bridge training step

500-step bridge between Stage B and Stage C. Prevents loss discontinuity from destabilizing the pair encoder.

```python
# After Stage B completes, before Stage C begins:
if bridge_config.get("enabled", True):
    bridge_steps = bridge_config.get("max_steps", 500)
    for step in range(bridge_steps):
        batch = next(stage_c_loader)  # use Stage C data
        margin_loss = pairwise_margin_loss(scores, pairs, margin=0.2)
        bce_loss = F.binary_cross_entropy(scores, grades_normalized)
        loss = 0.7 * margin_loss + 0.3 * bce_loss
        loss.backward()
        optimizer.step()
```

The bridge uses Stage C data but with Stage B's dominant loss signal (pairwise margin at 0.7 weight). This lets the pair encoder see graded relevance labels for the first time without abandoning its learned discrimination capacity.

#### Issue 4.6 — Add MGRH CLI mode

Add `--mgrh_train` CLI flag (or use config-driven `mgrh_training.enabled`):

- `--mgrh_stage a` → general NLI pre-training
- `--mgrh_stage b` → domain NLI adaptation
- `--mgrh_stage bridge` → B→C bridge (500 steps, auto-runs before C if enabled)
- `--mgrh_stage c` → relevance fine-tuning

Each stage loads the best checkpoint from the previous stage. The bridge is auto-triggered before Stage C unless `--skip_bridge` is passed. This sequential staging pattern already exists in the bakeoff script (stage_a → stage_b flow).

#### Issue 4.7 — Add domain saliency weighting from frozen TemporalHead + NERHead

The model already has trained, production-quality TemporalHead and GlobalPointerNERHead (ner_family). These output token-level logits/scores. Use them as attention biases in the pair encoder:

```python
# Get temporal saliency from frozen TemporalHead token logits:
temporal_logits = model.temporal_head(hidden_states, attention_mask)  # [B, L, num_temporal_labels]
temporal_saliency = temporal_logits.max(dim=-1).values.softmax(dim=-1)  # [B, L]

# Get entity saliency from frozen GlobalPointerNERHead span scores:
# (use diagonal of span matrix as per-token importance)
ner_scores = model.ner_family_head(hidden_states, attention_mask)  # [B, num_types, L, L]
entity_saliency = ner_scores.diagonal(dim1=-2, dim2=-1).max(dim=1).values.softmax(dim=-1)  # [B, L]

# Combine into saliency bias for pair encoder cross-attention:
saliency_bias = 0.5 * temporal_saliency + 0.5 * entity_saliency  # [B, L]
```

Pass `saliency_bias` as an additive attention bias in the CrossAttentionPairEncoder. This causes the pair encoder to attend more heavily to temporal tokens and entity-bearing tokens — directly targeting the two worst query types (temporal: separation=0.0064, entity: separation=0.0952).

No additional training data or parameters required — both heads are frozen and running at inference time anyway.

#### Issue 4.8 — Add ANCE dynamic hard negative mining

ANCE (Approximate Nearest Neighbor Negative Contrastive Estimation, Xiong et al. 2020) periodically re-encodes the corpus with current model weights and refreshes the hard negative pool.

Implementation:

```python
def refresh_hard_negatives(
    model: ModernBERTMultiTask,
    query_texts: list[str],
    corpus_texts: list[str],
    relevance_labels: dict[str, set[int]],  # query_id -> set of relevant corpus indices
    top_k: int = 20,
    max_negatives: int = 5000,
) -> list[dict]:
    """Re-mine hard negatives using current model scores."""
    # 1. Batch-encode all corpus texts with current MGRH
    # 2. For each query, score top-K corpus items
    # 3. Take highest-scoring NON-RELEVANT items as new hard negatives
    # 4. Return refreshed negative pool
```

Schedule: run every 3 epochs during Stage C. After epoch 2 the initial static negatives become easy — ANCE ensures the model always trains on the hardest current negatives.

The EmbeddingDataset already has slice/index structure that makes hot-swapping the negative pool feasible between epochs.

#### Issue 4.9 — Add post-training score calibration

After Stage C training is complete, calibrate the MGRH output scores on the holdout set using temperature scaling:

```python
# Single-parameter temperature scaling on holdout:
T = nn.Parameter(torch.ones(1))
optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=50)

def calibration_loss():
    optimizer.zero_grad()
    scaled_logits = raw_logits / T    # raw_logits before sigmoid
    loss = F.binary_cross_entropy_with_logits(scaled_logits, true_labels)
    loss.backward()
    return loss

optimizer.step(calibration_loss)
# Save T alongside MGRH checkpoint
```

This ensures the MGRH relevance score is properly calibrated for the `mu`-weighted interpolation with v4 fusion scores: `final = mu * v4_score + (1-mu) * calibrated_mgrh_score`. Without calibration, the interpolation weight mu is meaningless.

### Deliverable

Same script can now run: bakeoff, stage_b, teacher cache, distillation, AND mgrh training. No new script files.

---

## Milestone 5 — Wire MGRH through Client API

This is the "make it usable" milestone.

### Why

The NLI doc's Phase 7 probe showed the existing client has NO path for pair-input relevance scoring. The `analyze()` method is single-text only. The `similarity()` method is bi-encoder cosine only.

### Files

- `familyos_ultrabert/client.py`
- `familyos_ultrabert/runtime.py` (if inference routing lives here)
- `src/modeling_studio/models/unified_output.py` (pair tokenization already exists at line 996)

### Issues

#### Issue 5.1 — Add client.score_relevance(query, passage) -> float

New client method that:

1. Tokenizes `[CLS] query [SEP] passage [SEP]` (reuse existing pair tokenization from `unified_output.py:996`)
2. Forwards through frozen encoder
3. Forwards through MGRH head
4. Returns `float` relevance score 0.0-1.0

#### Issue 5.2 — Add client.rerank(query, candidates, top_k) -> list[dict]

Batch re-ranking method that:

1. Tokenizes all `(query, candidate)` pairs
2. Batched forward through encoder + MGRH head (single batch, not serial)
3. Returns sorted list of `{"text": str, "score": float, "original_index": int}`
4. Latency target: < 50ms for batch of 20

#### Issue 5.3 — Add MGRH weight loading in model initialization

When the client loads the model, it must also load the MGRH head weights and pair_encoder weights from the checkpoint. This extends the existing head loading path in `load_model_and_replace_embedding_head()`.

### Deliverable

Production-ready client API for cross-encoder re-ranking. `score_relevance()` and `rerank()` are callable alongside existing `analyze()`, `similarity()`, `find_similar()`.

---

## Milestone 6 — Validation and execution sequence

This is the "how we actually train and validate" milestone.

### Why

MGRH has a 3-stage training pipeline with specific data, hyperparameters, and evaluation gates at each stage. The execution order matters — Stage C depends on Stage A+B weights.

### Issues

#### Issue 6.1 — Stage A execution: general NLI pre-training

```
# In Colab or local:
python scripts/training/train_embedding_heads_bakeoff.py \
  --config configs/training/embedding_heads_bakeoff.yaml \
  --mgrh_train --mgrh_stage a
```

Data: ~854K general NLI pairs (MNLI + ANLI + WANLI + FEVER-NLI)
Duration: 5 epochs
Gate: NLI dev accuracy > 85% (MNLI matched dev)
Output: `outputs/mgrh-stage-a/best/`

#### Issue 6.2 — Stage B execution: domain NLI adaptation

```
python scripts/training/train_embedding_heads_bakeoff.py \
  --config configs/training/embedding_heads_bakeoff.yaml \
  --mgrh_train --mgrh_stage b \
  --checkpoint outputs/mgrh-stage-a/best/
```

Data: ~32K FamilyOS domain triplets (wrong_time, wrong_person, hard_negatives)
Duration: 5 epochs
Gate: pairwise accuracy > 80% on dev set
Output: `outputs/mgrh-stage-b/best/`

#### Issue 6.2b — Bridge B→C execution

```
# Auto-triggered before Stage C unless --skip_bridge is passed:
python scripts/training/train_embedding_heads_bakeoff.py \
  --config configs/training/embedding_heads_bakeoff.yaml \
  --mgrh_train --mgrh_stage bridge \
  --checkpoint outputs/mgrh-stage-b/best/
```

Data: Stage C data (human benchmark listwise)
Duration: 500 steps (fixed, not epoch-based)
Loss: 0.7 * PairwiseMargin + 0.3 * RelevanceBCE
No gate — bridge is a transition, not a training target
Output: `outputs/mgrh-bridge-bc/best/`

#### Issue 6.3 — Stage C execution: relevance fine-tuning

```
python scripts/training/train_embedding_heads_bakeoff.py \
  --config configs/training/embedding_heads_bakeoff.yaml \
  --mgrh_train --mgrh_stage c \
  --checkpoint outputs/mgrh-bridge-bc/best/
```

Data: ~8.9K relevance triples (human benchmark listwise + mined hard negatives)
Duration: 10 epochs
Gate: Spearman > 0.50, AUC-ROC > 0.80, nDCG@10 > 0.83
Output: `outputs/mgrh-stage-c/best/`

#### Issue 6.4 — End-to-end re-ranking validation

Run the v4 retrieval pipeline + MGRH re-ranking on the full human benchmark:

1. v4 pipeline produces top-20 candidates per query
2. MGRH re-ranks top-20 → top-10
3. Measure:
   - MRR lift over bi-encoder baseline (target: 0.81 → 0.85-0.89)
   - nDCG lift (target: 0.79 → 0.83-0.87)
   - Per-query-type improvement (temporal is primary target)
4. Determine optimal interpolation weight (mu) between v4 score and MGRH score
5. Latency validation: < 50ms for batched re-ranking of top-20

#### Issue 6.5 — Failure escalation paths

If Stage C gate metrics are not met:

- Spearman < 0.35 → enable MS MARCO warm-start (50K triplets, 2-3 epochs before Stage C)
- Spearman 0.35-0.50 → increase Stage C epochs to 20, add learning rate warmup restart
- AUC-ROC < 0.70 → switch from combined loss to pure LambdaRank, remove pairwise margin component

If overall re-ranking MRR lift < 5pp:

- Check if pair_encoder is undertrained → unfreeze for 2 additional epochs at 1e-5
- Check if Signal 3 (asymmetric embeddings) is contributing → ablation study removing Signal 3
- Last resort: escalate to Option B (dedicated RelevanceRegressionHead, simpler 768→256→1 architecture without 3-signal fusion)

### Deliverable

A trained MGRH head with documented performance on all gate metrics, integrated into the client API, with < 50ms batch re-ranking latency.

---

## Issue breakdown as GitHub-style tickets

### Epic

**Multi-Granularity Relevance Head (MGRH) — SOTA NLI Re-Ranking for Episode Retrieval**

### Milestone A — Head design and registration

- Issue A1: Implement `MultiGranularityRelevanceHead` in `heads.py` (4-signal fusion + LayerNorm(4609) + two-head output: `nli_head(256,3)` + `relevance_head(256,1)`)
- Issue A2: Add `relevance_score()` accessor method
- Issue A3: Add MGRH to `__all__` exports in `heads.py`
- Issue A4: Implement `LambdaRankLoss` and `CombinedRankingLoss` in `losses_ranking.py`
- Issue A5: Implement `RDropLoss` wrapper in `losses_ranking.py`

### Milestone B — Data curation and staging

- Issue B1: Download and format general NLI datasets (MNLI, ANLI, WANLI, FEVER-NLI) to unified JSONL
- Issue B2: Symlink/validate existing FamilyOS hard negatives for domain adaptation
- Issue B3: Format human benchmark as listwise JSONL for LambdaRank training
- Issue B4: Create train/dev/holdout splits (80/10/10, stratified by source)

### Milestone C — Training config extension

- Issue C1: Add `mgrh_training` section to `embedding_heads_bakeoff.yaml`
- Issue C2: Add stage-specific data sources and loss configs (stage_a, stage_b, stage_c)
- Issue C3: Add MGRH evaluation config with gate metrics
- Issue C4: Backward compatibility guard (`mgrh_training.enabled: false` default)

### Milestone D — Training script extension

- Issue D1: Add NLI dataset loader for Stage A premise/hypothesis pairs
- Issue D2: Add relevance listwise dataset loader for Stage C query groups
- Issue D3: Add MGRH model loading path (head + pair_encoder + frozen AgreementGatedHeadV2 + frozen TemporalHead + frozen NERHead)
- Issue D4: Add MGRH forward/train_step (4-signal forward, stage-specific loss, contrastive NLI aux for Stage A)
- Issue D4a: Add R-Drop regularization to Stage C train_step
- Issue D4b: Add Stage B→C bridge training step (500 steps, 0.7*margin + 0.3*BCE)
- Issue D5: Add MGRH evaluation functions (Spearman, AUC-ROC, nDCG@10, per-type breakdown)
- Issue D6: Add `--mgrh_train` / `--mgrh_stage` CLI flags (including `bridge` stage)
- Issue D7: Add domain saliency weighting from frozen TemporalHead + GlobalPointerNERHead
- Issue D8: Add ANCE dynamic hard negative mining (refresh every 3 epochs in Stage C)
- Issue D9: Add post-training temperature scaling calibration on holdout set

### Milestone E — Client API wiring

- Issue E1: Add `client.score_relevance(query, passage)` → float
- Issue E2: Add `client.rerank(query, candidates, top_k)` → sorted list
- Issue E3: Add MGRH weight loading in model initialization

### Milestone F — Validation and execution

- Issue F1: Execute Stage A (general NLI pre-training, gate: dev accuracy > 85%)
- Issue F2: Execute Stage B (domain adaptation, gate: pairwise accuracy > 80%)
- Issue F2b: Execute Bridge B→C (500 steps, no gate — transition only)
- Issue F3: Execute Stage C (relevance fine-tuning, gate: Spearman > 0.50, AUC > 0.80)
- Issue F4: End-to-end re-ranking validation (MRR lift, nDCG lift, latency)
- Issue F5: Document failure escalation paths and fallback options
- Issue F6: Ablation study — measure contribution of each SOTA enhancement (MaxSim, 2-layer ESIM, R-Drop, ANCE, domain saliency, contrastive NLI, calibration)

---

## Architecture summary

```
MGRH 4-Signal Architecture (SOTA-enhanced):

                     ┌──────────────────────────────────────────────────────┐
query text ──────────┤                                                       │
                     │  [CLS] query [SEP] episode [SEP]                      │
episode text ────────┤  → ModernBERT (22 layers, FROZEN)                    │
                     │  → h_joint  (768-dim per token)                       │
                     └───────────────┬───────────────────────────────────────┘
                                     │
          Signal 1: h_cls = cls_proj(h_joint[:, 0])  ─────────────────┐
                                     │                                  │
          Signal 2: h_cross = CrossAttentionPairEncoder(               │
                        text_a_hidden, text_b_hidden,                   │
                        saliency_bias=temporal+entity)  ───────────────┤
                        [2-layer ESIM, domain saliency weighted]        │
                                     │                                  │
          Signal 3: interaction = [q_emb, d_emb,                       │
                        q_emb*d_emb, |q_emb-d_emb|]                   │
                        (from frozen AgreementGatedHeadV2) ────────────┤
                                     │                                  │
          Signal 4: maxsim = ColBERT MaxSim(                           │
                        text_a_hidden, text_b_hidden)  ────────────────┤
                        [token-level alignment, no learned params]      │
                                     │                                  │
                     ┌───────────────▼──────────────────────────────────┤
                     │  LayerNorm(4609)                                  │
                     │  [h_cls | h_cross | interaction | maxsim_znorm]  │
                     │  4609 → 1024 → 256 → nli_head(3) + rel_head(1)  │
                     └───────────────┬──────────────────────────────────┘
                                     │
                         relevance_score ∈ [0.0, 1.0]
                                     │
                         (post-training: temperature calibration)

  Domain saliency path (frozen, no backward cost):
    TemporalHead token logits ──┐
                                 ├──→ saliency_bias → pair encoder attention bias
    NERHead (ner_family) spans ──┘
```

```
3-Stage Training Pipeline (SOTA-enhanced):
  Stage A: General NLI (854K pairs) ──→ teaches entailment reasoning
           + contrastive NLI auxiliary (SimCSE-style: entailment=pos, contradiction=hard neg)
  Stage B: Domain NLI (32K FamilyOS) ──→ teaches kinship/temporal patterns
  Stage C: Relevance fine-tune (8.9K) ──→ teaches graded re-ranking
           + R-Drop regularization (KL consistency, prevents small-data overfitting)
           + ANCE dynamic hard negative refresh (every 3 epochs)
           + post-training temperature calibration on holdout

  Loss progression:
  A: CrossEntropy(nli_head, 3-class) + 0.1*BCE(relevance_head, warmup) + 0.3*ContrastiveNLI
  B: PairwiseMargin(hard negatives) via relevance_head
  Bridge B→C: 500 steps of 0.7*PairwiseMargin + 0.3*RelevanceBCE (smooth transition)
  C: LambdaRank(nDCG) + PairwiseMargin(hard negatives) + R-Drop KL(alpha=1.0)
```

---

## What is correct about the approach

- MGRH reuses existing components (pair_encoder, AgreementGatedHeadV2, TemporalHead, NERHead) — no reinvention
- 4-signal fusion addresses every specific Phase 7 failure mode:
  - Signal 1 (CLS): global coherence — handles thematic/causal queries
  - Signal 2 (2-layer ESIM + domain saliency): token alignment with temporal/entity attention bias — targets worst query types
  - Signal 3 (asymmetric embeddings): query vs document role distinction — reuses AgreementGatedHeadV2 mode prompts
  - Signal 4 (MaxSim): fine-grained per-token alignment — directly catches keyword-overlap-but-wrong-semantics (S7-1)
- Data exists: hard_negatives (42K), mined_v2 (32K), human benchmark (4.4K) — no external data required for domain stages
- General NLI pre-training follows the proven MNLI + FEVER + ANLI recipe from DeBERTa-v3-base-mnli-fever-anli
- Training script reuse avoids pipeline fragmentation
- Freeze encoder policy matches existing bakeoff design — only head + pair_encoder are trainable
- 7 SOTA enhancements push estimated Spearman from 0.60-0.70 toward 0.70-0.80

## What needs careful attention

- ~~Stage A trains the MGRH as a 3-class NLI head first; Stage C switches to regression.~~ **RESOLVED (v2 review):** Two-head output design (`nli_head` + `relevance_head`) eliminates the cold-start weight reset. Both heads are instantiated at construction. Stage A trains `nli_head` as primary + `relevance_head` as 0.1-weighted auxiliary warmup. Stage B+C train only `relevance_head`. No weight reset, no representation loss.
- LambdaRank requires per-query document groups, not independent pairs. The dataloader and collator must batch by query group, not randomly.
- Signal 3 (asymmetric embeddings from AgreementGatedHeadV2) requires running the embedding head in inference mode during MGRH training. This adds forward-pass cost but not backward-pass cost (frozen).
- Signal 4 (MaxSim) and domain saliency also run frozen heads — total inference overhead is ~3 frozen head forwards per training step, but zero backward cost.
- R-Drop doubles the forward passes in Stage C. With frozen encoder + small head this is acceptable (~2x head forward cost, not encoder cost).
- ANCE hard negative refresh requires a full corpus re-encoding every 3 epochs. For ~88 episodes x top-20 this is fast (~5 seconds). For larger corpora, batch encoding is essential.
- Latency target (< 50ms for batch-20) requires efficient batched pair encoding. Serial pair encoding (20 x 15ms = 300ms) is not acceptable.
- Post-training calibration must use the holdout set (not dev). Do not leak calibration data into training or model selection.
- **MaxSim magnitude mismatch (v2 review fix):** Raw MaxSim is an unbounded dot-product scalar. Without normalization, the fusion MLP's first layer either ignores it or it dominates. Fix: z-score per batch + `LayerNorm(4609)` before the fusion MLP. Both are in the constructor.
- **Stage B→C loss discontinuity (v2 review fix):** Stage B trains purely on pairwise margin. Stage C starts cold on graded LambdaRank. A 500-step bridge with joint `0.7*margin + 0.3*BCE` prevents destabilizing the pair encoder's learned representations.
- **ANLI license (SHIP BLOCKER):** ANLI R1-R3 is CC-BY-NC-4.0. If FamilyOS has commercial use, ANLI must be replaced with DocNLI (Apache 2.0) or dropped entirely. Decision required before B1.

---

## SOTA enhancements summary

7 enhancements integrated into the plan, ordered by priority:

| # | Enhancement | Impact | Effort | Where Applied |
|---|---|---|---|---|
| 1 | ANCE dynamic hard negative mining | Very high — prevents stale negatives after epoch 2 | Medium (periodic re-index) | Stage C, every 3 epochs (Issue 4.8) |
| 2 | ColBERT MaxSim as Signal 4 | High — catches keyword-overlap-but-wrong-semantics | Low (~15 lines, no new params) | Head architecture (Issue 1.1) |
| 3 | 2-layer pair encoder | High — chain-of-alignment reasoning | Trivial (config: `num_layers: 2`) | Config (Issue 3.1) |
| 4 | R-Drop regularization | Medium-high — free quality, prevents Stage C overfitting | Very low (~5 lines in train_step) | Stage C loss (Issue 4.4a) |
| 5 | Domain saliency from TemporalHead + NERHead | High for temporal queries (worst type) | Medium | Pair encoder attention bias (Issue 4.7) |
| 6 | Contrastive NLI pre-training in Stage A | Medium — geometrically organizes NLI representations | Low | Stage A loss (Issue 3.2 config) |
| 7 | Post-training score calibration | Medium — critical for mu-weighted interpolation | Trivial (~10 lines) | Post Stage C (Issue 4.9) |

Expected cumulative effect on Spearman: 0.60-0.70 (base 3-signal) → **0.70-0.80** (with all 7 enhancements)

---

## What next

### Sequencing verdict (v2 review)

**A1 + A4 (+ LayerNorm + two-head design) → B1 (resolve ANLI license FIRST) → C1-C4 → D1-D3 → F1.**

Do not touch Stage C data until Stage A gates at 85%.

1. **Immediate:** Issue A1 — implement `MultiGranularityRelevanceHead` in `heads.py` with LayerNorm(4609), z-score MaxSim, and two-head output (`nli_head` + `relevance_head`). This is the design anchor.
2. **Parallel with A1:** Issue A4 + A5 — implement `LambdaRankLoss`, `CombinedRankingLoss`, `RDropLoss` in `losses_ranking.py`.
3. **Before B1:** Resolve ANLI license blocker. If commercial → replace with DocNLI (Apache 2.0) or drop ANLI and proceed with MNLI+WANLI+FEVER+SNLI (1.25M pairs).
4. **Then:** C1-C4 (config), D1-D3 (script loaders), F1 (execute Stage A, gate at 85% NLI dev accuracy).
5. **Only after F1 gates:** F2 → Bridge → F3 → F4 (end-to-end validation).

---

## Updated checklist

- [x] Read NLI reranking gap doc end-to-end
- [x] Explored heads.py registration pattern (14 heads, BaseHead hierarchy, NLIHead pair_encoder support)
- [x] Explored data/familyos dataset structure (embeddings/hard_negatives, mined_v2, silver_synthetic)
- [x] Explored training YAML config (6 data sources, 7-head bakeoff, stage_b, composite scoring)
- [x] Explored training script (4700 lines, dataset/collator/sampler/forward/eval/train loop)
- [x] Read existing plan template (milestone/issue structure, config-first, backward compatibility)
- [x] Created MGRH implementation plan with 6 milestones and 23 issues
- [x] Added 7 SOTA enhancements: ColBERT MaxSim (Signal 4), 2-layer ESIM, R-Drop, ANCE dynamic mining, domain saliency, contrastive NLI, score calibration
- [x] Upgraded architecture from 3-signal to 4-signal fusion
- [x] Upgraded targets from Spearman>0.50 to Spearman>0.70
- [x] Added 8 new issues (A5, D4a, D7, D8, D9, F6) — total: 31 issues across 6 milestones
- [x] **v2 review fix:** Two-head output design (`nli_head` + `relevance_head`) — eliminates cold-start weight reset between Stage A→C
- [x] **v2 review fix:** LayerNorm(4609) before fusion MLP + z-score MaxSim normalization — prevents magnitude mismatch
- [x] **v2 review fix:** ANLI CC-BY-NC-4.0 elevated to SHIP BLOCKER with DocNLI/NLI-in-the-wild alternatives
- [x] **v2 review enhancement:** Stage B→C bridge (500-step joint 0.7*margin + 0.3*BCE) — prevents loss discontinuity
- [x] Updated sequencing verdict: A1+A4 → resolve ANLI license → C1-C4 → D1-D3 → F1
- [x] Added issues D4b, F2b — total: 33 issues across 6 milestones
