# Fusion Tuning Results (Step 3)

Scope: UltraBERT retrieval fusion calibration on manual golden set (500 tools)

## Objective

Tune hybrid retrieval fusion parameters to maximize exact tool presence in top-10 while preserving domain and operation coverage.

Primary gate for Step 3:

- `TPR@10 >= 0.50`
- `DMR@10 >= 0.98`

## Baseline configuration (current)

From `POC/evaluate_ultrabert_embeddings.py`:

- `HYBRID_VECTOR_WEIGHT = 0.80`
- `HYBRID_BM25_WEIGHT = 0.20`
- `RERANK_DOMAIN_BONUS = 0.20`
- `RERANK_OPERATION_TOKEN_BONUS = 0.20`
- `RERANK_TOKEN_OVERLAP_WEIGHT = 0.20`
- `RERANK_CANDIDATE_K = 100`

Current baseline metrics (`report.txt`, 2026-02-27):

- Hybrid `TPR@10`: `0.961`
- Hybrid `DMR@10`: `1.000`
- Hybrid `OMR@10`: `0.963`
- Hybrid `P@5`: `0.848`
- Hybrid `MRR`: `0.875`

## Step 3 checklist (in progress)

- [x] Freeze baseline config and metrics
- [x] Define tuning grid (weights and bonus terms)
- [x] Execute grid search runs (mini-grid completed)
- [x] Rank candidates by gate compliance and tie-break metrics
- [x] Select best config and document rationale
- [ ] Add regression guardrails for chosen config

## Planned tuning grid

1. Fusion weights
   - `vector`: `[0.50, 0.60, 0.65, 0.70, 0.80]`
   - `bm25`: `[0.50, 0.40, 0.35, 0.30, 0.20]`
   - Constraint: `vector + bm25 = 1.0`

2. Bonus terms
   - `domain bonus`: `[0.10, 0.15, 0.20, 0.25]`
   - `operation bonus`: `[0.10, 0.15, 0.20, 0.25]`
   - `token overlap weight`: `[0.10, 0.20, 0.25, 0.30]`

3. Candidate pool
   - `RERANK_CANDIDATE_K`: `[50, 100, 150]`

## Experiment log

Run `baseline`

- vector: `0.65`
- bm25: `0.35`
- domain_bonus: `0.20`
- op_bonus: `0.15`
- token_overlap: `0.25`
- cand_k: `100`
- TPR@10: `0.957`
- DMR@10: `1.000`
- OMR@10: `0.959`
- P@5: `0.844`
- MRR: `0.872`
- status: `pass`

Run `tune_001_vector70_op20_tok20`

- vector: `0.70`
- bm25: `0.30`
- domain_bonus: `0.20`
- op_bonus: `0.20`
- token_overlap: `0.20`
- cand_k: `100`
- TPR@10: `0.961`
- DMR@10: `1.000`
- OMR@10: `0.963`
- P@5: `0.846`
- MRR: `0.873`
- status: `pass`

Run `tune_002_vec60_bm40_dom20_op20_tok20_k100`

- vector: `0.60`
- bm25: `0.40`
- domain_bonus: `0.20`
- op_bonus: `0.20`
- token_overlap: `0.20`
- cand_k: `100`
- TPR@10: `0.961`
- DMR@10: `1.000`
- OMR@10: `0.963`
- P@5: `0.844`
- MRR: `0.873`
- status: `pass`

Run `tune_003_vec80_bm20_dom20_op20_tok20_k100`

- vector: `0.80`
- bm25: `0.20`
- domain_bonus: `0.20`
- op_bonus: `0.20`
- token_overlap: `0.20`
- cand_k: `100`
- TPR@10: `0.961`
- DMR@10: `1.000`
- OMR@10: `0.963`
- P@5: `0.848`
- MRR: `0.875`
- status: `pass` (best tie-break)

Run `tune_004_vec70_bm30_dom15_op25_tok20_k100`

- vector: `0.70`
- bm25: `0.30`
- domain_bonus: `0.15`
- op_bonus: `0.25`
- token_overlap: `0.20`
- cand_k: `100`
- TPR@10: `0.961`
- DMR@10: `1.000`
- OMR@10: `0.963`
- P@5: `0.846`
- MRR: `0.873`
- status: `pass`

Run `tune_005_vec70_bm30_dom20_op20_tok20_k150`

- vector: `0.70`
- bm25: `0.30`
- domain_bonus: `0.20`
- op_bonus: `0.20`
- token_overlap: `0.20`
- cand_k: `150`
- TPR@10: `0.961`
- DMR@10: `1.000`
- OMR@10: `0.963`
- P@5: `0.846`
- MRR: `0.873`
- status: `pass`

## Ranking and selection

Ranking rule: maximize `TPR@10`, then `OMR@10`, then `P@5`, then `MRR`.

Top candidate after tie-break:

- `tune_003_vec80_bm20_dom20_op20_tok20_k100`
  - `TPR@10=0.961`, `DMR@10=1.000`, `OMR@10=0.963`, `P@5=0.848`, `MRR=0.875`

Recommended Step 3 config:

- `hybrid_vector_weight=0.80`
- `hybrid_bm25_weight=0.20`
- `rerank_domain_bonus=0.20`
- `rerank_operation_token_bonus=0.20`
- `rerank_token_overlap_weight=0.20`
- `rerank_candidate_k=100`

## Notes

- Because baseline already exceeds Step 3 gate strongly, tuning should prioritize robustness and regression resistance over incremental gain chasing.
- Final selection should preserve or improve `TPR@10` without reducing `DMR@10` below `0.98`.
