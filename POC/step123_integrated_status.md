# Integrated Status: Step 1 + Step 2 + Step 3

Date: 2026-02-27

This document captures an integrated execution snapshot where Step 1 normalization, Step 2 ontology/mapping QA, and Step 3 fusion calibration are validated together.

## Scope

- Corpus: manual golden batches (`500` tools)
- Model: `familyos_ultrabert(pytorch)`
- Evaluation script: `POC/evaluate_ultrabert_embeddings.py`

## Step 1 status (YAML normalization)

Evidence from latest integrated run:

- Loader marker present: `step1_normalized_items:100`
- Normalization scope remains first `100` tools (batches 001 and 002)

Current view:

- Step 1 is active and contributing to retrieval quality.
- Remaining long-term hardening item: convert loader-time normalization into durable per-file manual edits.

## Step 2 status (ontology expansion + mapping QA)

Latest integrated QA metrics:

- Sub-cluster mapping coverage: `100/100 (1.000)`
- Query token ontology coverage: `0.947`
- Unmapped sub-clusters: `none`

Step 2 gate check:

- Intermediate gate `OMR@10 >= 0.90`: `PASS`

## Step 3 status (fusion calibration)

Mini-grid executed with multiple candidate configs.

Best tie-break config from grid:

- `hybrid_vector_weight=0.80`
- `hybrid_bm25_weight=0.20`
- `rerank_domain_bonus=0.20`
- `rerank_operation_token_bonus=0.20`
- `rerank_token_overlap_weight=0.20`
- `rerank_candidate_k=100`

Best observed hybrid metrics:

- `TPR@10 = 0.961`
- `DMR@10 = 1.000`
- `OMR@10 = 0.963`
- `P@5 = 0.848`
- `MRR = 0.875`

Step 3 gate check:

- `TPR@10 >= 0.50`: `PASS`
- `DMR@10 >= 0.98`: `PASS`

## Combined interpretation

- Step 1 + Step 2 + Step 3 together are stable and high-performing on the manual golden benchmark.
- Current bottleneck is no longer ontology coverage; next gains should focus on regression-proofing and broader robustness checks.

## Next action

1. Add regression guardrails (automatic fail if key metrics dip below policy thresholds).
2. Run one larger sweep for stability across query style variants (`llm_clean`, `llm_noisy`).
3. Start Step 4 miss-case diagnostics for any remaining exact-target misses.

## Update

- Recommended Step 3 fusion config has been locked as default constants in `POC/evaluate_ultrabert_embeddings.py`.
