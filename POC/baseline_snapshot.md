# Baseline Snapshot

Date: 2026-02-27
Mode: golden
Model: familyos_ultrabert(pytorch)
Dataset source: `POC/manual_golden_batches/*.json` (10 files, 500 tools)

## Current baseline (hardwired UltraBERT pipeline run)

- Template key: `hardwired_hybrid_sentence`
- Input-style ablation winner: `hybrid_sentence`

### Input style scores

- structured_kv: p@5=0.605, mrr=0.716, tpr@10=0.756, dmr@10=0.956, omr@10=0.837
- hybrid_sentence: p@5=0.676, mrr=0.696, tpr@10=0.795, dmr@10=0.926, omr@10=0.795
- natural_sentence: p@5=0.503, mrr=0.525, tpr@10=0.634, dmr@10=0.883, omr@10=0.634

### KPI framing

- Exact tool presence is improving but below release goal.
- Domain/operation coverage is sensitive to input style and reranking strategy.
- Next leverage remains YAML normalization quality on first 100 tools.
