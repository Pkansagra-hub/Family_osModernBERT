# FamilyOS UltraBERT Release Guide

## Release model

FamilyOS UltraBERT releases are lightweight package releases.

- The wheel and source distribution contain Python code only.
- Runtime weights are downloaded from Hugging Face.
- Current encoder source of truth: `Pkansagra/ultrabert-weights` at `encoder/v2/fp32/`.

## Current Release: v4.0.9 — MGRH Production Release

### What shipped

1. **MultiGranularityRelevanceHead (46.3M params)** — cross-encoder reranking head fusing CLS, CrossAttention (2-layer bidirectional ESIM), embedding interaction, and ColBERT MaxSim signals.
2. **4-stage training curriculum** — NLI Warmup (A) -> Pairwise Margin (B) -> Bridge -> LambdaRank Listwise (C) with ANCE hard-negative mining.
3. **Client API**: `client.score_relevance(query, doc)` for pointwise relevance, `client.rerank(query, docs, top_k)` for batched reranking.
4. **MaxSim population z-normalization** — pre-computed statistics (mean=933.45, std=85.98) for stable single-pair scoring; batch z-norm for rerank batches.
5. **Temperature calibration** — inference temperature = 0.818 (learned post-training).
6. **All 5 gate checks passed** — Spearman, AUC-ROC, nDCG@10, holdout Spearman, holdout AUC.

### Results

- **Spearman 0.9090** on human benchmark (798 groups)
- **AUC-ROC 0.9882** — all metrics above threshold
- **nDCG@10 0.9997** — near-perfect top-10 ordering
- **Wrong Person 98.43%**, Wrong Time 94.40%, Hard Negatives 97.50%
- Grade 3 vs 0 separation: **0.889** | ECE: **0.030**

### Bug fixes since v4.0.8

- **ANCE OOM crash at epoch 4**: `refresh_hard_negatives_ance()` was missing `@torch.no_grad()`, building computation graphs for every sample. Fixed with decorator + batched processing.
- **Export head key mismatch**: Export script used `heads["mgrh"]` instead of `heads["relevance"]`. Fixed across export, metadata, and pair_encoder tensor cloning.
- **MaxSim z-norm priority**: Population z-norm was applied before batch z-norm, degrading batched rerank quality. Reversed priority: batch z-norm first (preserves trained behaviour), population z-norm as fallback for batch_size=1.
- **7 training bugs**: LambdaRank loss, per-group Spearman eval, collator max_length, MaxSim population stats, mgrh_metadata.json save, Grade 2 oversampling, query_doc drowning.

### What the MGRH is NOT

- It does **not** replace the existing `nli` head (3-class: entailment/neutral/contradiction). The NLI head remains one of the core capabilities in `client.analyze()`.
- MGRH is an **additional** cross-encoder reranking capability, complementary to the bi-encoder embedding path.

## Release artifacts

A standard release should produce these files under `familyos_ultrabert/dist/`:

- `familyos_ultrabert-<version>-py3-none-any.whl`
- `familyos_ultrabert-<version>.tar.gz`
- `SHA256SUMS.txt`
- `RELEASE_<version>.md`
- `familyos_ultrabert-<version>-release.zip`

The zip bundle is a convenience asset for GitHub Releases. It should contain the wheel, source distribution, checksums, and release summary.

## Build and validation

From the repository root, run:

```bash
python scripts/prepare_release.py --version 4.0.9 --test-install --generate-checksums --write-release-summary --build-bundle
```

This will:

1. Validate package structure
2. Build the wheel and source distribution
3. Verify weights are excluded from the package
4. Smoke-test installation in a clean virtual environment
5. Generate checksums
6. Write a GitHub-friendly release summary
7. Create a convenience zip bundle

## GitHub release flow

1. Ensure the runtime weights are already published to Hugging Face.
2. Commit and push the release-ready repository state.
3. Create and push the tag:

```bash
git tag -a v4.0.9 -m "FamilyOS UltraBERT v4.0.9"
git push origin v4.0.9
```

1. Publish the GitHub release for that tag.

The GitHub Actions workflow in `.github/workflows/release.yml` will:

- rebuild the package
- run the release prep script
- publish the package to PyPI
- attach the wheel, source distribution, checksum file, release summary, and zip bundle to the GitHub release

## Manual installation examples

Install from a release wheel:

```bash
pip install familyos_ultrabert-4.0.8-py3-none-any.whl
```

Install with PyTorch support:

```bash
pip install familyos_ultrabert-4.0.8-py3-none-any.whl torch
```

Install from source:

```bash
git clone https://github.com/Pkansagra-hub/Family_osModernBERT.git
cd Family_osModernBERT
pip install ./familyos_ultrabert[pytorch]
```

## Notes

- Keep `familyos_ultrabert/RELEASE_NOTES.md` as the cumulative feature log.
- Treat `familyos_ultrabert/dist/RELEASE_<version>.md` as the per-release asset summary.
- Do not bundle local model weights into the wheel or source distribution.

## License

Proprietary - All Rights Reserved. See `LICENSE` for details.
