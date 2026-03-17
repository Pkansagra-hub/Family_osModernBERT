# FamilyOS UltraBERT Release Guide

## Release model

FamilyOS UltraBERT releases are lightweight package releases.

- The wheel and source distribution contain Python code only.
- Runtime weights are downloaded from Hugging Face.
- Current encoder source of truth: `Pkansagra/ultrabert-weights` at `encoder/v2/fp32/`.

## Current Release: v4.0.8 — Multi-Granularity Relevance Head (MGRH)

### What shipped

1. **MultiGranularityRelevanceHead (46.3M params)** — cross-encoder reranking head fusing CLS, CrossAttention (2-layer bidirectional ESIM), embedding interaction, and ColBERT MaxSim signals.
2. **Client API**: `client.score_relevance(query, doc)` for pointwise relevance, `client.rerank(query, docs, top_k)` for batched reranking.
3. **Population z-normalization** for MaxSim — pre-computed statistics (mean=845.56, std=60.75) ensure consistent signal scaling at any batch size.
4. **Temperature calibration** — inference temperature = 0.300 (composite-optimal from sweep on 400 holdout triplets).
5. **Padding fix** — `rerank()` no longer pads to 512 tokens, only to longest in batch.

### Results

- **Spearman 0.9043** on human benchmark (798 groups) — +0.55 over bi-encoder
- **nDCG@10 0.9867** — near-perfect top-10 ordering
- **Wrong Person 93.13%**, Wrong Time 91.53%, Sentiment Flip 96.13%
- **Holdout pairwise accuracy 95.0%**
- `score_relevance()` and `rerank()` produce identical scores (verified to 6 decimal places)

### What the MGRH is NOT

- It does **not** replace the existing `nli` head (3-class: entailment/neutral/contradiction). The NLI head remains one of the 12 core capabilities in `client.analyze()`.
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
python scripts/prepare_release.py --version 4.0.8 --test-install --generate-checksums --write-release-summary --build-bundle
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
git tag -a v4.0.8 -m "FamilyOS UltraBERT v4.0.8"
git push origin v4.0.8
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
