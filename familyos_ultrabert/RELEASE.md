# FamilyOS UltraBERT Release Guide

## Release model

FamilyOS UltraBERT releases are lightweight package releases.

- The wheel and source distribution contain Python code only.
- Runtime weights are downloaded from Hugging Face.
- Current encoder source of truth: `Pkansagra/ultrabert-weights` at `encoder/v2/fp32/`.

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
python scripts/prepare_release.py --version 4.0.2 --test-install --generate-checksums --write-release-summary --build-bundle
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
git tag -a v4.0.2 -m "FamilyOS UltraBERT v4.0.2"
git push origin v4.0.2
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
pip install familyos_ultrabert-4.0.2-py3-none-any.whl
```

Install with PyTorch support:

```bash
pip install familyos_ultrabert-4.0.2-py3-none-any.whl torch
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
