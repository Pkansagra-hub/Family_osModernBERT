# FamilyOS UltraBERT v3.0 - Complete Modernization Plan

## Executive Summary

**Goal:** Transform familyos_ultrabert from a 1.6GB encoder-only package to a 10MB edge-ready package with:
- GPT-2 decoder for counterfactual generation (R5 dreaming)
- Lazy loading (decoder loads only when needed)
- ONNX export with INT8 quantization
- Multi-backend support: AMD NPU → NVIDIA CUDA → CPU fallback
- HuggingFace Hub weight distribution

**Timeline:** 4 Milestones over 3-4 weeks
**Target Version:** v3.0.0

---

# ═══════════════════════════════════════════════════════════════════════════
# MILESTONE 1: ONNX Export Infrastructure (Week 1)
# ═══════════════════════════════════════════════════════════════════════════

**Goal:** Export GPT-2 decoder to ONNX with quantization support
**Duration:** 3-4 days
**Dependencies:** Trained decoder checkpoint (v3)

---

## M1-Epic 1: Decoder ONNX Export

**Priority:** 🔥 Critical
**Effort:** 3 days
**Status:** 🔴 Not Started

### Issue 1.1: Copy Latest Model Files from Training Codebase

**Type:** Task
**Priority:** High
**Effort:** 2 hours
**Dependencies:** None

**Description:**
Update `familyos_ultrabert/models/` with latest architecture including GPT-2 decoder support.

**Acceptance Criteria:**

- [ ] Copy all model files from `src/modeling_studio/models/` to `familyos_ultrabert/models/`
- [ ] Files to copy:
  - `modernbert_multitask.py` (updated with GPT2DecoderHead)
  - `decoder_gpt2.py` (NEW)
  - `decoder_gpt2_config.py` (NEW)
  - `heads.py` (updated)
  - `adapters.py`
  - `poolers.py`
  - `pair_encoder.py`
  - `attention.py`
- [ ] Update imports to use `familyos_ultrabert.models` namespace
- [ ] Verify no dependencies on `modeling_studio` package

**Commands:**

```bash
cd d:/Modeling_studio
cp src/modeling_studio/models/modernbert_multitask.py familyos_ultrabert/models/
cp src/modeling_studio/models/decoder_gpt2.py familyos_ultrabert/models/
cp src/modeling_studio/models/decoder_gpt2_config.py familyos_ultrabert/models/
cp src/modeling_studio/models/heads.py familyos_ultrabert/models/
cp src/modeling_studio/models/adapters.py familyos_ultrabert/models/
cp src/modeling_studio/models/poolers.py familyos_ultrabert/models/
cp src/modeling_studio/models/pair_encoder.py familyos_ultrabert/models/
cp src/modeling_studio/models/attention.py familyos_ultrabert/models/
```

**Testing:**

```python
from familyos_ultrabert.models import ModernBertMultiTaskModel, GPT2DecoderHead
# Should import without errors
```

---

### Issue 1.2: Create Weight Downloader Module

**Type:** Feature
**Priority:** High
**Effort:** 4 hours
**Dependencies:** Issue 1.1

**Description:**
Create automatic weight downloader using HuggingFace Hub to fetch model weights on first use.

**Acceptance Criteria:**

- [ ] Create `familyos_ultrabert/weights_manager.py`
- [ ] Implement `download_encoder_weights(version="v1")` function
- [ ] Implement `download_decoder_weights(version="v3")` function
- [ ] Cache weights in `~/.cache/familyos_ultrabert/`
- [ ] Show download progress bar
- [ ] Handle resume for interrupted downloads
- [ ] Add checksum verification

**Implementation:**

```python
# familyos_ultrabert/weights_manager.py
from huggingface_hub import hf_hub_download
from pathlib import Path
import logging

HF_REPO = "Pkansagra/ultrabert-weights"
CACHE_DIR = Path.home() / ".cache" / "familyos_ultrabert"

def download_encoder_weights(version: str = "v1") -> Path:
    """Download encoder weights from HuggingFace Hub."""
    files = ["model.safetensors", "config.json", "capabilities.json"]

    for file in files:
        hf_hub_download(
            repo_id=HF_REPO,
            filename=f"encoder/{version}/{file}",
            cache_dir=CACHE_DIR,
            resume_download=True,
        )

    return CACHE_DIR / f"encoder/{version}"

def download_decoder_weights(version: str = "v3") -> Path:
    """Download decoder weights from HuggingFace Hub."""
    # Similar implementation
    pass
```

**Testing:**

```python
from familyos_ultrabert.weights_manager import download_encoder_weights
path = download_encoder_weights("v1")
assert path.exists()
assert (path / "model.safetensors").exists()
```

---

### Issue 1.3: Update Model Loader to Use Auto-Download

**Type:** Enhancement
**Priority:** High
**Effort:** 3 hours
**Dependencies:** Issue 1.2

**Description:**
Update `familyos_ultrabert/model.py` to automatically download weights instead of using bundled weights.

**Acceptance Criteria:**

- [ ] Remove hardcoded weight paths
- [ ] Call `download_encoder_weights()` on first load
- [ ] Call `download_decoder_weights()` on first load
- [ ] Support version selection (encoder_version, decoder_version)
- [ ] Show clear loading messages to user
- [ ] Handle offline mode gracefully

**Implementation:**

```python
# familyos_ultrabert/model.py
from familyos_ultrabert.weights_manager import (
    download_encoder_weights,
    download_decoder_weights,
)

class UltraBERT:
    @classmethod
    def load(
        cls,
        encoder_version: str = "v1",
        decoder_version: str = "v3",
        backend: Literal["pytorch", "onnx", "auto"] = "auto",
        device: str = "auto",
    ):
        # Download weights on first use
        print(f"Loading encoder weights (version={encoder_version})...")
        encoder_path = download_encoder_weights(encoder_version)

        print(f"Loading decoder weights (version={decoder_version})...")
        decoder_path = download_decoder_weights(decoder_version)

        # Load model
        model = ModernBertMultiTaskModel.load_checkpoint(
            encoder_path,
            device=device,
        )

        return cls(model, backend)
```

**Testing:**

```python
from familyos_ultrabert import UltraBERT
model = UltraBERT.load()  # Should download weights on first call
result = model.analyze("Hello world", capabilities=["sentiment"])
assert "sentiment" in result.capabilities
```

---

### Issue 1.4: Add Decoder Inference API

**Type:** Feature
**Priority:** High
**Effort:** 4 hours
**Dependencies:** Issue 1.1, Issue 1.3

**Description:**
Add support for counterfactual generation (decoder) in the Client API.

**Acceptance Criteria:**

- [ ] Add `generate_counterfactual(text: str)` method to Client
- [ ] Support `"counterfactual"` capability in `analyze()`
- [ ] Add convenience method `suggest_alternative(text: str)`
- [ ] Update ClientResult to include counterfactual output
- [ ] Add docstrings and examples

**Implementation:**

```python
# familyos_ultrabert/client.py
class Client:
    def generate_counterfactual(self, text: str) -> str:
        """Generate counterfactual suggestion."""
        return self.model.generate(text, capability="counterfactual")

    def suggest_alternative(self, text: str) -> str:
        """Convenience wrapper for counterfactual generation."""
        return self.generate_counterfactual(text)

    def analyze(self, text: str, capabilities: List[str] = ["sentiment"]):
        # Support "counterfactual" in capabilities list
        if "counterfactual" in capabilities:
            result["counterfactual"] = self.generate_counterfactual(text)
        return ClientResult(**result)
```

**Testing:**

```python
from familyos_ultrabert import Client
client = Client()
suggestion = client.suggest_alternative("I feel overwhelmed")
assert len(suggestion) > 0
assert "If you had" in suggestion
```

---

### Issue 1.5: Update pyproject.toml - Remove Weight Bundling

**Type:** Configuration
**Priority:** High
**Effort:** 1 hour
**Dependencies:** Issue 1.2

**Description:**
Update package configuration to exclude weights and add HuggingFace Hub dependency.

**Acceptance Criteria:**

- [ ] Remove `[tool.setuptools.package-data]` section
- [ ] Add `huggingface-hub>=0.20.0` to dependencies
- [ ] Bump version to `3.0.0` (breaking change)
- [ ] Update description to mention decoder support
- [ ] Update classifier to "Development Status :: 5 - Production/Stable"

**Changes:**

```toml
[project]
name = "familyos-ultrabert"
version = "3.0.0"  # BREAKING: Weights no longer bundled
description = "FamilyOS UltraBERT v3 - Multi-task NLP with GPT-2 decoder for counterfactual generation"

dependencies = [
    "numpy>=1.21.0",
    "transformers>=4.30.0",
    "tokenizers>=0.13.0",
    "huggingface-hub>=0.20.0",  # NEW - for weight downloading
]

# REMOVE THIS:
# [tool.setuptools.package-data]
# "familyos_ultrabert" = [
#     "weights/pytorch/*",
#     "weights/onnx/*",
# ]
```

---

### Issue 1.6: Delete Bundled Weights Directory

**Type:** Cleanup
**Priority:** High
**Effort:** 15 minutes
**Dependencies:** Issue 1.5

**Description:**
Remove the weights directory from the package to reduce wheel size.

**Acceptance Criteria:**

- [ ] Delete `familyos_ultrabert/weights/` directory
- [ ] Update `.gitignore` to exclude `weights/`
- [ ] Clean build artifacts

**Commands:**

```bash
cd d:/Modeling_studio/familyos_ultrabert
rm -rf weights/
rm -rf dist/ build/ *.egg-info/
echo "weights/" >> .gitignore
```

**Verification:**

```bash
# Verify weights directory is gone
ls familyos_ultrabert/
# Should NOT show weights/
```

---

## Epic 2: Upload Weights to HuggingFace Hub

**Priority:** 🔥 Critical
**Effort:** 1 day
**Status:** 🔴 Not Started

### Issue 2.1: Setup HuggingFace Repository

**Type:** Setup
**Priority:** High
**Effort:** 30 minutes
**Dependencies:** None

**Description:**
Create private HuggingFace repository for model weights.

**Acceptance Criteria:**

- [ ] Login to HuggingFace CLI: `huggingface-cli login`
- [ ] Create repo: `Pkansagra/ultrabert-weights` (private)
- [ ] Add repository description and README
- [ ] Setup repository structure:

  ```
  Pkansagra/ultrabert-weights/
  ├── encoder/
  │   └── v1/
  └── decoder/
      ├── v1/
      ├── v2/
      └── v3/
  ```

**Commands:**

```bash
huggingface-cli login
# Will run create_repo via upload script
```

---

### Issue 2.2: Upload Encoder Weights

**Type:** Task
**Priority:** High
**Effort:** 1 hour
**Dependencies:** Issue 2.1

**Description:**
Upload encoder weights (ModernBERT base + 12 task heads) to HuggingFace Hub.

**Acceptance Criteria:**

- [ ] Run upload script for encoder
- [ ] Verify files uploaded:
  - `model.safetensors` (~500MB)
  - `config.json`
  - `capabilities.json`
- [ ] Test download with `huggingface-hub` library
- [ ] Verify checksum integrity

**Commands:**

```bash
cd d:/Modeling_studio
python export_utility/upload_weights_to_hf.py --component encoder
```

**Verification:**

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="Pkansagra/ultrabert-weights",
    filename="encoder/v1/model.safetensors"
)
print(f"Downloaded to: {path}")
```

---

### Issue 2.3: Upload Decoder Weights (v1, v2, v3)

**Type:** Task
**Priority:** High
**Effort:** 2 hours
**Dependencies:** Issue 2.1

**Description:**
Upload all decoder checkpoints to HuggingFace Hub for version comparison.

**Acceptance Criteria:**

- [ ] Upload v1 (original baseline)
- [ ] Upload v2 (failed training with 3e-5 LR)
- [ ] Upload v3 checkpoint-2000 (best - +13% coherence)
- [ ] Add version tags and descriptions
- [ ] Document which version is recommended

**Commands:**

```bash
# Upload v1
python export_utility/upload_weights_to_hf.py --component decoder \
    --decoder-path outputs/ultrabert-gen-decoder-v1 \
    --version v1

# Upload v2 (for comparison)
python export_utility/upload_weights_to_hf.py --component decoder \
    --decoder-path outputs/ultrabert-gen-decoder-v2 \
    --version v2

# Upload v3 (RECOMMENDED)
python export_utility/upload_weights_to_hf.py --component decoder \
    --decoder-path outputs/ultrabert-gen-decoder-v3 \
    --version v3
```

**Repository Structure:**

```
Pkansagra/ultrabert-weights/
├── README.md
├── encoder/
│   └── v1/
│       ├── model.safetensors
│       ├── config.json
│       └── capabilities.json
└── decoder/
    ├── v1/  # Original (baseline)
    ├── v2/  # Regression (-15% coherence)
    └── v3/  # BEST (+13% coherence, use this!)
        ├── model.safetensors
        ├── config.json
        └── capabilities.json
```

---

### Issue 2.4: Create HuggingFace README with Model Card

**Type:** Documentation
**Priority:** Medium
**Effort:** 2 hours
**Dependencies:** Issue 2.2, Issue 2.3

**Description:**
Create comprehensive model card documenting architecture, performance, and usage.

**Acceptance Criteria:**

- [ ] Model overview and architecture
- [ ] Performance benchmarks (from evaluation results)
- [ ] Version comparison (v1 vs v2 vs v3)
- [ ] Usage examples
- [ ] License information

**Content:**

```markdown
# FamilyOS UltraBERT Weights

Private model weights for FamilyOS UltraBERT v3 multi-task NLP model.

## Architecture

- **Encoder**: ModernBERT-base (155M params) + 12 task heads
- **Decoder**: GPT-2 Medium (355M params) for counterfactual generation
- **Total**: 510M parameters

## Versions

### Encoder
- **v1**: Production encoder with 12 capabilities

### Decoder
- **v1**: Baseline counterfactual decoder
- **v2**: ❌ Regression (-15% coherence) - NOT RECOMMENDED
- **v3**: ✅ BEST (+13% coherence vs v1) - USE THIS

## Performance (v3 Decoder)

| Metric | v1 (Baseline) | v3 (Best) | Change |
|--------|---------------|-----------|--------|
| Coherence | 0.055 | 0.063 | **+13.4%** ⬆️ |
| health_mental | 0.056 | 0.066 | **+18.8%** ⬆️ |
| health_nutrition | 0.051 | 0.068 | **+33.3%** ⬆️ |
| Subdomains improved | - | 9/10 | **90%** ⬆️ |

## Usage

See `familyos-ultrabert` package on PyPI.

## License

Proprietary - All Rights Reserved
```

---

## Epic 3: Build and Test New Package

**Priority:** 🔥 Critical
**Effort:** 2 days
**Status:** 🔴 Not Started

### Issue 3.1: Build Lightweight Wheel

**Type:** Build
**Priority:** High
**Effort:** 1 hour
**Dependencies:** Epic 1 (all issues)

**Description:**
Build new v3.0.0 wheel without bundled weights.

**Acceptance Criteria:**

- [ ] Clean build environment
- [ ] Build wheel with `python -m build`
- [ ] Verify wheel size < 20MB (target: ~10MB)
- [ ] Verify no weights bundled inside wheel
- [ ] Build source distribution (sdist)

**Commands:**

```bash
cd d:/Modeling_studio/familyos_ultrabert
rm -rf dist/ build/ *.egg-info/
python -m build

# Check wheel size
ls -lh dist/familyos_ultrabert-3.0.0-py3-none-any.whl
# Should show ~10-15 MB (not 1.6 GB!)
```

**Verification:**

```bash
# Extract wheel and verify no weights
unzip -l dist/familyos_ultrabert-3.0.0-py3-none-any.whl | grep -i weights
# Should return nothing or just weights_manager.py
```

---

### Issue 3.2: Test Package in Clean Environment

**Type:** Testing
**Priority:** High
**Effort:** 2 hours
**Dependencies:** Issue 3.1, Epic 2 (weights uploaded)

**Description:**
Test package installation and weight downloading in isolated environment.

**Acceptance Criteria:**

- [ ] Create clean virtual environment
- [ ] Install wheel with `pip install dist/*.whl`
- [ ] Test auto-download on first use
- [ ] Verify weights cached in `~/.cache/`
- [ ] Test all capabilities including decoder
- [ ] Measure cold start time
- [ ] Test offline mode (with cached weights)

**Testing Script:**

```bash
# Create clean environment
python -m venv test_env
source test_env/bin/activate  # Windows: test_env\Scripts\activate

# Install package
pip install dist/familyos_ultrabert-3.0.0-py3-none-any.whl torch

# Test
python << EOF
from familyos_ultrabert import Client
import time

# First use - downloads weights
start = time.time()
client = Client()
print(f"Load time: {time.time() - start:.2f}s")

# Test encoder
result = client.analyze("I love my family", capabilities=["sentiment"])
print(f"Sentiment: {result.sentiment}")

# Test decoder
suggestion = client.suggest_alternative("I feel overwhelmed")
print(f"Suggestion: {suggestion[:100]}...")

# Verify cache
import pathlib
cache = pathlib.Path.home() / ".cache" / "familyos_ultrabert"
print(f"Cache dir exists: {cache.exists()}")
print(f"Cache size: {sum(f.stat().st_size for f in cache.rglob('*')) / 1024**3:.2f} GB")
EOF

deactivate
```

---

### Issue 3.3: Performance Benchmarks (v3.0 vs v2.2)

**Type:** Testing
**Priority:** Medium
**Effort:** 2 hours
**Dependencies:** Issue 3.2

**Description:**
Compare performance between bundled weights (v2.2) and auto-download (v3.0).

**Acceptance Criteria:**

- [ ] Benchmark cold start time (first import)
- [ ] Benchmark warm start time (cached weights)
- [ ] Measure memory usage
- [ ] Measure inference latency (should be same)
- [ ] Document results

**Expected Results:**

| Metric | v2.2.1 (Bundled) | v3.0.0 (Download) | Change |
|--------|------------------|-------------------|--------|
| Wheel size | 1.59 GB | ~10 MB | **160x smaller** |
| Install time | ~5 min | ~10 sec | **30x faster** |
| First load (download) | N/A | ~2-3 min | One-time only |
| Warm load (cached) | ~3 sec | ~3 sec | Same |
| Inference latency | 7-12 ms | 7-12 ms | Same |

---

### Issue 3.4: Update Documentation and Migration Guide

**Type:** Documentation
**Priority:** High
**Effort:** 3 hours
**Dependencies:** Issue 3.3

**Description:**
Update README, API docs, and create v3.0 migration guide.

**Acceptance Criteria:**

- [ ] Update README.md with v3.0 features
- [ ] Document decoder/counterfactual API
- [ ] Create MIGRATION_v2_to_v3.md guide
- [ ] Update examples with decoder usage
- [ ] Document offline mode

**Files to Update:**

- `familyos_ultrabert/README.md`
- `familyos_ultrabert/API.md`
- `familyos_ultrabert/examples/basic_usage.py`
- `familyos_ultrabert/MIGRATION_v2_to_v3.md` (NEW)

**Migration Guide Outline:**

```markdown
# Migrating from v2.x to v3.0

## Breaking Changes

1. **Weights no longer bundled** - Downloaded automatically on first use
2. **Internet required on first use** - Subsequent uses work offline
3. **New decoder capability** - `counterfactual` generation available

## What's New

- GPT-2 decoder for counterfactual suggestions
- 160x smaller package (10MB vs 1.6GB)
- Version-aware weight management
- Improved weak domain performance (+13-33%)

## Upgrade Steps

```bash
# Uninstall old version
pip uninstall familyos-ultrabert

# Install v3.0
pip install familyos-ultrabert==3.0.0

# First use will download weights (~1.2GB cached)
python -c "from familyos_ultrabert import Client; client = Client()"
```

## API Changes

### Decoder Support (NEW)

```python
from familyos_ultrabert import Client
client = Client()

# Generate counterfactual suggestion
suggestion = client.suggest_alternative("I feel overwhelmed")
# "If you had scheduled 15 minutes of 'me time' each day..."
```

## Offline Mode

Once weights are cached, the package works offline:

```python
# Weights cached in ~/.cache/familyos_ultrabert/
# ~1.2GB total (encoder + decoder)
```

```

---

## Epic 4: Publish and Cleanup
**Priority:** 🔥 Critical
**Effort:** 1 day
**Status:** 🔴 Not Started

### Issue 4.1: Delete Old GitHub Releases (v2.0.0 - v2.2.1)
**Type:** Cleanup
**Priority:** High
**Effort:** 30 minutes
**Dependencies:** Issue 3.4 (v3.0 ready)

**Description:**
Remove old releases with 1.6GB wheels from GitHub to save storage and avoid confusion.

**Acceptance Criteria:**
- [ ] Delete v2.2.1 release (1.59 GB)
- [ ] Delete v2.2.0 release (1.59 GB)
- [ ] Delete v2.1.0 release
- [ ] Delete v2.0.3 release
- [ ] Delete v2.0.2 release
- [ ] Delete v2.0.1 release
- [ ] Delete v2.0.0 release
- [ ] Keep git tags for version history

**Commands:**
```bash
# Delete releases (keeps tags)
gh release delete v2.2.1 --yes
gh release delete v2.2.0 --yes
gh release delete v2.1.0 --yes
gh release delete v2.0.3 --yes
gh release delete v2.0.2 --yes
gh release delete v2.0.1 --yes
gh release delete v2.0.0 --yes

# Verify deletions
gh release list
# Should only show v3.0.0 (after next issue)
```

**Space Saved:** ~11 GB (7 releases × ~1.6 GB each)

---

### Issue 4.2: Create v3.0.0 GitHub Release

**Type:** Release
**Priority:** High
**Effort:** 1 hour
**Dependencies:** Issue 3.4, Issue 4.1

**Description:**
Create new v3.0.0 release with lightweight wheel and comprehensive release notes.

**Acceptance Criteria:**

- [ ] Tag commit: `git tag v3.0.0`
- [ ] Push tag: `git push origin v3.0.0`
- [ ] Create GitHub release
- [ ] Upload wheel (~10MB)
- [ ] Upload source tarball
- [ ] Write detailed release notes
- [ ] Mark as "Latest Release"

**Release Notes Template:**

```markdown
# FamilyOS UltraBERT v3.0.0 - Major Architecture Update

## 🚀 Highlights

- **160x Smaller Package**: 10 MB wheel (down from 1.6 GB!)
- **GPT-2 Decoder**: Counterfactual generation for better suggestions
- **+13% Performance**: Improved coherence on weak domains
- **HuggingFace Integration**: Automatic weight downloading

## ⚠️ Breaking Changes

1. **Weights no longer bundled** - Downloaded automatically on first use (requires internet)
2. **Python 3.9+** required (was 3.8+)
3. **New dependency**: `huggingface-hub>=0.20.0`

## ✨ What's New

### Decoder Architecture
- Added GPT-2 Medium decoder (355M params) for counterfactual generation
- New API: `client.suggest_alternative(text)`
- Supports 13th capability: `"counterfactual"`

### Performance Improvements
| Domain | v1 Baseline | v3 | Improvement |
|--------|-------------|----|-----------|
| health_mental | 0.056 | 0.066 | **+18.8%** |
| health_nutrition | 0.051 | 0.068 | **+33.3%** |
| Overall coherence | 0.055 | 0.063 | **+13.4%** |

### Package Modernization
- Weights downloaded from HuggingFace Hub
- Cached in `~/.cache/familyos_ultrabert/`
- Version-aware weight management (encoder v1, decoder v3)
- 30x faster install time (~10 sec vs 5 min)

## 📦 Installation

```bash
pip install familyos-ultrabert==3.0.0
```

First use downloads weights (~1.2 GB cached):

```python
from familyos_ultrabert import Client
client = Client()  # Downloads weights on first call
```

## 🔧 Usage

### New Decoder API

```python
from familyos_ultrabert import Client

client = Client()

# Generate counterfactual suggestion
text = "I feel overwhelmed with work and family"
suggestion = client.suggest_alternative(text)
print(suggestion)
# "If you had scheduled 15 minutes of daily 'me time' to decompress..."

# Include in multi-capability analysis
result = client.analyze(text, capabilities=["sentiment", "emotions", "counterfactual"])
print(result.sentiment)        # "negative"
print(result.emotions)         # ["stress", "overwhelm"]
print(result.counterfactual)   # "If you had..."
```

### Migration from v2.x

See [MIGRATION_v2_to_v3.md](MIGRATION_v2_to_v3.md) for full guide.

**Quick migration:**

```python
# v2.x code still works!
from familyos_ultrabert import Client
client = Client()
result = client.analyze("Hello world")

# New v3.0 features:
suggestion = client.suggest_alternative("I feel stressed")
```

## 📊 Benchmarks

| Metric | v2.2.1 | v3.0.0 | Improvement |
|--------|--------|--------|-------------|
| Wheel size | 1.59 GB | 10 MB | **160x smaller** |
| Install time | ~5 min | ~10 sec | **30x faster** |
| Inference latency | 7-12 ms | 7-12 ms | Same |
| Coherence score | 0.055 | 0.063 | **+13.4%** |

## 🔗 Links

- [HuggingFace Model Weights](https://huggingface.co/Pkansagra/ultrabert-weights)
- [API Documentation](API.md)
- [Migration Guide](MIGRATION_v2_to_v3.md)
- [Changelog](RELEASE_NOTES.md)

## 📄 License

Proprietary - All Rights Reserved

```

**Commands:**
```bash
# Tag and push
git tag -a v3.0.0 -m "v3.0.0 - Decoder architecture + weight separation"
git push origin v3.0.0

# Create release
gh release create v3.0.0 \
  dist/familyos_ultrabert-3.0.0-py3-none-any.whl \
  dist/familyos_ultrabert-3.0.0.tar.gz \
  --title "FamilyOS UltraBERT v3.0.0 - Major Architecture Update" \
  --notes-file RELEASE_NOTES_v3.0.0.md
```

---

### Issue 4.3: Publish to PyPI

**Type:** Release
**Priority:** High
**Effort:** 30 minutes
**Dependencies:** Issue 4.2

**Description:**
Publish v3.0.0 to PyPI for public distribution.

**Acceptance Criteria:**

- [ ] Verify PyPI credentials
- [ ] Upload wheel to PyPI
- [ ] Upload source distribution
- [ ] Verify package appears on PyPI
- [ ] Test install from PyPI

**Commands:**

```bash
cd d:/Modeling_studio/familyos_ultrabert

# Test upload to TestPyPI first
twine upload --repository testpypi dist/*

# Verify on TestPyPI
pip install --index-url https://test.pypi.org/simple/ familyos-ultrabert==3.0.0

# Production upload
twine upload dist/*

# Verify on PyPI
pip install familyos-ultrabert==3.0.0
```

---

### Issue 4.4: Update Documentation Links

**Type:** Documentation
**Priority:** Medium
**Effort:** 1 hour
**Dependencies:** Issue 4.3

**Description:**
Update all documentation to point to v3.0.0 and HuggingFace weights.

**Acceptance Criteria:**

- [ ] Update main README.md
- [ ] Update docs/ folder
- [ ] Update GitHub repo description
- [ ] Add badge for PyPI version
- [ ] Add badge for wheel size

**Files to Update:**

- `README.md` (root)
- `familyos_ultrabert/README.md`
- `docs/` (all files)
- GitHub repo settings

**Badges to Add:**

```markdown
[![PyPI version](https://badge.fury.io/py/familyos-ultrabert.svg)](https://pypi.org/project/familyos-ultrabert/)
[![Wheel Size](https://img.shields.io/badge/wheel-10MB-brightgreen)](https://pypi.org/project/familyos-ultrabert/#files)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Weights-yellow)](https://huggingface.co/Pkansagra/ultrabert-weights)
```

---

## Epic 5: Monitoring and Rollback Plan

**Priority:** 🟡 Medium
**Effort:** 1 day
**Status:** 🔴 Not Started

### Issue 5.1: Create Rollback Documentation

**Type:** Documentation
**Priority:** Medium
**Effort:** 1 hour
**Dependencies:** Issue 4.3

**Description:**
Document rollback procedure in case v3.0 has issues.

**Acceptance Criteria:**

- [ ] Document how to downgrade to v2.2.1
- [ ] Keep v2.2.1 wheels accessible
- [ ] Create emergency contact list
- [ ] Document known issues

**Rollback Guide:**

```markdown
# v3.0 Rollback Procedure

If you encounter issues with v3.0, downgrade to v2.2.1:

```bash
pip uninstall familyos-ultrabert
pip install familyos-ultrabert==2.2.1
```

**Known v2.2.1 issues:**

- Large wheel size (1.6 GB)
- No decoder support
- Requires bundled weights

**When to rollback:**

- Weight download failures
- Network connectivity issues
- Compatibility problems

**Support:** <issues@familyos.dev>

```

---

### Issue 5.2: Setup Usage Analytics (Optional)
**Type:** Monitoring
**Priority:** Low
**Effort:** 2 hours
**Dependencies:** Issue 4.3

**Description:**
Add anonymous telemetry to track v3.0 adoption and errors.

**Acceptance Criteria:**
- [ ] Track package version usage
- [ ] Track weight download success/failure
- [ ] Track decoder API usage
- [ ] Respect do-not-track settings
- [ ] Document in privacy policy

---

## Summary

**Total Effort:** ~10-12 days
**Target Release:** v3.0.0
**Expected Impact:**
- 160x smaller package (10 MB vs 1.6 GB)
- 30x faster install time
- +13% performance improvement
- Modern architecture with decoder support
- Save ~11 GB GitHub storage

**Key Milestones:**
1. Week 1: Epic 1-2 (Code update + HF upload)
2. Week 2: Epic 3-4 (Build, test, publish)

**Success Metrics:**
- [ ] Wheel size < 20 MB
- [ ] Auto-download works in 99% of cases
- [ ] Inference performance maintained
- [ ] No user-facing breaking changes (API compatible)
