# UltraBERT v3 Release Architecture Strategy

> **Status**: Planning
> **Author**: Modeling Studio
> **Date**: December 16, 2025
> **Context**: Private HuggingFace repo for weights, lightweight pip package

---

## 1. Current vs. Proposed Architecture

### Current Release (v2.2.1) - Bundled Weights

```
familyos_ultrabert-2.2.1-py3-none-any.whl    # ~2.3 GB (HUGE!)
├── weights/
│   ├── pytorch/model.safetensors            # 592 MB
│   └── onnx/*.onnx                          # 1.7 GB (12 files)
├── client.py
├── model.py
└── labels.py
```

**Problems:**
- 2.3 GB wheel is slow to build/distribute
- Adding decoder = 2.5+ GB wheel
- Every update requires full re-download

### Proposed Release (v3.0) - HuggingFace Private Weights

```
familyos_ultrabert-3.0.0-py3-none-any.whl    # ~100 KB (TINY!)
├── client.py
├── decoder_client.py                         # NEW
├── weights_manager.py                        # NEW: HF download logic
├── model.py
└── labels.py

HuggingFace Private Repo: Pkansagra-hub/ultrabert-weights (PRIVATE)
├── encoder/
│   ├── model.safetensors                    # 592 MB
│   ├── config.json
│   └── capabilities.json
├── decoder/                                  # NEW (after Stage C)
│   ├── decoder.safetensors                  # 240 MB
│   └── config.json
└── onnx/                                     # Optional
    └── *.onnx
```

---

## 2. HuggingFace Private Repo Setup

### Step 1: Create Private Repo

```bash
# Install huggingface_hub
pip install huggingface_hub

# Login (one-time)
huggingface-cli login
# Enter your HF token from https://huggingface.co/settings/tokens
```

```python
from huggingface_hub import create_repo

# Create private repo
create_repo(
    repo_id="Pkansagra-hub/ultrabert-weights",
    private=True,  # PRIVATE - only you can access
    repo_type="model"
)
```

### Step 2: Upload Weights

```python
from huggingface_hub import HfApi

api = HfApi()

# Upload encoder weights
api.upload_folder(
    folder_path="familyos_ultrabert/weights/pytorch",
    repo_id="Pkansagra-hub/ultrabert-weights",
    path_in_repo="encoder",
)

# Upload ONNX weights (optional)
api.upload_folder(
    folder_path="familyos_ultrabert/weights/onnx",
    repo_id="Pkansagra-hub/ultrabert-weights",
    path_in_repo="onnx",
)

# Upload decoder (after training)
api.upload_folder(
    folder_path="outputs/ultrabert-gen-decoder-v1",
    repo_id="Pkansagra-hub/ultrabert-weights",
    path_in_repo="decoder",
)
```

### Step 3: Access Control

| Access Type | How to Use |
|-------------|------------|
| **Your machine** | `huggingface-cli login` (one-time) |
| **Colab** | `notebook_login()` or HF_TOKEN secret |
| **CI/CD** | `HF_TOKEN` environment variable |
| **New team member** | Add as collaborator on HF repo |

---

## 3. New Package Design

### weights_manager.py (NEW)

```python
"""
Weight manager for FamilyOS UltraBERT.
Downloads weights from private HuggingFace repo on first use.
"""
import os
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

# Private HuggingFace repo
HF_REPO_ID = "Pkansagra-hub/ultrabert-weights"

# Local cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "familyos_ultrabert"


def get_encoder_path(cache_dir: Path = None) -> Path:
    """
    Get path to encoder weights, downloading if needed.

    First use: Downloads ~600 MB from HuggingFace (one-time)
    Subsequent: Uses cached weights (instant)
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR

    # Download encoder weights
    local_dir = snapshot_download(
        repo_id=HF_REPO_ID,
        allow_patterns=["encoder/*"],
        local_dir=cache_dir,
        local_dir_use_symlinks=False,
    )

    return Path(local_dir) / "encoder"


def get_decoder_path(cache_dir: Path = None) -> Path:
    """
    Get path to decoder weights, downloading if needed.

    First use: Downloads ~250 MB from HuggingFace
    Subsequent: Uses cached weights
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR

    local_dir = snapshot_download(
        repo_id=HF_REPO_ID,
        allow_patterns=["decoder/*"],
        local_dir=cache_dir,
        local_dir_use_symlinks=False,
    )

    return Path(local_dir) / "decoder"


def get_onnx_path(cache_dir: Path = None) -> Path:
    """Get path to ONNX weights, downloading if needed."""
    cache_dir = cache_dir or DEFAULT_CACHE_DIR

    local_dir = snapshot_download(
        repo_id=HF_REPO_ID,
        allow_patterns=["onnx/*"],
        local_dir=cache_dir,
        local_dir_use_symlinks=False,
    )

    return Path(local_dir) / "onnx"


def clear_cache():
    """Clear all cached weights."""
    import shutil
    if DEFAULT_CACHE_DIR.exists():
        shutil.rmtree(DEFAULT_CACHE_DIR)
        print(f"Cleared cache: {DEFAULT_CACHE_DIR}")
```

### Updated model.py

```python
# In model.py - modify load() method

from .weights_manager import get_encoder_path, get_onnx_path

class UltraBERT:
    @classmethod
    def load(
        cls,
        model_path: str = None,           # Custom path (optional)
        backend: str = "auto",
        device: str = "auto",
        **kwargs
    ):
        # If no custom path, download from HuggingFace
        if model_path is None:
            if backend == "onnx":
                model_path = str(get_onnx_path())
            else:
                model_path = str(get_encoder_path())

        # Rest of existing load logic...
```

---

## 4. User Experience

### First Time (Downloads Weights)

```python
from familyos_ultrabert import Client

# First time: Downloads encoder weights (~600 MB)
# Shows progress bar, takes 1-2 minutes on good connection
client = Client()

# Now works instantly
result = client.analyze("Mom picked up the kids!")
```

Output:
```
Downloading encoder weights from HuggingFace...
encoder/model.safetensors: 100%|███████| 592M/592M [01:23<00:00, 7.1MB/s]
encoder/config.json: 100%|████████████████| 1.2k/1.2k [00:00<00:00]
Weights cached at: ~/.cache/familyos_ultrabert/encoder
[UltraBERT] Model loaded in 1523ms (backend: pytorch)
[UltraBERT] Warmup complete - ready for fast inference!
```

### Subsequent Use (Instant)

```python
from familyos_ultrabert import Client

client = Client()  # Uses cached weights, no download
result = client.analyze("text")
```

### Decoder On-Demand

```python
from familyos_ultrabert import DecoderClient

# First time: Downloads decoder weights (~250 MB)
with DecoderClient() as decoder:
    result = decoder.generate("text")
# Unloaded, memory freed
```

---

## 5. Backward Compatibility

### v2.x Behavior (Still Works!)

```python
# Old way - explicit model_path still works
client = Client(model_path="D:/my/custom/weights")

# Auto-detect bundled weights (if present) also works
model = UltraBERT.load()  # Checks local first, then HF
```

### Priority Order for Weight Discovery

1. **Explicit `model_path`** argument (user override)
2. **Environment variable** `ULTRABERT_WEIGHTS_DIR`
3. **Bundled weights** (in package, for legacy wheels)
4. **HuggingFace cache** (downloaded from private repo)
5. **Download from HuggingFace** (first-time setup)

---

## 6. Wheel Size Comparison

| Version | Contents | Size |
|---------|----------|------|
| v2.2.1 (current) | Code + PyTorch + ONNX | ~2.3 GB |
| v3.0 (proposed) | Code only | ~100 KB |
| v3.0 + encoder (cached) | After first download | +592 MB |
| v3.0 + decoder (cached) | After first decoder use | +240 MB |
| v3.0 full (cached) | All weights | ~1.0 GB |

**Benefits:**
- Instant pip install (~100 KB)
- Download only what you need
- Easy updates (just update HF repo)

---

## 7. HuggingFace Repo Structure

```
Pkansagra-hub/ultrabert-weights (PRIVATE)
│
├── README.md                 # Model card
├── encoder/
│   ├── model.safetensors     # 592 MB - Main encoder
│   ├── config.json           # Model config
│   └── capabilities.json     # Head configurations
│
├── decoder/                  # After Stage C training
│   ├── decoder.safetensors   # ~240 MB - MoE decoder
│   └── config.json           # Decoder config
│
├── onnx/                     # Optional ONNX models
│   ├── sentiment_int8.onnx
│   ├── emotions_int8.onnx
│   └── ... (12 files)
│
└── edge/                     # Future edge formats
    ├── tflite/
    └── coreml/
```

---

## 8. Decoder Client Design

```python
class DecoderClient:
    """
    On-demand decoder for counterfactual generation.
    Downloads weights from HuggingFace on first use.
    """

    def __init__(
        self,
        model_path: str = None,      # Custom path (optional)
        device: str = "auto",
        lazy_load: bool = True,
    ):
        self._model_path = model_path
        self._device = device
        self._decoder = None

        if not lazy_load:
            self.load()

    def load(self):
        """Load decoder into memory."""
        if self._model_path is None:
            # Download from HuggingFace if needed
            from .weights_manager import get_decoder_path
            self._model_path = str(get_decoder_path())

        self._decoder = load_decoder(self._model_path, self._device)

    def unload(self):
        """Free memory."""
        import torch
        del self._decoder
        self._decoder = None
        torch.cuda.empty_cache()

    def generate(self, text: str, **kwargs) -> str:
        """Generate counterfactual."""
        if self._decoder is None:
            self.load()
        return self._decoder.generate(text, **kwargs)

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *args):
        self.unload()
```

---

## 9. Implementation Plan

### Phase 1: Setup HuggingFace (1 hour)

| Task | Command |
|------|---------|
| Create HF account | https://huggingface.co/join |
| Get access token | https://huggingface.co/settings/tokens |
| Create private repo | `create_repo("Pkansagra-hub/ultrabert-weights", private=True)` |
| Upload encoder | `api.upload_folder(...)` |
| Upload ONNX | `api.upload_folder(...)` |

### Phase 2: Update Package (2 hours)

| File | Change |
|------|--------|
| `weights_manager.py` | NEW: Download logic |
| `model.py` | Use weights_manager |
| `pyproject.toml` | Add `huggingface_hub` dependency |
| `__init__.py` | Export DecoderClient |

### Phase 3: Build & Test (1 hour)

```bash
# Build lightweight wheel
cd familyos_ultrabert
python -m build

# Test installation
pip install dist/familyos_ultrabert-3.0.0-py3-none-any.whl

# Test download
python -c "from familyos_ultrabert import Client; Client()"
```

### Phase 4: Add Decoder (After Training)

```bash
# Upload decoder weights
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='outputs/ultrabert-gen-decoder-v1',
    repo_id='Pkansagra-hub/ultrabert-weights',
    path_in_repo='decoder',
)
"
```

---

## 10. File Changes Summary

### New Files

| File | Purpose |
|------|---------|
| `familyos_ultrabert/weights_manager.py` | HuggingFace download logic |
| `familyos_ultrabert/decoder_client.py` | Decoder on-demand loader |
| `scripts/upload_weights_to_hf.py` | One-time upload script |

### Modified Files

| File | Change |
|------|--------|
| `familyos_ultrabert/model.py` | Use weights_manager for path resolution |
| `familyos_ultrabert/__init__.py` | Export DecoderClient |
| `familyos_ultrabert/pyproject.toml` | Add huggingface_hub dependency |
| `familyos_ultrabert/MANIFEST.in` | Exclude weights/ from wheel |

### Manifest Changes (Exclude Weights)

```ini
# MANIFEST.in - UPDATED
include LICENSE
include README.md
include API.md

# Exclude weights from wheel (downloaded from HF instead)
prune weights
global-exclude *.safetensors
global-exclude *.onnx
```

---

## 11. Quick Reference

### Setup (One-Time)

```bash
# Login to HuggingFace
huggingface-cli login

# Install package
pip install familyos_ultrabert
```

### Daily Usage

```python
from familyos_ultrabert import Client

client = Client()  # Downloads weights on first use
result = client.analyze("Mom picked up the kids!")
print(result.sentiment)  # "very_positive"
```

### Nightly Decoder Job

```python
from familyos_ultrabert import DecoderClient

with DecoderClient() as decoder:  # Downloads decoder on first use
    result = decoder.generate("I yelled at my daughter...")
    print(result.counterfactual)
# Memory freed automatically
```

### Clear Cache

```python
from familyos_ultrabert.weights_manager import clear_cache
clear_cache()  # Re-downloads on next use
```

---

## 12. Memory Footprint

| Configuration | GPU RAM |
|---------------|---------|
| Encoder only | ~1.5 GB |
| Encoder + Decoder | ~3.0 GB |
| Decoder only (nightly) | ~1.5 GB |

---

## 13. Decision Summary

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Weight hosting** | HuggingFace private repo | Free, reliable, version controlled |
| **Wheel size** | Code only (~100 KB) | Fast install, easy updates |
| **First-use download** | Automatic with progress bar | Seamless UX |
| **Decoder loading** | On-demand with context manager | Memory efficient |
| **Backward compat** | model_path override still works | No breaking changes |

---

## 14. Next Steps

1. ✅ **Document strategy** (this document)
2. ⏳ **Create HuggingFace private repo**
3. ⏳ **Upload current weights**
4. ⏳ **Implement weights_manager.py**
5. ⏳ **Finish decoder training**
6. ⏳ **Upload decoder weights**
7. ⏳ **Build v3.0 wheel**
8. ⏳ **Test full flow**

---

*Document created: December 16, 2025*
*Last updated: December 16, 2025*
