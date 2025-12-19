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

## Current State Analysis

### Package Structure (`familyos_ultrabert/`)

```text
familyos_ultrabert/              # CURRENT STATE
├── __init__.py                  # Exports: Client, UltraBERT, ClientResult
├── model.py                     # UltraBERT.load() - backend selection
├── client.py                    # Client class with warmup
├── labels.py                    # Capability enum, label schemas
├── pytorch_inference.py         # PyTorch inference engine
├── onnx_inference.py            # ONNX inference engine (encoder only)
├── models/
│   ├── modernbert_multitask.py  # OLD: 12 heads, no decoder
│   ├── heads.py                 # Task-specific heads
│   ├── adapters.py              # Task adapters
│   ├── poolers.py               # Pooling strategies
│   └── pair_encoder.py          # Cross-attention for NLI
├── weights/                     # BUNDLED: 1.6GB (BAD!)
│   ├── pytorch/                 # PyTorch weights
│   └── onnx/                    # ONNX encoder models
├── data/                        # Label schemas
├── benchmarks/                  # Benchmark suite
└── examples/                    # Usage examples
```

### Training Codebase (`src/modeling_studio/models/`)

```text
src/modeling_studio/models/      # UP-TO-DATE
├── modernbert_multitask.py      # Updated with GPT2DecoderHead support
├── decoder_gpt2.py              # NEW: GPT-2 decoder implementation
├── decoder_gpt2_config.py       # NEW: Decoder configuration
├── heads.py                     # Updated task heads
├── adapters.py                  # Updated adapters
├── poolers.py                   # Updated poolers
├── pair_encoder.py              # Cross-attention
└── attention.py                 # Attention modules
```

### Gap Analysis

| Component | familyos_ultrabert | src/modeling_studio | Action |
|-----------|-------------------|---------------------|--------|
| Encoder | ✅ ModernBERT | ✅ ModernBERT | Keep |
| 12 Heads | ✅ Present | ✅ Present | Sync if updated |
| Decoder | ❌ Missing | ✅ GPT2DecoderHead | Copy |
| ONNX Decoder | ❌ Missing | ❌ Missing | Create |
| Lazy Loading | ❌ Missing | ❌ Missing | Create |
| NPU Support | ❌ Missing | ❌ Missing | Create |
| Weight Download | ❌ Missing | ❌ Missing | Create |
| INT8 Quantization | ❌ Missing | ❌ Missing | Create |

---

## Milestone 1: ONNX Export Infrastructure

**Goal:** Export GPT-2 decoder to ONNX with quantization support

**Duration:** 3-4 days

**Dependencies:** Trained decoder checkpoint (v3)

### M1-Epic 1: Decoder ONNX Export

**Priority:** 🔥 Critical

**Effort:** 2 days

#### Issue M1-1.1: Create `export_decoder_onnx.py`

**Type:** Feature

**File:** `export_utility/export_decoder_onnx.py`

**Description:**
Export GPT-2 decoder to ONNX format with split architecture for NPU efficiency.

**Split Architecture:**

```text
┌──────────────────────────────────────────────────────────────────┐
│ SPLIT EXPORT (Better for NPU scheduling)                        │
│                                                                  │
│ prefix_encoder.onnx          decoder_core.onnx                  │
│ ┌────────────────────┐       ┌────────────────────────────────┐ │
│ │ Input:             │       │ Input:                         │ │
│ │   encoder_hidden   │       │   input_embeds (batch, 1, 1024)│ │
│ │   (batch, seq, 768)│       │   past_key_values              │ │
│ │                    │       │   position_ids                 │ │
│ │ Output:            │       │                                │ │
│ │   prefix_embeds    │       │ Output:                        │ │
│ │   (batch, seq,1024)│       │   logits (batch, 1, vocab)     │ │
│ └────────────────────┘       │   new_past_key_values          │ │
│                              └────────────────────────────────┘ │
│                                                                  │
│ Runs ONCE per input          Runs N times (one per token)       │
└──────────────────────────────────────────────────────────────────┘
```

**Acceptance Criteria:**

- [ ] Export prefix_encoder.onnx (projection layer)
- [ ] Export decoder_core.onnx (GPT-2 transformer)
- [ ] Support dynamic batch size and sequence length
- [ ] Validate outputs match PyTorch within tolerance
- [ ] CLI: `python export_decoder_onnx.py --checkpoint outputs/ultrabert-gen-decoder-v3`

**Implementation Outline:**

```python
#!/usr/bin/env python3
"""
Export GPT-2 Decoder to ONNX - Split Architecture

Exports two ONNX models for efficient edge inference:
1. prefix_encoder.onnx - Projects encoder hidden states (runs once)
2. decoder_core.onnx - GPT-2 transformer (runs per token)

Usage:
    python export_decoder_onnx.py \
        --checkpoint outputs/ultrabert-gen-decoder-v3 \
        --output exports/decoder-onnx-v3 \
        --opset 17
"""

class PrefixEncoderWrapper(torch.nn.Module):
    """Wrapper for ONNX export of encoder projection."""

    def __init__(self, decoder):
        super().__init__()
        self.encoder_proj = decoder.encoder_proj
        self.adapter = decoder.adapter

    def forward(self, encoder_hidden_states):
        prefix_embeds = self.encoder_proj(encoder_hidden_states)
        if self.adapter is not None:
            prefix_embeds = self.adapter(prefix_embeds)
        return prefix_embeds


class DecoderCoreWrapper(torch.nn.Module):
    """Wrapper for ONNX export of GPT-2 core."""

    def __init__(self, decoder):
        super().__init__()
        self.gpt2 = decoder.gpt2
        self.wte = decoder.gpt2.transformer.wte

    def forward(self, input_embeds, attention_mask, position_ids, *past_key_values):
        # Reshape past_key_values from flat tuple
        # Forward through GPT-2
        # Return logits and new past_key_values
        pass
```

---

#### Issue M1-1.2: Create `quantize_onnx.py`

**Type:** Feature

**File:** `export_utility/quantize_onnx.py`

**Description:**
Quantize ONNX models to INT8 for edge deployment.

**Quantization Options:**

| Format | Size | Accuracy | Speed | Use Case |
|--------|------|----------|-------|----------|
| FP32 | 1400 MB | 100% | Baseline | Development |
| FP16 | 700 MB | ~99.9% | 1.5x | GPU inference |
| INT8 dynamic | 350 MB | ~99% | 2x | Edge (default) |
| INT8 static | 350 MB | ~99.5% | 3x | Edge (calibrated) |

**Acceptance Criteria:**

- [ ] Implement dynamic INT8 quantization (no calibration data)
- [ ] Implement static INT8 quantization (with calibration data)
- [ ] Implement FP16 conversion for GPU
- [ ] Validate quantized models produce acceptable outputs
- [ ] CLI: `python quantize_onnx.py --input decoder.onnx --output decoder_int8.onnx --mode dynamic`

**Implementation Outline:**

```python
#!/usr/bin/env python3
"""
ONNX Quantization Utility

Supports:
- Dynamic INT8 (no calibration, good for most cases)
- Static INT8 (requires calibration data, better accuracy)
- FP16 (for GPU inference)
"""

from onnxruntime.quantization import (
    quantize_dynamic,
    quantize_static,
    QuantType,
    CalibrationDataReader,
)

def quantize_dynamic_int8(input_path, output_path):
    """Dynamic INT8 quantization - no calibration needed."""
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )

def quantize_static_int8(input_path, output_path, calibration_data):
    """Static INT8 quantization - better accuracy with calibration."""
    # Create calibration data reader
    # Run quantization
    pass
```

---

### M1-Epic 2: Unified Inference Runtime

**Priority:** 🔥 Critical

**Effort:** 2 days

#### Issue M1-2.1: Create `runtime.py` - Multi-Backend Support

**Type:** Feature

**File:** `familyos_ultrabert/runtime.py`

**Description:**
Unified ONNX runtime with automatic backend selection and fallback.

**Backend Priority:**

```text
┌─────────────────────────────────────────────────────────────────┐
│ BACKEND FALLBACK CHAIN                                          │
│                                                                  │
│   AMD NPU ──────► NVIDIA CUDA ──────► CPU                       │
│   (DirectML)       (CUDA EP)         (Always works)             │
│                                                                  │
│ Priority: NPU first (power efficient for edge/nightly tasks)    │
│ Fallback: Silent logging, process never stops                   │
└─────────────────────────────────────────────────────────────────┘
```

**Acceptance Criteria:**

- [ ] Auto-detect available ONNX execution providers
- [ ] Implement fallback chain: NPU → CUDA → CPU
- [ ] Silent fallback with logging (process never stops)
- [ ] Configurable priority order
- [ ] Device capability detection (VRAM, NPU TOPS, etc.)

**Implementation Outline:**

```python
"""
Unified Inference Runtime - Multi-Backend ONNX Support

Supports:
- AMD NPU (DirectML) - Ryzen AI laptops
- NVIDIA CUDA - Gaming laptops, servers
- CPU - Universal fallback

Fallback is SILENT - logs warning but never crashes.
"""

import onnxruntime as ort
import logging

logger = logging.getLogger(__name__)

# Provider priority (configurable)
DEFAULT_PRIORITY = [
    "DmlExecutionProvider",      # AMD NPU (Ryzen AI)
    "CUDAExecutionProvider",      # NVIDIA GPU
    "ROCMExecutionProvider",      # AMD GPU (Linux)
    "CPUExecutionProvider",       # Always available
]

PROVIDER_NAMES = {
    "DmlExecutionProvider": "AMD NPU (DirectML)",
    "CUDAExecutionProvider": "NVIDIA CUDA",
    "ROCMExecutionProvider": "AMD ROCm",
    "CPUExecutionProvider": "CPU",
}


def get_best_provider(priority: list = None) -> tuple[str, str]:
    """Get best available execution provider.

    Returns:
        (provider_name, friendly_name)

    Note: Never raises - always falls back to CPU.
    """
    priority = priority or DEFAULT_PRIORITY
    available = ort.get_available_providers()

    for provider in priority:
        if provider in available:
            logger.info(f"Selected backend: {PROVIDER_NAMES.get(provider, provider)}")
            return provider, PROVIDER_NAMES.get(provider, provider)

    # Should never happen (CPU always available), but be safe
    logger.warning("No providers available, using CPU")
    return "CPUExecutionProvider", "CPU"


class ONNXSession:
    """Wrapper for ONNX Runtime InferenceSession with fallback."""

    def __init__(self, model_path: str, priority: list = None):
        provider, name = get_best_provider(priority)
        self.provider = provider
        self.provider_name = name

        try:
            self.session = ort.InferenceSession(
                model_path,
                providers=[provider],
            )
            logger.info(f"Loaded {model_path} with {name}")
        except Exception as e:
            # Fallback to CPU
            logger.warning(f"Failed to load with {name}: {e}")
            logger.warning("Falling back to CPU")
            self.session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
            )
            self.provider = "CPUExecutionProvider"
            self.provider_name = "CPU"
```

---

#### Issue M1-2.2: Verify AMD NPU Support

**Type:** Testing

**Description:**
Test DirectML execution provider on AMD Ryzen AI hardware.

**Acceptance Criteria:**

- [ ] Test on Ryzen AI 300 series (if available)
- [ ] Measure NPU power consumption vs GPU
- [ ] Document any NPU-specific limitations
- [ ] Create fallback test (NPU disabled → CUDA → CPU)

---

### M1 Deliverables

| Artifact | Location | Size |
|----------|----------|------|
| `export_decoder_onnx.py` | export_utility/ | ~400 lines |
| `quantize_onnx.py` | export_utility/ | ~200 lines |
| `runtime.py` | familyos_ultrabert/ | ~350 lines |
| `prefix_encoder.onnx` | exports/ | ~5 MB |
| `decoder_core_fp32.onnx` | exports/ | ~1400 MB |
| `decoder_core_int8.onnx` | exports/ | ~350 MB |

---

## Milestone 2: Package Architecture Update

**Goal:** Update familyos_ultrabert with decoder support and modern structure

**Duration:** 4-5 days

**Dependencies:** M1 complete

### M2-Epic 1: Sync Model Files

**Priority:** 🔥 Critical

**Effort:** 1 day

#### Issue M2-1.1: Copy Decoder Files from Training Codebase

**Type:** Task

**Description:**
Copy updated model files from `src/modeling_studio/models/` to `familyos_ultrabert/models/`.

**Files to Copy:**

| Source | Destination | Action |
|--------|-------------|--------|
| decoder_gpt2.py | models/decoder_gpt2.py | NEW |
| decoder_gpt2_config.py | models/decoder_gpt2_config.py | NEW |
| modernbert_multitask.py | models/modernbert_multitask.py | UPDATE |
| heads.py | models/heads.py | UPDATE if changed |
| adapters.py | models/adapters.py | UPDATE if changed |
| poolers.py | models/poolers.py | UPDATE if changed |

**Acceptance Criteria:**

- [ ] Copy decoder_gpt2.py with import fixes
- [ ] Copy decoder_gpt2_config.py
- [ ] Update modernbert_multitask.py imports
- [ ] Fix namespace: `modeling_studio` → `familyos_ultrabert`
- [ ] Verify no circular imports

**Commands:**

```powershell
cd D:\Modeling_studio

# Copy new decoder files
Copy-Item src/modeling_studio/models/decoder_gpt2.py familyos_ultrabert/models/
Copy-Item src/modeling_studio/models/decoder_gpt2_config.py familyos_ultrabert/models/

# Update imports in copied files
# sed -i 's/modeling_studio/familyos_ultrabert/g' familyos_ultrabert/models/decoder_*.py
```

---

#### Issue M2-1.2: Update models/**init**.py

**Type:** Task

**Description:**
Export new decoder classes from models package.

**Changes:**

```python
# familyos_ultrabert/models/__init__.py

from familyos_ultrabert.models.modernbert_multitask import ModernBertMultiTaskModel
from familyos_ultrabert.models.heads import (
    SequenceClassificationHead,
    TokenClassificationHead,
    EmbeddingHead,
    # ... existing heads
)

# NEW: Decoder exports
from familyos_ultrabert.models.decoder_gpt2 import GPT2DecoderHead
from familyos_ultrabert.models.decoder_gpt2_config import GPT2DecoderConfig

__all__ = [
    "ModernBertMultiTaskModel",
    "GPT2DecoderHead",      # NEW
    "GPT2DecoderConfig",    # NEW
    # ... existing exports
]
```

---

### M2-Epic 2: Weight Management

**Priority:** 🔥 Critical

**Effort:** 2 days

#### Issue M2-2.1: Create `weights_manager.py`

**Type:** Feature

**File:** `familyos_ultrabert/weights_manager.py`

**Description:**
Automatic weight downloading from HuggingFace Hub with caching.

**Features:**

- Download encoder weights (once)
- Download decoder weights (on demand)
- Cache in `~/.cache/familyos_ultrabert/`
- Resume interrupted downloads
- Checksum verification
- Progress bar

**Acceptance Criteria:**

- [ ] `download_encoder(version="v1")` - Returns path to encoder weights
- [ ] `download_decoder(version="v3")` - Returns path to decoder weights
- [ ] Support quantization variants: fp32, fp16, int8
- [ ] Cache management (get size, clear cache)
- [ ] Offline mode (use cached weights)

**Implementation Outline:**

```python
"""
Weight Manager - HuggingFace Hub Integration

Downloads model weights on first use, caches locally.
Supports version selection and quantization variants.
"""

from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
import logging

logger = logging.getLogger(__name__)

HF_REPO = "Pkansagra/ultrabert-weights"
CACHE_DIR = Path.home() / ".cache" / "familyos_ultrabert"


def get_cache_dir() -> Path:
    """Get cache directory, creating if needed."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def download_encoder(
    version: str = "v1",
    quantization: str = "int8",
    force: bool = False,
) -> Path:
    """Download encoder weights from HuggingFace Hub.

    Args:
        version: Encoder version (default: v1)
        quantization: "fp32", "fp16", or "int8" (default: int8)
        force: Re-download even if cached

    Returns:
        Path to local weights directory
    """
    cache_path = CACHE_DIR / f"encoder/{version}/{quantization}"

    if cache_path.exists() and not force:
        logger.info(f"Using cached encoder: {cache_path}")
        return cache_path

    logger.info(f"Downloading encoder v{version} ({quantization})...")

    # Download from HuggingFace
    snapshot_download(
        repo_id=HF_REPO,
        allow_patterns=[f"encoder/{version}/{quantization}/*"],
        local_dir=CACHE_DIR,
        local_dir_use_symlinks=False,
    )

    return cache_path


def download_decoder(
    version: str = "v3",
    quantization: str = "int8",
    force: bool = False,
) -> Path:
    """Download decoder weights from HuggingFace Hub.

    Args:
        version: Decoder version (default: v3)
        quantization: "fp32", "fp16", or "int8" (default: int8)
        force: Re-download even if cached

    Returns:
        Path to local weights directory
    """
    cache_path = CACHE_DIR / f"decoder/{version}/{quantization}"

    if cache_path.exists() and not force:
        logger.info(f"Using cached decoder: {cache_path}")
        return cache_path

    logger.info(f"Downloading decoder v{version} ({quantization})...")

    snapshot_download(
        repo_id=HF_REPO,
        allow_patterns=[f"decoder/{version}/{quantization}/*"],
        local_dir=CACHE_DIR,
        local_dir_use_symlinks=False,
    )

    return cache_path
```

---

#### Issue M2-2.2: Remove Bundled Weights

**Type:** Cleanup

**Description:**
Remove `familyos_ultrabert/weights/` directory and update pyproject.toml.

**Acceptance Criteria:**

- [ ] Delete `familyos_ultrabert/weights/` directory
- [ ] Remove `[tool.setuptools.package-data]` section from pyproject.toml
- [ ] Add `huggingface-hub>=0.20.0` to dependencies
- [ ] Update version to 3.0.0
- [ ] Verify wheel size < 50 MB

**Changes to pyproject.toml:**

```toml
[project]
name = "familyos-ultrabert"
version = "3.0.0"  # BREAKING CHANGE
description = "FamilyOS UltraBERT v3 - Multi-task NLP with decoder for counterfactual generation"

dependencies = [
    "numpy>=1.21.0",
    "transformers>=4.30.0",
    "tokenizers>=0.13.0",
    "huggingface-hub>=0.20.0",  # NEW
]

# REMOVE THIS SECTION:
# [tool.setuptools.package-data]
# "familyos_ultrabert" = [
#     "weights/pytorch/*",
#     "weights/onnx/*",
# ]
```

---

### M2-Epic 3: Update Package API

**Priority:** High

**Effort:** 1 day

#### Issue M2-3.1: Add Capability.COUNTERFACTUAL to labels.py

**Type:** Feature

**Description:**
Add counterfactual capability to enum.

**Changes:**

```python
# familyos_ultrabert/labels.py

class Capability(str, Enum):
    # Existing 12 capabilities
    SENTIMENT = "sentiment"
    EMOTIONS = "emotions"
    SAFETY_FAMILYOS = "safety_familyos"
    SAFETY_GENERIC = "safety_generic"
    INTENT = "intent"
    INGRESS = "ingress"
    NER_FAMILY = "ner_family"
    NER_GENERAL = "ner_general"
    TEMPORAL = "temporal"
    RELATION = "relation"
    NLI = "nli"
    EMBEDDING = "embedding"

    # NEW: 13th capability
    COUNTERFACTUAL = "counterfactual"


CAPABILITIES = {
    # ... existing mappings
    Capability.COUNTERFACTUAL: {
        "name": "Counterfactual Generation",
        "description": "Generate alternative phrasings and suggestions",
        "requires_decoder": True,  # NEW flag
    },
}
```

---

#### Issue M2-3.2: Update UltraBERT.load() for Decoder

**Type:** Enhancement

**Description:**
Support decoder loading in UltraBERT.load() method.

**Changes:**

```python
# familyos_ultrabert/model.py

@classmethod
def load(
    cls,
    model_path: Optional[str] = None,
    backend: Literal["auto", "pytorch", "onnx"] = "auto",
    device: Literal["auto", "cpu", "cuda", "npu"] = "auto",
    enable_cache: bool = True,
    cache_size: int = 1000,
    # NEW parameters
    encoder_version: str = "v1",
    decoder_version: str = "v3",
    quantization: str = "int8",
    load_decoder: bool = False,  # Don't load by default
) -> "UltraBERT":
    """
    Load FamilyOS UltraBERT model.

    Args:
        ...existing args...
        encoder_version: Version of encoder weights (default: v1)
        decoder_version: Version of decoder weights (default: v3)
        quantization: Weight format - "fp32", "fp16", "int8" (default: int8)
        load_decoder: Load decoder immediately (default: False for memory)
    """
    # Download weights if needed
    from .weights_manager import download_encoder, download_decoder

    encoder_path = download_encoder(encoder_version, quantization)

    decoder_path = None
    if load_decoder:
        decoder_path = download_decoder(decoder_version, quantization)

    # ... rest of loading logic
```

---

### M2 Deliverables

| Artifact | Location | Size |
|----------|----------|------|
| `decoder_gpt2.py` | familyos_ultrabert/models/ | ~600 lines |
| `decoder_gpt2_config.py` | familyos_ultrabert/models/ | ~100 lines |
| `weights_manager.py` | familyos_ultrabert/ | ~200 lines |
| Updated `pyproject.toml` | familyos_ultrabert/ | ~100 lines |
| Updated `labels.py` | familyos_ultrabert/ | +20 lines |

---

## Milestone 3: Lazy Loading & DecoderSession

**Goal:** Implement on-demand decoder loading for memory efficiency

**Duration:** 3-4 days

**Dependencies:** M2 complete

### M3-Epic 1: DecoderSession Context Manager

**Priority:** 🔥 Critical

**Effort:** 2 days

#### Issue M3-1.1: Create `decoder_session.py`

**Type:** Feature

**File:** `familyos_ultrabert/decoder_session.py`

**Description:**
Context manager for lazy loading decoder, perfect for R5 nightly processing.

**Memory Model:**

```text
┌──────────────────────────────────────────────────────────────────┐
│ MEMORY LIFECYCLE                                                  │
│                                                                   │
│ Daytime (R0-R4, R6-R8):                                          │
│ ┌─────────────────────────┐                                      │
│ │ Encoder + 12 Heads      │ 175 MB (INT8)                        │
│ └─────────────────────────┘                                      │
│                                                                   │
│ R5 Start: decoder.__enter__()                                    │
│ ┌─────────────────────────┐ ┌─────────────────────┐              │
│ │ Encoder + 12 Heads      │ │ Decoder             │ +350 MB      │
│ └─────────────────────────┘ └─────────────────────┘              │
│                             Total: 525 MB                         │
│                                                                   │
│ R5 End: decoder.__exit__()                                       │
│ ┌─────────────────────────┐                                      │
│ │ Encoder + 12 Heads      │ 175 MB (decoder freed)               │
│ └─────────────────────────┘                                      │
└──────────────────────────────────────────────────────────────────┘
```

**Acceptance Criteria:**

- [ ] Context manager pattern (`with DecoderSession() as decoder:`)
- [ ] Downloads weights on first use
- [ ] Loads ONNX model into memory on enter
- [ ] Unloads and frees memory on exit
- [ ] `generate(encoder_output) -> str`
- [ ] `generate_batch(encoder_outputs) -> List[str]`
- [ ] `generate_structured(encoder_output) -> dict`

**Implementation Outline:**

```python
"""
DecoderSession - Lazy-Loaded Decoder for R5 Dream Exploration

Memory-efficient context manager that loads decoder only when needed.
Perfect for nightly P03 consolidation where decoder is used in R5 only.

Usage:
    # Load encoder (always resident)
    client = Client(capabilities=["sentiment", "emotions"])

    # R5: Load decoder temporarily
    with DecoderSession(quantization="int8") as decoder:
        for episode in episodes:
            encoder_output = client.encode(episode.text)
            suggestion = decoder.generate(encoder_output)

    # Decoder automatically unloaded, memory freed
"""

import gc
import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .weights_manager import download_decoder
from .runtime import ONNXSession, get_best_provider

logger = logging.getLogger(__name__)


class DecoderSession:
    """Lazy-loaded decoder for counterfactual generation."""

    def __init__(
        self,
        version: str = "v3",
        quantization: str = "int8",
        device: str = "auto",  # "auto", "npu", "cuda", "cpu"
        batch_size: int = 16,
    ):
        self.version = version
        self.quantization = quantization
        self.device = device
        self.batch_size = batch_size

        # Sessions (loaded on __enter__)
        self._prefix_session = None
        self._decoder_session = None
        self._loaded = False

    def __enter__(self):
        """Load decoder into memory."""
        logger.info(f"Loading decoder v{self.version} ({self.quantization})...")

        # Download weights if needed
        weights_path = download_decoder(self.version, self.quantization)

        # Get best provider
        provider, name = get_best_provider()
        logger.info(f"Using backend: {name}")

        # Load ONNX sessions
        self._prefix_session = ONNXSession(
            str(weights_path / "prefix_encoder.onnx"),
        )
        self._decoder_session = ONNXSession(
            str(weights_path / f"decoder_core_{self.quantization}.onnx"),
        )

        self._loaded = True
        logger.info(f"Decoder loaded ({self._get_memory_mb():.1f} MB)")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Unload decoder, free memory."""
        logger.info("Unloading decoder...")

        del self._prefix_session
        del self._decoder_session
        self._prefix_session = None
        self._decoder_session = None
        self._loaded = False

        # Force garbage collection
        gc.collect()

        logger.info("Decoder unloaded, memory freed")
        return False  # Don't suppress exceptions

    def generate(
        self,
        encoder_hidden_states: np.ndarray,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> str:
        """Generate counterfactual text.

        Args:
            encoder_hidden_states: Encoder output (batch=1, seq, 768)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability

        Returns:
            Generated counterfactual text
        """
        if not self._loaded:
            raise RuntimeError("DecoderSession not loaded. Use 'with' context manager.")

        # Project encoder outputs
        prefix_embeds = self._prefix_session.run(encoder_hidden_states)

        # Autoregressive generation
        generated_ids = self._generate_tokens(
            prefix_embeds,
            max_new_tokens,
            temperature,
            top_p,
        )

        # Decode to text
        return self._decode_tokens(generated_ids)

    def generate_batch(
        self,
        encoder_hidden_states_list: List[np.ndarray],
        max_new_tokens: int = 128,
    ) -> List[str]:
        """Batch generate counterfactuals."""
        results = []
        for hidden_states in encoder_hidden_states_list:
            results.append(self.generate(hidden_states, max_new_tokens))
        return results

    def generate_structured(
        self,
        encoder_hidden_states: np.ndarray,
        max_new_tokens: int = 128,
    ) -> dict:
        """Generate with structured output.

        Returns:
            {
                "text": "If you had scheduled 15 minutes...",
                "raw": "If you had scheduled 15 minutes of daily...",
                "procedural_insight": {
                    "trigger": "feeling overwhelmed",
                    "action": "schedule personal time",
                    "expected_outcome": "reduced stress"
                }
            }
        """
        text = self.generate(encoder_hidden_states, max_new_tokens)

        return {
            "text": text,
            "raw": text,
            "procedural_insight": self._extract_insight(text),
        }

    def _extract_insight(self, text: str) -> dict:
        """Extract procedural insight from generated text."""
        # Simple pattern matching for now
        # Could be enhanced with another model call
        return {
            "trigger": "parsed from text",
            "action": "parsed from text",
            "expected_outcome": "parsed from text",
        }
```

---

#### Issue M3-1.2: Integrate DecoderSession with Client

**Type:** Enhancement

**File:** `familyos_ultrabert/client.py`

**Description:**
Add decoder methods to Client class.

**New Methods:**

```python
# familyos_ultrabert/client.py

class Client:
    # ... existing code ...

    def create_decoder_session(
        self,
        version: str = "v3",
        quantization: str = "int8",
        device: str = "auto",
    ) -> DecoderSession:
        """Create decoder session for batch processing.

        Usage:
            with client.create_decoder_session() as decoder:
                for text in texts:
                    encoder_output = client.encode(text)
                    suggestion = decoder.generate(encoder_output)
        """
        from .decoder_session import DecoderSession
        return DecoderSession(version, quantization, device)

    def suggest_alternative(self, text: str) -> str:
        """Generate counterfactual suggestion (one-off).

        Note: Loads decoder temporarily, slower than batch processing.
        For multiple texts, use create_decoder_session().
        """
        with self.create_decoder_session() as decoder:
            encoder_output = self.encode(text)
            return decoder.generate(encoder_output)

    def suggest_alternative_structured(self, text: str) -> dict:
        """Generate counterfactual with structured output."""
        with self.create_decoder_session() as decoder:
            encoder_output = self.encode(text)
            return decoder.generate_structured(encoder_output)

    def encode(self, text: str) -> np.ndarray:
        """Get encoder hidden states for a text.

        Used with DecoderSession for generation.
        """
        # Run encoder only, return hidden states
        pass
```

---

### M3-Epic 2: P03 Integration Example

**Priority:** Medium

**Effort:** 1 day

#### Issue M3-2.1: Create P03 Integration Example

**Type:** Documentation

**File:** `familyos_ultrabert/examples/p03_dreaming.py`

**Description:**
Show how to integrate with P03 dreaming pipeline.

```python
"""
Example: P03 Dreaming Pipeline Integration

Shows how to use DecoderSession for R5 counterfactual generation.
"""

from familyos_ultrabert import Client
from familyos_ultrabert.decoder_session import DecoderSession


class DreamingPipeline:
    """Example P03 dreaming pipeline."""

    def __init__(self):
        # Encoder always resident (175 MB with INT8)
        self.client = Client(
            capabilities=["sentiment", "emotions", "topics", "entities"],
            backend="onnx",
            quantization="int8",
        )

    def run_consolidation(self, events):
        """Run full nightly consolidation."""

        # R0-R4: Encoder-only phases
        episodes = self._run_r0_to_r4(events)

        # R5: Dream exploration (decoder loaded temporarily)
        counterfactuals = self._run_r5_dreams(episodes)

        # R6-R8: Encoder-only again
        self._run_r6_to_r8(counterfactuals)

    def _run_r5_dreams(self, episodes):
        """R5: Load decoder, generate counterfactuals, unload."""

        counterfactuals = []

        # Decoder loaded here, unloaded after context exits
        with self.client.create_decoder_session(
            quantization="int8",
            device="auto",  # NPU → CUDA → CPU fallback
        ) as decoder:

            for episode in episodes:
                # Get encoder representation
                encoder_output = self.client.encode(episode.text)

                # Generate counterfactual
                result = decoder.generate_structured(encoder_output)

                counterfactuals.append({
                    "episode_id": episode.id,
                    "original": episode.text,
                    **result,
                })

        # Decoder automatically unloaded
        # Memory: 525 MB → 175 MB

        return counterfactuals


if __name__ == "__main__":
    pipeline = DreamingPipeline()

    # Simulate events
    events = [
        {"text": "Had dinner with Mom at Luigi's"},
        {"text": "Felt overwhelmed with work deadlines"},
        {"text": "Kids argued about screen time"},
    ]

    pipeline.run_consolidation(events)
```

---

### M3 Deliverables

| Artifact | Location | Size |
|----------|----------|------|
| `decoder_session.py` | familyos_ultrabert/ | ~300 lines |
| Updated `client.py` | familyos_ultrabert/ | +50 lines |
| `p03_dreaming.py` | examples/ | ~100 lines |

---

## Milestone 4: HuggingFace Upload & Release ✅ COMPLETE

**Goal:** Upload weights to HuggingFace and publish v3.0.0

**Status:** ✅ COMPLETED (2024-12-18)

**Duration:** 2-3 days

**Dependencies:** M1, M2, M3 complete

### M4-Epic 1: Upload Weights to HuggingFace ✅

**Priority:** 🔥 Critical

**Effort:** 1 day

#### Issue M4-1.1: Setup HuggingFace Repository Structure ✅

**Type:** Setup

**Repository:** `Pkansagra/ultrabert-weights`

**Structure:**

```text
Pkansagra/ultrabert-weights/
├── README.md                    # Model card
├── encoder/
│   └── v1/
│       ├── fp32/
│       │   ├── model.safetensors
│       │   └── config.json
│       ├── fp16/
│       │   └── model.onnx
│       └── int8/
│           └── model.onnx       # DEFAULT
└── decoder/
    ├── v1/                      # Baseline
    ├── v2/                      # Failed experiment (for reference)
    └── v3/                      # BEST (+13% coherence)
        ├── fp32/
        │   ├── prefix_encoder.onnx
        │   ├── decoder_core.onnx
        │   └── config.json
        ├── fp16/
        │   ├── prefix_encoder.onnx
        │   └── decoder_core.onnx
        └── int8/
            ├── prefix_encoder.onnx
            └── decoder_core.onnx   # DEFAULT
```

---

#### Issue M4-1.2: Upload Encoder Weights

**Type:** Task

**Commands:**

```bash
huggingface-cli login

python export_utility/upload_weights_to_hf.py \
    --component encoder \
    --version v1 \
    --quantization all
```

---

#### Issue M4-1.3: Upload Decoder Weights

**Type:** Task

**Commands:**

```bash
python export_utility/upload_weights_to_hf.py \
    --component decoder \
    --version v3 \
    --checkpoint outputs/ultrabert-gen-decoder-v3 \
    --quantization all
```

---

### M4-Epic 2: Build and Test Package

**Priority:** 🔥 Critical

**Effort:** 1 day

#### Issue M4-2.1: Build Lightweight Wheel

**Type:** Build

**Commands:**

```powershell
cd D:\Modeling_studio\familyos_ultrabert

# Clean build artifacts
Remove-Item -Recurse -Force dist, build, *.egg-info -ErrorAction SilentlyContinue

# Build wheel
python -m build

# Check size (should be < 50 MB, target ~10 MB)
Get-ChildItem dist/*.whl | Select-Object Name, @{N="Size MB"; E={[math]::Round($_.Length/1MB, 2)}}
```

---

#### Issue M4-2.2: Test in Clean Environment

**Type:** Testing

**Script:**

```powershell
# Create clean virtualenv
python -m venv test_v3_env
.\test_v3_env\Scripts\Activate

# Install wheel
pip install dist/familyos_ultrabert-3.0.0-py3-none-any.whl torch onnxruntime

# Test encoder
python -c "
from familyos_ultrabert import Client
client = Client()
result = client.analyze('I love my family')
print(f'Sentiment: {result.sentiment}')
"

# Test decoder
python -c "
from familyos_ultrabert import Client
client = Client()
with client.create_decoder_session() as decoder:
    output = client.encode('I feel overwhelmed')
    suggestion = decoder.generate(output)
    print(f'Suggestion: {suggestion}')
"

deactivate
```

---

### M4-Epic 3: Publish Release

**Priority:** 🔥 Critical

**Effort:** 1 day

#### Issue M4-3.1: Delete Old GitHub Releases

**Type:** Cleanup

**Commands:**

```powershell
gh release delete v2.2.1 --yes
gh release delete v2.2.0 --yes
gh release delete v2.1.0 --yes
gh release delete v2.0.3 --yes
gh release delete v2.0.2 --yes
gh release delete v2.0.1 --yes
gh release delete v2.0.0 --yes
```

---

#### Issue M4-3.2: Create v3.0.0 GitHub Release

**Type:** Release

**Commands:**

```powershell
git tag -a v3.0.0 -m "v3.0.0 - Edge-ready decoder with lazy loading"
git push origin v3.0.0

gh release create v3.0.0 `
    familyos_ultrabert/dist/familyos_ultrabert-3.0.0-py3-none-any.whl `
    --title "FamilyOS UltraBERT v3.0.0 - Edge-Ready Architecture" `
    --notes "See RELEASE_NOTES.md for details"
```

---

#### Issue M4-3.3: Publish to PyPI

**Type:** Release

**Commands:**

```powershell
# Test PyPI first
twine upload --repository testpypi familyos_ultrabert/dist/*

# Production
twine upload familyos_ultrabert/dist/*
```

---

### M4 Deliverables

| Artifact | Location | Size |
|----------|----------|------|
| HuggingFace weights | Pkansagra/ultrabert-weights | ~2 GB total |
| v3.0.0 wheel | PyPI | ~10 MB |
| GitHub release | v3.0.0 | ~10 MB |

---

## Summary: Complete Timeline

| Milestone | Duration | Key Deliverables |
|-----------|----------|------------------|
| **M1: ONNX Infrastructure** | Week 1 | export_decoder_onnx.py, quantize_onnx.py, runtime.py |
| **M2: Package Update** | Week 2 | decoder_gpt2.py, weights_manager.py, updated API |
| **M3: Lazy Loading** | Week 2-3 | decoder_session.py, Client integration |
| **M4: Release** | Week 3-4 | HuggingFace upload, PyPI publish, GitHub release |

## Memory Comparison

| Scenario | Before (v2.x) | After (v3.0) | Reduction |
|----------|---------------|--------------|-----------|
| Encoder only | 620 MB | 175 MB (INT8) | 3.5x |
| Encoder + Decoder | 2020 MB | 525 MB (INT8) | 3.8x |
| Wheel size | 1590 MB | ~10 MB | 159x |

## Backend Support Matrix

| Backend | Windows | Linux | Docker |
|---------|---------|-------|--------|
| AMD NPU (DirectML) | ✅ | ❌ | ❌ |
| NVIDIA CUDA | ✅ | ✅ | ✅ |
| AMD ROCm | ❌ | ✅ | ✅ |
| CPU | ✅ | ✅ | ✅ |
