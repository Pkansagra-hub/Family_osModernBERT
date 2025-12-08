# Phase 0.5 Healing Training Plan

## ModernBERT v3 Ultra - Enhanced Healing Implementation Guide

**Version:** 1.1
**Date:** December 2025
**Status:** Production Implementation Specification
**Author:** FamilyOS Model Training Team

---

## Implementation Checklist

> **Start here.** Complete each epic in order. Check off items as you go.

### Epic 0: Pre-Flight Verification

- [x] **0.1** Verify v2 checkpoint exists at `checkpoints/modernbert-v2-for-v3-transfer/checkpoint-4000/model.safetensors`
- [x] **0.2** Verify tokenizer has hub tokens or can add them
- [x] **0.3** Verify GPU available with 16GB+ VRAM
- [x] **0.4** Verify all v3 trainer modules exist and import correctly
- [x] **0.5** Run module import test: model architecture aligned with ModernBERT

### Epic 1: Data Pipeline Setup

- [x] **1.1** Run `prepare_healing_data_enhanced.py` to generate healing data
- [x] **1.2** Verify output: `data/healing/healing_enhanced.jsonl` exists with 12,000 samples
- [x] **1.3** Validate data format with `--validate` flag
- [x] **1.4** Verify task distribution: SST-2 (3K), CoNLL (3K), MNLI (2K), SQuAD (2K), STS-B (2K)

### Epic 2: Model Initialization

- [x] **2.1** Run `initialize_from_v2()` to create v3 model from safetensors checkpoint
- [x] **2.2** Verify output checkpoint at `checkpoints/v3-initialized-from-v2/`
- [x] **2.3** Verify model has 28 layers, vocab size 50432 (50368 + 64 padding for alignment)
- [x] **2.4** Verify hub token embeddings initialized (semantic centroid method)
- [x] **2.5** Run function preservation verification (PASSED - all 6 tests)

### Epic 3: Training Script Creation

- [x] **3.1** Create `scripts/train_v3_phase0_5.py` with all imports
- [x] **3.2** Implement `Phase05Config` dataclass
- [x] **3.3** Implement `parse_args()` with dry-run, smoke-test, debug modes
- [x] **3.4** Implement `load_config()` to merge YAML + CLI args
- [x] **3.5** Implement `setup_model()` with layer freezing
- [x] **3.6** Implement `setup_optimizer()` with Zipper LR
- [x] **3.7** Implement `setup_data()` with healing data loading
- [x] **3.8** Implement `train_phase_0_5()` main loop
- [x] **3.9** Implement `evaluate()` function
- [x] **3.10** Implement `save_checkpoint()` and `load_checkpoint()`
- [x] **3.11** Implement `run_dry_run()` validation
- [x] **3.12** Implement `run_smoke_test()` with 10 steps
- [x] **3.13** Implement `main()` entry point

### Epic 4: Validation & Testing

- [x] **4.1** Run dry-run: `python scripts/train_v3_phase0_5.py --dry-run`
- [x] **4.2** Run smoke test: `python scripts/train_v3_phase0_5.py --smoke-test`
- [ ] **4.3** Run debug mode: `python scripts/train_v3_phase0_5.py --debug`
- [x] **4.4** Verify loss decreases in smoke test
- [x] **4.5** Verify no NaN losses
- [ ] **4.6** Verify gradient logging works

### Epic 5: Full Training Run

- [ ] **5.1** Start full training: `python scripts/train_v3_phase0_5.py`
- [ ] **5.2** Monitor W&B dashboard for metrics
- [ ] **5.3** Verify checkpoints saved at steps 500, 1000, 1500, 2000, 2500
- [ ] **5.4** Verify best model saved when val_loss improves
- [ ] **5.5** Verify final val_loss < 2.0

### Epic 6: Post-Training Validation

- [ ] **6.1** Load best checkpoint and run inference test
- [ ] **6.2** Verify interface activation similarity > 0.8
- [ ] **6.3** Verify hub token dispersion > 0.3
- [ ] **6.4** Verify embedding stability > 0.95
- [ ] **6.5** Run task-specific evaluation on each of 5 tasks

---

## Current Progress

| Epic | Status | Notes |
|------|--------|-------|
| Epic 0: Pre-Flight | COMPLETE | v2 checkpoint at safetensors format, architecture aligned |
| Epic 1: Data Pipeline | COMPLETE | 12K healing samples generated |
| Epic 2: Model Init | COMPLETE | 182M params, 149M transferred, 30M cloned |
| Epic 3: Training Script | In Progress | |
| Epic 4: Validation | Not Started | |
| Epic 5: Full Training | Not Started | |
| Epic 6: Post-Training | Not Started | |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Phase 0.5 Objectives](#2-phase-05-objectives)
3. [Architecture Context](#3-architecture-context)
4. [Data Pipeline](#4-data-pipeline)
5. [Model Configuration](#5-model-configuration)
6. [Training Infrastructure](#6-training-infrastructure)
7. [Training Script Implementation](#7-training-script-implementation)
8. [Execution Workflow](#8-execution-workflow)
9. [Monitoring & Validation](#9-monitoring--validation)
10. [Troubleshooting Guide](#10-troubleshooting-guide)
11. [File Inventory](#11-file-inventory)
12. [Acceptance Criteria](#12-acceptance-criteria)

---

## 1. Executive Summary

### What is Phase 0.5?

Phase 0.5 ("Enhanced Healing") is the critical first training phase after v3 model initialization. It **repairs the cloned layers** (L23-28, cloned from v2's L15-20) and **establishes smooth activation flow** across the L22→L23 interface boundary.

### Why is Phase 0.5 Necessary?

When we clone v2 layers 15-20 to create v3 layers 23-28, we create an artificial "clone boundary" at L22→L23:

```
L22 (original v2 layer 22) → L23 (clone of v2 layer 15)
                              ^^^
                          INTERFACE DISCONTINUITY
```

Without healing:

- L23 expects inputs that look like L14's outputs (not L22's)
- Gradient flow is unstable at the interface
- Hub tokens are randomly positioned in embedding space
- Model produces incoherent outputs

Phase 0.5 solves this by:

1. **Smooth Interface Training** - L23 learns to accept L22 outputs
2. **Graduated LR Strategy** - Maximum plasticity at L23, decreasing outward
3. **Hub Token Semantic Alignment** - Hub tokens find their semantic neighborhoods
4. **Capability Preservation** - L1-18 frozen to preserve v2 knowledge

### Training Overview

| Aspect | Specification |
|--------|---------------|
| **Duration** | 2,500 steps |
| **Warmup** | 500 steps (20%) |
| **Data** | 12,000 samples (5 tasks) |
| **Batch Size** | 32 |
| **Frozen Layers** | L1-18 (Foundation + Core) |
| **Trainable Layers** | L19-28 (SEMANTIC + Family) |
| **LR Strategy** | Zipper (L23 at 5e-5, graduated decay) |
| **Expected Time** | ~30 mins on A100 |

---

## 2. Phase 0.5 Objectives

### 2.1 Primary Objectives

1. **Heal L23-28 Activation Patterns**
   - Cloned layers must adapt to v3's deeper architecture
   - L23 must learn to process L22 outputs correctly
   - Family band (L23-28) must form coherent processing chain

2. **Smooth L22→L23 Interface**
   - Activation distributions must match across interface
   - Gradient flow must be stable (no explosions)
   - Interface similarity > 0.8 cosine

3. **Preserve L1-22 Capabilities**
   - Foundation (L1-6) and Core (L7-18) frozen
   - SEMANTIC (L19-22) trainable with low LR
   - v2 NLU capabilities must be intact

4. **Initialize Hub Token Semantics**
   - `[EMO]` → near emotion word embeddings
   - `[MEM]` → near memory/recall embeddings
   - `[REL]` → near family/relationship embeddings
   - `[TASK]` → near action/intent embeddings

### 2.2 Success Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Interface Activation Similarity | > 0.8 | Cosine similarity L22↔L23 |
| Embedding Space Stability | > 0.95 | Pre/post embedding similarity |
| Validation Loss | < 2.0 | Multi-task loss on held-out data |
| Gradient Norm (L23) | < 5.0 | Interface layer gradient stability |
| Hub Token Dispersion | > 0.3 | Hub tokens spread in embedding space |

### 2.3 Failure Modes to Avoid

| Failure Mode | Symptom | Prevention |
|--------------|---------|------------|
| Gradient Explosion | NaN loss, L23 grad > 10 | Interface clipping (0.5) |
| Embedding Collapse | All outputs identical | STS-B similarity task |
| Forgetting | L1-18 outputs drift | Freeze L1-18 completely |
| Hub Token Clustering | All hubs same position | Per-hub gradient masking |
| Interface Rejection | L23 ignores L22 | Maximum LR at L23 |

---

## 3. Architecture Context

### 3.1 v3 Layer Band Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  ModernBERT v3 Ultra (28 Layers)                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  INPUT: [CLS] [EMO] [MEM] [REL] [TASK] <text> [SEP]     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  FOUNDATION BAND (L1-6)    | Window: 64  | FROZEN       │   │
│  │  Copied from v2 L1-6       | LR: 0       | Params: ~20M │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  CORE BAND (L7-18)         | Window: 128 | FROZEN       │   │
│  │  Copied from v2 L7-18      | LR: 0       | Params: ~80M │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  SEMANTIC BAND (L19-22)      | Window: 256 | TRAINABLE    │   │
│  │  Copied from v2 L19-22     | LR: 1e-5    | Params: ~25M │   │
│  │  Purpose: Prepare for L23 interface                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  ╔═════════════════════════════════════════════════════════╗   │
│  ║  L22 → L23 INTERFACE BOUNDARY (Critical Healing Point)  ║   │
│  ╚═════════════════════════════════════════════════════════╝   │
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  FAMILY BAND (L23-28)      | Window: 512 | TRAINABLE    │   │
│  │  Cloned from v2 L15-20     | LR: 5e-5→3e-5 | Params: ~40M│   │
│  │  Purpose: Family-specific reasoning                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  POOLER + TASK HEADS                                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Hub Token Layout

| Position | Token | ID | Purpose | Capability Routes |
|----------|-------|-----|---------|-------------------|
| 0 | `[CLS]` | 0 | Sequence classification | Fallback |
| 1 | `[EMO]` | 50368 | Affective understanding | emotions, sentiment, safety |
| 2 | `[MEM]` | 50369 | Memory retrieval | embedding |
| 3 | `[REL]` | 50370 | Relational reasoning | nli, relation |
| 4 | `[TASK]` | 50371 | Action classification | intent, ingress |
| 5+ | Text | Variable | Input tokens | - |

### 3.3 Vocabulary Layout

```python
# Vocabulary structure
V2_VOCAB_SIZE = 50368        # Original ModernBERT vocab (0-50367)
HUB_TOKEN_START = 50368      # First hub token position
HUB_TOKEN_COUNT = 4          # [EMO], [MEM], [REL], [TASK]
V3_VOCAB_SIZE = 50372        # Total: 50368 + 4

# Hub token IDs
HUB_TOKEN_IDS = {
    "[EMO]":  50368,
    "[MEM]":  50369,
    "[REL]":  50370,
    "[TASK]": 50371,
}
```

---

## 4. Data Pipeline

### 4.1 Enhanced Healing Dataset Overview

Phase 0.5 uses a **carefully curated mix of 5 tasks** designed to:

1. Preserve classification capability (SST-2)
2. Ground token representations (CoNLL NER)
3. Maintain reasoning (MNLI)
4. Heal long-range attention (SQuAD)
5. Prevent embedding collapse (STS-B)

| Task | Source | Samples | Purpose | Hub Token |
|------|--------|---------|---------|-----------|
| Sentiment | SST-2 | 3,000 | Classification grounding | `[EMO]` |
| NER | CoNLL-2003 | 3,000 | Token representation | Token-level |
| NLI | MNLI | 2,000 | Reasoning preservation | `[REL]` |
| QA | SQuAD | 2,000 | Long-range attention | Context-aware |
| Similarity | STS-B | 2,000 | Embedding stability | `[MEM]` |
| **Total** | - | **12,000** | - | - |

### 4.2 Data Preparation Scripts

#### Basic Healing Data: `scripts/prepare_healing_data.py`

```bash
# Generate basic healing data (SST-2, CoNLL, MNLI)
python scripts/prepare_healing_data.py \
    --output data/healing/healing_generic.jsonl \
    --seed 42 \
    --validate
```

**Output Format (JSONL):**

```json
{
    "text": "This movie was absolutely wonderful!",
    "task": "sentiment",
    "task_type": "classification",
    "labels": {"sentiment": 1, "sentiment_label": "positive"},
    "source": "sst2",
    "split": "healing"
}
```

#### Enhanced Healing Data: `scripts/prepare_healing_data_enhanced.py`

```bash
# Generate enhanced healing data (adds SQuAD, STS-B)
python scripts/prepare_healing_data_enhanced.py \
    --output data/healing/healing_enhanced.jsonl \
    --seed 42 \
    --validate
```

**Additional Output Formats:**

```json
// SQuAD (QA)
{
    "text": "What is the capital of France? [SEP] Paris is the capital of France...",
    "question": "What is the capital of France?",
    "context": "Paris is the capital of France...",
    "task": "qa",
    "task_type": "span_extraction",
    "labels": {"answer_text": "Paris", "answer_start": 0, "answer_end": 5},
    "source": "squad",
    "healing_purpose": "Heal long-range attention by requiring context-question alignment"
}

// STS-B (Similarity)
{
    "text": "A man is playing guitar. [SEP] Someone is playing a musical instrument.",
    "sentence1": "A man is playing guitar.",
    "sentence2": "Someone is playing a musical instrument.",
    "task": "similarity",
    "task_type": "regression",
    "labels": {"similarity_score": 4.2, "normalized_score": 0.84},
    "source": "stsb",
    "healing_purpose": "Prevent embedding collapse, maintain semantic similarity"
}
```

### 4.3 Data Configuration: `configs/data/multitask/healing_enhanced.yaml`

```yaml
dataset:
  name: "healing_enhanced"
  version: "1.0"
  total_samples: 12000

paths:
  unified_file: "data/healing/healing_enhanced.jsonl"
  cache_dir: "data/cache/healing_enhanced/"

tasks:
  sentiment:
    source: "sst2"
    n_samples: 3000
    task_type: "classification"
    num_labels: 2
    weight: 1.0
    healing_target: "classification_head"

  ner:
    source: "conll2003"
    n_samples: 3000
    task_type: "token_classification"
    num_labels: 9
    weight: 1.0
    healing_target: "token_representations"

  nli:
    source: "mnli"
    n_samples: 2000
    task_type: "classification"
    num_labels: 3
    weight: 1.0
    healing_target: "reasoning_capability"

  qa:
    source: "squad"
    n_samples: 2000
    task_type: "span_extraction"
    weight: 1.2  # Higher weight for attention healing
    healing_target: "attention_patterns"

  similarity:
    source: "stsb"
    n_samples: 2000
    task_type: "regression"
    weight: 1.2  # Higher weight for embedding preservation
    healing_target: "embedding_space"
```

### 4.4 Data Loading Implementation

#### v3 Collators: `src/modeling_studio/data/collators_v3.py`

The v3 collators handle hub token insertion and label offsetting:

```python
# Key classes
V3CollatorConfig      # Configuration dataclass
V3BaseCollator        # Base with hub token insertion
V3ClassificationCollator   # For sentiment, NLI
V3TokenClassificationCollator  # For NER (with +5 offset)
V3MultiTaskCollator   # Unified multi-task batching

# Token layout transformation
# Input:  [CLS] <text tokens> [SEP] [PAD]...
# Output: [CLS] [EMO] [MEM] [REL] [TASK] <text tokens> [SEP] [PAD]...
```

**Critical: NER Label Offsetting**

```python
# Original NER labels (from CoNLL):
# labels = [0, 1, 2, 0, 0, 3, 4, 0]  # for 8 tokens

# After hub token insertion, labels must be offset:
# new_labels = [-100, -100, -100, -100, -100, 0, 1, 2, 0, 0, 3, 4, 0]
#              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  5 positions for hub tokens
```

---

## 5. Model Configuration

### 5.1 v3 Model Initialization

Before Phase 0.5, the model must be initialized from v2 weights:

```bash
# Initialize v3 from v2 checkpoint
python scripts/initialize_v3_from_v2.py \
    --v2-checkpoint checkpoints/v2_layers_15_20_for_v3.pt \
    --output-dir checkpoints/v3_initialized \
    --verify
```

**Initialization Process:**

1. Load v2 checkpoint (22 layers, 768 hidden)
2. Create v3 model (28 layers, 768 hidden)
3. Copy L1-22 weights directly
4. Clone L15-20 → L23-28
5. Add hub token embeddings (semantic centroid initialization)
6. Resize embedding matrix: 50368 → 50372
7. Verify function preservation (optional)
8. Save initialized checkpoint

### 5.2 Model Configuration: `configs/model/encoder/modernbert_v3_ultra.yaml`

```yaml
model:
  name: "ModernBERTv3Ultra"
  architecture: "encoder"

  # Dimensions
  num_layers: 28
  hidden_size: 768
  intermediate_size: 3072
  num_attention_heads: 12
  head_dim: 64

  # Vocabulary
  vocab_size: 50372  # 50368 + 4 hub tokens

  # Hub tokens
  num_hub_tokens: 4
  hub_token_names: ["[EMO]", "[MEM]", "[REL]", "[TASK]"]
  hub_token_start_id: 50368

  # Attention
  attention_type: "flash_attention_2"
  sliding_window_sizes:
    foundation: 64   # L1-6
    core: 128        # L7-18
    SEMANTIC: 256      # L19-22
    family: 512      # L23-28
  global_attention_positions: [0, 1, 2, 3, 4]  # CLS + hub tokens

  # FFN
  ffn_type: "gelu"  # Same as v2 for weight transfer

  # Other
  max_position_embeddings: 8192
  dropout: 0.1
  layer_norm_eps: 1e-5
```

### 5.3 Hub Token Semantic Initialization

Hub tokens are NOT randomly initialized. They use semantic centroid initialization:

```python
# From src/modeling_studio/models/hub_initialization_v3.py

SEMANTIC_SEEDS = {
    "[EMO]": ["happy", "sad", "angry", "fear", "joy", "love", "hate",
              "emotion", "feeling", "mood", "sentiment"],
    "[MEM]": ["remember", "memory", "recall", "forget", "past", "store",
              "retrieve", "history", "event", "experience"],
    "[REL]": ["family", "mother", "father", "child", "sibling", "parent",
              "relationship", "related", "connection", "bond"],
    "[TASK]": ["action", "intent", "goal", "task", "command", "request",
               "want", "need", "do", "perform"],
}

def initialize_hub_tokens_semantic(model, tokenizer, v2_embeddings):
    """Initialize hub tokens as centroids of semantic clusters."""
    for hub_name, seed_words in SEMANTIC_SEEDS.items():
        # Get embeddings of seed words
        seed_ids = [tokenizer.encode(w, add_special_tokens=False)[0]
                    for w in seed_words if w in tokenizer.get_vocab()]

        # Compute centroid
        centroid = v2_embeddings[seed_ids].mean(dim=0)

        # Assign to hub token
        hub_id = HUB_TOKEN_IDS[hub_name]
        model.embeddings.word_embeddings.weight.data[hub_id] = centroid
```

---

## 6. Training Infrastructure

### 6.1 Layer Freezing: `src/modeling_studio/trainers/freezing_v3.py`

**Layer Band Definitions:**

```python
class LayerBand(Enum):
    FOUNDATION = "foundation"  # L1-6 (indices 0-5)
    CORE = "core"              # L7-18 (indices 6-17)
    SEMANTIC = "SEMANTIC"          # L19-22 (indices 18-21)
    FAMILY = "family"          # L23-28 (indices 22-27)

LAYER_BANDS = {
    LayerBand.FOUNDATION: list(range(0, 6)),   # Frozen
    LayerBand.CORE: list(range(6, 18)),        # Frozen
    LayerBand.SEMANTIC: list(range(18, 22)),     # Trainable
    LayerBand.FAMILY: list(range(22, 28)),     # Trainable
}
```

**Phase 0.5 Freezing Strategy:**

```python
class LayerFreezer:
    def configure_for_phase(self, phase: TrainingPhase) -> dict:
        """Configure freezing for training phase."""

        if phase == TrainingPhase.PHASE_0_5:
            # Freeze Foundation + Core (L1-18)
            self.freeze_bands([LayerBand.FOUNDATION, LayerBand.CORE])

            # Trainable: SEMANTIC + Family (L19-28)
            self.unfreeze_bands([LayerBand.SEMANTIC, LayerBand.FAMILY])

            # Freeze embeddings except hub tokens
            self.freeze_embeddings(except_hub_tokens=True)
```

**Expected Parameter Counts:**

| Component | Parameters | Status |
|-----------|-----------|--------|
| Foundation (L1-6) | ~20M | Frozen |
| Core (L7-18) | ~80M | Frozen |
| SEMANTIC (L19-22) | ~25M | Trainable |
| Family (L23-28) | ~40M | Trainable |
| Embeddings (0-50367) | ~38M | Frozen |
| Hub Tokens (50368-50371) | ~3K | Trainable |
| Task Heads | ~5M | Trainable |
| **Total Frozen** | **~138M** | - |
| **Total Trainable** | **~70M** | - |

### 6.2 Zipper Learning Rate: `src/modeling_studio/trainers/zipper_lr_v3.py`

The Zipper LR strategy creates smooth transitions across the interface:

```
LR Profile (Phase 0.5):
           │
     5e-5 ─┼─────────────────────────────● L23 (Interface - MAX)
           │                            ╱
     4e-5 ─┼───────────────────────────● L24
           │                          ╱
   3.5e-5 ─┼──────────────────────────● L25
           │                         ╱
     3e-5 ─┼─────────────────────────●─●─● L26-28
           │
     1e-5 ─┼───●───●───●───●  L19-22 (SEMANTIC)
           │
       0 ──┼───────────────────────────────  L1-18 (Frozen)
           │
           └──────────────────────────────────
              L1  L18  L19  L22  L23  L28
```

**Zipper Configuration:**

```python
@dataclass
class ZipperLRConfig:
    base_lr: float = 3e-5

    # SEMANTIC band (L19-22) - uniform low LR
    SEMANTIC_lr: float = 1e-5

    # Interface layer (L23) - maximum plasticity
    interface_lr: float = 5e-5

    # Family band (L24-28) - graduated decay
    family_lr: float = 3e-5
    family_graduated: bool = True
    family_decay: float = 0.85  # Each layer = prev * 0.85

    # Frozen layers (L1-18)
    frozen_lr: float = 0.0

    # Components
    embeddings_lr: float = 0.0  # Frozen (except hub tokens)
    hub_tokens_lr: float = 1e-5
    task_heads_lr: float = 3e-5
```

**Optimizer Creation:**

```python
def create_zipper_optimizer(model, config: ZipperLRConfig):
    """Create optimizer with layer-specific LRs."""

    param_groups = []

    # Frozen layers (L1-18)
    for i in range(18):
        params = list(model.encoder.layers[i].parameters())
        param_groups.append({"params": params, "lr": 0.0})

    # SEMANTIC (L19-22)
    for i in range(18, 22):
        params = list(model.encoder.layers[i].parameters())
        param_groups.append({"params": params, "lr": config.SEMANTIC_lr})

    # Interface (L23)
    params = list(model.encoder.layers[22].parameters())
    param_groups.append({"params": params, "lr": config.interface_lr})

    # Family (L24-28) with decay
    for i, layer_idx in enumerate(range(23, 28)):
        lr = config.interface_lr * (config.family_decay ** (i + 1))
        params = list(model.encoder.layers[layer_idx].parameters())
        param_groups.append({"params": params, "lr": lr})

    return torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)
```

### 6.3 Gradient Clipping: `src/modeling_studio/trainers/gradient_utils_v3.py`

**Configuration:**

```python
@dataclass
class GradientClipConfig:
    max_grad_norm: float = 1.0        # Global clip

    per_layer_clip: bool = True
    interface_clip: float = 0.5       # L23: tighter clip
    family_clip: float = 1.0          # L24-28
    SEMANTIC_clip: float = 1.0          # L19-22

    log_grad_norms: bool = True
    log_every_n_steps: int = 100
    explosion_threshold: float = 10.0  # Warn if exceeded
    nan_check: bool = True
```

**Interface Gradient Monitor:**

```python
class InterfaceGradientMonitor:
    """Monitor gradient flow at L22→L23 interface."""

    def __init__(self, interface_layer: int = 22):
        self.interface_layer = interface_layer
        self.history = []

    def log_step(self, model, step: int) -> dict:
        """Log interface gradient statistics."""

        l22_grad_norm = self._get_layer_grad_norm(model, 21)  # L22
        l23_grad_norm = self._get_layer_grad_norm(model, 22)  # L23

        ratio = l23_grad_norm / (l22_grad_norm + 1e-8)

        stats = {
            "interface/l22_grad_norm": l22_grad_norm,
            "interface/l23_grad_norm": l23_grad_norm,
            "interface/grad_ratio": ratio,
        }

        # Warn if ratio too high (unstable)
        if ratio > 5.0:
            logger.warning(f"High interface grad ratio: {ratio:.2f}")

        return stats
```

### 6.4 Hub Token Gradient Masking: `src/modeling_studio/trainers/gradient_masking_v3.py`

**Purpose:** Selectively train hub tokens while freezing original vocabulary.

```python
@dataclass
class GradientMaskConfig:
    train_hub_tokens: list[str] = None  # Default: all 4
    freeze_original_vocab: bool = True  # Freeze 0-50367
    hub_token_grad_scale: float = 1.0   # Gradient scaling

class EmbeddingGradientHook:
    """Apply gradient mask to embeddings."""

    def _build_gradient_mask(self, embedding_weight):
        """Build mask: 0 for frozen, scale for trainable."""

        vocab_size = embedding_weight.shape[0]  # 50372
        mask = torch.zeros(vocab_size, 1)       # All frozen

        # Enable hub tokens
        for token_name in self.config.train_hub_tokens:
            position = HUB_TOKEN_POSITIONS[token_name]
            mask[position] = self.config.hub_token_grad_scale

        return mask

    def _gradient_hook(self, grad):
        """Mask gradients during backward pass."""
        return grad * self.grad_mask
```

### 6.5 Learning Rate Scheduler: `src/modeling_studio/trainers/schedulers_v3.py`

**Warmup + Cosine Decay:**

```python
class WarmupCosineScheduler:
    """
    LR Profile (2500 steps, 500 warmup):
        Step 0:    lr = 0
        Step 250:  lr = base_lr * 0.5
        Step 500:  lr = base_lr (peak)
        Step 1500: lr = ~base_lr * 0.5
        Step 2500: lr = base_lr * 0.01 (min_lr)
    """

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio

    def get_lr(self):
        step = self.last_epoch

        if step < self.warmup_steps:
            # Linear warmup
            return [base_lr * (step / self.warmup_steps)
                    for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * factor)
                    for base_lr in self.base_lrs]
```

---

## 7. Training Script Implementation

### 7.1 Script Location

**File:** `scripts/train_v3_phase0_5.py`

**Status:** To be created (implementation plan Issue 5.4.1)

### 7.2 Script Structure

```python
#!/usr/bin/env python3
"""
Phase 0.5 Enhanced Healing Training Script for ModernBERT v3

Training Strategy:
    - Freeze: L1-18 (Foundation + Core bands)
    - Train: L19-28 (SEMANTIC + Family bands), Hub tokens
    - LR: Zipper strategy with L23 at maximum plasticity
    - Data: Enhanced healing (SST-2, CoNLL, MNLI, SQuAD, STS-B)

Usage:
    python scripts/train_v3_phase0_5.py \
        --config configs/training/multitask/stage_v3_phase0_5_enhanced.yaml \
        --model-path checkpoints/v3_initialized \
        --output-dir outputs/v3_phase0_5

Modes:
    --dry-run     Validate configuration without training
    --smoke-test  Run 10 steps to verify pipeline
    --debug       Run 5 steps with verbose logging
"""

import argparse
import logging
import sys
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

# Local imports
from modeling_studio.models.modernbert_v3 import ModernBERTv3Model
from modeling_studio.trainers.freezing_v3 import LayerFreezer, LayerBand, TrainingPhase
from modeling_studio.trainers.zipper_lr_v3 import create_zipper_optimizer, ZipperLRConfig
from modeling_studio.trainers.schedulers_v3 import WarmupCosineScheduler
from modeling_studio.trainers.gradient_utils_v3 import GradientClipper, GradientClipConfig
from modeling_studio.trainers.gradient_utils_v3 import InterfaceGradientMonitor
from modeling_studio.trainers.gradient_masking_v3 import setup_hub_token_gradient_masking
from modeling_studio.data.collators_v3 import create_v3_collator

logger = logging.getLogger(__name__)
```

### 7.3 Configuration Dataclass

```python
@dataclass
class Phase05Config:
    """Configuration for Phase 0.5 training."""

    # Model
    model_path: str = ""
    model_config: str = "configs/model/encoder/modernbert_v3_ultra.yaml"

    # Training
    max_steps: int = 2500
    warmup_steps: int = 500
    eval_steps: int = 250
    save_steps: int = 500
    logging_steps: int = 50

    # Batch
    train_batch_size: int = 32
    eval_batch_size: int = 64
    gradient_accumulation_steps: int = 1

    # Optimizer
    base_lr: float = 3e-5
    weight_decay: float = 0.01
    zipper_preset: str = "phase_0.5_healing"

    # Gradient
    max_grad_norm: float = 1.0
    interface_grad_clip: float = 0.5

    # Data
    healing_data_config: str = "configs/data/multitask/healing_enhanced.yaml"

    # Output
    output_dir: str = "outputs/v3_phase0_5"

    # Logging
    use_wandb: bool = True
    wandb_project: str = "modernbert-v3"
    wandb_run_name: str = "phase0_5_healing"

    # Device
    device: str = "cuda"
    bf16: bool = True
    seed: int = 42
```

### 7.4 Training Loop Skeleton

```python
def train_phase_0_5(model, train_loader, val_loader, config):
    """Execute Phase 0.5 training loop."""

    device = torch.device(config.device)
    model = model.to(device)

    # =========================================
    # 1. Setup Layer Freezing
    # =========================================
    freezer = LayerFreezer(model)
    freezer.configure_for_phase(TrainingPhase.PHASE_0_5)

    frozen_count, trainable_count = freezer.get_param_counts()
    logger.info(f"Frozen: {frozen_count:,} | Trainable: {trainable_count:,}")

    # =========================================
    # 2. Setup Hub Token Gradient Masking
    # =========================================
    hub_grad_manager = setup_hub_token_gradient_masking(
        model,
        freeze_original_vocab=True,
        train_hub_tokens=["[EMO]", "[MEM]", "[REL]", "[TASK]"],
    )

    # =========================================
    # 3. Setup Zipper LR Optimizer
    # =========================================
    optimizer = create_zipper_optimizer(
        model,
        preset=config.zipper_preset,
        base_lr=config.base_lr,
    )

    # =========================================
    # 4. Setup Scheduler
    # =========================================
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=config.max_steps,
        min_lr_ratio=0.01,
    )

    # =========================================
    # 5. Setup Gradient Clipping
    # =========================================
    grad_clipper = GradientClipper(model, GradientClipConfig(
        max_grad_norm=config.max_grad_norm,
        per_layer_clip=True,
        interface_clip=config.interface_grad_clip,
    ))

    interface_monitor = InterfaceGradientMonitor(interface_layer=23)

    # =========================================
    # 6. Training Loop
    # =========================================
    model.train()
    global_step = 0
    best_val_loss = float("inf")

    pbar = tqdm(total=config.max_steps, desc="Phase 0.5")
    train_iter = iter(train_loader)

    while global_step < config.max_steps:
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Forward
        with torch.cuda.amp.autocast(dtype=torch.bfloat16 if config.bf16 else torch.float32):
            outputs = model(**batch)
            loss = outputs.loss

        # Backward
        loss.backward()

        # Gradient operations
        grad_norm = grad_clipper.clip_gradients()
        interface_stats = interface_monitor.log_step(model, global_step)

        # Optimizer step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Logging
        global_step += 1
        pbar.update(1)

        if global_step % config.logging_steps == 0:
            log_training_step(loss, grad_norm, interface_stats, scheduler, global_step)

        # Evaluation
        if global_step % config.eval_steps == 0:
            val_loss = evaluate(model, val_loader, device, config)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, scheduler, global_step,
                               config.output_dir / "best_model")

        # Checkpointing
        if global_step % config.save_steps == 0:
            save_checkpoint(model, optimizer, scheduler, global_step,
                           config.output_dir / f"checkpoint-{global_step}")

    pbar.close()
    return {"best_val_loss": best_val_loss, "total_steps": global_step}
```

### 7.5 Evaluation Function

```python
def evaluate(model, val_loader, device, config):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            with torch.cuda.amp.autocast(dtype=torch.bfloat16 if config.bf16 else torch.float32):
                outputs = model(**batch)
                total_loss += outputs.loss.item()

            num_batches += 1

    model.train()
    return total_loss / max(num_batches, 1)
```

### 7.6 Checkpoint Management

```python
def save_checkpoint(model, optimizer, scheduler, step, path):
    """Save training checkpoint."""
    path.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "phase": "0.5",
    }, path / "checkpoint.pt")

    # Save model config
    if hasattr(model, "config"):
        model.config.save_pretrained(path)

    logger.info(f"Checkpoint saved: {path}")


def load_checkpoint(model, optimizer, scheduler, path):
    """Load training checkpoint."""
    ckpt = torch.load(path / "checkpoint.pt")

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return ckpt["step"]
```

---

## 8. Execution Workflow

### 8.1 Pre-Training Checklist

```bash
# 1. Verify v3 model is initialized
ls -la checkpoints/v3_initialized/
# Expected: config.json, model.safetensors

# 2. Verify healing data exists
ls -la data/healing/
# Expected: healing_enhanced.jsonl (or per-task files)

# 3. Check GPU availability
nvidia-smi
# Expected: GPU with 16GB+ VRAM

# 4. Verify dependencies
python -c "import torch; print(torch.cuda.is_available())"
python -c "from flash_attn import flash_attn_func; print('Flash Attention OK')"
```

### 8.2 Data Preparation

```bash
# Prepare enhanced healing data (if not exists)
python scripts/prepare_healing_data_enhanced.py \
    --output data/healing/healing_enhanced.jsonl \
    --seed 42 \
    --validate

# Verify output
wc -l data/healing/healing_enhanced.jsonl
# Expected: 12000
```

### 8.3 Dry Run (Configuration Validation)

```bash
python scripts/train_v3_phase0_5.py \
    --config configs/training/multitask/stage_v3_phase0_5_enhanced.yaml \
    --model-path checkpoints/v3_initialized \
    --output-dir outputs/v3_phase0_5 \
    --dry-run
```

**Expected Output:**

```
============================================================
Phase 0.5 Dry Run Validation
============================================================
Configuration: configs/training/multitask/stage_v3_phase0_5_enhanced.yaml

Model:
  Layers: 28
  Hidden: 768
  Hub tokens: 4

Data:
  Healing samples: 12000
  Tasks: sentiment, ner, nli, qa, similarity

Training:
  Max steps: 2500
  Warmup: 500
  Batch size: 32
  Effective batch: 32

Layer Freezing:
  Frozen (L1-18): 100,234,567 params
  Trainable (L19-28): 50,123,456 params

Learning Rate:
  Strategy: zipper
  Interface (L23): 5e-05
  Family (L24-28): graduated decay

DRY RUN PASSED
============================================================
```

### 8.4 Smoke Test (Pipeline Validation)

```bash
python scripts/train_v3_phase0_5.py \
    --config configs/training/multitask/stage_v3_phase0_5_enhanced.yaml \
    --model-path checkpoints/v3_initialized \
    --output-dir outputs/v3_phase0_5_smoke \
    --smoke-test
```

**Expected Output:**

```
============================================================
Phase 0.5 Smoke Test (10 steps)
============================================================
Step 1/10: loss=5.234, lr=1.00e-05, grad_norm=0.45
Step 2/10: loss=5.198, lr=2.00e-05, grad_norm=0.52
...
Step 10/10: loss=4.876, lr=1.00e-04, grad_norm=0.61

SMOKE TEST PASSED
  Initial loss: 5.234
  Final loss: 4.876
  Loss decreased: YES
  No NaN: YES
  Interface stable: YES
============================================================
```

### 8.5 Full Training

```bash
python scripts/train_v3_phase0_5.py \
    --config configs/training/multitask/stage_v3_phase0_5_enhanced.yaml \
    --model-path checkpoints/v3_initialized \
    --output-dir outputs/v3_phase0_5 \
    --wandb-run-name "phase0_5_enhanced_v1"
```

### 8.6 Resume from Checkpoint

```bash
python scripts/train_v3_phase0_5.py \
    --config configs/training/multitask/stage_v3_phase0_5_enhanced.yaml \
    --resume-from outputs/v3_phase0_5/checkpoint-1500 \
    --output-dir outputs/v3_phase0_5
```

---

## 9. Monitoring & Validation

### 9.1 W&B Metrics to Track

| Metric | Description | Target |
|--------|-------------|--------|
| `train/loss` | Training loss | Decreasing |
| `eval/loss` | Validation loss | < 2.0 |
| `train/learning_rate` | Current LR | Peak at step 500 |
| `train/grad_norm` | Global gradient norm | < 5.0 |
| `interface/l23_grad_norm` | L23 gradient norm | < 3.0 |
| `interface/grad_ratio` | L23/L22 gradient ratio | 0.5-2.0 |
| `hub/emo_grad_norm` | [EMO] gradient norm | > 0 |
| `hub/mem_grad_norm` | [MEM] gradient norm | > 0 |
| `hub/rel_grad_norm` | [REL] gradient norm | > 0 |
| `hub/task_grad_norm` | [TASK] gradient norm | > 0 |

### 9.2 Verification Checkpoints

At steps 500, 1000, 1500, 2000, 2500, run verification:

```python
def verify_healing_progress(model, step):
    """Verify healing is progressing correctly."""

    checks = {}

    # 1. Interface activation similarity
    sim = compute_interface_similarity(model)
    checks["interface_similarity"] = sim
    checks["interface_ok"] = sim > 0.7

    # 2. Hub token dispersion
    disp = compute_hub_dispersion(model)
    checks["hub_dispersion"] = disp
    checks["hub_ok"] = disp > 0.2

    # 3. Embedding stability
    stab = compute_embedding_stability(model)
    checks["embedding_stability"] = stab
    checks["embedding_ok"] = stab > 0.9

    return checks
```

### 9.3 Expected Training Curves

```
Loss Curve:
   5.5 ─┼─●
       │   ╲
   4.5 ─┼────●
       │       ╲
   3.5 ─┼────────●
       │           ╲
   2.5 ─┼────────────●──
       │               ╲──────●
   1.5 ─┼──────────────────────────
       │
       └──────────────────────────────
          500  1000  1500  2000  2500
                    Steps

Interface Gradient Ratio:
   3.0 ─┼─●
       │  ╲
   2.0 ─┼───●
       │     ╲
   1.5 ─┼──────●
       │        ╲
   1.0 ─┼─────────●────●────●────●
       │
       └──────────────────────────────
          500  1000  1500  2000  2500
                    Steps
```

---

## 10. Troubleshooting Guide

### 10.1 Common Issues

#### Issue: NaN Loss

**Symptoms:**

- Loss becomes NaN after few steps
- Gradient norm explodes

**Causes:**

- Learning rate too high at interface
- Numerical instability in attention

**Solutions:**

```bash
# Reduce interface LR
--learning-rate 2e-5

# Enable tighter gradient clipping
# In config: interface_clip: 0.3

# Switch from bf16 to fp32
--no-bf16
```

#### Issue: Loss Not Decreasing

**Symptoms:**

- Loss stays flat or increases
- Validation loss matches training loss

**Causes:**

- Frozen layers include trainable bands
- Data loading issue
- Wrong loss function

**Solutions:**

```python
# Verify freezing
for name, param in model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")

# Verify data
for batch in train_loader:
    print(batch.keys())
    print(batch["input_ids"].shape)
    break
```

#### Issue: Hub Tokens Not Learning

**Symptoms:**

- Hub token embeddings don't change
- Hub gradient norms are zero

**Causes:**

- Gradient masking not registered
- Hub tokens not in tokenizer

**Solutions:**

```python
# Verify gradient masking
if hub_grad_manager.is_setup():
    print("Hook registered")
else:
    print("Hook NOT registered - problem!")

# Verify hub tokens in tokenizer
print(tokenizer.additional_special_tokens)
# Expected: ['[EMO]', '[MEM]', '[REL]', '[TASK]']
```

#### Issue: Interface Gradient Explosion

**Symptoms:**

- L23 gradient norm >> L22 gradient norm
- Training unstable

**Causes:**

- Interface layer receiving unexpected activations
- LR too high at L23

**Solutions:**

```bash
# Reduce interface LR
# In config: layer_23: 3e-5 (instead of 5e-5)

# Add gradient norm clipping specifically for L23
# In config: interface_clip: 0.3
```

### 10.2 Diagnostic Commands

```python
# Check model parameter status
def diagnose_model(model):
    for name, param in model.named_parameters():
        if "layer" in name:
            layer_num = int(name.split(".")[2])  # Extract layer number
            print(f"Layer {layer_num}: {name.split('.')[-1]} - "
                  f"requires_grad={param.requires_grad}")

# Check embedding gradient mask
def diagnose_embedding_gradients(model):
    emb_weight = model.embeddings.word_embeddings.weight
    print(f"Embedding shape: {emb_weight.shape}")
    print(f"Requires grad: {emb_weight.requires_grad}")

    # Check if hook is registered
    if hasattr(emb_weight, "_backward_hooks"):
        print(f"Hooks registered: {len(emb_weight._backward_hooks)}")

# Check hub token positions
def diagnose_hub_tokens(tokenizer):
    for token in ["[EMO]", "[MEM]", "[REL]", "[TASK]"]:
        if token in tokenizer.get_vocab():
            print(f"{token}: ID={tokenizer.get_vocab()[token]}")
        else:
            print(f"{token}: NOT FOUND")
```

---

## 11. File Inventory

### 11.1 Scripts

| File | Purpose | Status |
|------|---------|--------|
| `scripts/train_v3_phase0_5.py` | Phase 0.5 training script | To Create |
| `scripts/prepare_healing_data.py` | Basic healing data prep | Created |
| `scripts/prepare_healing_data_enhanced.py` | Enhanced healing data prep | Created |
| `scripts/initialize_v3_from_v2.py` | v2→v3 initialization | Created |

### 11.2 Configurations

| File | Purpose | Status |
|------|---------|--------|
| `configs/training/multitask/stage_v3_phase0_5_enhanced.yaml` | Training config | Created |
| `configs/data/multitask/healing_enhanced.yaml` | Data config | Created |
| `configs/data/multitask/healing_datasets.yaml` | Basic data config | Created |
| `configs/model/encoder/modernbert_v3_ultra.yaml` | Model config | To Verify |

### 11.3 Source Modules

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `src/modeling_studio/trainers/freezing_v3.py` | Layer freezing | `LayerFreezer`, `LayerBand`, `TrainingPhase` |
| `src/modeling_studio/trainers/zipper_lr_v3.py` | Zipper LR | `ZipperLRConfig`, `create_zipper_optimizer` |
| `src/modeling_studio/trainers/gradient_utils_v3.py` | Gradient ops | `GradientClipper`, `InterfaceGradientMonitor` |
| `src/modeling_studio/trainers/gradient_masking_v3.py` | Hub masking | `EmbeddingGradientHook`, `setup_hub_token_gradient_masking` |
| `src/modeling_studio/trainers/schedulers_v3.py` | LR schedulers | `WarmupCosineScheduler` |
| `src/modeling_studio/data/collators_v3.py` | Data collators | `V3MultiTaskCollator`, `create_v3_collator` |
| `src/modeling_studio/models/modernbert_v3.py` | v3 model | `ModernBERTv3Model`, `ModernBERTv3Config` |
| `src/modeling_studio/models/hub_tokens.py` | Hub tokens | `HUB_TOKEN_IDS`, `HUB_TOKENS` |

### 11.4 Output Artifacts

| Path | Contents |
|------|----------|
| `outputs/v3_phase0_5/best_model/` | Best checkpoint by val loss |
| `outputs/v3_phase0_5/checkpoint-{step}/` | Step checkpoints |
| `outputs/v3_phase0_5/training_log.json` | Training metrics |
| `data/healing/healing_enhanced.jsonl` | Prepared healing data |

---

## 12. Acceptance Criteria

### 12.1 Pre-Training Criteria

- [ ] v3 model initialized from v2 checkpoint
- [ ] Model has 28 layers, 50372 vocab size
- [ ] Hub tokens exist in tokenizer at positions 50368-50371
- [ ] Healing data prepared (12,000 samples across 5 tasks)
- [ ] Configuration files validated

### 12.2 Training Criteria

- [ ] Training completes 2,500 steps without crashes
- [ ] No NaN losses during training
- [ ] Loss decreases from ~5.0 to < 2.0
- [ ] Gradient norms stay < 5.0
- [ ] Interface gradient ratio stabilizes to ~1.0

### 12.3 Post-Training Criteria

- [ ] Best model checkpoint saved
- [ ] Interface activation similarity > 0.8
- [ ] Embedding space stability > 0.95
- [ ] Hub token dispersion > 0.3
- [ ] Validation loss < 2.0
- [ ] All task heads functional (sentiment, NER, NLI, QA, similarity)

### 12.4 Quality Gates

| Gate | Metric | Threshold | Action if Failed |
|------|--------|-----------|------------------|
| G1 | Training loss @ step 500 | < 4.5 | Check LR, restart |
| G2 | Interface similarity @ step 1000 | > 0.6 | Increase replay ratio |
| G3 | Hub dispersion @ step 1500 | > 0.2 | Check gradient masking |
| G4 | Val loss @ step 2000 | < 2.5 | Continue, may need more steps |
| G5 | Final val loss | < 2.0 | Training success |

---

## Appendix A: Quick Reference Commands

```bash
# Full pipeline
python scripts/prepare_healing_data_enhanced.py --output data/healing/healing_enhanced.jsonl --validate
python scripts/train_v3_phase0_5.py --config configs/training/multitask/stage_v3_phase0_5_enhanced.yaml --model-path checkpoints/v3_initialized --output-dir outputs/v3_phase0_5

# Dry run
python scripts/train_v3_phase0_5.py --dry-run

# Smoke test
python scripts/train_v3_phase0_5.py --smoke-test

# Resume
python scripts/train_v3_phase0_5.py --resume-from outputs/v3_phase0_5/checkpoint-1500

# Debug mode
python scripts/train_v3_phase0_5.py --debug --max-steps 5
```

---

## Appendix B: Expected Timeline

| Step | Duration | Cumulative |
|------|----------|------------|
| Data preparation | 5 min | 5 min |
| Model loading | 2 min | 7 min |
| Warmup (500 steps) | 5 min | 12 min |
| Training (2000 steps) | 20 min | 32 min |
| Final evaluation | 3 min | 35 min |
| **Total** | **35 min** | - |

*Timeline based on A100 GPU with batch size 32.*

---

**Document Version:** 1.0
**Last Updated:** December 2025
**Next Phase:** Phase 1 Multi-Task Training (after Phase 0.5 completion)
