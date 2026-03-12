<div align="center">

# FamilyOS UltraBERT v4

### Production Multi-Task Encoder for Family AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Flash Attention 2](https://img.shields.io/badge/Flash_Attention-2-orange.svg)](https://github.com/Dao-AILab/flash-attention)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/familyos-ultrabert)](https://pypi.org/project/familyos-ultrabert/)

**22 layers | 12 task heads | 149M params | 20ms inference | 89% FCCS**

[Quick Start](#-quick-start) | [Architecture](#-architecture) | [Benchmarks](#-benchmarks) | [Installation](#-installation)

</div>

---

## Overview

**FamilyOS UltraBERT v4** is a production-ready multi-task NLP encoder for family assistant AI. It performs 12 NLU tasks in a single forward pass with ~20ms latency.

```bash
pip install familyos-ultrabert
```

```python
from familyos_ultrabert import Client
client = Client()
result = client.analyze("Mom called about grandma's birthday!")
print(result.emotions)  # ["joy", "excitement"]
print(result.safety)    # "GREEN"
```

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           FamilyOS UltraBERT v4 Architecture                            │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │  v4 ENCODER (Production) - 149M params                                          │    │
│  │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                          │    │
│  │  ModernBERT-base (22 layers, 768-dim, Flash Attention 2, RoPE)                  │    │
│  │                                                                                 │    │
│  │  12 Task Heads:                                                                 │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │    │
│  │  │   NER    │ │ Emotions │ │ Sentiment│ │  Safety  │ │   NLI    │ │Embedding │  │    │
│  │  │ General  │ │  (44)    │ │   (5)    │ │ (4-band) │ │   (3)    │ │  (768d)  │  │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │    │
│  │  │   NER    │ │ Temporal │ │ Relation │ │  Intent  │ │  Ingress │ │  Safety  │  │    │
│  │  │  Family  │ │   (7)    │ │  (15)    │ │   (8)    │ │   (6)    │ │ Generic  │  │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                         │
│  Total: 149M params │ Edge-friendly (~1GB VRAM inference)                              │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 📊 Model Summary (v4)

| Component         | Params | Pre-trained | Status        | Use Case                                 |
|-------------------|--------|-------------|---------------|------------------------------------------|
| **v4 Encoder**    | 149M   | ModernBERT  | ✅ Production | Classification, NER, embeddings, safety  |
| **Full Model**    | 149M   | Yes         | ✅ Production | End-to-end family AI                     |

### 🎯 Key Capabilities (v4)

| # | Capability         | Type        | Output         | Head     |
|---|-------------------|-------------|---------------|----------|
| 1 | `ner_general`     | Span        | 4 entities    | Encoder  |
| 2 | `ner_family`      | Span        | 10 entities   | Encoder  |
| 3 | `sentiment`       | Sequence    | 5 classes     | Encoder  |
| 4 | `emotions`        | Multi-label | 44 emotions   | Encoder  |
| 5 | `safety_generic`  | Multi-label | 8 types       | Encoder  |
| 6 | `safety_familyos` | Sequence    | 4 bands       | Encoder  |
| 7 | `nli`             | Pair        | 3 classes     | Encoder  |
| 8 | `embedding`       | Vector      | 768-dim       | Encoder  |
| 9 | `temporal`        | Span        | 6 time types  | Encoder  |
| 10| `relation`        | Pair        | 15 types      | Encoder  |
| 11| `intent`          | Sequence    | 8 classes     | Encoder  |
| 12| `ingress`         | Sequence    | 12 domains    | Encoder  |

---

## 🧠 v4 Multi-Task Encoder

The production encoder powering FamilyOS classification, NER, embeddings, safety detection, and all core tasks.

### Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  ModernBERT-base v4 (Production)                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  Base: answerdotai/ModernBERT-base                                          │
│  Layers: 22 │ Hidden: 768 │ Heads: 12 │ Params: 149M                        │
│  Context: 8192 tokens │ Flash Attention 2 │ RoPE Positional Encoding        │
├─────────────────────────────────────────────────────────────────────────────┤
│  12 Task-Specific Heads (~6M params total)                                  │
│                                                                             │
│  Sequence Classification:                                                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │Sentiment │ │ Emotions │ │ Safety   │ │  Intent  │ │  Ingress │           │
│  │ (5-cls)  │ │(44 multi)│ │(4-band)  │ │  (8-cls) │ │  (6-cls) │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
│                                                                             │
│  Token Classification:                                                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                                     │
│  │   NER    │ │   NER    │ │ Temporal │                                     │
│  │ General  │ │  Family  │ │  (7 BIO) │                                     │
│  └──────────┘ └──────────┘ └──────────┘                                     │
│                                                                             │
│  Pair Classification:                                                       │
│  ┌──────────┐ ┌──────────┐                                                  │
│  │   NLI    │ │ Relation │     Dense Embeddings:                            │
│  │  (3-cls) │ │ (15-cls) │     ┌──────────┐                                 │
│  └──────────┘ └──────────┘     │ 768-dim  │                                 │
│                                └──────────┘                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Training Pipeline

| Stage | Description | Data | Output |
|-------|-------------|------|--------|
| **Stage A** | Generic multi-task pretraining | CoNLL, SST-2, GoEmotions, MNLI, etc. | 7 heads trained |
| **Stage B** | FamilyOS domain adaptation | FamilyOS unified shards + 15% replay | 12 heads trained |

### Key Features

- **Flash Attention 2** — 2x speedup on A100/H100
- **EMA Checkpointing** — +0.8-1.5 pt consistent improvement
- **Head-wise Learning Rates** — Encoder 2e-5, heads 1e-4
- **Uncertainty Weighting** — Auto-balanced multi-task loss
- **Safety Oversampling** — CRISIS 20x, RED 5x for recall ≥98%

### Checkpoint Status

| Checkpoint         | Step   | Status        | Weighted Score |
|--------------------|--------|--------------|----------------|
| `checkpoint-18000` | 18,000 | Production | **90.58%**     |

### Retrieval Embedding Head: AgreementGatedHeadV2

The old generic embedding path has been replaced by `AgreementGatedHeadV2`, the retrieval head used by the released `distil_stage_b_bestema` model.

Design principle:

> Keep masked mean pooling as the safe anchor, then allow bounded residual refinement only where auxiliary views agree.

```text
             AgreementGatedHeadV2

hidden_states [B, L, 768]
    |
    +---------------------------+
    |                           |
    v                           v
   mode prompt                 masked mean pool
 (query/document)                  e_mean
    |
    v
  normalized token states
    |
   +------+------+------+------+------+
   |      |      |      |      |      |
   v      v      v      v      v      v
  CLS   latent  max    role  temporal safety
 pool   cross   pool   view    view   salience
 e_cls  attn    e_max  e_role  e_temp features
    \      |      |      |      /
     \-----+------+------+-----/
        project to 768-d
        |
        v
    concat([e_cls, e_lat, e_max, e_role, e_temp])
        |
         refine_mlp
        |
        v
      e_refined = e_mean + delta
        |
         agreement features across 6 views
    (cosines, norm ratios, salience stats, mode bit)
        |
        v
       gate_mlp -> gate_expand -> sigmoid
        |
        v
      e_out = e_mean + g * (e_refined - e_mean)
        |
        v
     layer norm + L2 normalization
        |
        v
       final embedding [B, 768]
```

What changed in the release embedding head:

- adds lightweight query/document asymmetry via mode prompts
- uses six views instead of a single pooled representation
- computes a per-dimension vector gate instead of applying an unconstrained projection
- keeps the original `get_embedding(...)` API intact while adding:
  - `get_query_embedding(...)`
  - `get_document_embedding(...)`

### Retrieval Release Checkpoint

| Checkpoint | Purpose | Head | Status |
|------------|---------|------|--------|
| `distil_stage_b_bestema` | Hosted retrieval release | `AgreementGatedHeadV2` | ✅ Released as `encoder/v2/fp32/` |

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU training)
- 8GB+ VRAM (16GB+ recommended)

### Quick Install

```bash
# Clone repository
git clone https://github.com/Pkansagra-hub/Family_osModernBERT.git
cd Family_osModernBERT

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Optional: Flash Attention for A100/H100 (2x speedup)
pip install flash-attn --no-build-isolation
```

### Verify Installation

```bash
# Run tests
pytest tests/ -v

# Check model loads correctly
python -c "from modeling_studio.models import ModernBertMultiTaskModel; print('✅ Installation successful!')"
```

---

## Safety System

FamilyOS implements a **4-band safety classification** system designed for family contexts:

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                     │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐                      │
│   │  GREEN   │    │  AMBER   │    │   RED    │    │  CRISIS  │                      │
│   │   ✓ ✓   │    │    ⚠     │    │    ⛔   │    │   🚨🚨  │                      │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘                      │
│                                                                                     │
│   Safe content     Needs review    Escalate        Immediate                        │
│   Normal flow      Flag for K1     to supervisor   intervention                     │
│                                                                                     │
│   Examples:        Examples:       Examples:       Examples:                        │
│   • Daily logs     • Stress        • Persistent    • Self-harm                      │
│   • Happy moments  • Frustration     sadness       • Suicide                        │
│   • Routines       • Mild concern  • Isolation       ideation                       │
│                                    • Hopelessness  • Harm to                        │
│                                                      others                         │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Safety Subcategories (12 Types)

| Band | Subcategories |
|------|--------------|
| GREEN | `none` |
| AMBER | `stress`, `mild_sadness`, `frustration`, `health_mention` |
| RED | `persistent_sadness`, `isolation`, `hopelessness`, `substance` |
| CRISIS | `self_harm_ideation`, `suicide_ideation`, `harm_to_others`, `abuse_disclosure` |

### Cultural Awareness

The model is trained to understand **Indian English patterns** and avoid false positives on cultural expressions:

```python
# These are recognized as normal venting, NOT crisis
"I'm dying of laughter!"       # → GREEN (hyperbole)
"Kill me now, so embarrassing" # → GREEN (expression)
"This traffic is killing me"   # → GREEN (metaphor)
```

---

## Benchmarks

### v4.0.1 Core Performance

| Capability | Metric | Score | Latency P95 |
|------------|--------|-------|-------------|
| NER General | F1 | **95.2%** | 19.4ms |
| NER Family | F1 | **80.0%** | 23.4ms |
| Temporal | F1 | **100.0%** | 20.5ms |
| Intent | Accuracy | **90.0%** | 20.1ms |
| Ingress | Accuracy | **100.0%** | 19.1ms |
| Emotions | Hit Rate | **95.3%** | ~7ms |
| Sentiment | Direction Acc | **100.0%** | 18.5ms |
| Safety | Band Accuracy | **87.5%** | 18.7ms |
| Safety | CRISIS Recall | **100%** | - |

### Holistic Coherence Benchmark (FCCS)

Cross-head consistency metrics measuring how well the 12 heads work together:

| Metric | Score | Threshold | Description |
|--------|-------|-----------|-------------|
| Head Agreement (HAS) | 81.56% | 70% | Sentiment-emotion valence alignment |
| Entity Grounding (EGS) | 85.96% | 65% | Relations grounded in detected entities |
| Safety-Emotion (SEC) | 99.36% | 75% | Distress emotions trigger elevated safety |
| Temporal Completeness (TCS) | 99.72% | 80% | Reminder intents have temporal info |
| Family Context (FCS) | 79.80% | 35% | Richness of family understanding |
| Intent-Ingress (IIC) | 86.00% | 55% | Intent aligns with ingress domain |
| Ingress-Emotion (IEC) | 92.80% | 60% | Domain has expected emotions |
| **FCCS Overall** | **89.12%** | 70% | Weighted holistic score |

**Performance:** 20.3ms/sample for all 12 heads in single forward pass.

### Embedding Quality (AgreementGatedHeadV2 release)

These are the current retrieval-release numbers from `scripts/evaluation/evaluate_retrieval_checkpoints.py` on `checkpoints/distil_stage_b_bestema`.

| Split | Queries | Recall@1 | Recall@5 | Recall@10 | nDCG@10 | MRR | Triplet Accuracy | Selection Score |
|-------|---------|----------|----------|-----------|---------|-----|------------------|-----------------|
| Dev | 300 | **0.8800** | **0.9933** | **1.0000** | **0.9492** | **0.9319** | **0.7975** | **0.9134** |
| Holdout | 300 | **0.8833** | **0.9933** | **1.0000** | **0.9499** | **0.9329** | **0.8175** | **0.9170** |

Release readout:

- `Recall@5` is effectively saturated on both splits
- holdout slightly exceeds dev on `Recall@1`, `MRR`, and selection score
- the released embedding head is optimized for FamilyOS retrieval rather than generic STS-only quality

### Throughput

| Metric | Value |
|--------|-------|
| Throughput | 93.6 inferences/sec |
| P95 Latency | ~20ms |
| Average Latency | 10.7ms |

---

## Task Reference

### Capability Details

| Capability | Hub | Type | Labels | Description |
|------------|-----|------|--------|-------------|
| `ner_general` | - | Span | PER, ORG, LOC, MISC | Standard entities (GlobalPointer) |
| `ner_family` | - | Span | PERSON, KINSHIP, NICKNAME, PET, HOME_LOC, FAMILY_EVENT, ROUTINE, TRADITION, MILESTONE, HEIRLOOM | Family entities (GlobalPointer) |
| `sentiment` | [EMO] | Sequence | very_negative, negative, neutral, positive, very_positive | 5-point sentiment scale |
| `emotions` | [EMO] | Multi-label | neutral, joy, sadness, anger, fear, surprise, love, disgust, admiration, amusement, approval, caring, excitement, gratitude, optimism, pride, relief, contentment, hope, tenderness, annoyance, disappointment, disapproval, embarrassment, grief, nervousness, remorse, frustration, overwhelmed, emptiness, nostalgia, protectiveness, togetherness, longing, warmth, playfulness, celebration, belonging, parental_pride, parental_guilt, patience, worry, bittersweet, homesickness | 44 FamilyOS emotions |
| `safety_generic` | [EMO] | Multi-label | Standard | Jigsaw toxicity types |
| `safety_familyos` | [EMO] | Hierarchical | GREEN, AMBER, RED, CRISIS | 4-band safety hierarchy |
| `nli` | [REL] | Pair | entailment, neutral, contradiction | Natural language inference |
| `embedding` | [MEM] | Vector | 768-dim | Dense vector representations |
| `temporal` | - | Span | DATE_ABS, DATE_REL, TIME, DURATION, FREQUENCY, AGE | Time expressions (GlobalPointer) |
| `relation` | [REL] | Pair | no_relation, parent_of, child_of, spouse_of, sibling_of, grandparent_of, grandchild_of, aunt_uncle_of, niece_nephew_of, cousin_of, pet_of, friend_of, colleague_of, lives_at, owns | 15 relationship types |
| `intent` | [TASK] | Sequence | log_memory, query_memory, set_reminder, express_feeling, seek_advice, share_news, reflect, other | 8 user intents |
| `ingress` | [TASK] | Sequence | DIARY, TASK, HEALTH, FINANCE, RELATIONSHIP, WORK, META, MEMORY, PLANNING, CELEBRATION, CONCERN, GRATITUDE | 12 domain categories |

### FamilyOS Emotion Schema (44 Classes)

<details>
<summary>Click to expand full emotion list</summary>

**Core Emotions (8)**

- neutral, joy, sadness, anger, fear, surprise, love, disgust

**Positive Emotions (12)**

- admiration, amusement, approval, caring, excitement, gratitude, optimism, pride, relief, contentment, hope, tenderness

**Negative Emotions (10)**

- annoyance, disappointment, disapproval, embarrassment, grief, nervousness, remorse, frustration, overwhelmed, emptiness

**Family-Specific Emotions (14)**

- nostalgia, protectiveness, togetherness, longing, warmth, playfulness, celebration, belonging, parental_pride, parental_guilt, patience, worry, bittersweet, homesickness

</details>

---

## 📁 Project Structure

```
FamilyOS-ModernBERT/
│
├── 📂 configs/                          # Configuration files
│   ├── data/
│   │   └── multitask/                   # Dataset configs
│   │       ├── healing_datasets.yaml
│   │       ├── healing_enhanced.yaml
│   │       └── familyos_unified.yaml
│   ├── model/
│   │   └── encoder/                     # Model configs
│   │       ├── modernbert_v2.yaml       # v2 (22 layers)
│   │       └── modernbert_v3_ultra.yaml # v3 Ultra (28 layers)
│   └── training/
│       └── multitask/                   # Training configs
│           ├── stage_v3_phase0.yaml         # Phase 0 (initialization)
│           ├── stage_v3_phase0_5.yaml       # Phase 0.5 (basic healing)
│           ├── stage_v3_phase0_5_enhanced.yaml  # Enhanced healing (5-task)
│           ├── stage_v3_phase1.yaml         # Phase 1 (multi-task)
│           ├── stage_v3_phase1_a100_80gb.yaml
│           └── stage_v3_phase1_h100.yaml
│
├── 📂 src/modeling_studio/              # Main package
│   ├── 📂 data/                         # Data pipeline
│   │   ├── labels.py                    # Label schemas (12 tasks)
│   │   ├── loaders.py                   # Dataset loaders (v2)
│   │   ├── loaders_v3.py                # Unified dataset loaders (v3)
│   │   ├── tokenization.py              # Tokenization functions
│   │   ├── tokenization_v3.py           # Hub token injection
│   │   ├── collators_v3.py              # v3 collators with hub offsets
│   │   ├── unified_dataset.py           # Unified JSONL dataset
│   │   ├── healing_dataset.py           # Phase 0.5 healing data
│   │   └── augmentation.py              # Data augmentation
│   │
│   ├── 📂 models/                       # Model architecture
│   │   ├── modernbert_v2.py             # v2 model (22 layers)
│   │   ├── modernbert_v3.py             # v3 Ultra (28 layers)
│   │   ├── config_v3.py                 # v3 configuration
│   │   ├── embeddings_v3.py             # v3 embeddings with hub tokens
│   │   ├── encoder_v3.py                # 28-layer encoder
│   │   ├── layers_v3.py                 # v3 transformer layers
│   │   ├── attention_v3.py              # Multi-scale + global hub attention
│   │   ├── ffn_v3.py                    # GELU FFN
│   │   ├── lora_v3.py                   # LoRA for L23-28
│   │   ├── hub_tokens.py                # Hub token definitions
│   │   ├── hub_initialization_v3.py     # Semantic centroid init
│   │   ├── initialization_v3.py         # Function preserving growth
│   │   ├── poolers_v3.py                # Hub token pooler
│   │   ├── pair_encoder_v3.py           # Cross-attention with [REL]
│   │   ├── heads.py                     # Task-specific heads (v2)
│   │   ├── heads_v3.py                  # Hub-aware heads (v3)
│   │   └── losses.py                    # Custom loss functions
│   │
│   ├── 📂 trainers/                     # Training logic
│   │   ├── multitask_trainer.py         # Multi-task trainer (v2)
│   │   ├── trainer_v3.py                # v3 trainer with phase control
│   │   ├── hub_token_trainer.py         # Hub token training utilities
│   │   ├── zipper_lr.py                 # Zipper LR for Phase 0.5
│   │   ├── healing_scheduler.py         # Warmup + cosine for healing
│   │   ├── collators.py                 # Data collators (v2)
│   │   ├── task_sampler.py              # Task sampling strategies
│   │   ├── task_weighting.py            # Uncertainty weighting
│   │   ├── ema.py                       # EMA model
│   │   ├── optimizer.py                 # Layer-wise LR (v3: L19-22 vs L23-28)
│   │   └── callbacks.py                 # Training callbacks
│   │
│   ├── 📂 evaluation/                   # Evaluation
│   │   ├── evaluator.py                 # Evaluation runner
│   │   ├── metrics.py                   # Per-task metrics
│   │   ├── benchmarks.py                # Benchmark suite
│   │   └── forgetting_eval.py           # Catastrophic forgetting
│   │
│   └── 📂 inference/                    # Inference utilities
│       └── unified_inference.py         # Production inference
│
├── 📂 scripts/                          # Training & utility scripts
│   ├── initialize_v3_from_v2.py         # Function preserving growth
│   ├── verify_function_preserving.py    # Verify L1-22 match v2
│   ├── train_v3.py                      # v3 multi-phase training
│   ├── prepare_healing_data.py          # Generate Phase 0.5 data (3-task)
│   ├── prepare_healing_data_enhanced.py # Generate enhanced healing (5-task)
│   ├── validate_unified_data.py         # Validate unified JSONL format
│   ├── evaluate_stage_a.py              # Stage A benchmark evaluation
│   ├── forgetting_eval.py               # Catastrophic forgetting check
│   ├── calibrate_safety.py              # Safety threshold calibration
│   ├── export_model.py                  # Model export (v2)
│   ├── 📂 export_utility/               # v3 export utilities
│   │   ├── lora_merge_v3.py             # Merge LoRA weights
│   │   ├── temperature_calibration_v3.py # Per-head calibration
│   │   ├── export_v3_model.py           # Export v3 model
│   │   └── export_onnx_v3.py            # ONNX with hub tokens
│   └── 📂 agents/                       # Data generation agents
│
├── 📂 data/                             # Data directory
│   ├── public/                          # Public datasets (Stage A)
│   ├── healing/                         # Phase 0.5 healing datasets
│   │   ├── healing_generic.jsonl        # 3-task (SST2, CoNLL, MNLI)
│   │   └── healing_enhanced.jsonl       # 5-task (+SQuAD, +STS-B)
│   └── familyos/                        # FamilyOS-specific data
│       ├── unified/output/              # Unified JSONL shards
│       │   ├── shard_001.jsonl
│       │   ├── shard_002.jsonl
│       │   └── ...
│       ├── emotions/
│       ├── safety/
│       ├── intents/
│       ├── relations/
│       └── temporal/
│
├── 📂 tests/                            # Test suite
├── 📂 docs/                             # Documentation
├── 📂 checkpoints/                      # Model checkpoints
│
├── requirements.txt                     # Python dependencies
├── pyproject.toml                       # Package configuration
├── Makefile                             # Build commands
└── setup_gcp.sh                         # GCP setup script
```

---

## ☁️ Cloud Training

### Google Cloud Platform (A100/H100)

```bash
# 1. Create VM with Deep Learning image
gcloud compute instances create training-vm \
    --zone=us-central1-a \
    --machine-type=a2-highgpu-1g \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --accelerator="type=nvidia-tesla-a100,count=1" \
    --maintenance-policy=TERMINATE

# 2. SSH and setup
gcloud compute ssh training-vm --zone=us-central1-a
git clone https://github.com/Pkansagra-hub/Family_osModernBERT.git
cd Family_osModernBERT
./setup_gcp.sh

# 3. Train
python scripts/train_stage_a.py \
    --config configs/training/multitask/stage_a_a100.yaml
```

### Estimated Costs

| GPU | Time (Stage A) | Cost |
|-----|---------------|------|
| A100 40GB | 2-3 hours | ~$8-12 |
| A100 80GB | 1.5-2 hours | ~$12-16 |
| H100 | 1-1.5 hours | ~$15-22 |

---

## Detailed Benchmark Results (v4.0.1)

> **Evaluation Date:** February 1, 2026
> **Model:** ModernBERT + GlobalPointer Decoder + LabelDescriptionHeads (22 layers, 149M params, 768-dim)
> **Hardware:** NVIDIA RTX 5070 Laptop GPU (CUDA 12.8)

### Summary: All Heads Pass Production Thresholds

| Head | Type | Primary Metric | Score | Status |
|------|------|----------------|-------|--------|
| ner_general | GlobalPointer | F1 | **95.2%** | PASS |
| ner_family | GlobalPointer | F1 | **80.0%** | PASS |
| temporal | GlobalPointer | F1 | **100.0%** | PASS |
| sentiment | Classification | Direction Acc | **100.0%** | PASS |
| emotions | Multi-label | Hit Rate | **95.3%** | PASS |
| safety_familyos | Hierarchical | Band Accuracy | **87.5%** | PASS |
| safety_familyos | Hierarchical | CRISIS Recall | **100%** | CRITICAL PASS |
| intent | LabelDescriptionHead | Accuracy | **90.0%** | PASS |
| ingress | LabelDescriptionHead | Accuracy | **100.0%** | PASS |
| embedding | Vector | Recall@10 | **100%** | PASS |
| relation | Pair | N/A | Trained | READY |
| nli | Pair | N/A | Trained | READY |

### Entity Recognition Results (GlobalPointer Heads)

| Test Category | Test Cases | Issues Resolved | Resolution Rate |
|---------------|------------|-----------------|-----------------|
| Original Issues | 33 | 33/33 | 100% |
| Expanded Issues | 47 | 47/47 | 100% |
| Hard Cases | 14 | 14/14 | 100% |
| **Total** | **80** | **80/80** | **100%** |

#### Quality Issue Categories Resolved

| Category | Examples | Status |
|----------|----------|--------|
| MILESTONE tags verbs | learned, passed, promoted, accepted, started, graduated, bought | ✅ RESOLVED |
| FAMILY_EVENT tags pronouns | our, we, 3, 4, 2, 5 | ✅ RESOLVED |
| HEIRLOOM tags prepositions | old, to, from, in, safe, displayed, contains | ✅ RESOLVED |
| PET tags determiners | the (before pet names) | ✅ RESOLVED |
| PERSON tags verbs | met, asked, called, thinking, went, came, worked, drove | ✅ RESOLVED |
| PERSON tags emotions | anxious, grateful, stressed, excited, worried, happy, sad, proud | ✅ RESOLVED |
| ORG tags common nouns | meeting, email, afternoon, manager, conference, accounting, kitchen, hospital | ✅ RESOLVED |
| Partial entity extraction | Lincoln School, San Francisco, Bella Notte, Johnson & Johnson, Madison Square Garden, L.A. International Airport, MIT | ✅ RESOLVED |
| Time fragments | 3pm, 10am, 12pm, 6pm, 9pm, 8am, 7pm | ✅ RESOLVED |
| Verb forms | learned, working, organized, developed, created, managed | ✅ RESOLVED |
| Complex family contexts | Multi-generational relationships, pets, heirlooms, addresses | ✅ RESOLVED |
| Ambiguous entities | Names vs verbs, organizations vs common nouns, mixed contexts | ✅ RESOLVED |
| Cultural challenges | International names, cultural foods, traditions, family terms | ✅ RESOLVED |

#### Key Improvement: No Post-Processing Required

**V2 UltraBERT** required 10-step post-processing pipeline with 15+ filters to handle garbage entities:

- Verb/emotion filters
- Pronoun filters
- Determiner filters
- Time fragment filters
- Partial span mergers
- Cultural awareness filters

**V4 UltraBERT** with GlobalPointer architecture eliminates all garbage entities at the source:

- Span-based scoring instead of token classification
- Complete entity extraction prevents partial spans
- Context-aware boundaries reduce false positives
- No filters needed - 100% clean output

### Embedding Quality Benchmarks

#### Retrieval Accuracy

| Benchmark | Metric | Score | Status |
|-----------|--------|-------|--------|
| **10 distractors** | Recall@1 | **84.5%** | PASS |
| **100 distractors** | Recall@1 | **75.0%** | PASS |
| **100 distractors** | Recall@5 | **100%** | PASS |
| **100 distractors** | Recall@10 | **100%** | PASS |
| **Triplet Accuracy** | Binary | **70%** | PASS |

#### Advanced Embedding Metrics

| Metric | Score |
|--------|-------|
| **MRR** | 1.0 |
| **NDCG@5** | 1.0 |
| **Precision@1** | 1.0 |
| **Precision@3** | 0.78 |
| **Precision@5** | 0.47 |

---

### Inference Latency Benchmarks (RTX 5070 - v4.0.1)

#### Full Multi-Task Inference

| Metric | Value |
|--------|-------|
| **P95 Latency** | **~20 ms** |
| **Average** | **10.7 ms** |
| **Throughput** | **93.6 inferences/sec** |

#### Per-Head Latency (P95)

| Head | Type | Latency P95 |
|------|------|-------------|
| sentiment | Classification | 18.5 ms |
| safety_familyos | Hierarchical | 18.7 ms |
| ingress | LabelDescriptionHead | 19.1 ms |
| ner_general | GlobalPointer | 19.4 ms |
| intent | LabelDescriptionHead | 20.1 ms |
| temporal | GlobalPointer | 20.5 ms |
| ner_family | GlobalPointer | 23.4 ms |

---

### Embedding Query Performance (RTX 5070)

#### Corpus Indexing

| Metric | Value |
|--------|-------|
| **Embedding Throughput** | **69 docs/sec** |
| **Embedding Dimension** | 768 |

#### Query Latency (1000 doc corpus)

| Metric | Value |
|--------|-------|
| **Average (embed + search)** | **1.86 ms** |
| **P50** | 0.87 ms |
| **P95** | 9.97 ms |

#### Latency Breakdown

| Component | Time | % |
|-----------|------|---|
| Query Embedding | 1.51 ms | 82% |
| Search (1000 docs) | 0.34 ms | 18% |

#### Search Scaling (Pre-computed Embeddings)

| Corpus Size | Search Time |
|-------------|-------------|
| 100 docs | 0.011 ms |
| 500 docs | 0.034 ms |
| 1000 docs | 0.253 ms |
| 5000 docs | 0.464 ms |
| 10000 docs | 0.954 ms |

---

### Robustness Tests

| Test Category | Status |
|---------------|--------|
| Edge Cases (empty, whitespace, special chars) | PASS |
| Unicode/Emoji Handling | PASS |
| Adversarial Inputs | PASS |
| Unicode Normalization | PASS |
| Length Scaling (tiny to very_long) | PASS |

---

### Key Improvements in v4.0.1

| Metric | v4.0.0 | v4.0.1 | Change |
|--------|--------|--------|--------|
| NER General F1 | 73.0% | **95.2%** | +22.2% |
| Temporal F1 | 63.9% | **100.0%** | +36.1% |
| Intent Architecture | Linear | **LabelDescriptionHead** | Zero-shot capable |
| Ingress Architecture | Linear | **LabelDescriptionHead** | Zero-shot capable |
| Throughput | 61.5/sec | **93.6/sec** | +52% |
| Label Expansion | Not supported | **Dynamic API** | Add labels at runtime |

### Key Improvements from V2

| Metric | V2 (checkpoint-18000) | V4.0.1 (UltraBERT) | Change |
|--------|----------------------|----------------|--------|
| Full Inference P95 | 102.25 ms | **~20 ms** | **5x faster** |
| Throughput | 11.3/sec | **93.6/sec** | **8.3x higher** |
| CRISIS Recall | - | **100%** | Guaranteed |
| NER Architecture | Token Classification | **GlobalPointer** | Better spans |
| Intent/Ingress | Static labels | **LabelDescriptionHead** | Zero-shot expandable |

---

### Sample Inference Output (v4.0.1)

```json
{
  "text": "My grandmother called yesterday to remind me about the family reunion next Sunday. I am so excited!",
  "sentiment": "very_positive",
  "sentiment_confidence": 0.9719,
  "emotions": ["joy", "excitement", "togetherness", "warmth"],
  "safety": "GREEN",
  "safety_confidence": 1.0,
  "entities": [
    {"text": "grandmother", "label": "KINSHIP", "score": 0.89}
  ],
  "temporal": [
    {"text": "yesterday", "label": "DATE_REL", "score": 0.92},
    {"text": "next Sunday", "label": "DATE_REL", "score": 0.88}
  ],
  "intent": "share_news",
  "ingress": "CELEBRATION",
  "relations": ["grandparent_of", "grandchild_of"],
  "latency_ms": 18.5
}
```

---

### Production Readiness Assessment (v4.0.1)

| Capability | Status | Notes |
|------------|--------|-------|
| **Safety** | PASS | 87.5% band accuracy, 100% explicit CRISIS recall |
| **Intent (LabelDescriptionHead)** | PASS | 90% accuracy, zero-shot expandable |
| **Ingress (LabelDescriptionHead)** | PASS | 100% accuracy, zero-shot expandable |
| **Emotions** | PASS | 95.3% hit rate, multi-label |
| **Sentiment** | PASS | 100% direction accuracy |
| **NER General (GlobalPointer)** | PASS | 95.2% F1, span extraction |
| **NER Family (GlobalPointer)** | PASS | 80.0% F1, family entities |
| **Temporal (GlobalPointer)** | PASS | 100% F1, time expressions |
| **Embeddings** | PASS | 84.5% R@1 (10d), 100% R@10 (100d) |
| **Throughput** | PASS | 93.6 inferences/sec |
| **Latency** | PASS | P95 < 25ms |
| **Robustness** | PASS | All edge/unicode/adversarial tests |

**Overall Verdict:** **Production Ready** for FamilyOS deployment.

---

## 🧪 Embedding Head Training Recipe

The embedding release was built with the bake-off and Stage B specialization flow implemented in:

- `scripts/training/train_embedding_heads_bakeoff.py`
- `configs/training/embedding_heads_bakeoff.yaml`

### What the trainer actually does

The bake-off trainer loads the shared multi-task checkpoint, freezes the encoder and all non-embedding heads, replaces only the embedding head, and trains multiple candidate retrieval heads under matched conditions.

Core setup:

- source checkpoint: `outputs/globalpointer-unified-v1/checkpoint-8000`
- encoder: frozen
- non-embedding heads: frozen
- trainable module: embedding head only
- hidden size: `768`
- tokenizer max length: `128`

### Candidate heads in the bake-off

- `mean_baseline`
- `residual_mlp_mean`
- `latent_residual`
- `agreement_gated`
- `multi_pool_low_rank`
- `anisotropy_corrected`
- `agreement_gated_v2`

### Shared optimization recipe

From `configs/training/embedding_heads_bakeoff.yaml`:

- learning rate: `2.0e-4`
- encoder learning rate: `0.0`
- weight decay: `0.01`
- Adam betas: `0.9`, `0.999`
- Adam epsilon: `1.0e-8`
- max grad norm: `1.0`
- scheduler: cosine
- warmup steps: `200`
- num epochs: `12`
- batch size: `2048`
- gradient accumulation: `1`
- BF16: enabled
- TF32: enabled
- Flash Attention 2: enabled when available
- EMA: enabled
- early stopping patience: `5`
- curriculum warmup: `2` epochs
- matryoshka dimensions: `[768, 512, 256, 128]`

### Loss recipe

- loss family: InfoNCE / contrastive retrieval loss
- base temperature: `0.07`
- learnable temperature: enabled
- temperature LR: `1.0e-3`
- hard negative weight: `1.5`
- in-batch negatives: enabled

### Training data slices

Configured slices:

- `silver_synthetic`
- `hard_negatives`
- `wrong_person`
- `wrong_time`
- `safety_emotion`
- `query_doc`

The trainer uses per-slice holdout and balanced slice-aware sampling.

Global data settings:

- eval split per slice: `0.15`
- sampler strategy: `balanced`

### Stage A core bake-off profile

Stage A uses:

- `silver_synthetic`
- `hard_negatives`

Stage A slice weights:

- `silver_synthetic: 1.0`
- `hard_negatives: 2.0`

### Stage B specialization profile used for the release head

Stage B config is explicitly retrieval-focused:

- data profile: `stage_b_v2`
- target head: `agreement_gated_v2`
- reuse checkpoint head as-is: `true`
- learning rate: `7.5e-5`
- num epochs: `10`
- warmup steps: `40`
- early stopping patience: `4`

Stage B slices:

- `hard_negatives`
- `wrong_person`
- `wrong_time`
- `safety_emotion`
- `query_doc`

Stage B slice weights:

- `hard_negatives: 1.5`
- `wrong_person: 3.5`
- `wrong_time: 5.0`
- `safety_emotion: 5.0`
- `query_doc: 8.0`

Stage B routing and auxiliary recipe:

- anchor mode defaults to `document`
- positives default to `document`
- negatives default to `document`
- `query_doc` anchors route through `query` mode
- role entropy weight: `0.02`
- temporal entropy weight: `0.02`
- safety entropy weight: `0.02`

Teacher and distillation scaffolding are present in the trainer for future runs, but the base config keeps them disabled by default:

- `teacher.enabled: false`
- `distillation.enabled: false`

### Commands

Run a single head:

```bash
python scripts/training/train_embedding_heads_bakeoff.py \
  --config configs/training/embedding_heads_bakeoff.yaml \
  --head_type agreement_gated_v2
```

Run the full bake-off:

```bash
python scripts/training/train_embedding_heads_bakeoff.py \
  --config configs/training/embedding_heads_bakeoff.yaml \
  --run_all
```

Run a debug slice:

```bash
python scripts/training/train_embedding_heads_bakeoff.py \
  --config configs/training/embedding_heads_bakeoff.yaml \
  --head_type mean_baseline \
  --debug \
  --max_samples 500
```

---

## 🔧 Advanced Configuration

### Head-wise Learning Rates

```yaml
optimizer:
  encoder_lr: 2e-5          # Backbone (slower)
  head_lr: 1e-4             # Classification heads
  token_head_lr: 5e-5       # NER/Temporal heads
  layer_decay: 0.95         # Bottom layers learn slower
```

### Task Sampling Strategies

```yaml
mixing:
  strategy: temperature     # proportional, uniform, sqrt, temperature, uncertainty
  temperature: 2.0          # Higher = more uniform sampling
  max_samples_per_task: 50000
```

### Asymmetric Loss (SOTA Multi-label)

```yaml
heads:
  emotions:
    use_asl: true           # ICCV 2021 SOTA for multi-label
    asl_gamma_neg: 4.0      # Suppress easy negatives
    asl_gamma_pos: 1.0      # Preserve hard positives
    asl_clip: 0.05          # Probability clipping
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [Enhanced Design v3.3](src/modeling_studio/plans/enhanced_design_v3.md) | Complete v3 Ultra architecture specification |
| [Implementation Plan v3](src/modeling_studio/plans/implementation_plan_v3.md) | Detailed v3 implementation guide with wiring |
| [Stage A README](docs/STAGE_A_README.md) | Stage A benchmark documentation |
| [Stage B README](docs/STAGE_B_README.md) | FamilyOS data documentation |
| [Annotation Guidelines](docs/annotation/README.md) | Data annotation standards |
| [Safety Guidelines](docs/annotation/safety_guidelines.md) | Safety classification rules |
| [K0 Module Migration](docs/k0_module_migration.md) | Integration with K0 runtime |
| [Lightning AI Training](docs/LIGHTNING_AI_TRAINING.md) | Cloud training guide |

---

## 🚀 Releases

### Automated Release Process

FamilyOS UltraBERT uses automated releases that exclude model weights (hosted on HuggingFace) for lightweight distributions.

#### Release Workflow

1. **Prepare Release** (Local)

   ```bash
   # Set version and prepare release
   make release-prep VERSION=4.0.0

   # Test package installation
   make release-test VERSION=4.0.0
   ```

2. **Create GitHub Release**
   - Go to [GitHub Releases](https://github.com/Pkansagra-hub/Family_osModernBERT/releases)
   - Click "Create a new release"
   - Tag: `v4.0.0` (with 'v' prefix)
   - Title: `FamilyOS UltraBERT v4.0.0`
   - Description: Copy from generated `RELEASE_NOTES_v4.0.0.md`

3. **Automated Publishing**
   - GitHub Actions automatically builds and publishes to PyPI
   - Release assets are attached to the GitHub release
   - Weights are downloaded from HuggingFace at runtime

#### Package Contents

The released package **excludes weights** for:

- **Smaller size**: ~50MB vs ~700MB with weights
- **Faster installs**: No large model files
- **Automatic updates**: Weights downloaded on first use
- **Security**: No sensitive model artifacts in distribution

**Included in release:**

```text
familyos_ultrabert/
├── __init__.py          # Package initialization
├── client.py            # Main inference client
├── model.py             # Model architecture
├── weights_manager.py   # HuggingFace weight management
├── labels.py            # Label definitions
├── onnx_inference.py    # ONNX runtime support
├── pytorch_inference.py # PyTorch inference
├── runtime.py           # Runtime utilities
├── benchmarks/          # Benchmark suite
├── examples/            # Usage examples
└── tests/               # Test suite
```

**Excluded from release:**

- `weights/` directory (636MB model files)
- Build artifacts and caches
- Development files

#### Installation from PyPI

```bash
# Install the lightweight package
pip install familyos-ultrabert==4.0.0

# First run automatically downloads weights from HuggingFace
python -c "import familyos_ultrabert; client = familyos_ultrabert.Client()"
```

#### Manual Release Process

If you need to release manually:

```bash
# 1. Prepare package
cd familyos_ultrabert
python -m build

# 2. Test installation
pip install dist/familyos_ultrabert-4.0.0-py3-none-any.whl --force-reinstall

# 3. Upload to PyPI
twine upload dist/*
```

---

We welcome contributions! Please see our contribution guidelines for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **ModernBERT** by Answer.AI - Backbone architecture
- **Flash Attention 2** by Dao-AILab - Efficient attention
- **HuggingFace Transformers** - Model infrastructure
- **GoEmotions** by Google - Emotion classification dataset

---

<div align="center">

**FamilyOS UltraBERT v4** - Multi-Task Encoder for Family AI

`pip install familyos-ultrabert`

[Back to Top](#familyos-ultrabert-v4)

</div>
