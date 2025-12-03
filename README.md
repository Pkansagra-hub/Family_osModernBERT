<div align="center">

# 🏠 FamilyOS ModernBERT

### Unified Multi-Task Encoder for Family Assistant Applications

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow)](https://huggingface.co/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*One model. 12 capabilities. Built for families.*

[Features](#-key-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [Training](#-training) • [Documentation](#-documentation)

</div>

---

## 📖 Overview

**FamilyOS ModernBERT** is a unified multi-task encoder that consolidates 9+ separate NLP models into a single, efficient architecture. Built on the modern `answerdotai/ModernBERT-base` backbone, it provides **12 specialized capabilities** for family assistant applications — from emotion detection and entity recognition to safety classification and relationship extraction.

### 🎯 The Problem We Solve

| Before (Model Zoo) | After (Unified) |
|:------------------:|:---------------:|
| 9 separate models | 1 unified model |
| ~4,350 MB memory | ~500 MB memory |
| ~62s load time | ~8s load time |
| ~150ms per query | ~35ms per query |
| 5 different architectures | 1 architecture |

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🧠 12 Capabilities
- **NER General** — 17 BIO entity tags
- **NER Family** — 21 family-specific entities
- **Sentiment** — 5-point scale analysis
- **Emotions** — 44 emotion classes (family-aware)
- **Safety Generic** — 8 toxicity types
- **Safety FamilyOS** — 4-band policy system
- **NLI** — Natural language inference
- **Embeddings** — 768-dim with Matryoshka
- **Temporal** — Time expression extraction
- **Relations** — Family relationship mapping
- **Intent** — User intent classification
- **Ingress** — 12 domain classification

</td>
<td width="50%">

### ⚡ Modern Architecture
- **ModernBERT backbone** — 2T tokens, 8192 context
- **Flash Attention 2** — Optimized for A100/H100
- **RoPE embeddings** — Better position encoding
- **Task adapters** — Efficient fine-tuning
- **EMA checkpointing** — Smoother training
- **Uncertainty weighting** — Auto task balancing

### 🛡️ Family-First Safety
- **4-band system** — GREEN → AMBER → RED → CRISIS
- **12 subcategories** — Fine-grained classification
- **Cultural awareness** — Indian English patterns
- **Keyword override** — Zero false negatives on crisis

</td>
</tr>
</table>

---

## 🏗️ Architecture

```
                                    ┌─────────────────────────────────────┐
                                    │           Input Text                │
                                    │   "Mom took Panda to the park"      │
                                    └─────────────────┬───────────────────┘
                                                      │
                                                      ▼
                              ┌────────────────────────────────────────────────┐
                              │                                                │
                              │             ModernBERT Encoder                 │
                              │         (149M params, 8192 context)            │
                              │                                                │
                              │    ┌──────────────────────────────────────┐    │
                              │    │     22 Transformer Layers            │    │
                              │    │     • Flash Attention 2              │    │
                              │    │     • RoPE Position Embeddings       │    │
                              │    │     • 768 Hidden Dimension           │    │
                              │    └──────────────────────────────────────┘    │
                              │                                                │
                              │    ┌──────────────────────────────────────┐    │
                              │    │     Task Group Adapters (Optional)   │    │
                              │    │     • Token Tasks Adapter            │    │
                              │    │     • Sequence Tasks Adapter         │    │
                              │    │     • Pair Tasks Adapter             │    │
                              │    └──────────────────────────────────────┘    │
                              │                                                │
                              └────────────────────────┬───────────────────────┘
                                                       │
                    ┌──────────────────────────────────┼──────────────────────────────────┐
                    │                                  │                                  │
                    ▼                                  ▼                                  ▼
        ┌───────────────────────┐        ┌───────────────────────┐        ┌───────────────────────┐
        │    Token Outputs      │        │     CLS Pooling       │        │   Cross-Attention     │
        │   (All Positions)     │        │   + Mean Pooling      │        │   Pair Encoder        │
        └───────────┬───────────┘        └───────────┬───────────┘        └───────────┬───────────┘
                    │                                │                                │
        ┌───────────┴───────────┐        ┌───────────┴───────────┐        ┌───────────┴───────────┐
        │                       │        │                       │        │                       │
        ▼                       ▼        ▼                       ▼        ▼                       ▼
┌───────────────┐     ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│  NER General  │     │   Temporal    │ │   Sentiment   │ │   Emotions    │ │      NLI      │ │   Relation    │
│   17 tags     │     │   13 tags     │ │   5 classes   │ │  44 classes   │ │   3 classes   │ │  15 classes   │
└───────────────┘     └───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘

┌───────────────┐     ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│  NER Family   │     │    Intent     │ │Safety Generic │ │Safety FamilyOS│ │    Ingress    │ │   Embedding   │
│   21 tags     │     │   8 classes   │ │   8 types     │ │   4 bands     │ │  12 domains   │ │   768-dim     │
└───────────────┘     └───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘
```

### 🔄 Single Forward Pass

```python
from modeling_studio.models import ModernBertMultiTaskModel
from modeling_studio.data.labels import Capability

# Load unified model
model = ModernBertMultiTaskModel.from_pretrained("checkpoints/modernbert-unified-v2")

# Single inference for multiple tasks
outputs = model.infer(
    text="Mom took Panda to the park, feeling so happy today!",
    capabilities=[
        Capability.NER_FAMILY,      # Extract: Mom (KINSHIP), Panda (NICKNAME), park (LOC)
        Capability.EMOTIONS,         # Detect: joy, togetherness, love
        Capability.SENTIMENT,        # Classify: very_positive
        Capability.SAFETY_FAMILYOS,  # Classify: GREEN
        Capability.INTENT,           # Classify: log_memory
    ]
)
```

---

## 📦 Installation

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

## 🚀 Quick Start

### 1️⃣ Validate Pipeline (Always First!)

```bash
python scripts/validate_full_pipeline.py \
    --config configs/training/multitask/stage_a_generic.yaml \
    --samples 200 \
    --steps 20
```

### 2️⃣ Train Stage A (Generic Multi-Task)

```bash
# Full training (~2-3 hours on A100)
python scripts/train_stage_a.py \
    --config configs/training/multitask/stage_a_generic.yaml

# Quick test (100 steps)
python scripts/train_stage_a.py \
    --config configs/training/multitask/stage_a_generic.yaml \
    training.max_steps=100
```

### 3️⃣ Train Stage B (FamilyOS Domain)

```bash
python scripts/train_stage_b.py \
    --config configs/training/multitask/stage_b_familyos.yaml \
    --base_model checkpoints/modernbert-multitask-v0
```

### 4️⃣ Inference

```python
from modeling_studio.inference import UnifiedInference

# Load model
inference = UnifiedInference.from_pretrained("checkpoints/modernbert-unified-v2")

# Run inference
result = inference.predict(
    "Had a wonderful family dinner, everyone was laughing!"
)

print(result.sentiment)      # "very_positive"
print(result.emotions)       # ["joy", "togetherness", "love", "gratitude"]
print(result.safety_band)    # "GREEN"
print(result.entities)       # []
```

---

## 📊 Training Pipeline

### Two-Stage Training Strategy

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                         │
│   STAGE A: Generic Multi-Task                    STAGE B: FamilyOS Domain               │
│   ─────────────────────────────                  ────────────────────────               │
│                                                                                         │
│   ┌─────────────────────────┐                    ┌─────────────────────────┐            │
│   │   ModernBERT-base       │                    │   Stage A Checkpoint    │            │
│   │   (149M parameters)     │ ──────────────────▶│   + LoRA Adapters      │            │
│   └─────────────────────────┘                    └─────────────────────────┘            │
│                                                                                         │
│   Datasets:                                      Datasets:                              │
│   • CoNLL-2003 (NER)                            • FamilyOS NER (3-5K)                   │
│   • SST-2 (Sentiment)                           • FamilyOS Ingress (5-7K)               │
│   • GoEmotions (Emotions)                       • FamilyOS Safety (3-4K)                │
│   • Jigsaw (Safety)                             • FamilyOS Relations (2-3K)             │
│   • MNLI/SNLI (NLI)                             • FamilyOS Intents (4-5K)               │
│   • STS-B (Embeddings)                          • FamilyOS Embeddings (2K)              │
│                                                                                         │
│   Training:                                      Training:                              │
│   • 10-12 epochs                                 • 5-8 epochs                           │
│   • Full fine-tuning                             • LoRA (r=32, α=64)                    │
│   • ~200K samples                                • Replay 10% Stage A                   │
│   • EMA decay 0.999                              • Safety weight 10-20×                 │
│                                                                                         │
│   Output: modernbert-multitask-v0                Output: modernbert-unified-v2          │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### Training Configurations

| Config | GPU | Batch Size | Gradient Accum | Est. Time |
|--------|-----|------------|----------------|-----------|
| `stage_a_generic.yaml` | Any | 16 | 4 | 4-6 hrs |
| `stage_a_a100.yaml` | A100 40GB | 32 | 2 | 2-3 hrs |
| `stage_a_a100_80gb.yaml` | A100 80GB | 64 | 1 | 1.5-2 hrs |
| `stage_a_h100.yaml` | H100 | 64 | 1 | 1-1.5 hrs |
| `stage_a_a100_fast.yaml` | A100 | 32 | 1 | ~1 hr (subset) |

---

## 🛡️ Safety System

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

## 📋 Task Reference

### Capability Details

| Capability | Type | Labels | Description |
|------------|------|--------|-------------|
| `ner_general` | Token | 17 BIO | Standard entities: PER, ORG, LOC, DATE, TIME, etc. |
| `ner_family` | Token | 21 BIO | Family entities: KINSHIP, NICKNAME, PET, TRADITION, etc. |
| `sentiment` | Sequence | 5 | very_negative → very_positive scale |
| `emotions` | Multi-label | 44 | FamilyOS emotions including family-specific feelings |
| `safety_generic` | Multi-label | 8 | Toxicity types (Jigsaw + self-harm + dangerous advice) |
| `safety_familyos` | Sequence | 4 | Policy bands: GREEN, AMBER, RED, CRISIS |
| `nli` | Pair | 3 | Entailment, neutral, contradiction |
| `embedding` | Vector | 768-dim | Dense representations with Matryoshka support |
| `temporal` | Token | 13 BIO | Time expressions: DATE, TIME, DURATION, etc. |
| `relation` | Pair | 15 | Family relationships: parent_of, sibling_of, etc. |
| `intent` | Sequence | 8 | User intents: log_memory, query_memory, remind, etc. |
| `ingress` | Sequence | 12 | Domains: DIARY, TASK, HEALTH, MEMORY, etc. |

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
│   │       ├── stage_a_datasets.yaml
│   │       └── stage_b_datasets.yaml
│   ├── model/
│   │   └── encoder/                     # Model configs
│   │       └── modernbert_base.yaml
│   └── training/
│       └── multitask/                   # Training configs
│           ├── stage_a_generic.yaml     # Stage A (CPU/basic GPU)
│           ├── stage_a_a100.yaml        # Stage A (A100 40GB)
│           ├── stage_a_a100_80gb.yaml   # Stage A (A100 80GB)
│           ├── stage_a_h100.yaml        # Stage A (H100)
│           └── stage_b_familyos.yaml    # Stage B (LoRA)
│
├── 📂 src/modeling_studio/              # Main package
│   ├── 📂 data/                         # Data pipeline
│   │   ├── labels.py                    # Label schemas (12 tasks)
│   │   ├── loaders.py                   # Dataset loaders
│   │   ├── tokenization.py              # Tokenization functions
│   │   ├── multitask_dataset.py         # Combined dataset
│   │   └── augmentation.py              # Data augmentation
│   │
│   ├── 📂 models/                       # Model architecture
│   │   ├── modernbert_multitask.py      # Main model class
│   │   ├── heads.py                     # Task-specific heads
│   │   ├── adapters.py                  # Task adapters (LoRA)
│   │   ├── poolers.py                   # Pooling strategies
│   │   ├── pair_encoder.py              # Cross-attention encoder
│   │   └── losses.py                    # Custom loss functions
│   │
│   ├── 📂 trainers/                     # Training logic
│   │   ├── multitask_trainer.py         # Multi-task trainer
│   │   ├── collators.py                 # Data collators
│   │   ├── task_sampler.py              # Task sampling
│   │   ├── task_weighting.py            # Uncertainty weighting
│   │   ├── ema.py                       # EMA model
│   │   ├── optimizer.py                 # Head-wise LR
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
│   ├── train_stage_a.py                 # Stage A training
│   ├── train_stage_b.py                 # Stage B training
│   ├── evaluate.py                      # Evaluation script
│   ├── validate_full_pipeline.py        # Pipeline validation
│   ├── calibrate_safety.py              # Safety calibration
│   ├── export_model.py                  # Model export
│   └── 📂 agents/                       # Data generation agents
│
├── 📂 data/                             # Data directory
│   ├── public/                          # Public datasets
│   └── familyos/                        # FamilyOS-specific data
│       ├── ner_family/
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

## 📈 Benchmarks & Quality Targets

### Performance Targets

| Capability | Metric | Stage A Target | Stage B Target |
|------------|--------|----------------|----------------|
| NER General | F1 | 88%+ | 91%+ |
| NER Family | F1 | - | 88%+ |
| Sentiment | Accuracy | 92%+ | 94%+ |
| Emotions | Macro F1 | 75%+ | 78%+ |
| Safety FamilyOS | CRISIS Recall | - | **98%+** ⚠️ |
| Safety FamilyOS | Cultural FP | - | ≤2% |
| NLI | Accuracy | 85%+ | 87%+ |
| Embeddings | Spearman | 0.85+ | 0.87+ |
| Ingress | Accuracy | - | 92%+ |

### Catastrophic Forgetting Gates

After Stage B, re-evaluate on Stage A benchmarks:

| Benchmark | Max Allowed Drop |
|-----------|------------------|
| CoNLL-2003 (NER) | ≤ 2% F1 |
| SST-2 (Sentiment) | ≤ 2% Accuracy |
| MNLI (NLI) | ≤ 2% Accuracy |

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
| [Stage A README](docs/STAGE_A_README.md) | Generic multi-task training guide |
| [Stage B README](docs/STAGE_B_README.md) | FamilyOS domain adaptation guide |
| [Annotation Guidelines](docs/annotation/README.md) | Data annotation standards |
| [Safety Guidelines](docs/annotation/safety_guidelines.md) | Safety classification rules |
| [K0 Module Migration](docs/k0_module_migration.md) | Integration with K0 runtime |
| [Lightning AI Training](docs/LIGHTNING_AI_TRAINING.md) | Cloud training guide |

---

## 🤝 Contributing

We welcome contributions! Please see our contribution guidelines for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **ModernBERT** by Answer.AI — The backbone architecture
- **HuggingFace Transformers** — Model infrastructure
- **GoEmotions** by Google — Emotion classification dataset
- **Jigsaw/Perspective API** — Toxicity detection data
- **CoNLL-2003** — NER benchmark dataset

---

<div align="center">

**Built with ❤️ for families**

[⬆ Back to Top](#-familyos-modernbert)

</div>
