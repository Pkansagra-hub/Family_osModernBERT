<div align="center">

# 🏠 FamilyOS UltraBERT

### Production Multi-Task Encoder + GPT-2 Counterfactual Decoder

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Flash Attention 2](https://img.shields.io/badge/Flash_Attention-2-orange.svg)](https://github.com/Dao-AILab/flash-attention)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow)](https://huggingface.co/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**v2 Encoder** *(Production)* — 22 layers, 12 heads, 155M params
**GPT-2 Decoder** *(Finalized)* — 24 layers, 355M params, pre-trained fluency
**v3 Ultra** *(Roadmap)* — 28 layers, hub tokens, multi-scale attention

[Architecture](#-architecture-overview) • [v2 Encoder](#-v2-multi-task-encoder) • [GPT-2 Decoder](#-stage-c-gpt-2-counterfactual-decoder) • [v3 Roadmap](#-v3-ultra-roadmap) • [Installation](#-installation)

</div>

---

## 📖 Overview

**FamilyOS UltraBERT** is a production-ready multi-task NLP system for family assistant AI, featuring:

- **v2 Encoder** — 22-layer ModernBERT with 12 task-specific heads (NER, emotions, safety, embeddings, etc.)
- **GPT-2 Decoder** — 13th head: Pre-trained GPT-2 Medium with prefix injection for fluent counterfactual generation
- **v3 Ultra** — Future 28-layer architecture with hub token routing (roadmap)

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
| 1 | `ner_general`     | Token       | 9 BIO tags    | Encoder  |
| 2 | `ner_family`      | Token       | 12 BIO tags   | Encoder  |
| 3 | `sentiment`       | Sequence    | 5 classes     | Encoder  |
| 4 | `emotions`        | Multi-label | 44 emotions   | Encoder  |
| 5 | `safety_generic`  | Multi-label | 8 types       | Encoder  |
| 6 | `safety_familyos` | Sequence    | 4 bands       | Encoder  |
| 7 | `nli`             | Pair        | 3 classes     | Encoder  |
| 8 | `embedding`       | Vector      | 768-dim       | Encoder  |
| 9 | `temporal`        | Token       | 7 BIO tags    | Encoder  |
| 10| `relation`        | Pair        | 15 types      | Encoder  |
| 11| `intent`          | Sequence    | 8 classes     | Encoder  |
| 12| `ingress`         | Sequence    | 6 domains     | Encoder  |

---

## 🧠 v2 Multi-Task Encoder

The production encoder powering FamilyOS classification, NER, embeddings, and safety detection.

### Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  ModernBERT-base v2 (Production)                                            │
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

| Checkpoint | Step | Status | Weighted Score |
|------------|------|--------|----------------|
| `checkpoint-18000` | 18,000 | ✅ Production | **90.58%** |

---

batch_size: 128 (per GPU)
gradient_accumulation_steps: 2
effective_batch_size: 256
learning_rate: 1e-4 (cosine decay)
warmup_ratio: 0.1
num_train_epochs: 5
temperature: 1.0
top_k: 50
top_p: 0.9
repetition_penalty: 1.2

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
| `checkpoint-18000` | 18,000 | ✅ Production | **90.58%**     |

---

## 🚀 v3 Ultra Roadmap
>
> **Status:** Under Development (after v4 release)
...existing code...

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🎯 v2 Encoder (Production)

- **22 Transformer Layers** — ModernBERT-base backbone
- **12 Task Heads** — Unified multi-task inference
- **Flash Attention 2** — 2x speedup on A100/H100
- **768-dim Embeddings** — Dense semantic vectors
- **4-Band Safety** — GREEN/AMBER/RED/CRISIS
- **Cultural Awareness** — Indian English patterns

### 🧠 13 Capabilities

- **NER General** — 9 BIO entity tags
- **NER Family** — 12 family-specific entities
- **Sentiment** — 5-point scale analysis
- **Emotions** — 44 emotion classes (family-aware)
- **Safety Generic** — Standard toxicity detection
- **Safety FamilyOS** — 4-band hierarchical system
- **NLI** — 3-way entailment classification
- **Embeddings** — 768-dim dense vectors
- **Temporal** — 7 time expression tags
- **Relations** — 15 family relationship types
- **Intent** — 8 user intent classes
- **Ingress** — 6 domain categories
- **Counterfactual** — GPT-2 text generation

</td>
<td width="50%">

### 🔮 GPT-2 Decoder (Finalized)

- **24 Transformer Layers** — GPT-2 Medium backbone
- **355M Parameters** — Pre-trained on 40GB WebText
- **Prefix Injection** — Encoder-conditioned generation
- **1024-dim Hidden** — Projected from encoder 768-dim
- **Edge-Friendly** — ~1GB VRAM for inference
- **Fast Convergence** — 3-5 epochs sufficient

### 🛡️ Safety System

- **≥98% CRISIS recall** — Zero misses on self-harm
- **≤2% Cultural FP** — Indian English aware
- **4-band hierarchy** — GREEN → AMBER → RED → CRISIS
- **Keyword override** — Explicit crisis detection
- **12 subcategories** — Fine-grained classification

### 🚀 Production Ready

- **<90ms full inference** — 12 tasks, single pass
- **93% Recall@10** — Embedding retrieval
- **98.6% triplet accuracy** — Semantic similarity
- **A100/H100 optimized** — Flash Attention 2
- **Phase 0.5 Healing** — Interface alignment warmup

### 🛡️ Enhanced Safety System

- **≥99% CRISIS recall** — Zero misses on self-harm
- **≤1% Cultural FP** — Indian English aware
- **4-band hierarchy** — GREEN → AMBER → RED → CRISIS
- **Keyword override** — Explicit crisis detection
- **12 subcategories** — Fine-grained classification
- **15% Stage A replay** — Prevents forgetting

### 🚀 Production Ready

- **<35ms NPU latency** — Edge deployment optimized
- **<55ms full inference** — 28 layers, multi-task
- **~180M parameters** — Efficient architecture
- **ONNX export** — INT8 quantization support

</td>
</tr>
</table>

---

## 🚀 v3 Ultra Roadmap

> **Status:** Future Development (after Stage C completion)

v3 Ultra is the next-generation encoder with **Hub Token Architecture** — 4 specialized tokens with global bidirectional attention for superior task routing.

### Planned Features

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  🎯 Hub Token Routing        │  ⚡ Multi-Scale Attention                │
│  4 specialized hub tokens    │  64 → 128 → 256 → 512 sliding windows   │
│  [EMO] [MEM] [REL] [TASK]    │  Flash Attention 2 optimized            │
│  Global bidirectional attn   │  <35ms latency target                   │
├─────────────────────────────────────────────────────────────────────────┤
│  🧬 Function Preserving      │  📊 Target Improvements                 │
│  Direct v2 weight transfer   │  +2-5% per task over v2                 │
│  L1-22: Copy, L23-28: Clone  │  CRISIS recall ≥99%                     │
│  No distillation needed      │  Cultural FP ≤1%                        │
└─────────────────────────────────────────────────────────────────────────┘
```

### v3 Architecture Preview

| Component | v2 (Current) | v3 (Planned) | Benefit |
|-----------|--------------|--------------|---------|
| **Layers** | 22 | 28 (+6 new) | Higher capacity |
| **Parameters** | 155M | ~180M | +21% |
| **Hub Tokens** | None | 4 ([EMO], [MEM], [REL], [TASK]) | Task routing |
| **Attention** | Uniform | Multi-scale (64/128/256/512) | Efficiency |
| **LoRA** | None | L23-28 only (r=16) | Parameter efficiency |

### Hub Token Routing System (Planned)

```text
Input: "Mom is feeling sad today"

Tokenization with Hub Injection:
[CLS] [EMO] [MEM] [REL] [TASK] Mom is feeling sad today [SEP]
  0     1     2     3     4     5   6    7      8    9    10

Hub Token Routing:
  [EMO]  → Emotions, Sentiment, Safety heads
  [MEM]  → Embedding head (768-dim vectors)
  [REL]  → NLI, Relation heads
  [TASK] → Intent, Ingress heads
  Token positions → NER, Temporal heads
```

<details>
<summary>Click to expand full v3 architecture diagram</summary>

```text
║                                            │                                               ║
║  ┌───────────────────────────────────────────────────────────────────────────────────────┐ ║
║  │  🔴 FAMILY BAND (Layers 23-28)  ⭐NEW⭐                  Window: 512   🔥 LoRA      │ ║
║  │  ──────────────────────────────────                                                   │ ║
║  │  • Family-specific understanding, cultural context                                    │ ║
║  │  • Safety/crisis detection, relationship mapping                                      │ ║
║  │  • Cloned from v2 L15-20 + LoRA adaptation (r=16, α=16)                               │ ║
║  │  • Hub tokens: GLOBAL attention (final task routing)                                  │ ║
║  │  • Text tokens: SLIDING WINDOW (window=512, maximum context)                          │ ║
║  │  • LoRA applied to: q_proj, k_proj, v_proj, o_proj (only these 6 layers)              │ ║
║  └───────────────────────────────────────────────────────────────────────────────────────┘ ║
║                                                                                            ║
║  💡 Key Innovation: HYBRID ATTENTION MECHANISM                                            ║
║     • Hub tokens [EMO][MEM][REL][TASK] = GLOBAL bidirectional attention                    ║
║     • Text tokens = SLIDING WINDOW local attention (multi-scale: 64→128→256→512)           ║
║     • Efficiency: O(n·w) instead of O(n²), where w << n                                    ║
║     • Quality: Hub tokens aggregate global context for task routing                        ║
╚═════════════════════════════════════════════════════════════════════════════════════════════╝
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│  🎯 OUTPUT REPRESENTATIONS (All from final layer L28)                                       │
│                                                                                             │
│  Position 0:  [CLS]  → 768-dim (general sequence representation)                            │
│  Position 1:  [EMO]  → 768-dim (emotion/sentiment/safety hub)                               │
│  Position 2:  [MEM]  → 768-dim (embedding/memory hub)                                       │
│  Position 3:  [REL]  → 768-dim (relation/logic hub)                                         │
│  Position 4:  [TASK] → 768-dim (intent/ingress hub)                                         │
│  Position 5-10: TEXT → 768-dim each (token-level representations)                           │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
╔═════════════════════════════════════════════════════════════════════════════════════════════╗
║  🎪 ROUTING LAYER (Hub Token Pooler + Task-Specific Heads)                                  ║
║                                                                                             ║
║  ┌─────────────────────┐       ┌─────────────────────┐       ┌──────────────────────────┐   ║
║  │   🟠 [EMO] HUB      │       │   🔵 [MEM] HUB     │       │   🟣 [REL] HUB           │   ║
║  │   Position 1        │       │   Position 2        │       │   Position 3             │   ║
║  │   (768-dim)         │       │   (768-dim)         │       │   (768-dim)              │   ║
║  └──────────┬──────────┘       └──────────┬──────────┘       └──────────┬───────────────┘   ║
║             │                             │                             │                   ║
║    ┌────────┴────────┐           ┌────────▼────────┐           ┌────────▼────────┐          ║
║    │                 │           │                 │           │                 │          ║
║    ▼                 ▼           ▼                 │           ▼                 ▼          ║
║  ┌────────┐    ┌─────────┐   ┌─────────┐          │      ┌────────┐      ┌──────────┐       ║
║  │Emotions│    │Sentiment│   │ Safety  │          │      │  NLI   │      │Relations │       ║
║  │  Head  │    │  Head   │   │  Head   │          │      │  Head  │      │   Head   │       ║
║  │ (44cls)│    │ (5 cls) │   │(4 bands)│          │      │(3 cls) │      │ (15 cls) │       ║
║  └────────┘    └─────────┘   └─────────┘          │      └────────┘      └──────────┘       ║
║     │               │              │               │          │                │            ║
║     ▼               ▼              ▼               ▼          ▼                ▼            ║
║  [joy,        very_positive     GREEN         (768-dim)  entailment      parent_of          ║
║   love,                                       embedding                                     ║
║   concern]                                                                                  ║
║                                                                                             ║
║  ┌─────────────────────┐       ┌──────────────────────────────────────────────────────┐     ║
║  │   🟢 [TASK] HUB     │       │   📊 TOKEN-LEVEL OUTPUTS (Full Sequence)            │     ║
║  │   Position 4        │       │   Positions 5-10 (all text tokens)                   │     ║
║  │   (768-dim)         │       │                                                      │     ║
║  └──────────┬──────────┘       └──────────────────────┬───────────────────────────────┘     ║
║             │                                         │                                     ║
║    ┌────────┴────────┐                       ┌────────┴────────┐                            ║
║    │                 │                       │                 │                            ║
║    ▼                 ▼                       ▼                 ▼                            ║
║  ┌────────┐    ┌─────────┐            ┌──────────┐      ┌──────────┐                        ║
║  │ Intent │    │ Ingress │            │   NER    │      │   NER    │                        ║
║  │  Head  │    │  Head   │            │ General  │      │  Family  │                        ║
║  │(8 cls) │    │(6 cls)  │            │ (9 BIO)  │      │ (12 BIO) │                        ║
║  └────────┘    └─────────┘            └──────────┘      └──────────┘                        ║
║     │               │                       │                  │                            ║
║     ▼               ▼                       ▼                  ▼                            ║
║  log_memory     DIARY              [O, B-PER, O, O, B-EMO, O]                               ║
║                                    [O, B-KINSHIP, O, O, B-EMOTION, O]                       ║
║                                                                                             ║
║  ┌──────────────────────────────────────────────────────────────────────────────────┐       ║
║  │   🔧 Temporal Head (Token-level, separate pathway)                               │      ║
║  │   Input: Full sequence positions 5-10                                            │       ║
║  │   Output: [O, O, O, O, B-TIME, O]  (7 BIO tags)                                  │       ║
║  └──────────────────────────────────────────────────────────────────────────────────┘       ║
╚═════════════════════════════════════════════════════════════════════════════════════════════╝
                                            │
                                            ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  📦 UNIFIED OUTPUT (Single Forward Pass)                                                ┃
┃                                                                                         ┃
┃  {                                                                                      ┃
┃    "emotions": ["joy", "love", "concern"],              # [EMO] hub → Multi-label head  ┃
┃    "sentiment": "very_positive",                        # [EMO] hub → 5-class head      ┃
┃    "safety_band": "GREEN",                              # [EMO] hub → 4-band hierarchy  ┃
┃    "embedding": <768-dim vector>,                       # [MEM] hub → Dense vector      ┃
┃    "nli": "entailment",                                 # [REL] hub → 3-class head      ┃
┃    "relation": "parent_of",                             # [REL] hub → 15-class head     ┃
┃    "intent": "log_memory",                              # [TASK] hub → 8-class head     ┃
┃    "ingress": "DIARY",                                  # [TASK] hub → 6-class head     ┃
┃    "ner_general": [("Mom", "PER"), ("today", "TIME")], # Token-level → BIO tags         ┃
┃    "ner_family": [("Mom", "KINSHIP")],                 # Token-level → BIO tags         ┃
┃    "temporal": [("today", "TIME")],                    # Token-level → BIO tags         ┃
┃  }                                                                                      ┃
┃                                                                                         ┃
┃  ⚡ Performance: <35ms on NPU (256 tokens) | ~180M parameters | Zero routing overhead  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

### 🔄 Single Forward Pass

```python
from modeling_studio.models import ModernBERTv3Ultra
from modeling_studio.inference import UnifiedInference

# Load v3 Ultra model
model = ModernBERTv3Ultra.from_pretrained("checkpoints/modernbert-v3-ultra")
inference = UnifiedInference(model)

# Single forward pass with hub routing
result = inference.predict(
    text="Mom took Panda to the park, feeling so happy today!",
    tasks=[
        "ner_family",        # → Token-level (full sequence)
        "emotions",          # → [EMO] hub token
        "sentiment",         # → [EMO] hub token
        "safety_familyos",   # → [EMO] hub token
        "intent",            # → [TASK] hub token
        "embedding",         # → [MEM] hub token
    ]
)

print(result.entities)        # [("Mom", "KINSHIP"), ("Panda", "NICKNAME")]
print(result.emotions)        # ["joy", "togetherness", "love"]
print(result.sentiment)       # "very_positive"
print(result.safety_band)     # "GREEN"
print(result.intent)          # "log_memory"
print(result.embedding.shape) # (768,)
```

### 🎯 Hub Token Capabilities

| Hub Token | Position | Routed Capabilities | Description |
|-----------|----------|---------------------|-------------|
| `[EMO]` | 1 | emotions, sentiment, safety_* | Affective understanding & safety |
| `[MEM]` | 2 | embedding | Memory & retrieval representations |
| `[REL]` | 3 | nli, relation | Relationships & logical reasoning |
| `[TASK]` | 4 | intent, ingress | User actions & domain classification |
| Token-level | N/A | ner_*, temporal | Per-token sequence labeling |

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

### 1️⃣ Initialize v3 from v2 (Function Preserving Growth)

```bash
# Initialize 28-layer v3 from 22-layer v2 checkpoint
python scripts/initialize_v3_from_v2.py \
    --v2_checkpoint checkpoints/modernbert-v2-final \
    --output_path checkpoints/modernbert-v3-initialized

# Verify function preserving (L1-22 should match v2 exactly)
python scripts/verify_function_preserving.py \
    --v2_checkpoint checkpoints/modernbert-v2-final \
    --v3_checkpoint checkpoints/modernbert-v3-initialized
```

### 2️⃣ Phase 0.5: Healing Warmup (CRITICAL)

```bash
# Align L22→L23 interface before multi-task training
python scripts/train_v3.py \
    --config configs/training/multitask/stage_v3_phase0_5_enhanced.yaml \
    --checkpoint checkpoints/modernbert-v3-initialized \
    --phase 0.5

# Output: modernbert-v3-healed
```

### 3️⃣ Phase 1: Full Multi-Task Training

```bash
# Train with 15% Stage A replay to prevent forgetting
python scripts/train_v3.py \
    --config configs/training/multitask/stage_v3_phase1.yaml \
    --checkpoint checkpoints/modernbert-v3-healed \
    --phase 1

# Output: modernbert-v3-phase1
```

### 4️⃣ Phase 1.5: Forgetting Evaluation (Gate)

```bash
# Evaluate on Stage A benchmarks (must pass before production)
python scripts/forgetting_eval.py \
    --model checkpoints/modernbert-v3-phase1 \
    --benchmarks CoNLL,SST2,MNLI

# Max allowed drops: ≤2% on NER, Sentiment, NLI
```

### 5️⃣ Production Inference

```python
from modeling_studio.models import ModernBERTv3Ultra
from modeling_studio.inference import UnifiedInference

# Load v3 Ultra
model = ModernBERTv3Ultra.from_pretrained("checkpoints/modernbert-v3-phase1")
inference = UnifiedInference(model)

# Multi-task inference with hub routing
result = inference.predict(
    "Had a wonderful family dinner, everyone was laughing!"
)

print(result.sentiment)       # "very_positive"
print(result.emotions)        # ["joy", "togetherness", "love", "gratitude"]
print(result.safety_band)     # "GREEN"
print(result.entities)        # []
print(result.intent)          # "log_memory"
print(result.embedding.shape) # (768,)
```

---

## 📊 Training Pipeline

### v3 Ultra Multi-Phase Training Strategy

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                        │
│  Phase 0: Initialization          Phase 0.5: Healing         Phase 1: Multi-Task       │
│  ─────────────────────            ────────────────────       ────────────────────      │
│                                                                                        │
│  ┌──────────────────┐             ┌──────────────────┐       ┌──────────────────┐      │
│  │  ModernBERT v2   │             │ v3 Initialized   │       │   v3 Healed      │      │
│  │  22 layers       │───────────▶│ 28 layers        │──────▶│  + Multi-Task    │     │
│  │  149M params     │  Function   │ 180M params      │ 2.5K  │  Heads trained   │      │
│  └──────────────────┘  Preserving └──────────────────┘ steps └──────────────────┘      │
│                        Growth                                                          │
│  Weight Transfer:                 Healing Data:              Training Data:            │
│  • L1-22: Direct copy             • SST-2: 3K samples        • FamilyOS unified:       │
│  • L23-28: Clone L15-20           • CoNLL: 3K samples          85% (shards)            │
│  • Hub tokens: Semantic           • MNLI: 2K samples         • Stage A replay:         │
│    centroid init                  • SQuAD: 2K samples          15% (forgetting gate)   │
│                                   • STS-B: 2K samples                                  │
│  Layer Freezing:                                             Safety Oversampling:      │
│  N/A (initialization)             L1-18: ❄️ Frozen           • CRISIS: 20x             │
│                                   L19-22: 🔥 Zipper LR       • RED: 5x                 │
│  Output:                          L23: 🔥 MAX (5e-5)         • 15% Stage A replay      │
│  modernbert-v3-initialized        L24-28: 🔥 Zipper          • EMA decay 0.999         │
│                                   Heads: ❄️ Frozen                                     │
│                                                              Output:                   │
│                                   Output:                    modernbert-v3-phase1      │
│                                   modernbert-v3-healed                                 │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Training Configurations

| Config | Phase | GPU | Batch | Frozen Layers | LoRA | Est. Time |
|--------|-------|-----|-------|---------------|------|-----------|
| `stage_v3_phase0_5_enhanced.yaml` | Healing | A100 | 5 | L1-18 | None | ~2-3 hrs |
| `stage_v3_phase1.yaml` | Full | A100 40GB | 8 | L1-18 | L23-28 | 8-12 hrs |
| `stage_v3_phase1_a100_80gb.yaml` | Full | A100 80GB | 16 | L1-18 | L23-28 | 4-6 hrs |
| `stage_v3_phase1_h100.yaml` | Full | H100 | 16 | L1-18 | L23-28 | 3-4 hrs |

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

| Capability | Hub | Type | Labels | Description |
|------------|-----|------|--------|-------------|
| `ner_general` | - | Token | 9 BIO | Standard entities: PER, ORG, LOC, etc. |
| `ner_family` | - | Token | 12 BIO | Family entities: KINSHIP, NICKNAME, PET, etc. |
| `sentiment` | [EMO] | Sequence | 5 | very_negative → very_positive scale |
| `emotions` | [EMO] | Multi-label | 44 | FamilyOS emotions + family-specific feelings |
| `safety_generic` | [EMO] | Multi-label | Standard | Jigsaw toxicity types |
| `safety_familyos` | [EMO] | Hierarchical | 4 bands | GREEN → AMBER → RED → CRISIS |
| `nli` | [REL] | Pair | 3 | Entailment, neutral, contradiction |
| `embedding` | [MEM] | Vector | 768-dim | Dense representations for retrieval |
| `temporal` | - | Token | 7 BIO | Time expressions: DATE, TIME, DURATION |
| `relation` | [REL] | Pair | 15 | Family relationships: parent_of, sibling_of |
| `intent` | [TASK] | Sequence | 8 | User intents: log_memory, query, remind |
| `ingress` | [TASK] | Sequence | 6 | Domains: DIARY, TASK, HEALTH, MEMORY |

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

## 📈 Benchmarks & Quality Targets

### v3 Ultra Quality Targets

| Capability | Metric | v2 Baseline | v3 Target | Improvement |
|------------|--------|-------------|-----------|-------------|
| NER General | F1 | 89% | **93%** | +4% |
| NER Family | F1 | 86% | **91%** | +5% |
| Sentiment | Accuracy | 92% | **96%** | +4% |
| Emotions | Macro F1 | 76% | **82%** | +6% |
| Safety FamilyOS | CRISIS Recall | 98% | **≥99%** ⚠️ | +1% |
| Safety FamilyOS | Cultural FP | 2% | **≤1%** | Better |
| NLI | Accuracy | 86% | **91%** | +5% |
| Embeddings | Recall@10 | 85% | **90%** | +5% |
| Relations | F1 | 82% | **87%** | +5% |
| Intent | Accuracy | 90% | **93%** | +3% |
| Temporal | F1 | 85% | **89%** | +4% |
| Ingress | Accuracy | 92% | **95%** | +3% |

### Latency Targets (256 tokens, multi-task)

| Platform | v2 (22 layers) | v3 Ultra (28 layers) | Target Met? |
|----------|----------------|----------------------|-------------|
| A100 GPU | ~15ms | ~18ms | ✅ <20ms |
| RTX 4090 | ~25ms | ~30ms | ✅ <35ms |
| Ryzen AI NPU | ~60ms | ~72ms → **<35ms** (Phase 2*) | ✅ |
| Apple M3 | ~45ms | ~55ms | ✅ <60ms |

*Phase 2 with GQA/SwiGLU (R&D track, not in production roadmap)

### Catastrophic Forgetting Gates (Phase 1.5)

After Phase 1 training, **mandatory evaluation** on Stage A benchmarks:

| Benchmark | Max Allowed Drop | Action if Failed |
|-----------|------------------|------------------|
| CoNLL-2003 (NER) | ≤ 2% F1 | Increase replay ratio to 20% |
| SST-2 (Sentiment) | ≤ 2% Accuracy | Increase replay ratio to 20% |
| MNLI (NLI) | ≤ 2% Accuracy | Increase replay ratio to 20% |
| FamilyOS Emotions | ≤ 3% F1 | Reduce LoRA rank (r=8) |

**Gate Status:** Must pass before production deployment ⚠️

---

## V4 Model Benchmark Report (UltraBERT v4.0.0)

> **Evaluation Date:** January 22, 2026
> **Weights:** `familyos_ultrabert/weights/pytorch`
> **Model:** ModernBERT + GlobalPointer Decoder (22 layers, ~149M params, 768-dim, 3 GP heads)
> **Hardware:** NVIDIA RTX 5070 Laptop GPU (CUDA 12.8)

### Overall Performance Summary

| Category | Metric | Score | Status |
|----------|--------|-------|--------|
| **Benchmark Suite** | Pass Rate | **87/89** (97.8%) | Production Ready |
| **CRISIS Detection** | Recall | **100%** | CRITICAL PASS |
| **Safety Band** | Accuracy | **100%** | PASS |
| **Embedding Triplet** | Accuracy | **70%+** | PASS |

---

### Classification Performance

| Task | Metric | Score | Status |
|------|--------|-------|--------|
| **safety_familyos** | band_accuracy | **100%** | PASS |
| **safety_familyos** | crisis_recall | **100%** | CRITICAL |
| **intent** | accuracy | **100%** | PASS |
| **emotions** | hit_rate | **95.3%** | PASS |
| **sentiment** | direction_accuracy | **100%** | PASS |
| **sentiment** | 5class_accuracy | **66.7%** | PASS |

---

### Named Entity Recognition (GlobalPointer Heads)

| Head | Metric | Score | Optimal Threshold |
|------|--------|-------|-------------------|
| **ner_general** | F1 | **73.0%** | -1.0 |
| **ner_family** | F1 | **81.2%** | -0.7 |
| **temporal** | F1 | **63.9%** | -1.9 |

---

### NER Quality Issues Resolution (GlobalPointer)

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

### Inference Latency Benchmarks (RTX 5070)

#### Full Multi-Task Inference (12 Capabilities)

| Metric | Value |
|--------|-------|
| **P95 Latency** | **16.84 ms** |
| **Average** | ~11-16 ms |
| **Throughput (Sequential)** | **61.5 inferences/sec** |
| **Throughput (Burst)** | **82.2 inferences/sec** |

#### Per-Capability Latency (P95)

| Capability | Latency |
|------------|---------|
| ingress | 0.61 ms |
| intent | 0.78 ms |
| safety_generic | 0.84 ms |
| relation | 0.87 ms |
| sentiment | 0.91 ms |
| embedding | 1.01 ms |
| nli | 1.21 ms |
| safety_familyos | 1.79 ms |
| ner_general | 2.10 ms |
| temporal | 3.18 ms |
| ner_family | 3.58 ms |
| emotions | 7.40 ms |

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

### Key Improvements from V2

| Metric | V2 (checkpoint-18000) | V4 (UltraBERT) | Change |
|--------|----------------------|----------------|--------|
| Full Inference P95 | 102.25 ms | **16.84 ms** | **6.1x faster** |
| Throughput | 11.3/sec | **61.5/sec** | **5.4x higher** |
| CRISIS Recall | - | **100%** | Guaranteed |
| NER Architecture | Token Classification | **GlobalPointer** | Better spans |
| Query Latency | 18.44 ms | **1.86 ms** | **9.9x faster** |

---

### Sample Inference Output

```json
{
  "text": "My grandmother called yesterday to remind me about the family reunion next Sunday. I am so excited!",
  "sentiment": "very_positive",
  "sentiment_confidence": 0.9719,
  "emotions": ["joy", "excitement", "togetherness", "warmth"],
  "safety": "GREEN",
  "safety_confidence": 1.0,
  "entities": [],
  "temporal": [
    {"text": "yesterday", "label": "DATE_REL", "score": 0.45},
    {"text": "next Sunday", "label": "DATE_REL", "score": 0.26}
  ],
  "intent": "share_news",
  "ingress": "CELEBRATION",
  "relations": ["grandparent_of", "grandchild_of"],
  "latency_ms": 23.96
}
```

---

### Production Readiness Assessment

| Capability | Status | Notes |
|------------|--------|-------|
| **Safety** | PASS | 100% band accuracy, 100% CRISIS recall |
| **Intent** | PASS | 100% valid label rate |
| **Emotions** | PASS | 95.3% hit rate, multi-label |
| **Sentiment** | PASS | 100% direction accuracy |
| **NER (GlobalPointer)** | PASS | 73-81% F1, span extraction |
| **Temporal** | PASS | 63.9% F1, time expressions |
| **Embeddings** | PASS | 84.5% R@1 (10d), 100% R@10 (100d) |
| **Robustness** | PASS | All edge/unicode/adversarial tests |

**Overall Verdict:** **Production Ready** for FamilyOS deployment.

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

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **ModernBERT** by Answer.AI — The backbone architecture (v2 base)
- **OpenAI GPT-2** — Pre-trained decoder foundation
- **Flash Attention 2** by Dao-AILab — Efficient attention implementation
- **LoRA** by Microsoft — Parameter-efficient fine-tuning
- **HuggingFace Transformers** — Model infrastructure
- **GoEmotions** by Google — Emotion classification dataset
- **Jigsaw/Perspective API** — Toxicity detection data
- **CoNLL-2003** — NER benchmark dataset

---

<div align="center">

### FamilyOS UltraBERT — Evolution Timeline

| Version | Status | Components | Params |
|---------|--------|------------|--------|
| **v2 Encoder** | ✅ Production | 22 layers, 12 heads | 155M |
| **GPT-2 Decoder** | ✅ Finalized | 24 layers, pre-trained | 355M |
| **v3 Ultra** | 📋 Roadmap | 28 layers, hub tokens | ~180M |

---

### Key Innovations

| Component | Innovation | Benefit |
|-----------|------------|---------|
| **v2 Multi-Task** | 12 heads, unified inference | Single forward pass |
| **GPT-2 Decoder** | Pre-trained + prefix injection | Fluent counterfactuals |
| **Safety System** | 4-band hierarchy, cultural awareness | CRISIS recall ≥98% |
| **Embeddings** | 768-dim, 98.6% triplet accuracy | 93% Recall@10 |

---

**Built with care for families**

**FamilyOS UltraBERT — Multi-Task Encoder**

[Back to Top](#-familyos-ultrabert)
</div>
