# UltraBERT-Gen MoE Decoder Architecture

## Overview

**Component**: 13th Head - Counterfactual Generation Decoder
**Codename**: UltraBERT-Gen MoE
**Total Params**: ~420M decoder + 155M encoder = **575M total**
**Active Params**: ~230M decoder + 155M encoder = **385M active per token**
**Target**: Best-in-class counterfactual generation for FamilyOS (P03 offline)

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         UltraBERT-Gen MoE v1.0                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  ENCODER (Frozen, 155M params)                                      │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │  ModernBERT-base                                              │  │    │
│  │  │  - Hidden: 768                                                │  │    │
│  │  │  - Layers: 22                                                 │  │    │
│  │  │  - Heads: 12                                                  │  │    │
│  │  │  - Vocab: 50,280                                              │  │    │
│  │  │  → Output: (B, S_enc, 768)                                    │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  ENCODER PROJECTION (1.0M params)                                   │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │  Linear(768 → 1280)                                           │  │    │
│  │  │  RMSNorm(1280)                                                │  │    │
│  │  │  GELU activation                                              │  │    │
│  │  │  Dropout(0.1)                                                 │  │    │
│  │  │  → Output: (B, S_enc, 1280)                                   │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    ↓                                        │
│                         [Encoder Context for Cross-Attention]               │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  DECODER (Trainable, ~420M params)                                  │    │
│  │                                                                     │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │  TOKEN EMBEDDING (64.4M params, tied with LM head)            │  │    │
│  │  │  - Vocab: 50,280 × Hidden: 1,280                              │  │    │
│  │  │  - RoPE Positional Encoding (θ=10000, max_len=512)            │  │    │
│  │  │  → Output: (B, S_dec, 1280)                                   │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  │                              ↓                                      │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │  DENSE LAYERS 0-1 (27.5M params)                              │  │    │
│  │  │  ┌─────────────────────────────────────────────────────────┐  │  │    │
│  │  │  │  For each layer:                                        │  │  │    │
│  │  │  │  1. RMSNorm → GQA Self-Attention (causal) → Residual    │  │  │    │
│  │  │  │  2. RMSNorm → Cross-Attention (to encoder) → Residual   │  │  │    │
│  │  │  │  3. RMSNorm → SwiGLU FFN (dense) → Residual             │  │  │    │
│  │  │  └─────────────────────────────────────────────────────────┘  │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  │                              ↓                                      │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │  MoE LAYERS 2-7 (274.1M params, 82.7M active)                 │  │    │
│  │  │  ┌─────────────────────────────────────────────────────────┐  │  │    │
│  │  │  │  For each layer:                                        │  │  │    │
│  │  │  │  1. RMSNorm → GQA Self-Attention (causal) → Residual    │  │  │    │
│  │  │  │  2. RMSNorm → Cross-Attention (to encoder) → Residual   │  │  │    │
│  │  │  │  3. RMSNorm → Sparse MoE FFN → Residual                 │  │  │    │
│  │  │  │     ├── Router: Linear(1280 → 8) + TopK(2)              │  │  │    │
│  │  │  │     ├── 8 Expert FFNs (SwiGLU)                          │  │  │    │
│  │  │  │     └── 1 Shared Expert (always active)                 │  │  │    │
│  │  │  └─────────────────────────────────────────────────────────┘  │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  │                              ↓                                      │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │  OUTPUT (tied with embeddings)                                │  │    │
│  │  │  - RMSNorm(1280)                                              │  │    │
│  │  │  - LM Head: Linear(1280 → 50,280)                             │  │    │
│  │  │  → Output: (B, S_dec, 50,280) logits                          │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Component Specifications

### 1. Encoder Projection

```python
EncoderProjection:
    input_dim: 768          # ModernBERT hidden size
    output_dim: 1280        # Decoder hidden size

    layers:
        - Linear(768, 1280, bias=True)
        - RMSNorm(1280, eps=1e-6)
        - GELU()
        - Dropout(0.1)

    params: ~1.0M
```

### 2. Token Embeddings + RoPE

```python
TokenEmbedding:
    vocab_size: 50280       # ModernBERT tokenizer
    hidden_size: 1280
    tie_weights: True       # Shared with LM head

    params: 64.4M (shared)

RotaryPositionalEmbedding:
    dim: 64                 # head_dim
    max_seq_len: 512
    base: 10000.0           # theta

    # RoPE is computed, no learned params
    params: 0
```

### 3. GQA Self-Attention (All Layers)

```python
GroupedQueryAttention:
    hidden_size: 1280
    num_heads: 20           # Query heads
    num_kv_heads: 4         # Key/Value heads (5:1 GQA ratio)
    head_dim: 64            # 1280 / 20

    projections:
        Q: Linear(1280, 1280)   # 20 heads × 64
        K: Linear(1280, 256)    # 4 heads × 64
        V: Linear(1280, 256)    # 4 heads × 64
        O: Linear(1280, 1280)

    is_causal: True         # Causal mask for autoregressive
    use_flash_attention: True
    dropout: 0.0            # No dropout in attention (modern practice)

    params_per_layer: 1280² + 2×(1280×256) + 1280² = 3.93M
    total (8 layers): 31.4M
```

### 4. Cross-Attention (All Layers)

```python
CrossAttention:
    hidden_size: 1280
    num_heads: 20           # Full heads (no GQA for cross-attn)
    head_dim: 64

    projections:
        Q: Linear(1280, 1280)   # From decoder
        K: Linear(1280, 1280)   # From encoder context
        V: Linear(1280, 1280)   # From encoder context
        O: Linear(1280, 1280)

    is_causal: False        # Attend to all encoder positions
    use_flash_attention: True
    dropout: 0.0

    params_per_layer: 4 × 1280² = 6.55M
    total (8 layers): 52.4M
```

### 5. Dense FFN (Layers 0-1)

```python
DenseSwiGLUFFN:
    hidden_size: 1280
    intermediate_size: 3584  # 2.8× hidden

    layers:
        gate_proj: Linear(1280, 3584, bias=False)
        up_proj: Linear(1280, 3584, bias=False)
        down_proj: Linear(3584, 1280, bias=False)

    forward:
        gate = gate_proj(x)
        up = up_proj(x)
        x = down_proj(SiLU(gate) * up)

    params_per_layer: 3 × 1280 × 3584 = 13.76M
    total (2 layers): 27.5M
```

### 6. Sparse MoE FFN (Layers 2-7)

```python
SparseMoEFFN:
    hidden_size: 1280
    num_experts: 8
    num_experts_per_token: 2        # Top-2 routing
    expert_intermediate_size: 2048  # 1.6× hidden per expert

    # Router
    router:
        gate: Linear(1280, 8, bias=False)
        top_k: 2
        routing: softmax + top-k selection

    # 8 Expert FFNs (SwiGLU)
    experts[0..7]:
        gate_proj: Linear(1280, 2048, bias=False)
        up_proj: Linear(1280, 2048, bias=False)
        down_proj: Linear(2048, 1280, bias=False)

        params_per_expert: 3 × 1280 × 2048 = 7.86M
        total_experts: 8 × 7.86M = 62.9M

    # Shared Expert (always active)
    shared_expert:
        gate_proj: Linear(1280, 1280, bias=False)
        up_proj: Linear(1280, 1280, bias=False)
        down_proj: Linear(1280, 1280, bias=False)

        params: 3 × 1280 × 1280 = 4.92M

    params_per_layer:
        router: 0.01M
        experts: 62.9M
        shared: 4.92M
        total: 67.83M

    active_per_layer:
        router: 0.01M
        top-2 experts: 15.72M (2/8 × 62.9M)
        shared: 4.92M
        total: 20.65M

    total (6 layers): 407.0M params, 123.9M active
```

### 7. RMSNorm

```python
RMSNorm:
    hidden_size: 1280
    eps: 1e-6

    # Per layer: 3 norms (pre-attn, pre-cross, pre-ffn)
    params_per_layer: 3 × 1280 = 3.84K

    # Final norm before LM head
    final_norm: 1280

    total: 8 × 3.84K + 1.28K = 32K
```

### 8. LM Head

```python
LMHead:
    hidden_size: 1280
    vocab_size: 50280

    # Weights tied with token embeddings
    weight: shared with TokenEmbedding

    params: 0 (tied)
```

---

## Parameter Summary

```
┌──────────────────────────────────────────────────────────────────────────┐
│ COMPONENT                          │ TOTAL PARAMS │ ACTIVE PARAMS       │
├──────────────────────────────────────────────────────────────────────────┤
│ Token Embeddings (tied)            │ 64.4M        │ 64.4M               │
│ Encoder Projection                 │ 1.0M         │ 1.0M                │
├──────────────────────────────────────────────────────────────────────────┤
│ GQA Self-Attention (8 layers)      │ 31.4M        │ 31.4M               │
│ Cross-Attention (8 layers)         │ 52.4M        │ 52.4M               │
├──────────────────────────────────────────────────────────────────────────┤
│ Dense FFN (layers 0-1)             │ 27.5M        │ 27.5M               │
├──────────────────────────────────────────────────────────────────────────┤
│ MoE FFN (layers 2-7)               │              │                     │
│   - Router                         │ 0.06M        │ 0.06M               │
│   - 8 Experts × 6 layers           │ 377.4M       │ 94.4M (top-2)       │
│   - Shared Expert × 6 layers       │ 29.5M        │ 29.5M               │
├──────────────────────────────────────────────────────────────────────────┤
│ RMSNorm (all)                      │ 0.03M        │ 0.03M               │
│ LM Head (tied)                     │ 0            │ 0                   │
├──────────────────────────────────────────────────────────────────────────┤
│ DECODER TOTAL                      │ 419.7M       │ 236.7M              │
│ + Encoder (frozen)                 │ 155M         │ 155M                │
├──────────────────────────────────────────────────────────────────────────┤
│ FULL MODEL                         │ 574.7M       │ 391.7M              │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## MoE Robustness Features

### 1. Load Balancing Loss

```python
# Prevents expert collapse (all tokens going to 1 expert)
def load_balancing_loss(router_probs, expert_indices):
    # router_probs: (batch, seq, num_experts)
    # expert_indices: (batch, seq, top_k)

    # Fraction of tokens routed to each expert
    tokens_per_expert = expert_indices.flatten().bincount(minlength=num_experts)
    tokens_per_expert = tokens_per_expert / tokens_per_expert.sum()

    # Average router probability for each expert
    prob_per_expert = router_probs.mean(dim=[0, 1])

    # Auxiliary loss: minimize imbalance
    aux_loss = num_experts * (tokens_per_expert * prob_per_expert).sum()

    return aux_loss

weight: 0.01  # Added to main loss
```

### 2. Router Z-Loss

```python
# Prevents router logits from growing unbounded
def router_z_loss(router_logits):
    # router_logits: (batch, seq, num_experts)
    z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
    return z_loss

weight: 0.001
```

### 3. Expert Capacity

```python
# Prevents overflow during training
capacity_factor: 1.5  # Start high, can reduce to 1.25 later
max_tokens_per_expert = (total_tokens / num_experts) * capacity_factor

# Tokens exceeding capacity are handled by shared expert
```

### 4. Expert Dropout (Training Only)

```python
# Regularization: randomly skip experts
expert_dropout: 0.05  # 5% chance to skip non-selected experts
```

### 5. Router Initialization

```python
# Small uniform init prevents early expert bias
router.weight.data.uniform_(-0.01, 0.01)
```

---

## Configuration Schema

```python
@dataclass
class DecoderMoEConfig:
    """Configuration for UltraBERT-Gen MoE Decoder."""

    # Core dimensions
    hidden_size: int = 1280
    num_layers: int = 8
    vocab_size: int = 50280
    max_position_embeddings: int = 512

    # Attention
    num_attention_heads: int = 20
    num_kv_heads: int = 4  # GQA
    head_dim: int = 64
    attention_dropout: float = 0.0

    # FFN
    dense_layers: tuple = (0, 1)
    moe_layers: tuple = (2, 3, 4, 5, 6, 7)
    dense_intermediate_size: int = 3584  # 2.8× hidden

    # MoE
    num_experts: int = 8
    num_experts_per_token: int = 2
    expert_intermediate_size: int = 2048  # 1.6× hidden
    use_shared_expert: bool = True
    shared_expert_intermediate_size: int = 1280  # 1× hidden

    # Robustness
    load_balancing_loss_weight: float = 0.01
    router_z_loss_weight: float = 0.001
    capacity_factor: float = 1.5
    expert_dropout: float = 0.05

    # Encoder interface
    encoder_hidden_size: int = 768  # ModernBERT

    # RoPE
    rope_theta: float = 10000.0

    # Regularization
    hidden_dropout: float = 0.1

    # Weight tying
    tie_word_embeddings: bool = True
```

---

## Memory & Compute Estimates

### Training Memory (bf16, batch=8, seq=256)

| Component | Memory |
|-----------|--------|
| Encoder (frozen, fp16) | 0.3 GB |
| Decoder weights (bf16) | 0.84 GB |
| Decoder gradients (bf16) | 0.84 GB |
| Optimizer states (fp32) | 3.4 GB |
| Activations | 4.0 GB |
| KV Cache | 0.1 GB |
| **Total** | **~9.5 GB** |

### With Gradient Checkpointing

| Component | Memory |
|-----------|--------|
| Activations | 1.5 GB (saved) |
| **Total** | **~7.0 GB** |

### Inference Memory (bf16, batch=1)

| Component | Memory |
|-----------|--------|
| Full model (bf16) | 1.15 GB |
| KV Cache (seq=512) | 0.05 GB |
| Activations | 0.2 GB |
| **Total** | **~1.4 GB** |

---

## Expert Domain Specialization (Expected)

During training, experts naturally specialize:

| Expert | Expected Specialization | Related Domains |
|--------|------------------------|-----------------|
| E0 | Parenting & Children | discipline, education, bonding, toddlers, teens, milestones |
| E1 | Relationships | spouse, in-laws, extended family, conflicts, trust |
| E2 | Health & Wellness | sleep, nutrition, exercise, mental health, chronic |
| E3 | Daily Routines | morning, evening, meals, chores, self-care |
| E4 | Work & Career | boundaries, remote, burnout, career, childcare |
| E5 | Finances | budgeting, savings, debt, education funds |
| E6 | Emotions & Communication | stress, grief, anger, arguments, listening |
| E7 | Cultural & Family Events | festivals, rituals, heritage, weddings, traditions |
| **Shared** | Common patterns | Grammar, style, universal advice structures |

---

## File Structure

```
src/modeling_studio/models/
├── decoder_moe.py              # Main decoder module
├── moe_components.py           # MoE layer, router, experts
├── attention.py                # GQA, Cross-attention, RoPE
├── decoder_config.py           # DecoderMoEConfig dataclass
└── modernbert_with_decoder.py  # Unified model (encoder + decoder)

scripts/
└── train_stage_c.py            # Decoder training script

configs/training/
└── stage_c_decoder.yaml        # Training configuration
```

---

## Files Needed & Wiring Plan

### Existing Files to REUSE (No Modifications)

These files provide battle-tested infrastructure that the decoder will leverage directly:

| File | Purpose | Reuse Strategy |
|------|---------|----------------|
| [losses.py](../models/losses.py) | FocalLoss, LabelSmoothingCE, ASL | Use LabelSmoothingCrossEntropy for decoder LM loss |
| [poolers.py](../models/poolers.py) | CLSPooler, MeanPooler, AttentionPooler | Not needed for decoder (uses full sequence) |
| [task_weighting.py](../../trainers/task_weighting.py) | UncertaintyWeighting | Optional: weight decoder vs encoder losses |
| [callbacks.py](../../trainers/callbacks.py) | TaskMetricsCallback, GradientMonitorCallback | Extend for decoder-specific metrics |
| [evaluator.py](../../evaluation/evaluator.py) | Evaluation orchestration | Add decoder evaluation methods |

### Existing Files to MODIFY

| File | Location | Changes Required |
|------|----------|------------------|
| [labels.py](../../data/labels.py#L745) | `src/modeling_studio/data/` | Add `COUNTERFACTUAL = "counterfactual"` to `Capability` enum at line 745 |
| [modernbert_multitask.py](../models/modernbert_multitask.py#L129) | `src/modeling_studio/models/` | Add decoder to `CAPABILITY_TO_HEAD_TYPE` mapping at line 129 |
| [models/**init**.py](../models/__init__.py) | `src/modeling_studio/models/` | Export `CounterfactualDecoderHead`, `DecoderMoEConfig`, `MoELayer` |
| [loaders.py](../../data/loaders.py) | `src/modeling_studio/data/` | Add `load_counterfactual_dataset()` function |
| [collators.py](../../trainers/collators.py) | `src/modeling_studio/trainers/` | Add `CounterfactualCollator` for seq2seq batching |
| [multitask_trainer.py](../../trainers/multitask_trainer.py) | `src/modeling_studio/trainers/` | Handle decoder training mode with encoder freezing |

### NEW Files to CREATE

```
src/modeling_studio/
├── models/
│   ├── decoder_config.py           # DecoderMoEConfig dataclass (NEW)
│   ├── moe_components.py           # Router, Expert, MoELayer classes (NEW)
│   ├── decoder_moe.py              # CounterfactualDecoderHead module (NEW)
│   └── attention.py                # GQA, CrossAttention, RoPE (NEW)
│
├── data/
│   └── counterfactual_dataset.py   # Dataset for counterfactual pairs (NEW)
│
└── trainers/
    └── decoder_collator.py         # Seq2seq collator with encoder context (NEW)

scripts/
└── train_stage_c.py                # Decoder training script (NEW)

configs/training/multitask/
└── stage_c_decoder.yaml            # Training configuration (NEW)
```

---

### New File Specifications

#### 1. `decoder_config.py` (~100 lines)

```python
# src/modeling_studio/models/decoder_config.py
"""Configuration dataclass for UltraBERT-Gen MoE Decoder."""

from dataclasses import dataclass, field

@dataclass
class DecoderMoEConfig:
    """Configuration for the MoE decoder head."""

    # Core dimensions
    hidden_size: int = 1280
    num_layers: int = 8
    vocab_size: int = 50280
    max_position_embeddings: int = 512

    # Attention
    num_attention_heads: int = 20
    num_kv_heads: int = 4
    head_dim: int = 64
    attention_dropout: float = 0.0

    # FFN - Dense layers
    dense_layers: tuple[int, ...] = (0, 1)
    dense_intermediate_size: int = 3584

    # FFN - MoE layers
    moe_layers: tuple[int, ...] = (2, 3, 4, 5, 6, 7)
    num_experts: int = 8
    num_experts_per_token: int = 2
    expert_intermediate_size: int = 2048
    use_shared_expert: bool = True
    shared_expert_intermediate_size: int = 1280

    # MoE robustness
    load_balancing_loss_weight: float = 0.01
    router_z_loss_weight: float = 0.001
    capacity_factor: float = 1.5
    expert_dropout: float = 0.05

    # Encoder interface
    encoder_hidden_size: int = 768

    # RoPE
    rope_theta: float = 10000.0

    # Regularization
    hidden_dropout: float = 0.1
    tie_word_embeddings: bool = True
```

#### 2. `moe_components.py` (~400 lines)

```python
# src/modeling_studio/models/moe_components.py
"""MoE components: Router, Expert FFN, Shared Expert, MoE Layer."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUExpert(nn.Module):
    """Single SwiGLU expert FFN."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

class TopKRouter(nn.Module):
    """Top-K routing with load balancing loss."""
    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        ...
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        # Returns: (routing_weights, expert_indices, aux_losses)
        ...

class MoELayer(nn.Module):
    """Sparse Mixture-of-Experts FFN layer."""
    def __init__(self, config: DecoderMoEConfig, layer_idx: int):
        ...
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # Returns: (output, aux_losses)
        ...

class SharedExpert(nn.Module):
    """Always-active shared expert for common patterns."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        ...
```

#### 3. `attention.py` (~500 lines)

```python
# src/modeling_studio/models/attention.py
"""GQA Self-Attention, Cross-Attention, and RoPE implementations."""

import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0):
        ...
    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        ...

class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention with RoPE and causal masking."""
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int):
        ...
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        ...

class CrossAttention(nn.Module):
    """Cross-attention to encoder context."""
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        ...
    def forward(self, x: torch.Tensor, encoder_hidden_states: torch.Tensor, encoder_attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        ...
```

#### 4. `decoder_moe.py` (~600 lines)

```python
# src/modeling_studio/models/decoder_moe.py
"""UltraBERT-Gen MoE Decoder - 13th Head for counterfactual generation."""

import torch
import torch.nn as nn
from modeling_studio.models.decoder_config import DecoderMoEConfig
from modeling_studio.models.heads import BaseHead

class EncoderProjection(nn.Module):
    """Project encoder outputs (768) to decoder dimension (1280)."""
    def __init__(self, encoder_dim: int, decoder_dim: int):
        ...

class DecoderBlock(nn.Module):
    """Single decoder block: Self-Attn → Cross-Attn → FFN."""
    def __init__(self, config: DecoderMoEConfig, layer_idx: int):
        ...

class CounterfactualDecoderHead(BaseHead):
    """
    13th head for counterfactual generation.

    Inherits from BaseHead for loss computation compatibility.
    """
    def __init__(self, encoder_config, config: DecoderMoEConfig | None = None):
        ...

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            encoder_hidden_states: (B, S_enc, 768) from ModernBERT
            encoder_attention_mask: (B, S_enc)
            decoder_input_ids: (B, S_dec) target token IDs (teacher forcing)
            decoder_attention_mask: (B, S_dec)
            labels: (B, S_dec) for loss computation

        Returns:
            dict with 'loss', 'logits', 'aux_loss' (MoE balancing)
        """
        ...

    def generate(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        max_length: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Autoregressive generation with nucleus sampling."""
        ...
```

#### 5. `counterfactual_dataset.py` (~200 lines)

```python
# src/modeling_studio/data/counterfactual_dataset.py
"""Dataset for counterfactual generation training."""

import json
from pathlib import Path
from torch.utils.data import Dataset
import h5py  # for precomputed embeddings

class CounterfactualDataset(Dataset):
    """
    Dataset that pairs input text with counterfactual outputs.

    Supports two modes:
    1. Live encoder: Encodes input on-the-fly (slower, flexible)
    2. Precomputed: Loads encoder embeddings from HDF5 (faster, Stage C)
    """
    def __init__(
        self,
        samples_path: Path,
        embeddings_path: Path | None = None,  # HDF5 with precomputed
        tokenizer = None,
        max_input_length: int = 256,
        max_output_length: int = 256,
    ):
        ...

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            {
                'input_text': str,
                'encoder_embeddings': Tensor (S_enc, 768) if precomputed,
                'decoder_input_ids': Tensor (S_dec,),
                'labels': Tensor (S_dec,),
            }
        """
        ...
```

#### 6. `decoder_collator.py` (~150 lines)

```python
# src/modeling_studio/trainers/decoder_collator.py
"""Collator for seq2seq counterfactual generation."""

import torch
from dataclasses import dataclass

@dataclass
class CounterfactualCollator:
    """
    Collates counterfactual samples for decoder training.

    Handles:
    - Padding encoder sequences (from precomputed or live)
    - Creating decoder input_ids (with BOS token prepended)
    - Creating labels (with -100 for padding)
    - Creating attention masks
    """
    tokenizer: ...
    max_input_length: int = 256
    max_output_length: int = 256
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict]) -> dict:
        ...
```

#### 7. `train_stage_c.py` (~500 lines)

```python
# scripts/train_stage_c.py
"""
Stage C Training Script: Counterfactual Decoder

Trains the 13th head (MoE decoder) for counterfactual generation.

Training Strategy:
1. Load checkpoint-18000 with 12 heads
2. Freeze encoder + all 12 existing heads
3. Initialize and train decoder head only
4. Use precomputed encoder embeddings for efficiency

Usage:
    python scripts/train_stage_c.py \
        --config configs/training/multitask/stage_c_decoder.yaml
"""

import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer

from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel
from modeling_studio.models.decoder_moe import CounterfactualDecoderHead
from modeling_studio.models.decoder_config import DecoderMoEConfig
from modeling_studio.data.counterfactual_dataset import CounterfactualDataset
from modeling_studio.trainers.decoder_collator import CounterfactualCollator

def main():
    # 1. Load existing model
    model = ModernBertMultiTaskModel.load_checkpoint(
        "outputs/modernbert-v2-for-v3-transfer/checkpoint-18000"
    )

    # 2. Freeze encoder + existing heads
    for param in model.encoder.parameters():
        param.requires_grad = False
    for head in model.heads.values():
        for param in head.parameters():
            param.requires_grad = False

    # 3. Initialize decoder head
    decoder_config = DecoderMoEConfig()
    decoder_head = CounterfactualDecoderHead(
        encoder_config=model.config,
        config=decoder_config,
    )

    # 4. Add decoder to model
    model.add_head(Capability.COUNTERFACTUAL, decoder_head)

    # 5. Train with standard Trainer
    ...
```

#### 8. `stage_c_decoder.yaml` (~80 lines)

```yaml
# configs/training/multitask/stage_c_decoder.yaml
# Stage C: Counterfactual Decoder Training

model:
  checkpoint_path: "outputs/modernbert-v2-for-v3-transfer/checkpoint-18000"
  freeze_encoder: true
  freeze_existing_heads: true

decoder:
  hidden_size: 1280
  num_layers: 8
  num_attention_heads: 20
  num_kv_heads: 4
  num_experts: 8
  num_experts_per_token: 2
  use_shared_expert: true
  load_balancing_loss_weight: 0.01
  router_z_loss_weight: 0.001

data:
  train_path: "data/counterfactual/training"
  embeddings_mode: "precomputed"  # or "live"
  max_input_length: 256
  max_output_length: 256

training:
  output_dir: "outputs/ultrabert-gen-decoder-v1"
  num_train_epochs: 10
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 2e-4
  warmup_ratio: 0.1
  weight_decay: 0.01
  bf16: true
  gradient_checkpointing: true

  # Save best by perplexity
  metric_for_best_model: "eval_perplexity"
  greater_is_better: false
  save_strategy: "steps"
  save_steps: 1000
  eval_steps: 500

logging:
  report_to: "wandb"
  project: "ultrabert-gen"
  run_name: "stage-c-decoder-moe"
```

---

### Wiring Diagram

```
                    WIRING PLAN: 13th Head Integration
                    ═══════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: Update Capability Enum                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  File: src/modeling_studio/data/labels.py (line 745)                        │
│                                                                             │
│  class Capability(str, Enum):                                               │
│      NER_GENERAL = "ner_general"                                            │
│      SENTIMENT = "sentiment"                                                │
│      ...                                                                    │
│      INTENT = "intent"                                                      │
│  +   COUNTERFACTUAL = "counterfactual"   ← ADD THIS                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: Register Head Type Mapping                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  File: src/modeling_studio/models/modernbert_multitask.py (line 129)        │
│                                                                             │
│  from modeling_studio.models.decoder_moe import CounterfactualDecoderHead   │
│                                                                             │
│  CAPABILITY_TO_HEAD_TYPE = {                                                │
│      Capability.NER_GENERAL: TokenClassificationHead,                       │
│      ...                                                                    │
│      Capability.INTENT: IntentHead,                                         │
│  +   Capability.COUNTERFACTUAL: CounterfactualDecoderHead,  ← ADD THIS      │
│  }                                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: Create Decoder Modules                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  NEW FILES:                                                                 │
│  ├── decoder_config.py      → DecoderMoEConfig dataclass                    │
│  ├── moe_components.py      → TopKRouter, SwiGLUExpert, MoELayer            │
│  ├── attention.py           → RoPE, GQA, CrossAttention                     │
│  └── decoder_moe.py         → CounterfactualDecoderHead(BaseHead)           │
│                                                                             │
│  Key: CounterfactualDecoderHead MUST inherit from BaseHead                  │
│       to integrate with existing loss computation flow                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: Data Pipeline                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  EXISTING:                                                                  │
│  └── scripts/agents/prepare_decoder_training_data.py                        │
│      (Already creates embeddings.h5 + samples.jsonl)                        │
│                                                                             │
│  NEW FILES:                                                                 │
│  ├── counterfactual_dataset.py → Loads HDF5 embeddings + text pairs         │
│  └── decoder_collator.py       → Pads seq2seq batches                       │
│                                                                             │
│  ADD to loaders.py:                                                         │
│  def load_counterfactual_dataset(path, tokenizer, mode="precomputed"):      │
│      return CounterfactualDataset(...)                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: Training Integration                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  MODIFY multitask_trainer.py:                                               │
│  - Add decoder training mode flag                                           │
│  - Handle seq2seq loss (decoder returns 'loss' + 'aux_loss')                │
│  - Combine MoE auxiliary losses with main loss                              │
│                                                                             │
│  def compute_loss(self, model, inputs):                                     │
│      outputs = model(**inputs)                                              │
│      loss = outputs.loss                                                    │
│      if hasattr(outputs, 'aux_loss'):                                       │
│          loss = loss + outputs.aux_loss  # MoE balancing                    │
│      return loss                                                            │
│                                                                             │
│  NEW: scripts/train_stage_c.py                                              │
│  - Loads checkpoint-18000                                                   │
│  - Freezes encoder + 12 heads                                               │
│  - Trains decoder only                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 6: Export Updates                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  File: src/modeling_studio/models/__init__.py                               │
│                                                                             │
│  # Add to exports                                                           │
│  from modeling_studio.models.decoder_config import DecoderMoEConfig         │
│  from modeling_studio.models.decoder_moe import CounterfactualDecoderHead   │
│  from modeling_studio.models.moe_components import MoELayer, TopKRouter     │
│  from modeling_studio.models.attention import (                             │
│      GroupedQueryAttention, CrossAttention, RotaryEmbedding                 │
│  )                                                                          │
│                                                                             │
│  __all__ = [                                                                │
│      ...existing exports...,                                                │
│      "DecoderMoEConfig",                                                    │
│      "CounterfactualDecoderHead",                                           │
│      "MoELayer",                                                            │
│      "TopKRouter",                                                          │
│      "GroupedQueryAttention",                                               │
│      "CrossAttention",                                                      │
│      "RotaryEmbedding",                                                     │
│  ]                                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Training Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE C TRAINING FLOW                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐                                                        │
│  │ checkpoint-18000│ ─────────────────────────────────────┐                 │
│  │ (12 heads)      │                                      │                 │
│  └────────┬────────┘                                      │                 │
│           │                                               │                 │
│           ▼                                               │                 │
│  ┌─────────────────┐     ┌──────────────────┐            │                 │
│  │ Freeze Encoder  │     │ Freeze 12 Heads  │            │                 │
│  │ (155M params)   │     │ (existing tasks) │            │                 │
│  └────────┬────────┘     └────────┬─────────┘            │                 │
│           │                       │                       │                 │
│           └───────────┬───────────┘                       │                 │
│                       │                                   │                 │
│                       ▼                                   │                 │
│  ┌─────────────────────────────────────────┐             │                 │
│  │  Add CounterfactualDecoderHead (420M)   │◄────────────┘                 │
│  │  • Encoder projection (768→1280)        │                               │
│  │  • 8 decoder layers (2 dense + 6 MoE)   │                               │
│  │  • LM head (tied embeddings)            │                               │
│  └────────────────────┬────────────────────┘                               │
│                       │                                                     │
│                       ▼                                                     │
│  ┌─────────────────────────────────────────┐                               │
│  │  Load Precomputed Embeddings            │                               │
│  │  • data/counterfactual/training/        │                               │
│  │  • embeddings.h5 (N, 768) float16       │                               │
│  │  • samples.jsonl (input→counterfactual) │                               │
│  └────────────────────┬────────────────────┘                               │
│                       │                                                     │
│                       ▼                                                     │
│  ┌─────────────────────────────────────────┐                               │
│  │  Train Decoder Only (~420M trainable)   │                               │
│  │  • LR: 2e-4                             │                               │
│  │  • Epochs: 10                           │                               │
│  │  • Batch: 8 × 4 grad accum = 32         │                               │
│  │  • Loss: CE + 0.01 * load_balance       │                               │
│  │         + 0.001 * router_z              │                               │
│  └────────────────────┬────────────────────┘                               │
│                       │                                                     │
│                       ▼                                                     │
│  ┌─────────────────────────────────────────┐                               │
│  │  Output: ultrabert-gen-decoder-v1       │                               │
│  │  • 13 heads (12 frozen + 1 new)         │                               │
│  │  • Total: ~575M params                  │                               │
│  │  • Active: ~390M params                 │                               │
│  └─────────────────────────────────────────┘                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Implementation Order

```
Phase 1: Core Infrastructure (Week 1)
├── 1.1 decoder_config.py          # 2 hours - dataclass only
├── 1.2 moe_components.py          # 1 day - router, experts, MoE layer
├── 1.3 attention.py               # 1 day - RoPE, GQA, cross-attn
└── 1.4 decoder_moe.py             # 2 days - full decoder + BaseHead

Phase 2: Data & Integration (Week 2)
├── 2.1 counterfactual_dataset.py  # 4 hours - HDF5 loading
├── 2.2 decoder_collator.py        # 2 hours - padding logic
├── 2.3 Update labels.py           # 15 min - add enum value
├── 2.4 Update modernbert_multitask.py  # 30 min - head mapping
└── 2.5 Update models/__init__.py  # 15 min - exports

Phase 3: Training (Week 3)
├── 3.1 stage_c_decoder.yaml       # 1 hour - config file
├── 3.2 train_stage_c.py           # 4 hours - training script
├── 3.3 Update multitask_trainer.py  # 2 hours - decoder mode
└── 3.4 Run training               # 2-3 days (GPU time)

Phase 4: Evaluation & Polish (Week 4)
├── 4.1 Add decoder metrics        # 4 hours - perplexity, BLEU
├── 4.2 Generation tests           # 4 hours - quality checks
└── 4.3 Documentation              # 2 hours - update READMEs
```

---

### Quick Reference: File Counts

| Category | Count | Details |
|----------|-------|---------|
| **Files to REUSE** | 5 | losses.py, poolers.py, task_weighting.py, callbacks.py, evaluator.py |
| **Files to MODIFY** | 6 | labels.py, modernbert_multitask.py, **init**.py, loaders.py, collators.py, multitask_trainer.py |
| **NEW Python files** | 6 | decoder_config.py, moe_components.py, attention.py, decoder_moe.py, counterfactual_dataset.py, decoder_collator.py |
| **NEW Script** | 1 | train_stage_c.py |
| **NEW Config** | 1 | stage_c_decoder.yaml |
| **TOTAL NEW** | 8 files | ~2,050 lines estimated |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2025-12-15 | Initial architecture design |
| v1.1 | 2025-12-15 | Added Files Needed & Wiring Plan section |

---

## Implementation Plan: Milestones, Epics & Issues

### Overview

| Milestone | Name | Duration | Issues | Tests |
|-----------|------|----------|--------|-------|
| **M10** | Core MoE Components | 5 days | 6 issues | 18 tests |
| **M11** | Decoder Architecture | 5 days | 5 issues | 15 tests |
| **M12** | Data Pipeline | 3 days | 4 issues | 12 tests |
| **M13** | Training Infrastructure | 4 days | 5 issues | 14 tests |
| **M14** | Integration & Wiring | 3 days | 4 issues | 10 tests |
| **M15** | Evaluation & Quality | 3 days | 4 issues | 12 tests |
| **Total** | | **23 days** | **28 issues** | **81 tests** |

---

## Milestone 10: Core MoE Components

**Goal**: Build foundational MoE building blocks
**Duration**: 5 days
**Output**: `src/modeling_studio/models/moe_components.py`

### Epic 10.1: Router Implementation

#### Issue 10.1.1: TopKRouter Base Class

**File**: `src/modeling_studio/models/moe_components.py`
**Priority**: P0 (Blocker)
**Estimate**: 4 hours

**Implementation**:

```python
class TopKRouter(nn.Module):
    """Top-K router with softmax gating."""
    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2):
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.top_k = top_k
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        # Returns: (routing_weights, expert_indices, aux_losses)
```

**Acceptance Criteria**:

- [ ] AC1: Router returns top-k expert indices per token
- [ ] AC2: Routing weights sum to 1.0 for selected experts
- [ ] AC3: Small uniform initialization (-0.01, 0.01)

**Tests** (`tests/milestone10/test_router.py`):

```python
def test_router_initialization():
    """10.1.1-T1: Router gate weights are small uniform."""

def test_router_top_k_selection():
    """10.1.1-T2: Router selects exactly top_k experts."""

def test_router_weights_normalized():
    """10.1.1-T3: Selected expert weights sum to 1.0."""
```

---

#### Issue 10.1.2: Load Balancing Loss

**File**: `src/modeling_studio/models/moe_components.py`
**Priority**: P0 (Blocker)
**Estimate**: 3 hours
**Depends On**: 10.1.1

**Implementation**:

```python
def compute_load_balancing_loss(
    router_probs: torch.Tensor,
    expert_indices: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Auxiliary loss to prevent expert collapse."""
    # tokens_per_expert × prob_per_expert
```

**Acceptance Criteria**:

- [ ] AC1: Loss is 0 when tokens distributed perfectly evenly
- [ ] AC2: Loss increases when experts are imbalanced
- [ ] AC3: Gradient flows back to router

**Tests**:

```python
def test_load_balance_loss_even_distribution():
    """10.1.2-T1: Loss ~1.0 when perfectly balanced."""

def test_load_balance_loss_imbalanced():
    """10.1.2-T2: Loss > 1.0 when imbalanced."""

def test_load_balance_gradient_flow():
    """10.1.2-T3: Gradients reach router gate."""
```

---

#### Issue 10.1.3: Router Z-Loss

**File**: `src/modeling_studio/models/moe_components.py`
**Priority**: P1
**Estimate**: 2 hours
**Depends On**: 10.1.1

**Implementation**:

```python
def compute_router_z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    """Prevents router logits from growing unbounded."""
    return torch.logsumexp(router_logits, dim=-1).pow(2).mean()
```

**Acceptance Criteria**:

- [ ] AC1: Z-loss is scalar
- [ ] AC2: Z-loss increases with larger logits
- [ ] AC3: Z-loss is differentiable

**Tests**:

```python
def test_z_loss_scalar():
    """10.1.3-T1: Z-loss returns scalar tensor."""

def test_z_loss_increases_with_logits():
    """10.1.3-T2: Larger logits → larger z-loss."""
```

---

### Epic 10.2: Expert FFN Implementation

#### Issue 10.2.1: SwiGLU Expert

**File**: `src/modeling_studio/models/moe_components.py`
**Priority**: P0 (Blocker)
**Estimate**: 3 hours

**Implementation**:

```python
class SwiGLUExpert(nn.Module):
    """Single SwiGLU expert FFN."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

**Acceptance Criteria**:

- [ ] AC1: Output shape matches input shape
- [ ] AC2: No bias in any linear layer
- [ ] AC3: SiLU activation applied correctly

**Tests**:

```python
def test_swiglu_output_shape():
    """10.2.1-T1: Output shape == input shape."""

def test_swiglu_no_bias():
    """10.2.1-T2: All linear layers have bias=False."""

def test_swiglu_parameter_count():
    """10.2.1-T3: Params = 3 × hidden × intermediate."""
```

---

#### Issue 10.2.2: Shared Expert

**File**: `src/modeling_studio/models/moe_components.py`
**Priority**: P1
**Estimate**: 2 hours
**Depends On**: 10.2.1

**Implementation**:

```python
class SharedExpert(nn.Module):
    """Always-active shared expert."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        self.expert = SwiGLUExpert(hidden_size, intermediate_size)
```

**Acceptance Criteria**:

- [ ] AC1: Shared expert always processes all tokens
- [ ] AC2: Output added to sparse expert output

**Tests**:

```python
def test_shared_expert_always_active():
    """10.2.2-T1: Shared expert processes all tokens."""

def test_shared_expert_output_added():
    """10.2.2-T2: Shared output added to MoE output."""
```

---

#### Issue 10.2.3: MoELayer Assembly

**File**: `src/modeling_studio/models/moe_components.py`
**Priority**: P0 (Blocker)
**Estimate**: 6 hours
**Depends On**: 10.1.1, 10.1.2, 10.2.1, 10.2.2

**Implementation**:

```python
class MoELayer(nn.Module):
    """Complete sparse MoE FFN layer."""
    def __init__(self, config: DecoderMoEConfig, layer_idx: int):
        self.router = TopKRouter(config.hidden_size, config.num_experts, config.num_experts_per_token)
        self.experts = nn.ModuleList([
            SwiGLUExpert(config.hidden_size, config.expert_intermediate_size)
            for _ in range(config.num_experts)
        ])
        self.shared_expert = SharedExpert(...) if config.use_shared_expert else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # Efficient batched expert computation
```

**Acceptance Criteria**:

- [ ] AC1: MoELayer routes tokens to top-k experts
- [ ] AC2: Returns (output, aux_losses_dict)
- [ ] AC3: Capacity factor limits tokens per expert
- [ ] AC4: Shared expert output combined correctly

**Tests**:

```python
def test_moe_layer_routing():
    """10.2.3-T1: Tokens routed to exactly top_k experts."""

def test_moe_layer_aux_losses():
    """10.2.3-T2: aux_losses contains load_balance and z_loss."""

def test_moe_layer_capacity_factor():
    """10.2.3-T3: Excess tokens handled by shared expert."""

def test_moe_layer_output_shape():
    """10.2.3-T4: Output shape matches input."""
```

---

## Milestone 11: Decoder Architecture

**Goal**: Build decoder blocks and full decoder
**Duration**: 5 days
**Output**: `src/modeling_studio/models/decoder_moe.py`, `attention.py`, `decoder_config.py`

### Epic 11.1: Configuration

#### Issue 11.1.1: DecoderMoEConfig Dataclass

**File**: `src/modeling_studio/models/decoder_config.py`
**Priority**: P0 (Blocker)
**Estimate**: 2 hours

**Implementation**:

```python
@dataclass
class DecoderMoEConfig:
    hidden_size: int = 1280
    num_layers: int = 8
    vocab_size: int = 50280
    # ... all config fields
```

**Acceptance Criteria**:

- [ ] AC1: All architecture params have sensible defaults
- [ ] AC2: Config is JSON serializable
- [ ] AC3: Config validates constraints (e.g., num_kv_heads divides num_heads)

**Tests** (`tests/milestone11/test_decoder_config.py`):

```python
def test_config_defaults():
    """11.1.1-T1: Default config matches architecture spec."""

def test_config_serialization():
    """11.1.1-T2: Config round-trips through JSON."""

def test_config_validation():
    """11.1.1-T3: Invalid configs raise ValueError."""
```

---

### Epic 11.2: Attention Mechanisms

#### Issue 11.2.1: Rotary Position Embedding (RoPE)

**File**: `src/modeling_studio/models/attention.py`
**Priority**: P0 (Blocker)
**Estimate**: 4 hours

**Implementation**:

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0):
        # Precompute sin/cos tables

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Return (cos, sin) for RoPE application
```

**Acceptance Criteria**:

- [ ] AC1: RoPE produces position-dependent rotations
- [ ] AC2: Different positions have different embeddings
- [ ] AC3: Works with arbitrary sequence lengths up to max

**Tests** (`tests/milestone11/test_attention.py`):

```python
def test_rope_different_positions():
    """11.2.1-T1: Different positions have different cos/sin."""

def test_rope_deterministic():
    """11.2.1-T2: Same position always gives same embedding."""

def test_rope_max_length():
    """11.2.1-T3: Works at max_seq_len boundary."""
```

---

#### Issue 11.2.2: Grouped Query Attention (GQA)

**File**: `src/modeling_studio/models/attention.py`
**Priority**: P0 (Blocker)
**Estimate**: 6 hours
**Depends On**: 11.2.1

**Implementation**:

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int):
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size)
        self.rotary_emb = RotaryEmbedding(head_dim)

    def forward(self, x, attention_mask=None, past_key_value=None):
        # GQA with RoPE and causal masking
```

**Acceptance Criteria**:

- [ ] AC1: KV heads are shared across query head groups
- [ ] AC2: Causal mask prevents attending to future
- [ ] AC3: KV cache supported for inference
- [ ] AC4: RoPE applied to Q and K

**Tests**:

```python
def test_gqa_kv_sharing():
    """11.2.2-T1: KV heads shared correctly (5:1 ratio)."""

def test_gqa_causal_mask():
    """11.2.2-T2: Cannot attend to future positions."""

def test_gqa_kv_cache():
    """11.2.2-T3: KV cache enables incremental decoding."""

def test_gqa_rope_applied():
    """11.2.2-T4: Q and K have RoPE applied."""
```

---

#### Issue 11.2.3: Cross-Attention

**File**: `src/modeling_studio/models/attention.py`
**Priority**: P0 (Blocker)
**Estimate**: 4 hours

**Implementation**:

```python
class CrossAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)  # From encoder
        self.v_proj = nn.Linear(hidden_size, hidden_size)  # From encoder
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, encoder_hidden_states, encoder_attention_mask=None):
        # Cross-attend to encoder context
```

**Acceptance Criteria**:

- [ ] AC1: Q from decoder, K/V from encoder
- [ ] AC2: No causal mask (attend to all encoder positions)
- [ ] AC3: Encoder mask applied correctly

**Tests**:

```python
def test_cross_attention_qkv_sources():
    """11.2.3-T1: Q from decoder, K/V from encoder."""

def test_cross_attention_no_causal():
    """11.2.3-T2: All encoder positions visible."""

def test_cross_attention_encoder_mask():
    """11.2.3-T3: Padding tokens masked correctly."""
```

---

### Epic 11.3: Decoder Blocks

#### Issue 11.3.1: DecoderBlock (Dense + MoE variants)

**File**: `src/modeling_studio/models/decoder_moe.py`
**Priority**: P0 (Blocker)
**Estimate**: 6 hours
**Depends On**: 11.2.2, 11.2.3, 10.2.3

**Implementation**:

```python
class DecoderBlock(nn.Module):
    def __init__(self, config: DecoderMoEConfig, layer_idx: int):
        self.self_attn = GroupedQueryAttention(...)
        self.cross_attn = CrossAttention(...)
        self.norm1 = RMSNorm(config.hidden_size)
        self.norm2 = RMSNorm(config.hidden_size)
        self.norm3 = RMSNorm(config.hidden_size)

        # Dense or MoE FFN based on layer_idx
        if layer_idx in config.dense_layers:
            self.ffn = DenseSwiGLUFFN(...)
        else:
            self.ffn = MoELayer(...)
```

**Acceptance Criteria**:

- [ ] AC1: Layers 0-1 use dense FFN
- [ ] AC2: Layers 2-7 use MoE FFN
- [ ] AC3: Pre-norm architecture (RMSNorm before each sublayer)
- [ ] AC4: Residual connections after each sublayer

**Tests** (`tests/milestone11/test_decoder_block.py`):

```python
def test_decoder_block_dense_layers():
    """11.3.1-T1: Layers 0-1 have DenseSwiGLUFFN."""

def test_decoder_block_moe_layers():
    """11.3.1-T2: Layers 2-7 have MoELayer."""

def test_decoder_block_prenorm():
    """11.3.1-T3: RMSNorm applied before sublayers."""

def test_decoder_block_residual():
    """11.3.1-T4: Residual connections preserve gradients."""
```

---

#### Issue 11.3.2: CounterfactualDecoderHead

**File**: `src/modeling_studio/models/decoder_moe.py`
**Priority**: P0 (Blocker)
**Estimate**: 8 hours
**Depends On**: 11.1.1, 11.3.1

**Implementation**:

```python
class CounterfactualDecoderHead(BaseHead):
    """13th head for counterfactual generation."""

    def __init__(self, encoder_config, config: DecoderMoEConfig | None = None):
        super().__init__(hidden_size=config.hidden_size, num_labels=config.vocab_size)
        self.encoder_projection = EncoderProjection(encoder_config.hidden_size, config.hidden_size)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderBlock(config, i) for i in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Tie weights
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, encoder_hidden_states, encoder_attention_mask,
                decoder_input_ids=None, labels=None) -> dict:
        # Full forward with loss computation

    def generate(self, encoder_hidden_states, encoder_attention_mask,
                 max_length=128, temperature=0.7, top_p=0.9) -> torch.Tensor:
        # Autoregressive generation
```

**Acceptance Criteria**:

- [ ] AC1: Inherits from BaseHead for compatibility
- [ ] AC2: Weight tying between embeddings and LM head
- [ ] AC3: Returns dict with 'loss', 'logits', 'aux_loss'
- [ ] AC4: generate() produces valid token sequences
- [ ] AC5: Total params ~420M

**Tests** (`tests/milestone11/test_decoder_head.py`):

```python
def test_decoder_inherits_basehead():
    """11.3.2-T1: CounterfactualDecoderHead inherits BaseHead."""

def test_decoder_weight_tying():
    """11.3.2-T2: lm_head.weight is embed_tokens.weight."""

def test_decoder_forward_output_format():
    """11.3.2-T3: Forward returns dict with loss, logits, aux_loss."""

def test_decoder_generate_valid_tokens():
    """11.3.2-T4: Generated tokens are in vocab range."""

def test_decoder_parameter_count():
    """11.3.2-T5: Total params ~420M (±5%)."""
```

---

## Milestone 12: Data Pipeline

**Goal**: Create dataset and collator for counterfactual training
**Duration**: 3 days
**Output**: `src/modeling_studio/data/counterfactual_dataset.py`, `trainers/decoder_collator.py`

### Epic 12.1: Dataset Implementation

#### Issue 12.1.1: CounterfactualDataset (Precomputed Mode)

**File**: `src/modeling_studio/data/counterfactual_dataset.py`
**Priority**: P0 (Blocker)
**Estimate**: 4 hours

**Implementation**:

```python
class CounterfactualDataset(Dataset):
    def __init__(self, samples_path: Path, embeddings_path: Path, tokenizer,
                 max_input_length=256, max_output_length=256):
        self.samples = self._load_jsonl(samples_path)
        self.embeddings = h5py.File(embeddings_path, 'r')['embeddings']
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        return {
            'encoder_embeddings': torch.tensor(self.embeddings[idx]),
            'decoder_input_ids': self._tokenize_output(self.samples[idx]['counterfactual_full_text']),
            'labels': ...,
        }
```

**Acceptance Criteria**:

- [ ] AC1: Loads samples from JSONL
- [ ] AC2: Loads precomputed embeddings from HDF5
- [ ] AC3: Tokenizes counterfactual text with BOS/EOS
- [ ] AC4: Labels shifted for causal LM (-100 for padding)

**Tests** (`tests/milestone12/test_counterfactual_dataset.py`):

```python
def test_dataset_loads_samples():
    """12.1.1-T1: Dataset loads samples from JSONL."""

def test_dataset_loads_embeddings():
    """12.1.1-T2: Dataset loads embeddings from HDF5."""

def test_dataset_tokenization():
    """12.1.1-T3: Output text tokenized with BOS/EOS."""

def test_dataset_label_shifting():
    """12.1.1-T4: Labels are shifted, padding is -100."""
```

---

#### Issue 12.1.2: CounterfactualDataset (Live Encoder Mode)

**File**: `src/modeling_studio/data/counterfactual_dataset.py`
**Priority**: P2
**Estimate**: 3 hours
**Depends On**: 12.1.1

**Implementation**:

```python
class CounterfactualDataset(Dataset):
    def __init__(self, ..., encoder=None, mode='precomputed'):
        if mode == 'live':
            self.encoder = encoder
            self.embeddings = None

    def __getitem__(self, idx):
        if self.mode == 'live':
            encoder_embeddings = self.encoder.encode(self.samples[idx]['input_text'])
        else:
            encoder_embeddings = self.embeddings[idx]
```

**Acceptance Criteria**:

- [ ] AC1: Live mode computes embeddings on-the-fly
- [ ] AC2: Same output format as precomputed mode
- [ ] AC3: Encoder is called with input_text

**Tests**:

```python
def test_live_mode_encodes_text():
    """12.1.2-T1: Live mode calls encoder.encode()."""

def test_live_mode_same_format():
    """12.1.2-T2: Live mode output matches precomputed format."""
```

---

### Epic 12.2: Collator Implementation

#### Issue 12.2.1: CounterfactualCollator

**File**: `src/modeling_studio/trainers/decoder_collator.py`
**Priority**: P0 (Blocker)
**Estimate**: 4 hours

**Implementation**:

```python
@dataclass
class CounterfactualCollator:
    tokenizer: ...
    max_input_length: int = 256
    max_output_length: int = 256
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict]) -> dict:
        # Pad encoder embeddings to max length in batch
        # Pad decoder input_ids
        # Create attention masks
        # Ensure labels have -100 for padding
```

**Acceptance Criteria**:

- [ ] AC1: Pads encoder embeddings to batch max
- [ ] AC2: Pads decoder sequences to batch max
- [ ] AC3: Creates encoder_attention_mask
- [ ] AC4: Creates decoder_attention_mask
- [ ] AC5: Labels use -100 for padding positions

**Tests** (`tests/milestone12/test_decoder_collator.py`):

```python
def test_collator_pads_encoder():
    """12.2.1-T1: Encoder embeddings padded to batch max."""

def test_collator_pads_decoder():
    """12.2.1-T2: Decoder sequences padded to batch max."""

def test_collator_encoder_mask():
    """12.2.1-T3: Encoder mask is 0 for padding."""

def test_collator_label_padding():
    """12.2.1-T4: Labels are -100 for padding positions."""
```

---

#### Issue 12.2.2: Loader Function

**File**: `src/modeling_studio/data/loaders.py`
**Priority**: P1
**Estimate**: 2 hours
**Depends On**: 12.1.1

**Implementation**:

```python
def load_counterfactual_dataset(
    path: Path,
    tokenizer,
    mode: str = 'precomputed',
    encoder = None,
    max_input_length: int = 256,
    max_output_length: int = 256,
) -> CounterfactualDataset:
    """Load counterfactual dataset for Stage C training."""
```

**Acceptance Criteria**:

- [ ] AC1: Function exported from loaders.py
- [ ] AC2: Returns CounterfactualDataset instance
- [ ] AC3: Validates path exists

**Tests**:

```python
def test_load_counterfactual_exists():
    """12.2.2-T1: load_counterfactual_dataset is importable."""

def test_load_counterfactual_returns_dataset():
    """12.2.2-T2: Returns CounterfactualDataset instance."""
```

---

## Milestone 13: Training Infrastructure

**Goal**: Training script and trainer modifications
**Duration**: 4 days
**Output**: `scripts/train_stage_c.py`, `configs/training/multitask/stage_c_decoder.yaml`

### Epic 13.1: Configuration

#### Issue 13.1.1: Stage C Config File

**File**: `configs/training/multitask/stage_c_decoder.yaml`
**Priority**: P0 (Blocker)
**Estimate**: 2 hours

**Implementation**:

```yaml
# Stage C: Counterfactual Decoder Training
model:
  checkpoint_path: "outputs/modernbert-v2-for-v3-transfer/checkpoint-18000"
  freeze_encoder: true
  freeze_existing_heads: true

decoder:
  hidden_size: 1280
  num_layers: 8
  num_attention_heads: 20
  num_kv_heads: 4
  num_experts: 8
  num_experts_per_token: 2
  use_shared_expert: true
  load_balancing_loss_weight: 0.01
  router_z_loss_weight: 0.001

data:
  train_path: "data/counterfactual/training"
  embeddings_mode: "precomputed"
  max_input_length: 256
  max_output_length: 256

training:
  output_dir: "outputs/ultrabert-gen-decoder-v1"
  num_train_epochs: 10
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 2e-4
  warmup_ratio: 0.1
  bf16: true
  gradient_checkpointing: true
```

**Acceptance Criteria**:

- [ ] AC1: Config parseable by OmegaConf
- [ ] AC2: All decoder params specified
- [ ] AC3: Checkpoint path valid
- [ ] AC4: Training hyperparams match architecture doc

**Tests** (`tests/milestone13/test_stage_c_config.py`):

```python
def test_config_loads():
    """13.1.1-T1: Config loads without errors."""

def test_config_decoder_params():
    """13.1.1-T2: Decoder params match architecture."""

def test_config_freezing_flags():
    """13.1.1-T3: freeze_encoder and freeze_existing_heads are True."""
```

---

### Epic 13.2: Trainer Modifications

#### Issue 13.2.1: Decoder Training Mode

**File**: `src/modeling_studio/trainers/multitask_trainer.py`
**Priority**: P0 (Blocker)
**Estimate**: 4 hours

**Implementation**:

```python
class MultiTaskTrainer:
    def __init__(self, ..., decoder_mode: bool = False):
        self.decoder_mode = decoder_mode

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss

        # Add MoE auxiliary losses in decoder mode
        if self.decoder_mode and hasattr(outputs, 'aux_loss'):
            loss = loss + outputs.aux_loss

        return (loss, outputs) if return_outputs else loss
```

**Acceptance Criteria**:

- [ ] AC1: decoder_mode flag added to trainer
- [ ] AC2: aux_loss added to total loss when present
- [ ] AC3: Logs aux_loss separately to wandb

**Tests** (`tests/milestone13/test_trainer_decoder_mode.py`):

```python
def test_trainer_decoder_mode_flag():
    """13.2.1-T1: Trainer accepts decoder_mode flag."""

def test_trainer_aux_loss_added():
    """13.2.1-T2: aux_loss added to total loss."""

def test_trainer_logs_aux_loss():
    """13.2.1-T3: aux_loss logged separately."""
```

---

#### Issue 13.2.2: Encoder Freezing Utility

**File**: `src/modeling_studio/trainers/multitask_trainer.py`
**Priority**: P0 (Blocker)
**Estimate**: 2 hours

**Implementation**:

```python
def freeze_encoder_and_heads(model, freeze_encoder=True, freeze_heads=True):
    """Freeze encoder and/or existing heads for Stage C."""
    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
    if freeze_heads:
        for name, head in model.heads.items():
            if name != 'counterfactual':
                for param in head.parameters():
                    param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {trainable:,}")
```

**Acceptance Criteria**:

- [ ] AC1: Encoder params frozen when flag True
- [ ] AC2: Existing 12 heads frozen when flag True
- [ ] AC3: Counterfactual head remains trainable
- [ ] AC4: Logs trainable param count

**Tests**:

```python
def test_freeze_encoder():
    """13.2.2-T1: Encoder params require_grad=False."""

def test_freeze_existing_heads():
    """13.2.2-T2: 12 existing heads require_grad=False."""

def test_decoder_head_trainable():
    """13.2.2-T3: Counterfactual head require_grad=True."""
```

---

### Epic 13.3: Training Script

#### Issue 13.3.1: train_stage_c.py Main Script

**File**: `scripts/train_stage_c.py`
**Priority**: P0 (Blocker)
**Estimate**: 6 hours
**Depends On**: 13.1.1, 13.2.1, 13.2.2

**Implementation**:

```python
#!/usr/bin/env python
"""Stage C Training Script: Counterfactual Decoder"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    # 1. Load checkpoint-18000
    model = ModernBertMultiTaskModel.load_checkpoint(config.model.checkpoint_path)

    # 2. Initialize decoder head
    decoder_config = DecoderMoEConfig(**config.decoder)
    decoder_head = CounterfactualDecoderHead(model.config, decoder_config)
    model.add_head(Capability.COUNTERFACTUAL, decoder_head)

    # 3. Freeze encoder + existing heads
    freeze_encoder_and_heads(model, config.model.freeze_encoder, config.model.freeze_existing_heads)

    # 4. Load dataset
    dataset = load_counterfactual_dataset(config.data.train_path, tokenizer)

    # 5. Train
    trainer = MultiTaskTrainer(model, ..., decoder_mode=True)
    trainer.train()

    # 6. Save
    model.save_pretrained(config.training.output_dir)

if __name__ == '__main__':
    main()
```

**Acceptance Criteria**:

- [ ] AC1: Script loads config from YAML
- [ ] AC2: Loads checkpoint-18000 correctly
- [ ] AC3: Initializes decoder with config params
- [ ] AC4: Freezes encoder + 12 heads
- [ ] AC5: Trains decoder only
- [ ] AC6: Saves final checkpoint

**Tests** (`tests/milestone13/test_train_stage_c.py`):

```python
def test_script_parses_args():
    """13.3.1-T1: Script parses --config argument."""

def test_script_loads_checkpoint():
    """13.3.1-T2: Script loads checkpoint-18000."""

def test_script_initializes_decoder():
    """13.3.1-T3: Decoder head initialized with config."""

def test_script_freezes_correctly():
    """13.3.1-T4: Encoder + 12 heads frozen."""
```

---

## Milestone 14: Integration & Wiring

**Goal**: Wire decoder into existing codebase
**Duration**: 3 days
**Output**: Updated `labels.py`, `modernbert_multitask.py`, `models/__init__.py`

### Epic 14.1: Capability Registration

#### Issue 14.1.1: Add COUNTERFACTUAL Capability

**File**: `src/modeling_studio/data/labels.py`
**Priority**: P0 (Blocker)
**Estimate**: 1 hour

**Implementation**:

```python
class Capability(str, Enum):
    NER_GENERAL = "ner_general"
    # ... existing 12 ...
    INTENT = "intent"
    COUNTERFACTUAL = "counterfactual"  # NEW
```

**Acceptance Criteria**:

- [ ] AC1: COUNTERFACTUAL added to Capability enum
- [ ] AC2: Value is "counterfactual"
- [ ] AC3: Enum has 13 members total

**Tests** (`tests/milestone14/test_capability_enum.py`):

```python
def test_counterfactual_capability_exists():
    """14.1.1-T1: Capability.COUNTERFACTUAL exists."""

def test_capability_count():
    """14.1.1-T2: Capability enum has 13 members."""
```

---

#### Issue 14.1.2: Register Head Type Mapping

**File**: `src/modeling_studio/models/modernbert_multitask.py`
**Priority**: P0 (Blocker)
**Estimate**: 1 hour
**Depends On**: 14.1.1, 11.3.2

**Implementation**:

```python
from modeling_studio.models.decoder_moe import CounterfactualDecoderHead

CAPABILITY_TO_HEAD_TYPE = {
    # ... existing 12 ...
    Capability.INTENT: IntentHead,
    Capability.COUNTERFACTUAL: CounterfactualDecoderHead,  # NEW
}
```

**Acceptance Criteria**:

- [ ] AC1: CounterfactualDecoderHead imported
- [ ] AC2: Mapping includes COUNTERFACTUAL
- [ ] AC3: No import errors

**Tests**:

```python
def test_counterfactual_head_mapped():
    """14.1.2-T1: COUNTERFACTUAL maps to CounterfactualDecoderHead."""

def test_import_no_errors():
    """14.1.2-T2: Import succeeds without errors."""
```

---

### Epic 14.2: Module Exports

#### Issue 14.2.1: Update models/**init**.py

**File**: `src/modeling_studio/models/__init__.py`
**Priority**: P1
**Estimate**: 1 hour
**Depends On**: 10.2.3, 11.3.2

**Implementation**:

```python
from modeling_studio.models.decoder_config import DecoderMoEConfig
from modeling_studio.models.decoder_moe import CounterfactualDecoderHead, EncoderProjection
from modeling_studio.models.moe_components import MoELayer, TopKRouter, SwiGLUExpert
from modeling_studio.models.attention import GroupedQueryAttention, CrossAttention, RotaryEmbedding

__all__ = [
    # ... existing ...
    "DecoderMoEConfig",
    "CounterfactualDecoderHead",
    "MoELayer",
    "TopKRouter",
    "GroupedQueryAttention",
    "CrossAttention",
    "RotaryEmbedding",
]
```

**Acceptance Criteria**:

- [ ] AC1: All new classes exported
- [ ] AC2: Import from models package works

**Tests**:

```python
def test_decoder_classes_exported():
    """14.2.1-T1: Decoder classes importable from models."""

def test_moe_classes_exported():
    """14.2.1-T2: MoE classes importable from models."""
```

---

#### Issue 14.2.2: Add Collator Export

**File**: `src/modeling_studio/trainers/collators.py`
**Priority**: P1
**Estimate**: 1 hour
**Depends On**: 12.2.1

**Implementation**:

```python
# At end of collators.py
from modeling_studio.trainers.decoder_collator import CounterfactualCollator

__all__ = [
    # ... existing ...
    "CounterfactualCollator",
]
```

**Acceptance Criteria**:

- [ ] AC1: CounterfactualCollator importable from collators
- [ ] AC2: No circular imports

**Tests**:

```python
def test_collator_exported():
    """14.2.2-T1: CounterfactualCollator importable from trainers.collators."""
```

---

## Milestone 15: Evaluation & Quality

**Goal**: Add decoder evaluation metrics and quality tests
**Duration**: 3 days
**Output**: Evaluation utilities, quality benchmarks

### Epic 15.1: Generation Metrics

#### Issue 15.1.1: Perplexity Calculation

**File**: `src/modeling_studio/evaluation/decoder_metrics.py`
**Priority**: P0 (Blocker)
**Estimate**: 3 hours

**Implementation**:

```python
def compute_perplexity(model, dataloader, device='cuda') -> float:
    """Compute perplexity on validation set."""
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            outputs = model(**batch.to(device))
            total_loss += outputs.loss.item() * batch['labels'].ne(-100).sum()
            total_tokens += batch['labels'].ne(-100).sum()

    return torch.exp(total_loss / total_tokens).item()
```

**Acceptance Criteria**:

- [ ] AC1: Returns perplexity as float
- [ ] AC2: Handles -100 padding in labels
- [ ] AC3: Works with batched evaluation

**Tests** (`tests/milestone15/test_decoder_metrics.py`):

```python
def test_perplexity_calculation():
    """15.1.1-T1: Perplexity computed correctly."""

def test_perplexity_ignores_padding():
    """15.1.1-T2: -100 labels excluded from calculation."""
```

---

#### Issue 15.1.2: BLEU Score Calculation

**File**: `src/modeling_studio/evaluation/decoder_metrics.py`
**Priority**: P1
**Estimate**: 3 hours

**Implementation**:

```python
from sacrebleu import corpus_bleu

def compute_bleu(predictions: list[str], references: list[str]) -> float:
    """Compute corpus BLEU score."""
    return corpus_bleu(predictions, [references]).score
```

**Acceptance Criteria**:

- [ ] AC1: Uses sacrebleu for consistent scores
- [ ] AC2: Returns BLEU as float (0-100 scale)
- [ ] AC3: Handles empty predictions gracefully

**Tests**:

```python
def test_bleu_perfect_match():
    """15.1.2-T1: BLEU = 100 for identical text."""

def test_bleu_no_match():
    """15.1.2-T2: BLEU ~ 0 for completely different text."""
```

---

### Epic 15.2: Quality Benchmarks

#### Issue 15.2.1: Generation Quality Test Suite

**File**: `tests/milestone15/test_generation_quality.py`
**Priority**: P0 (Blocker)
**Estimate**: 4 hours

**Implementation**:

```python
class TestGenerationQuality:
    """Quality benchmarks for counterfactual generation."""

    @pytest.fixture
    def golden_samples(self):
        """Load golden test samples."""
        return [
            {
                "input": "I yelled at my kids and now I feel terrible.",
                "expected_counterfactual_contains": ["calm", "alternative", "outcome"],
            },
            # ... more samples
        ]

    def test_generation_coherent():
        """15.2.1-T1: Generated text is grammatically coherent."""

    def test_generation_preserves_entities():
        """15.2.1-T2: Family entities preserved in counterfactual."""

    def test_generation_changes_outcome():
        """15.2.1-T3: Counterfactual shows different outcome."""
```

**Acceptance Criteria**:

- [ ] AC1: 10+ golden test samples defined
- [ ] AC2: Tests check coherence
- [ ] AC3: Tests check entity preservation
- [ ] AC4: Tests check outcome change

**Tests**:

```python
def test_generation_coherent():
    """15.2.1-T1: Generated text is grammatically coherent."""

def test_generation_preserves_entities():
    """15.2.1-T2: Family entities preserved in counterfactual."""

def test_generation_changes_outcome():
    """15.2.1-T3: Counterfactual shows different outcome."""
```

---

#### Issue 15.2.2: MoE Expert Utilization Test

**File**: `tests/milestone15/test_expert_utilization.py`
**Priority**: P1
**Estimate**: 3 hours

**Implementation**:

```python
class TestExpertUtilization:
    """Tests for MoE expert load balancing."""

    def test_expert_utilization_balanced():
        """15.2.2-T1: Experts receive roughly equal tokens."""

    def test_no_expert_collapse():
        """15.2.2-T2: No expert receives 0 tokens."""

    def test_shared_expert_always_active():
        """15.2.2-T3: Shared expert processes all tokens."""
```

**Acceptance Criteria**:

- [ ] AC1: Each expert gets 10-15% of tokens (for 8 experts)
- [ ] AC2: No expert gets 0 tokens
- [ ] AC3: Shared expert gets 100% of tokens

**Tests**:

```python
def test_expert_utilization_balanced():
    """15.2.2-T1: Experts receive roughly equal tokens."""

def test_no_expert_collapse():
    """15.2.2-T2: No expert receives 0 tokens."""

def test_shared_expert_always_active():
    """15.2.2-T3: Shared expert processes all tokens."""
```

---

## Implementation Schedule

```
Week 1 (Days 1-5): Milestone 10 + 11
├── Day 1: Issues 10.1.1, 10.1.2, 10.1.3 (Router)
├── Day 2: Issues 10.2.1, 10.2.2 (Experts)
├── Day 3: Issue 10.2.3 (MoELayer), 11.1.1 (Config)
├── Day 4: Issues 11.2.1, 11.2.2 (RoPE, GQA)
└── Day 5: Issues 11.2.3, 11.3.1 (CrossAttn, DecoderBlock)

Week 2 (Days 6-10): Milestone 11 (cont) + 12 + 13
├── Day 6: Issue 11.3.2 (CounterfactualDecoderHead)
├── Day 7: Issues 12.1.1, 12.1.2 (Dataset)
├── Day 8: Issues 12.2.1, 12.2.2 (Collator, Loader)
├── Day 9: Issues 13.1.1, 13.2.1 (Config, Trainer)
└── Day 10: Issues 13.2.2, 13.3.1 (Freezing, Script)

Week 3 (Days 11-15): Milestone 14 + 15 + Training
├── Day 11: Issues 14.1.1, 14.1.2 (Capability, Mapping)
├── Day 12: Issues 14.2.1, 14.2.2 (Exports)
├── Day 13: Issues 15.1.1, 15.1.2 (Metrics)
├── Day 14: Issues 15.2.1, 15.2.2 (Quality Tests)
└── Day 15: Buffer / Integration testing

Week 4 (Days 16-23): Training + Polish
├── Days 16-21: GPU training (~6 days at 10 epochs)
├── Day 22: Evaluation, threshold tuning
└── Day 23: Documentation, final review
```

---

## Dependency Graph

```
                            ┌─────────────────────────────────────────┐
                            │  MILESTONE 10: Core MoE Components      │
                            │                                         │
                            │  10.1.1 TopKRouter ──┬──► 10.1.2 LoadBal│
                            │         │            │                  │
                            │         │            └──► 10.1.3 Z-Loss │
                            │         │                               │
                            │  10.2.1 SwiGLUExpert ──► 10.2.2 Shared  │
                            │         │                     │         │
                            │         └───────────┬─────────┘         │
                            │                     ▼                   │
                            │              10.2.3 MoELayer            │
                            └────────────────────┬────────────────────┘
                                                 │
                                                 ▼
                            ┌─────────────────────────────────────────┐
                            │  MILESTONE 11: Decoder Architecture     │
                            │                                         │
                            │  11.1.1 Config                          │
                            │         │                               │
                            │  11.2.1 RoPE ──► 11.2.2 GQA             │
                            │                     │                   │
                            │  11.2.3 CrossAttn ──┼───► 11.3.1 Block  │
                            │                     │           │       │
                            │                     └───────────┼───────┤
                            │                                 ▼       │
                            │                     11.3.2 DecoderHead  │
                            └────────────────────┬────────────────────┘
                                                 │
                    ┌────────────────────────────┼────────────────────────────┐
                    │                            │                            │
                    ▼                            ▼                            ▼
┌─────────────────────────────┐  ┌─────────────────────────────┐  ┌─────────────────────────────┐
│  MILESTONE 12: Data         │  │  MILESTONE 13: Training     │  │  MILESTONE 14: Integration  │
│                             │  │                             │  │                             │
│  12.1.1 Dataset ──► 12.1.2  │  │  13.1.1 Config              │  │  14.1.1 Capability          │
│         │                   │  │         │                   │  │         │                   │
│         └──► 12.2.1 Collator│  │  13.2.1 Trainer ──► 13.2.2  │  │  14.1.2 Mapping ──► 14.2.1  │
│                    │        │  │         │                   │  │                    │        │
│              12.2.2 Loader  │  │  13.3.1 Script              │  │              14.2.2 Export  │
└────────────────────┬────────┘  └────────────────────┬────────┘  └────────────────────┬────────┘
                     │                                │                                │
                     └───────────────┬────────────────┴────────────────┬───────────────┘
                                     │                                 │
                                     ▼                                 ▼
                            ┌─────────────────────────────────────────┐
                            │  MILESTONE 15: Evaluation & Quality     │
                            │                                         │
                            │  15.1.1 Perplexity ──► 15.1.2 BLEU      │
                            │                             │           │
                            │  15.2.1 QualityTests ◄──────┘           │
                            │         │                               │
                            │  15.2.2 ExpertUtilization               │
                            └─────────────────────────────────────────┘
```

---

## Test Summary by Milestone

| Milestone | Test File | Tests | Coverage |
|-----------|-----------|-------|----------|
| M10 | `tests/milestone10/test_router.py` | 8 | Router, losses |
| M10 | `tests/milestone10/test_experts.py` | 6 | SwiGLU, MoELayer |
| M10 | `tests/milestone10/test_moe_integration.py` | 4 | Full MoE |
| M11 | `tests/milestone11/test_decoder_config.py` | 3 | Config |
| M11 | `tests/milestone11/test_attention.py` | 10 | RoPE, GQA, Cross |
| M11 | `tests/milestone11/test_decoder_block.py` | 4 | DecoderBlock |
| M11 | `tests/milestone11/test_decoder_head.py` | 5 | Full head |
| M12 | `tests/milestone12/test_counterfactual_dataset.py` | 6 | Dataset |
| M12 | `tests/milestone12/test_decoder_collator.py` | 4 | Collator |
| M13 | `tests/milestone13/test_stage_c_config.py` | 3 | Config |
| M13 | `tests/milestone13/test_trainer_decoder_mode.py` | 5 | Trainer |
| M13 | `tests/milestone13/test_train_stage_c.py` | 4 | Script |
| M14 | `tests/milestone14/test_capability_enum.py` | 2 | Capability |
| M14 | `tests/milestone14/test_integration.py` | 4 | Wiring |
| M15 | `tests/milestone15/test_decoder_metrics.py` | 4 | Metrics |
| M15 | `tests/milestone15/test_generation_quality.py` | 3 | Quality |
| M15 | `tests/milestone15/test_expert_utilization.py` | 3 | MoE balance |
| **Total** | | **81** | |

---

## Quality Gates

### Gate 1: After Milestone 10 (MoE Components)

```bash
pytest tests/milestone10/ -v
# All 18 tests must pass
# MoELayer output shape correct
# Load balance loss < 1.5 on uniform input
```

### Gate 2: After Milestone 11 (Decoder)

```bash
pytest tests/milestone11/ -v
# All 22 tests must pass
# CounterfactualDecoderHead params = 420M ± 5%
# generate() produces valid tokens
```

### Gate 3: After Milestone 12-13 (Data + Training)

```bash
pytest tests/milestone12/ tests/milestone13/ -v
# All 22 tests must pass
# train_stage_c.py runs without errors (1 epoch test)
```

### Gate 4: After Milestone 14-15 (Integration + Quality)

```bash
pytest tests/milestone14/ tests/milestone15/ -v
# All 16 tests must pass
# COUNTERFACTUAL capability registered
# Perplexity < 50 on validation set
```

### Final Gate: Before Deployment

```bash
pytest tests/ -v --ignore=tests/v3  # Full test suite
# All 81+ tests pass
# Perplexity < 20 on validation set
# Expert utilization balanced (no collapse)
# BLEU > 30 on golden set
```

---

## References

- Mixtral 8x7B (Mistral AI) - MoE architecture
- LLaMA 2/3 (Meta) - GQA, RoPE, SwiGLU
- Switch Transformer (Google) - Load balancing, capacity factor
- GPT-4 (OpenAI) - Top-2 routing speculation
