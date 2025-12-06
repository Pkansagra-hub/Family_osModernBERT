# FamilyOS Unified Encoder - Implementation Plan v3 (Ultra)

> **Base Model:** `answerdotai/ModernBERT-base` (Apache 2.0) → Extended to 28 layers
> **Goal:** Multi-task encoder with 12 capabilities + Hub Token routing + Sliding Window Attention
> **Reference:** `enhanced_design_v3.md`
> **Updated:** December 2025
> **Strategy:** Function Preserving Growth (Weight Transfer from v2)

---

## 🎯 Training Strategy Overview

### v3.3 Critical Fixes Addressed

| Fix | Problem | Solution | Implementation |
|-----|---------|----------|----------------|
| **Blind Hub** | Hub tokens can't see beyond sliding window | Global Bidirectional Attention for positions 0-4 | `attention_v3.py` |
| **Random Init Waste** | Random hub embeddings burn training steps | Semantic Centroid Initialization | `hub_initialization_v3.py` |
| **Transplant Rejection** | L22→L23 interface mismatch | Enhanced Phase 0.5 Healing with Zipper LR | `trainer_v3.py` |
| **Catastrophic Forgetting** | FamilyOS-only training forgets English | 15% Stage A Replay in training mix | Training config |
| **Feature Distribution Shift** | Task-only healing causes overfitting | 5-Task Structural Healing (SST2, CoNLL, MNLI, SQuAD, STS-B) | `prepare_healing_data_enhanced.py` |
| **Embedding Collapse** | Embeddings drift during fine-tuning | STS-B similarity data in healing mix | Healing config |

### Enhanced Phase 0.5: Zipper Healing Strategy

| Feature | Standard Plan | **Enhanced Plan (Ultra)** | Why Better |
|---------|---------------|---------------------------|------------|
| **Data Scope** | 3 Tasks (Sentiment, NER, NLI) | **5 Tasks (+SQuAD, +STS-B)** | Prevents overfitting to classification; forces context understanding |
| **Learning Rate** | Constant `1e-5` | **Linear Warmup + Cosine Decay** | Prevents "shock" at step 1; settles weights gently |
| **Layer Strategy** | Train L19-28 equally | **Zipper LR (differential by layer)** | L23 (interface) needs more plasticity than L19 (feeder) |
| **Batching** | Standard | **Gradient Clipping (1.0)** | Prevents exploding gradients at L22→L23 interface |

### Zipper Layer Strategy (Differential Learning Rates)

| Layers | Role | Learning Rate | Rationale |
|--------|------|---------------|-----------|
| 1-18 | Foundation | ❄️ Frozen | Core v2 knowledge preserved |
| 19-22 | Feeders | `1e-5` (Low) | Nudge outputs to match L23 expectations |
| **23** | **Interface** | **`5e-5` (High)** | Maximum plasticity to fix the "scar tissue" |
| 24-28 | Clones | `3e-5` (Medium) | Adapt to new signals from L23 |

### Training Phases

| Phase | Description | Layers Trainable | Data | LR Strategy |
|-------|-------------|------------------|------|-------------|
| 0 (Init) | Build v3 from v2 checkpoint | None | None | N/A |
| 0.5 (Heal) | Structural healing with Zipper LR | L19-28 (differential) | 5-Task Mix (12k samples) | Warmup + Cosine |
| 1 (Train) | Multi-task training | L19-28 + Hub + Heads | 85% FamilyOS + 15% Stage A | Standard |
| 1.5 (Eval) | Forgetting evaluation | None (eval only) | Benchmarks | N/A |

### Enhanced Healing Data Mix (Phase 0.5)

| Dataset | Task | Samples | Purpose |
|---------|------|---------|---------|
| SST-2 | Sentiment | 3,000 | Classification grounding |
| CoNLL-2003 | NER | 3,000 | Structural/syntax understanding |
| MNLI | NLI | 2,000 | Logic and reasoning |
| SQuAD | QA/Context | 2,000 | Attention mechanism healing (span finding) |
| STS-B | Similarity | 2,000 | Embedding stability (prevents collapse) |
| **Total** | **5 Tasks** | **12,000** | **Comprehensive structural healing** |

---

## ⚠️ Risk Assessment & Mandatory Mitigations

> **CRITICAL**: The following risks have been identified through architecture review. These are NOT optional fixes—they are **mandatory** for successful v3 training.

### Risk Matrix

| Risk | Severity | Status | Mitigation |
|------|----------|--------|------------|
| Flash Attention + Global Hubs | 🔴 **BLOCKER** | Fixed | Use pure PyTorch `MultiScaleAttentionWithGlobals` for training |
| Function-Preserving Growth Collapse | 🔴 **HIGH** | Mitigated | Zipper Healing Phase 0.5 is **MANDATORY** |
| Embedding Collapse | 🔴 **HIGH** | Mitigated | STS-B in healing mix is **MANDATORY** |
| L22→L23 Exploding Gradients | 🔴 **HIGH** | Mitigated | Gradient clipping (1.0) + Zipper LR |
| Catastrophic Forgetting | 🟡 **MEDIUM** | Mitigated | 15% Stage A replay (monitor, may need 20-25%) |
| 8k Sequence Stability | 🟡 **MEDIUM** | Planned | Gradient checkpointing + ZeRO-3 |

---

### Risk 1: Flash Attention + Global Hubs (BLOCKER)

**Problem:** Flash Attention 2 with `window_size` parameter breaks Text → Hub visibility. When a text token at position 500 uses a 64-token window, it CANNOT attend to hub tokens at positions 1-4.

```
Text token at position 500, window=64:
  ✗ Cannot see [EMO] at position 1 (outside window)
  ✗ Cannot see [MEM] at position 2 (outside window)
  ✗ Hub tokens become "invisible" to distant text
```

**Root Cause:** Flash Attention 2 cannot efficiently handle arbitrary sparse attention patterns. The `window_size` parameter creates a strict sliding window with no exceptions.

**MANDATORY FIX: The Safety Switch** (See Issue 2.1.4 for full implementation)

| Phase | Implementation | Why |
|-------|----------------|-----|
| Training (all) | `MultiScaleAttentionWithGlobals` + SDPA | Correctness is non-negotiable |
| Inference <2k | `MultiScaleAttentionWithGlobals` | Negligible speed difference |
| Inference 8k+ | `FlashAttentionWithGlobals` | Accept Text→Hub blindness |

**Key Optimization:** Standard attention uses `F.scaled_dot_product_attention` (SDPA) which is memory-efficient and often accelerated in PyTorch 2.0+, preventing OOM errors.

**Config Enforcement:**

```yaml
# configs/training/v3_phase1.yaml
model:
  attention:
    use_flash_attention: false  # ⚠️ CRITICAL: Preserve Text->Hub attention
```

**Acceptance Test:**

```python
def test_text_can_see_hub_tokens():
    """MUST PASS: Text at position 500 can attend to [EMO] at position 1."""
    attention = create_attention_layer(layer_idx=1, use_flash_attention=False)

    hidden = torch.randn(1, 512, 768)
    output, weights = attention(hidden, output_attentions=True)

    # Position 500 attending to position 1 ([EMO])
    assert weights[0, :, 500, 1].mean() > 0, "Text cannot see [EMO] hub token!"
```

---

### Risk 2: Function-Preserving Growth Collapse (HIGH)

**Problem:** Cloning L15-20 → L23-28 creates a "scar" at the L22→L23 interface. Layer 23 (cloned from L15) expects L14-style input features, but receives L22-style features that have been "rotated" by 8 extra layers of processing.

**Without healing:** 5-10 point drop on GLUE-style benchmarks.

**Evidence:** This pattern is well-documented in:

- DeepSeek-V2 (layer extension via cloning)
- Llama-3 MoE (upcycling from dense)
- OpenAI scaling laws papers

**MANDATORY:** Phase 0.5 Zipper Healing is NOT optional. You MUST run it.

```yaml
# Phase 0.5 is MANDATORY - DO NOT SKIP
phase_0_5:
  enabled: true  # ← MUST be true
  skip: false    # ← NEVER set to true

  # If you skip this, expect:
  # - 5-10 pt drop on MNLI
  # - 3-5 pt drop on SST-2
  # - Unstable training in Phase 1
```

---

### Risk 3: Embedding Collapse (HIGH)

**Problem:** During fine-tuning, embeddings can "collapse" to a narrow subspace, losing representational diversity. This manifests as:

- STS-B scores dropping 10+ points
- Embedding retrieval quality degrading
- Memory/relation tasks failing

**MANDATORY:** Keep STS-B (2,000 samples) in Phase 0.5 healing mix.

```python
# In prepare_healing_data_enhanced.py

healing_mix = {
    "sst2": 3000,
    "conll": 3000,
    "mnli": 2000,
    "squad": 2000,
    "stsb": 2000,   # ← DO NOT REMOVE - prevents embedding collapse
}

# STS-B forces the model to maintain:
# 1. Sentence-level semantic similarity understanding
# 2. Embedding space coherence
# 3. [MEM] hub token utility for retrieval
```

---

### Risk 4: L22→L23 Exploding Gradients (HIGH)

**Problem:** The interface between L22 (v2 original) and L23 (cloned from L15) has a feature distribution mismatch. This causes gradient magnitudes to spike during early training.

**Symptoms:**

- Loss spikes at steps 50-200
- NaN gradients
- Training divergence

**MANDATORY:** Keep the following Phase 0.5 settings:

```yaml
phase_0_5:
  gradient_clipping: 1.0   # ← MANDATORY - prevents explosion

  learning_rate:
    layers_19_22: 1e-5     # Feeders: gentle
    layer_23: 5e-5         # Interface: HIGH plasticity (the fix!)
    layers_24_28: 3e-5     # Clones: moderate

  warmup_steps: 500        # ← MANDATORY - prevents step-1 shock
```

**Why this works:** High LR on L23 gives it maximum plasticity to adapt its input expectations. Low LR on L19-22 gently nudges their outputs. This is the "Zipper" method used by DeepSeek, Llama-3-MoE, etc.

---

### Risk 5: Catastrophic Forgetting of English (MEDIUM)

**Problem:** Training only on FamilyOS data causes the model to "forget" general English understanding.

**Current Mitigation:** 15% Stage A replay in Phase 1.

**Monitoring Required:**

```python
# In trainer_v3.py - add forgetting monitor

class ForgettingMonitor:
    """Monitor for catastrophic forgetting during Phase 1."""

    THRESHOLDS = {
        "mnli": 0.02,   # Max 2% drop
        "squad": 0.02,  # Max 2% drop
        "sst2": 0.02,   # Max 2% drop
    }

    def check_and_adjust(self, current_metrics, baseline_metrics):
        for task, threshold in self.THRESHOLDS.items():
            drop = baseline_metrics[task] - current_metrics[task]

            if drop > threshold:
                print(f"⚠️ FORGETTING DETECTED: {task} dropped {drop:.1%}")
                print(f"   Recommendation: Increase replay ratio to 20-25%")
                return {"increase_replay": True, "new_ratio": 0.25}

        return {"increase_replay": False}
```

**Be Ready to Adjust:**

```yaml
# If forgetting detected, update Phase 1 config:
data_mix:
  familyos_ratio: 0.75  # Reduce from 0.85
  stage_a_replay_ratio: 0.25  # Increase from 0.15
```

---

### Risk 6: Training Stability on 8k Sequences (MEDIUM)

**Problem:** 8192-token sequences with 28 layers consume massive GPU memory. OOM errors likely without optimizations.

**MANDATORY for 8k training:**

```yaml
# In training config

training:
  # Memory optimizations (REQUIRED for 8k)
  gradient_checkpointing: true

  # DeepSpeed ZeRO-3 (REQUIRED for multi-GPU)
  deepspeed:
    zero_stage: 3
    offload_optimizer: true
    offload_param: false  # Keep params on GPU for speed

  # Batch size management
  per_device_batch_size: 1      # Small due to memory
  gradient_accumulation: 64     # Effective batch = 64

  # Mixed precision (REQUIRED)
  fp16: false
  bf16: true  # Use BF16 for stability
```

**Memory Estimate (per GPU, 8k context):**

- Hidden states: 28 layers × 8192 tokens × 768 dim × 4 bytes = ~700MB
- Attention: 12 heads × 8192² × 28 layers ≈ Huge without windows
- With 512-window on top layers: ~60% reduction
- With gradient checkpointing: ~50% activation memory reduction

---

## 📁 Project File Inventory

### New v3 Model Files (🆕 Required)

| File | Status | Purpose |
|------|--------|---------|
| `src/modeling_studio/models/config_v3.py` | 📝 NEW | v3 configuration (28 layers, hub tokens, sliding windows) |
| `src/modeling_studio/models/hub_tokens.py` | 📝 NEW | Hub token definitions and capability mappings |
| `src/modeling_studio/models/hub_initialization_v3.py` | 📝 NEW | Semantic centroid initialization for hub tokens |
| `src/modeling_studio/models/tokenization_v3.py` | 📝 NEW | Hub token injection wrapper |
| `src/modeling_studio/models/attention_v3.py` | 📝 NEW | MHA + sliding windows + global hub tokens |
| `src/modeling_studio/models/ffn_v3.py` | 📝 NEW | GELU FFN (modularized) |
| `src/modeling_studio/models/layers_v3.py` | 📝 NEW | v3 transformer layer with LoRA attachment |
| `src/modeling_studio/models/lora_v3.py` | 📝 NEW | LoRA implementation for layers 23-28 |
| `src/modeling_studio/models/poolers_v3.py` | 📝 NEW | Hub token pooler |
| `src/modeling_studio/models/pair_encoder_v3.py` | 📝 NEW | Cross-attention with [REL] hub + span masks |
| `src/modeling_studio/models/modernbert_v3.py` | 📝 NEW | Main v3 model (28-layer integration) |
| `src/modeling_studio/models/initialization_v3.py` | 📝 NEW | Function Preserving Growth from v2 |

### v3 Training Files (🆕 Required)

| File | Status | Purpose |
|------|--------|---------|
| `src/modeling_studio/trainers/trainer_v3.py` | 📝 NEW | Phase-based trainer with layer freezing |
| `src/modeling_studio/trainers/hub_token_trainer.py` | 📝 NEW | Hub token specific training utilities |
| `src/modeling_studio/trainers/zipper_lr.py` | 📝 NEW | Zipper LR strategy (differential by layer) |
| `src/modeling_studio/trainers/healing_scheduler.py` | 📝 NEW | Warmup + Cosine decay for Phase 0.5 |
| `src/modeling_studio/data/collators_v3.py` | 📝 NEW | Collators with hub token offset handling |
| `src/modeling_studio/data/loaders_v3.py` | 📝 NEW | Unified FamilyOS loader with hub_routing support |
| `src/modeling_studio/data/unified_dataset.py` | 📝 NEW | Dataset class for unified JSONL format |
| `src/modeling_studio/data/healing_dataset.py` | 📝 NEW | Enhanced healing dataset (5-task mix) |

### v3 Scripts (🆕 Required)

| File | Status | Purpose |
|------|--------|---------|
| `scripts/prepare_healing_data.py` | 📝 NEW | Basic Phase 0.5 healing data (3 tasks) |
| `scripts/prepare_healing_data_enhanced.py` | 📝 NEW | Enhanced healing data (5 tasks: +SQuAD, +STS-B) |
| `scripts/initialize_v3_from_v2.py` | 📝 NEW | Build v3 model from v2 checkpoint |
| `scripts/train_v3.py` | 📝 NEW | Multi-phase v3 training script |
| `scripts/verify_function_preserving.py` | 📝 NEW | Verify L1-22 output matches v2 |
| `scripts/validate_unified_data.py` | 📝 NEW | Validate generated unified JSONL format |

### v3 Config Files (🆕 Required)

| File | Status | Purpose |
|------|--------|---------|
| `configs/model/encoder/modernbert_v3_ultra.yaml` | 📝 NEW | v3 model configuration |
| `configs/training/multitask/stage_v3_phase0.yaml` | 📝 NEW | Phase 0 initialization config |
| `configs/training/multitask/stage_v3_phase0_5.yaml` | 📝 NEW | Basic Phase 0.5 healing config |
| `configs/training/multitask/stage_v3_phase0_5_enhanced.yaml` | 📝 NEW | Enhanced healing with Zipper LR + warmup |
| `configs/training/multitask/stage_v3_phase1.yaml` | 📝 NEW | Phase 1 training config |
| `configs/data/multitask/healing_datasets.yaml` | 📝 NEW | Basic healing data config |
| `configs/data/multitask/healing_enhanced.yaml` | 📝 NEW | Enhanced 5-task healing config |
| `configs/data/multitask/familyos_unified.yaml` | 📝 NEW | Unified FamilyOS data config (shard_*.jsonl) |

### Generated Data Directory (🗂️ Data Source)

| Path | Status | Purpose |
|------|--------|---------|
| `data/familyos/unified/output/shard_*.jsonl` | ✅ Generated | Unified training data with hub_routing |
| `data/healing/healing_generic.jsonl` | 📝 To Generate | Basic healing (3 tasks: SST2, CoNLL, MNLI) |
| `data/healing/healing_enhanced.jsonl` | 📝 To Generate | Enhanced healing (5 tasks: +SQuAD, +STS-B) |

### Reused from v2 (✅ No Changes)

| File | Status | Purpose |
|------|--------|---------|
| `src/modeling_studio/data/labels.py` | ✅ Reuse | Label schemas for all 12 capabilities |
| `src/modeling_studio/data/loaders.py` | ✅ Reuse | All 12 dataset loaders |
| `src/modeling_studio/data/multitask_dataset.py` | ✅ Reuse | MultiTaskDataset |
| `src/modeling_studio/data/preprocessing.py` | ✅ Reuse | TextPreprocessor |
| `src/modeling_studio/data/augmentation.py` | ✅ Reuse | FamilyAugmenter |
| `src/modeling_studio/models/heads.py` | ✅ Reuse | All 9 head types |
| `src/modeling_studio/models/losses.py` | ✅ Reuse | All loss functions |
| `src/modeling_studio/models/adapters.py` | ✅ Reuse | LoRA, Bottleneck adapters |
| `src/modeling_studio/trainers/task_sampler.py` | ✅ Reuse | 5 samplers + factory |
| `src/modeling_studio/trainers/ema.py` | ✅ Reuse | EMA model |
| `src/modeling_studio/trainers/task_weighting.py` | ✅ Reuse | Uncertainty weighting |
| `src/modeling_studio/trainers/callbacks.py` | ✅ Reuse | Training callbacks |
| `src/modeling_studio/evaluation/metrics.py` | ✅ Reuse | All 12 metric functions |
| `src/modeling_studio/evaluation/evaluator.py` | ✅ Reuse | Evaluator class |
| `src/modeling_studio/evaluation/benchmarks.py` | ✅ Reuse | Benchmark suite |
| `src/modeling_studio/evaluation/safety_eval.py` | ✅ Reuse | Safety evaluation |
| `src/modeling_studio/evaluation/forgetting_eval.py` | ✅ Reuse | Forgetting evaluation |

### Extended from v2 (⚠️ Modifications Required)

| File | Status | Changes Needed |
|------|--------|----------------|
| `src/modeling_studio/data/tokenization.py` | ⚠️ Extend | Add `get_v3_tokenizer()` wrapper |
| `src/modeling_studio/trainers/collators.py` | ⚠️ Extend | Handle hub token position offsets |
| `src/modeling_studio/trainers/optimizer.py` | ⚠️ Extend | Add layer-group LRs (L19-22 vs L23-28) |
| `src/modeling_studio/models/poolers.py` | ⚠️ Extend | Add `HubTokenPooler` |
| `src/modeling_studio/inference/unified_output.py` | ⚠️ Extend | Update for hub token routing |
| `src/modeling_studio/k0/runtime/model_registry.py` | ⚠️ Extend | Add `familyos_unified_v3` entry |

---

## 🔌 Complete Wiring Documentation

### Overview: Component Assembly & Data Flow

This section provides a complete wiring diagram showing how all v3 components connect together, from tokenization through to task heads. Use this as the authoritative reference for understanding the full model architecture integration.

---

### 1. Component Inventory

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPONENT INVENTORY                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  LAYER 0: INPUT PIPELINE                                                    │
│  ├── HubTokenizer (tokenization_v3.py) → Injects [EMO][MEM][REL][TASK]     │
│  └── ModernBERTEmbeddingsV3 (embeddings_v3.py) → Word + Position embeds    │
│                                                                              │
│  LAYER 1: ENCODER STACK (28 layers)                                         │
│  ├── ModernBERTEncoderV3 (encoder_v3.py) → Layer orchestrator               │
│  ├── ModernBERTLayerV3 (layers_v3.py) → Single transformer layer            │
│  ├── MultiScaleAttentionWithGlobals (attention_v3.py) → MHA + sliding       │
│  ├── GELUFFN (ffn_v3.py) → Feed-forward network                             │
│  └── LoRALayer (lora_v3.py) → Adapters for L23-28                           │
│                                                                              │
│  LAYER 2: POOLING                                                           │
│  ├── HubTokenPooler (poolers_v3.py) → Extracts [EMO][MEM][REL][TASK]       │
│  ├── CombinedPooler (poolers_v3.py) → CLS + Mean + Hub                      │
│  └── PairEncoderV3 (pair_encoder_v3.py) → NLI/Relation fusion               │
│                                                                              │
│  LAYER 3: TASK HEADS (heads_v3.py)                                          │
│  ├── HubAwareClassificationHead → emotions, sentiment, safety, intent       │
│  ├── HubAwareTokenClassificationHead → NER, temporal                        │
│  ├── HierarchicalSafetyHead → GREEN→AMBER→RED→CRISIS                        │
│  └── EmbeddingHead → [MEM] → similarity/retrieval                           │
│                                                                              │
│  ORCHESTRATOR: ModernBERTv3Ultra (modernbert_v3.py)                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 2. Data Flow Wiring

```
INPUT: "Mom is feeling sad today"
        ↓
┌───────────────────────────────────────────────────────────────────────────┐
│  STEP 1: TOKENIZATION (HubTokenizer)                                      │
│  Raw text → [CLS][EMO][MEM][REL][TASK] Mom is feeling sad today [SEP]    │
│  Positions:   0    1    2    3    4     5   6    7     8   9     10      │
└───────────────────────────────────────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────────────────────────────────────┐
│  STEP 2: EMBEDDINGS (ModernBERTEmbeddingsV3)                              │
│  input_ids → word_embeddings → LayerNorm → dropout                        │
│  Output: [batch, seq_len, 768]                                            │
│                                                                            │
│  Hub token embeddings initialized via SEMANTIC CENTROID:                  │
│  • [EMO] ← avg("emotion", "feeling", "happy", "sad", "angry")             │
│  • [MEM] ← avg("memory", "remember", "recall", "store", "retrieve")       │
│  • [REL] ← avg("relation", "family", "mother", "father", "sibling")       │
│  • [TASK] ← avg("task", "intent", "action", "request", "command")         │
└───────────────────────────────────────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────────────────────────────────────┐
│  STEP 3: ENCODER (ModernBERTEncoderV3 - 28 layers)                        │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐      │
│  │  FOUNDATION BAND (L1-6) | Window: 64 | ❄️ FROZEN                │      │
│  │  → MultiScaleAttentionWithGlobals(window=64, global_tokens=[0-4])│      │
│  │  → GELUFFN(intermediate=3072)                                    │      │
│  │  → LayerNorm + Residual                                          │      │
│  └─────────────────────────────────────────────────────────────────┘      │
│        ↓                                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐      │
│  │  CONTEXT BAND (L7-18) | Window: 128 | ❄️ FROZEN                 │      │
│  │  → Same structure as Foundation                                  │      │
│  └─────────────────────────────────────────────────────────────────┘      │
│        ↓                                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐      │
│  │  SEMANTIC BAND (L19-22) | Window: 256 | 🔥 TRAINABLE            │      │
│  │  → Same structure, trainable weights                             │      │
│  └─────────────────────────────────────────────────────────────────┘      │
│        ↓                                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐      │
│  │  FAMILY BAND (L23-28) | Window: 512 | 🔥 TRAINABLE + LoRA       │      │
│  │  → LoRALayer(r=16, alpha=16) on q_proj, k_proj, v_proj, o_proj  │      │
│  │  → Cloned from v2 L15-20 weights                                 │      │
│  └─────────────────────────────────────────────────────────────────┘      │
│                                                                            │
│  GLOBAL ATTENTION: Hub tokens (pos 0-4) attend to ALL tokens              │
│                    ALL tokens attend to hub tokens                         │
│  Output: [batch, seq_len, 768]                                            │
└───────────────────────────────────────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────────────────────────────────────┐
│  STEP 4: POOLING (HubTokenPooler + CombinedPooler)                        │
│                                                                            │
│  last_hidden_state: [batch, seq_len, 768]                                 │
│        ↓                                                                   │
│  pooled_outputs = {                                                        │
│      "[CLS]": last_hidden_state[:, 0, :],   # [batch, 768]                │
│      "[EMO]": last_hidden_state[:, 1, :],   # [batch, 768]                │
│      "[MEM]": last_hidden_state[:, 2, :],   # [batch, 768]                │
│      "[REL]": last_hidden_state[:, 3, :],   # [batch, 768]                │
│      "[TASK]": last_hidden_state[:, 4, :],  # [batch, 768]                │
│      "mean": mean_pool(last_hidden_state),  # [batch, 768]                │
│  }                                                                         │
└───────────────────────────────────────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────────────────────────────────────┐
│  STEP 5: HUB ROUTING (get_representation_for_capability)                  │
│                                                                            │
│  capability="emotions" → hub_token="[EMO]" → pooled_outputs["[EMO]"]      │
│  capability="sentiment" → hub_token="[EMO]" → pooled_outputs["[EMO]"]     │
│  capability="safety_*" → hub_token="[EMO]" → pooled_outputs["[EMO]"]      │
│  capability="embedding" → hub_token="[MEM]" → pooled_outputs["[MEM]"]     │
│  capability="nli" → hub_token="[REL]" → PairEncoderV3                     │
│  capability="relation" → hub_token="[REL]" → PairEncoderV3                │
│  capability="intent" → hub_token="[TASK]" → pooled_outputs["[TASK]"]      │
│  capability="ingress" → hub_token="[TASK]" → pooled_outputs["[TASK]"]     │
│  capability="ner_*" → TOKEN_LEVEL → full last_hidden_state                │
│  capability="temporal" → TOKEN_LEVEL → full last_hidden_state             │
└───────────────────────────────────────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────────────────────────────────────┐
│  STEP 6: TASK HEADS                                                       │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐      │
│  │  HubAwareClassificationHead(hub="[EMO]", labels=44)             │      │
│  │  Input: pooled_outputs["[EMO]"] → dropout → Linear → logits     │      │
│  │  Output: emotions_logits [batch, 44]                             │      │
│  └─────────────────────────────────────────────────────────────────┘      │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐      │
│  │  HierarchicalSafetyHead(hub="[EMO]", hierarchy=4 levels)        │      │
│  │  Input: pooled_outputs["[EMO]"]                                  │      │
│  │  Output: safety_level ∈ {GREEN, AMBER, RED, CRISIS}             │      │
│  └─────────────────────────────────────────────────────────────────┘      │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐      │
│  │  PairEncoderV3(pooling="rel_hub")                               │      │
│  │  Input: last_hidden_state[:, 3, :] ([REL] position)             │      │
│  │  Output: nli_logits [batch, 3] or relation_logits [batch, 15]   │      │
│  └─────────────────────────────────────────────────────────────────┘      │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐      │
│  │  HubAwareTokenClassificationHead(labels=9)                      │      │
│  │  Input: full last_hidden_state [batch, seq, 768]                │      │
│  │  Output: ner_logits [batch, seq, 9]                              │      │
│  └─────────────────────────────────────────────────────────────────┘      │
└───────────────────────────────────────────────────────────────────────────┘
```

---

### 3. Module Import Wiring

```python
# src/modeling_studio/models/modernbert_v3.py — Main orchestrator imports

from .config_v3 import ModernBERTv3Config
from .embeddings_v3 import ModernBERTEmbeddingsV3
from .encoder_v3 import ModernBERTEncoderV3
from .layers_v3 import ModernBERTLayerV3, create_layer_stack
from .attention_v3 import MultiScaleAttentionWithGlobals, GLOBAL_TOKEN_POSITIONS
from .ffn_v3 import GELUFFN
from .lora_v3 import LoRALayer
from .poolers_v3 import HubTokenPooler, CombinedPooler
from .pair_encoder_v3 import PairEncoderV3
from .hub_tokens import (
    HUB_TOKEN_REGISTRY,
    get_hub_positions,
    get_hub_for_capability,
    TOKEN_LEVEL_CAPABILITIES,
)
from .heads_v3 import (
    HubAwareClassificationHead,
    HubAwareTokenClassificationHead,
    HierarchicalSafetyHead,
    EmbeddingHead,
)
from .initialization_v3 import initialize_from_v2
```

---

### 4. Constructor Wiring (ModernBERTv3Ultra.**init**)

```python
class ModernBERTv3Ultra(nn.Module):
    """
    ModernBERT v3.3 Ultra - Complete Multi-Task Encoder

    This is the main orchestrator that wires together all v3 components.
    """

    def __init__(self, config: ModernBERTv3Config):
        super().__init__()
        self.config = config

        # ===================================================================
        # COMPONENT 1: EMBEDDINGS
        # ===================================================================
        self.embeddings = ModernBERTEmbeddingsV3(
            vocab_size=config.vocab_size,  # 50264 (v2) + 4 hub tokens = 50268
            hidden_size=config.hidden_size,  # 768
            max_position_embeddings=config.max_position_embeddings,  # 8192
            hidden_dropout_prob=config.hidden_dropout_prob,
            pad_token_id=config.pad_token_id,
            use_rotary_embeddings=config.use_rotary_embeddings,
        )

        # ===================================================================
        # COMPONENT 2: ENCODER (28 Transformer Layers)
        # ===================================================================
        self.encoder = ModernBERTEncoderV3(
            num_layers=config.num_hidden_layers,  # 28
            hidden_size=config.hidden_size,  # 768
            num_attention_heads=config.num_attention_heads,  # 12
            intermediate_size=config.intermediate_size,  # 3072 (4x hidden)
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            use_flash_attention=config.use_flash_attention,
            gradient_checkpointing=config.gradient_checkpointing,
            lora_layers=config.lora_target_layers,  # [23, 24, 25, 26, 27, 28]
            lora_r=config.lora_r,  # 16
            lora_alpha=config.lora_alpha,  # 16
        )

        # ===================================================================
        # COMPONENT 3: POOLERS
        # ===================================================================
        # Hub token pooler extracts [CLS], [EMO], [MEM], [REL], [TASK]
        self.hub_pooler = HubTokenPooler(
            hidden_size=config.hidden_size,
            add_projection=False,  # No projection, use raw representations
        )

        # Combined pooler provides CLS, mean, and hub pooling
        self.combined_pooler = CombinedPooler(
            hidden_size=config.hidden_size,
        )

        # ===================================================================
        # COMPONENT 4: PAIR ENCODER (for NLI/Relation tasks)
        # ===================================================================
        self.pair_encoder = PairEncoderV3(
            hidden_size=config.hidden_size,
            num_labels=3,  # Default for NLI (entailment, neutral, contradiction)
            classifier_dropout=config.hidden_dropout_prob,
            pooling_strategy="rel_hub",  # Use [REL] hub token
        )

        # ===================================================================
        # COMPONENT 5: FINAL LAYER NORM (optional post-encoder normalization)
        # ===================================================================
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

        # ===================================================================
        # COMPONENT 6: TASK HEADS (12 capabilities)
        # NOTE: Label counts MUST match src/modeling_studio/data/labels.py
        # ===================================================================
        self.heads = nn.ModuleDict({
            # [EMO] Hub - Affective capabilities
            "emotions": HubAwareClassificationHead(
                hidden_size=config.hidden_size,
                num_labels=44,  # EMOTIONS_FAMILYOS_LABELS: 44 FamilyOS emotions
                hub_token="[EMO]",
            ),
            "sentiment": HubAwareClassificationHead(
                hidden_size=config.hidden_size,
                num_labels=5,  # SENTIMENT_LABELS: very_negative → very_positive
                hub_token="[EMO]",
            ),
            "safety_generic": HierarchicalSafetyHead(
                hidden_size=config.hidden_size,
                num_labels=8,  # SAFETY_GENERIC_LABELS: 8 toxicity types
                hub_token="[EMO]",
            ),
            "safety_familyos": HierarchicalSafetyHead(
                hidden_size=config.hidden_size,
                num_labels=4,  # SAFETY_FAMILYOS_LABELS: GREEN, AMBER, RED, CRISIS
                hub_token="[EMO]",
            ),

            # [MEM] Hub - Memory/Embedding capability
            "embedding": EmbeddingHead(
                hidden_size=config.hidden_size,
                hub_token="[MEM]",
            ),

            # [REL] Hub - Relationship capabilities
            "nli": HubAwareClassificationHead(
                hidden_size=config.hidden_size,
                num_labels=3,  # NLI_LABELS: Entailment, Neutral, Contradiction
                hub_token="[REL]",
            ),
            "relation": HubAwareClassificationHead(
                hidden_size=config.hidden_size,
                num_labels=15,  # RELATION_LABELS: 15 family relations
                hub_token="[REL]",
            ),

            # [TASK] Hub - Intent/Action capabilities
            "intent": HubAwareClassificationHead(
                hidden_size=config.hidden_size,
                num_labels=8,  # INTENT_LABELS: 8 FamilyOS intents
                hub_token="[TASK]",
            ),
            "ingress": HubAwareClassificationHead(
                hidden_size=config.hidden_size,
                num_labels=12,  # INGRESS_LABELS: 12 domains
                hub_token="[TASK]",
            ),

            # Token-level capabilities (no hub routing)
            "ner_general": HubAwareTokenClassificationHead(
                hidden_size=config.hidden_size,
                num_labels=17,  # NER_GENERAL_LABELS: 17 BIO tags
            ),
            "ner_family": HubAwareTokenClassificationHead(
                hidden_size=config.hidden_size,
                num_labels=21,  # NER_FAMILY_LABELS: 21 BIO tags
            ),
            "temporal": HubAwareTokenClassificationHead(
                hidden_size=config.hidden_size,
                num_labels=13,  # TEMPORAL_LABELS: 13 BIO tags
            ),
        })

        # ===================================================================
        # METADATA
        # ===================================================================
        self.hub_positions = get_hub_positions()
        self.num_hub_tokens = len(HUB_TOKEN_REGISTRY)

        # Initialize weights
        self.apply(self._init_weights)

        print(f"\n✓ ModernBERTv3Ultra initialized:")
        print(f"  - Layers: {config.num_hidden_layers}")
        print(f"  - Hidden: {config.hidden_size}")
        print(f"  - Heads: {config.num_attention_heads}")
        print(f"  - Hub tokens: {list(HUB_TOKEN_REGISTRY.keys())}")
        print(f"  - LoRA layers: {config.lora_target_layers}")
        print(f"  - Total parameters: {self.num_parameters:,}")
```

---

### 5. Forward Pass Wiring

```python
def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    token_type_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    task: Optional[str] = None,
    output_hidden_states: bool = False,
    output_attentions: bool = False,
    return_dict: bool = True,
) -> Union[ModernBERTv3Output, Dict[str, torch.Tensor]]:
    """
    Complete forward pass through v3 architecture.

    Flow:
        input_ids → embeddings → encoder (28 layers) → pooling → task heads

    Args:
        input_ids: [batch, seq_len] token IDs
        attention_mask: [batch, seq_len] padding mask (1=valid, 0=pad)
        token_type_ids: [batch, seq_len] segment IDs (unused in ModernBERT)
        position_ids: [batch, seq_len] position IDs (optional)
        task: Single task name or None for all tasks
        output_hidden_states: Return all layer hidden states
        output_attentions: Return all attention weights
        return_dict: Return structured output or tuple

    Returns:
        ModernBERTv3Output with:
            - last_hidden_state: [batch, seq_len, 768]
            - pooled_outputs: Dict of hub representations
            - task_outputs: Dict of task-specific logits (if task specified)
    """

    # ===================================================================
    # STEP 1: EMBEDDINGS
    # ===================================================================
    # Input: [batch, seq_len] token IDs
    # Output: [batch, seq_len, 768] embeddings
    hidden_states = self.embeddings(
        input_ids=input_ids,
        position_ids=position_ids,
        token_type_ids=token_type_ids,
    )

    # ===================================================================
    # STEP 2: ENCODER (28 Transformer Layers)
    # ===================================================================
    # Pass through all 28 layers with sliding windows + global hub attention
    encoder_output, all_hidden_states, all_attentions = self.encoder(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        output_hidden_states=output_hidden_states,
        output_attentions=output_attentions,
    )

    # ===================================================================
    # STEP 3: FINAL LAYER NORM
    # ===================================================================
    last_hidden_state = self.final_layer_norm(encoder_output)

    # ===================================================================
    # STEP 4: POOLING
    # ===================================================================
    # Extract hub token representations: [EMO], [MEM], [REL], [TASK]
    pooled_outputs = self.hub_pooler(last_hidden_state, attention_mask)
    # Result: {"[CLS]": ..., "[EMO]": ..., "[MEM]": ..., "[REL]": ..., "[TASK]": ...}

    # ===================================================================
    # STEP 5: TASK HEAD ROUTING (if task specified)
    # ===================================================================
    task_outputs = {}
    if task is not None:
        task_outputs = self._route_to_head(
            task=task,
            last_hidden_state=last_hidden_state,
            pooled_outputs=pooled_outputs,
            attention_mask=attention_mask,
        )

    # ===================================================================
    # STEP 6: RETURN
    # ===================================================================
    if return_dict:
        return ModernBERTv3Output(
            last_hidden_state=last_hidden_state,
            pooled_outputs=pooled_outputs,
            task_outputs=task_outputs,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
    else:
        return (last_hidden_state, pooled_outputs, task_outputs,
                all_hidden_states, all_attentions)


def _route_to_head(
    self,
    task: str,
    last_hidden_state: torch.Tensor,
    pooled_outputs: Dict[str, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Route to appropriate task head based on capability.

    Routing logic:
        - Token-level tasks (NER, temporal) → full sequence
        - Pair tasks (NLI, relation) → PairEncoderV3
        - Hub-routed tasks → specific hub token representation
    """
    outputs = {}

    # Token-level capabilities
    if task in TOKEN_LEVEL_CAPABILITIES:
        # NER, temporal need full sequence
        outputs[task] = self.heads[task](
            hidden_states=last_hidden_state,
            attention_mask=attention_mask,
        )

    # Pair encoder tasks
    elif task in ["nli", "relation"]:
        # Use PairEncoderV3 with [REL] hub token
        outputs[task] = self.pair_encoder(
            encoder_output=last_hidden_state,
            attention_mask=attention_mask,
        )

    # Hub-routed tasks
    else:
        # Get appropriate hub token representation
        hub_token = get_hub_for_capability(task)
        hub_repr = pooled_outputs[hub_token]

        outputs[task] = self.heads[task](
            hidden_states=last_hidden_state,
            pooled_outputs=pooled_outputs,
        )

    return outputs
```

---

### 6. Layer Stack Wiring (create_layer_stack)

```python
def create_layer_stack(
    num_layers: int = 28,
    hidden_size: int = 768,
    num_attention_heads: int = 12,
    intermediate_size: int = 3072,
    hidden_dropout_prob: float = 0.1,
    attention_probs_dropout_prob: float = 0.1,
    use_flash_attention: bool = False,
    lora_layers: Optional[List[int]] = None,
    lora_r: int = 16,
    lora_alpha: int = 16,
) -> nn.ModuleList:
    """
    Create the complete 28-layer transformer stack with proper configuration.

    Layer bands:
        - L1-6 (Foundation): Window=64, Frozen
        - L7-18 (Context): Window=128, Frozen
        - L19-22 (Semantic): Window=256, Trainable
        - L23-28 (Family): Window=512, Trainable + LoRA
    """
    if lora_layers is None:
        lora_layers = [23, 24, 25, 26, 27, 28]  # Family Band only

    layers = nn.ModuleList()

    for layer_idx in range(1, num_layers + 1):
        # Determine window size based on layer band
        window_size = LAYER_WINDOW_CONFIG.get(layer_idx, 512)

        # Enable LoRA for Family Band (L23-28)
        enable_lora = layer_idx in lora_layers

        layer = ModernBERTLayerV3(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_flash_attention=use_flash_attention,
            window_size=window_size,
            layer_idx=layer_idx,
            enable_lora=enable_lora,
            lora_r=lora_r if enable_lora else 0,
            lora_alpha=lora_alpha if enable_lora else 0,
        )

        layers.append(layer)

    return layers


# Layer window configuration by band
LAYER_WINDOW_CONFIG = {
    # Foundation Band (L1-6): Local patterns
    1: 64, 2: 64, 3: 64, 4: 64, 5: 64, 6: 64,

    # Context Band (L7-18): Phrase patterns
    7: 128, 8: 128, 9: 128, 10: 128, 11: 128, 12: 128,
    13: 128, 14: 128, 15: 128, 16: 128, 17: 128, 18: 128,

    # Semantic Band (L19-22): Sentence patterns
    19: 256, 20: 256, 21: 256, 22: 256,

    # Family Band (L23-28): Full context + LoRA
    23: 512, 24: 512, 25: 512, 26: 512, 27: 512, 28: 512,
}
```

---

### 7. Attention Wiring (Global + Sliding Window)

```python
class MultiScaleAttentionWithGlobals(nn.Module):
    """
    Multi-Head Attention with:
        1. Sliding windows for text tokens (efficiency)
        2. Global bidirectional attention for hub tokens (capability)

    Key innovation: Hub tokens (pos 0-4) can see ENTIRE sequence,
                    and entire sequence can see hub tokens.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, hidden_size = hidden_states.shape

        # ===============================================================
        # STEP 1: Q, K, V PROJECTIONS (with optional LoRA)
        # ===============================================================
        # Q = X @ W_q + (X @ A @ B) * (alpha / r)  [if LoRA enabled]
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        queries = self._split_heads(queries)  # [batch, heads, seq, head_dim]
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        # ===============================================================
        # STEP 2: CREATE HYBRID ATTENTION MASK
        # ===============================================================
        # Combine sliding window + global hub attention
        attn_mask = create_global_local_attention_mask(
            seq_len=seq_len,
            window_size=self.window_size,
            global_token_positions=GLOBAL_TOKEN_POSITIONS,  # [0, 1, 2, 3, 4]
            device=hidden_states.device,
        )

        # Mask format: [seq, seq] where 1 = can attend, 0 = masked
        #
        # Visual example (seq_len=10, window=4):
        #           0  1  2  3  4  5  6  7  8  9   (keys)
        #        +--------------------------------
        #    0   |  1  1  1  1  1  1  1  1  1  1   <- [CLS] global
        #    1   |  1  1  1  1  1  1  1  1  1  1   <- [EMO] global
        #    2   |  1  1  1  1  1  1  1  1  1  1   <- [MEM] global
        #    3   |  1  1  1  1  1  1  1  1  1  1   <- [REL] global
        #    4   |  1  1  1  1  1  1  1  1  1  1   <- [TASK] global
        #    5   |  1  1  1  1  1  1  1  1  0  0   <- text: globals + window
        #    6   |  1  1  1  1  1  0  1  1  1  0   <- text: globals + window
        #    7   |  1  1  1  1  1  0  0  1  1  1   <- text: globals + window
        #    8   |  1  1  1  1  1  0  0  0  1  1   <- text: globals + window
        #    9   |  1  1  1  1  1  0  0  0  0  1   <- text: globals + window

        # ===============================================================
        # STEP 3: SCALED DOT-PRODUCT ATTENTION
        # ===============================================================
        if self.use_flash_attention:
            # Flash Attention 2 with mask support
            attn_output = flash_attn_func(
                queries, keys, values,
                causal=False,
                window_size=(self.window_size, self.window_size),
            )
        else:
            # Standard PyTorch SDPA
            attn_output = F.scaled_dot_product_attention(
                queries, keys, values,
                attn_mask=attn_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
            )

        # ===============================================================
        # STEP 4: OUTPUT PROJECTION
        # ===============================================================
        attn_output = self._merge_heads(attn_output)  # [batch, seq, hidden]
        output = self.o_proj(attn_output)  # + LoRA if enabled

        return output, None  # (output, attention_weights)
```

---

### 8. Weight Initialization Wiring (Function Preserving Growth)

```python
def initialize_from_v2(
    v3_model: ModernBERTv3Ultra,
    v2_checkpoint_path: str,
) -> None:
    """
    Initialize v3 model from v2 checkpoint via Function Preserving Growth.

    Transfer strategy:
        1. Word embeddings: Direct copy (768-dim match)
        2. L1-22: Direct copy from v2 L1-22
        3. L23-28: Clone from v2 L15-20 (mature semantic processors)
        4. Hub tokens: Semantic centroid initialization
    """
    print(f"\n🔄 Initializing v3 from v2 checkpoint: {v2_checkpoint_path}")

    # Load v2 model
    v2_model = load_v2_checkpoint(v2_checkpoint_path)
    v2_embeddings = v2_model.embeddings.word_embeddings.weight

    with torch.no_grad():
        # ===============================================================
        # COMPONENT 1: WORD EMBEDDINGS
        # ===============================================================
        vocab_size_v2 = v2_embeddings.size(0)  # 50264

        # Copy v2 embeddings
        v3_model.embeddings.word_embeddings.weight[:vocab_size_v2].copy_(
            v2_embeddings
        )

        print(f"  ✓ Copied {vocab_size_v2} word embeddings from v2")

        # ===============================================================
        # COMPONENT 2: HUB TOKEN EMBEDDINGS (Semantic Centroid Init)
        # ===============================================================
        from .hub_initialization_v3 import initialize_hub_tokens_semantic

        initialize_hub_tokens_semantic(
            v3_model=v3_model,
            v2_tokenizer=v2_model.tokenizer,
            v2_embeddings=v2_embeddings,
        )
        # Hub tokens now initialized as centroids of related words

        # ===============================================================
        # COMPONENT 3: TRANSFORMER LAYERS 1-22 (Direct Copy)
        # ===============================================================
        for i in range(22):
            v3_model.encoder.layers[i].load_state_dict(
                v2_model.encoder.layers[i].state_dict(),
                strict=True,
            )

        print(f"  ✓ Copied layers 1-22 from v2")

        # ===============================================================
        # COMPONENT 4: TRANSFORMER LAYERS 23-28 (Clone from v2 L15-20)
        # ===============================================================
        source_layers = [14, 15, 16, 17, 18, 19]  # 0-indexed
        target_layers = [22, 23, 24, 25, 26, 27]  # 0-indexed

        for src_idx, tgt_idx in zip(source_layers, target_layers):
            # Clone weights (not LoRA, just base transformer)
            v3_model.encoder.layers[tgt_idx].attention.load_state_dict(
                v2_model.encoder.layers[src_idx].attention.state_dict(),
                strict=False,  # Skip LoRA weights
            )
            v3_model.encoder.layers[tgt_idx].ffn.load_state_dict(
                v2_model.encoder.layers[src_idx].ffn.state_dict()
            )

        print(f"  ✓ Cloned layers 23-28 from v2 layers 15-20")

        # LoRA weights remain randomly initialized
        print(f"  ⚠️  LoRA adapters (L23-28) remain random (will train)")

    print(f"\n✅ v3 initialization complete!")
    print(f"  - Layers 1-22: Exact copy from v2")
    print(f"  - Layers 23-28: Cloned from v2 L15-20")
    print(f"  - Hub tokens: Semantic centroid initialized")
    print(f"  - LoRA: Random (r=16, alpha=16, trainable)")
```

---

### 9. Training Phase Wiring

```python
def configure_training_phase(
    model: ModernBERTv3Ultra,
    phase: str,
) -> None:
    """
    Configure model freezing/unfreezing for different training phases.

    Phases:
        - "phase0.5": Healing warmup (L19-28 trainable, L1-18 frozen)
        - "phase1": Full training (L19-28 trainable, L1-18 frozen)
    """

    if phase == "phase0.5":
        # ===============================================================
        # PHASE 0.5: HEALING WARMUP
        # ===============================================================
        print("\n🏥 Configuring Phase 0.5 (Healing Warmup)")

        # Freeze embeddings (except hub tokens)
        for name, param in model.embeddings.named_parameters():
            if "word_embeddings" in name:
                param.requires_grad_(False)

        # Freeze Foundation + Context bands (L1-18)
        for i in range(18):
            for param in model.encoder.layers[i].parameters():
                param.requires_grad_(False)

        # Unfreeze Semantic + Family bands (L19-28)
        for i in range(18, 28):
            for param in model.encoder.layers[i].parameters():
                param.requires_grad_(True)

        # Freeze hub pooler and heads
        for param in model.hub_pooler.parameters():
            param.requires_grad_(False)
        for head in model.heads.values():
            for param in head.parameters():
                param.requires_grad_(False)

        print("  ❄️  Frozen: Embeddings, L1-18, Poolers, Heads")
        print("  🔥 Trainable: L19-28 (Semantic + Family bands)")

    elif phase == "phase1":
        # ===============================================================
        # PHASE 1: FULL TRAINING
        # ===============================================================
        print("\n🚀 Configuring Phase 1 (Full Training)")

        # Same freezing as Phase 0.5 for encoder
        for i in range(18):
            for param in model.encoder.layers[i].parameters():
                param.requires_grad_(False)

        for i in range(18, 28):
            for param in model.encoder.layers[i].parameters():
                param.requires_grad_(True)

        # Unfreeze poolers and heads
        for param in model.hub_pooler.parameters():
            param.requires_grad_(True)
        for head in model.heads.values():
            for param in head.parameters():
                param.requires_grad_(True)

        print("  ❄️  Frozen: L1-18")
        print("  🔥 Trainable: L19-28, Hub Poolers, All Heads, LoRA")
```

---

### 10. Capability-to-Hub Routing Table

```python
# Complete routing table for all 12 capabilities
CAPABILITY_HUB_ROUTING = {
    # [EMO] Hub - Affective understanding
    "emotions": "[EMO]",
    "sentiment": "[EMO]",
    "safety_generic": "[EMO]",
    "safety_familyos": "[EMO]",

    # [MEM] Hub - Memory and retrieval
    "embedding": "[MEM]",
    "memory_logging": "[MEM]",

    # [REL] Hub - Relationships and logic
    "nli": "[REL]",
    "relation": "[REL]",

    # [TASK] Hub - Actions and intents
    "intent": "[TASK]",
    "ingress": "[TASK]",

    # Token-level (no hub routing)
    "ner_general": None,  # Uses full sequence
    "ner_family": None,
    "temporal": None,
}


def get_hub_for_capability(capability: str) -> str:
    """
    Get the hub token that routes to a given capability.

    Args:
        capability: Capability name (e.g., "emotions", "nli")

    Returns:
        Hub token name (e.g., "[EMO]", "[REL]") or None for token-level
    """
    return CAPABILITY_HUB_ROUTING.get(capability, "[CLS]")  # Fallback to [CLS]
```

---

### 11. Summary: Complete Wiring Chain

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  COMPLETE DATA FLOW: INPUT → OUTPUT                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. Raw Text                                                                 │
│     "Mom is feeling sad today"                                               │
│                  ↓                                                           │
│  2. HubTokenizer (tokenization_v3.py)                                        │
│     [CLS][EMO][MEM][REL][TASK] Mom is feeling sad today [SEP]               │
│                  ↓                                                           │
│  3. ModernBERTEmbeddingsV3 (embeddings_v3.py)                                │
│     [batch, seq_len, 768] embeddings                                         │
│                  ↓                                                           │
│  4. ModernBERTEncoderV3 (encoder_v3.py)                                      │
│     28 layers: L1-6 (W=64), L7-18 (W=128), L19-22 (W=256), L23-28 (W=512)   │
│     Each layer: MultiScaleAttentionWithGlobals → GELUFFN → LayerNorm        │
│                  ↓                                                           │
│  5. HubTokenPooler (poolers_v3.py)                                           │
│     Extract {"[CLS]": ..., "[EMO]": ..., "[MEM]": ..., "[REL]": ..., ...}   │
│                  ↓                                                           │
│  6. Hub Routing (get_representation_for_capability)                          │
│     capability="emotions" → hub="[EMO]" → pooled_outputs["[EMO]"]           │
│                  ↓                                                           │
│  7. HubAwareClassificationHead (heads_v3.py)                                 │
│     pooled_repr → dropout → Linear(768, 44) → emotions_logits               │
│                  ↓                                                           │
│  8. Loss & Backprop                                                          │
│     CrossEntropyLoss(logits, labels) → gradients → LoRA + L19-28            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

**This wiring documentation provides the complete blueprint for assembling ModernBERT v3.3 Ultra. All components, data flows, and routing logic are now explicitly documented.**

---

## 🏁 Milestone 1: v3 Configuration & Hub Token Foundation

**Goal:** Define v3 architecture configuration and implement hub token system
**Estimated Effort:** 5 days
**Dependencies:** v2 checkpoint available, tokenizer from v2

### Epic 1.1: v3 Configuration

#### Issue 1.1.1: Implement v3 Configuration Dataclass

**File:** `src/modeling_studio/models/config_v3.py`
**Effort:** 4 hours
**Dependencies:** None

**Description:**
Create the configuration dataclass for ModernBERT v3.3 Ultra with all architecture parameters.

**Implementation:**

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class ModernBERTv3Config:
    """Configuration for ModernBERT v3.3 Ultra."""

    # Architecture
    hidden_size: int = 768                    # Same as v2 (enables weight transfer)
    num_layers: int = 28                      # 22 from v2 + 6 cloned
    num_attention_heads: int = 12             # MHA (no GQA)
    intermediate_size: int = 3072             # 4x hidden (GELU FFN)
    max_position_embeddings: int = 8192
    vocab_size: int = 50368                   # v2 vocab + 4 hub tokens

    # Hub Tokens
    hub_tokens: List[str] = field(default_factory=lambda: ["[EMO]", "[MEM]", "[REL]", "[TASK]"])
    hub_token_positions: Dict[str, int] = field(default_factory=lambda: {
        "[CLS]": 0, "[EMO]": 1, "[MEM]": 2, "[REL]": 3, "[TASK]": 4
    })
    global_attention_positions: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])

    # Sliding Window by Layer Band
    window_sizes: Dict[str, int] = field(default_factory=lambda: {
        "foundation": 64,    # Layers 1-6
        "context": 128,      # Layers 7-18
        "semantic": 256,     # Layers 19-22
        "family": 512,       # Layers 23-28
    })

    # Layer Bands
    layer_bands: Dict[str, List[int]] = field(default_factory=lambda: {
        "foundation": list(range(1, 7)),      # 1-6
        "context": list(range(7, 19)),        # 7-18
        "semantic": list(range(19, 23)),      # 19-22
        "family": list(range(23, 29)),        # 23-28
    })

    # LoRA Configuration
    lora_enabled: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_layers: List[int] = field(default_factory=lambda: [23, 24, 25, 26, 27, 28])

    # Pair Encoder
    pair_encoder_enabled: bool = True
    pair_encoder_heads: int = 8
    pair_encoder_dropout: float = 0.1

    # Training
    frozen_layers_phase1: List[int] = field(default_factory=lambda: list(range(1, 19)))

    # FFN
    ffn_activation: str = "gelu"              # No SwiGLU (removed from roadmap)
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
```

**Acceptance Criteria:**

- [ ] Dataclass validates all required fields
- [ ] Default values match enhanced_design_v3.md specifications
- [ ] Layer bands correctly map to layer indices
- [ ] Hub token positions are 0-indexed correctly

**Tests:** `tests/v3/test_config_v3.py::test_config_defaults`

---

#### Issue 1.1.2: Create v3 Model YAML Configuration

**File:** `configs/model/encoder/modernbert_v3_ultra.yaml`
**Effort:** 2 hours
**Dependencies:** Issue 1.1.1

**Description:**
Create YAML configuration file that can be loaded by Hydra/OmegaConf for v3 model instantiation.

**Implementation:**

```yaml
# configs/model/encoder/modernbert_v3_ultra.yaml

_target_: modeling_studio.models.modernbert_v3.ModernBERTv3Ultra

name: "ModernBERTv3Ultra"
version: "3.3"
codename: "Ultra"

architecture:
  hidden_size: 768
  num_layers: 28
  num_attention_heads: 12
  intermediate_size: 3072
  max_position_embeddings: 8192
  vocab_size: 50368
  ffn_activation: "gelu"
  hidden_dropout_prob: 0.1
  attention_probs_dropout_prob: 0.1

hub_tokens:
  enabled: true
  tokens: ["[EMO]", "[MEM]", "[REL]", "[TASK]"]
  positions:
    "[CLS]": 0
    "[EMO]": 1
    "[MEM]": 2
    "[REL]": 3
    "[TASK]": 4
  global_attention: [0, 1, 2, 3, 4]
  initialization: "semantic_centroid"

attention:
  type: "multi_scale_with_globals"
  window_sizes:
    foundation: 64    # Layers 1-6
    context: 128      # Layers 7-18
    semantic: 256     # Layers 19-22
    family: 512       # Layers 23-28

layer_bands:
  foundation: [1, 2, 3, 4, 5, 6]
  context: [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
  semantic: [19, 20, 21, 22]
  family: [23, 24, 25, 26, 27, 28]

lora:
  enabled: true
  r: 16
  alpha: 16
  dropout: 0.05
  target_layers: [23, 24, 25, 26, 27, 28]

pair_encoder:
  enabled: true
  num_heads: 8
  dropout: 0.1

initialization:
  method: "function_preserving_growth"
  source_checkpoint: "checkpoints/modernbert-unified-v2"
  layers_1_22: "copy"
  layers_23_28: "clone_from_15_20"
  hub_tokens: "semantic_centroid"
  verify_function_preserving: true
```

**Acceptance Criteria:**

- [ ] YAML loads without errors via OmegaConf
- [ ] All values match config_v3.py defaults
- [ ] Hub token positions are correct
- [ ] Layer bands sum to 28 total layers

**Tests:** `tests/v3/test_config_v3.py::test_yaml_loading`

---

#### Issue 1.1.3: Implement Layer Source Mapping

**File:** `src/modeling_studio/models/config_v3.py` (add to existing)
**Effort:** 2 hours
**Dependencies:** Issue 1.1.1

**Description:**
Create mapping that defines where each v3 layer's weights come from during initialization.

**Implementation:**

```python
# Add to config_v3.py

from enum import Enum
from typing import NamedTuple

class LayerSource(Enum):
    """Source of layer weights during v3 initialization."""
    COPY = "copy"           # Direct copy from v2 same layer
    CLONE = "clone"         # Clone from different v2 layer
    RANDOM = "random"       # Random initialization

class LayerMapping(NamedTuple):
    """Mapping of v3 layer to its weight source."""
    v3_layer: int
    source: LayerSource
    v2_layer: int           # Source layer in v2 (ignored if RANDOM)

def get_layer_source_mapping() -> Dict[int, LayerMapping]:
    """
    Get the complete layer source mapping for v3 initialization.

    Returns:
        Dict mapping v3 layer index to LayerMapping

    Strategy:
        - Layers 1-22: Copy directly from v2 layers 1-22
        - Layer 23: Clone from v2 layer 15
        - Layer 24: Clone from v2 layer 16
        - Layer 25: Clone from v2 layer 17
        - Layer 26: Clone from v2 layer 18
        - Layer 27: Clone from v2 layer 19
        - Layer 28: Clone from v2 layer 20
    """
    mapping = {}

    # Layers 1-22: Direct copy
    for i in range(1, 23):
        mapping[i] = LayerMapping(v3_layer=i, source=LayerSource.COPY, v2_layer=i)

    # Layers 23-28: Clone from v2 layers 15-20
    for i, v2_layer in enumerate([15, 16, 17, 18, 19, 20], start=23):
        mapping[i] = LayerMapping(v3_layer=i, source=LayerSource.CLONE, v2_layer=v2_layer)

    return mapping

def get_layer_band(layer_idx: int) -> str:
    """Get the band name for a given layer index."""
    if 1 <= layer_idx <= 6:
        return "foundation"
    elif 7 <= layer_idx <= 18:
        return "context"
    elif 19 <= layer_idx <= 22:
        return "semantic"
    elif 23 <= layer_idx <= 28:
        return "family"
    else:
        raise ValueError(f"Invalid layer index: {layer_idx}")

def get_window_size(layer_idx: int, config: ModernBERTv3Config) -> int:
    """Get sliding window size for a given layer."""
    band = get_layer_band(layer_idx)
    return config.window_sizes[band]
```

**Acceptance Criteria:**

- [ ] Layers 1-22 map to COPY from same v2 layer
- [ ] Layers 23-28 map to CLONE from v2 layers 15-20
- [ ] `get_layer_band()` returns correct band for all 28 layers
- [ ] `get_window_size()` returns correct window for each band

**Tests:** `tests/v3/test_config_v3.py::test_layer_source_mapping`

---

### Epic 1.2: Hub Token System

#### Issue 1.2.1: Define Hub Token Registry

**File:** `src/modeling_studio/models/hub_tokens.py`
**Effort:** 3 hours
**Dependencies:** None

**Description:**
Create the hub token registry with capability mappings and semantic seed words for initialization.

**Implementation:**

```python
# src/modeling_studio/models/hub_tokens.py

from dataclasses import dataclass
from typing import Dict, List, Set
from enum import Enum

class HubToken(Enum):
    """Hub token identifiers."""
    CLS = "[CLS]"
    EMO = "[EMO]"
    MEM = "[MEM]"
    REL = "[REL]"
    TASK = "[TASK]"

@dataclass
class HubTokenSpec:
    """Specification for a hub token."""
    token: str
    position: int
    capabilities: List[str]
    semantic_seeds: List[str]
    description: str

# Hub Token Registry
HUB_TOKEN_REGISTRY: Dict[str, HubTokenSpec] = {
    "[EMO]": HubTokenSpec(
        token="[EMO]",
        position=1,
        capabilities=["emotions", "sentiment", "safety_generic", "safety_familyos"],
        semantic_seeds=["happy", "sad", "angry", "fear", "joy", "anxious", "love", "feeling"],
        description="Affective understanding - routes to emotion/sentiment/safety heads"
    ),
    "[MEM]": HubTokenSpec(
        token="[MEM]",
        position=2,
        capabilities=["embedding"],
        semantic_seeds=["remember", "memory", "past", "history", "recall", "yesterday"],
        description="Memory retrieval & storage - routes to embedding head"
    ),
    "[REL]": HubTokenSpec(
        token="[REL]",
        position=3,
        capabilities=["nli", "relation"],
        semantic_seeds=["family", "mother", "father", "sister", "brother", "parent", "child"],
        description="Entity & logical relationships - routes to NLI/relation heads"
    ),
    "[TASK]": HubTokenSpec(
        token="[TASK]",
        position=4,
        capabilities=["intent", "ingress"],
        semantic_seeds=["action", "do", "want", "need", "help", "schedule", "plan"],
        description="User action classification - routes to intent/ingress heads"
    ),
}

# Capabilities that use token-level classification (not hub routing)
TOKEN_LEVEL_CAPABILITIES: Set[str] = {"ner_general", "ner_family", "temporal"}

# Hub token IDs (to be set after tokenizer initialization)
HUB_TOKEN_IDS: Dict[str, int] = {}

def get_hub_for_capability(capability: str) -> str:
    """
    Get the hub token that routes to a given capability.

    Args:
        capability: Name of the capability (e.g., "emotions", "intent")

    Returns:
        Hub token string (e.g., "[EMO]") or "[CLS]" for token-level tasks
    """
    if capability in TOKEN_LEVEL_CAPABILITIES:
        return "[CLS]"  # Token-level tasks use CLS or per-token representations

    for hub_token, spec in HUB_TOKEN_REGISTRY.items():
        if capability in spec.capabilities:
            return hub_token

    return "[CLS]"  # Fallback to CLS

def get_capabilities_for_hub(hub_token: str) -> List[str]:
    """Get all capabilities routed through a hub token."""
    if hub_token not in HUB_TOKEN_REGISTRY:
        return []
    return HUB_TOKEN_REGISTRY[hub_token].capabilities

def get_hub_positions() -> Dict[str, int]:
    """Get position indices for all hub tokens (including CLS)."""
    positions = {"[CLS]": 0}
    for token, spec in HUB_TOKEN_REGISTRY.items():
        positions[token] = spec.position
    return positions

def get_global_attention_positions() -> List[int]:
    """Get positions that should have global attention (CLS + all hubs)."""
    return [0, 1, 2, 3, 4]  # [CLS], [EMO], [MEM], [REL], [TASK]

def get_semantic_seeds(hub_token: str) -> List[str]:
    """Get semantic seed words for hub token initialization."""
    if hub_token not in HUB_TOKEN_REGISTRY:
        return []
    return HUB_TOKEN_REGISTRY[hub_token].semantic_seeds
```

**Acceptance Criteria:**

- [ ] All 4 hub tokens defined with correct positions (1-4)
- [ ] Each hub maps to correct capabilities per enhanced_design_v3.md
- [ ] Semantic seeds match the design document
- [ ] `get_hub_for_capability()` returns correct hub for all 12 capabilities
- [ ] Token-level capabilities (NER, temporal) correctly excluded from hub routing

**Tests:** `tests/v3/test_hub_tokens.py::test_hub_token_registry`

---

#### Issue 1.2.2: Implement Hub Token Injection Tokenizer

**File:** `src/modeling_studio/models/tokenization_v3.py`
**Effort:** 4 hours
**Dependencies:** Issue 1.2.1

**Description:**
Create a tokenizer wrapper that automatically injects hub tokens after [CLS] for all inputs.

**Implementation:**

```python
# src/modeling_studio/models/tokenization_v3.py

from typing import Dict, List, Optional, Union
from transformers import AutoTokenizer, BatchEncoding
import torch

from .hub_tokens import HUB_TOKEN_REGISTRY, get_hub_positions

class HubTokenizer:
    """
    Wrapper tokenizer that injects hub tokens after [CLS].

    Input:  "Mom is happy today"
    Output: "[CLS] [EMO] [MEM] [REL] [TASK] Mom is happy today [SEP]"
    """

    def __init__(self, base_tokenizer_name: str = "answerdotai/ModernBERT-base"):
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name)

        # Add hub tokens to vocabulary
        hub_tokens = list(HUB_TOKEN_REGISTRY.keys())
        num_added = self.base_tokenizer.add_special_tokens({
            "additional_special_tokens": hub_tokens
        })
        print(f"Added {num_added} hub tokens to vocabulary")

        # Cache hub token IDs
        self.hub_token_ids = {
            token: self.base_tokenizer.convert_tokens_to_ids(token)
            for token in hub_tokens
        }
        self.cls_token_id = self.base_tokenizer.cls_token_id
        self.sep_token_id = self.base_tokenizer.sep_token_id
        self.pad_token_id = self.base_tokenizer.pad_token_id

        # Hub token sequence: [EMO], [MEM], [REL], [TASK]
        self.hub_sequence = [
            self.hub_token_ids["[EMO]"],
            self.hub_token_ids["[MEM]"],
            self.hub_token_ids["[REL]"],
            self.hub_token_ids["[TASK]"],
        ]
        self.num_hub_tokens = len(self.hub_sequence)

    @property
    def vocab_size(self) -> int:
        return len(self.base_tokenizer)

    def __call__(
        self,
        text: Union[str, List[str]],
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: str = "pt",
        **kwargs
    ) -> BatchEncoding:
        """
        Tokenize text with hub token injection.

        Args:
            text: Input text or list of texts
            max_length: Maximum sequence length (including hub tokens)

        Returns:
            BatchEncoding with input_ids, attention_mask, and hub_token_mask
        """
        # Adjust max_length to account for hub tokens
        # Original: [CLS] text [SEP] -> New: [CLS] [EMO] [MEM] [REL] [TASK] text [SEP]
        adjusted_max_length = max_length - self.num_hub_tokens

        # Tokenize without special tokens (we'll add them manually)
        if isinstance(text, str):
            text = [text]

        batch_input_ids = []
        batch_attention_mask = []
        batch_hub_token_mask = []

        for t in text:
            # Tokenize text only (no CLS/SEP)
            encoded = self.base_tokenizer.encode(
                t,
                add_special_tokens=False,
                max_length=adjusted_max_length - 2,  # Reserve for CLS and SEP
                truncation=truncation,
            )

            # Build sequence: [CLS] [EMO] [MEM] [REL] [TASK] <text> [SEP]
            input_ids = (
                [self.cls_token_id] +
                self.hub_sequence +
                encoded +
                [self.sep_token_id]
            )

            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = [1] * len(input_ids)

            # Create hub token mask (1 for hub positions, 0 otherwise)
            hub_token_mask = [0] + [1] * self.num_hub_tokens + [0] * (len(encoded) + 1)

            # Pad to max_length
            padding_length = max_length - len(input_ids)
            if padding_length > 0:
                input_ids += [self.pad_token_id] * padding_length
                attention_mask += [0] * padding_length
                hub_token_mask += [0] * padding_length

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_hub_token_mask.append(hub_token_mask)

        # Convert to tensors
        result = BatchEncoding({
            "input_ids": torch.tensor(batch_input_ids),
            "attention_mask": torch.tensor(batch_attention_mask),
            "hub_token_mask": torch.tensor(batch_hub_token_mask),
        })

        return result

    def get_hub_token_positions(self) -> Dict[str, int]:
        """Get the positions of hub tokens in the sequence."""
        return {
            "[CLS]": 0,
            "[EMO]": 1,
            "[MEM]": 2,
            "[REL]": 3,
            "[TASK]": 4,
        }

    def get_text_start_position(self) -> int:
        """Get the position where actual text tokens start."""
        return 5  # After [CLS] + 4 hub tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        return self.base_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def save_pretrained(self, path: str):
        """Save tokenizer to disk."""
        self.base_tokenizer.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path: str) -> "HubTokenizer":
        """Load tokenizer from disk."""
        instance = cls.__new__(cls)
        instance.base_tokenizer = AutoTokenizer.from_pretrained(path)
        # Re-cache hub token IDs
        hub_tokens = list(HUB_TOKEN_REGISTRY.keys())
        instance.hub_token_ids = {
            token: instance.base_tokenizer.convert_tokens_to_ids(token)
            for token in hub_tokens
        }
        instance.cls_token_id = instance.base_tokenizer.cls_token_id
        instance.sep_token_id = instance.base_tokenizer.sep_token_id
        instance.pad_token_id = instance.base_tokenizer.pad_token_id
        instance.hub_sequence = [
            instance.hub_token_ids["[EMO]"],
            instance.hub_token_ids["[MEM]"],
            instance.hub_token_ids["[REL]"],
            instance.hub_token_ids["[TASK]"],
        ]
        instance.num_hub_tokens = len(instance.hub_sequence)
        return instance
```

**Acceptance Criteria:**

- [ ] Hub tokens added to vocabulary correctly
- [ ] Tokenization produces `[CLS] [EMO] [MEM] [REL] [TASK] <text> [SEP]` format
- [ ] `hub_token_mask` correctly identifies positions 1-4
- [ ] Text start position is 5 (after CLS + 4 hubs)
- [ ] Padding and truncation work correctly with hub token overhead

**Tests:** `tests/v3/test_hub_tokens.py::test_hub_tokenizer`

---

#### Issue 1.2.3: Implement Semantic Centroid Initialization

**File:** `src/modeling_studio/models/hub_initialization_v3.py`
**Effort:** 4 hours
**Dependencies:** Issues 1.2.1, 1.2.2

**Description:**
Initialize hub token embeddings as the centroid (mean) of semantically related word embeddings from v2.

**Implementation:**

```python
# src/modeling_studio/models/hub_initialization_v3.py

import torch
import torch.nn as nn
from typing import Dict, List
from transformers import AutoTokenizer, AutoModel

from .hub_tokens import HUB_TOKEN_REGISTRY, get_semantic_seeds

def compute_semantic_centroid(
    word_list: List[str],
    tokenizer,
    embeddings: nn.Embedding,
) -> torch.Tensor:
    """
    Compute the centroid embedding for a list of semantic seed words.

    Args:
        word_list: List of seed words (e.g., ["happy", "sad", "angry"])
        tokenizer: Tokenizer to convert words to IDs
        embeddings: Embedding layer to look up vectors

    Returns:
        Centroid vector (mean of all seed word embeddings)
    """
    vectors = []

    for word in word_list:
        # Tokenize word (may produce multiple subword tokens)
        token_ids = tokenizer.encode(word, add_special_tokens=False)

        if len(token_ids) == 0:
            continue

        # Get embeddings for all subword tokens
        with torch.no_grad():
            word_embeds = embeddings(torch.tensor(token_ids))
            # Mean pool if multiple subwords
            word_vector = word_embeds.mean(dim=0)
            vectors.append(word_vector)

    if len(vectors) == 0:
        raise ValueError(f"No valid embeddings found for words: {word_list}")

    # Stack and compute centroid
    stacked = torch.stack(vectors, dim=0)
    centroid = stacked.mean(dim=0)

    return centroid


def initialize_hub_tokens_semantic(
    model,
    v2_tokenizer,
    v2_embeddings: torch.Tensor,
) -> None:
    """
    Initialize hub token embeddings using semantic centroids.

    This positions hub tokens in meaningful regions of the embedding space:
    - [EMO] near emotion words (happy, sad, angry, ...)
    - [MEM] near memory words (remember, past, history, ...)
    - [REL] near relationship words (family, mother, father, ...)
    - [TASK] near action words (do, want, need, help, ...)

    Args:
        model: The v3 model with hub tokens in vocabulary
        v2_tokenizer: Original v2 tokenizer (without hub tokens)
        v2_embeddings: v2 word embedding matrix [vocab_size, hidden_size]
    """
    print("\n🎯 Initializing hub tokens with semantic centroids...")

    # Create temporary embedding layer from v2 weights
    temp_embeddings = nn.Embedding.from_pretrained(v2_embeddings, freeze=True)

    # Get the v3 model's embedding layer
    v3_embeddings = model.get_input_embeddings()

    for hub_token, spec in HUB_TOKEN_REGISTRY.items():
        seed_words = spec.semantic_seeds

        print(f"  {hub_token}: Computing centroid from {seed_words[:3]}...")

        # Compute centroid
        centroid = compute_semantic_centroid(
            word_list=seed_words,
            tokenizer=v2_tokenizer,
            embeddings=temp_embeddings,
        )

        # Get hub token ID in v3 vocabulary
        hub_token_id = model.config.hub_token_ids.get(hub_token)
        if hub_token_id is None:
            # Try to get from tokenizer
            hub_token_id = model.tokenizer.convert_tokens_to_ids(hub_token)

        # Set the embedding
        with torch.no_grad():
            v3_embeddings.weight[hub_token_id] = centroid

        print(f"    ✓ {hub_token} initialized at position {hub_token_id}")

    print("\n✓ Hub tokens initialized with semantic centroids")


def verify_hub_token_initialization(
    model,
    v2_tokenizer,
    v2_embeddings: torch.Tensor,
) -> Dict[str, float]:
    """
    Verify hub tokens are positioned near their semantic domains.

    Returns cosine similarity between hub token and its seed word centroid.
    """
    from torch.nn.functional import cosine_similarity

    temp_embeddings = nn.Embedding.from_pretrained(v2_embeddings, freeze=True)
    v3_embeddings = model.get_input_embeddings()

    results = {}

    for hub_token, spec in HUB_TOKEN_REGISTRY.items():
        # Compute expected centroid
        centroid = compute_semantic_centroid(
            word_list=spec.semantic_seeds,
            tokenizer=v2_tokenizer,
            embeddings=temp_embeddings,
        )

        # Get actual hub token embedding
        hub_token_id = model.tokenizer.convert_tokens_to_ids(hub_token)
        actual = v3_embeddings.weight[hub_token_id]

        # Compute similarity
        sim = cosine_similarity(actual.unsqueeze(0), centroid.unsqueeze(0)).item()
        results[hub_token] = sim

        status = "✓" if sim > 0.99 else "⚠️"
        print(f"  {status} {hub_token}: cosine_sim = {sim:.4f}")

    return results
```

**Acceptance Criteria:**

- [ ] Centroid correctly computed as mean of seed word embeddings
- [ ] Multi-subword tokens handled (mean pooled)
- [ ] Hub token embeddings updated in-place in v3 model
- [ ] Verification shows cosine similarity > 0.99 for all hub tokens
- [ ] Handles OOV seed words gracefully

**Tests:** `tests/v3/test_hub_tokens.py::test_semantic_centroid_initialization`

---

#### Issue 1.2.4: Implement Hub Token Pooler

**File:** `src/modeling_studio/models/poolers_v3.py`
**Effort:** 3 hours
**Dependencies:** Issue 1.2.1

**Description:**
Implement a pooler that extracts hub token representations for routing to capability heads.

**Implementation:**

```python
# src/modeling_studio/models/poolers_v3.py

import torch
import torch.nn as nn
from typing import Dict, Optional

from .hub_tokens import get_hub_positions, HUB_TOKEN_REGISTRY

class HubTokenPooler(nn.Module):
    """
    Extracts hub token representations from the final hidden states.

    Given sequence: [CLS] [EMO] [MEM] [REL] [TASK] <text> [SEP] [PAD]...
    Returns dict of hub token representations for routing to heads.
    """

    def __init__(self, hidden_size: int = 768, add_projection: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.hub_positions = get_hub_positions()

        # Optional projection layer (like BERT's pooler)
        self.add_projection = add_projection
        if add_projection:
            self.projections = nn.ModuleDict({
                token.replace("[", "").replace("]", ""): nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                )
                for token in HUB_TOKEN_REGISTRY.keys()
            })

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract hub token representations.

        Args:
            hidden_states: Final layer output [batch, seq_len, hidden]
            attention_mask: Optional attention mask [batch, seq_len]

        Returns:
            Dict mapping hub token names to their representations [batch, hidden]
        """
        batch_size = hidden_states.size(0)

        pooled = {}

        # Extract each hub token
        for token, position in self.hub_positions.items():
            # Get representation at hub position
            hub_repr = hidden_states[:, position, :]  # [batch, hidden]

            # Apply projection if enabled
            if self.add_projection and token in HUB_TOKEN_REGISTRY:
                key = token.replace("[", "").replace("]", "")
                hub_repr = self.projections[key](hub_repr)

            pooled[token] = hub_repr

        return pooled

    def get_pooled_for_capability(
        self,
        hidden_states: torch.Tensor,
        capability: str,
    ) -> torch.Tensor:
        """
        Get the pooled representation for a specific capability.

        Args:
            hidden_states: Final layer output [batch, seq_len, hidden]
            capability: Capability name (e.g., "emotions", "intent")

        Returns:
            Pooled representation [batch, hidden] from the appropriate hub
        """
        from .hub_tokens import get_hub_for_capability

        hub_token = get_hub_for_capability(capability)
        position = self.hub_positions[hub_token]

        return hidden_states[:, position, :]


class CombinedPooler(nn.Module):
    """
    Combined pooler that provides CLS, Mean, and Hub token pooling.
    """

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size
        self.hub_pooler = HubTokenPooler(hidden_size)

        # CLS projection (standard BERT-style)
        self.cls_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Get all pooled representations.

        Returns:
            Dict with:
                - "[CLS]": CLS token representation (projected)
                - "[EMO]", "[MEM]", "[REL]", "[TASK]": Hub representations
                - "mean": Mean-pooled representation (excluding special tokens)
        """
        # Hub token pooling
        pooled = self.hub_pooler(hidden_states, attention_mask)

        # CLS projection
        pooled["[CLS]_projected"] = self.cls_projection(pooled["[CLS]"])

        # Mean pooling (exclude special tokens at positions 0-4)
        if attention_mask is not None:
            # Mask out CLS and hub tokens from mean pooling
            mean_mask = attention_mask.clone()
            mean_mask[:, :5] = 0  # Zero out [CLS] and 4 hub positions

            # Expand mask for broadcasting
            mask_expanded = mean_mask.unsqueeze(-1).float()
            sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled["mean"] = sum_hidden / sum_mask
        else:
            # Simple mean over text positions (5 onwards)
            pooled["mean"] = hidden_states[:, 5:, :].mean(dim=1)

        return pooled
```

**Acceptance Criteria:**

- [ ] Correctly extracts representations at positions 0-4
- [ ] `get_pooled_for_capability()` returns correct hub for each capability
- [ ] Mean pooling excludes CLS and hub tokens
- [ ] Optional projection layer works correctly
- [ ] Handles variable sequence lengths

**Tests:** `tests/v3/test_hub_tokens.py::test_hub_token_pooler`

---

#### Issue 1.2.5: Implement Hub-to-Capability Routing

**File:** `src/modeling_studio/models/hub_tokens.py` (extend) + `src/modeling_studio/models/routing_v3.py`
**Effort:** 3 hours
**Dependencies:** Issues 1.2.1, 1.2.4

**Description:**
Implement the routing logic that directs hub token representations to appropriate capability heads.

**Implementation:**

```python
# src/modeling_studio/models/routing_v3.py

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .hub_tokens import (
    HUB_TOKEN_REGISTRY,
    TOKEN_LEVEL_CAPABILITIES,
    get_hub_for_capability,
    get_capabilities_for_hub,
)

class HubRouter(nn.Module):
    """
    Routes hub token representations to capability heads.

    For each capability, determines:
    1. Which hub token provides the representation
    2. Whether to use hub pooling or per-token representations
    """

    # Routing table: capability -> (pool_type, hub_token)
    ROUTING_TABLE = {
        # EMO hub capabilities (sequence-level)
        "emotions": ("hub", "[EMO]"),
        "sentiment": ("hub", "[EMO]"),
        "safety_generic": ("hub", "[EMO]"),
        "safety_familyos": ("hub", "[EMO]"),

        # MEM hub capabilities (sequence-level)
        "embedding": ("hub", "[MEM]"),

        # REL hub capabilities (sequence-level, may use pair encoder)
        "nli": ("hub", "[REL]"),
        "relation": ("hub", "[REL]"),

        # TASK hub capabilities (sequence-level)
        "intent": ("hub", "[TASK]"),
        "ingress": ("hub", "[TASK]"),

        # Token-level capabilities (use per-token representations)
        "ner_general": ("token", None),
        "ner_family": ("token", None),
        "temporal": ("token", None),
    }

    def __init__(self):
        super().__init__()

    def get_representation_for_capability(
        self,
        hidden_states: torch.Tensor,
        pooled_outputs: Dict[str, torch.Tensor],
        capability: str,
    ) -> Tuple[torch.Tensor, str]:
        """
        Get the appropriate representation for a capability.

        Args:
            hidden_states: Full sequence hidden states [batch, seq_len, hidden]
            pooled_outputs: Dict of hub token representations from pooler
            capability: Target capability name

        Returns:
            Tuple of (representation, pool_type)
            - For hub capabilities: ([batch, hidden], "hub")
            - For token capabilities: ([batch, seq_len, hidden], "token")
        """
        pool_type, hub_token = self.ROUTING_TABLE.get(capability, ("hub", "[CLS]"))

        if pool_type == "token":
            # Return full sequence for token-level classification
            return hidden_states, "token"
        else:
            # Return hub token representation
            return pooled_outputs[hub_token], "hub"

    def get_hub_gradient_mask(
        self,
        active_capabilities: List[str],
        batch_size: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Create gradient masks for hub tokens based on active capabilities.

        Used during training to ensure gradients only flow through relevant hubs.

        Args:
            active_capabilities: List of capabilities being trained this batch
            batch_size: Current batch size
            device: Target device

        Returns:
            Dict mapping hub tokens to gradient masks [batch]
        """
        masks = {}

        for hub_token in HUB_TOKEN_REGISTRY.keys():
            hub_capabilities = get_capabilities_for_hub(hub_token)

            # Hub should receive gradient if any of its capabilities are active
            should_train = any(cap in active_capabilities for cap in hub_capabilities)

            masks[hub_token] = torch.ones(batch_size, device=device) * float(should_train)

        return masks


class CapabilityHead(nn.Module):
    """
    Wrapper for a capability head that handles hub routing.
    """

    def __init__(
        self,
        capability: str,
        head: nn.Module,
        hidden_size: int = 768,
    ):
        super().__init__()
        self.capability = capability
        self.head = head
        self.hidden_size = hidden_size

        pool_type, hub_token = HubRouter.ROUTING_TABLE.get(capability, ("hub", "[CLS]"))
        self.pool_type = pool_type
        self.hub_token = hub_token

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooled_outputs: Dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with automatic hub routing.

        Args:
            hidden_states: Full sequence [batch, seq_len, hidden]
            pooled_outputs: Hub token representations
            **kwargs: Additional arguments for the head

        Returns:
            Head output logits
        """
        if self.pool_type == "token":
            # Token-level head (NER, temporal)
            return self.head(hidden_states, **kwargs)
        else:
            # Hub-routed head
            representation = pooled_outputs[self.hub_token]
            return self.head(representation, **kwargs)


def create_hub_routing_info(capability: str) -> Dict:
    """
    Get routing information for a capability.

    Returns:
        Dict with pool_type, hub_token, and description
    """
    pool_type, hub_token = HubRouter.ROUTING_TABLE.get(capability, ("hub", "[CLS]"))

    info = {
        "capability": capability,
        "pool_type": pool_type,
        "hub_token": hub_token,
    }

    if hub_token and hub_token in HUB_TOKEN_REGISTRY:
        info["hub_description"] = HUB_TOKEN_REGISTRY[hub_token].description

    return info
```

**Acceptance Criteria:**

- [ ] All 12 capabilities correctly mapped to routing types
- [ ] Hub capabilities (9) route through appropriate hub tokens
- [ ] Token-level capabilities (3) receive full sequence representations
- [ ] Gradient masks correctly identify which hubs should be trained
- [ ] `CapabilityHead` wrapper correctly handles both routing types

**Tests:** `tests/v3/test_hub_tokens.py::test_hub_routing`

---

## 🏁 Milestone 2: v3 Attention & Transformer Layers

**Goal:** Implement multi-scale sliding window attention with global hub tokens
**Estimated Effort:** 6 days
**Dependencies:** Milestone 1 complete (config_v3.py, hub_tokens.py)

### Epic 2.1: Sliding Window Attention

#### Issue 2.1.1: Implement Global-Local Attention Mask Creation

**File:** `src/modeling_studio/models/attention_v3.py`
**Effort:** 4 hours
**Dependencies:** Issue 1.1.1 (config with global_attention_positions)

**Description:**
Create attention mask that combines sliding windows for text tokens with global attention for hub tokens (positions 0-4). This solves the "Blind Hub" problem where hub tokens couldn't see beyond their local window.

**Implementation:**

```python
# src/modeling_studio/models/attention_v3.py

import torch
from typing import List, Optional

# Global token positions (exempt from sliding window)
GLOBAL_TOKEN_POSITIONS = [0, 1, 2, 3, 4]  # [CLS], [EMO], [MEM], [REL], [TASK]

def create_global_local_attention_mask(
    seq_len: int,
    window_size: int,
    global_positions: List[int] = GLOBAL_TOKEN_POSITIONS,
    device: torch.device = None,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """
    Create attention mask with global tokens + sliding windows.

    Global tokens (positions 0-4) can attend to ALL positions.
    ALL positions can attend to global tokens.
    Text tokens use sliding window attention for other text tokens.

    Args:
        seq_len: Sequence length
        window_size: Sliding window size for text tokens
        global_positions: Positions with global attention (default: 0-4)
        device: Target device
        dtype: Output dtype (bool for mask, float for additive)

    Returns:
        Attention mask [seq_len, seq_len] where True/1 = can attend

    Visual example (seq_len=10, window=4, globals=[0,1,2,3,4]):

              0  1  2  3  4  5  6  7  8  9   (keys)
           +--------------------------------
       0   |  1  1  1  1  1  1  1  1  1  1   <- [CLS] global
       1   |  1  1  1  1  1  1  1  1  1  1   <- [EMO] global
       2   |  1  1  1  1  1  1  1  1  1  1   <- [MEM] global
       3   |  1  1  1  1  1  1  1  1  1  1   <- [REL] global
       4   |  1  1  1  1  1  1  1  1  1  1   <- [TASK] global
       5   |  1  1  1  1  1  1  1  1  0  0   <- text: globals + window
       6   |  1  1  1  1  1  0  1  1  1  0   <- text: globals + window
       7   |  1  1  1  1  1  0  0  1  1  1   <- text: globals + window
       8   |  1  1  1  1  1  0  0  0  1  1   <- text: globals + window
       9   |  1  1  1  1  1  0  0  0  0  1   <- text: globals + window
    (queries)
    """
    if device is None:
        device = torch.device("cpu")

    # Start with zeros (no attention)
    mask = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)

    # Global tokens can attend to everything (rows)
    for pos in global_positions:
        if pos < seq_len:
            mask[pos, :] = 1

    # Everything can attend to global tokens (columns)
    for pos in global_positions:
        if pos < seq_len:
            mask[:, pos] = 1

    # Sliding window for non-global positions
    half_window = window_size // 2
    for i in range(seq_len):
        if i in global_positions:
            continue  # Already handled

        # Window range
        start = max(0, i - half_window)
        end = min(seq_len, i + half_window + 1)
        mask[i, start:end] = 1

    return mask


def create_causal_global_local_mask(
    seq_len: int,
    window_size: int,
    global_positions: List[int] = GLOBAL_TOKEN_POSITIONS,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Create CAUSAL attention mask (for decoder-style, if needed).
    Combines global attention + sliding window + causal masking.
    """
    mask = create_global_local_attention_mask(
        seq_len, window_size, global_positions, device
    )

    # Apply causal mask (upper triangle = 0)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    mask = mask * causal_mask

    return mask


def expand_mask_for_batch(
    mask: torch.Tensor,
    batch_size: int,
    num_heads: int,
) -> torch.Tensor:
    """
    Expand 2D mask to 4D for multi-head attention.

    Args:
        mask: [seq_len, seq_len]
        batch_size: Batch size
        num_heads: Number of attention heads

    Returns:
        Expanded mask [batch, num_heads, seq_len, seq_len]
    """
    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
```

**Acceptance Criteria:**

- [ ] Global positions (0-4) have full row attention (can see everything)
- [ ] Global positions have full column attention (everyone sees them)
- [ ] Text tokens use sliding window for non-global positions
- [ ] Mask shape is [seq_len, seq_len] or [batch, heads, seq_len, seq_len]
- [ ] Works with variable sequence lengths

**Tests:** `tests/v3/test_attention_v3.py::test_global_local_mask`

---

#### Issue 2.1.2: Implement Layer-wise Window Size Configuration

**File:** `src/modeling_studio/models/attention_v3.py` (extend)
**Effort:** 2 hours
**Dependencies:** Issue 1.1.1, Issue 2.1.1

**Description:**
Configure different sliding window sizes per layer band (64→128→256→512).

**Implementation:**

```python
# Add to attention_v3.py

from typing import Dict

# Window sizes by layer band
LAYER_WINDOW_CONFIG: Dict[int, int] = {
    # Foundation Band: Local token interactions (64)
    1: 64, 2: 64, 3: 64, 4: 64, 5: 64, 6: 64,

    # Context Band: Phrase-level patterns (128)
    7: 128, 8: 128, 9: 128, 10: 128, 11: 128, 12: 128,
    13: 128, 14: 128, 15: 128, 16: 128, 17: 128, 18: 128,

    # Semantic Band: Sentence-level semantics (256)
    19: 256, 20: 256, 21: 256, 22: 256,

    # Family Band: Full context (512)
    23: 512, 24: 512, 25: 512, 26: 512, 27: 512, 28: 512,
}

# Band definitions for easy lookup
LAYER_BANDS = {
    "foundation": (1, 6, 64),     # (start, end, window)
    "context": (7, 18, 128),
    "semantic": (19, 22, 256),
    "family": (23, 28, 512),
}


def get_window_size_for_layer(layer_idx: int) -> int:
    """
    Get the sliding window size for a given layer.

    Args:
        layer_idx: 1-indexed layer number (1-28)

    Returns:
        Window size (64, 128, 256, or 512)
    """
    if layer_idx in LAYER_WINDOW_CONFIG:
        return LAYER_WINDOW_CONFIG[layer_idx]

    # Fallback: determine from band
    if 1 <= layer_idx <= 6:
        return 64
    elif 7 <= layer_idx <= 18:
        return 128
    elif 19 <= layer_idx <= 22:
        return 256
    elif 23 <= layer_idx <= 28:
        return 512
    else:
        raise ValueError(f"Invalid layer index: {layer_idx}. Must be 1-28.")


def get_layer_band_name(layer_idx: int) -> str:
    """Get the band name for a layer."""
    if 1 <= layer_idx <= 6:
        return "foundation"
    elif 7 <= layer_idx <= 18:
        return "context"
    elif 19 <= layer_idx <= 22:
        return "semantic"
    elif 23 <= layer_idx <= 28:
        return "family"
    else:
        raise ValueError(f"Invalid layer index: {layer_idx}")


def get_attention_mask_for_layer(
    layer_idx: int,
    seq_len: int,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Get the appropriate attention mask for a specific layer.

    Args:
        layer_idx: 1-indexed layer number
        seq_len: Sequence length
        device: Target device

    Returns:
        Attention mask [seq_len, seq_len]
    """
    window_size = get_window_size_for_layer(layer_idx)

    return create_global_local_attention_mask(
        seq_len=seq_len,
        window_size=window_size,
        global_positions=GLOBAL_TOKEN_POSITIONS,
        device=device,
    )


def print_layer_config():
    """Print the layer configuration for debugging."""
    print("\n📊 Layer Window Configuration:")
    print("-" * 50)
    for band_name, (start, end, window) in LAYER_BANDS.items():
        print(f"  {band_name.capitalize():12} (L{start:2}-{end:2}): window = {window}")
    print("-" * 50)
```

**Acceptance Criteria:**

- [ ] Foundation (L1-6) uses 64-token window
- [ ] Context (L7-18) uses 128-token window
- [ ] Semantic (L19-22) uses 256-token window
- [ ] Family (L23-28) uses 512-token window
- [ ] Invalid layer indices raise ValueError
- [ ] All 28 layers have defined window sizes

**Tests:** `tests/v3/test_attention_v3.py::test_layer_window_config`

---

#### Issue 2.1.3: Implement MultiScaleAttentionWithGlobals

**File:** `src/modeling_studio/models/attention_v3.py` (extend)
**Effort:** 6 hours
**Dependencies:** Issues 2.1.1, 2.1.2

**Description:**
Implement the full multi-head attention module with sliding windows and global hub tokens.

**Implementation:**

```python
# Add to attention_v3.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class MultiScaleAttentionWithGlobals(nn.Module):
    """
    Multi-head attention with:
    - Sliding window for text tokens
    - Global attention for hub tokens (positions 0-4)
    - Layer-specific window sizes

    This is the v3.3 solution to the "Blind Hub" problem.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        attention_dropout: float = 0.1,
        layer_idx: int = 1,
        max_position_embeddings: int = 8192,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.layer_idx = layer_idx
        self.window_size = get_window_size_for_layer(layer_idx)

        assert hidden_size % num_attention_heads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_attention_heads})"

        # QKV projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(attention_dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Cache for attention mask (avoid recomputing)
        self._cached_mask: Optional[torch.Tensor] = None
        self._cached_seq_len: int = 0

    def _get_attention_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Get or create cached attention mask."""
        if self._cached_mask is None or self._cached_seq_len != seq_len:
            self._cached_mask = create_global_local_attention_mask(
                seq_len=seq_len,
                window_size=self.window_size,
                global_positions=GLOBAL_TOKEN_POSITIONS,
                device=device,
                dtype=torch.float32,
            )
            self._cached_seq_len = seq_len

        return self._cached_mask.to(device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with global-local attention.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Optional padding mask [batch, seq_len]
            output_attentions: Whether to return attention weights

        Returns:
            Tuple of (output, attention_weights or None)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape for multi-head attention: [batch, heads, seq, head_dim]
        query = query.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        # Shape: [batch, heads, seq, seq]

        # Apply global-local mask
        global_local_mask = self._get_attention_mask(seq_len, hidden_states.device)
        # Convert to additive mask: 0 -> 0, 1 -> -inf
        additive_mask = (1.0 - global_local_mask) * torch.finfo(attn_weights.dtype).min
        attn_weights = attn_weights + additive_mask.unsqueeze(0).unsqueeze(0)

        # Apply padding mask if provided
        if attention_mask is not None:
            # attention_mask: [batch, seq_len] where 1 = valid, 0 = padding
            padding_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).float()) * torch.finfo(attn_weights.dtype).min
            attn_weights = attn_weights + padding_mask

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value)
        # Shape: [batch, heads, seq, head_dim]

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        # Output projection
        attn_output = self.out_proj(attn_output)

        if output_attentions:
            return attn_output, attn_weights
        return attn_output, None

    def extra_repr(self) -> str:
        return f"layer={self.layer_idx}, window={self.window_size}, heads={self.num_attention_heads}"
```

**Acceptance Criteria:**

- [ ] QKV projections correctly sized (768 → 768)
- [ ] Multi-head reshape is correct (12 heads × 64 dim)
- [ ] Global-local mask applied correctly
- [ ] Padding mask combined with global-local mask
- [ ] Output shape matches input shape
- [ ] Attention weights can be returned for debugging

**Tests:** `tests/v3/test_attention_v3.py::test_multi_scale_attention`

---

#### Issue 2.1.4: Integrate Flash Attention 2 with Safety Switch

**File:** `src/modeling_studio/models/attention_v3.py` (extend)
**Effort:** 6 hours
**Dependencies:** Issue 2.1.3

> ⚠️ **BLOCKER RISK**: Flash Attention 2's sliding window kernels prevent text tokens from
> seeing Hub tokens (indices 0-4) if they are outside the window radius.
> See Risk Assessment section for full analysis.

**Description:**
Implement a "Safety Switch" strategy that enforces the correct attention implementation
based on the training/inference phase.

### The "Blind Hub" Problem

Standard Flash Attention sliding window:

- ✅ Hub→Text: Solvable via manual calculation (Hubs see everything)
- ❌ Text→Hub: NOT supported - text tokens outside window cannot see `[EMO]`/`[REL]`

This breaks the core v3 architecture where text embeddings must be conditioned on hub tokens.

### Mitigation Strategy: The Safety Switch

**Decision Matrix:**

| Phase | Implementation | Reason |
|-------|----------------|--------|
| Training (Phase 1) | `MultiScaleAttentionWithGlobals` + SDPA | Correctness is non-negotiable |
| Inference (short <2k) | `MultiScaleAttentionWithGlobals` | Negligible speed difference |
| Inference (long 8k+) | `FlashAttentionWithGlobals` | Accept Text→Hub blindness trade-off |

**Key Insight:** The standard implementation will use `F.scaled_dot_product_attention` (SDPA),
which is memory-efficient and often accelerated in PyTorch 2.0+, preventing OOM errors
even without Flash Attention.

**Implementation:**

```python
# Add to attention_v3.py

import torch.nn.functional as F
import math

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class FlashAttentionWithGlobals(nn.Module):
    """
    Flash Attention 2 implementation with Global Hub Token support.

    ⚠️ MITIGATION STRATEGY:
    1. Hub->Text Attention: ✅ Solved via manual calculation (Hubs see everything).
    2. Text->Hub Attention: ❌ NOT natively supported in Flash sliding window.
       - Impact: Text tokens may not see [EMO]/[REL] if window is small.
       - Use: ONLY for long-context inference where speed > perfect topology.
       - Training: Use MultiScaleAttentionWithGlobals instead.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        attention_dropout: float = 0.1,
        layer_idx: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.layer_idx = layer_idx
        self.window_size = get_window_size_for_layer(layer_idx)
        self.dropout_p = attention_dropout

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward with Flash Attention 2 + Hub correction.

        ⚠️ Text->Hub attention is NOT preserved. Use only for long-context inference.
        """
        # Flash Attention doesn't support output_attentions
        if output_attentions:
            raise ValueError(
                "Flash Attention does not support output_attentions=True. "
                "Use MultiScaleAttentionWithGlobals for debugging."
            )

        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape [batch, seq, heads, dim] for Flash Attention
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)

        # 1. Main Flash Attention (Sliding Window)
        # ⚠️ WARNING: Text tokens > window_size/2 away from 0 will NOT see global tokens 0-4
        attn_output = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            causal=False,
            window_size=(self.window_size // 2, self.window_size // 2),
        )

        # 2. Hub->Text Correction (Global Tokens 0-4 see EVERYTHING)
        # We manually compute attention for indices 0-4 using standard attention
        global_q = q[:, :5, :, :]  # [batch, 5, heads, dim]

        # Standard attention scores for global query positions
        global_scores = torch.einsum('bqhd,bkhd->bhqk', global_q, k) / math.sqrt(self.head_dim)

        # Apply padding mask if provided
        if attention_mask is not None:
            # Expand mask: 1.0 is keep, 0.0 is mask -> additive: 0.0 keep, -inf mask
            padding_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).float()) * -10000.0
            global_scores = global_scores + padding_mask

        global_probs = F.softmax(global_scores, dim=-1)
        global_out_correction = torch.einsum('bhqk,bkhd->bqhd', global_probs, v)

        # Overwrite Flash output for positions 0-4 with correct global attention
        attn_output[:, :5, :, :] = global_out_correction

        # Reshape and project
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        return self.out_proj(attn_output), None

    def extra_repr(self) -> str:
        return (
            f"layer={self.layer_idx}, window={self.window_size}, "
            f"heads={self.num_attention_heads}, "
            f"⚠️ Text->Hub blind (use for inference only)"
        )


def create_attention_layer(
    hidden_size: int = 768,
    num_attention_heads: int = 12,
    attention_dropout: float = 0.1,
    layer_idx: int = 1,
    use_flash_attention: bool = True,
) -> nn.Module:
    """
    Factory function implementing the DECISION MATRIX (Safety Switch).

    Decision Logic:
    1. If Flash Attention missing → Standard (SDPA optimized)
    2. If use_flash_attention=False → Standard (for Training Phase)
    3. If use_flash_attention=True & available → Flash (for Long Inference)

    ⚠️ CRITICAL: For Phase 1 Training, set use_flash_attention=False in config
    to ensure Text->Hub attention is preserved.

    Args:
        use_flash_attention: Whether to use Flash Attention.
            - Training: Set to False (correctness > speed)
            - Inference 8k+: Set to True (speed, accept Text->Hub blindness)

    Returns:
        Attention module (Flash or Standard with SDPA)
    """
    # SAFETY SWITCH: Force standard attention for correctness
    # Controlled via 'use_flash_attention' in model config
    # Phase 1 Training config MUST set: use_flash_attention: false

    if use_flash_attention and FLASH_ATTN_AVAILABLE:
        return FlashAttentionWithGlobals(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            layer_idx=layer_idx,
        )
    else:
        # Standard attention with SDPA optimization
        # MultiScaleAttentionWithGlobals should use F.scaled_dot_product_attention
        # for memory efficiency even without Flash Attention
        return MultiScaleAttentionWithGlobals(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            layer_idx=layer_idx,
        )
```

### Issue 2.1.4b: Upgrade MultiScaleAttentionWithGlobals to use SDPA

**Description:**
Update the standard attention implementation to use `F.scaled_dot_product_attention` (SDPA)
for memory efficiency. This provides near-Flash-Attention performance while supporting
custom attention masks required for global hub tokens.

**Implementation Snippet (update to Issue 2.1.3):**

```python
# Update forward() method in MultiScaleAttentionWithGlobals

def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Forward pass with global-local attention using SDPA optimization.
    """
    batch_size, seq_len, _ = hidden_states.shape

    # Project to Q, K, V
    query = self.q_proj(hidden_states)
    key = self.k_proj(hidden_states)
    value = self.v_proj(hidden_states)

    # Reshape: [batch, heads, seq, head_dim]
    query = query.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
    key = key.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
    value = value.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

    # Create combined attention mask (global-local + padding)
    global_local_mask = self._get_attention_mask(seq_len, hidden_states.device)

    if attention_mask is not None:
        # Combine: global_local_mask AND padding_mask
        padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq]
        combined_mask = global_local_mask.unsqueeze(0) * padding_mask.float()
    else:
        combined_mask = global_local_mask.unsqueeze(0).expand(batch_size, -1, -1)

    # Convert to boolean mask for SDPA (True = MASK OUT, False = attend)
    attn_mask = combined_mask == 0

    if output_attentions:
        # Fall back to manual attention for debugging
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn_weights = attn_weights.masked_fill(attn_mask.unsqueeze(1), float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value)
    else:
        # Use SDPA for memory-efficient attention (PyTorch 2.0+)
        # SDPA automatically uses Flash/Memory-Efficient kernels when possible
        attn_output = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_mask.unsqueeze(1).expand(-1, self.num_attention_heads, -1, -1),
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )
        attn_weights = None

    # Reshape back: [batch, seq, hidden]
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

    # Output projection
    attn_output = self.out_proj(attn_output)

    return attn_output, attn_weights
```

**Acceptance Criteria:**

- [ ] **SAFETY SWITCH**: Training config sets `use_flash_attention: false`
- [ ] `MultiScaleAttentionWithGlobals` uses `F.scaled_dot_product_attention` (SDPA)
- [ ] SDPA fallback works when custom mask provided (no OOM on 8k sequences)
- [ ] `FlashAttentionWithGlobals` correctly handles Hub→Text (positions 0-4)
- [ ] ⚠️ Text→Hub blindness is DOCUMENTED and accepted for inference only
- [ ] Factory function implements Decision Matrix correctly
- [ ] `output_attentions=True` raises clear error for Flash path
- [ ] `extra_repr()` includes visibility warning for Flash variant

**Tests:**

```
tests/v3/test_attention_v3.py::test_flash_attention_hub_correction
tests/v3/test_attention_v3.py::test_sdpa_memory_efficiency
tests/v3/test_attention_v3.py::test_safety_switch_factory
tests/v3/test_attention_v3.py::test_text_to_hub_visibility_standard
```

**Config Example (Training - Safe):**

```yaml
# configs/training/v3_phase1.yaml
model:
  attention:
    use_flash_attention: false  # ⚠️ CRITICAL: Preserve Text->Hub attention
```

**Config Example (Inference - Fast):**

```yaml
# configs/inference/v3_long_context.yaml
model:
  attention:
    use_flash_attention: true  # Accept Text->Hub blindness for 8k+ speed
```

---

### Epic 2.2: FFN & Transformer Layer

#### Issue 2.2.1: Implement GELU FFN Module

**File:** `src/modeling_studio/models/ffn_v3.py`
**Effort:** 2 hours
**Dependencies:** Issue 1.1.1 (config)

**Description:**
Implement the GELU Feed-Forward Network used in all 28 layers. This is the same as v2 (no SwiGLU upgrade per v3.3 decision).

**Implementation:**

```python
# src/modeling_studio/models/ffn_v3.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class GELUFFN(nn.Module):
    """
    GELU Feed-Forward Network (same as v2).

    Architecture:
        hidden → intermediate (4x) → GELU → hidden
        768 → 3072 → GELU → 768

    Note: SwiGLU was considered for v3 Phase 2 but removed from roadmap
    per v3.3 decision (stability > marginal gains).
    """

    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Up projection: 768 → 3072
        self.up_proj = nn.Linear(hidden_size, intermediate_size)

        # Down projection: 3072 → 768
        self.down_proj = nn.Linear(intermediate_size, hidden_size)

        # Dropout
        self.dropout = nn.Dropout(hidden_dropout_prob)

        # Activation
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "gelu_new":
            self.activation = self._gelu_new
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unknown activation: {activation}")

    @staticmethod
    def _gelu_new(x: torch.Tensor) -> torch.Tensor:
        """
        GELU approximation (used in some models).
        """
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
        ))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            Output: [batch, seq_len, hidden_size]
        """
        # Up project
        intermediate = self.up_proj(hidden_states)

        # Activation
        intermediate = self.activation(intermediate)

        # Down project
        output = self.down_proj(intermediate)

        # Dropout
        output = self.dropout(output)

        return output

    def extra_repr(self) -> str:
        return f"hidden={self.hidden_size}, intermediate={self.intermediate_size}"


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network (DEPRECATED - R&D only).

    NOT used in v3 production. Kept for research experiments.

    Architecture:
        hidden → gate (4x) → SiLU
        hidden → up (4x)
        gate * up → down → hidden
    """

    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
    ):
        super().__init__()

        # SwiGLU uses 2/3 of intermediate for gate and up each
        # to maintain same param count as GELU FFN
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """SwiGLU forward: SiLU(gate) * up → down"""
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        intermediate = gate * up
        output = self.down_proj(intermediate)
        output = self.dropout(output)
        return output


def create_ffn(
    hidden_size: int = 768,
    intermediate_size: int = 3072,
    hidden_dropout_prob: float = 0.1,
    ffn_type: str = "gelu",
) -> nn.Module:
    """
    Factory function to create FFN module.

    Args:
        ffn_type: "gelu" (default) or "swiglu" (R&D only)
    """
    if ffn_type == "gelu":
        return GELUFFN(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
        )
    elif ffn_type == "swiglu":
        print("⚠️ SwiGLU is R&D only - not recommended for production")
        return SwiGLUFFN(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
        )
    else:
        raise ValueError(f"Unknown FFN type: {ffn_type}")
```

**Acceptance Criteria:**

- [ ] GELU activation applied correctly
- [ ] Dimensions: 768 → 3072 → 768
- [ ] Dropout applied after down projection
- [ ] SwiGLU available for R&D (not production)
- [ ] Factory function returns correct type

**Tests:** `tests/v3/test_layers_v3.py::test_gelu_ffn`

---

#### Issue 2.2.2: Implement v3 LoRA Layer

**File:** `src/modeling_studio/models/lora_v3.py`
**Effort:** 4 hours
**Dependencies:** Issue 1.1.1 (config with LoRA settings)

**Description:**
Implement LoRA (Low-Rank Adaptation) for layers 23-28 (Family Band). LoRA enables efficient fine-tuning by adding trainable low-rank matrices to frozen weights.

**Implementation:**

```python
# src/modeling_studio/models/lora_v3.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Set

class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer for efficient fine-tuning.

    Adds trainable low-rank matrices A and B to a frozen weight matrix W:
        output = (W + BA) @ x = W @ x + B @ (A @ x)

    Where:
        - W: Original frozen weights [out_features, in_features]
        - A: Down projection [r, in_features]
        - B: Up projection [out_features, r]
        - r: LoRA rank (default: 16)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        alpha: int = 16,
        dropout: float = 0.05,
    ):
        super().__init__()

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # LoRA matrices
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Initialize: A with normal, B with zeros
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        LoRA forward pass.

        Returns the LoRA delta (to be added to base layer output).
        """
        # x: [batch, seq, in_features]
        lora_output = self.lora_B(self.lora_A(self.dropout(x)))
        return lora_output * self.scaling


class LinearWithLoRA(nn.Module):
    """
    Linear layer with optional LoRA adapter.

    Can be used as drop-in replacement for nn.Linear.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        r: int = 16,
        alpha: int = 16,
        dropout: float = 0.05,
        enable_lora: bool = True,
    ):
        super().__init__()

        # Base linear layer (frozen during LoRA training)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # LoRA adapter
        self.enable_lora = enable_lora
        if enable_lora:
            self.lora = LoRALayer(
                in_features=in_features,
                out_features=out_features,
                r=r,
                alpha=alpha,
                dropout=dropout,
            )
        else:
            self.lora = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with base + LoRA."""
        output = self.linear(x)

        if self.lora is not None and self.enable_lora:
            output = output + self.lora(x)

        return output

    def merge_lora(self) -> None:
        """
        Merge LoRA weights into base weights (for inference).

        After merging, the model behaves as a standard linear layer
        with no LoRA overhead.
        """
        if self.lora is None:
            return

        with torch.no_grad():
            # W' = W + B @ A * scaling
            lora_weight = self.lora.lora_B.weight @ self.lora.lora_A.weight
            self.linear.weight.add_(lora_weight * self.lora.scaling)

        # Disable LoRA after merging
        self.lora = None
        self.enable_lora = False

    def freeze_base(self) -> None:
        """Freeze base weights, keep LoRA trainable."""
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)


def apply_lora_to_layer(
    layer: nn.Module,
    r: int = 16,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: Set[str] = {"q_proj", "k_proj", "v_proj", "out_proj"},
) -> Dict[str, LoRALayer]:
    """
    Apply LoRA adapters to specific modules in a layer.

    Args:
        layer: Transformer layer
        target_modules: Module names to add LoRA to

    Returns:
        Dict of LoRA modules added
    """
    lora_modules = {}

    for name, module in layer.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Create and attach LoRA
                lora = LoRALayer(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                )
                lora_modules[name] = lora

    return lora_modules


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Get all LoRA parameters for optimizer."""
    lora_params = []
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            lora_params.append(param)
    return lora_params


def count_lora_parameters(model: nn.Module) -> int:
    """Count trainable LoRA parameters."""
    return sum(
        p.numel() for n, p in model.named_parameters()
        if "lora" in n.lower() and p.requires_grad
    )
```

**Acceptance Criteria:**

- [ ] LoRA A initialized with Kaiming, B with zeros
- [ ] Scaling factor = alpha / r applied correctly
- [ ] Dropout applied before LoRA projection
- [ ] `merge_lora()` correctly fuses weights
- [ ] `freeze_base()` freezes only base weights
- [ ] Works with Q, K, V, and output projections

**Tests:** `tests/v3/test_layers_v3.py::test_lora_layer`

---

#### Issue 2.2.3: Implement ModernBERTLayerV3

**File:** `src/modeling_studio/models/layers_v3.py`
**Effort:** 4 hours
**Dependencies:** Issues 2.1.3, 2.2.1, 2.2.2

**Description:**
Implement the complete v3 transformer layer combining attention, FFN, and optional LoRA.

**Implementation:**

```python
# src/modeling_studio/models/layers_v3.py

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .attention_v3 import create_attention_layer, get_window_size_for_layer, get_layer_band_name
from .ffn_v3 import create_ffn
from .lora_v3 import LoRALayer, apply_lora_to_layer

class ModernBERTLayerV3(nn.Module):
    """
    Single transformer layer for ModernBERT v3.3 Ultra.

    Components:
        1. Multi-Scale Attention (with global hub tokens)
        2. GELU FFN
        3. Pre-LayerNorm (like GPT-2, not BERT)
        4. Optional LoRA for layers 23-28

    Architecture:
        x → LN → Attention → + → LN → FFN → + → output
            └──────────────────┘   └─────────┘
                (residual)          (residual)
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        layer_idx: int = 1,
        use_flash_attention: bool = True,
        enable_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ):
        super().__init__()

        self.layer_idx = layer_idx
        self.band = get_layer_band_name(layer_idx)
        self.window_size = get_window_size_for_layer(layer_idx)
        self.enable_lora = enable_lora

        # Layer norms (pre-norm architecture)
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        # Attention
        self.attention = create_attention_layer(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_probs_dropout_prob,
            layer_idx=layer_idx,
            use_flash_attention=use_flash_attention,
        )

        # FFN
        self.ffn = create_ffn(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_type="gelu",
        )

        # Dropout for residuals
        self.dropout = nn.Dropout(hidden_dropout_prob)

        # LoRA adapters (only for layers 23-28)
        self.lora_q: Optional[LoRALayer] = None
        self.lora_k: Optional[LoRALayer] = None
        self.lora_v: Optional[LoRALayer] = None
        self.lora_o: Optional[LoRALayer] = None

        if enable_lora and 23 <= layer_idx <= 28:
            self._init_lora(hidden_size, lora_r, lora_alpha, lora_dropout)

    def _init_lora(
        self,
        hidden_size: int,
        r: int,
        alpha: int,
        dropout: float,
    ) -> None:
        """Initialize LoRA adapters for attention projections."""
        self.lora_q = LoRALayer(hidden_size, hidden_size, r, alpha, dropout)
        self.lora_k = LoRALayer(hidden_size, hidden_size, r, alpha, dropout)
        self.lora_v = LoRALayer(hidden_size, hidden_size, r, alpha, dropout)
        self.lora_o = LoRALayer(hidden_size, hidden_size, r, alpha, dropout)
        print(f"  ✓ LoRA initialized for layer {self.layer_idx}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, seq_len] padding mask
            output_attentions: Return attention weights

        Returns:
            (output_hidden_states, attention_weights or None)
        """
        # === Attention Block ===
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)

        # Attention (with optional LoRA)
        attn_output, attn_weights = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

        # Add LoRA contributions if enabled
        if self.lora_q is not None:
            # LoRA is applied to the pre-attention hidden states
            # and added to attention output
            # This is a simplified approach; full LoRA would modify Q/K/V
            lora_contrib = self.lora_o(hidden_states)
            attn_output = attn_output + lora_contrib

        attn_output = self.dropout(attn_output)
        hidden_states = residual + attn_output

        # === FFN Block ===
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        ffn_output = self.ffn(hidden_states)
        hidden_states = residual + ffn_output

        return hidden_states, attn_weights

    def freeze_base_weights(self) -> None:
        """Freeze all weights except LoRA."""
        for name, param in self.named_parameters():
            if "lora" not in name.lower():
                param.requires_grad_(False)

    def merge_lora_weights(self) -> None:
        """Merge LoRA into base weights for inference."""
        # This would require modifying attention projections
        # Implementation depends on how LoRA is integrated with attention
        pass

    def extra_repr(self) -> str:
        lora_status = "enabled" if self.enable_lora and self.lora_q else "disabled"
        return f"layer={self.layer_idx}, band={self.band}, window={self.window_size}, lora={lora_status}"


def create_layer_stack(
    num_layers: int = 28,
    hidden_size: int = 768,
    num_attention_heads: int = 12,
    intermediate_size: int = 3072,
    hidden_dropout_prob: float = 0.1,
    attention_probs_dropout_prob: float = 0.1,
    use_flash_attention: bool = True,
    lora_layers: Optional[list] = None,
    lora_r: int = 16,
    lora_alpha: int = 16,
) -> nn.ModuleList:
    """
    Create the full 28-layer transformer stack.

    Args:
        lora_layers: List of layer indices to apply LoRA (default: 23-28)
    """
    if lora_layers is None:
        lora_layers = [23, 24, 25, 26, 27, 28]

    layers = nn.ModuleList()

    print("\n🏗️ Building v3 layer stack...")
    for i in range(1, num_layers + 1):
        enable_lora = i in lora_layers
        layer = ModernBERTLayerV3(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            layer_idx=i,
            use_flash_attention=use_flash_attention,
            enable_lora=enable_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )
        layers.append(layer)

    print(f"✓ Created {num_layers} layers with LoRA on layers {lora_layers}")
    return layers
```

**Acceptance Criteria:**

- [ ] Pre-LayerNorm architecture (not post-norm)
- [ ] Residual connections around attention and FFN
- [ ] LoRA only applied to layers 23-28
- [ ] Window size correctly set per layer band
- [ ] `freeze_base_weights()` preserves LoRA trainability
- [ ] All 28 layers can be created with correct config

**Tests:** `tests/v3/test_layers_v3.py::test_modernbert_layer_v3`

---

#### Issue 2.2.4: Implement Layer Band Configuration

**File:** `src/modeling_studio/models/layers_v3.py` (extend)
**Effort:** 2 hours
**Dependencies:** Issue 2.2.3

**Description:**
Implement utilities for managing layer bands (Foundation, Context, Semantic, Family) with different configurations.

**Implementation:**

```python
# Add to layers_v3.py

from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import Enum

class LayerBand(Enum):
    """Layer band identifiers."""
    FOUNDATION = "foundation"
    CONTEXT = "context"
    SEMANTIC = "semantic"
    FAMILY = "family"

@dataclass
class BandConfig:
    """Configuration for a layer band."""
    name: str
    layers: Tuple[int, int]  # (start, end) inclusive
    window_size: int
    trainable_phase1: bool
    lora_enabled: bool
    description: str

# Band configurations
BAND_CONFIGS: Dict[LayerBand, BandConfig] = {
    LayerBand.FOUNDATION: BandConfig(
        name="Foundation",
        layers=(1, 6),
        window_size=64,
        trainable_phase1=False,
        lora_enabled=False,
        description="Local token interactions, morphology, subwords"
    ),
    LayerBand.CONTEXT: BandConfig(
        name="Context",
        layers=(7, 18),
        window_size=128,
        trainable_phase1=False,
        lora_enabled=False,
        description="Phrase-level patterns, entities, short phrases"
    ),
    LayerBand.SEMANTIC: BandConfig(
        name="Semantic",
        layers=(19, 22),
        window_size=256,
        trainable_phase1=True,
        lora_enabled=False,
        description="Clause/sentence patterns, syntax, semantics"
    ),
    LayerBand.FAMILY: BandConfig(
        name="Family",
        layers=(23, 28),
        window_size=512,
        trainable_phase1=True,
        lora_enabled=True,
        description="Family-specific representations, hub specialization"
    ),
}


def get_band_for_layer(layer_idx: int) -> LayerBand:
    """Get the band enum for a layer index."""
    for band, config in BAND_CONFIGS.items():
        start, end = config.layers
        if start <= layer_idx <= end:
            return band
    raise ValueError(f"Layer {layer_idx} not in any band")


def get_layers_in_band(band: LayerBand) -> List[int]:
    """Get all layer indices in a band."""
    start, end = BAND_CONFIGS[band].layers
    return list(range(start, end + 1))


def get_trainable_layers(phase: str = "phase1") -> List[int]:
    """Get layers that should be trainable in a given phase."""
    if phase == "phase0.5":
        # Healing phase: L19-28 trainable
        return list(range(19, 29))
    elif phase == "phase1":
        # Full training: L19-28 trainable
        return list(range(19, 29))
    else:
        return []


def get_frozen_layers(phase: str = "phase1") -> List[int]:
    """Get layers that should be frozen in a given phase."""
    if phase in ["phase0.5", "phase1"]:
        # Freeze L1-18
        return list(range(1, 19))
    else:
        return list(range(1, 29))


def freeze_layers_by_band(
    model: nn.Module,
    frozen_bands: List[LayerBand],
) -> None:
    """
    Freeze layers in specified bands.

    Args:
        model: Model with .layers attribute
        frozen_bands: Bands to freeze
    """
    frozen_layers = []
    for band in frozen_bands:
        frozen_layers.extend(get_layers_in_band(band))

    for layer_idx in frozen_layers:
        layer = model.layers[layer_idx - 1]  # 0-indexed
        for param in layer.parameters():
            param.requires_grad_(False)

    print(f"❄️ Frozen bands: {[b.value for b in frozen_bands]}")
    print(f"   Layers: {frozen_layers}")


def unfreeze_layers_by_band(
    model: nn.Module,
    trainable_bands: List[LayerBand],
) -> None:
    """
    Unfreeze layers in specified bands.
    """
    trainable_layers = []
    for band in trainable_bands:
        trainable_layers.extend(get_layers_in_band(band))

    for layer_idx in trainable_layers:
        layer = model.layers[layer_idx - 1]
        for param in layer.parameters():
            param.requires_grad_(True)

    print(f"🔥 Trainable bands: {[b.value for b in trainable_bands]}")
    print(f"   Layers: {trainable_layers}")


def print_band_summary():
    """Print summary of all band configurations."""
    print("\n📊 Layer Band Configuration:")
    print("=" * 70)
    for band, config in BAND_CONFIGS.items():
        start, end = config.layers
        trainable = "🔥 Trainable" if config.trainable_phase1 else "❄️ Frozen"
        lora = "+ LoRA" if config.lora_enabled else ""
        print(f"  {config.name:12} (L{start:2}-{end:2}): "
              f"window={config.window_size:3}, {trainable} {lora}")
        print(f"    └── {config.description}")
    print("=" * 70)
```

**Acceptance Criteria:**

- [ ] All 4 bands correctly defined with layer ranges
- [ ] `get_band_for_layer()` returns correct band
- [ ] `get_trainable_layers()` returns correct layers per phase
- [ ] `freeze_layers_by_band()` freezes all parameters in band
- [ ] Foundation + Context frozen in Phase 1
- [ ] Semantic + Family trainable in Phase 1
- [ ] Only Family band has LoRA

**Tests:** `tests/v3/test_layers_v3.py::test_layer_band_config`

---

## 🏁 Milestone 3: v3 Model Assembly

**Goal:** Assemble complete ModernBERTv3Ultra model with all components
**Estimated Effort:** 5 days
**Dependencies:** Milestone 2 complete (attention_v3.py, layers_v3.py, lora_v3.py)

### Epic 3.1: Model Integration

#### Issue 3.1.1: Implement v3 Embeddings Module

**File:** `src/modeling_studio/models/embeddings_v3.py`
**Effort:** 4 hours
**Dependencies:** Issue 1.2.2 (HubTokenizer)

**Description:**
Implement the v3 embeddings module that handles word embeddings, position embeddings, and the 4 hub token slots. The embedding layer must be compatible with v2 weight transfer while adding space for hub tokens.

**Implementation:**

```python
# src/modeling_studio/models/embeddings_v3.py

import torch
import torch.nn as nn
from typing import Optional, Dict

from .hub_tokens import HUB_TOKEN_REGISTRY, get_hub_positions

class ModernBERTEmbeddingsV3(nn.Module):
    """
    Embeddings module for ModernBERT v3.3 Ultra.

    Token layout:
        [CLS] [EMO] [MEM] [REL] [TASK] <text tokens> [SEP] [PAD]...
        pos 0   1     2     3     4        5+

    Components:
        1. Word embeddings (v2 vocab + 4 hub tokens)
        2. Position embeddings (RoPE-style or learned)
        3. LayerNorm
        4. Dropout

    The hub token embeddings (positions 1-4 in vocab) are initialized
    via semantic centroid initialization from v2 embeddings.
    """

    def __init__(
        self,
        vocab_size: int = 50268,  # v2 vocab (50264) + 4 hub tokens
        hidden_size: int = 768,
        max_position_embeddings: int = 8192,
        hidden_dropout_prob: float = 0.1,
        pad_token_id: int = 0,
        use_rotary_embeddings: bool = True,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id
        self.use_rotary_embeddings = use_rotary_embeddings

        # Word embeddings
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_token_id
        )

        # Position embeddings (if not using RoPE)
        if not use_rotary_embeddings:
            self.position_embeddings = nn.Embedding(
                max_position_embeddings, hidden_size
            )
        else:
            self.position_embeddings = None
            # RoPE is applied in attention layers, not here

        # Token type embeddings (optional, not used in ModernBERT)
        self.token_type_embeddings = None

        # LayerNorm and Dropout
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        # Hub token position indices
        self.hub_positions = get_hub_positions()
        self.num_hub_tokens = len(HUB_TOKEN_REGISTRY)
        self.text_start_position = 5  # After [CLS] + 4 hubs

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for embeddings.

        Args:
            input_ids: [batch, seq_len] token IDs
            position_ids: [batch, seq_len] position IDs (optional)
            token_type_ids: [batch, seq_len] type IDs (unused)
            inputs_embeds: [batch, seq_len, hidden] pre-computed embeddings

        Returns:
            Embeddings [batch, seq_len, hidden_size]
        """
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        batch_size, seq_len = inputs_embeds.shape[:2]

        # Add position embeddings if not using RoPE
        if self.position_embeddings is not None:
            if position_ids is None:
                position_ids = torch.arange(
                    seq_len, dtype=torch.long, device=inputs_embeds.device
                )
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

            position_embeds = self.position_embeddings(position_ids)
            embeddings = inputs_embeds + position_embeds
        else:
            embeddings = inputs_embeds

        # LayerNorm and Dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

    def get_hub_token_embeddings(self) -> Dict[str, torch.Tensor]:
        """
        Extract hub token embeddings for inspection.

        Returns:
            Dict mapping hub token names to their embedding vectors
        """
        hub_embeds = {}
        for token_name, position in self.hub_positions.items():
            if token_name == "[CLS]":
                continue  # Skip CLS
            # Hub tokens are at positions 1-4 after special tokens
            # The actual vocab index depends on tokenizer setup
            hub_embeds[token_name] = self.word_embeddings.weight[position].detach()
        return hub_embeds

    def resize_token_embeddings(self, new_vocab_size: int) -> None:
        """
        Resize embedding matrix to accommodate new vocabulary size.

        Used when adding hub tokens to v2 vocabulary.
        """
        old_vocab_size = self.word_embeddings.num_embeddings
        if new_vocab_size == old_vocab_size:
            return

        # Create new embedding matrix
        new_embeddings = nn.Embedding(
            new_vocab_size,
            self.hidden_size,
            padding_idx=self.pad_token_id,
        )

        # Copy old embeddings
        num_to_copy = min(old_vocab_size, new_vocab_size)
        new_embeddings.weight.data[:num_to_copy] = \
            self.word_embeddings.weight.data[:num_to_copy]

        # Initialize new embeddings (hub tokens) with small random values
        if new_vocab_size > old_vocab_size:
            nn.init.normal_(
                new_embeddings.weight.data[old_vocab_size:],
                mean=0.0,
                std=0.02,
            )

        self.word_embeddings = new_embeddings
        self.vocab_size = new_vocab_size
        print(f"✓ Resized embeddings: {old_vocab_size} → {new_vocab_size}")

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.vocab_size}, hidden_size={self.hidden_size}, "
            f"max_position={self.max_position_embeddings}, "
            f"rotary={'yes' if self.use_rotary_embeddings else 'no'}"
        )
```

**Acceptance Criteria:**

- [ ] Word embeddings sized for v2 vocab + 4 hub tokens
- [ ] Position embeddings support up to 8192 tokens
- [ ] Hub token positions (1-4) accessible via `get_hub_token_embeddings()`
- [ ] `resize_token_embeddings()` works for adding hub tokens
- [ ] RoPE mode skips position embedding addition (applied in attention)
- [ ] LayerNorm and Dropout applied correctly

**Tests:** `tests/v3/test_modernbert_v3.py::test_embeddings_v3`

---

#### Issue 3.1.2: Implement v3 Encoder Stack (28 layers)

**File:** `src/modeling_studio/models/encoder_v3.py`
**Effort:** 4 hours
**Dependencies:** Issue 2.2.3 (ModernBERTLayerV3)

**Description:**
Implement the encoder stack that chains 28 transformer layers with proper gradient checkpointing support for memory efficiency during 8k sequence training.

**Implementation:**

```python
# src/modeling_studio/models/encoder_v3.py

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, List, Dict

from .layers_v3 import ModernBERTLayerV3, create_layer_stack, get_layer_band_name
from .attention_v3 import GLOBAL_TOKEN_POSITIONS

class ModernBERTEncoderV3(nn.Module):
    """
    28-layer encoder stack for ModernBERT v3.3 Ultra.

    Layer Structure:
        - Layers 1-6:   Foundation Band (window=64, frozen)
        - Layers 7-18:  Context Band (window=128, frozen)
        - Layers 19-22: Semantic Band (window=256, trainable)
        - Layers 23-28: Family Band (window=512, trainable + LoRA)

    Features:
        - Gradient checkpointing for memory efficiency
        - Per-layer hidden state output (optional)
        - Hub token preservation through all layers
    """

    def __init__(
        self,
        num_layers: int = 28,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        use_flash_attention: bool = False,  # Default to safe mode
        gradient_checkpointing: bool = False,
        lora_layers: Optional[List[int]] = None,
        lora_r: int = 16,
        lora_alpha: int = 16,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gradient_checkpointing = gradient_checkpointing

        # Create layer stack
        if lora_layers is None:
            lora_layers = [23, 24, 25, 26, 27, 28]  # Family Band

        self.layers = create_layer_stack(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_flash_attention=use_flash_attention,
            lora_layers=lora_layers,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )

        # Track layer bands for debugging
        self.layer_bands = {
            i: get_layer_band_name(i) for i in range(1, num_layers + 1)
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
        """
        Forward pass through all 28 layers.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, seq_len] padding mask
            output_hidden_states: Return all layer hidden states
            output_attentions: Return all attention weights

        Returns:
            Tuple of:
                - Final hidden states [batch, seq_len, hidden_size]
                - All hidden states (if output_hidden_states=True)
                - All attention weights (if output_attentions=True)
        """
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None

        for i, layer in enumerate(self.layers):
            layer_idx = i + 1  # 1-indexed

            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            # Gradient checkpointing for memory efficiency
            if self.gradient_checkpointing and self.training:
                hidden_states, attn_weights = self._checkpoint_forward(
                    layer, hidden_states, attention_mask, output_attentions
                )
            else:
                hidden_states, attn_weights = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                )

            if output_attentions and attn_weights is not None:
                all_attentions.append(attn_weights)

        # Add final hidden states
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return hidden_states, all_hidden_states, all_attentions

    def _checkpoint_forward(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        output_attentions: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward with gradient checkpointing.

        Note: Gradient checkpointing doesn't work well with output_attentions
        """
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(inputs[0], attention_mask=inputs[1], output_attentions=False)
            return custom_forward

        hidden_states, _ = checkpoint(
            create_custom_forward(layer),
            hidden_states,
            attention_mask,
            use_reentrant=False,
        )
        return hidden_states, None

    def freeze_layers(self, layer_indices: List[int]) -> None:
        """
        Freeze specific layers.

        Args:
            layer_indices: 1-indexed layer numbers to freeze
        """
        for idx in layer_indices:
            if 1 <= idx <= self.num_layers:
                layer = self.layers[idx - 1]  # 0-indexed
                for param in layer.parameters():
                    param.requires_grad_(False)
        print(f"❄️ Froze layers: {layer_indices}")

    def unfreeze_layers(self, layer_indices: List[int]) -> None:
        """
        Unfreeze specific layers.

        Args:
            layer_indices: 1-indexed layer numbers to unfreeze
        """
        for idx in layer_indices:
            if 1 <= idx <= self.num_layers:
                layer = self.layers[idx - 1]
                for param in layer.parameters():
                    param.requires_grad_(True)
        print(f"🔥 Unfroze layers: {layer_indices}")

    def get_layer_by_band(self, band: str) -> List[nn.Module]:
        """Get all layers in a specific band."""
        layers = []
        for i, layer in enumerate(self.layers):
            if self.layer_bands[i + 1] == band:
                layers.append(layer)
        return layers

    def print_layer_summary(self) -> None:
        """Print summary of encoder layers."""
        print("\n📊 Encoder Layer Summary:")
        print("=" * 60)

        current_band = None
        for i, layer in enumerate(self.layers):
            layer_idx = i + 1
            band = self.layer_bands[layer_idx]

            if band != current_band:
                print(f"\n  [{band.upper()} BAND]")
                current_band = band

            trainable = sum(p.requires_grad for p in layer.parameters())
            total = sum(1 for _ in layer.parameters())
            lora_status = "🔧 LoRA" if layer.enable_lora else ""
            print(f"    Layer {layer_idx:2d}: window={layer.window_size:3d}, "
                  f"trainable={trainable}/{total} {lora_status}")

        print("=" * 60)

    def extra_repr(self) -> str:
        return f"num_layers={self.num_layers}, checkpointing={self.gradient_checkpointing}"
```

**Acceptance Criteria:**

- [ ] 28 layers created with correct band configurations
- [ ] Gradient checkpointing reduces memory for 8k sequences
- [ ] `freeze_layers()` and `unfreeze_layers()` work correctly
- [ ] All hidden states returned when `output_hidden_states=True`
- [ ] Attention weights returned when `output_attentions=True`
- [ ] Layer band lookup works correctly

**Tests:** `tests/v3/test_modernbert_v3.py::test_encoder_v3`

---

#### Issue 3.1.3: Implement v3 Pair Encoder with [REL] Hub

**File:** `src/modeling_studio/models/pair_encoder_v3.py`
**Effort:** 5 hours
**Dependencies:** Issues 3.1.1, 3.1.2, Issue 1.2.1 (hub tokens)

**Description:**
Implement the pair encoder for sentence-pair tasks (NLI, relation extraction) that leverages the `[REL]` hub token for relationship representation. The pair encoder takes two texts and produces a unified representation for classification.

**Implementation:**

```python
# src/modeling_studio/models/pair_encoder_v3.py

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, Union

from .hub_tokens import get_hub_positions, HubToken

class PairEncoderV3(nn.Module):
    """
    Pair Encoder for sentence-pair tasks in v3.

    Token Layout for Pairs:
        [CLS] [EMO] [MEM] [REL] [TASK] <text_a> [SEP] <text_b> [SEP] [PAD]...

    Key Innovation: The [REL] hub token (position 3) captures the
    relationship between text_a and text_b through cross-attention
    across the full sequence.

    Use Cases:
        - NLI (Natural Language Inference)
        - Relation extraction
        - Semantic similarity
        - Paraphrase detection
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_labels: int = 3,  # NLI: entailment, neutral, contradiction
        classifier_dropout: float = 0.1,
        use_rel_hub: bool = True,
        pooling_strategy: str = "rel_hub",  # "rel_hub", "cls", "mean", "concat"
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.use_rel_hub = use_rel_hub
        self.pooling_strategy = pooling_strategy

        # Get hub positions
        self.hub_positions = get_hub_positions()
        self.rel_position = self.hub_positions["[REL]"]  # Position 3
        self.cls_position = self.hub_positions["[CLS]"]  # Position 0

        # Determine classifier input size based on strategy
        if pooling_strategy == "concat":
            classifier_input_size = hidden_size * 3  # CLS + REL + mean_diff
        elif pooling_strategy == "rel_hub":
            classifier_input_size = hidden_size
        else:
            classifier_input_size = hidden_size

        # Classifier head
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(classifier_input_size, num_labels)

        # Optional: Cross-attention refinement layer
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=classifier_dropout,
            batch_first=True,
        )
        self.use_cross_attention = False  # Can be enabled for enhanced fusion

    def forward(
        self,
        encoder_output: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        text_a_mask: Optional[torch.Tensor] = None,
        text_b_mask: Optional[torch.Tensor] = None,
        return_pooled: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for pair classification.

        Args:
            encoder_output: [batch, seq_len, hidden_size] from encoder
            attention_mask: [batch, seq_len] padding mask
            text_a_mask: [batch, seq_len] mask for first sentence (optional)
            text_b_mask: [batch, seq_len] mask for second sentence (optional)
            return_pooled: Also return the pooled representation

        Returns:
            Classification logits [batch, num_labels]
            Optionally: (logits, pooled_representation)
        """
        batch_size = encoder_output.size(0)

        # Extract pooled representation based on strategy
        if self.pooling_strategy == "rel_hub":
            # Use [REL] hub token - designed for relationship representation
            pooled = encoder_output[:, self.rel_position, :]  # [batch, hidden]

        elif self.pooling_strategy == "cls":
            # Traditional CLS pooling
            pooled = encoder_output[:, self.cls_position, :]

        elif self.pooling_strategy == "mean":
            # Mean pooling over non-special tokens
            if attention_mask is not None:
                # Mask out positions 0-4 (CLS + hubs)
                text_mask = attention_mask.clone()
                text_mask[:, :5] = 0  # Exclude special tokens
                mask_expanded = text_mask.unsqueeze(-1).float()
                sum_hidden = (encoder_output * mask_expanded).sum(dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                pooled = sum_hidden / sum_mask
            else:
                pooled = encoder_output[:, 5:, :].mean(dim=1)

        elif self.pooling_strategy == "concat":
            # Concatenate CLS + REL + mean difference
            cls_repr = encoder_output[:, self.cls_position, :]
            rel_repr = encoder_output[:, self.rel_position, :]

            # Compute mean representations for text_a and text_b if masks provided
            if text_a_mask is not None and text_b_mask is not None:
                a_mask = text_a_mask.unsqueeze(-1).float()
                b_mask = text_b_mask.unsqueeze(-1).float()
                mean_a = (encoder_output * a_mask).sum(1) / a_mask.sum(1).clamp(min=1e-9)
                mean_b = (encoder_output * b_mask).sum(1) / b_mask.sum(1).clamp(min=1e-9)
                mean_diff = torch.abs(mean_a - mean_b)
            else:
                mean_diff = rel_repr  # Fallback

            pooled = torch.cat([cls_repr, rel_repr, mean_diff], dim=-1)

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        # Optional cross-attention refinement
        if self.use_cross_attention and text_a_mask is not None and text_b_mask is not None:
            # Get text_a and text_b representations
            # This is a placeholder for more sophisticated fusion
            pass

        # Classification
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        if return_pooled:
            return logits, pooled
        return logits

    def get_rel_hub_representation(
        self,
        encoder_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract the [REL] hub token representation.

        This is the primary representation for relationship tasks.

        Args:
            encoder_output: [batch, seq_len, hidden_size]

        Returns:
            [REL] representation [batch, hidden_size]
        """
        return encoder_output[:, self.rel_position, :]

    def set_pooling_strategy(self, strategy: str) -> None:
        """
        Change pooling strategy at runtime.

        Strategies:
            - "rel_hub": Use [REL] hub token (default, recommended)
            - "cls": Traditional CLS token
            - "mean": Mean pooling over text tokens
            - "concat": Concatenate CLS + REL + mean_diff
        """
        valid_strategies = ["rel_hub", "cls", "mean", "concat"]
        if strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}")
        self.pooling_strategy = strategy
        print(f"✓ Pair encoder pooling strategy set to: {strategy}")

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, num_labels={self.num_labels}, "
            f"pooling={self.pooling_strategy}"
        )


class SiamesePairEncoderV3(nn.Module):
    """
    Siamese-style pair encoder for semantic similarity.

    Uses the [MEM] hub token for embedding representation
    and [REL] hub for explicit relationship modeling.

    Good for:
        - Semantic textual similarity (STS)
        - Duplicate detection
        - Embedding-based retrieval ranking
    """

    def __init__(
        self,
        hidden_size: int = 768,
        similarity_function: str = "cosine",  # "cosine", "euclidean", "learned"
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.similarity_function = similarity_function
        self.hub_positions = get_hub_positions()

        # For learned similarity
        if similarity_function == "learned":
            self.similarity_layer = nn.Sequential(
                nn.Linear(hidden_size * 4, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, 1),
            )

    def forward(
        self,
        encoder_output_a: torch.Tensor,
        encoder_output_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute similarity between two encoded sequences.

        Args:
            encoder_output_a: [batch, seq_len, hidden] for text A
            encoder_output_b: [batch, seq_len, hidden] for text B

        Returns:
            Similarity scores [batch] or [batch, 1]
        """
        # Use [MEM] hub for embedding representation
        mem_position = self.hub_positions["[MEM]"]
        embed_a = encoder_output_a[:, mem_position, :]  # [batch, hidden]
        embed_b = encoder_output_b[:, mem_position, :]  # [batch, hidden]

        if self.similarity_function == "cosine":
            # Cosine similarity
            sim = nn.functional.cosine_similarity(embed_a, embed_b, dim=-1)

        elif self.similarity_function == "euclidean":
            # Negative euclidean distance (higher = more similar)
            dist = torch.norm(embed_a - embed_b, p=2, dim=-1)
            sim = -dist

        elif self.similarity_function == "learned":
            # Learned similarity with element-wise operations
            concat = torch.cat([
                embed_a,
                embed_b,
                embed_a * embed_b,  # Element-wise product
                torch.abs(embed_a - embed_b),  # Absolute difference
            ], dim=-1)
            sim = self.similarity_layer(concat).squeeze(-1)

        else:
            raise ValueError(f"Unknown similarity function: {self.similarity_function}")

        return sim
```

**Acceptance Criteria:**

- [ ] `[REL]` hub token (position 3) used as primary pair representation
- [ ] Multiple pooling strategies supported (rel_hub, cls, mean, concat)
- [ ] NLI classification (3 labels) works correctly
- [ ] `SiamesePairEncoderV3` supports cosine, euclidean, and learned similarity
- [ ] `[MEM]` hub used for embedding similarity
- [ ] Text A/B masks correctly applied for mean pooling

**Tests:** `tests/v3/test_modernbert_v3.py::test_pair_encoder_v3`

---

#### Issue 3.1.4: Implement ModernBERTv3Ultra Main Class

**File:** `src/modeling_studio/models/modernbert_v3.py`
**Effort:** 6 hours
**Dependencies:** Issues 3.1.1, 3.1.2, 3.1.3

**Description:**
Implement the main model class that combines embeddings, encoder, poolers, and provides the unified interface for all downstream tasks. This is the primary entry point for v3.

**Implementation:**

```python
# src/modeling_studio/models/modernbert_v3.py

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass

from .config_v3 import ModernBERTv3Config
from .embeddings_v3 import ModernBERTEmbeddingsV3
from .encoder_v3 import ModernBERTEncoderV3
from .poolers_v3 import HubTokenPooler, CombinedPooler
from .hub_tokens import (
    HUB_TOKEN_REGISTRY,
    get_hub_positions,
    get_hub_for_capability,
    TOKEN_LEVEL_CAPABILITIES,
)
from .pair_encoder_v3 import PairEncoderV3


@dataclass
class ModernBERTv3Output:
    """
    Output container for ModernBERT v3 forward pass.

    Attributes:
        last_hidden_state: Final layer output [batch, seq, hidden]
        pooled_outputs: Dict of hub token representations
        hidden_states: All layer outputs (if output_hidden_states=True)
        attentions: All attention weights (if output_attentions=True)
    """
    last_hidden_state: torch.Tensor
    pooled_outputs: Dict[str, torch.Tensor]
    hidden_states: Optional[List[torch.Tensor]] = None
    attentions: Optional[List[torch.Tensor]] = None


class ModernBERTv3Ultra(nn.Module):
    """
    ModernBERT v3.3 Ultra - Unified FamilyOS Encoder.

    Architecture:
        - 28 transformer layers (vs 22 in v2)
        - 4 hub tokens: [EMO], [MEM], [REL], [TASK]
        - Multi-scale sliding window attention (64→128→256→512)
        - Global attention for hub tokens (positions 0-4)
        - LoRA adapters on Family Band (L23-28)

    Token Layout:
        [CLS] [EMO] [MEM] [REL] [TASK] <text> [SEP] [PAD]...
        pos 0   1     2     3     4     5+

    Capabilities (12 total):
        Hub-routed (9): emotions, sentiment, safety_*, embedding, nli, relation, intent, ingress
        Token-level (3): ner_general, ner_family, temporal
    """

    def __init__(self, config: ModernBERTv3Config):
        super().__init__()
        self.config = config

        # Embeddings
        self.embeddings = ModernBERTEmbeddingsV3(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            max_position_embeddings=config.max_position_embeddings,
            hidden_dropout_prob=config.hidden_dropout_prob,
            pad_token_id=config.pad_token_id,
            use_rotary_embeddings=config.use_rotary_embeddings,
        )

        # Encoder (28 layers)
        self.encoder = ModernBERTEncoderV3(
            num_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            use_flash_attention=config.use_flash_attention,
            gradient_checkpointing=config.gradient_checkpointing,
            lora_layers=config.lora_layers,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
        )

        # Poolers
        self.hub_pooler = HubTokenPooler(
            hidden_size=config.hidden_size,
            add_projection=False,
        )
        self.combined_pooler = CombinedPooler(hidden_size=config.hidden_size)

        # Pair encoder for NLI/relation tasks
        self.pair_encoder = PairEncoderV3(
            hidden_size=config.hidden_size,
            num_labels=3,  # Will be reconfigured per task
            pooling_strategy="rel_hub",
        )

        # Final LayerNorm (optional, some models use this)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

        # Hub positions cache
        self.hub_positions = get_hub_positions()
        self.num_hub_tokens = len(HUB_TOKEN_REGISTRY)

        # Initialize weights
        self.apply(self._init_weights)

        print(f"\n✓ ModernBERTv3Ultra initialized:")
        print(f"  - Layers: {config.num_hidden_layers}")
        print(f"  - Hidden: {config.hidden_size}")
        print(f"  - Heads: {config.num_attention_heads}")
        print(f"  - Hub tokens: {list(HUB_TOKEN_REGISTRY.keys())}")
        print(f"  - LoRA layers: {config.lora_layers}")

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
    ) -> Union[ModernBERTv3Output, Tuple]:
        """
        Forward pass for ModernBERT v3.

        Args:
            input_ids: [batch, seq_len] token IDs
            attention_mask: [batch, seq_len] padding mask (1=valid, 0=pad)
            token_type_ids: [batch, seq_len] type IDs (unused)
            position_ids: [batch, seq_len] position IDs (optional)
            output_hidden_states: Return all layer hidden states
            output_attentions: Return all attention weights
            return_dict: Return ModernBERTv3Output or tuple

        Returns:
            ModernBERTv3Output or tuple of tensors
        """
        # Embeddings
        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )

        # Encoder
        encoder_output, all_hidden_states, all_attentions = self.encoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        # Final LayerNorm
        last_hidden_state = self.final_layer_norm(encoder_output)

        # Pool hub token representations
        pooled_outputs = self.hub_pooler(last_hidden_state, attention_mask)

        if return_dict:
            return ModernBERTv3Output(
                last_hidden_state=last_hidden_state,
                pooled_outputs=pooled_outputs,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
            )
        else:
            return (last_hidden_state, pooled_outputs, all_hidden_states, all_attentions)

    def get_representation_for_capability(
        self,
        last_hidden_state: torch.Tensor,
        pooled_outputs: Dict[str, torch.Tensor],
        capability: str,
    ) -> torch.Tensor:
        """
        Get the appropriate representation for a capability.

        Hub-routed capabilities get the hub token representation.
        Token-level capabilities get the full sequence.

        Args:
            last_hidden_state: [batch, seq, hidden]
            pooled_outputs: Dict of hub representations
            capability: Capability name

        Returns:
            Representation tensor
        """
        if capability in TOKEN_LEVEL_CAPABILITIES:
            # Token-level tasks (NER, temporal) need full sequence
            return last_hidden_state
        else:
            # Hub-routed tasks
            hub_token = get_hub_for_capability(capability)
            return pooled_outputs[hub_token]

    def get_embedding_representation(
        self,
        last_hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get embedding for retrieval/similarity tasks.

        Uses the [MEM] hub token at position 2.

        Args:
            last_hidden_state: [batch, seq, hidden]

        Returns:
            Embedding [batch, hidden]
        """
        mem_position = self.hub_positions["[MEM]"]
        return last_hidden_state[:, mem_position, :]

    def freeze_for_phase(self, phase: str) -> None:
        """
        Configure model freezing for a training phase.

        Args:
            phase: "phase0.5" (healing) or "phase1" (full training)
        """
        if phase in ["phase0.5", "phase1"]:
            # Freeze embeddings (except hub tokens - handled separately)
            for param in self.embeddings.parameters():
                param.requires_grad_(False)

            # Freeze Foundation + Context bands (L1-18)
            self.encoder.freeze_layers(list(range(1, 19)))

            # Unfreeze Semantic + Family bands (L19-28)
            self.encoder.unfreeze_layers(list(range(19, 29)))

            print(f"✓ Model configured for {phase}:")
            print(f"  ❄️ Frozen: Embeddings, L1-18")
            print(f"  🔥 Trainable: L19-28")

    def merge_lora_weights(self) -> None:
        """
        Merge LoRA weights into base weights for inference.

        Call this before exporting the model.
        """
        for layer in self.encoder.layers:
            if hasattr(layer, 'merge_lora_weights'):
                layer.merge_lora_weights()
        print("✓ LoRA weights merged into base model")

    def get_input_embeddings(self) -> nn.Embedding:
        """Get word embeddings."""
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        """Set word embeddings."""
        self.embeddings.word_embeddings = new_embeddings

    def resize_token_embeddings(self, new_vocab_size: int) -> None:
        """Resize embeddings for new vocabulary (hub tokens)."""
        self.embeddings.resize_token_embeddings(new_vocab_size)
        self.config.vocab_size = new_vocab_size

    @property
    def num_parameters(self) -> int:
        """Total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_model_summary(self) -> None:
        """Print summary of model architecture."""
        print("\n" + "=" * 70)
        print("📊 ModernBERT v3.3 Ultra - Model Summary")
        print("=" * 70)
        print(f"  Total parameters:     {self.num_parameters:,}")
        print(f"  Trainable parameters: {self.num_trainable_parameters:,}")
        print(f"  Layers: {self.config.num_hidden_layers}")
        print(f"  Hidden size: {self.config.hidden_size}")
        print(f"  Attention heads: {self.config.num_attention_heads}")
        print(f"  Hub tokens: {list(self.hub_positions.keys())}")
        print("=" * 70)
        self.encoder.print_layer_summary()


def create_modernbert_v3_ultra(
    from_v2_checkpoint: Optional[str] = None,
    **config_overrides,
) -> ModernBERTv3Ultra:
    """
    Factory function to create ModernBERT v3 Ultra.

    Args:
        from_v2_checkpoint: Path to v2 checkpoint for initialization
        **config_overrides: Override default config values

    Returns:
        Initialized ModernBERTv3Ultra model
    """
    # Create config with defaults
    config = ModernBERTv3Config(**config_overrides)

    # Create model
    model = ModernBERTv3Ultra(config)

    # Initialize from v2 if provided
    if from_v2_checkpoint:
        from .initialization_v3 import initialize_from_v2
        initialize_from_v2(model, from_v2_checkpoint)

    return model
```

**Acceptance Criteria:**

- [ ] Combines embeddings, encoder, and poolers correctly
- [ ] `ModernBERTv3Output` contains all required fields
- [ ] Hub token representations extracted via `get_representation_for_capability()`
- [ ] `freeze_for_phase()` correctly configures L1-18 frozen, L19-28 trainable
- [ ] `merge_lora_weights()` works for inference export
- [ ] `print_model_summary()` shows complete architecture info
- [ ] Factory function supports v2 checkpoint initialization

**Tests:** `tests/v3/test_modernbert_v3.py::test_modernbert_v3_ultra`

---

#### Issue 3.1.5: Implement v3 Forward Pass with Hub Routing

**File:** `src/modeling_studio/models/modernbert_v3.py` (extend)
**Effort:** 5 hours
**Dependencies:** Issues 3.1.4, 1.2.5 (HubRouter)

**Description:**
Implement the multi-task forward pass that routes hub token representations to the appropriate capability heads based on the active tasks in a batch. This enables true multi-task learning with hub-specialized representations.

**Implementation:**

```python
# Add to modernbert_v3.py

from typing import Dict, List, Any, Optional
from .routing_v3 import HubRouter, create_hub_routing_info
from .hub_tokens import TOKEN_LEVEL_CAPABILITIES, get_hub_for_capability


class ModernBERTv3ForMultiTask(ModernBERTv3Ultra):
    """
    ModernBERT v3 with multi-task heads and hub routing.

    Extends the base model with:
        - Task-specific classification/regression heads
        - Hub token routing to appropriate heads
        - Multi-task loss computation
        - Gradient masking for hub specialization
    """

    def __init__(self, config: ModernBERTv3Config, task_heads: Dict[str, nn.Module] = None):
        super().__init__(config)

        # Hub router
        self.hub_router = HubRouter()

        # Task heads registry
        self.task_heads = nn.ModuleDict()
        if task_heads:
            for task_name, head in task_heads.items():
                self.register_task_head(task_name, head)

        # Loss weights per task (can be adjusted during training)
        self.task_loss_weights: Dict[str, float] = {}

        # Active capabilities for current batch
        self._active_capabilities: List[str] = []

    def register_task_head(
        self,
        task_name: str,
        head: nn.Module,
        loss_weight: float = 1.0,
    ) -> None:
        """
        Register a task head.

        Args:
            task_name: Capability name (e.g., "emotions", "ner_general")
            head: Classification/regression head module
            loss_weight: Weight for this task's loss (default: 1.0)
        """
        self.task_heads[task_name] = head
        self.task_loss_weights[task_name] = loss_weight

        routing_info = create_hub_routing_info(task_name)
        print(f"  ✓ Registered head: {task_name} → {routing_info['hub_token']} "
              f"({routing_info['pool_type']})")

    def forward_for_task(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        task: str = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for a single task.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            task: Task/capability name
            labels: Ground truth labels (optional, for loss computation)

        Returns:
            Dict with 'logits', optionally 'loss', 'hidden_states'
        """
        if task not in self.task_heads:
            raise ValueError(f"Unknown task: {task}. Registered: {list(self.task_heads.keys())}")

        # Get encoder output
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Route to appropriate representation
        representation = self.get_representation_for_capability(
            last_hidden_state=outputs.last_hidden_state,
            pooled_outputs=outputs.pooled_outputs,
            capability=task,
        )

        # Get task head and compute logits
        head = self.task_heads[task]
        logits = head(representation)

        result = {"logits": logits}

        # Compute loss if labels provided
        if labels is not None:
            loss = self._compute_task_loss(task, logits, labels, attention_mask)
            result["loss"] = loss

        return result

    def forward_multitask(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        task_labels: Dict[str, torch.Tensor] = None,
        active_tasks: List[str] = None,
        return_all_logits: bool = True,
    ) -> Dict[str, Any]:
        """
        Multi-task forward pass with hub routing.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            task_labels: Dict mapping task names to label tensors
            active_tasks: List of tasks to compute (default: all registered)
            return_all_logits: Return logits for all tasks

        Returns:
            Dict with:
                - 'total_loss': Weighted sum of all task losses
                - 'task_losses': Dict of individual task losses
                - 'task_logits': Dict of task logits (if return_all_logits)
                - 'hub_representations': Dict of hub token vectors
        """
        if active_tasks is None:
            active_tasks = list(self.task_heads.keys())

        self._active_capabilities = active_tasks

        # Get encoder output (single forward pass)
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        task_losses = {}
        task_logits = {}
        total_loss = torch.tensor(0.0, device=input_ids.device)

        # Process each active task
        for task in active_tasks:
            if task not in self.task_heads:
                continue

            # Get appropriate representation via hub routing
            representation = self.get_representation_for_capability(
                last_hidden_state=outputs.last_hidden_state,
                pooled_outputs=outputs.pooled_outputs,
                capability=task,
            )

            # Compute logits
            head = self.task_heads[task]
            logits = head(representation)

            if return_all_logits:
                task_logits[task] = logits

            # Compute loss if labels provided for this task
            if task_labels and task in task_labels:
                labels = task_labels[task]
                loss = self._compute_task_loss(task, logits, labels, attention_mask)
                task_losses[task] = loss

                # Add weighted loss to total
                weight = self.task_loss_weights.get(task, 1.0)
                total_loss = total_loss + weight * loss

        return {
            "total_loss": total_loss if task_losses else None,
            "task_losses": task_losses,
            "task_logits": task_logits,
            "hub_representations": outputs.pooled_outputs,
            "last_hidden_state": outputs.last_hidden_state,
        }

    def _compute_task_loss(
        self,
        task: str,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute loss for a task.

        Handles different loss types:
            - Classification: CrossEntropyLoss
            - Token-level: CrossEntropyLoss with mask
            - Regression: MSELoss
        """
        if task in TOKEN_LEVEL_CAPABILITIES:
            # Token-level classification (NER, temporal)
            # logits: [batch, seq, num_labels]
            # labels: [batch, seq]
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
        elif task == "stsb" or task == "similarity":
            # Regression
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.squeeze(-1), labels.float())
        else:
            # Sequence classification
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return loss

    def get_hub_gradient_mask(
        self,
        device: torch.device,
        batch_size: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Get gradient masks for hub tokens based on active capabilities.

        Used to ensure gradients only flow through hubs that are
        being used for active tasks.

        Returns:
            Dict mapping hub tokens to masks [batch]
        """
        return self.hub_router.get_hub_gradient_mask(
            active_capabilities=self._active_capabilities,
            batch_size=batch_size,
            device=device,
        )

    def set_task_loss_weight(self, task: str, weight: float) -> None:
        """Set loss weight for a task."""
        if task not in self.task_heads:
            raise ValueError(f"Unknown task: {task}")
        self.task_loss_weights[task] = weight
        print(f"✓ Loss weight for '{task}' set to {weight}")

    def print_routing_table(self) -> None:
        """Print hub routing configuration."""
        print("\n📊 Hub Routing Table:")
        print("-" * 60)
        print(f"{'Task':<20} {'Pool Type':<12} {'Hub Token':<12}")
        print("-" * 60)
        for task in self.task_heads.keys():
            info = create_hub_routing_info(task)
            hub = info['hub_token'] or "N/A (token-level)"
            print(f"{task:<20} {info['pool_type']:<12} {hub:<12}")
        print("-" * 60)


# Convenience heads for common tasks
class ClassificationHead(nn.Module):
    """Simple classification head for hub-routed tasks."""

    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(pooled_output))


class TokenClassificationHead(nn.Module):
    """Token-level classification head for NER/temporal."""

    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        # sequence_output: [batch, seq, hidden]
        return self.classifier(self.dropout(sequence_output))


class RegressionHead(nn.Module):
    """Regression head for similarity tasks."""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.dropout(pooled_output))


def create_v3_multitask_model(
    config: ModernBERTv3Config,
    task_configs: Dict[str, Dict],
) -> ModernBERTv3ForMultiTask:
    """
    Factory function to create v3 with task heads.

    Args:
        config: Model config
        task_configs: Dict mapping task names to head configs
            Example:
            {
                "emotions": {"type": "classification", "num_labels": 7},
                "ner_general": {"type": "token_classification", "num_labels": 9},
                "embedding": {"type": "none"},  # Uses raw hub output
            }

    Returns:
        Configured multi-task model
    """
    model = ModernBERTv3ForMultiTask(config)

    for task_name, head_config in task_configs.items():
        head_type = head_config.get("type", "classification")

        if head_type == "classification":
            head = ClassificationHead(
                hidden_size=config.hidden_size,
                num_labels=head_config["num_labels"],
                dropout=head_config.get("dropout", 0.1),
            )
        elif head_type == "token_classification":
            head = TokenClassificationHead(
                hidden_size=config.hidden_size,
                num_labels=head_config["num_labels"],
                dropout=head_config.get("dropout", 0.1),
            )
        elif head_type == "regression":
            head = RegressionHead(
                hidden_size=config.hidden_size,
                dropout=head_config.get("dropout", 0.1),
            )
        elif head_type == "none":
            # No head - uses raw hub output (e.g., for embeddings)
            continue
        else:
            raise ValueError(f"Unknown head type: {head_type}")

        model.register_task_head(
            task_name,
            head,
            loss_weight=head_config.get("loss_weight", 1.0),
        )

    return model
```

**Acceptance Criteria:**

- [ ] `forward_for_task()` routes single task to correct hub
- [ ] `forward_multitask()` handles multiple tasks in one forward pass
- [ ] Hub routing uses `[EMO]` for emotions/sentiment, `[REL]` for NLI, etc.
- [ ] Token-level tasks (NER) receive full sequence, not hub pooling
- [ ] Loss computation handles classification, token-level, and regression
- [ ] `get_hub_gradient_mask()` returns masks for active capabilities
- [ ] `print_routing_table()` shows all task→hub mappings
- [ ] Factory function creates model with configured heads

**Tests:** `tests/v3/test_modernbert_v3.py::test_multitask_forward`

---

### Epic 3.2: Head Integration

#### Issue 3.2.1: Wire Hub Tokens to Capability Heads

**File:** `src/modeling_studio/models/heads_v3.py`
**Effort:** 5 hours
**Dependencies:** Issues 3.1.4, 3.1.5, 1.2.5 (HubRouter)

**Description:**
Create hub-aware task heads that automatically receive the correct representation based on their capability's hub routing. This extends the v2 heads with hub token awareness.

**Implementation:**

```python
# src/modeling_studio/models/heads_v3.py

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from .hub_tokens import (
    get_hub_for_capability,
    get_hub_positions,
    TOKEN_LEVEL_CAPABILITIES,
    HUB_TOKEN_REGISTRY,
)


@dataclass
class HeadConfig:
    """Configuration for a task head."""
    name: str
    num_labels: int
    head_type: str  # "classification", "token", "regression", "hierarchical"
    hub_token: str  # Which hub routes to this head
    hidden_size: int = 768
    dropout: float = 0.1
    loss_weight: float = 1.0
    # For hierarchical heads
    hierarchy: Optional[Dict] = None


class HubAwareClassificationHead(nn.Module):
    """
    Classification head that receives input from a specific hub token.

    Used for: emotions, sentiment, safety_*, intent, ingress
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_labels: int = 2,
        dropout: float = 0.1,
        hub_token: str = "[CLS]",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.hub_token = hub_token
        self.hub_position = get_hub_positions()[hub_token]

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooled_outputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: [batch, seq, hidden] - full sequence
            pooled_outputs: Dict of hub representations (preferred)

        Returns:
            Logits [batch, num_labels]
        """
        if pooled_outputs is not None and self.hub_token in pooled_outputs:
            # Use pre-pooled hub representation
            pooled = pooled_outputs[self.hub_token]
        else:
            # Extract from sequence
            pooled = hidden_states[:, self.hub_position, :]

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

    def extra_repr(self) -> str:
        return f"hub={self.hub_token}, labels={self.num_labels}"


class HubAwareTokenClassificationHead(nn.Module):
    """
    Token-level classification head for sequence labeling.

    Used for: ner_general, ner_family, temporal

    Note: Token-level tasks do NOT use hub pooling - they need
    the full sequence output.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_labels: int = 9,  # NER tags
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: [batch, seq, hidden]
            attention_mask: [batch, seq] (for masking predictions)

        Returns:
            Logits [batch, seq, num_labels]
        """
        # Skip hub token positions (0-4) for NER predictions
        # Only predict on actual text tokens
        sequence_output = self.dropout(hidden_states)
        logits = self.classifier(sequence_output)

        return logits

    def get_predictions(
        self,
        logits: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get predicted labels, masking special tokens.

        Returns:
            Predictions [batch, seq] with -100 for special positions
        """
        predictions = logits.argmax(dim=-1)

        # Mask positions 0-4 (CLS + hub tokens)
        predictions[:, :5] = -100

        # Mask padding
        predictions = predictions.masked_fill(attention_mask == 0, -100)

        return predictions


class HubAwareHierarchicalHead(nn.Module):
    """
    Hierarchical classification head for emotions.

    Structure:
        - Primary: ekman (7 classes)
        - Secondary: goemotions (28 classes)

    Uses [EMO] hub token for representation.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        primary_labels: int = 7,    # Ekman emotions
        secondary_labels: int = 28,  # GoEmotions
        dropout: float = 0.1,
        hub_token: str = "[EMO]",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.hub_token = hub_token
        self.hub_position = get_hub_positions()[hub_token]

        self.dropout = nn.Dropout(dropout)

        # Primary classifier (Ekman)
        self.primary_classifier = nn.Linear(hidden_size, primary_labels)

        # Secondary classifier (GoEmotions) - conditioned on primary
        self.secondary_classifier = nn.Sequential(
            nn.Linear(hidden_size + primary_labels, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, secondary_labels),
        )

        # Emotion hierarchy mapping (Ekman -> GoEmotions indices)
        self.hierarchy_mask = self._build_hierarchy_mask(
            primary_labels, secondary_labels
        )

    def _build_hierarchy_mask(
        self,
        primary: int,
        secondary: int,
    ) -> torch.Tensor:
        """Build mask enforcing hierarchy constraints."""
        # This would be populated from labels.py emotion hierarchy
        # For now, return identity (no masking)
        return torch.ones(primary, secondary)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooled_outputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with hierarchical predictions.

        Returns:
            Tuple of (primary_logits, secondary_logits)
        """
        if pooled_outputs is not None and self.hub_token in pooled_outputs:
            pooled = pooled_outputs[self.hub_token]
        else:
            pooled = hidden_states[:, self.hub_position, :]

        pooled = self.dropout(pooled)

        # Primary prediction (Ekman)
        primary_logits = self.primary_classifier(pooled)
        primary_probs = torch.softmax(primary_logits, dim=-1)

        # Secondary prediction conditioned on primary
        secondary_input = torch.cat([pooled, primary_probs], dim=-1)
        secondary_logits = self.secondary_classifier(secondary_input)

        return primary_logits, secondary_logits


class HubAwareSafetyHead(nn.Module):
    """
    Safety classification head with calibrated outputs.

    Uses [EMO] hub token (safety correlates with emotional content).

    Features:
        - Binary classification (safe/unsafe)
        - Confidence calibration
        - Threshold-based prediction
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_labels: int = 2,  # Safe / Unsafe
        dropout: float = 0.1,
        hub_token: str = "[EMO]",
        confidence_threshold: float = 0.5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.hub_token = hub_token
        self.hub_position = get_hub_positions()[hub_token]
        self.confidence_threshold = confidence_threshold

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Temperature for calibration (learned or fixed)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooled_outputs: Optional[Dict[str, torch.Tensor]] = None,
        return_confidence: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with optional confidence.

        Returns:
            Logits [batch, 2] or (logits, confidence) if return_confidence
        """
        if pooled_outputs is not None and self.hub_token in pooled_outputs:
            pooled = pooled_outputs[self.hub_token]
        else:
            pooled = hidden_states[:, self.hub_position, :]

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled) / self.temperature

        if return_confidence:
            probs = torch.softmax(logits, dim=-1)
            confidence = probs.max(dim=-1).values
            return logits, confidence

        return logits

    def predict_with_threshold(
        self,
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with confidence threshold.

        Returns:
            (predictions, is_confident) where is_confident indicates
            if prediction meets threshold
        """
        probs = torch.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values
        predictions = logits.argmax(dim=-1)
        is_confident = confidence >= self.confidence_threshold

        return predictions, is_confident


class HubAwareNLIHead(nn.Module):
    """
    NLI head using [REL] hub token.

    Labels: entailment (0), neutral (1), contradiction (2)
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_labels: int = 3,
        dropout: float = 0.1,
        hub_token: str = "[REL]",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.hub_token = hub_token
        self.hub_position = get_hub_positions()[hub_token]

        self.dropout = nn.Dropout(dropout)

        # Two-layer classifier for NLI
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooled_outputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward using [REL] hub."""
        if pooled_outputs is not None and self.hub_token in pooled_outputs:
            pooled = pooled_outputs[self.hub_token]
        else:
            pooled = hidden_states[:, self.hub_position, :]

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


# Head registry mapping capabilities to their head classes
HEAD_REGISTRY: Dict[str, type] = {
    # EMO hub heads
    "emotions": HubAwareHierarchicalHead,
    "sentiment": HubAwareClassificationHead,
    "safety_generic": HubAwareSafetyHead,
    "safety_familyos": HubAwareSafetyHead,

    # MEM hub heads
    "embedding": None,  # No head - uses raw [MEM] output

    # REL hub heads
    "nli": HubAwareNLIHead,
    "relation": HubAwareClassificationHead,

    # TASK hub heads
    "intent": HubAwareClassificationHead,
    "ingress": HubAwareClassificationHead,

    # Token-level heads (no hub pooling)
    "ner_general": HubAwareTokenClassificationHead,
    "ner_family": HubAwareTokenClassificationHead,
    "temporal": HubAwareTokenClassificationHead,
}


def create_head_for_capability(
    capability: str,
    hidden_size: int = 768,
    num_labels: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create appropriate head for a capability.

    Args:
        capability: Task/capability name
        hidden_size: Model hidden size
        num_labels: Number of output labels (task-dependent)
        **kwargs: Additional head-specific arguments

    Returns:
        Configured head module
    """
    if capability not in HEAD_REGISTRY:
        raise ValueError(f"Unknown capability: {capability}")

    head_class = HEAD_REGISTRY[capability]

    if head_class is None:
        return None  # Embedding task - no head needed

    # Get hub token for this capability
    hub_token = get_hub_for_capability(capability)

    # Default label counts per capability
    default_labels = {
        "emotions": 7,  # Ekman primary
        "sentiment": 3,  # pos/neg/neu
        "safety_generic": 2,
        "safety_familyos": 2,
        "nli": 3,
        "relation": 10,  # Family relations
        "intent": 15,  # Intent types
        "ingress": 8,  # Ingress categories
        "ner_general": 9,  # BIO tags
        "ner_family": 9,
        "temporal": 5,
    }

    if num_labels is None:
        num_labels = default_labels.get(capability, 2)

    # Handle special cases
    if capability == "emotions":
        return HubAwareHierarchicalHead(
            hidden_size=hidden_size,
            primary_labels=7,
            secondary_labels=28,
            hub_token=hub_token,
            **kwargs,
        )

    if capability in TOKEN_LEVEL_CAPABILITIES:
        return HubAwareTokenClassificationHead(
            hidden_size=hidden_size,
            num_labels=num_labels,
            **kwargs,
        )

    # Standard classification head
    return head_class(
        hidden_size=hidden_size,
        num_labels=num_labels,
        hub_token=hub_token,
        **kwargs,
    )


def create_all_heads(
    hidden_size: int = 768,
    capabilities: Optional[List[str]] = None,
) -> nn.ModuleDict:
    """
    Create heads for all (or specified) capabilities.

    Returns:
        ModuleDict of capability -> head
    """
    if capabilities is None:
        capabilities = list(HEAD_REGISTRY.keys())

    heads = nn.ModuleDict()

    for cap in capabilities:
        head = create_head_for_capability(cap, hidden_size)
        if head is not None:
            heads[cap] = head

    print(f"✓ Created {len(heads)} task heads")
    return heads
```

**Acceptance Criteria:**

- [ ] `HubAwareClassificationHead` extracts correct hub token
- [ ] `HubAwareTokenClassificationHead` uses full sequence (not hub)
- [ ] `HubAwareHierarchicalHead` implements primary→secondary cascade for emotions
- [ ] `HubAwareSafetyHead` includes temperature calibration
- [ ] `HubAwareNLIHead` uses `[REL]` hub token
- [ ] `HEAD_REGISTRY` maps all 12 capabilities to correct head types
- [ ] `create_head_for_capability()` factory works for all capabilities

**Tests:** `tests/v3/test_heads_v3.py::test_hub_aware_heads`

---

#### Issue 3.2.2: Implement Hub-Aware Loss Computation

**File:** `src/modeling_studio/models/losses_v3.py`
**Effort:** 4 hours
**Dependencies:** Issue 3.2.1

**Description:**
Implement loss computation that respects hub token routing and supports multi-task training with weighted losses, focal loss for imbalanced classes, and hierarchical loss for emotions.

**Implementation:**

```python
# src/modeling_studio/models/losses_v3.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

from .hub_tokens import TOKEN_LEVEL_CAPABILITIES, get_hub_for_capability


@dataclass
class LossOutput:
    """Container for loss computation results."""
    total_loss: torch.Tensor
    task_losses: Dict[str, torch.Tensor]
    task_weights: Dict[str, float]


class HubAwareLossComputer(nn.Module):
    """
    Computes losses for all tasks with hub routing awareness.

    Features:
        - Per-task loss weighting
        - Focal loss for imbalanced classes
        - Hierarchical loss for emotions
        - Label smoothing support
        - Hub gradient masking
    """

    def __init__(
        self,
        task_configs: Dict[str, Dict],
        label_smoothing: float = 0.0,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
    ):
        super().__init__()

        self.task_configs = task_configs
        self.label_smoothing = label_smoothing
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma

        # Per-task loss weights (can be learned or fixed)
        self.loss_weights = nn.ParameterDict()
        for task, config in task_configs.items():
            weight = config.get("loss_weight", 1.0)
            # Use buffer for fixed weights, Parameter for learnable
            self.register_buffer(f"weight_{task}", torch.tensor(weight))

        # Loss functions per task type
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=label_smoothing,
        )
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

    def compute_task_loss(
        self,
        task: str,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute loss for a single task.

        Args:
            task: Task name
            logits: Model predictions
            labels: Ground truth
            attention_mask: For token-level tasks

        Returns:
            Scalar loss tensor
        """
        config = self.task_configs.get(task, {})
        loss_type = config.get("loss_type", "cross_entropy")

        if task in TOKEN_LEVEL_CAPABILITIES:
            # Token-level loss (NER, temporal)
            return self._compute_token_level_loss(logits, labels, attention_mask)

        elif loss_type == "regression" or task in ["stsb", "similarity"]:
            # Regression loss
            return self.mse_loss(logits.squeeze(-1), labels.float())

        elif loss_type == "hierarchical" or task == "emotions":
            # Hierarchical loss for emotions
            if isinstance(logits, tuple):
                primary_logits, secondary_logits = logits
                primary_labels, secondary_labels = labels  # Assume tuple
                return self._compute_hierarchical_loss(
                    primary_logits, secondary_logits,
                    primary_labels, secondary_labels,
                )
            else:
                return self.ce_loss(logits, labels)

        elif self.use_focal_loss:
            # Focal loss for imbalanced classification
            return self._compute_focal_loss(logits, labels)

        else:
            # Standard cross-entropy
            return self.ce_loss(logits, labels)

    def _compute_token_level_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute loss for token-level tasks.

        Masks out hub token positions (0-4) and padding.
        """
        batch_size, seq_len, num_labels = logits.shape

        # Flatten for loss computation
        logits_flat = logits.view(-1, num_labels)
        labels_flat = labels.view(-1)

        # Mask hub token positions (they shouldn't contribute to loss)
        # Create mask: 1 for valid positions, 0 for hub/padding
        if attention_mask is not None:
            valid_mask = attention_mask.clone()
            valid_mask[:, :5] = 0  # Mask hub positions
            valid_mask = valid_mask.view(-1)

            # Set invalid positions to ignore_index
            labels_flat = labels_flat.masked_fill(valid_mask == 0, -100)

        loss = self.ce_loss(logits_flat, labels_flat)
        return loss

    def _compute_focal_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Focal loss for handling class imbalance.

        FL = -α(1-p)^γ * log(p)
        """
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        probs = torch.softmax(logits, dim=-1)
        pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)

        focal_weight = (1 - pt) ** self.focal_gamma
        focal_loss = focal_weight * ce_loss

        return focal_loss.mean()

    def _compute_hierarchical_loss(
        self,
        primary_logits: torch.Tensor,
        secondary_logits: torch.Tensor,
        primary_labels: torch.Tensor,
        secondary_labels: torch.Tensor,
        primary_weight: float = 0.4,
        secondary_weight: float = 0.6,
    ) -> torch.Tensor:
        """
        Hierarchical loss for emotions (Ekman + GoEmotions).

        Weights primary (coarse) vs secondary (fine) predictions.
        """
        primary_loss = self.ce_loss(primary_logits, primary_labels)
        secondary_loss = self.ce_loss(secondary_logits, secondary_labels)

        total_loss = primary_weight * primary_loss + secondary_weight * secondary_loss
        return total_loss

    def compute_multitask_loss(
        self,
        task_logits: Dict[str, torch.Tensor],
        task_labels: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        active_tasks: Optional[List[str]] = None,
    ) -> LossOutput:
        """
        Compute weighted sum of all task losses.

        Args:
            task_logits: Dict of task -> logits
            task_labels: Dict of task -> labels
            attention_mask: For token-level tasks
            active_tasks: Only compute loss for these tasks

        Returns:
            LossOutput with total and per-task losses
        """
        if active_tasks is None:
            active_tasks = list(task_logits.keys())

        task_losses = {}
        total_loss = torch.tensor(0.0, device=next(iter(task_logits.values())).device)

        for task in active_tasks:
            if task not in task_logits or task not in task_labels:
                continue

            logits = task_logits[task]
            labels = task_labels[task]

            # Compute task loss
            loss = self.compute_task_loss(task, logits, labels, attention_mask)
            task_losses[task] = loss

            # Add weighted loss
            weight = getattr(self, f"weight_{task}", torch.tensor(1.0))
            total_loss = total_loss + weight * loss

        return LossOutput(
            total_loss=total_loss,
            task_losses=task_losses,
            task_weights={t: getattr(self, f"weight_{t}", 1.0).item()
                         for t in task_losses},
        )

    def update_task_weight(self, task: str, weight: float) -> None:
        """Update loss weight for a task."""
        buffer_name = f"weight_{task}"
        if hasattr(self, buffer_name):
            setattr(self, buffer_name, torch.tensor(weight))
            print(f"✓ Updated weight for '{task}': {weight}")


class UncertaintyWeightedLoss(nn.Module):
    """
    Multi-task loss with learned uncertainty weighting.

    Based on "Multi-Task Learning Using Uncertainty to Weigh Losses"
    (Kendall et al., 2018)

    Loss = Σ (1/2σ²) * L_i + log(σ)

    The σ parameters are learned per-task.
    """

    def __init__(self, task_names: List[str]):
        super().__init__()

        self.task_names = task_names

        # Learnable log-variance for each task
        self.log_vars = nn.ParameterDict({
            task: nn.Parameter(torch.zeros(1))
            for task in task_names
        })

    def forward(
        self,
        task_losses: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute uncertainty-weighted total loss.

        Args:
            task_losses: Dict of task -> scalar loss

        Returns:
            (total_loss, effective_weights)
        """
        total_loss = torch.tensor(0.0, device=next(iter(task_losses.values())).device)
        effective_weights = {}

        for task, loss in task_losses.items():
            if task not in self.log_vars:
                continue

            log_var = self.log_vars[task]
            precision = torch.exp(-log_var)

            # Weighted loss + regularization
            weighted_loss = precision * loss + log_var
            total_loss = total_loss + weighted_loss

            effective_weights[task] = precision.item()

        return total_loss, effective_weights

    def get_task_weights(self) -> Dict[str, float]:
        """Get current effective weights (inverse variance)."""
        return {
            task: torch.exp(-log_var).item()
            for task, log_var in self.log_vars.items()
        }


def create_loss_computer(
    task_configs: Dict[str, Dict],
    use_uncertainty_weighting: bool = False,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create loss computer.

    Args:
        task_configs: Per-task configuration
        use_uncertainty_weighting: Use learned weights
        **kwargs: Additional HubAwareLossComputer args

    Returns:
        Loss computation module
    """
    if use_uncertainty_weighting:
        base_loss = HubAwareLossComputer(task_configs, **kwargs)
        uncertainty_loss = UncertaintyWeightedLoss(list(task_configs.keys()))
        # Return combined module
        return nn.ModuleDict({
            "base": base_loss,
            "uncertainty": uncertainty_loss,
        })
    else:
        return HubAwareLossComputer(task_configs, **kwargs)
```

**Acceptance Criteria:**

- [ ] Token-level loss masks hub token positions (0-4)
- [ ] Focal loss correctly implements γ-weighted cross entropy
- [ ] Hierarchical loss combines primary + secondary for emotions
- [ ] Multi-task loss aggregates with configurable weights
- [ ] `UncertaintyWeightedLoss` learns per-task σ parameters
- [ ] Label smoothing works with cross-entropy
- [ ] Factory function supports both fixed and uncertainty weighting

**Tests:** `tests/v3/test_losses_v3.py::test_hub_aware_loss`

---

#### Issue 3.2.3: Update Task Head Registry for v3

**File:** `src/modeling_studio/models/registry_v3.py`
**Effort:** 3 hours
**Dependencies:** Issues 3.2.1, 3.2.2

**Description:**
Create a unified registry for v3 task heads that tracks hub routing, label schemas, and default configurations for all 12 capabilities.

**Implementation:**

```python
# src/modeling_studio/models/registry_v3.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type, Any
from enum import Enum
import torch.nn as nn

from .hub_tokens import get_hub_for_capability, TOKEN_LEVEL_CAPABILITIES
from .heads_v3 import (
    HubAwareClassificationHead,
    HubAwareTokenClassificationHead,
    HubAwareHierarchicalHead,
    HubAwareSafetyHead,
    HubAwareNLIHead,
)


class TaskType(Enum):
    """Types of tasks supported."""
    CLASSIFICATION = "classification"
    TOKEN_CLASSIFICATION = "token_classification"
    REGRESSION = "regression"
    HIERARCHICAL = "hierarchical"
    EMBEDDING = "embedding"


@dataclass
class TaskSpec:
    """Complete specification for a task/capability."""
    name: str
    task_type: TaskType
    hub_token: str
    head_class: Optional[Type[nn.Module]]
    num_labels: int
    label_names: List[str]
    loss_type: str = "cross_entropy"
    loss_weight: float = 1.0
    metrics: List[str] = field(default_factory=list)
    description: str = ""


# Complete task registry for v3
# NOTE: Label counts MUST match src/modeling_studio/data/labels.py (ground truth)
TASK_REGISTRY_V3: Dict[str, TaskSpec] = {
    # ═══════════════════════════════════════════════════════════════
    # [EMO] Hub Tasks - Emotional/Affective Understanding
    # ═══════════════════════════════════════════════════════════════
    "emotions": TaskSpec(
        name="emotions",
        task_type=TaskType.CLASSIFICATION,  # Flat classification for 44 FamilyOS emotions
        hub_token="[EMO]",
        head_class=HubAwareClassificationHead,
        num_labels=44,  # EMOTIONS_FAMILYOS_LABELS: Core(8) + Positive(12) + Negative(10) + Family(14)
        label_names=[
            # Core Emotions (8)
            "neutral", "joy", "sadness", "anger", "fear", "surprise", "love", "disgust",
            # Positive Emotions (12)
            "admiration", "amusement", "approval", "caring", "curiosity", "desire",
            "excitement", "gratitude", "hope", "optimism", "pride", "tenderness",
            # Negative Emotions (10)
            "annoyance", "confusion", "disappointment", "disapproval", "embarrassment",
            "grief", "nervousness", "remorse", "worry", "emptiness",
            # Family-Specific Emotions (14)
            "nostalgia", "protectiveness", "relief", "contentment", "longing",
            "resentment", "guilt", "overwhelmed", "belonging", "abandonment",
            "jealousy", "trust", "vulnerability", "homesickness",
        ],
        loss_type="cross_entropy",
        loss_weight=1.0,
        metrics=["macro_f1", "accuracy"],
        description="Flat emotion classification (44 FamilyOS emotions)",
    ),

    "sentiment": TaskSpec(
        name="sentiment",
        task_type=TaskType.CLASSIFICATION,
        hub_token="[EMO]",
        head_class=HubAwareClassificationHead,
        num_labels=5,  # SENTIMENT_LABELS: 5-class scale
        label_names=["very_negative", "negative", "neutral", "positive", "very_positive"],
        loss_type="cross_entropy",
        loss_weight=0.8,
        metrics=["accuracy", "macro_f1"],
        description="5-class sentiment polarity classification",
    ),

    "safety_generic": TaskSpec(
        name="safety_generic",
        task_type=TaskType.MULTI_LABEL,  # Multi-label classification
        hub_token="[EMO]",
        head_class=HubAwareSafetyHead,
        num_labels=8,  # SAFETY_GENERIC_LABELS: 6 Jigsaw + 2 new
        label_names=[
            "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate",
            "self_harm", "dangerous_advice",
        ],
        loss_type="binary_cross_entropy",  # Multi-label uses BCE
        loss_weight=1.5,  # Higher weight for safety
        metrics=["recall", "precision", "f1"],
        description="Multi-label toxicity detection (8 types)",
    ),

    "safety_familyos": TaskSpec(
        name="safety_familyos",
        task_type=TaskType.CLASSIFICATION,
        hub_token="[EMO]",
        head_class=HubAwareSafetyHead,
        num_labels=4,  # SAFETY_FAMILYOS_LABELS: GREEN, AMBER, RED, CRISIS
        label_names=["GREEN", "AMBER", "RED", "CRISIS"],
        loss_type="cross_entropy",
        loss_weight=2.0,  # Highest weight - CRISIS recall is critical
        metrics=["recall", "precision", "f1", "crisis_recall"],
        description="FamilyOS safety policy bands (GREEN to CRISIS)",
    ),

    # ═══════════════════════════════════════════════════════════════
    # [MEM] Hub Tasks - Memory/Embedding
    # ═══════════════════════════════════════════════════════════════
    "embedding": TaskSpec(
        name="embedding",
        task_type=TaskType.EMBEDDING,
        hub_token="[MEM]",
        head_class=None,  # No head - uses raw [MEM] output
        num_labels=0,
        label_names=[],
        loss_type="contrastive",
        loss_weight=1.0,
        metrics=["recall@10", "mrr"],
        description="Sentence embedding for retrieval/similarity",
    ),

    # ═══════════════════════════════════════════════════════════════
    # [REL] Hub Tasks - Relationship Understanding
    # ═══════════════════════════════════════════════════════════════
    "nli": TaskSpec(
        name="nli",
        task_type=TaskType.CLASSIFICATION,
        hub_token="[REL]",
        head_class=HubAwareNLIHead,
        num_labels=3,
        label_names=["entailment", "neutral", "contradiction"],
        loss_type="cross_entropy",
        loss_weight=1.0,
        metrics=["accuracy"],
        description="Natural Language Inference",
    ),

    "relation": TaskSpec(
        name="relation",
        task_type=TaskType.CLASSIFICATION,
        hub_token="[REL]",
        head_class=HubAwareClassificationHead,
        num_labels=15,  # RELATION_LABELS: no_relation + 14 relations
        label_names=[
            "no_relation",
            "parent_of", "child_of", "spouse_of", "sibling_of",
            "grandparent_of", "grandchild_of", "aunt_uncle_of", "niece_nephew_of",
            "cousin_of", "pet_of", "friend_of", "colleague_of", "lives_at", "owns",
        ],
        loss_type="cross_entropy",
        loss_weight=1.2,
        metrics=["macro_f1", "accuracy"],
        description="Family relationship extraction (15 relations)",
    ),

    # ═══════════════════════════════════════════════════════════════
    # [TASK] Hub Tasks - Intent/Action Understanding
    # ═══════════════════════════════════════════════════════════════
    "intent": TaskSpec(
        name="intent",
        task_type=TaskType.CLASSIFICATION,
        hub_token="[TASK]",
        head_class=HubAwareClassificationHead,
        num_labels=8,  # INTENT_LABELS: 8 FamilyOS intents
        label_names=[
            "log_memory", "query_memory", "set_reminder", "express_feeling",
            "seek_advice", "share_news", "reflect", "other",
        ],
        loss_type="cross_entropy",
        loss_weight=1.0,
        metrics=["accuracy", "macro_f1"],
        description="FamilyOS user intent classification (8 intents)",
    ),

    "ingress": TaskSpec(
        name="ingress",
        task_type=TaskType.CLASSIFICATION,
        hub_token="[TASK]",
        head_class=HubAwareClassificationHead,
        num_labels=12,  # INGRESS_LABELS: 7 original + 5 extended
        label_names=[
            "DIARY", "TASK", "HEALTH", "FINANCE", "RELATIONSHIP", "WORK", "META",
            "MEMORY", "PLANNING", "CELEBRATION", "CONCERN", "GRATITUDE",
        ],
        loss_type="cross_entropy",
        loss_weight=1.0,
        metrics=["accuracy"],
        description="Extended domain classification (12 domains)",
    ),

    # ═══════════════════════════════════════════════════════════════
    # Token-Level Tasks (No Hub Pooling)
    # ═══════════════════════════════════════════════════════════════
    "ner_general": TaskSpec(
        name="ner_general",
        task_type=TaskType.TOKEN_CLASSIFICATION,
        hub_token="[CLS]",  # Not used - full sequence
        head_class=HubAwareTokenClassificationHead,
        num_labels=17,  # NER_GENERAL_LABELS: 17 BIO tags
        label_names=[
            "O",
            "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC",
            "B-MISC", "I-MISC", "B-DATE", "I-DATE", "B-TIME", "I-TIME",
            "B-EVENT", "I-EVENT", "B-PRODUCT", "I-PRODUCT",
        ],
        loss_type="cross_entropy",
        loss_weight=1.0,
        metrics=["entity_f1", "precision", "recall"],
        description="General named entity recognition (17 BIO tags)",
    ),

    "ner_family": TaskSpec(
        name="ner_family",
        task_type=TaskType.TOKEN_CLASSIFICATION,
        hub_token="[CLS]",
        head_class=HubAwareTokenClassificationHead,
        num_labels=21,  # NER_FAMILY_LABELS: 21 BIO tags
        label_names=[
            "O",
            "B-PERSON", "I-PERSON", "B-KINSHIP", "I-KINSHIP",
            "B-NICKNAME", "I-NICKNAME", "B-PET", "I-PET",
            "B-HOME_LOC", "I-HOME_LOC", "B-FAMILY_EVENT", "I-FAMILY_EVENT",
            "B-ROUTINE", "I-ROUTINE", "B-TRADITION", "I-TRADITION",
            "B-MILESTONE", "I-MILESTONE", "B-HEIRLOOM", "I-HEIRLOOM",
        ],
        loss_type="cross_entropy",
        loss_weight=1.2,
        metrics=["entity_f1", "precision", "recall"],
        description="Family-specific entity recognition (21 BIO tags)",
    ),

    "temporal": TaskSpec(
        name="temporal",
        task_type=TaskType.TOKEN_CLASSIFICATION,
        hub_token="[CLS]",
        head_class=HubAwareTokenClassificationHead,
        num_labels=13,  # TEMPORAL_LABELS: 13 BIO tags
        label_names=[
            "O",
            "B-DATE_ABS", "I-DATE_ABS", "B-DATE_REL", "I-DATE_REL",
            "B-TIME", "I-TIME", "B-DURATION", "I-DURATION",
            "B-FREQUENCY", "I-FREQUENCY", "B-AGE", "I-AGE",
        ],
        loss_type="cross_entropy",
        loss_weight=1.0,
        metrics=["entity_f1"],
        description="Temporal expression extraction (13 BIO tags)",
    ),
}


class TaskRegistry:
    """
    Registry for managing v3 task configurations.

    Provides:
        - Task lookup by name
        - Head creation
        - Hub routing information
        - Metric configuration
    """

    def __init__(self, custom_registry: Optional[Dict[str, TaskSpec]] = None):
        self.registry = TASK_REGISTRY_V3.copy()
        if custom_registry:
            self.registry.update(custom_registry)

    def get_task(self, name: str) -> TaskSpec:
        """Get task specification by name."""
        if name not in self.registry:
            raise ValueError(f"Unknown task: {name}. Available: {list(self.registry.keys())}")
        return self.registry[name]

    def get_all_tasks(self) -> List[str]:
        """Get all registered task names."""
        return list(self.registry.keys())

    def get_tasks_by_hub(self, hub_token: str) -> List[str]:
        """Get all tasks routed through a hub token."""
        return [
            name for name, spec in self.registry.items()
            if spec.hub_token == hub_token
        ]

    def get_hub_routed_tasks(self) -> List[str]:
        """Get tasks that use hub token pooling (not token-level)."""
        return [
            name for name, spec in self.registry.items()
            if spec.task_type != TaskType.TOKEN_CLASSIFICATION
            and spec.task_type != TaskType.EMBEDDING
        ]

    def get_token_level_tasks(self) -> List[str]:
        """Get tasks that use token-level classification."""
        return [
            name for name, spec in self.registry.items()
            if spec.task_type == TaskType.TOKEN_CLASSIFICATION
        ]

    def create_head(
        self,
        task_name: str,
        hidden_size: int = 768,
        **kwargs,
    ) -> Optional[nn.Module]:
        """Create head for a task."""
        spec = self.get_task(task_name)

        if spec.head_class is None:
            return None

        return spec.head_class(
            hidden_size=hidden_size,
            num_labels=spec.num_labels,
            hub_token=spec.hub_token,
            **kwargs,
        )

    def create_all_heads(
        self,
        hidden_size: int = 768,
        tasks: Optional[List[str]] = None,
    ) -> nn.ModuleDict:
        """Create heads for multiple tasks."""
        if tasks is None:
            tasks = self.get_all_tasks()

        heads = nn.ModuleDict()
        for task in tasks:
            head = self.create_head(task, hidden_size)
            if head is not None:
                heads[task] = head

        return heads

    def get_loss_weights(self) -> Dict[str, float]:
        """Get default loss weights for all tasks."""
        return {name: spec.loss_weight for name, spec in self.registry.items()}

    def get_metrics(self, task_name: str) -> List[str]:
        """Get metrics for a task."""
        return self.get_task(task_name).metrics

    def print_registry(self) -> None:
        """Print registry summary."""
        print("\n" + "=" * 80)
        print("📋 v3 Task Registry")
        print("=" * 80)

        by_hub = {}
        for name, spec in self.registry.items():
            hub = spec.hub_token
            if hub not in by_hub:
                by_hub[hub] = []
            by_hub[hub].append((name, spec))

        for hub, tasks in by_hub.items():
            print(f"\n  {hub} Hub:")
            for name, spec in tasks:
                type_str = spec.task_type.value[:12]
                print(f"    {name:<18} {type_str:<15} labels={spec.num_labels:<3} weight={spec.loss_weight}")

        print("\n" + "=" * 80)


# Singleton registry instance
_registry = None

def get_registry() -> TaskRegistry:
    """Get global task registry."""
    global _registry
    if _registry is None:
        _registry = TaskRegistry()
    return _registry
```

**Acceptance Criteria:**

- [ ] All 12 capabilities registered with complete specifications
- [ ] `TaskSpec` includes hub_token, head_class, labels, metrics
- [ ] `get_tasks_by_hub()` returns correct tasks per hub
- [ ] `create_head()` instantiates correct head class
- [ ] Loss weights configured (safety tasks have higher weight)
- [ ] Token-level tasks correctly identified
- [ ] `print_registry()` shows organized summary

**Tests:** `tests/v3/test_registry_v3.py::test_task_registry`

---

## 🏁 Milestone 4: Function Preserving Growth & Initialization

**Goal:** Initialize v3 from v2 weights via direct transfer

### Epic 4.1: Weight Transfer

#### Issue 4.1.1: Implement v2 Checkpoint Loader

**File:** `src/modeling_studio/models/initialization_v3.py`
**Effort:** 4 hours
**Dependencies:** Issue 1.1.1 (ModernBERTv3Config)

**Description:**
Implement a robust checkpoint loader that can read v2 (22-layer) checkpoints and map them to v3 (28-layer) architecture. This handles differences in layer naming, vocabulary size, and special token configuration.

**Implementation:**

```python
# src/modeling_studio/models/initialization_v3.py

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List, Any
from pathlib import Path
import re
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class V2CheckpointInfo:
    """Information about a v2 checkpoint."""
    path: Path
    num_layers: int  # Should be 22
    hidden_size: int
    vocab_size: int
    has_pooler: bool
    has_task_heads: bool
    state_dict_keys: List[str]


@dataclass
class WeightTransferStats:
    """Statistics from weight transfer."""
    total_params: int
    transferred_params: int
    initialized_params: int  # New params (hub tokens, new layers)
    skipped_params: int
    layer_mapping: Dict[int, int]  # v3_layer -> v2_layer


class V2CheckpointLoader:
    """
    Loads and parses ModernBERT v2 checkpoints.

    v2 Architecture (22 layers):
        - Foundation Band: L1-6 (window=64)
        - Core Band: L7-18 (window=128)
        - Family Band: L19-22 (window=256)

    v3 Architecture (28 layers):
        - Foundation Band: L1-6 (window=64) ← COPY from v2 L1-6
        - Core Band: L7-18 (window=128) ← COPY from v2 L7-18
        - Feeder Band: L19-22 (window=256) ← COPY from v2 L19-22
        - Family Band: L23-28 (window=512) ← CLONE from v2 L15-20
    """

    V2_NUM_LAYERS = 22
    V3_NUM_LAYERS = 28

    # Layer mapping: v3_layer -> v2_layer (None = newly initialized)
    LAYER_MAPPING = {
        # Foundation Band: Direct copy
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
        # Core Band: Direct copy
        6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11,
        12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17,
        # Feeder Band: Direct copy from v2 Family Band
        18: 18, 19: 19, 20: 20, 21: 21,
        # Family Band: Clone from v2 Core/Family layers 15-20
        22: 14, 23: 15, 24: 16, 25: 17, 26: 18, 27: 19,
    }

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = Path(checkpoint_path)
        self._state_dict: Optional[Dict[str, torch.Tensor]] = None
        self._info: Optional[V2CheckpointInfo] = None

    def load(self) -> Dict[str, torch.Tensor]:
        """Load checkpoint state dict."""
        if self._state_dict is None:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )

            # Handle different checkpoint formats
            if "state_dict" in checkpoint:
                self._state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                self._state_dict = checkpoint["model_state_dict"]
            elif "model" in checkpoint:
                self._state_dict = checkpoint["model"]
            else:
                self._state_dict = checkpoint

            # Clean up module. prefix if present
            self._state_dict = self._clean_state_dict(self._state_dict)

        return self._state_dict

    def _clean_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Remove 'module.' prefix from DDP checkpoints."""
        cleaned = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[7:]  # Remove 'module.' prefix
            cleaned[key] = value
        return cleaned

    def get_info(self) -> V2CheckpointInfo:
        """Extract checkpoint metadata."""
        if self._info is None:
            state_dict = self.load()

            # Detect layer count
            layer_pattern = re.compile(r"encoder\.layers\.(\d+)\.")
            layer_indices = set()
            for key in state_dict.keys():
                match = layer_pattern.search(key)
                if match:
                    layer_indices.add(int(match.group(1)))

            num_layers = max(layer_indices) + 1 if layer_indices else 0

            # Detect hidden size from first layer norm
            hidden_size = 768  # default
            for key, tensor in state_dict.items():
                if "layer_norm" in key and tensor.dim() == 1:
                    hidden_size = tensor.shape[0]
                    break

            # Detect vocab size from embeddings
            vocab_size = 50368  # default v2 vocab
            for key, tensor in state_dict.items():
                if "word_embeddings" in key and tensor.dim() == 2:
                    vocab_size = tensor.shape[0]
                    break

            # Check for pooler and task heads
            has_pooler = any("pooler" in k for k in state_dict.keys())
            has_task_heads = any("head" in k.lower() for k in state_dict.keys())

            self._info = V2CheckpointInfo(
                path=self.checkpoint_path,
                num_layers=num_layers,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                has_pooler=has_pooler,
                has_task_heads=has_task_heads,
                state_dict_keys=list(state_dict.keys()),
            )

        return self._info

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate checkpoint is compatible with v2→v3 transfer.

        Returns:
            (is_valid, list of issues)
        """
        info = self.get_info()
        issues = []

        if info.num_layers != self.V2_NUM_LAYERS:
            issues.append(
                f"Expected {self.V2_NUM_LAYERS} layers, found {info.num_layers}"
            )

        if info.hidden_size != 768:
            issues.append(
                f"Expected hidden_size=768, found {info.hidden_size}"
            )

        # Check for required keys
        required_patterns = [
            "embeddings.word_embeddings",
            "encoder.layers.0",
            "encoder.layers.21",  # Last v2 layer
        ]

        state_dict = self.load()
        for pattern in required_patterns:
            if not any(pattern in k for k in state_dict.keys()):
                issues.append(f"Missing required pattern: {pattern}")

        return len(issues) == 0, issues

    def get_layer_weights(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """
        Get all weights for a specific layer.

        Args:
            layer_idx: v2 layer index (0-21)

        Returns:
            Dict of weight name -> tensor
        """
        state_dict = self.load()
        prefix = f"encoder.layers.{layer_idx}."

        layer_weights = {}
        for key, tensor in state_dict.items():
            if key.startswith(prefix):
                # Remove prefix for cleaner mapping
                short_key = key[len(prefix):]
                layer_weights[short_key] = tensor

        return layer_weights

    def get_embedding_weights(self) -> Dict[str, torch.Tensor]:
        """Get embedding layer weights."""
        state_dict = self.load()
        embedding_weights = {}

        for key, tensor in state_dict.items():
            if key.startswith("embeddings."):
                short_key = key[len("embeddings."):]
                embedding_weights[short_key] = tensor

        return embedding_weights

    def print_summary(self) -> None:
        """Print checkpoint summary."""
        info = self.get_info()
        print("\n" + "=" * 60)
        print("📦 v2 Checkpoint Summary")
        print("=" * 60)
        print(f"  Path: {info.path}")
        print(f"  Layers: {info.num_layers}")
        print(f"  Hidden Size: {info.hidden_size}")
        print(f"  Vocab Size: {info.vocab_size}")
        print(f"  Has Pooler: {info.has_pooler}")
        print(f"  Has Task Heads: {info.has_task_heads}")
        print(f"  Total Keys: {len(info.state_dict_keys)}")
        print("=" * 60)


def load_v2_checkpoint(path: str) -> V2CheckpointLoader:
    """
    Factory function to load v2 checkpoint.

    Args:
        path: Path to checkpoint file

    Returns:
        Configured loader
    """
    loader = V2CheckpointLoader(path)

    # Validate
    is_valid, issues = loader.validate()
    if not is_valid:
        logger.warning(f"Checkpoint validation issues: {issues}")

    loader.print_summary()
    return loader
```

**Acceptance Criteria:**

- [ ] Loads PyTorch v2 checkpoints (22 layers)
- [ ] Handles different checkpoint formats (state_dict, model, etc.)
- [ ] Cleans `module.` prefix from DDP checkpoints
- [ ] Extracts metadata (layers, hidden_size, vocab_size)
- [ ] `validate()` checks compatibility
- [ ] `get_layer_weights()` extracts per-layer tensors
- [ ] `get_embedding_weights()` extracts embedding tensors

**Tests:** `tests/v3/test_initialization_v3.py::test_v2_checkpoint_loader`

---

#### Issue 4.1.2: Implement Layer 1-22 Direct Copy

**File:** `src/modeling_studio/models/initialization_v3.py` (extend)
**Effort:** 3 hours
**Dependencies:** Issue 4.1.1

**Description:**
Implement direct weight copying from v2 layers 1-22 to v3 layers 1-22. This preserves all learned representations from the Foundation, Core, and Feeder bands exactly.

**Implementation:**

```python
# Add to initialization_v3.py

class LayerCopier:
    """
    Copies layer weights from v2 to v3.

    Layer Mapping (v3 ← v2):
        L1-6 (Foundation) ← L1-6: Direct copy (window 64)
        L7-18 (Core) ← L7-18: Direct copy (window 128)
        L19-22 (Feeder) ← L19-22: Direct copy (window 256)
    """

    def __init__(
        self,
        v2_loader: V2CheckpointLoader,
        strict: bool = True,
    ):
        self.v2_loader = v2_loader
        self.strict = strict
        self.copy_stats = {
            "matched": 0,
            "mismatched_shape": 0,
            "missing_in_v2": 0,
        }

    def copy_layer(
        self,
        v3_layer: nn.Module,
        v2_layer_idx: int,
        v3_layer_idx: int,
    ) -> int:
        """
        Copy weights from v2 layer to v3 layer.

        Args:
            v3_layer: Target v3 layer module
            v2_layer_idx: Source layer index in v2
            v3_layer_idx: Target layer index in v3 (for logging)

        Returns:
            Number of parameters copied
        """
        v2_weights = self.v2_loader.get_layer_weights(v2_layer_idx)
        copied_params = 0

        v3_state = v3_layer.state_dict()

        for v3_key in v3_state.keys():
            # Map v3 key to v2 key (same structure)
            v2_key = v3_key

            if v2_key not in v2_weights:
                if self.strict:
                    logger.warning(
                        f"Layer {v3_layer_idx}: Missing v2 weight for '{v3_key}'"
                    )
                self.copy_stats["missing_in_v2"] += 1
                continue

            v2_tensor = v2_weights[v2_key]
            v3_tensor = v3_state[v3_key]

            if v2_tensor.shape != v3_tensor.shape:
                logger.warning(
                    f"Layer {v3_layer_idx}: Shape mismatch for '{v3_key}': "
                    f"v2={v2_tensor.shape}, v3={v3_tensor.shape}"
                )
                self.copy_stats["mismatched_shape"] += 1
                continue

            # Copy weight
            v3_state[v3_key] = v2_tensor.clone()
            copied_params += v2_tensor.numel()
            self.copy_stats["matched"] += 1

        # Load updated state
        v3_layer.load_state_dict(v3_state)

        return copied_params

    def copy_layers_1_to_22(
        self,
        v3_encoder: nn.Module,
    ) -> int:
        """
        Copy all v2 layers 0-21 to v3 layers 0-21 (direct copy).

        Args:
            v3_encoder: v3 encoder with layers attribute

        Returns:
            Total parameters copied
        """
        total_copied = 0

        for v3_idx in range(22):  # Layers 0-21 (1-22 in 1-indexed)
            v2_idx = v3_idx  # Direct mapping

            v3_layer = v3_encoder.layers[v3_idx]
            copied = self.copy_layer(v3_layer, v2_idx, v3_idx)
            total_copied += copied

            logger.info(f"  Layer {v3_idx}: Copied {copied:,} params from v2 L{v2_idx}")

        return total_copied

    def get_stats(self) -> Dict[str, int]:
        """Get copy statistics."""
        return self.copy_stats.copy()


def copy_layers_direct(
    v3_model: nn.Module,
    v2_checkpoint_path: str,
) -> int:
    """
    Copy v2 layers 1-22 directly to v3 layers 1-22.

    Args:
        v3_model: Target v3 model
        v2_checkpoint_path: Path to v2 checkpoint

    Returns:
        Number of parameters copied
    """
    loader = V2CheckpointLoader(v2_checkpoint_path)
    copier = LayerCopier(loader)

    print("\n🔄 Copying v2 Layers 1-22 to v3 Layers 1-22...")

    # Get encoder from model
    encoder = v3_model.encoder if hasattr(v3_model, "encoder") else v3_model

    total_copied = copier.copy_layers_1_to_22(encoder)

    stats = copier.get_stats()
    print(f"\n✓ Direct copy complete:")
    print(f"  - Matched: {stats['matched']}")
    print(f"  - Shape mismatches: {stats['mismatched_shape']}")
    print(f"  - Missing in v2: {stats['missing_in_v2']}")
    print(f"  - Total params: {total_copied:,}")

    return total_copied
```

**Acceptance Criteria:**

- [ ] Copies all 22 v2 layers to first 22 v3 layers
- [ ] Handles shape mismatches gracefully with warnings
- [ ] Reports statistics (matched, mismatched, missing)
- [ ] Works with both strict and non-strict modes
- [ ] Preserves parameter values exactly (no modifications)

**Tests:** `tests/v3/test_initialization_v3.py::test_layer_direct_copy`

---

#### Issue 4.1.3: Implement Layer 23-28 Cloning from L15-20

**File:** `src/modeling_studio/models/initialization_v3.py` (extend)
**Effort:** 4 hours
**Dependencies:** Issue 4.1.2

**Description:**
Clone v2 layers 15-20 (middle Core + early Family) to initialize v3 layers 23-28 (new Family Band). This provides strong initialization for the new layers with proven representations.

**Implementation:**

```python
# Add to initialization_v3.py

class LayerCloner:
    """
    Clones layer weights from v2 to new v3 layers.

    Clone Mapping (v3 ← v2):
        L23 ← L15: First Family Band layer
        L24 ← L16: Second layer
        L25 ← L17: Third layer
        L26 ← L18: Fourth layer
        L27 ← L19: Fifth layer
        L28 ← L20: Sixth layer

    Why L15-20?
        - L15-18: Late Core Band - good general representations
        - L19-20: Early Family Band - task-relevant features
        - Together: balanced mix of general + specialized
    """

    # Clone mapping: v3_layer_idx -> v2_layer_idx
    CLONE_MAPPING = {
        22: 14,  # L23 ← L15 (0-indexed)
        23: 15,  # L24 ← L16
        24: 16,  # L25 ← L17
        25: 17,  # L26 ← L18
        26: 18,  # L27 ← L19
        27: 19,  # L28 ← L20
    }

    def __init__(
        self,
        v2_loader: V2CheckpointLoader,
        add_noise: bool = False,
        noise_std: float = 0.01,
    ):
        self.v2_loader = v2_loader
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.clone_stats = {
            "cloned": 0,
            "noise_added": 0,
        }

    def clone_layer(
        self,
        v3_layer: nn.Module,
        v2_layer_idx: int,
        v3_layer_idx: int,
    ) -> int:
        """
        Clone weights from v2 layer to v3 layer.

        Optionally adds small noise to break symmetry between
        cloned layers (helps them specialize during training).

        Args:
            v3_layer: Target v3 layer module
            v2_layer_idx: Source layer index in v2
            v3_layer_idx: Target layer index in v3

        Returns:
            Number of parameters cloned
        """
        v2_weights = self.v2_loader.get_layer_weights(v2_layer_idx)
        cloned_params = 0

        v3_state = v3_layer.state_dict()

        for v3_key in v3_state.keys():
            v2_key = v3_key

            if v2_key not in v2_weights:
                logger.warning(
                    f"Layer {v3_layer_idx}: No v2 weight for '{v3_key}', "
                    "using random init"
                )
                continue

            v2_tensor = v2_weights[v2_key]
            v3_tensor = v3_state[v3_key]

            if v2_tensor.shape != v3_tensor.shape:
                logger.warning(
                    f"Layer {v3_layer_idx}: Shape mismatch for '{v3_key}': "
                    f"v2={v2_tensor.shape}, v3={v3_tensor.shape}"
                )
                continue

            # Clone with optional noise
            cloned = v2_tensor.clone()

            if self.add_noise and v3_key.endswith(".weight"):
                # Add small noise to weights (not biases/norms)
                noise = torch.randn_like(cloned) * self.noise_std
                cloned = cloned + noise
                self.clone_stats["noise_added"] += 1

            v3_state[v3_key] = cloned
            cloned_params += cloned.numel()
            self.clone_stats["cloned"] += 1

        v3_layer.load_state_dict(v3_state)
        return cloned_params

    def clone_layers_23_to_28(
        self,
        v3_encoder: nn.Module,
    ) -> int:
        """
        Clone v2 layers 15-20 to v3 layers 23-28.

        Args:
            v3_encoder: v3 encoder with layers attribute

        Returns:
            Total parameters cloned
        """
        total_cloned = 0

        print("\n🧬 Cloning v2 Layers 15-20 to v3 Layers 23-28...")

        for v3_idx, v2_idx in self.CLONE_MAPPING.items():
            v3_layer = v3_encoder.layers[v3_idx]
            cloned = self.clone_layer(v3_layer, v2_idx, v3_idx)
            total_cloned += cloned

            noise_str = " (+noise)" if self.add_noise else ""
            logger.info(
                f"  Layer {v3_idx}: Cloned {cloned:,} params from v2 L{v2_idx}{noise_str}"
            )

        return total_cloned

    def get_stats(self) -> Dict[str, int]:
        """Get clone statistics."""
        return self.clone_stats.copy()


def clone_layers_for_growth(
    v3_model: nn.Module,
    v2_checkpoint_path: str,
    add_noise: bool = True,
    noise_std: float = 0.01,
) -> int:
    """
    Clone v2 layers 15-20 to v3 layers 23-28.

    Args:
        v3_model: Target v3 model
        v2_checkpoint_path: Path to v2 checkpoint
        add_noise: Add small noise to break symmetry
        noise_std: Standard deviation of noise

    Returns:
        Number of parameters cloned
    """
    loader = V2CheckpointLoader(v2_checkpoint_path)
    cloner = LayerCloner(loader, add_noise=add_noise, noise_std=noise_std)

    encoder = v3_model.encoder if hasattr(v3_model, "encoder") else v3_model

    total_cloned = cloner.clone_layers_23_to_28(encoder)

    stats = cloner.get_stats()
    print(f"\n✓ Layer cloning complete:")
    print(f"  - Cloned weights: {stats['cloned']}")
    print(f"  - Noise added to: {stats['noise_added']} tensors")
    print(f"  - Total params: {total_cloned:,}")

    return total_cloned


# Layer band configuration for v3
V3_LAYER_BANDS = {
    "foundation": list(range(0, 6)),    # L1-6: window=64
    "core": list(range(6, 18)),         # L7-18: window=128
    "feeder": list(range(18, 22)),      # L19-22: window=256
    "family": list(range(22, 28)),      # L23-28: window=512
}

def get_clone_source_for_layer(v3_layer_idx: int) -> Optional[int]:
    """Get the v2 layer that was cloned to create this v3 layer."""
    return LayerCloner.CLONE_MAPPING.get(v3_layer_idx)
```

**Acceptance Criteria:**

- [ ] Clones v2 L15-20 to v3 L23-28 correctly
- [ ] Optional noise addition breaks symmetry
- [ ] Noise only added to weights, not biases/LayerNorm
- [ ] Reports cloning statistics
- [ ] Layer band configuration exported for training

**Tests:** `tests/v3/test_initialization_v3.py::test_layer_cloning`

---

#### Issue 4.1.4: Implement Embedding Transfer with Hub Token Slots

**File:** `src/modeling_studio/models/initialization_v3.py` (extend)
**Effort:** 4 hours
**Dependencies:** Issues 4.1.1, 1.2.1 (hub tokens)

**Description:**
Transfer v2 word embeddings to v3 while creating slots for the 4 new hub tokens. The v3 vocabulary is v2_vocab + 4 hub tokens, with hub tokens placed at specific indices.

**Implementation:**

```python
# Add to initialization_v3.py

from .hub_tokens import HUB_TOKEN_REGISTRY, get_hub_positions


class EmbeddingTransfer:
    """
    Transfers embeddings from v2 to v3 with hub token slot creation.

    v2 Vocabulary: 50,368 tokens (ModernBERT-base)
    v3 Vocabulary: 50,372 tokens (+4 hub tokens)

    Hub Token Positions (added at end of vocab):
        [EMO] = 50368
        [MEM] = 50369
        [REL] = 50370
        [TASK] = 50371

    Embedding Layout:
        v2: [vocab_embeddings: 50368]
        v3: [vocab_embeddings: 50368, hub_tokens: 4]
    """

    V2_VOCAB_SIZE = 50368
    NUM_HUB_TOKENS = 4
    V3_VOCAB_SIZE = V2_VOCAB_SIZE + NUM_HUB_TOKENS

    def __init__(self, v2_loader: V2CheckpointLoader):
        self.v2_loader = v2_loader
        self.transfer_stats = {
            "vocab_transferred": 0,
            "hub_slots_created": 0,
            "position_embeddings_transferred": 0,
        }

    def transfer_word_embeddings(
        self,
        v3_embeddings: nn.Module,
    ) -> int:
        """
        Transfer word embeddings from v2, creating hub token slots.

        Args:
            v3_embeddings: v3 embedding module with word_embeddings

        Returns:
            Number of parameters transferred
        """
        v2_emb_weights = self.v2_loader.get_embedding_weights()

        if "word_embeddings.weight" not in v2_emb_weights:
            raise ValueError("v2 checkpoint missing word_embeddings.weight")

        v2_word_emb = v2_emb_weights["word_embeddings.weight"]
        v2_vocab_size, hidden_size = v2_word_emb.shape

        # Verify expected size
        if v2_vocab_size != self.V2_VOCAB_SIZE:
            logger.warning(
                f"Unexpected v2 vocab size: {v2_vocab_size} "
                f"(expected {self.V2_VOCAB_SIZE})"
            )

        # Get v3 word embeddings
        v3_word_emb = v3_embeddings.word_embeddings.weight

        # Verify v3 has correct size
        if v3_word_emb.shape[0] != self.V3_VOCAB_SIZE:
            raise ValueError(
                f"v3 vocab size mismatch: {v3_word_emb.shape[0]} "
                f"(expected {self.V3_VOCAB_SIZE})"
            )

        # Copy v2 vocab embeddings to v3 (first 50,368 positions)
        with torch.no_grad():
            v3_embeddings.word_embeddings.weight[:v2_vocab_size] = v2_word_emb.clone()

        self.transfer_stats["vocab_transferred"] = v2_vocab_size * hidden_size
        self.transfer_stats["hub_slots_created"] = self.NUM_HUB_TOKENS

        logger.info(
            f"  Transferred {v2_vocab_size:,} vocab embeddings, "
            f"created {self.NUM_HUB_TOKENS} hub token slots"
        )

        return self.transfer_stats["vocab_transferred"]

    def transfer_position_embeddings(
        self,
        v3_embeddings: nn.Module,
    ) -> int:
        """
        Transfer position embeddings from v2 to v3.

        v3 supports 8192 positions vs v2's 8192, so this is a direct copy.
        If v3 has more positions, the extra are left randomly initialized.

        Args:
            v3_embeddings: v3 embedding module

        Returns:
            Number of parameters transferred
        """
        v2_emb_weights = self.v2_loader.get_embedding_weights()

        # Check for position embeddings (may not exist if using RoPE)
        if "position_embeddings.weight" not in v2_emb_weights:
            logger.info("  No position embeddings in v2 (using RoPE)")
            return 0

        v2_pos_emb = v2_emb_weights["position_embeddings.weight"]
        v2_max_pos, hidden_size = v2_pos_emb.shape

        # Get v3 position embeddings
        if not hasattr(v3_embeddings, "position_embeddings"):
            logger.info("  v3 uses RoPE, skipping position embedding transfer")
            return 0

        v3_pos_emb = v3_embeddings.position_embeddings.weight
        v3_max_pos = v3_pos_emb.shape[0]

        # Copy up to min of both sizes
        copy_length = min(v2_max_pos, v3_max_pos)

        with torch.no_grad():
            v3_embeddings.position_embeddings.weight[:copy_length] = \
                v2_pos_emb[:copy_length].clone()

        self.transfer_stats["position_embeddings_transferred"] = \
            copy_length * hidden_size

        logger.info(
            f"  Transferred {copy_length:,} position embeddings"
        )

        return self.transfer_stats["position_embeddings_transferred"]

    def transfer_layer_norm(
        self,
        v3_embeddings: nn.Module,
    ) -> int:
        """Transfer embedding LayerNorm from v2."""
        v2_emb_weights = self.v2_loader.get_embedding_weights()
        transferred = 0

        # LayerNorm weight
        if "LayerNorm.weight" in v2_emb_weights:
            with torch.no_grad():
                v3_embeddings.LayerNorm.weight.copy_(
                    v2_emb_weights["LayerNorm.weight"]
                )
            transferred += v2_emb_weights["LayerNorm.weight"].numel()

        # LayerNorm bias
        if "LayerNorm.bias" in v2_emb_weights:
            with torch.no_grad():
                v3_embeddings.LayerNorm.bias.copy_(
                    v2_emb_weights["LayerNorm.bias"]
                )
            transferred += v2_emb_weights["LayerNorm.bias"].numel()

        if transferred > 0:
            logger.info(f"  Transferred embedding LayerNorm: {transferred:,} params")

        return transferred

    def transfer_all(
        self,
        v3_embeddings: nn.Module,
    ) -> int:
        """
        Transfer all embedding components from v2 to v3.

        Args:
            v3_embeddings: v3 embedding module

        Returns:
            Total parameters transferred
        """
        total = 0

        print("\n📝 Transferring Embeddings (with Hub Token Slots)...")

        total += self.transfer_word_embeddings(v3_embeddings)
        total += self.transfer_position_embeddings(v3_embeddings)
        total += self.transfer_layer_norm(v3_embeddings)

        print(f"\n✓ Embedding transfer complete: {total:,} params")

        return total


def transfer_embeddings(
    v3_model: nn.Module,
    v2_checkpoint_path: str,
) -> int:
    """
    Transfer embeddings from v2 to v3 with hub token slots.

    Args:
        v3_model: Target v3 model
        v2_checkpoint_path: Path to v2 checkpoint

    Returns:
        Number of parameters transferred
    """
    loader = V2CheckpointLoader(v2_checkpoint_path)
    transfer = EmbeddingTransfer(loader)

    embeddings = v3_model.embeddings if hasattr(v3_model, "embeddings") else v3_model

    return transfer.transfer_all(embeddings)
```

**Acceptance Criteria:**

- [ ] Transfers v2 vocab embeddings (50,368 tokens) to v3
- [ ] Creates 4 hub token slots at positions 50368-50371
- [ ] Transfers position embeddings if present (handles RoPE)
- [ ] Transfers embedding LayerNorm weights
- [ ] Reports transfer statistics
- [ ] Hub token slots left uninitialized (for Issue 4.1.5)

**Tests:** `tests/v3/test_initialization_v3.py::test_embedding_transfer`

---

#### Issue 4.1.5: Implement Hub Token Semantic Initialization

**File:** `src/modeling_studio/models/initialization_v3.py` (extend)
**Effort:** 5 hours
**Dependencies:** Issues 4.1.4, 1.2.2 (hub semantic init)

**Description:**
Initialize hub token embeddings with semantic meaning by averaging embeddings of related tokens. This gives each hub a meaningful starting point that reflects its intended capability.

**Implementation:**

```python
# Add to initialization_v3.py

from transformers import AutoTokenizer
from typing import List


class HubTokenSemanticInitializer:
    """
    Initialize hub token embeddings with semantic meaning.

    Strategy: Average embeddings of semantically related tokens

    [EMO] ← avg("emotion", "feeling", "mood", "sentiment", "affect")
    [MEM] ← avg("memory", "remember", "recall", "history", "context")
    [REL] ← avg("relation", "relationship", "connection", "link", "between")
    [TASK] ← avg("task", "intent", "action", "goal", "purpose")
    """

    # Seed tokens for each hub
    HUB_SEED_TOKENS = {
        "[EMO]": [
            "emotion", "feeling", "mood", "sentiment", "affect",
            "happy", "sad", "angry", "fear", "joy", "surprise",
        ],
        "[MEM]": [
            "memory", "remember", "recall", "history", "context",
            "past", "experience", "store", "retrieve", "knowledge",
        ],
        "[REL]": [
            "relation", "relationship", "connection", "link", "between",
            "entail", "contradict", "similar", "compare", "associate",
        ],
        "[TASK]": [
            "task", "intent", "action", "goal", "purpose",
            "do", "request", "question", "command", "want",
        ],
    }

    # Hub token vocab positions
    HUB_POSITIONS = {
        "[EMO]": 50368,
        "[MEM]": 50369,
        "[REL]": 50370,
        "[TASK]": 50371,
    }

    def __init__(
        self,
        tokenizer_name: str = "answerdotai/ModernBERT-base",
        fallback_std: float = 0.02,
    ):
        """
        Args:
            tokenizer_name: HuggingFace tokenizer to use for token IDs
            fallback_std: Std for random init if seed tokens not found
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.fallback_std = fallback_std
        self.init_stats = {}

    def get_seed_token_ids(self, hub_name: str) -> List[int]:
        """
        Get token IDs for seed tokens.

        Returns only tokens that exist in vocabulary and are single-token.
        """
        seed_tokens = self.HUB_SEED_TOKENS.get(hub_name, [])
        valid_ids = []

        for token in seed_tokens:
            # Tokenize and check if single token
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(token_ids) == 1:
                valid_ids.append(token_ids[0])

        return valid_ids

    def initialize_hub_token(
        self,
        word_embeddings: nn.Embedding,
        hub_name: str,
    ) -> torch.Tensor:
        """
        Initialize a single hub token embedding.

        Args:
            word_embeddings: Full embedding matrix
            hub_name: Hub token name (e.g., "[EMO]")

        Returns:
            Initialized embedding vector
        """
        seed_ids = self.get_seed_token_ids(hub_name)

        if len(seed_ids) == 0:
            # Fallback to random init with small std
            logger.warning(
                f"  {hub_name}: No valid seed tokens, using random init"
            )
            hidden_size = word_embeddings.weight.shape[1]
            init_emb = torch.randn(hidden_size) * self.fallback_std
            self.init_stats[hub_name] = {"method": "random", "seeds": 0}
            return init_emb

        # Average seed token embeddings
        with torch.no_grad():
            seed_embeddings = word_embeddings.weight[seed_ids]
            avg_embedding = seed_embeddings.mean(dim=0)

        self.init_stats[hub_name] = {
            "method": "semantic_avg",
            "seeds": len(seed_ids),
            "seed_tokens": [self.tokenizer.decode([sid]) for sid in seed_ids[:5]],
        }

        logger.info(
            f"  {hub_name}: Initialized from {len(seed_ids)} seed tokens"
        )

        return avg_embedding

    def initialize_all_hubs(
        self,
        v3_embeddings: nn.Module,
    ) -> None:
        """
        Initialize all 4 hub token embeddings.

        Args:
            v3_embeddings: v3 embedding module with word_embeddings
        """
        print("\n🎯 Initializing Hub Token Embeddings (Semantic)...")

        word_emb = v3_embeddings.word_embeddings

        for hub_name, hub_position in self.HUB_POSITIONS.items():
            init_emb = self.initialize_hub_token(word_emb, hub_name)

            with torch.no_grad():
                word_emb.weight[hub_position] = init_emb

        self._print_summary()

    def _print_summary(self) -> None:
        """Print initialization summary."""
        print("\n✓ Hub Token Initialization Summary:")
        print("-" * 50)

        for hub, stats in self.init_stats.items():
            method = stats["method"]
            if method == "semantic_avg":
                seeds = stats["seeds"]
                examples = stats.get("seed_tokens", [])[:3]
                print(f"  {hub}: {method} ({seeds} seeds: {', '.join(examples)}...)")
            else:
                print(f"  {hub}: {method}")

        print("-" * 50)


def initialize_hub_tokens_semantic(
    v3_model: nn.Module,
    tokenizer_name: str = "answerdotai/ModernBERT-base",
) -> None:
    """
    Initialize hub token embeddings with semantic meaning.

    Args:
        v3_model: Target v3 model
        tokenizer_name: HuggingFace tokenizer for seed token lookup
    """
    initializer = HubTokenSemanticInitializer(tokenizer_name)

    embeddings = v3_model.embeddings if hasattr(v3_model, "embeddings") else v3_model

    initializer.initialize_all_hubs(embeddings)


# ═══════════════════════════════════════════════════════════════════════════
# Main Initialization Function
# ═══════════════════════════════════════════════════════════════════════════

def initialize_from_v2(
    v3_model: nn.Module,
    v2_checkpoint_path: str,
    add_clone_noise: bool = True,
    clone_noise_std: float = 0.01,
    tokenizer_name: str = "answerdotai/ModernBERT-base",
) -> WeightTransferStats:
    """
    Complete initialization of v3 model from v2 checkpoint.

    Steps:
        1. Load and validate v2 checkpoint
        2. Copy layers 1-22 directly
        3. Clone layers 15-20 to layers 23-28
        4. Transfer embeddings with hub token slots
        5. Initialize hub tokens semantically

    Args:
        v3_model: Target v3 model
        v2_checkpoint_path: Path to v2 checkpoint
        add_clone_noise: Add noise to cloned layers
        clone_noise_std: Std of clone noise
        tokenizer_name: Tokenizer for hub semantic init

    Returns:
        WeightTransferStats with transfer details
    """
    print("\n" + "=" * 70)
    print("🚀 ModernBERT v2 → v3 Weight Transfer")
    print("=" * 70)

    # Step 1: Load and validate
    loader = V2CheckpointLoader(v2_checkpoint_path)
    is_valid, issues = loader.validate()
    if not is_valid:
        print(f"⚠️  Checkpoint issues: {issues}")
    loader.print_summary()

    # Step 2: Copy layers 1-22
    copier = LayerCopier(loader)
    encoder = v3_model.encoder if hasattr(v3_model, "encoder") else v3_model
    direct_copied = copier.copy_layers_1_to_22(encoder)

    # Step 3: Clone layers 23-28
    cloner = LayerCloner(loader, add_noise=add_clone_noise, noise_std=clone_noise_std)
    cloned = cloner.clone_layers_23_to_28(encoder)

    # Step 4: Transfer embeddings
    embeddings = v3_model.embeddings if hasattr(v3_model, "embeddings") else v3_model
    emb_transfer = EmbeddingTransfer(loader)
    emb_transferred = emb_transfer.transfer_all(embeddings)

    # Step 5: Initialize hub tokens
    hub_init = HubTokenSemanticInitializer(tokenizer_name)
    hub_init.initialize_all_hubs(embeddings)

    # Create stats
    stats = WeightTransferStats(
        total_params=v3_model.num_parameters if hasattr(v3_model, "num_parameters")
                     else sum(p.numel() for p in v3_model.parameters()),
        transferred_params=direct_copied + emb_transferred,
        initialized_params=cloned,
        skipped_params=0,
        layer_mapping=LayerCloner.CLONE_MAPPING,
    )

    print("\n" + "=" * 70)
    print("✅ Weight Transfer Complete!")
    print("=" * 70)
    print(f"  Total v3 params: {stats.total_params:,}")
    print(f"  Direct transferred: {stats.transferred_params:,}")
    print(f"  Cloned (new layers): {stats.initialized_params:,}")
    print("=" * 70)

    return stats
```

**Acceptance Criteria:**

- [ ] `[EMO]` initialized from emotion-related tokens
- [ ] `[MEM]` initialized from memory-related tokens
- [ ] `[REL]` initialized from relation-related tokens
- [ ] `[TASK]` initialized from task/intent-related tokens
- [ ] Falls back to random init if seed tokens not found
- [ ] Uses tokenizer to find valid single-token seeds
- [ ] `initialize_from_v2()` orchestrates complete transfer

**Tests:** `tests/v3/test_initialization_v3.py::test_hub_semantic_init`

---

### Epic 4.2: Verification

#### Issue 4.2.1: Implement Function Preserving Verification

**File:** `src/modeling_studio/models/verification_v3.py`
**Effort:** 5 hours
**Dependencies:** Issues 4.1.1-4.1.5

**Description:**
Implement verification tests to ensure v3 produces identical outputs to v2 for the first 22 layers when processing the same input. This validates that the "function preserving growth" property holds.

**Implementation:**

```python
# src/modeling_studio/models/verification_v3.py

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Results from function preserving verification."""
    passed: bool
    max_diff: float
    mean_diff: float
    layer_diffs: Dict[int, float]
    embedding_diff: float
    failed_layers: List[int]
    message: str


@dataclass
class LayerComparisonResult:
    """Comparison result for a single layer."""
    layer_idx: int
    v2_norm: float
    v3_norm: float
    diff_norm: float
    relative_diff: float
    passed: bool


class FunctionPreservingVerifier:
    """
    Verifies that v3 model preserves v2 function for layers 1-22.

    Function Preserving Property:
        For layers L1-L22, given identical input embeddings,
        the layer outputs should be identical (within numerical precision).

    Tolerance Levels:
        - Strict: max_diff < 1e-5 (bit-exact on same hardware)
        - Normal: max_diff < 1e-4 (accounts for precision differences)
        - Relaxed: max_diff < 1e-3 (allows minor floating point drift)
    """

    TOLERANCE_STRICT = 1e-5
    TOLERANCE_NORMAL = 1e-4
    TOLERANCE_RELAXED = 1e-3

    def __init__(
        self,
        v2_model: nn.Module,
        v3_model: nn.Module,
        tolerance: float = TOLERANCE_NORMAL,
    ):
        """
        Args:
            v2_model: Original v2 model (22 layers)
            v3_model: Initialized v3 model (28 layers)
            tolerance: Maximum allowed difference
        """
        self.v2_model = v2_model
        self.v3_model = v3_model
        self.tolerance = tolerance

        # Put both models in eval mode
        self.v2_model.eval()
        self.v3_model.eval()

    def verify_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[bool, float]:
        """
        Verify embedding layer produces identical output.

        Note: Only compares non-hub positions (0 and 5+),
        since hub tokens are new in v3.

        Returns:
            (passed, max_diff)
        """
        with torch.no_grad():
            # Get v2 embeddings
            v2_emb = self.v2_model.embeddings(input_ids)

            # Get v3 embeddings
            v3_emb = self.v3_model.embeddings(input_ids)

            # Compare only shared token positions
            # v3 has hub tokens at positions 1-4, so compare:
            # - Position 0 (CLS)
            # - Positions 5+ (text tokens)
            v2_cls = v2_emb[:, 0, :]
            v3_cls = v3_emb[:, 0, :]

            v2_text = v2_emb[:, 1:, :]  # v2 text starts at position 1
            v3_text = v3_emb[:, 5:, :]  # v3 text starts at position 5

            cls_diff = (v2_cls - v3_cls).abs().max().item()

            # For text comparison, need to align lengths
            min_len = min(v2_text.shape[1], v3_text.shape[1])
            text_diff = (v2_text[:, :min_len] - v3_text[:, :min_len]).abs().max().item()

            max_diff = max(cls_diff, text_diff)
            passed = max_diff < self.tolerance

            return passed, max_diff

    def verify_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> LayerComparisonResult:
        """
        Verify a single layer produces identical output.

        Args:
            layer_idx: Layer index (0-21 for shared layers)
            hidden_states: Input hidden states
            attention_mask: Attention mask

        Returns:
            LayerComparisonResult
        """
        with torch.no_grad():
            v2_layer = self.v2_model.encoder.layers[layer_idx]
            v3_layer = self.v3_model.encoder.layers[layer_idx]

            # Forward through both layers
            v2_output = v2_layer(hidden_states, attention_mask)
            v3_output = v3_layer(hidden_states, attention_mask)

            # Handle tuple outputs (hidden_states, attention_weights)
            if isinstance(v2_output, tuple):
                v2_output = v2_output[0]
            if isinstance(v3_output, tuple):
                v3_output = v3_output[0]

            # Compute differences
            v2_norm = v2_output.norm().item()
            v3_norm = v3_output.norm().item()
            diff = (v2_output - v3_output).abs()
            diff_norm = diff.max().item()

            relative_diff = diff_norm / (v2_norm + 1e-8)
            passed = diff_norm < self.tolerance

            return LayerComparisonResult(
                layer_idx=layer_idx,
                v2_norm=v2_norm,
                v3_norm=v3_norm,
                diff_norm=diff_norm,
                relative_diff=relative_diff,
                passed=passed,
            )

    def verify_all_layers(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> VerificationResult:
        """
        Verify all 22 shared layers produce identical outputs.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            VerificationResult with detailed comparison
        """
        print("\n" + "=" * 70)
        print("🔍 Function Preserving Verification")
        print("=" * 70)

        failed_layers = []
        layer_diffs = {}

        with torch.no_grad():
            # Step 1: Verify embeddings
            emb_passed, emb_diff = self.verify_embeddings(input_ids, attention_mask)

            if not emb_passed:
                print(f"❌ Embedding verification failed: diff={emb_diff:.2e}")
            else:
                print(f"✓ Embeddings: diff={emb_diff:.2e}")

            # Get initial hidden states from v2
            v2_hidden = self.v2_model.embeddings(input_ids)
            v3_hidden = self.v3_model.embeddings(input_ids)

            # Step 2: Verify each layer
            print("\nLayer-by-layer verification:")
            print("-" * 50)

            for layer_idx in range(22):  # First 22 layers
                result = self.verify_layer(layer_idx, v2_hidden, attention_mask)
                layer_diffs[layer_idx] = result.diff_norm

                status = "✓" if result.passed else "❌"
                print(
                    f"  L{layer_idx:2d}: {status} "
                    f"diff={result.diff_norm:.2e} "
                    f"rel={result.relative_diff:.2e}"
                )

                if not result.passed:
                    failed_layers.append(layer_idx)

                # Propagate hidden states
                v2_output = self.v2_model.encoder.layers[layer_idx](
                    v2_hidden, attention_mask
                )
                v3_output = self.v3_model.encoder.layers[layer_idx](
                    v3_hidden, attention_mask
                )

                v2_hidden = v2_output[0] if isinstance(v2_output, tuple) else v2_output
                v3_hidden = v3_output[0] if isinstance(v3_output, tuple) else v3_output

        # Compute summary statistics
        max_diff = max(layer_diffs.values()) if layer_diffs else 0.0
        mean_diff = sum(layer_diffs.values()) / len(layer_diffs) if layer_diffs else 0.0
        passed = len(failed_layers) == 0 and emb_passed

        # Create result
        result = VerificationResult(
            passed=passed,
            max_diff=max_diff,
            mean_diff=mean_diff,
            layer_diffs=layer_diffs,
            embedding_diff=emb_diff,
            failed_layers=failed_layers,
            message=self._create_message(passed, failed_layers, max_diff),
        )

        # Print summary
        print("-" * 50)
        if passed:
            print(f"\n✅ PASSED: All layers within tolerance ({self.tolerance:.0e})")
        else:
            print(f"\n❌ FAILED: {len(failed_layers)} layers exceeded tolerance")
            print(f"   Failed layers: {failed_layers}")

        print(f"   Max diff: {max_diff:.2e}")
        print(f"   Mean diff: {mean_diff:.2e}")
        print("=" * 70)

        return result

    def _create_message(
        self,
        passed: bool,
        failed_layers: List[int],
        max_diff: float,
    ) -> str:
        """Create human-readable result message."""
        if passed:
            return f"Function preserving property verified (max_diff={max_diff:.2e})"
        else:
            return (
                f"Function preserving property VIOLATED: "
                f"{len(failed_layers)} layers failed, max_diff={max_diff:.2e}"
            )


def verify_function_preserving(
    v2_model: nn.Module,
    v3_model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tolerance: float = 1e-4,
) -> VerificationResult:
    """
    Verify v3 preserves v2 function for shared layers.

    Args:
        v2_model: Original v2 model
        v3_model: Initialized v3 model
        input_ids: Test input token IDs
        attention_mask: Test attention mask
        tolerance: Maximum allowed difference

    Returns:
        VerificationResult
    """
    verifier = FunctionPreservingVerifier(v2_model, v3_model, tolerance)
    return verifier.verify_all_layers(input_ids, attention_mask)
```

**Acceptance Criteria:**

- [ ] Compares v2 and v3 outputs layer-by-layer
- [ ] Handles hub token offset (v3 positions 1-4)
- [ ] Supports strict/normal/relaxed tolerance levels
- [ ] Reports per-layer differences
- [ ] Clear pass/fail result with message
- [ ] Works with both eval and train mode

**Tests:** `tests/v3/test_verification_v3.py::test_function_preserving`

---

#### Issue 4.2.2: Implement Layer Output Comparison Tests

**File:** `tests/v3/test_initialization_v3.py`
**Effort:** 4 hours
**Dependencies:** Issue 4.2.1

**Description:**
Implement comprehensive test suite for layer output comparison, including edge cases like different sequence lengths, batch sizes, and attention patterns.

**Implementation:**

```python
# tests/v3/test_initialization_v3.py

import pytest
import torch
import torch.nn as nn
from typing import Dict, Optional
from pathlib import Path

from modeling_studio.models.initialization_v3 import (
    V2CheckpointLoader,
    LayerCopier,
    LayerCloner,
    EmbeddingTransfer,
    HubTokenSemanticInitializer,
    initialize_from_v2,
    WeightTransferStats,
)
from modeling_studio.models.verification_v3 import (
    FunctionPreservingVerifier,
    VerificationResult,
    verify_function_preserving,
)
from modeling_studio.models.config_v3 import ModernBERTv3Config
from modeling_studio.models.modernbert_v3 import ModernBERTv3Ultra


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def v2_checkpoint_path() -> str:
    """Path to v2 checkpoint for testing."""
    # Use a test checkpoint or mock
    return "checkpoints/modernbert-multitask-v2/pytorch_model.bin"


@pytest.fixture
def v3_config() -> ModernBERTv3Config:
    """v3 configuration for testing."""
    return ModernBERTv3Config(
        hidden_size=768,
        num_layers=28,
        num_attention_heads=12,
        intermediate_size=3072,
        vocab_size=50372,  # v2 vocab + 4 hub tokens
        max_position_embeddings=8192,
    )


@pytest.fixture
def sample_input(v3_config: ModernBERTv3Config) -> Dict[str, torch.Tensor]:
    """Sample input for testing."""
    batch_size = 2
    seq_length = 128

    return {
        "input_ids": torch.randint(
            0, v3_config.vocab_size - 4,  # Avoid hub token IDs
            (batch_size, seq_length),
        ),
        "attention_mask": torch.ones(batch_size, seq_length, dtype=torch.long),
    }


# ═══════════════════════════════════════════════════════════════════════════
# V2 Checkpoint Loader Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestV2CheckpointLoader:
    """Tests for v2 checkpoint loading."""

    def test_load_checkpoint(self, v2_checkpoint_path: str):
        """Test basic checkpoint loading."""
        if not Path(v2_checkpoint_path).exists():
            pytest.skip("v2 checkpoint not available")

        loader = V2CheckpointLoader(v2_checkpoint_path)
        state_dict = loader.load()

        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0

    def test_get_info(self, v2_checkpoint_path: str):
        """Test checkpoint info extraction."""
        if not Path(v2_checkpoint_path).exists():
            pytest.skip("v2 checkpoint not available")

        loader = V2CheckpointLoader(v2_checkpoint_path)
        info = loader.get_info()

        assert info.num_layers == 22
        assert info.hidden_size == 768
        assert info.vocab_size >= 50000

    def test_validate(self, v2_checkpoint_path: str):
        """Test checkpoint validation."""
        if not Path(v2_checkpoint_path).exists():
            pytest.skip("v2 checkpoint not available")

        loader = V2CheckpointLoader(v2_checkpoint_path)
        is_valid, issues = loader.validate()

        assert is_valid, f"Validation failed: {issues}"

    def test_get_layer_weights(self, v2_checkpoint_path: str):
        """Test per-layer weight extraction."""
        if not Path(v2_checkpoint_path).exists():
            pytest.skip("v2 checkpoint not available")

        loader = V2CheckpointLoader(v2_checkpoint_path)

        for layer_idx in [0, 10, 21]:
            weights = loader.get_layer_weights(layer_idx)
            assert len(weights) > 0
            assert any("attention" in k for k in weights.keys())


# ═══════════════════════════════════════════════════════════════════════════
# Layer Copy Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestLayerCopy:
    """Tests for direct layer copying."""

    def test_layer_copy_preserves_weights(
        self,
        v2_checkpoint_path: str,
        v3_config: ModernBERTv3Config,
    ):
        """Test that copied layers have identical weights."""
        if not Path(v2_checkpoint_path).exists():
            pytest.skip("v2 checkpoint not available")

        # Create v3 model
        v3_model = ModernBERTv3Ultra(v3_config)

        # Copy layers
        loader = V2CheckpointLoader(v2_checkpoint_path)
        copier = LayerCopier(loader)
        copier.copy_layers_1_to_22(v3_model.encoder)

        # Verify weights match
        v2_weights = loader.get_layer_weights(0)
        v3_state = v3_model.encoder.layers[0].state_dict()

        for key in v2_weights:
            if key in v3_state:
                diff = (v2_weights[key] - v3_state[key]).abs().max()
                assert diff < 1e-6, f"Layer 0 weight {key} differs by {diff}"


# ═══════════════════════════════════════════════════════════════════════════
# Layer Clone Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestLayerClone:
    """Tests for layer cloning (L15-20 → L23-28)."""

    def test_clone_mapping_correct(self):
        """Test clone mapping is correct."""
        expected_mapping = {
            22: 14, 23: 15, 24: 16, 25: 17, 26: 18, 27: 19
        }
        assert LayerCloner.CLONE_MAPPING == expected_mapping

    def test_clone_with_noise(
        self,
        v2_checkpoint_path: str,
        v3_config: ModernBERTv3Config,
    ):
        """Test that noise breaks symmetry between cloned layers."""
        if not Path(v2_checkpoint_path).exists():
            pytest.skip("v2 checkpoint not available")

        v3_model = ModernBERTv3Ultra(v3_config)

        loader = V2CheckpointLoader(v2_checkpoint_path)
        cloner = LayerCloner(loader, add_noise=True, noise_std=0.01)
        cloner.clone_layers_23_to_28(v3_model.encoder)

        # Check that consecutive cloned layers are different
        l23_weights = v3_model.encoder.layers[22].state_dict()
        l24_weights = v3_model.encoder.layers[23].state_dict()

        for key in l23_weights:
            if "weight" in key:
                diff = (l23_weights[key] - l24_weights[key]).abs().max()
                assert diff > 0, f"Layers 23 and 24 should differ due to noise"


# ═══════════════════════════════════════════════════════════════════════════
# Embedding Transfer Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestEmbeddingTransfer:
    """Tests for embedding transfer with hub token slots."""

    def test_vocab_embeddings_transferred(
        self,
        v2_checkpoint_path: str,
        v3_config: ModernBERTv3Config,
    ):
        """Test that v2 vocab embeddings are transferred correctly."""
        if not Path(v2_checkpoint_path).exists():
            pytest.skip("v2 checkpoint not available")

        v3_model = ModernBERTv3Ultra(v3_config)

        loader = V2CheckpointLoader(v2_checkpoint_path)
        transfer = EmbeddingTransfer(loader)
        transfer.transfer_all(v3_model.embeddings)

        # Check first 100 vocab embeddings match
        v2_emb = loader.get_embedding_weights()["word_embeddings.weight"]
        v3_emb = v3_model.embeddings.word_embeddings.weight

        diff = (v2_emb[:100] - v3_emb[:100]).abs().max()
        assert diff < 1e-6, f"Vocab embeddings differ by {diff}"

    def test_hub_token_slots_created(
        self,
        v2_checkpoint_path: str,
        v3_config: ModernBERTv3Config,
    ):
        """Test that hub token slots exist at correct positions."""
        if not Path(v2_checkpoint_path).exists():
            pytest.skip("v2 checkpoint not available")

        v3_model = ModernBERTv3Ultra(v3_config)

        # Check v3 has 4 more embeddings than v2
        v3_vocab_size = v3_model.embeddings.word_embeddings.weight.shape[0]
        assert v3_vocab_size == 50372  # 50368 + 4 hub tokens


# ═══════════════════════════════════════════════════════════════════════════
# Hub Token Semantic Init Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestHubSemanticInit:
    """Tests for hub token semantic initialization."""

    def test_hub_tokens_not_zero(
        self,
        v3_config: ModernBERTv3Config,
    ):
        """Test that hub tokens are initialized (not zeros)."""
        v3_model = ModernBERTv3Ultra(v3_config)

        # Initialize hub tokens
        initializer = HubTokenSemanticInitializer()
        initializer.initialize_all_hubs(v3_model.embeddings)

        # Check hub positions are non-zero
        hub_positions = [50368, 50369, 50370, 50371]
        for pos in hub_positions:
            hub_emb = v3_model.embeddings.word_embeddings.weight[pos]
            assert hub_emb.abs().sum() > 0, f"Hub token at {pos} is zero"

    def test_hub_tokens_different(
        self,
        v3_config: ModernBERTv3Config,
    ):
        """Test that each hub token has unique embedding."""
        v3_model = ModernBERTv3Ultra(v3_config)

        initializer = HubTokenSemanticInitializer()
        initializer.initialize_all_hubs(v3_model.embeddings)

        # Compare hub embeddings pairwise
        hub_positions = [50368, 50369, 50370, 50371]
        hub_embs = [
            v3_model.embeddings.word_embeddings.weight[pos]
            for pos in hub_positions
        ]

        for i in range(len(hub_embs)):
            for j in range(i + 1, len(hub_embs)):
                diff = (hub_embs[i] - hub_embs[j]).abs().sum()
                assert diff > 0.1, f"Hub tokens {i} and {j} are too similar"


# ═══════════════════════════════════════════════════════════════════════════
# Function Preserving Verification Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFunctionPreserving:
    """Tests for function preserving verification."""

    @pytest.mark.slow
    def test_full_initialization_preserves_function(
        self,
        v2_checkpoint_path: str,
        v3_config: ModernBERTv3Config,
        sample_input: Dict[str, torch.Tensor],
    ):
        """Test that complete initialization preserves v2 function."""
        if not Path(v2_checkpoint_path).exists():
            pytest.skip("v2 checkpoint not available")

        # Load v2 model
        from modeling_studio.models import load_v2_model
        v2_model = load_v2_model(v2_checkpoint_path)

        # Create and initialize v3 model
        v3_model = ModernBERTv3Ultra(v3_config)
        initialize_from_v2(v3_model, v2_checkpoint_path)

        # Verify function preserving property
        result = verify_function_preserving(
            v2_model,
            v3_model,
            sample_input["input_ids"],
            sample_input["attention_mask"],
            tolerance=1e-4,
        )

        assert result.passed, result.message

    def test_verification_detects_differences(
        self,
        v3_config: ModernBERTv3Config,
        sample_input: Dict[str, torch.Tensor],
    ):
        """Test that verification catches non-preserved functions."""
        # Create two different models (not initialized from same source)
        v3_model_a = ModernBERTv3Ultra(v3_config)
        v3_model_b = ModernBERTv3Ultra(v3_config)

        # These should NOT pass verification
        verifier = FunctionPreservingVerifier(
            v3_model_a, v3_model_b, tolerance=1e-4
        )

        # Get embeddings and compare first layer
        with torch.no_grad():
            hidden_a = v3_model_a.embeddings(sample_input["input_ids"])
            result = verifier.verify_layer(
                0, hidden_a, sample_input["attention_mask"]
            )

        # Random init should differ significantly
        assert not result.passed or result.diff_norm > 1e-4


# ═══════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFullInitialization:
    """Integration tests for complete initialization pipeline."""

    @pytest.mark.slow
    def test_initialize_from_v2_returns_stats(
        self,
        v2_checkpoint_path: str,
        v3_config: ModernBERTv3Config,
    ):
        """Test that initialization returns proper statistics."""
        if not Path(v2_checkpoint_path).exists():
            pytest.skip("v2 checkpoint not available")

        v3_model = ModernBERTv3Ultra(v3_config)
        stats = initialize_from_v2(v3_model, v2_checkpoint_path)

        assert isinstance(stats, WeightTransferStats)
        assert stats.transferred_params > 0
        assert stats.initialized_params > 0
        assert len(stats.layer_mapping) == 6  # 6 cloned layers
```

**Acceptance Criteria:**

- [ ] Tests for V2CheckpointLoader (load, info, validate)
- [ ] Tests for LayerCopier (direct copy verification)
- [ ] Tests for LayerCloner (clone mapping, noise)
- [ ] Tests for EmbeddingTransfer (vocab, hub slots)
- [ ] Tests for HubTokenSemanticInitializer (non-zero, unique)
- [ ] Tests for FunctionPreservingVerifier (integration)
- [ ] Proper pytest fixtures and marks (@pytest.mark.slow)

**Tests:** `tests/v3/test_initialization_v3.py`

---

#### Issue 4.2.3: Create Initialization Script

**File:** `scripts/initialize_v3_from_v2.py`
**Effort:** 3 hours
**Dependencies:** Issues 4.1.1-4.1.5, 4.2.1

**Description:**
Create a command-line script that initializes a v3 model from a v2 checkpoint, runs verification, and saves the initialized model.

**Implementation:**

```python
#!/usr/bin/env python3
"""
Initialize ModernBERT v3 from v2 checkpoint.

Usage:
    python scripts/initialize_v3_from_v2.py \
        --v2-checkpoint checkpoints/modernbert-v2/pytorch_model.bin \
        --output-dir checkpoints/modernbert-v3-init \
        --verify

This script:
    1. Loads v2 checkpoint (22 layers)
    2. Creates v3 model (28 layers)
    3. Copies layers 1-22 directly
    4. Clones layers 15-20 to 23-28
    5. Transfers embeddings with hub token slots
    6. Initializes hub tokens semantically
    7. Verifies function preserving property
    8. Saves initialized v3 model
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modeling_studio.models.config_v3 import ModernBERTv3Config
from modeling_studio.models.modernbert_v3 import ModernBERTv3Ultra
from modeling_studio.models.initialization_v3 import (
    initialize_from_v2,
    V2CheckpointLoader,
    WeightTransferStats,
)
from modeling_studio.models.verification_v3 import (
    verify_function_preserving,
    VerificationResult,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Initialize ModernBERT v3 from v2 checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--v2-checkpoint",
        type=str,
        required=True,
        help="Path to v2 checkpoint file",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for initialized v3 model",
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run function preserving verification after initialization",
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Tolerance for verification (default: 1e-4)",
    )

    parser.add_argument(
        "--add-clone-noise",
        action="store_true",
        default=True,
        help="Add noise to cloned layers (default: True)",
    )

    parser.add_argument(
        "--clone-noise-std",
        type=float,
        default=0.01,
        help="Std of noise for cloned layers (default: 0.01)",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="answerdotai/ModernBERT-base",
        help="Tokenizer for hub semantic initialization",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for verification (cpu or cuda)",
    )

    return parser.parse_args()


def create_v3_config(v2_loader: V2CheckpointLoader) -> ModernBERTv3Config:
    """Create v3 config based on v2 checkpoint info."""
    info = v2_loader.get_info()

    return ModernBERTv3Config(
        hidden_size=info.hidden_size,
        num_layers=28,  # v3 has 28 layers
        num_attention_heads=12,
        intermediate_size=info.hidden_size * 4,
        vocab_size=info.vocab_size + 4,  # +4 hub tokens
        max_position_embeddings=8192,
    )


def run_verification(
    v2_checkpoint_path: str,
    v3_model: torch.nn.Module,
    tolerance: float,
    device: str,
) -> VerificationResult:
    """Run function preserving verification."""
    logger.info("Running function preserving verification...")

    # Load v2 model for comparison
    from modeling_studio.models import load_v2_model
    v2_model = load_v2_model(v2_checkpoint_path)

    # Move to device
    v2_model = v2_model.to(device)
    v3_model = v3_model.to(device)

    # Create test input
    batch_size = 2
    seq_length = 128

    input_ids = torch.randint(0, 50000, (batch_size, seq_length)).to(device)
    attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long).to(device)

    # Run verification
    result = verify_function_preserving(
        v2_model,
        v3_model,
        input_ids,
        attention_mask,
        tolerance=tolerance,
    )

    return result


def save_model(
    model: torch.nn.Module,
    config: ModernBERTv3Config,
    output_dir: Path,
    stats: WeightTransferStats,
    verification_result: VerificationResult = None,
) -> None:
    """Save initialized model and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    model_path = output_dir / "pytorch_model.bin"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model weights to {model_path}")

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config.__dict__, f, indent=2)
    logger.info(f"Saved config to {config_path}")

    # Save initialization metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "source": "v2_checkpoint",
        "transfer_stats": {
            "total_params": stats.total_params,
            "transferred_params": stats.transferred_params,
            "initialized_params": stats.initialized_params,
            "layer_mapping": stats.layer_mapping,
        },
    }

    if verification_result:
        metadata["verification"] = {
            "passed": verification_result.passed,
            "max_diff": verification_result.max_diff,
            "mean_diff": verification_result.mean_diff,
            "failed_layers": verification_result.failed_layers,
        }

    metadata_path = output_dir / "initialization_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")


def main() -> int:
    """Main entry point."""
    args = parse_args()

    print("\n" + "=" * 70)
    print("🚀 ModernBERT v3 Initialization from v2")
    print("=" * 70)

    # Validate inputs
    v2_path = Path(args.v2_checkpoint)
    if not v2_path.exists():
        logger.error(f"v2 checkpoint not found: {v2_path}")
        return 1

    output_dir = Path(args.output_dir)

    # Step 1: Load v2 checkpoint info
    logger.info(f"Loading v2 checkpoint from {v2_path}")
    v2_loader = V2CheckpointLoader(str(v2_path))
    v2_loader.print_summary()

    # Step 2: Create v3 model
    logger.info("Creating v3 model...")
    config = create_v3_config(v2_loader)
    v3_model = ModernBERTv3Ultra(config)

    logger.info(f"  v3 layers: {config.num_layers}")
    logger.info(f"  v3 vocab size: {config.vocab_size}")
    logger.info(f"  v3 params: {v3_model.num_parameters:,}")

    # Step 3: Initialize from v2
    stats = initialize_from_v2(
        v3_model,
        str(v2_path),
        add_clone_noise=args.add_clone_noise,
        clone_noise_std=args.clone_noise_std,
        tokenizer_name=args.tokenizer,
    )

    # Step 4: Verification (optional)
    verification_result = None
    if args.verify:
        try:
            verification_result = run_verification(
                str(v2_path),
                v3_model,
                args.tolerance,
                args.device,
            )

            if not verification_result.passed:
                logger.warning("⚠️  Verification FAILED - proceeding anyway")
        except Exception as e:
            logger.error(f"Verification failed with error: {e}")
            logger.info("Continuing without verification...")

    # Step 5: Save model
    logger.info(f"Saving initialized v3 model to {output_dir}")
    save_model(
        v3_model,
        config,
        output_dir,
        stats,
        verification_result,
    )

    # Summary
    print("\n" + "=" * 70)
    print("✅ Initialization Complete!")
    print("=" * 70)
    print(f"  Output: {output_dir}")
    print(f"  Total params: {stats.total_params:,}")
    print(f"  Transferred: {stats.transferred_params:,}")
    print(f"  Cloned: {stats.initialized_params:,}")
    if verification_result:
        status = "PASSED ✓" if verification_result.passed else "FAILED ✗"
        print(f"  Verification: {status}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Usage:**

```bash
# Basic initialization
python scripts/initialize_v3_from_v2.py \
    --v2-checkpoint checkpoints/modernbert-v2/pytorch_model.bin \
    --output-dir checkpoints/modernbert-v3-init

# With verification
python scripts/initialize_v3_from_v2.py \
    --v2-checkpoint checkpoints/modernbert-v2/pytorch_model.bin \
    --output-dir checkpoints/modernbert-v3-init \
    --verify \
    --device cuda

# With custom noise settings
python scripts/initialize_v3_from_v2.py \
    --v2-checkpoint checkpoints/modernbert-v2/pytorch_model.bin \
    --output-dir checkpoints/modernbert-v3-init \
    --add-clone-noise \
    --clone-noise-std 0.005 \
    --verify
```

**Acceptance Criteria:**

- [ ] CLI with all necessary arguments
- [ ] Validates v2 checkpoint exists
- [ ] Creates v3 config from v2 info
- [ ] Runs complete initialization pipeline
- [ ] Optional verification with configurable tolerance
- [ ] Saves model weights, config, and metadata
- [ ] Clear progress output and summary
- [ ] Proper error handling and exit codes

**Tests:** `tests/v3/test_scripts.py::test_initialize_v3_script`

---

## 🏁 Milestone 5: v3 Training Infrastructure

**Goal:** Implement phase-based trainer with layer freezing and replay

### Epic 5.1: v3 Trainer

#### Issue 5.1.1: Implement Layer Freezing by Band

**File:** `src/modeling_studio/trainers/freezing_v3.py`
**Effort:** 4 hours
**Dependencies:** Issue 3.1.2 (ModernBERTEncoderV3)

**Description:**
Implement layer freezing utilities that freeze/unfreeze layers by band for phase-based training. This is critical for preserving v2 capabilities while training new layers.

**Implementation:**

```python
# src/modeling_studio/trainers/freezing_v3.py

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Set, Literal
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LayerBand(Enum):
    """Layer bands in v3 architecture."""
    FOUNDATION = "foundation"  # L1-6: window=64
    CORE = "core"              # L7-18: window=128
    FEEDER = "feeder"          # L19-22: window=256
    FAMILY = "family"          # L23-28: window=512


# Layer indices for each band (0-indexed)
LAYER_BANDS: Dict[LayerBand, List[int]] = {
    LayerBand.FOUNDATION: list(range(0, 6)),    # L1-6
    LayerBand.CORE: list(range(6, 18)),         # L7-18
    LayerBand.FEEDER: list(range(18, 22)),      # L19-22
    LayerBand.FAMILY: list(range(22, 28)),      # L23-28
}


class TrainingPhase(Enum):
    """Training phases for v3."""
    PHASE_0_5 = "phase_0.5"    # Healing: L19-28 trainable
    PHASE_1 = "phase_1"        # Multi-task: L19-28 trainable
    PHASE_2 = "phase_2"        # Full fine-tune: all trainable
    INFERENCE = "inference"    # All frozen


# Which bands are trainable in each phase
PHASE_TRAINABLE_BANDS: Dict[TrainingPhase, List[LayerBand]] = {
    TrainingPhase.PHASE_0_5: [LayerBand.FEEDER, LayerBand.FAMILY],
    TrainingPhase.PHASE_1: [LayerBand.FEEDER, LayerBand.FAMILY],
    TrainingPhase.PHASE_2: [LayerBand.FOUNDATION, LayerBand.CORE,
                            LayerBand.FEEDER, LayerBand.FAMILY],
    TrainingPhase.INFERENCE: [],
}


class LayerFreezer:
    """
    Manages layer freezing for phase-based training.

    Freeze Strategy:
        Phase 0.5 (Healing):
            - Frozen: L1-18 (Foundation + Core)
            - Trainable: L19-28 (Feeder + Family)
            - Purpose: Heal cloned layers without forgetting

        Phase 1 (Multi-task):
            - Frozen: L1-18
            - Trainable: L19-28 + task heads
            - Purpose: Learn FamilyOS tasks

        Phase 2 (Full fine-tune):
            - All trainable with low LR on L1-18
            - Purpose: Optional final polish
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: ModernBERTv3Ultra or similar with encoder.layers
        """
        self.model = model
        self.encoder = model.encoder if hasattr(model, "encoder") else model
        self._frozen_layers: Set[int] = set()
        self._frozen_components: Set[str] = set()

    def get_layer(self, layer_idx: int) -> nn.Module:
        """Get layer by index."""
        return self.encoder.layers[layer_idx]

    def freeze_layer(self, layer_idx: int) -> None:
        """Freeze a single layer."""
        layer = self.get_layer(layer_idx)
        for param in layer.parameters():
            param.requires_grad = False
        self._frozen_layers.add(layer_idx)

    def unfreeze_layer(self, layer_idx: int) -> None:
        """Unfreeze a single layer."""
        layer = self.get_layer(layer_idx)
        for param in layer.parameters():
            param.requires_grad = True
        self._frozen_layers.discard(layer_idx)

    def freeze_band(self, band: LayerBand) -> int:
        """
        Freeze all layers in a band.

        Returns:
            Number of parameters frozen
        """
        frozen_params = 0
        for layer_idx in LAYER_BANDS[band]:
            layer = self.get_layer(layer_idx)
            for param in layer.parameters():
                if param.requires_grad:
                    param.requires_grad = False
                    frozen_params += param.numel()
            self._frozen_layers.add(layer_idx)

        logger.info(f"Froze {band.value} band (L{min(LAYER_BANDS[band])+1}-{max(LAYER_BANDS[band])+1}): {frozen_params:,} params")
        return frozen_params

    def unfreeze_band(self, band: LayerBand) -> int:
        """
        Unfreeze all layers in a band.

        Returns:
            Number of parameters unfrozen
        """
        unfrozen_params = 0
        for layer_idx in LAYER_BANDS[band]:
            layer = self.get_layer(layer_idx)
            for param in layer.parameters():
                if not param.requires_grad:
                    param.requires_grad = True
                    unfrozen_params += param.numel()
            self._frozen_layers.discard(layer_idx)

        logger.info(f"Unfroze {band.value} band: {unfrozen_params:,} params")
        return unfrozen_params

    def freeze_embeddings(self) -> int:
        """Freeze embedding layer."""
        frozen = 0
        embeddings = self.model.embeddings if hasattr(self.model, "embeddings") else None

        if embeddings:
            for param in embeddings.parameters():
                if param.requires_grad:
                    param.requires_grad = False
                    frozen += param.numel()
            self._frozen_components.add("embeddings")

        logger.info(f"Froze embeddings: {frozen:,} params")
        return frozen

    def unfreeze_embeddings(self) -> int:
        """Unfreeze embedding layer."""
        unfrozen = 0
        embeddings = self.model.embeddings if hasattr(self.model, "embeddings") else None

        if embeddings:
            for param in embeddings.parameters():
                if not param.requires_grad:
                    param.requires_grad = True
                    unfrozen += param.numel()
            self._frozen_components.discard("embeddings")

        return unfrozen

    def freeze_hub_tokens(self, freeze: bool = True) -> None:
        """
        Freeze or unfreeze hub token embeddings only.

        Hub tokens are at positions 50368-50371 in word_embeddings.
        """
        embeddings = self.model.embeddings if hasattr(self.model, "embeddings") else None
        if not embeddings:
            return

        word_emb = embeddings.word_embeddings.weight
        hub_start = 50368  # First hub token position

        # Note: Can't selectively freeze parts of a parameter
        # This is handled via gradient masking instead (Issue 5.1.5)
        logger.info(f"Hub token freezing handled via gradient masking")

    def configure_for_phase(self, phase: TrainingPhase) -> Dict[str, int]:
        """
        Configure model freezing for a training phase.

        Args:
            phase: Training phase

        Returns:
            Stats about frozen/trainable params
        """
        print(f"\n🔧 Configuring model for {phase.value}...")

        trainable_bands = PHASE_TRAINABLE_BANDS[phase]

        # Freeze all bands first
        for band in LayerBand:
            self.freeze_band(band)

        # Unfreeze trainable bands
        for band in trainable_bands:
            self.unfreeze_band(band)

        # Always freeze embeddings in Phase 0.5/1
        if phase in [TrainingPhase.PHASE_0_5, TrainingPhase.PHASE_1]:
            self.freeze_embeddings()
        else:
            self.unfreeze_embeddings()

        # Compute stats
        stats = self.get_freeze_stats()
        self._print_freeze_summary(phase, stats)

        return stats

    def get_freeze_stats(self) -> Dict[str, int]:
        """Get freezing statistics."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "frozen_layers": len(self._frozen_layers),
            "trainable_layers": 28 - len(self._frozen_layers),
        }

    def _print_freeze_summary(self, phase: TrainingPhase, stats: Dict[str, int]) -> None:
        """Print freeze configuration summary."""
        print("\n" + "-" * 50)
        print(f"Phase: {phase.value}")
        print(f"  Frozen layers: {sorted(self._frozen_layers)}")
        print(f"  Trainable layers: {sorted(set(range(28)) - self._frozen_layers)}")
        print(f"  Total params: {stats['total_params']:,}")
        print(f"  Trainable: {stats['trainable_params']:,} ({100*stats['trainable_params']/stats['total_params']:.1f}%)")
        print(f"  Frozen: {stats['frozen_params']:,} ({100*stats['frozen_params']/stats['total_params']:.1f}%)")
        print("-" * 50)


def configure_model_for_phase(
    model: nn.Module,
    phase: str,
) -> Dict[str, int]:
    """
    Configure model freezing for a training phase.

    Args:
        model: ModernBERTv3 model
        phase: Phase name ("phase_0.5", "phase_1", "phase_2", "inference")

    Returns:
        Freeze statistics
    """
    phase_enum = TrainingPhase(phase)
    freezer = LayerFreezer(model)
    return freezer.configure_for_phase(phase_enum)
```

**Acceptance Criteria:**

- [ ] `LayerBand` enum defines all 4 bands correctly
- [ ] `freeze_band()` and `unfreeze_band()` work correctly
- [ ] `configure_for_phase()` sets up correct freezing
- [ ] Phase 0.5/1: L1-18 frozen, L19-28 trainable
- [ ] Phase 2: All layers trainable
- [ ] Embeddings frozen in Phase 0.5/1
- [ ] `get_freeze_stats()` returns accurate counts

**Tests:** `tests/v3/test_trainer_v3.py::test_layer_freezing`

---

#### Issue 5.1.2: Implement Phase-Aware Training Loop

**File:** `src/modeling_studio/trainers/trainer_v3.py`
**Effort:** 6 hours
**Dependencies:** Issue 5.1.1

**Description:**
Implement the main training loop that supports phase-based training with automatic phase transitions, loss tracking, and checkpoint management.

**Implementation:**

```python
# src/modeling_studio/trainers/trainer_v3.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging
from tqdm import tqdm
import wandb

from .freezing_v3 import (
    LayerFreezer,
    TrainingPhase,
    configure_model_for_phase,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for v3 training."""
    # Phase settings
    phase: str = "phase_0.5"

    # Training hyperparameters
    max_steps: int = 2500
    warmup_steps: int = 500
    eval_steps: int = 250
    save_steps: int = 500
    logging_steps: int = 50

    # Learning rates (per layer group)
    learning_rate: float = 3e-5
    lr_layers_1_18: float = 0.0      # Frozen in Phase 0.5/1
    lr_layers_19_22: float = 1e-5    # Feeders
    lr_layer_23: float = 5e-5        # Interface
    lr_layers_24_28: float = 3e-5    # Clones

    # Optimization
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # Scheduler
    lr_scheduler_type: str = "cosine"

    # Mixed precision
    fp16: bool = False
    bf16: bool = True

    # Paths
    output_dir: str = "outputs/v3_training"
    checkpoint_dir: Optional[str] = None

    # Logging
    use_wandb: bool = True
    wandb_project: str = "modernbert-v3"
    wandb_run_name: Optional[str] = None


@dataclass
class TrainingState:
    """Tracks training state."""
    global_step: int = 0
    epoch: int = 0
    best_metric: float = 0.0
    phase: str = "phase_0.5"
    losses: List[float] = field(default_factory=list)
    metrics_history: List[Dict] = field(default_factory=list)


class ModernBERTv3Trainer:
    """
    Phase-aware trainer for ModernBERT v3.

    Training Phases:
        Phase 0.5 (Healing): ~2500 steps
            - Heal cloned layers L23-28
            - Smooth L22→L23 interface
            - Use generic benchmark data

        Phase 1 (Multi-task): ~5000 steps
            - Train on FamilyOS unified data
            - All 12 tasks active
            - LoRA on L23-28

        Phase 2 (Polish): ~1000 steps (optional)
            - Full fine-tune with low LR
            - Focus on safety/emotions
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        compute_metrics: Optional[Callable] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.compute_metrics = compute_metrics

        # State tracking
        self.state = TrainingState(phase=config.phase)

        # Layer freezer
        self.freezer = LayerFreezer(model)

        # Optimizer and scheduler (created in setup)
        self.optimizer = None
        self.scheduler = None
        self.scaler = None  # For mixed precision

        # Device
        self.device = next(model.parameters()).device

    def setup(self) -> None:
        """Setup training components."""
        # Configure freezing for phase
        self.freezer.configure_for_phase(TrainingPhase(self.config.phase))

        # Create optimizer with layer-group LRs
        self.optimizer = self._create_optimizer()

        # Create scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision scaler
        if self.config.fp16:
            self.scaler = torch.cuda.amp.GradScaler()

        # Wandb logging
        if self.config.use_wandb:
            self._init_wandb()

        logger.info("Training setup complete")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with per-layer-group learning rates."""
        param_groups = self._get_parameter_groups()

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.weight_decay,
        )

        return optimizer

    def _get_parameter_groups(self) -> List[Dict]:
        """
        Create parameter groups with layer-specific LRs.

        Groups:
            - layers_1_18: Foundation + Core (frozen or very low LR)
            - layers_19_22: Feeder band
            - layer_23: Interface layer (highest LR)
            - layers_24_28: Family band clones
            - embeddings: Usually frozen
            - task_heads: Same as layers_24_28
        """
        param_groups = []

        # Layers 1-18 (Foundation + Core)
        layers_1_18_params = []
        for i in range(18):
            layers_1_18_params.extend(self.model.encoder.layers[i].parameters())

        if any(p.requires_grad for p in layers_1_18_params):
            param_groups.append({
                "params": [p for p in layers_1_18_params if p.requires_grad],
                "lr": self.config.lr_layers_1_18,
                "name": "layers_1_18",
            })

        # Layers 19-22 (Feeder)
        layers_19_22_params = []
        for i in range(18, 22):
            layers_19_22_params.extend(self.model.encoder.layers[i].parameters())

        if any(p.requires_grad for p in layers_19_22_params):
            param_groups.append({
                "params": [p for p in layers_19_22_params if p.requires_grad],
                "lr": self.config.lr_layers_19_22,
                "name": "layers_19_22",
            })

        # Layer 23 (Interface - highest plasticity)
        layer_23_params = list(self.model.encoder.layers[22].parameters())
        if any(p.requires_grad for p in layer_23_params):
            param_groups.append({
                "params": [p for p in layer_23_params if p.requires_grad],
                "lr": self.config.lr_layer_23,
                "name": "layer_23",
            })

        # Layers 24-28 (Family clones)
        layers_24_28_params = []
        for i in range(23, 28):
            layers_24_28_params.extend(self.model.encoder.layers[i].parameters())

        if any(p.requires_grad for p in layers_24_28_params):
            param_groups.append({
                "params": [p for p in layers_24_28_params if p.requires_grad],
                "lr": self.config.lr_layers_24_28,
                "name": "layers_24_28",
            })

        # Embeddings
        if hasattr(self.model, "embeddings"):
            emb_params = list(self.model.embeddings.parameters())
            if any(p.requires_grad for p in emb_params):
                param_groups.append({
                    "params": [p for p in emb_params if p.requires_grad],
                    "lr": self.config.lr_layers_1_18,  # Low LR for embeddings
                    "name": "embeddings",
                })

        # Task heads
        if hasattr(self.model, "task_heads"):
            head_params = list(self.model.task_heads.parameters())
            if any(p.requires_grad for p in head_params):
                param_groups.append({
                    "params": [p for p in head_params if p.requires_grad],
                    "lr": self.config.lr_layers_24_28,
                    "name": "task_heads",
                })

        # Log parameter groups
        logger.info("Parameter groups:")
        for group in param_groups:
            n_params = sum(p.numel() for p in group["params"])
            logger.info(f"  {group['name']}: {n_params:,} params, lr={group['lr']}")

        return param_groups

    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        total_steps = self.config.max_steps
        warmup_steps = self.config.warmup_steps

        if self.config.lr_scheduler_type == "cosine":
            from transformers import get_cosine_schedule_with_warmup
            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        elif self.config.lr_scheduler_type == "linear":
            from transformers import get_linear_schedule_with_warmup
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.lr_scheduler_type}")

        return scheduler

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases logging."""
        wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_run_name or f"v3_{self.config.phase}",
            config=self.config.__dict__,
        )

    def train(self) -> Dict[str, Any]:
        """
        Main training loop.

        Returns:
            Training results and metrics
        """
        self.setup()

        logger.info(f"Starting training: {self.config.phase}")
        logger.info(f"  Max steps: {self.config.max_steps}")
        logger.info(f"  Warmup steps: {self.config.warmup_steps}")

        self.model.train()

        progress_bar = tqdm(
            total=self.config.max_steps,
            desc=f"Training ({self.config.phase})",
        )

        accumulated_loss = 0.0

        while self.state.global_step < self.config.max_steps:
            for batch in self.train_dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                loss = self._training_step(batch)

                # Backward pass with gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                accumulated_loss += loss.item()

                if self.config.fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Optimizer step
                if (self.state.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.max_grad_norm > 0:
                        if self.config.fp16:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm,
                        )

                    # Optimizer step
                    if self.config.fp16:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    # Logging
                    if self.state.global_step % self.config.logging_steps == 0:
                        self._log_step(accumulated_loss)
                        accumulated_loss = 0.0

                self.state.global_step += 1
                progress_bar.update(1)

                # Evaluation
                if self.state.global_step % self.config.eval_steps == 0:
                    if self.eval_dataloader:
                        metrics = self.evaluate()
                        self._log_eval(metrics)

                # Checkpointing
                if self.state.global_step % self.config.save_steps == 0:
                    self._save_checkpoint()

                if self.state.global_step >= self.config.max_steps:
                    break

        progress_bar.close()

        # Final evaluation and save
        if self.eval_dataloader:
            final_metrics = self.evaluate()
            self._log_eval(final_metrics)

        self._save_checkpoint(final=True)

        return {"final_step": self.state.global_step, "metrics": self.state.metrics_history}

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Single training step."""
        outputs = self.model(**batch)

        if hasattr(outputs, "loss"):
            loss = outputs.loss
        elif isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"]
        else:
            raise ValueError("Model output must contain 'loss'")

        return loss

    def evaluate(self) -> Dict[str, float]:
        """Run evaluation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)

                if hasattr(outputs, "loss"):
                    total_loss += outputs.loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        metrics = {"eval_loss": avg_loss}

        if self.compute_metrics:
            custom_metrics = self.compute_metrics(self.model, self.eval_dataloader)
            metrics.update(custom_metrics)

        self.model.train()
        return metrics

    def _log_step(self, loss: float) -> None:
        """Log training step."""
        lr = self.scheduler.get_last_lr()[0]

        log_dict = {
            "train/loss": loss,
            "train/lr": lr,
            "train/step": self.state.global_step,
        }

        if self.config.use_wandb:
            wandb.log(log_dict)

        logger.info(f"Step {self.state.global_step}: loss={loss:.4f}, lr={lr:.2e}")

    def _log_eval(self, metrics: Dict[str, float]) -> None:
        """Log evaluation metrics."""
        if self.config.use_wandb:
            wandb.log({f"eval/{k}": v for k, v in metrics.items()})

        logger.info(f"Eval: {metrics}")
        self.state.metrics_history.append(metrics)

    def _save_checkpoint(self, final: bool = False) -> None:
        """Save training checkpoint."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_name = "final" if final else f"step_{self.state.global_step}"
        checkpoint_path = output_dir / checkpoint_name

        checkpoint_path.mkdir(exist_ok=True)

        # Save model
        torch.save(
            self.model.state_dict(),
            checkpoint_path / "pytorch_model.bin",
        )

        # Save training state
        torch.save(
            {
                "global_step": self.state.global_step,
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
            },
            checkpoint_path / "trainer_state.bin",
        )

        logger.info(f"Saved checkpoint: {checkpoint_path}")
```

**Acceptance Criteria:**

- [ ] `TrainingConfig` supports all phase-specific settings
- [ ] Per-layer-group learning rates applied correctly
- [ ] Warmup + cosine decay scheduler works
- [ ] Gradient clipping prevents exploding gradients
- [ ] Mixed precision (bf16) supported
- [ ] Checkpointing at configurable intervals
- [ ] Wandb logging integration
- [ ] Evaluation at configurable intervals

**Tests:** `tests/v3/test_trainer_v3.py::test_phase_aware_training`

---

#### Issue 5.1.3: Implement LoRA Application to Layers 23-28

**File:** `src/modeling_studio/trainers/lora_v3.py`
**Effort:** 5 hours
**Dependencies:** Issue 2.2.4 (LoRA adapters)

**Description:**
Implement LoRA (Low-Rank Adaptation) specifically for layers 23-28 (Family Band) to enable efficient fine-tuning with fewer parameters while preserving the base model's capabilities.

**Implementation:**

```python
# src/modeling_studio/trainers/lora_v3.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters."""
    rank: int = 16                     # LoRA rank (r)
    alpha: float = 32.0                # Scaling factor (α)
    dropout: float = 0.1               # Dropout on LoRA path
    target_modules: List[str] = None   # Which modules to apply LoRA to
    layers: List[int] = None           # Which layers to apply LoRA to

    def __post_init__(self):
        if self.target_modules is None:
            # Default: apply to attention Q, K, V and output projection
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if self.layers is None:
            # Default: apply to Family Band (L23-28)
            self.layers = list(range(22, 28))

    @property
    def scaling(self) -> float:
        """LoRA scaling factor."""
        return self.alpha / self.rank


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adapter.

    LoRA decomposes weight update:
        W' = W + ΔW = W + BA

    Where:
        - W: Original frozen weights [out, in]
        - B: Low-rank down projection [out, r]
        - A: Low-rank up projection [r, in]
        - r: LoRA rank (much smaller than in/out)

    Forward:
        y = Wx + (α/r) * BAx
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Original linear layer (frozen)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # LoRA adapters
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout)

        # Initialize LoRA weights
        self._init_lora_weights()

        # Initially disabled
        self.merged = False
        self.enabled = True

    def _init_lora_weights(self) -> None:
        """Initialize LoRA weights for stable training start."""
        # A: Kaiming uniform (matches PyTorch default)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        # B: Zero init (LoRA starts as identity)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with LoRA."""
        # Original forward
        result = self.linear(x)

        if self.enabled and not self.merged:
            # Add LoRA contribution
            lora_out = self.lora_B(self.lora_A(self.lora_dropout(x)))
            result = result + self.scaling * lora_out

        return result

    def merge_weights(self) -> None:
        """Merge LoRA weights into base linear layer."""
        if self.merged:
            return

        with torch.no_grad():
            # W' = W + (α/r) * B @ A
            delta_w = self.scaling * (self.lora_B.weight @ self.lora_A.weight)
            self.linear.weight.add_(delta_w)

        self.merged = True
        logger.debug("Merged LoRA weights")

    def unmerge_weights(self) -> None:
        """Unmerge LoRA weights from base linear layer."""
        if not self.merged:
            return

        with torch.no_grad():
            delta_w = self.scaling * (self.lora_B.weight @ self.lora_A.weight)
            self.linear.weight.sub_(delta_w)

        self.merged = False

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.1,
    ) -> "LoRALinear":
        """Create LoRALinear from existing Linear layer."""
        lora_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            bias=linear.bias is not None,
        )

        # Copy original weights
        lora_linear.linear.weight.data = linear.weight.data.clone()
        if linear.bias is not None:
            lora_linear.linear.bias.data = linear.bias.data.clone()

        # Freeze original weights
        lora_linear.linear.weight.requires_grad = False
        if lora_linear.linear.bias is not None:
            lora_linear.linear.bias.requires_grad = False

        return lora_linear


class LoRAManager:
    """
    Manages LoRA application to a model.

    Applies LoRA to specific layers and modules in ModernBERT v3.
    """

    def __init__(self, model: nn.Module, config: LoRAConfig):
        self.model = model
        self.config = config
        self.lora_modules: Dict[str, LoRALinear] = {}
        self._original_modules: Dict[str, nn.Linear] = {}

    def apply_lora(self) -> int:
        """
        Apply LoRA to configured layers and modules.

        Returns:
            Number of LoRA parameters added
        """
        lora_params = 0

        encoder = self.model.encoder if hasattr(self.model, "encoder") else self.model

        for layer_idx in self.config.layers:
            layer = encoder.layers[layer_idx]

            for module_name in self.config.target_modules:
                # Find the module in the layer
                full_name = f"layer_{layer_idx}.{module_name}"
                module = self._get_module(layer, module_name)

                if module is None:
                    logger.warning(f"Module {module_name} not found in layer {layer_idx}")
                    continue

                if not isinstance(module, nn.Linear):
                    logger.warning(f"Module {full_name} is not Linear, skipping")
                    continue

                # Store original
                self._original_modules[full_name] = module

                # Create LoRA version
                lora_module = LoRALinear.from_linear(
                    module,
                    rank=self.config.rank,
                    alpha=self.config.alpha,
                    dropout=self.config.dropout,
                )

                # Replace in model
                self._set_module(layer, module_name, lora_module)
                self.lora_modules[full_name] = lora_module

                # Count params
                lora_params += self.config.rank * (module.in_features + module.out_features)

                logger.debug(f"Applied LoRA to {full_name}")

        logger.info(f"Applied LoRA to {len(self.lora_modules)} modules")
        logger.info(f"  LoRA rank: {self.config.rank}")
        logger.info(f"  LoRA params: {lora_params:,}")

        return lora_params

    def _get_module(self, parent: nn.Module, name: str) -> Optional[nn.Module]:
        """Get a nested module by dot-separated name."""
        parts = name.split(".")
        module = parent

        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None

        return module

    def _set_module(self, parent: nn.Module, name: str, new_module: nn.Module) -> None:
        """Set a nested module by dot-separated name."""
        parts = name.split(".")

        for part in parts[:-1]:
            parent = getattr(parent, part)

        setattr(parent, parts[-1], new_module)

    def get_lora_parameters(self) -> List[nn.Parameter]:
        """Get all LoRA parameters for optimizer."""
        params = []
        for lora_module in self.lora_modules.values():
            params.extend(lora_module.lora_A.parameters())
            params.extend(lora_module.lora_B.parameters())
        return params

    def merge_all(self) -> None:
        """Merge all LoRA weights into base model."""
        for name, lora_module in self.lora_modules.items():
            lora_module.merge_weights()
            logger.debug(f"Merged {name}")

        logger.info(f"Merged {len(self.lora_modules)} LoRA modules")

    def unmerge_all(self) -> None:
        """Unmerge all LoRA weights from base model."""
        for name, lora_module in self.lora_modules.items():
            lora_module.unmerge_weights()

    def enable_lora(self, enable: bool = True) -> None:
        """Enable or disable LoRA."""
        for lora_module in self.lora_modules.values():
            lora_module.enabled = enable

    def save_lora_weights(self, path: str) -> None:
        """Save only LoRA weights."""
        lora_state = {}
        for name, lora_module in self.lora_modules.items():
            lora_state[f"{name}.lora_A"] = lora_module.lora_A.state_dict()
            lora_state[f"{name}.lora_B"] = lora_module.lora_B.state_dict()

        torch.save(lora_state, path)
        logger.info(f"Saved LoRA weights to {path}")

    def load_lora_weights(self, path: str) -> None:
        """Load LoRA weights."""
        lora_state = torch.load(path)

        for name, lora_module in self.lora_modules.items():
            if f"{name}.lora_A" in lora_state:
                lora_module.lora_A.load_state_dict(lora_state[f"{name}.lora_A"])
            if f"{name}.lora_B" in lora_state:
                lora_module.lora_B.load_state_dict(lora_state[f"{name}.lora_B"])

        logger.info(f"Loaded LoRA weights from {path}")


def apply_lora_to_family_band(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.1,
) -> LoRAManager:
    """
    Apply LoRA to Family Band (L23-28).

    Args:
        model: ModernBERTv3 model
        rank: LoRA rank
        alpha: LoRA scaling
        dropout: LoRA dropout

    Returns:
        LoRAManager for controlling LoRA
    """
    config = LoRAConfig(
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        layers=list(range(22, 28)),  # L23-28
    )

    manager = LoRAManager(model, config)
    manager.apply_lora()

    return manager
```

**Acceptance Criteria:**

- [ ] `LoRALinear` implements low-rank adaptation correctly
- [ ] `merge_weights()` combines LoRA into base layer
- [ ] `unmerge_weights()` reverses merge correctly
- [ ] `LoRAManager.apply_lora()` targets L23-28 attention projections
- [ ] `get_lora_parameters()` returns only trainable LoRA params
- [ ] `save_lora_weights()` / `load_lora_weights()` work correctly
- [ ] LoRA rank 16 adds ~2M params (vs 150M base)

**Tests:** `tests/v3/test_trainer_v3.py::test_lora_application`

---

#### Issue 5.1.4: Implement Layer-Group Learning Rates

**File:** `src/modeling_studio/trainers/lr_groups_v3.py`
**Effort:** 4 hours
**Dependencies:** Issue 5.1.2 (Trainer)

**Description:**
Implement layer-group specific learning rates for optimal training. Different layer bands require different learning rates based on their role and whether they're transferred vs cloned.

**Implementation:**

```python
# src/modeling_studio/trainers/lr_groups_v3.py

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class LayerGroupLRConfig:
    """
    Configuration for layer-group learning rates.

    Rationale:
        - Foundation/Core (L1-18): Very low or frozen - preserve v2 knowledge
        - Feeder (L19-22): Low LR - gentle refinement of interface
        - Interface (L23): Highest LR - needs most adaptation
        - Family (L24-28): Moderate LR - learning new capabilities
    """
    # Base learning rate
    base_lr: float = 3e-5

    # Layer band multipliers (relative to base_lr)
    foundation_mult: float = 0.0     # L1-6: Frozen or no training
    core_mult: float = 0.0           # L7-18: Frozen or no training
    feeder_mult: float = 0.33        # L19-22: 1/3 of base LR
    interface_mult: float = 1.67     # L23: 5/3 of base LR (highest)
    family_mult: float = 1.0         # L24-28: Base LR

    # Component-specific multipliers
    embeddings_mult: float = 0.1     # Usually frozen or very low
    task_heads_mult: float = 1.0     # Same as family band
    hub_tokens_mult: float = 0.5     # Careful with hub token gradients

    # Warmup settings
    warmup_ratio: float = 0.1        # 10% warmup
    min_lr_ratio: float = 0.01       # End at 1% of peak

    def get_layer_lr(self, layer_idx: int) -> float:
        """Get learning rate for a specific layer."""
        if layer_idx < 6:  # Foundation
            return self.base_lr * self.foundation_mult
        elif layer_idx < 18:  # Core
            return self.base_lr * self.core_mult
        elif layer_idx < 22:  # Feeder
            return self.base_lr * self.feeder_mult
        elif layer_idx == 22:  # Interface (L23, 0-indexed)
            return self.base_lr * self.interface_mult
        else:  # Family
            return self.base_lr * self.family_mult


# Preset configurations for different phases
PHASE_LR_CONFIGS = {
    "phase_0.5": LayerGroupLRConfig(
        base_lr=3e-5,
        foundation_mult=0.0,
        core_mult=0.0,
        feeder_mult=0.33,
        interface_mult=1.67,
        family_mult=1.0,
    ),
    "phase_1": LayerGroupLRConfig(
        base_lr=2e-5,
        foundation_mult=0.0,
        core_mult=0.0,
        feeder_mult=0.5,
        interface_mult=1.5,
        family_mult=1.0,
    ),
    "phase_2": LayerGroupLRConfig(
        base_lr=1e-5,
        foundation_mult=0.1,
        core_mult=0.2,
        feeder_mult=0.5,
        interface_mult=1.0,
        family_mult=1.0,
    ),
}


class LayerGroupOptimizer:
    """
    Creates optimizer with layer-group specific learning rates.

    Usage:
        config = LayerGroupLRConfig(base_lr=3e-5)
        group_optimizer = LayerGroupOptimizer(model, config)
        optimizer = group_optimizer.create_optimizer()
    """

    def __init__(
        self,
        model: nn.Module,
        config: LayerGroupLRConfig,
        weight_decay: float = 0.01,
    ):
        self.model = model
        self.config = config
        self.weight_decay = weight_decay

        # Get encoder reference
        self.encoder = model.encoder if hasattr(model, "encoder") else model

    def create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create AdamW optimizer with layer-group LRs.

        Returns:
            Configured AdamW optimizer
        """
        param_groups = self._build_param_groups()

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.weight_decay,
        )

        self._log_param_groups(param_groups)
        return optimizer

    def _build_param_groups(self) -> List[Dict]:
        """Build parameter groups with appropriate LRs."""
        param_groups = []

        # Track which parameters we've assigned
        assigned_params = set()

        # Layer groups
        layer_groups = {
            "foundation": (range(0, 6), self.config.foundation_mult),
            "core": (range(6, 18), self.config.core_mult),
            "feeder": (range(18, 22), self.config.feeder_mult),
            "interface": ([22], self.config.interface_mult),
            "family": (range(23, 28), self.config.family_mult),
        }

        for group_name, (layer_indices, mult) in layer_groups.items():
            lr = self.config.base_lr * mult

            if lr == 0:
                # Skip frozen groups (they're not trainable anyway)
                continue

            params = []
            for layer_idx in layer_indices:
                layer = self.encoder.layers[layer_idx]
                for p in layer.parameters():
                    if p.requires_grad and id(p) not in assigned_params:
                        params.append(p)
                        assigned_params.add(id(p))

            if params:
                param_groups.append({
                    "params": params,
                    "lr": lr,
                    "name": group_name,
                })

        # Embeddings
        if hasattr(self.model, "embeddings"):
            emb_params = [
                p for p in self.model.embeddings.parameters()
                if p.requires_grad and id(p) not in assigned_params
            ]
            if emb_params:
                param_groups.append({
                    "params": emb_params,
                    "lr": self.config.base_lr * self.config.embeddings_mult,
                    "name": "embeddings",
                })
                for p in emb_params:
                    assigned_params.add(id(p))

        # Task heads
        if hasattr(self.model, "task_heads"):
            head_params = [
                p for p in self.model.task_heads.parameters()
                if p.requires_grad and id(p) not in assigned_params
            ]
            if head_params:
                param_groups.append({
                    "params": head_params,
                    "lr": self.config.base_lr * self.config.task_heads_mult,
                    "name": "task_heads",
                })
                for p in head_params:
                    assigned_params.add(id(p))

        # Any remaining parameters (e.g., poolers, classifiers)
        remaining_params = [
            p for p in self.model.parameters()
            if p.requires_grad and id(p) not in assigned_params
        ]
        if remaining_params:
            param_groups.append({
                "params": remaining_params,
                "lr": self.config.base_lr,
                "name": "other",
            })

        return param_groups

    def _log_param_groups(self, param_groups: List[Dict]) -> None:
        """Log parameter group configuration."""
        print("\n" + "=" * 60)
        print("Layer Group Learning Rates")
        print("=" * 60)

        total_params = 0
        for group in param_groups:
            n_params = sum(p.numel() for p in group["params"])
            total_params += n_params
            print(f"  {group['name']:15} | lr={group['lr']:.2e} | params={n_params:,}")

        print("-" * 60)
        print(f"  {'TOTAL':15} | base_lr={self.config.base_lr:.2e} | params={total_params:,}")
        print("=" * 60 + "\n")


def create_layer_group_optimizer(
    model: nn.Module,
    phase: str = "phase_0.5",
    base_lr: Optional[float] = None,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    """
    Create optimizer with phase-appropriate layer-group LRs.

    Args:
        model: ModernBERTv3 model
        phase: Training phase name
        base_lr: Override base learning rate
        weight_decay: Weight decay for AdamW

    Returns:
        Configured optimizer
    """
    config = PHASE_LR_CONFIGS.get(phase, PHASE_LR_CONFIGS["phase_0.5"])

    if base_lr is not None:
        config.base_lr = base_lr

    group_optimizer = LayerGroupOptimizer(model, config, weight_decay)
    return group_optimizer.create_optimizer()
```

**Acceptance Criteria:**

- [ ] `LayerGroupLRConfig` supports all layer bands
- [ ] Foundation/Core get 0 LR in Phase 0.5/1
- [ ] Interface layer (L23) gets highest LR
- [ ] Feeder band gets lower LR than Family
- [ ] `create_optimizer()` creates valid AdamW
- [ ] Parameter groups logged clearly
- [ ] Preset configs for all phases

**Tests:** `tests/v3/test_trainer_v3.py::test_layer_group_lr`

---

#### Issue 5.1.5: Implement Hub Token Gradient Masking

**File:** `src/modeling_studio/trainers/gradient_masking_v3.py`
**Effort:** 4 hours
**Dependencies:** Issue 1.1.4 (Hub Tokens)

**Description:**
Implement gradient masking for hub tokens to enable selective training. Hub token embeddings need special handling since they're part of the word embedding matrix but may need different training dynamics.

**Implementation:**

```python
# src/modeling_studio/trainers/gradient_masking_v3.py

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# Hub token positions in vocabulary
HUB_TOKEN_POSITIONS = {
    "[EMO]": 50368,
    "[MEM]": 50369,
    "[REL]": 50370,
    "[TASK]": 50371,
}

# Vocabulary layout
V2_VOCAB_SIZE = 50368  # Original ModernBERT vocab
HUB_TOKEN_START = 50368
HUB_TOKEN_COUNT = 4
V3_VOCAB_SIZE = 50372  # V2 + hub tokens


@dataclass
class GradientMaskConfig:
    """Configuration for gradient masking."""
    # Which hub tokens to train
    train_hub_tokens: List[str] = None
    # Freeze original vocabulary
    freeze_original_vocab: bool = True
    # Hub token gradient scaling
    hub_token_grad_scale: float = 1.0

    def __post_init__(self):
        if self.train_hub_tokens is None:
            # Default: train all hub tokens
            self.train_hub_tokens = list(HUB_TOKEN_POSITIONS.keys())


class EmbeddingGradientHook:
    """
    Gradient hook for selective embedding training.

    Applies gradient masking to word embeddings to:
    1. Zero gradients for frozen token positions
    2. Scale gradients for hub tokens
    3. Enable per-token training control
    """

    def __init__(
        self,
        embedding_weight: nn.Parameter,
        config: GradientMaskConfig,
    ):
        """
        Args:
            embedding_weight: Word embedding weight [vocab_size, hidden_size]
            config: Gradient mask configuration
        """
        self.embedding_weight = embedding_weight
        self.config = config
        self.hook_handle = None

        # Build gradient mask
        self.grad_mask = self._build_gradient_mask()

    def _build_gradient_mask(self) -> torch.Tensor:
        """
        Build gradient mask tensor.

        Returns:
            Mask [vocab_size, 1] where 0=frozen, >0=trainable
        """
        vocab_size = self.embedding_weight.shape[0]
        device = self.embedding_weight.device
        dtype = self.embedding_weight.dtype

        # Start with all frozen or all trainable
        if self.config.freeze_original_vocab:
            mask = torch.zeros(vocab_size, 1, device=device, dtype=dtype)
        else:
            mask = torch.ones(vocab_size, 1, device=device, dtype=dtype)

        # Set hub token masks
        for token_name, position in HUB_TOKEN_POSITIONS.items():
            if position < vocab_size:
                if token_name in self.config.train_hub_tokens:
                    # Trainable with scaling
                    mask[position] = self.config.hub_token_grad_scale
                else:
                    # Frozen
                    mask[position] = 0.0

        logger.info(f"Built gradient mask:")
        logger.info(f"  Original vocab (0-{V2_VOCAB_SIZE-1}): {'frozen' if self.config.freeze_original_vocab else 'trainable'}")
        logger.info(f"  Hub tokens ({HUB_TOKEN_START}-{V3_VOCAB_SIZE-1}): {self.config.train_hub_tokens}")
        logger.info(f"  Hub gradient scale: {self.config.hub_token_grad_scale}")

        return mask

    def _gradient_hook(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Hook function applied to gradients.

        Args:
            grad: Gradient tensor [vocab_size, hidden_size]

        Returns:
            Masked gradient tensor
        """
        # Apply mask
        masked_grad = grad * self.grad_mask.to(grad.device)
        return masked_grad

    def register(self) -> None:
        """Register gradient hook on embedding weight."""
        if self.hook_handle is not None:
            self.hook_handle.remove()

        self.hook_handle = self.embedding_weight.register_hook(self._gradient_hook)
        logger.info("Registered embedding gradient hook")

    def remove(self) -> None:
        """Remove gradient hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
            logger.info("Removed embedding gradient hook")

    def update_trainable_tokens(self, token_names: List[str]) -> None:
        """Update which hub tokens are trainable."""
        self.config.train_hub_tokens = token_names
        self.grad_mask = self._build_gradient_mask()


class HubTokenGradientManager:
    """
    Manages hub token gradient masking for a model.

    Provides high-level interface for controlling hub token training.
    """

    def __init__(self, model: nn.Module, config: Optional[GradientMaskConfig] = None):
        """
        Args:
            model: Model with embeddings.word_embeddings
            config: Gradient mask configuration
        """
        self.model = model
        self.config = config or GradientMaskConfig()
        self.hooks: List[EmbeddingGradientHook] = []

    def get_embedding_weight(self) -> Optional[nn.Parameter]:
        """Get word embedding weight from model."""
        if hasattr(self.model, "embeddings"):
            if hasattr(self.model.embeddings, "word_embeddings"):
                return self.model.embeddings.word_embeddings.weight
        if hasattr(self.model, "encoder"):
            if hasattr(self.model.encoder, "embeddings"):
                if hasattr(self.model.encoder.embeddings, "word_embeddings"):
                    return self.model.encoder.embeddings.word_embeddings.weight
        return None

    def setup(self) -> bool:
        """
        Setup gradient masking for hub tokens.

        Returns:
            True if setup successful
        """
        embedding_weight = self.get_embedding_weight()

        if embedding_weight is None:
            logger.warning("Could not find embedding weight in model")
            return False

        # Create and register hook
        hook = EmbeddingGradientHook(embedding_weight, self.config)
        hook.register()
        self.hooks.append(hook)

        logger.info("Hub token gradient masking setup complete")
        return True

    def cleanup(self) -> None:
        """Remove all gradient hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def freeze_all_hub_tokens(self) -> None:
        """Freeze all hub token gradients."""
        for hook in self.hooks:
            hook.update_trainable_tokens([])
        logger.info("Froze all hub tokens")

    def unfreeze_all_hub_tokens(self) -> None:
        """Enable gradients for all hub tokens."""
        for hook in self.hooks:
            hook.update_trainable_tokens(list(HUB_TOKEN_POSITIONS.keys()))
        logger.info("Unfroze all hub tokens")

    def train_specific_hub_tokens(self, token_names: List[str]) -> None:
        """
        Train only specific hub tokens.

        Args:
            token_names: List of hub token names to train
        """
        valid_tokens = [t for t in token_names if t in HUB_TOKEN_POSITIONS]
        for hook in self.hooks:
            hook.update_trainable_tokens(valid_tokens)
        logger.info(f"Training hub tokens: {valid_tokens}")

    def get_hub_token_gradients(self) -> Dict[str, Optional[torch.Tensor]]:
        """
        Get current gradients for hub tokens.

        Returns:
            Dict mapping token name to gradient tensor
        """
        embedding_weight = self.get_embedding_weight()
        if embedding_weight is None or embedding_weight.grad is None:
            return {name: None for name in HUB_TOKEN_POSITIONS}

        gradients = {}
        for token_name, position in HUB_TOKEN_POSITIONS.items():
            if position < embedding_weight.grad.shape[0]:
                gradients[token_name] = embedding_weight.grad[position].clone()
            else:
                gradients[token_name] = None

        return gradients

    def get_hub_token_embeddings(self) -> Dict[str, torch.Tensor]:
        """
        Get current hub token embeddings.

        Returns:
            Dict mapping token name to embedding tensor
        """
        embedding_weight = self.get_embedding_weight()
        if embedding_weight is None:
            return {}

        embeddings = {}
        for token_name, position in HUB_TOKEN_POSITIONS.items():
            if position < embedding_weight.shape[0]:
                embeddings[token_name] = embedding_weight[position].clone()

        return embeddings


def setup_hub_token_gradient_masking(
    model: nn.Module,
    train_hub_tokens: Optional[List[str]] = None,
    freeze_original_vocab: bool = True,
    hub_token_grad_scale: float = 1.0,
) -> HubTokenGradientManager:
    """
    Setup hub token gradient masking for a model.

    Args:
        model: ModernBERTv3 model
        train_hub_tokens: Which hub tokens to train (None = all)
        freeze_original_vocab: Whether to freeze original vocab embeddings
        hub_token_grad_scale: Gradient scaling for hub tokens

    Returns:
        Configured HubTokenGradientManager
    """
    config = GradientMaskConfig(
        train_hub_tokens=train_hub_tokens,
        freeze_original_vocab=freeze_original_vocab,
        hub_token_grad_scale=hub_token_grad_scale,
    )

    manager = HubTokenGradientManager(model, config)
    manager.setup()

    return manager
```

**Acceptance Criteria:**

- [ ] `EmbeddingGradientHook` masks gradients correctly
- [ ] Original vocab (0-50367) gradients zeroed when frozen
- [ ] Hub token gradients preserved/scaled
- [ ] `train_specific_hub_tokens()` selects specific tokens
- [ ] `get_hub_token_gradients()` returns correct values
- [ ] Hooks properly registered and removable
- [ ] No memory leaks from hook registration

**Tests:** `tests/v3/test_trainer_v3.py::test_hub_token_gradient_masking`

---

#### Issue 5.1.6: Implement Zipper Learning Rate Strategy

**File:** `src/modeling_studio/trainers/zipper_lr_v3.py`
**Effort:** 4 hours
**Dependencies:** Issue 5.1.4 (Layer-Group LRs)

**Description:**
Implement the Zipper Learning Rate strategy that provides smooth LR transitions across the v2→v3 interface boundary. This prevents the "cliff effect" at L22→L23 transition.

**Implementation:**

```python
# src/modeling_studio/trainers/zipper_lr_v3.py

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class ZipperLRConfig:
    """
    Configuration for Zipper Learning Rate strategy.

    The Zipper strategy creates a smooth LR transition across the
    v2→v3 interface boundary to prevent gradient discontinuities.

    Layer Layout:
        L1-18:  Foundation + Core (frozen, lr=0)
        L19-22: Feeder band (low lr, interface preparation)
        L23:    Interface layer (highest lr, maximum plasticity)
        L24-28: Family band (moderate lr, learning new tasks)

    LR Profile (Phase 0.5):
        L19: 1e-5 ─┐
        L20: 1e-5  │ Feeder: gentle adaptation
        L21: 1e-5  │
        L22: 1e-5 ─┘
        L23: 5e-5 ← Interface: highest plasticity
        L24: 4e-5 ─┐
        L25: 3.5e-5│ Family: decreasing toward output
        L26: 3e-5  │
        L27: 3e-5  │
        L28: 3e-5 ─┘
    """
    # Base learning rate (reference point)
    base_lr: float = 3e-5

    # Feeder band (L19-22) - uniform low LR
    feeder_lr: float = 1e-5

    # Interface layer (L23) - maximum plasticity
    interface_lr: float = 5e-5

    # Family band (L24-28) - can be uniform or graduated
    family_lr: float = 3e-5
    family_graduated: bool = True   # Decrease LR toward output
    family_decay: float = 0.9       # Each layer = prev * decay

    # Frozen layers (L1-18)
    frozen_lr: float = 0.0

    # Additional components
    embeddings_lr: float = 0.0      # Usually frozen
    task_heads_lr: float = 3e-5     # Same as family

    def get_layer_lr(self, layer_idx: int) -> float:
        """Get learning rate for a specific layer (0-indexed)."""
        if layer_idx < 18:
            # Foundation + Core: frozen
            return self.frozen_lr
        elif layer_idx < 22:
            # Feeder (L19-22)
            return self.feeder_lr
        elif layer_idx == 22:
            # Interface (L23)
            return self.interface_lr
        else:
            # Family (L24-28)
            if self.family_graduated:
                # Decay from interface
                steps_from_interface = layer_idx - 22
                return self.interface_lr * (self.family_decay ** steps_from_interface)
            else:
                return self.family_lr


# Preset Zipper configurations for different phases
ZIPPER_PRESETS = {
    "phase_0.5_healing": ZipperLRConfig(
        base_lr=3e-5,
        feeder_lr=1e-5,
        interface_lr=5e-5,
        family_lr=3e-5,
        family_graduated=True,
        family_decay=0.85,
    ),
    "phase_1_multitask": ZipperLRConfig(
        base_lr=2e-5,
        feeder_lr=1e-5,
        interface_lr=4e-5,
        family_lr=2e-5,
        family_graduated=True,
        family_decay=0.9,
    ),
    "phase_2_polish": ZipperLRConfig(
        base_lr=1e-5,
        feeder_lr=5e-6,
        interface_lr=2e-5,
        family_lr=1e-5,
        family_graduated=False,
    ),
    "conservative": ZipperLRConfig(
        base_lr=1e-5,
        feeder_lr=5e-6,
        interface_lr=3e-5,
        family_lr=1e-5,
        family_graduated=False,
    ),
    "aggressive": ZipperLRConfig(
        base_lr=5e-5,
        feeder_lr=2e-5,
        interface_lr=1e-4,
        family_lr=5e-5,
        family_graduated=True,
        family_decay=0.8,
    ),
}


class ZipperLROptimizer:
    """
    Creates optimizer using Zipper Learning Rate strategy.

    The Zipper method ensures:
    1. Smooth LR transition at v2→v3 interface
    2. Maximum plasticity at L23 (interface layer)
    3. Graduated LR decay in Family band
    4. Preserved v2 knowledge via frozen Foundation/Core
    """

    def __init__(
        self,
        model: nn.Module,
        config: ZipperLRConfig,
        weight_decay: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.model = model
        self.config = config
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps

        # Get encoder reference
        self.encoder = model.encoder if hasattr(model, "encoder") else model

    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with Zipper LR strategy."""
        param_groups = self._build_zipper_param_groups()

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.weight_decay,
            betas=self.betas,
            eps=self.eps,
        )

        self._print_zipper_summary()
        return optimizer

    def _build_zipper_param_groups(self) -> List[Dict]:
        """Build parameter groups with Zipper LR pattern."""
        param_groups = []
        assigned_params = set()

        # Per-layer groups for L19-28 (trainable layers)
        for layer_idx in range(18, 28):
            layer = self.encoder.layers[layer_idx]
            lr = self.config.get_layer_lr(layer_idx)

            if lr <= 0:
                continue

            params = [p for p in layer.parameters()
                     if p.requires_grad and id(p) not in assigned_params]

            if params:
                param_groups.append({
                    "params": params,
                    "lr": lr,
                    "name": f"layer_{layer_idx + 1}",  # 1-indexed for display
                })
                for p in params:
                    assigned_params.add(id(p))

        # Embeddings (usually frozen)
        if hasattr(self.model, "embeddings"):
            emb_lr = self.config.embeddings_lr
            if emb_lr > 0:
                emb_params = [p for p in self.model.embeddings.parameters()
                             if p.requires_grad and id(p) not in assigned_params]
                if emb_params:
                    param_groups.append({
                        "params": emb_params,
                        "lr": emb_lr,
                        "name": "embeddings",
                    })
                    for p in emb_params:
                        assigned_params.add(id(p))

        # Task heads
        if hasattr(self.model, "task_heads"):
            head_lr = self.config.task_heads_lr
            head_params = [p for p in self.model.task_heads.parameters()
                          if p.requires_grad and id(p) not in assigned_params]
            if head_params:
                param_groups.append({
                    "params": head_params,
                    "lr": head_lr,
                    "name": "task_heads",
                })
                for p in head_params:
                    assigned_params.add(id(p))

        # Any remaining trainable parameters
        remaining = [p for p in self.model.parameters()
                    if p.requires_grad and id(p) not in assigned_params]
        if remaining:
            param_groups.append({
                "params": remaining,
                "lr": self.config.base_lr,
                "name": "other",
            })

        return param_groups

    def _print_zipper_summary(self) -> None:
        """Print Zipper LR visualization."""
        print("\n" + "=" * 60)
        print("⚡ Zipper Learning Rate Strategy")
        print("=" * 60)

        # ASCII visualization
        print("\nLR Profile:")
        print("  Layer │ LR        │ Band")
        print("  ──────┼───────────┼─────────")

        for layer_idx in range(28):
            lr = self.config.get_layer_lr(layer_idx)
            layer_num = layer_idx + 1

            # Band name
            if layer_idx < 6:
                band = "Foundation"
            elif layer_idx < 18:
                band = "Core"
            elif layer_idx < 22:
                band = "Feeder"
            elif layer_idx == 22:
                band = "Interface ★"
            else:
                band = "Family"

            # LR bar
            if lr > 0:
                bar_len = min(20, int(lr * 400000))
                bar = "█" * bar_len
                print(f"  L{layer_num:02d}   │ {lr:.1e} │ {band:12} {bar}")
            else:
                print(f"  L{layer_num:02d}   │ frozen    │ {band}")

        print("=" * 60 + "\n")

    def get_lr_dict(self) -> Dict[str, float]:
        """Get dictionary of layer→LR mappings."""
        lr_dict = {}
        for layer_idx in range(28):
            lr_dict[f"layer_{layer_idx + 1}"] = self.config.get_layer_lr(layer_idx)
        lr_dict["embeddings"] = self.config.embeddings_lr
        lr_dict["task_heads"] = self.config.task_heads_lr
        return lr_dict


def create_zipper_optimizer(
    model: nn.Module,
    preset: str = "phase_0.5_healing",
    weight_decay: float = 0.01,
    **overrides,
) -> torch.optim.Optimizer:
    """
    Create optimizer with Zipper LR strategy.

    Args:
        model: ModernBERTv3 model
        preset: Preset name from ZIPPER_PRESETS
        weight_decay: Weight decay for AdamW
        **overrides: Override specific config values

    Returns:
        Configured optimizer
    """
    config = ZIPPER_PRESETS.get(preset, ZIPPER_PRESETS["phase_0.5_healing"])

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    zipper = ZipperLROptimizer(model, config, weight_decay)
    return zipper.create_optimizer()


# Quick reference for Zipper LR values
ZIPPER_LR_QUICK_REF = """
╔══════════════════════════════════════════════════════════╗
║            Zipper Learning Rate Quick Reference           ║
╠══════════════════════════════════════════════════════════╣
║ Layer    │ Phase 0.5  │ Phase 1    │ Phase 2             ║
║──────────┼────────────┼────────────┼─────────────────────║
║ L1-18    │ 0 (frozen) │ 0 (frozen) │ 1e-6 (low)          ║
║ L19-22   │ 1e-5       │ 1e-5       │ 5e-6                ║
║ L23 ★    │ 5e-5       │ 4e-5       │ 2e-5                ║
║ L24-28   │ 3e-5→      │ 2e-5→      │ 1e-5                ║
╚══════════════════════════════════════════════════════════╝

★ = Interface layer (maximum plasticity)
→ = Graduated decay toward output layer
"""
```

**Acceptance Criteria:**

- [ ] `ZipperLRConfig` defines all layer LRs
- [ ] Interface layer (L23) gets highest LR
- [ ] Graduated decay in Family band works correctly
- [ ] Feeder band gets uniform low LR
- [ ] `create_optimizer()` creates valid AdamW
- [ ] ASCII visualization shows LR profile clearly
- [ ] Presets for all phases available
- [ ] Override mechanism works

**Tests:** `tests/v3/test_trainer_v3.py::test_zipper_lr_strategy`

---

#### Issue 5.1.7: Implement Warmup + Cosine Decay Scheduler

**File:** `src/modeling_studio/trainers/schedulers_v3.py`
**Effort:** 3 hours
**Dependencies:** Issue 5.1.2 (Trainer)

**Description:**
Implement learning rate schedulers with warmup and cosine decay for smooth training. Warmup prevents gradient shock at step 1, while cosine decay settles weights gently at the end of training.

**Implementation:**

```python
# src/modeling_studio/trainers/schedulers_v3.py

import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import List, Optional, Callable
import math
import logging

logger = logging.getLogger(__name__)


class WarmupCosineScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warmup and cosine decay.

    LR Profile:
        Warmup Phase (steps 0 to warmup_steps):
            lr = base_lr * (step / warmup_steps)

        Cosine Decay Phase (steps warmup_steps to total_steps):
            lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(π * progress))

        where progress = (step - warmup_steps) / (total_steps - warmup_steps)

    Example (2500 total, 500 warmup):
        Step 0:    lr = 0
        Step 250:  lr = base_lr * 0.5
        Step 500:  lr = base_lr (peak)
        Step 1500: lr = base_lr * 0.5
        Step 2500: lr = min_lr
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.01,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer: Wrapped optimizer
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            min_lr_ratio: Minimum LR as ratio of base LR (default 1%)
            last_epoch: The index of last epoch (for resuming)
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio

        # Store base LRs before calling parent __init__
        self.base_lrs_list = [group["lr"] for group in optimizer.param_groups]

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Calculate learning rate for current step."""
        step = self.last_epoch

        if step < self.warmup_steps:
            # Linear warmup
            warmup_factor = step / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        elif step >= self.total_steps:
            # After training complete
            return [base_lr * self.min_lr_ratio for base_lr in self.base_lrs]

        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))

            # Interpolate between base_lr and min_lr
            return [
                base_lr * self.min_lr_ratio + (base_lr - base_lr * self.min_lr_ratio) * cosine_factor
                for base_lr in self.base_lrs
            ]


class WarmupLinearScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warmup and linear decay.

    Simpler than cosine but can be effective for shorter training runs.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Calculate learning rate for current step."""
        step = self.last_epoch

        if step < self.warmup_steps:
            # Linear warmup
            warmup_factor = step / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        else:
            # Linear decay
            decay_steps = self.total_steps - self.warmup_steps
            steps_since_warmup = step - self.warmup_steps
            decay_factor = 1.0 - (steps_since_warmup / max(1, decay_steps))
            decay_factor = max(self.min_lr_ratio, decay_factor)

            return [base_lr * decay_factor for base_lr in self.base_lrs]


class WarmupConstantScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warmup then constant LR.

    Useful for short fine-tuning runs where decay isn't beneficial.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Calculate learning rate for current step."""
        step = self.last_epoch

        if step < self.warmup_steps:
            warmup_factor = step / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            return list(self.base_lrs)


class PhaseAwareScheduler:
    """
    Scheduler that handles phase transitions in v3 training.

    Manages separate schedulers for each phase and handles
    transitions between phases.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        phase_configs: dict,
    ):
        """
        Args:
            optimizer: Wrapped optimizer
            phase_configs: Dict of phase_name -> config dict
                Each config: {warmup_steps, total_steps, scheduler_type, min_lr_ratio}
        """
        self.optimizer = optimizer
        self.phase_configs = phase_configs
        self.current_phase = None
        self.current_scheduler = None
        self.phase_step = 0

    def set_phase(self, phase: str) -> None:
        """
        Switch to a new training phase.

        Args:
            phase: Phase name (e.g., "phase_0.5", "phase_1")
        """
        if phase not in self.phase_configs:
            raise ValueError(f"Unknown phase: {phase}")

        config = self.phase_configs[phase]
        self.current_phase = phase
        self.phase_step = 0

        # Create scheduler for this phase
        scheduler_type = config.get("scheduler_type", "cosine")
        warmup_steps = config.get("warmup_steps", 500)
        total_steps = config.get("total_steps", 2500)
        min_lr_ratio = config.get("min_lr_ratio", 0.01)

        if scheduler_type == "cosine":
            self.current_scheduler = WarmupCosineScheduler(
                self.optimizer,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                min_lr_ratio=min_lr_ratio,
            )
        elif scheduler_type == "linear":
            self.current_scheduler = WarmupLinearScheduler(
                self.optimizer,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                min_lr_ratio=min_lr_ratio,
            )
        elif scheduler_type == "constant":
            self.current_scheduler = WarmupConstantScheduler(
                self.optimizer,
                warmup_steps=warmup_steps,
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        logger.info(f"Switched to {phase} with {scheduler_type} scheduler")
        logger.info(f"  Warmup: {warmup_steps} steps, Total: {total_steps} steps")

    def step(self) -> None:
        """Advance scheduler by one step."""
        if self.current_scheduler is not None:
            self.current_scheduler.step()
            self.phase_step += 1

    def get_last_lr(self) -> List[float]:
        """Get current learning rates."""
        if self.current_scheduler is not None:
            return self.current_scheduler.get_last_lr()
        return [group["lr"] for group in self.optimizer.param_groups]


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    warmup_steps: int = 500,
    total_steps: int = 2500,
    min_lr_ratio: float = 0.01,
) -> _LRScheduler:
    """
    Create a learning rate scheduler.

    Args:
        optimizer: Wrapped optimizer
        scheduler_type: "cosine", "linear", or "constant"
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr_ratio: Minimum LR as ratio of peak

    Returns:
        Configured scheduler
    """
    if scheduler_type == "cosine":
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=min_lr_ratio,
        )
    elif scheduler_type == "linear":
        scheduler = WarmupLinearScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=min_lr_ratio,
        )
    elif scheduler_type == "constant":
        scheduler = WarmupConstantScheduler(
            optimizer,
            warmup_steps=warmup_steps,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    logger.info(f"Created {scheduler_type} scheduler:")
    logger.info(f"  Warmup steps: {warmup_steps}")
    logger.info(f"  Total steps: {total_steps}")
    if scheduler_type != "constant":
        logger.info(f"  Min LR ratio: {min_lr_ratio}")

    return scheduler


# Default phase configurations
DEFAULT_PHASE_SCHEDULER_CONFIGS = {
    "phase_0.5": {
        "scheduler_type": "cosine",
        "warmup_steps": 500,
        "total_steps": 2500,
        "min_lr_ratio": 0.01,
    },
    "phase_1": {
        "scheduler_type": "cosine",
        "warmup_steps": 1000,
        "total_steps": 5000,
        "min_lr_ratio": 0.01,
    },
    "phase_2": {
        "scheduler_type": "cosine",
        "warmup_steps": 200,
        "total_steps": 1000,
        "min_lr_ratio": 0.1,
    },
}
```

**Acceptance Criteria:**

- [ ] `WarmupCosineScheduler` implements warmup + cosine correctly
- [ ] LR starts at 0, peaks at warmup_steps, decays to min_lr
- [ ] `WarmupLinearScheduler` provides linear alternative
- [ ] `WarmupConstantScheduler` for short runs
- [ ] `PhaseAwareScheduler` handles phase transitions
- [ ] `create_scheduler()` factory function works
- [ ] Compatible with per-layer-group LRs

**Tests:** `tests/v3/test_trainer_v3.py::test_warmup_cosine_scheduler`

---

#### Issue 5.1.8: Implement Gradient Clipping for Phase 0.5

**File:** `src/modeling_studio/trainers/gradient_utils_v3.py`
**Effort:** 3 hours
**Dependencies:** Issue 5.1.2 (Trainer)

**Description:**
Implement gradient clipping and monitoring utilities for Phase 0.5 training. The L22→L23 interface is particularly sensitive to gradient explosions during healing, requiring careful gradient management.

**Implementation:**

```python
# src/modeling_studio/trainers/gradient_utils_v3.py

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class GradientClipConfig:
    """Configuration for gradient clipping."""
    # Global gradient clipping
    max_grad_norm: float = 1.0

    # Per-layer gradient clipping (optional, more fine-grained)
    per_layer_clip: bool = False
    interface_clip: float = 0.5      # L23: tighter clip at interface
    family_clip: float = 1.0         # L24-28
    feeder_clip: float = 1.0         # L19-22

    # Gradient monitoring
    log_grad_norms: bool = True
    log_every_n_steps: int = 100

    # Gradient explosion detection
    explosion_threshold: float = 10.0  # Warn if grad norm > threshold
    nan_check: bool = True             # Check for NaN gradients


@dataclass
class GradientStats:
    """Statistics about gradients."""
    total_norm: float = 0.0
    layer_norms: Dict[str, float] = field(default_factory=dict)
    max_grad: float = 0.0
    min_grad: float = 0.0
    has_nan: bool = False
    has_inf: bool = False
    clipped: bool = False


class GradientClipper:
    """
    Gradient clipping and monitoring for v3 training.

    Provides:
    1. Global gradient clipping (standard)
    2. Per-layer gradient clipping (for interface sensitivity)
    3. Gradient norm monitoring
    4. NaN/Inf detection
    5. Gradient explosion warnings
    """

    def __init__(
        self,
        model: nn.Module,
        config: GradientClipConfig,
    ):
        self.model = model
        self.config = config
        self.encoder = model.encoder if hasattr(model, "encoder") else model

        # Tracking
        self.step = 0
        self.gradient_history: List[GradientStats] = []
        self.explosion_count = 0

    def clip_gradients(self) -> GradientStats:
        """
        Clip gradients and return statistics.

        Returns:
            GradientStats with clipping info
        """
        stats = GradientStats()

        # Check for NaN/Inf first
        if self.config.nan_check:
            stats.has_nan, stats.has_inf = self._check_gradient_health()
            if stats.has_nan or stats.has_inf:
                logger.warning(f"Step {self.step}: NaN={stats.has_nan}, Inf={stats.has_inf}")
                self._zero_bad_gradients()

        # Calculate gradient norms per layer
        if self.config.log_grad_norms:
            stats.layer_norms = self._compute_layer_norms()

        # Apply clipping
        if self.config.per_layer_clip:
            stats = self._per_layer_clip(stats)
        else:
            stats = self._global_clip(stats)

        # Check for gradient explosion
        if stats.total_norm > self.config.explosion_threshold:
            self.explosion_count += 1
            logger.warning(
                f"Step {self.step}: Gradient explosion detected! "
                f"Norm={stats.total_norm:.2f} > {self.config.explosion_threshold}"
            )

        # Log periodically
        if self.config.log_grad_norms and self.step % self.config.log_every_n_steps == 0:
            self._log_gradient_stats(stats)

        self.step += 1
        self.gradient_history.append(stats)

        return stats

    def _check_gradient_health(self) -> Tuple[bool, bool]:
        """Check for NaN or Inf gradients."""
        has_nan = False
        has_inf = False

        for param in self.model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    has_nan = True
                if torch.isinf(param.grad).any():
                    has_inf = True

            if has_nan and has_inf:
                break

        return has_nan, has_inf

    def _zero_bad_gradients(self) -> int:
        """Zero out NaN and Inf gradients."""
        zeroed = 0
        for param in self.model.parameters():
            if param.grad is not None:
                bad_mask = torch.isnan(param.grad) | torch.isinf(param.grad)
                if bad_mask.any():
                    param.grad[bad_mask] = 0.0
                    zeroed += bad_mask.sum().item()
        return zeroed

    def _compute_layer_norms(self) -> Dict[str, float]:
        """Compute gradient norm per layer."""
        layer_norms = {}

        for layer_idx in range(len(self.encoder.layers)):
            layer = self.encoder.layers[layer_idx]
            layer_grad_norm = 0.0

            for param in layer.parameters():
                if param.grad is not None:
                    layer_grad_norm += param.grad.data.norm(2).item() ** 2

            layer_norms[f"layer_{layer_idx + 1}"] = math.sqrt(layer_grad_norm)

        return layer_norms

    def _global_clip(self, stats: GradientStats) -> GradientStats:
        """Apply global gradient clipping."""
        # Get all parameters with gradients
        params = [p for p in self.model.parameters() if p.grad is not None]

        if not params:
            return stats

        # Compute total norm
        total_norm = torch.nn.utils.clip_grad_norm_(
            params,
            max_norm=self.config.max_grad_norm,
        )

        stats.total_norm = total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm
        stats.clipped = stats.total_norm > self.config.max_grad_norm

        return stats

    def _per_layer_clip(self, stats: GradientStats) -> GradientStats:
        """Apply per-layer gradient clipping."""
        total_norm_sq = 0.0

        for layer_idx in range(len(self.encoder.layers)):
            layer = self.encoder.layers[layer_idx]

            # Determine clip value based on layer position
            if layer_idx == 22:  # Interface layer (L23)
                max_norm = self.config.interface_clip
            elif layer_idx >= 23:  # Family band (L24-28)
                max_norm = self.config.family_clip
            elif layer_idx >= 18:  # Feeder band (L19-22)
                max_norm = self.config.feeder_clip
            else:  # Foundation/Core (should be frozen)
                max_norm = self.config.max_grad_norm

            # Clip this layer
            layer_params = [p for p in layer.parameters() if p.grad is not None]
            if layer_params:
                layer_norm = torch.nn.utils.clip_grad_norm_(
                    layer_params,
                    max_norm=max_norm,
                )
                total_norm_sq += layer_norm.item() ** 2

        stats.total_norm = math.sqrt(total_norm_sq)
        stats.clipped = True  # Per-layer always applies clipping

        return stats

    def _log_gradient_stats(self, stats: GradientStats) -> None:
        """Log gradient statistics."""
        logger.info(f"Step {self.step} gradient stats:")
        logger.info(f"  Total norm: {stats.total_norm:.4f}")
        logger.info(f"  Clipped: {stats.clipped}")

        if stats.layer_norms:
            # Show key layers
            for key in ["layer_22", "layer_23", "layer_24", "layer_28"]:
                if key in stats.layer_norms:
                    logger.info(f"  {key}: {stats.layer_norms[key]:.4f}")

    def get_gradient_summary(self) -> Dict:
        """Get summary of gradient history."""
        if not self.gradient_history:
            return {}

        norms = [s.total_norm for s in self.gradient_history]
        clipped = sum(1 for s in self.gradient_history if s.clipped)

        return {
            "mean_norm": sum(norms) / len(norms),
            "max_norm": max(norms),
            "min_norm": min(norms),
            "clip_count": clipped,
            "clip_ratio": clipped / len(self.gradient_history),
            "explosion_count": self.explosion_count,
        }


class InterfaceGradientMonitor:
    """
    Specialized monitor for L22→L23 interface gradients.

    The interface between v2 (L22) and v3 (L23) is the most sensitive
    region during healing. This monitor tracks gradient flow across
    this boundary.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.encoder = model.encoder if hasattr(model, "encoder") else model
        self.history: List[Dict] = []

    def record(self) -> Dict[str, float]:
        """Record gradient statistics at interface."""
        stats = {}

        # L22 (last v2 layer) gradients
        l22 = self.encoder.layers[21]
        l22_norm = self._layer_grad_norm(l22)
        stats["l22_grad_norm"] = l22_norm

        # L23 (first v3 layer) gradients
        l23 = self.encoder.layers[22]
        l23_norm = self._layer_grad_norm(l23)
        stats["l23_grad_norm"] = l23_norm

        # Ratio (measures gradient flow)
        if l22_norm > 0:
            stats["l23_l22_ratio"] = l23_norm / l22_norm
        else:
            stats["l23_l22_ratio"] = 0.0

        # Gradient alignment (cosine similarity between layer outputs)
        # This would require hooks, simplified here
        stats["interface_healthy"] = 0.1 < stats["l23_l22_ratio"] < 10.0

        self.history.append(stats)
        return stats

    def _layer_grad_norm(self, layer: nn.Module) -> float:
        """Compute gradient L2 norm for a layer."""
        norm_sq = 0.0
        for param in layer.parameters():
            if param.grad is not None:
                norm_sq += param.grad.data.norm(2).item() ** 2
        return math.sqrt(norm_sq)

    def get_interface_health(self) -> Dict:
        """Get interface health summary."""
        if not self.history:
            return {"healthy": True, "message": "No data yet"}

        healthy_count = sum(1 for h in self.history if h.get("interface_healthy", False))
        health_ratio = healthy_count / len(self.history)

        ratios = [h["l23_l22_ratio"] for h in self.history if "l23_l22_ratio" in h]
        mean_ratio = sum(ratios) / len(ratios) if ratios else 0.0

        return {
            "healthy": health_ratio > 0.9,
            "health_ratio": health_ratio,
            "mean_l23_l22_ratio": mean_ratio,
            "message": "OK" if health_ratio > 0.9 else "WARNING: Interface gradient imbalance",
        }


def clip_gradients(
    model: nn.Module,
    max_norm: float = 1.0,
    per_layer: bool = False,
) -> float:
    """
    Clip gradients for a model.

    Args:
        model: Model to clip
        max_norm: Maximum gradient norm
        per_layer: Whether to clip per-layer

    Returns:
        Total gradient norm before clipping
    """
    config = GradientClipConfig(
        max_grad_norm=max_norm,
        per_layer_clip=per_layer,
        log_grad_norms=False,
    )

    clipper = GradientClipper(model, config)
    stats = clipper.clip_gradients()

    return stats.total_norm
```

**Acceptance Criteria:**

- [ ] `GradientClipper` implements global clipping (max_norm=1.0)
- [ ] Per-layer clipping applies tighter clip to L23 (0.5)
- [ ] NaN/Inf gradient detection and zeroing
- [ ] Gradient explosion warnings logged
- [ ] `InterfaceGradientMonitor` tracks L22→L23 gradient flow
- [ ] Gradient statistics logged periodically
- [ ] `clip_gradients()` convenience function works
- [ ] No memory leaks from gradient history

**Tests:** `tests/v3/test_trainer_v3.py::test_gradient_clipping`

---

### Epic 5.2: Enhanced Healing Data Pipeline

#### Issue 5.2.1: Implement v3 Collators with Hub Token Offsets

**File:** `src/modeling_studio/data/collators_v3.py`
**Effort:** 4 hours
**Dependencies:** Issue 1.1.4 (Hub Tokens)

**Description:**
Implement data collators that handle the v3 token layout with hub token positions. The collator must shift token positions to account for [CLS] + 4 hub tokens at the start.

**Implementation:**

```python
# src/modeling_studio/data/collators_v3.py

import torch
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# V3 Token Layout Constants
HUB_TOKEN_COUNT = 4
V3_SPECIAL_PREFIX_LEN = 5  # [CLS] + [EMO] + [MEM] + [REL] + [TASK]

# Position mapping
POSITION_CLS = 0
POSITION_EMO = 1
POSITION_MEM = 2
POSITION_REL = 3
POSITION_TASK = 4
POSITION_TEXT_START = 5


@dataclass
class V3CollatorConfig:
    """Configuration for v3 collators."""
    # Tokenizer settings
    max_length: int = 512
    padding: str = "max_length"
    truncation: bool = True

    # Hub token handling
    include_hub_tokens: bool = True
    hub_token_ids: Dict[str, int] = None  # Populated from tokenizer

    # Task-specific settings
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __post_init__(self):
        if self.hub_token_ids is None:
            self.hub_token_ids = {
                "[EMO]": 50368,
                "[MEM]": 50369,
                "[REL]": 50370,
                "[TASK]": 50371,
            }


class V3BaseCollator:
    """
    Base collator for v3 models with hub token support.

    Handles the v3 token layout:
        [CLS] [EMO] [MEM] [REL] [TASK] <text tokens> [SEP] [PAD]...

    All position-based labels (NER, etc.) must be offset by +5 to
    account for the hub token prefix.
    """

    def __init__(
        self,
        tokenizer,
        config: Optional[V3CollatorConfig] = None,
    ):
        self.tokenizer = tokenizer
        self.config = config or V3CollatorConfig()

        # Validate tokenizer has hub tokens
        self._validate_tokenizer()

    def _validate_tokenizer(self) -> None:
        """Ensure tokenizer has v3 hub tokens."""
        vocab = self.tokenizer.get_vocab()
        for token_name, expected_id in self.config.hub_token_ids.items():
            if token_name not in vocab:
                logger.warning(f"Hub token {token_name} not in tokenizer vocab")
            elif vocab[token_name] != expected_id:
                logger.warning(
                    f"Hub token {token_name} has ID {vocab[token_name]}, "
                    f"expected {expected_id}"
                )

    def _add_hub_tokens(
        self,
        input_ids: List[int],
        attention_mask: List[int],
    ) -> tuple:
        """
        Insert hub tokens after [CLS].

        Input:  [CLS] <text> [SEP] [PAD]...
        Output: [CLS] [EMO] [MEM] [REL] [TASK] <text> [SEP] [PAD]...
        """
        cls_id = self.tokenizer.cls_token_id
        hub_ids = list(self.config.hub_token_ids.values())

        # Find [CLS] position (should be 0)
        if input_ids[0] != cls_id:
            logger.warning("First token is not [CLS], inserting hub tokens at position 1")
            insert_pos = 1
        else:
            insert_pos = 1  # After [CLS]

        # Insert hub tokens
        new_input_ids = (
            input_ids[:insert_pos] +
            hub_ids +
            input_ids[insert_pos:]
        )

        # Extend attention mask (hub tokens are always attended)
        new_attention_mask = (
            attention_mask[:insert_pos] +
            [1] * HUB_TOKEN_COUNT +
            attention_mask[insert_pos:]
        )

        # Truncate to max_length if needed
        if len(new_input_ids) > self.config.max_length:
            new_input_ids = new_input_ids[:self.config.max_length]
            new_attention_mask = new_attention_mask[:self.config.max_length]

            # Ensure [SEP] at end
            sep_id = self.tokenizer.sep_token_id
            if new_input_ids[-1] != sep_id:
                new_input_ids[-1] = sep_id

        return new_input_ids, new_attention_mask

    def _offset_labels(
        self,
        labels: List[int],
        offset: int = V3_SPECIAL_PREFIX_LEN,
    ) -> List[int]:
        """
        Offset position-based labels for hub token prefix.

        For NER/token classification, label positions must shift by +5.
        """
        # Prepend ignore labels for [CLS] + hub tokens
        hub_labels = [self.config.label_pad_token_id] * offset
        return hub_labels + labels

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate features into batch."""
        raise NotImplementedError("Subclasses must implement __call__")


class V3ClassificationCollator(V3BaseCollator):
    """
    Collator for sequence classification tasks (sentiment, safety, etc.).

    No label offsetting needed - just single label per sequence.
    """

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate classification features."""
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        for feature in features:
            input_ids = feature["input_ids"]
            attention_mask = feature.get("attention_mask", [1] * len(input_ids))

            # Add hub tokens
            if self.config.include_hub_tokens:
                input_ids, attention_mask = self._add_hub_tokens(
                    input_ids, attention_mask
                )

            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)

            if "label" in feature:
                batch["labels"].append(feature["label"])
            elif "labels" in feature:
                batch["labels"].append(feature["labels"])

        # Pad and convert to tensors
        batch = self._pad_batch(batch)
        return batch

    def _pad_batch(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Pad batch to uniform length."""
        max_len = max(len(ids) for ids in batch["input_ids"])

        padded_input_ids = []
        padded_attention_mask = []

        for input_ids, attn_mask in zip(batch["input_ids"], batch["attention_mask"]):
            pad_len = max_len - len(input_ids)
            padded_input_ids.append(
                input_ids + [self.tokenizer.pad_token_id] * pad_len
            )
            padded_attention_mask.append(attn_mask + [0] * pad_len)

        result = {
            "input_ids": torch.tensor(padded_input_ids),
            "attention_mask": torch.tensor(padded_attention_mask),
        }

        if batch["labels"]:
            result["labels"] = torch.tensor(batch["labels"])

        return result


class V3TokenClassificationCollator(V3BaseCollator):
    """
    Collator for token classification tasks (NER, etc.).

    Labels must be offset by +5 for hub token prefix.
    """

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate token classification features."""
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        for feature in features:
            input_ids = feature["input_ids"]
            attention_mask = feature.get("attention_mask", [1] * len(input_ids))
            labels = feature.get("labels", feature.get("ner_tags", []))

            # Add hub tokens
            if self.config.include_hub_tokens:
                input_ids, attention_mask = self._add_hub_tokens(
                    input_ids, attention_mask
                )
                # Offset labels for hub tokens
                labels = self._offset_labels(labels)

            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(labels)

        # Pad and convert to tensors
        batch = self._pad_batch(batch)
        return batch

    def _pad_batch(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Pad batch to uniform length."""
        max_len = max(len(ids) for ids in batch["input_ids"])

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        pad_id = self.tokenizer.pad_token_id
        label_pad = self.config.label_pad_token_id

        for input_ids, attn_mask, labels in zip(
            batch["input_ids"], batch["attention_mask"], batch["labels"]
        ):
            pad_len = max_len - len(input_ids)
            padded_input_ids.append(input_ids + [pad_id] * pad_len)
            padded_attention_mask.append(attn_mask + [0] * pad_len)
            padded_labels.append(labels + [label_pad] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids),
            "attention_mask": torch.tensor(padded_attention_mask),
            "labels": torch.tensor(padded_labels),
        }


class V3MultiTaskCollator(V3BaseCollator):
    """
    Collator for multi-task training with multiple label types.

    Handles unified samples with multiple task labels.
    """

    def __init__(
        self,
        tokenizer,
        config: Optional[V3CollatorConfig] = None,
        task_configs: Optional[Dict[str, Dict]] = None,
    ):
        super().__init__(tokenizer, config)

        # Task-specific label handling
        self.task_configs = task_configs or {
            "sentiment": {"type": "classification", "num_labels": 3},
            "emotions": {"type": "multilabel", "num_labels": 8},
            "safety": {"type": "classification", "num_labels": 3},
            "ner": {"type": "token_classification", "num_labels": 9},
            "intent": {"type": "classification", "num_labels": 12},
            "ingress": {"type": "classification", "num_labels": 4},
        }

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate multi-task features."""
        batch = {
            "input_ids": [],
            "attention_mask": [],
        }

        # Initialize task label lists
        for task_name in self.task_configs:
            batch[f"{task_name}_labels"] = []

        for feature in features:
            input_ids = feature["input_ids"]
            attention_mask = feature.get("attention_mask", [1] * len(input_ids))

            # Add hub tokens
            if self.config.include_hub_tokens:
                input_ids, attention_mask = self._add_hub_tokens(
                    input_ids, attention_mask
                )

            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)

            # Extract task-specific labels
            tasks = feature.get("tasks", {})
            for task_name, task_config in self.task_configs.items():
                if task_name in tasks:
                    label = tasks[task_name]
                    if task_config["type"] == "token_classification":
                        label = self._offset_labels(label)
                    batch[f"{task_name}_labels"].append(label)
                else:
                    # Missing task - use ignore index
                    batch[f"{task_name}_labels"].append(None)

        # Pad and convert to tensors
        batch = self._pad_multitask_batch(batch)
        return batch

    def _pad_multitask_batch(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Pad multi-task batch."""
        max_len = max(len(ids) for ids in batch["input_ids"])

        # Pad input_ids and attention_mask
        padded_input_ids = []
        padded_attention_mask = []

        for input_ids, attn_mask in zip(batch["input_ids"], batch["attention_mask"]):
            pad_len = max_len - len(input_ids)
            padded_input_ids.append(
                input_ids + [self.tokenizer.pad_token_id] * pad_len
            )
            padded_attention_mask.append(attn_mask + [0] * pad_len)

        result = {
            "input_ids": torch.tensor(padded_input_ids),
            "attention_mask": torch.tensor(padded_attention_mask),
        }

        # Pad task labels
        for task_name, task_config in self.task_configs.items():
            labels = batch[f"{task_name}_labels"]

            if task_config["type"] == "token_classification":
                # Pad token-level labels
                padded_labels = []
                for label in labels:
                    if label is None:
                        padded_labels.append([self.config.label_pad_token_id] * max_len)
                    else:
                        pad_len = max_len - len(label)
                        padded_labels.append(
                            label + [self.config.label_pad_token_id] * pad_len
                        )
                result[f"{task_name}_labels"] = torch.tensor(padded_labels)
            else:
                # Sequence-level labels
                processed_labels = [
                    l if l is not None else self.config.label_pad_token_id
                    for l in labels
                ]
                result[f"{task_name}_labels"] = torch.tensor(processed_labels)

        return result


def create_v3_collator(
    tokenizer,
    task_type: str = "classification",
    **kwargs,
) -> V3BaseCollator:
    """
    Factory function to create appropriate v3 collator.

    Args:
        tokenizer: Tokenizer with v3 hub tokens
        task_type: "classification", "token_classification", or "multitask"
        **kwargs: Additional config options

    Returns:
        Appropriate collator instance
    """
    config = V3CollatorConfig(**kwargs)

    if task_type == "classification":
        return V3ClassificationCollator(tokenizer, config)
    elif task_type == "token_classification":
        return V3TokenClassificationCollator(tokenizer, config)
    elif task_type == "multitask":
        return V3MultiTaskCollator(tokenizer, config)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
```

**Acceptance Criteria:**

- [ ] `V3BaseCollator` inserts hub tokens after [CLS]
- [ ] `V3ClassificationCollator` handles sentiment/safety tasks
- [ ] `V3TokenClassificationCollator` offsets NER labels by +5
- [ ] `V3MultiTaskCollator` handles unified samples
- [ ] Proper padding and truncation
- [ ] Label positions aligned with token positions
- [ ] Works with v3 tokenizer

**Tests:** `tests/v3/test_collators_v3.py::test_hub_token_insertion`

---

#### Issue 5.2.2: Implement Stage A Replay Sampler

**File:** `src/modeling_studio/data/replay_sampler_v3.py`
**Effort:** 5 hours
**Dependencies:** Issue 5.2.1

**Description:**
Implement a replay sampler that mixes healing data with new task data during training. This prevents catastrophic forgetting by replaying Stage A benchmark samples.

**Implementation:**

```python
# src/modeling_studio/data/replay_sampler_v3.py

import torch
from torch.utils.data import Sampler, Dataset, ConcatDataset
from typing import Iterator, List, Dict, Optional, Tuple
import random
import math
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class ReplayConfig:
    """Configuration for replay sampling."""

    def __init__(
        self,
        replay_ratio: float = 0.15,       # 15% of samples from replay buffer
        task_balanced: bool = True,        # Balance across replay tasks
        min_replay_per_epoch: int = 100,   # Minimum replay samples per epoch
        dynamic_ratio: bool = True,        # Adjust ratio based on loss
        loss_threshold: float = 0.5,       # Increase replay if loss > threshold
        max_replay_ratio: float = 0.3,     # Maximum replay ratio
    ):
        self.replay_ratio = replay_ratio
        self.task_balanced = task_balanced
        self.min_replay_per_epoch = min_replay_per_epoch
        self.dynamic_ratio = dynamic_ratio
        self.loss_threshold = loss_threshold
        self.max_replay_ratio = max_replay_ratio


class ReplaySampler(Sampler):
    """
    Sampler that mixes primary training data with replay data.

    The replay mechanism ensures v3 doesn't forget Stage A capabilities
    by periodically sampling from SST-2, CoNLL, MNLI during training.

    Sampling Strategy:
        1. For each batch, select (1 - replay_ratio) samples from primary data
        2. Select replay_ratio samples from replay buffer
        3. If task_balanced, ensure equal representation across replay tasks

    Dynamic Adjustment:
        If forgetting_loss > threshold, automatically increase replay_ratio
    """

    def __init__(
        self,
        primary_dataset: Dataset,
        replay_dataset: Dataset,
        config: Optional[ReplayConfig] = None,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.primary_dataset = primary_dataset
        self.replay_dataset = replay_dataset
        self.config = config or ReplayConfig()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        # Current ratio (can be adjusted dynamically)
        self.current_replay_ratio = self.config.replay_ratio

        # Calculate sample counts
        self._calculate_sample_counts()

        # RNG for reproducibility
        self.rng = random.Random(seed)

        # Task indices for balanced sampling
        self._build_task_indices()

        logger.info(f"ReplaySampler initialized:")
        logger.info(f"  Primary samples: {len(self.primary_dataset)}")
        logger.info(f"  Replay samples: {len(self.replay_dataset)}")
        logger.info(f"  Replay ratio: {self.current_replay_ratio:.2%}")

    def _calculate_sample_counts(self) -> None:
        """Calculate number of samples per epoch."""
        total_primary = len(self.primary_dataset)

        # Replay samples based on ratio
        n_replay = max(
            int(total_primary * self.current_replay_ratio / (1 - self.current_replay_ratio)),
            self.config.min_replay_per_epoch,
        )

        self.n_primary_per_epoch = total_primary
        self.n_replay_per_epoch = n_replay
        self.total_samples = self.n_primary_per_epoch + self.n_replay_per_epoch

    def _build_task_indices(self) -> None:
        """Build index mapping for task-balanced sampling."""
        if not self.config.task_balanced:
            self.task_indices = None
            return

        # Group replay samples by task
        self.task_indices = defaultdict(list)

        for idx in range(len(self.replay_dataset)):
            sample = self.replay_dataset[idx]
            task = sample.get("task", sample.get("task_name", "unknown"))
            self.task_indices[task].append(idx)

        logger.info(f"Replay task distribution:")
        for task, indices in self.task_indices.items():
            logger.info(f"  {task}: {len(indices)} samples")

    def _sample_replay_indices(self) -> List[int]:
        """Sample replay indices with optional task balancing."""
        if not self.config.task_balanced or self.task_indices is None:
            # Simple random sampling
            all_indices = list(range(len(self.replay_dataset)))
            return self.rng.sample(
                all_indices,
                min(self.n_replay_per_epoch, len(all_indices)),
            )

        # Task-balanced sampling
        tasks = list(self.task_indices.keys())
        samples_per_task = max(1, self.n_replay_per_epoch // len(tasks))

        replay_indices = []
        for task in tasks:
            task_pool = self.task_indices[task]
            n_samples = min(samples_per_task, len(task_pool))
            replay_indices.extend(self.rng.sample(task_pool, n_samples))

        # Fill remaining with random samples if needed
        remaining = self.n_replay_per_epoch - len(replay_indices)
        if remaining > 0:
            all_indices = list(range(len(self.replay_dataset)))
            extra = self.rng.sample(all_indices, min(remaining, len(all_indices)))
            replay_indices.extend(extra)

        return replay_indices[:self.n_replay_per_epoch]

    def __iter__(self) -> Iterator[Tuple[str, int]]:
        """
        Generate interleaved indices.

        Yields tuples of (source, index):
            - ("primary", idx) for primary dataset samples
            - ("replay", idx) for replay dataset samples
        """
        # Get all indices
        primary_indices = list(range(len(self.primary_dataset)))
        replay_indices = self._sample_replay_indices()

        if self.shuffle:
            self.rng.shuffle(primary_indices)
            self.rng.shuffle(replay_indices)

        # Create tagged indices
        tagged_primary = [("primary", idx) for idx in primary_indices]
        tagged_replay = [("replay", idx) for idx in replay_indices]

        # Interleave based on ratio
        all_indices = []
        p_ptr, r_ptr = 0, 0

        while p_ptr < len(tagged_primary) or r_ptr < len(tagged_replay):
            # Decide next sample source based on ratio
            if r_ptr >= len(tagged_replay):
                # No more replay samples
                all_indices.append(tagged_primary[p_ptr])
                p_ptr += 1
            elif p_ptr >= len(tagged_primary):
                # No more primary samples
                all_indices.append(tagged_replay[r_ptr])
                r_ptr += 1
            else:
                # Choose based on ratio
                if self.rng.random() < self.current_replay_ratio:
                    all_indices.append(tagged_replay[r_ptr])
                    r_ptr += 1
                else:
                    all_indices.append(tagged_primary[p_ptr])
                    p_ptr += 1

        for item in all_indices:
            yield item

    def __len__(self) -> int:
        return self.total_samples

    def update_replay_ratio(self, forgetting_loss: float) -> None:
        """
        Dynamically adjust replay ratio based on forgetting loss.

        If forgetting_loss is high, increase replay to prevent forgetting.
        """
        if not self.config.dynamic_ratio:
            return

        old_ratio = self.current_replay_ratio

        if forgetting_loss > self.config.loss_threshold:
            # Increase replay ratio
            self.current_replay_ratio = min(
                self.current_replay_ratio * 1.2,
                self.config.max_replay_ratio,
            )
            logger.info(
                f"Forgetting loss {forgetting_loss:.3f} > {self.config.loss_threshold}, "
                f"increasing replay ratio: {old_ratio:.2%} -> {self.current_replay_ratio:.2%}"
            )
        elif forgetting_loss < self.config.loss_threshold * 0.5:
            # Decrease replay ratio (learning is stable)
            self.current_replay_ratio = max(
                self.current_replay_ratio * 0.9,
                self.config.replay_ratio,  # Don't go below initial
            )

        # Recalculate sample counts
        self._calculate_sample_counts()


class ReplayDataset(Dataset):
    """
    Dataset wrapper that handles replay sampling.

    Combines primary and replay datasets with automatic source tracking.
    """

    def __init__(
        self,
        primary_dataset: Dataset,
        replay_dataset: Dataset,
        sampler: ReplaySampler,
    ):
        self.primary_dataset = primary_dataset
        self.replay_dataset = replay_dataset
        self.sampler = sampler

        # Pre-compute epoch indices
        self._epoch_indices: List[Tuple[str, int]] = []
        self._refresh_epoch()

    def _refresh_epoch(self) -> None:
        """Refresh indices for new epoch."""
        self._epoch_indices = list(self.sampler)
        self._current_idx = 0

    def __len__(self) -> int:
        return len(self._epoch_indices)

    def __getitem__(self, idx: int) -> Dict:
        """Get item by index."""
        source, source_idx = self._epoch_indices[idx]

        if source == "primary":
            item = self.primary_dataset[source_idx]
        else:
            item = self.replay_dataset[source_idx]

        # Add source info
        item["_source"] = source
        item["_is_replay"] = (source == "replay")

        return item


def create_replay_sampler(
    primary_dataset: Dataset,
    replay_dataset: Dataset,
    replay_ratio: float = 0.15,
    batch_size: int = 32,
    task_balanced: bool = True,
    **kwargs,
) -> Tuple[ReplayDataset, ReplaySampler]:
    """
    Create replay-enabled dataset and sampler.

    Args:
        primary_dataset: Main training dataset
        replay_dataset: Stage A replay dataset
        replay_ratio: Fraction of samples from replay
        batch_size: Training batch size
        task_balanced: Balance replay across tasks

    Returns:
        (ReplayDataset, ReplaySampler) tuple
    """
    config = ReplayConfig(
        replay_ratio=replay_ratio,
        task_balanced=task_balanced,
        **kwargs,
    )

    sampler = ReplaySampler(
        primary_dataset=primary_dataset,
        replay_dataset=replay_dataset,
        config=config,
        batch_size=batch_size,
    )

    dataset = ReplayDataset(
        primary_dataset=primary_dataset,
        replay_dataset=replay_dataset,
        sampler=sampler,
    )

    return dataset, sampler
```

**Acceptance Criteria:**

- [ ] `ReplaySampler` mixes primary and replay data correctly
- [ ] Task-balanced sampling ensures equal task representation
- [ ] `update_replay_ratio()` dynamically adjusts based on loss
- [ ] Interleaving creates well-mixed batches
- [ ] `ReplayDataset` tracks sample source
- [ ] Supports SST-2, CoNLL, MNLI replay tasks
- [ ] Minimum replay samples guaranteed per epoch

**Tests:** `tests/v3/test_replay_sampler.py::test_replay_ratio`

---

#### Issue 5.2.3: Implement Basic Healing Data Preparation Script

**File:** `scripts/prepare_healing_data.py`
**Effort:** 4 hours
**Dependencies:** None (uses HuggingFace datasets)

**Description:**
Create a script that prepares healing data from standard benchmarks (SST-2, CoNLL-2003, MNLI) for Phase 0.5 training. This data helps "heal" the cloned layers.

**Implementation:**

```python
#!/usr/bin/env python3
"""
Prepare healing data for ModernBERT v3 Phase 0.5.

This script downloads and preprocesses standard NLP benchmarks
to create a healing dataset that preserves v2 capabilities while
training the new v3 layers.

Usage:
    python scripts/prepare_healing_data.py --output data/healing/healing_generic.jsonl
    python scripts/prepare_healing_data.py --output data/healing --split-by-task
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict
import random

from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Healing data configuration
HEALING_CONFIG = {
    "sst2": {
        "hf_name": "glue",
        "hf_subset": "sst2",
        "split": "train",
        "n_samples": 3000,
        "task_type": "sentiment",
        "text_field": "sentence",
        "label_field": "label",
        "label_map": {0: "negative", 1: "positive"},
    },
    "conll": {
        "hf_name": "conll2003",
        "hf_subset": None,
        "split": "train",
        "n_samples": 3000,
        "task_type": "ner",
        "text_field": "tokens",
        "label_field": "ner_tags",
        "label_map": {
            0: "O",
            1: "B-PER", 2: "I-PER",
            3: "B-ORG", 4: "I-ORG",
            5: "B-LOC", 6: "I-LOC",
            7: "B-MISC", 8: "I-MISC",
        },
    },
    "mnli": {
        "hf_name": "glue",
        "hf_subset": "mnli",
        "split": "train",
        "n_samples": 4000,
        "task_type": "nli",
        "text_field": ["premise", "hypothesis"],
        "label_field": "label",
        "label_map": {0: "entailment", 1: "neutral", 2: "contradiction"},
    },
}

TOTAL_SAMPLES = sum(cfg["n_samples"] for cfg in HEALING_CONFIG.values())


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare healing data for v3 training"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/healing/healing_generic.jsonl",
        help="Output file or directory",
    )
    parser.add_argument(
        "--split-by-task",
        action="store_true",
        help="Create separate files per task",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate output after creation",
    )
    return parser.parse_args()


def load_and_sample_dataset(
    config: Dict,
    n_samples: int,
    seed: int,
) -> List[Dict]:
    """Load dataset from HuggingFace and sample."""
    logger.info(f"Loading {config['hf_name']}...")

    # Load dataset
    if config["hf_subset"]:
        dataset = load_dataset(
            config["hf_name"],
            config["hf_subset"],
            split=config["split"],
        )
    else:
        dataset = load_dataset(
            config["hf_name"],
            split=config["split"],
        )

    # Sample
    if len(dataset) > n_samples:
        dataset = dataset.shuffle(seed=seed).select(range(n_samples))

    return list(dataset)


def convert_sst2_sample(sample: Dict, config: Dict) -> Dict:
    """Convert SST-2 sample to unified format."""
    return {
        "text": sample[config["text_field"]],
        "task": "sentiment",
        "task_type": "classification",
        "labels": {
            "sentiment": sample[config["label_field"]],
            "sentiment_label": config["label_map"][sample[config["label_field"]]],
        },
        "source": "sst2",
        "split": "healing",
    }


def convert_conll_sample(sample: Dict, config: Dict) -> Dict:
    """Convert CoNLL-2003 sample to unified format."""
    tokens = sample[config["text_field"]]
    ner_tags = sample[config["label_field"]]

    # Convert to text
    text = " ".join(tokens)

    # Create span annotations
    spans = []
    current_entity = None
    current_start = None

    for i, (token, tag) in enumerate(zip(tokens, ner_tags)):
        tag_str = config["label_map"][tag]

        if tag_str.startswith("B-"):
            # End previous entity
            if current_entity:
                spans.append({
                    "start": current_start,
                    "end": i,
                    "type": current_entity,
                    "text": " ".join(tokens[current_start:i]),
                })
            # Start new entity
            current_entity = tag_str[2:]
            current_start = i

        elif tag_str.startswith("I-"):
            # Continue entity (if matching)
            if current_entity and tag_str[2:] != current_entity:
                # Mismatched I- tag, treat as new entity
                if current_entity:
                    spans.append({
                        "start": current_start,
                        "end": i,
                        "type": current_entity,
                        "text": " ".join(tokens[current_start:i]),
                    })
                current_entity = tag_str[2:]
                current_start = i

        else:  # O tag
            if current_entity:
                spans.append({
                    "start": current_start,
                    "end": i,
                    "type": current_entity,
                    "text": " ".join(tokens[current_start:i]),
                })
                current_entity = None
                current_start = None

    # Handle final entity
    if current_entity:
        spans.append({
            "start": current_start,
            "end": len(tokens),
            "type": current_entity,
            "text": " ".join(tokens[current_start:]),
        })

    return {
        "text": text,
        "tokens": tokens,
        "task": "ner",
        "task_type": "token_classification",
        "labels": {
            "ner_tags": ner_tags,
            "spans": spans,
        },
        "source": "conll2003",
        "split": "healing",
    }


def convert_mnli_sample(sample: Dict, config: Dict) -> Dict:
    """Convert MNLI sample to unified format."""
    premise = sample["premise"]
    hypothesis = sample["hypothesis"]

    return {
        "text": f"{premise} [SEP] {hypothesis}",
        "premise": premise,
        "hypothesis": hypothesis,
        "task": "nli",
        "task_type": "classification",
        "labels": {
            "nli": sample[config["label_field"]],
            "nli_label": config["label_map"][sample[config["label_field"]]],
        },
        "source": "mnli",
        "split": "healing",
    }


def prepare_healing_data(seed: int = 42) -> Dict[str, List[Dict]]:
    """Prepare all healing datasets."""
    random.seed(seed)

    healing_data = {}

    for task_name, config in HEALING_CONFIG.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {task_name}...")
        logger.info(f"{'='*50}")

        # Load and sample
        raw_samples = load_and_sample_dataset(
            config,
            config["n_samples"],
            seed,
        )

        # Convert to unified format
        converted = []
        converter = {
            "sst2": convert_sst2_sample,
            "conll": convert_conll_sample,
            "mnli": convert_mnli_sample,
        }[task_name]

        for sample in tqdm(raw_samples, desc=f"Converting {task_name}"):
            try:
                converted.append(converter(sample, config))
            except Exception as e:
                logger.warning(f"Failed to convert sample: {e}")
                continue

        healing_data[task_name] = converted
        logger.info(f"  Converted {len(converted)} samples")

    return healing_data


def save_healing_data(
    healing_data: Dict[str, List[Dict]],
    output_path: str,
    split_by_task: bool = False,
) -> None:
    """Save healing data to disk."""
    output = Path(output_path)

    if split_by_task:
        # Create directory and save per-task files
        output.mkdir(parents=True, exist_ok=True)

        for task_name, samples in healing_data.items():
            task_file = output / f"healing_{task_name}.jsonl"
            with open(task_file, "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample) + "\n")
            logger.info(f"Saved {len(samples)} samples to {task_file}")

    else:
        # Single file with all samples
        output.parent.mkdir(parents=True, exist_ok=True)

        all_samples = []
        for samples in healing_data.values():
            all_samples.extend(samples)

        # Shuffle for mixed training
        random.shuffle(all_samples)

        with open(output, "w") as f:
            for sample in all_samples:
                f.write(json.dumps(sample) + "\n")

        logger.info(f"Saved {len(all_samples)} samples to {output}")


def validate_healing_data(output_path: str) -> bool:
    """Validate the created healing data."""
    output = Path(output_path)

    if output.is_dir():
        files = list(output.glob("healing_*.jsonl"))
    else:
        files = [output]

    total_samples = 0
    task_counts = defaultdict(int)

    for file_path in files:
        with open(file_path) as f:
            for line in f:
                sample = json.loads(line)
                total_samples += 1
                task_counts[sample["task"]] += 1

    logger.info("\n" + "=" * 50)
    logger.info("Validation Results:")
    logger.info("=" * 50)
    logger.info(f"Total samples: {total_samples}")
    for task, count in sorted(task_counts.items()):
        logger.info(f"  {task}: {count}")

    # Check expected counts
    expected = TOTAL_SAMPLES
    if total_samples < expected * 0.95:
        logger.warning(f"Lower than expected sample count: {total_samples} < {expected}")
        return False

    logger.info("✅ Validation passed!")
    return True


def main() -> int:
    """Main entry point."""
    args = parse_args()

    print("\n" + "=" * 60)
    print("ModernBERT v3 Healing Data Preparation")
    print("=" * 60)
    print(f"Output: {args.output}")
    print(f"Split by task: {args.split_by_task}")
    print(f"Seed: {args.seed}")
    print()

    # Prepare data
    healing_data = prepare_healing_data(seed=args.seed)

    # Save
    save_healing_data(
        healing_data,
        args.output,
        split_by_task=args.split_by_task,
    )

    # Validate
    if args.validate:
        if not validate_healing_data(args.output):
            return 1

    print("\n" + "=" * 60)
    print("✅ Healing data preparation complete!")
    print("=" * 60)

    # Print summary
    total = sum(len(samples) for samples in healing_data.values())
    print(f"\nTotal samples: {total}")
    for task, samples in healing_data.items():
        print(f"  {task}: {len(samples)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Usage:**

```bash
# Create unified healing data file
python scripts/prepare_healing_data.py \
    --output data/healing/healing_generic.jsonl \
    --validate

# Create separate files per task
python scripts/prepare_healing_data.py \
    --output data/healing \
    --split-by-task \
    --validate
```

**Output Format:**

```json
{
    "text": "This movie is absolutely fantastic!",
    "task": "sentiment",
    "task_type": "classification",
    "labels": {"sentiment": 1, "sentiment_label": "positive"},
    "source": "sst2",
    "split": "healing"
}
```

**Acceptance Criteria:**

- [ ] Downloads SST-2, CoNLL-2003, MNLI from HuggingFace
- [ ] Samples 3000/3000/4000 samples respectively
- [ ] Converts to unified JSONL format
- [ ] NER samples include both tags and spans
- [ ] MNLI samples include premise/hypothesis
- [ ] Output is shuffled for mixed training
- [ ] Validation checks sample counts
- [ ] Supports split-by-task mode

**Tests:** `tests/v3/test_data_prep.py::test_healing_data_script`

---

#### Issue 5.2.4: Implement Enhanced Healing Data Preparation Script

**File:** `scripts/prepare_healing_data_enhanced.py`
**Effort:** 5 hours
**Dependencies:** Issue 5.2.3

**Description:**
Create an enhanced healing data script that adds SQuAD and STS-B to the healing mix. SQuAD heals attention patterns for long-range dependencies, while STS-B prevents embedding collapse.

**Implementation:**

```python
#!/usr/bin/env python3
"""
Prepare ENHANCED healing data for ModernBERT v3 Phase 0.5.

This extends the basic healing script with:
- SQuAD: Question answering (heals attention for context understanding)
- STS-B: Semantic similarity (prevents embedding collapse)

Usage:
    python scripts/prepare_healing_data_enhanced.py \
        --output data/healing/healing_enhanced.jsonl \
        --validate
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict
import random

from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Enhanced healing data configuration (5 tasks, 12k samples)
ENHANCED_HEALING_CONFIG = {
    "sst2": {
        "hf_name": "glue",
        "hf_subset": "sst2",
        "split": "train",
        "n_samples": 3000,
        "task_type": "sentiment",
        "text_field": "sentence",
        "label_field": "label",
        "label_map": {0: "negative", 1: "positive"},
        "purpose": "Sentiment classification - core capability",
    },
    "conll": {
        "hf_name": "conll2003",
        "hf_subset": None,
        "split": "train",
        "n_samples": 3000,
        "task_type": "ner",
        "text_field": "tokens",
        "label_field": "ner_tags",
        "label_map": {
            0: "O",
            1: "B-PER", 2: "I-PER",
            3: "B-ORG", 4: "I-ORG",
            5: "B-LOC", 6: "I-LOC",
            7: "B-MISC", 8: "I-MISC",
        },
        "purpose": "NER structural grounding - preserves token understanding",
    },
    "mnli": {
        "hf_name": "glue",
        "hf_subset": "mnli",
        "split": "train",
        "n_samples": 2000,
        "task_type": "nli",
        "text_field": ["premise", "hypothesis"],
        "label_field": "label",
        "label_map": {0: "entailment", 1: "neutral", 2: "contradiction"},
        "purpose": "NLI logic/reasoning - preserves inference capability",
    },
    "squad": {
        "hf_name": "squad",
        "hf_subset": None,
        "split": "train",
        "n_samples": 2000,
        "task_type": "qa",
        "context_field": "context",
        "question_field": "question",
        "answer_field": "answers",
        "purpose": "QA context understanding - heals long-range attention",
    },
    "stsb": {
        "hf_name": "glue",
        "hf_subset": "stsb",
        "split": "train",
        "n_samples": 2000,
        "task_type": "similarity",
        "text_field": ["sentence1", "sentence2"],
        "label_field": "label",  # 0-5 float
        "purpose": "Semantic similarity - prevents embedding collapse",
    },
}

TOTAL_ENHANCED_SAMPLES = sum(cfg["n_samples"] for cfg in ENHANCED_HEALING_CONFIG.values())


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare enhanced healing data for v3 training"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/healing/healing_enhanced.jsonl",
        help="Output file or directory",
    )
    parser.add_argument(
        "--split-by-task",
        action="store_true",
        help="Create separate files per task",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate output after creation",
    )
    parser.add_argument(
        "--include-basic",
        action="store_true",
        default=True,
        help="Include basic tasks (SST-2, CoNLL, MNLI)",
    )
    return parser.parse_args()


def load_and_sample_dataset(
    config: Dict,
    n_samples: int,
    seed: int,
) -> List[Dict]:
    """Load dataset from HuggingFace and sample."""
    logger.info(f"Loading {config['hf_name']}...")

    if config["hf_subset"]:
        dataset = load_dataset(
            config["hf_name"],
            config["hf_subset"],
            split=config["split"],
        )
    else:
        dataset = load_dataset(
            config["hf_name"],
            split=config["split"],
        )

    if len(dataset) > n_samples:
        dataset = dataset.shuffle(seed=seed).select(range(n_samples))

    return list(dataset)


# Converters for basic tasks (same as prepare_healing_data.py)
def convert_sst2_sample(sample: Dict, config: Dict) -> Dict:
    """Convert SST-2 sample to unified format."""
    return {
        "text": sample[config["text_field"]],
        "task": "sentiment",
        "task_type": "classification",
        "labels": {
            "sentiment": sample[config["label_field"]],
            "sentiment_label": config["label_map"][sample[config["label_field"]]],
        },
        "source": "sst2",
        "split": "healing",
        "healing_purpose": config["purpose"],
    }


def convert_conll_sample(sample: Dict, config: Dict) -> Dict:
    """Convert CoNLL-2003 sample to unified format."""
    tokens = sample[config["text_field"]]
    ner_tags = sample[config["label_field"]]
    text = " ".join(tokens)

    # Extract spans
    spans = []
    current_entity = None
    current_start = None

    for i, (token, tag) in enumerate(zip(tokens, ner_tags)):
        tag_str = config["label_map"][tag]

        if tag_str.startswith("B-"):
            if current_entity:
                spans.append({
                    "start": current_start,
                    "end": i,
                    "type": current_entity,
                    "text": " ".join(tokens[current_start:i]),
                })
            current_entity = tag_str[2:]
            current_start = i
        elif tag_str == "O":
            if current_entity:
                spans.append({
                    "start": current_start,
                    "end": i,
                    "type": current_entity,
                    "text": " ".join(tokens[current_start:i]),
                })
                current_entity = None

    if current_entity:
        spans.append({
            "start": current_start,
            "end": len(tokens),
            "type": current_entity,
            "text": " ".join(tokens[current_start:]),
        })

    return {
        "text": text,
        "tokens": tokens,
        "task": "ner",
        "task_type": "token_classification",
        "labels": {
            "ner_tags": ner_tags,
            "spans": spans,
        },
        "source": "conll2003",
        "split": "healing",
        "healing_purpose": config["purpose"],
    }


def convert_mnli_sample(sample: Dict, config: Dict) -> Dict:
    """Convert MNLI sample to unified format."""
    return {
        "text": f"{sample['premise']} [SEP] {sample['hypothesis']}",
        "premise": sample["premise"],
        "hypothesis": sample["hypothesis"],
        "task": "nli",
        "task_type": "classification",
        "labels": {
            "nli": sample[config["label_field"]],
            "nli_label": config["label_map"][sample[config["label_field"]]],
        },
        "source": "mnli",
        "split": "healing",
        "healing_purpose": config["purpose"],
    }


# NEW: Enhanced task converters
def convert_squad_sample(sample: Dict, config: Dict) -> Dict:
    """
    Convert SQuAD sample to unified format.

    SQuAD heals long-range attention by requiring the model to:
    1. Attend to question tokens
    2. Find relevant context spans
    3. Extract precise answer boundaries
    """
    context = sample[config["context_field"]]
    question = sample[config["question_field"]]
    answers = sample[config["answer_field"]]

    # Get first answer (SQuAD can have multiple)
    if answers["text"]:
        answer_text = answers["text"][0]
        answer_start = answers["answer_start"][0]
    else:
        answer_text = ""
        answer_start = -1

    return {
        "text": f"{question} [SEP] {context}",
        "question": question,
        "context": context,
        "task": "qa",
        "task_type": "span_extraction",
        "labels": {
            "answer_text": answer_text,
            "answer_start": answer_start,
            "answer_end": answer_start + len(answer_text) if answer_start >= 0 else -1,
        },
        "source": "squad",
        "split": "healing",
        "healing_purpose": config["purpose"],
    }


def convert_stsb_sample(sample: Dict, config: Dict) -> Dict:
    """
    Convert STS-B sample to unified format.

    STS-B prevents embedding collapse by requiring:
    1. Meaningful sentence representations
    2. Continuous similarity scores (0-5)
    3. Symmetric sentence understanding
    """
    sentence1 = sample["sentence1"]
    sentence2 = sample["sentence2"]
    score = sample[config["label_field"]]  # Float 0-5

    # Normalize to 0-1 for easier loss computation
    normalized_score = score / 5.0

    return {
        "text": f"{sentence1} [SEP] {sentence2}",
        "sentence1": sentence1,
        "sentence2": sentence2,
        "task": "similarity",
        "task_type": "regression",
        "labels": {
            "similarity_score": score,
            "normalized_score": normalized_score,
        },
        "source": "stsb",
        "split": "healing",
        "healing_purpose": config["purpose"],
    }


# Converter registry
CONVERTERS = {
    "sst2": convert_sst2_sample,
    "conll": convert_conll_sample,
    "mnli": convert_mnli_sample,
    "squad": convert_squad_sample,
    "stsb": convert_stsb_sample,
}


def prepare_enhanced_healing_data(seed: int = 42) -> Dict[str, List[Dict]]:
    """Prepare all enhanced healing datasets."""
    random.seed(seed)

    healing_data = {}

    for task_name, config in ENHANCED_HEALING_CONFIG.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {task_name}...")
        logger.info(f"Purpose: {config['purpose']}")
        logger.info(f"{'='*50}")

        raw_samples = load_and_sample_dataset(
            config,
            config["n_samples"],
            seed,
        )

        converter = CONVERTERS[task_name]
        converted = []

        for sample in tqdm(raw_samples, desc=f"Converting {task_name}"):
            try:
                converted.append(converter(sample, config))
            except Exception as e:
                logger.warning(f"Failed to convert sample: {e}")
                continue

        healing_data[task_name] = converted
        logger.info(f"  Converted {len(converted)} samples")

    return healing_data


def save_enhanced_healing_data(
    healing_data: Dict[str, List[Dict]],
    output_path: str,
    split_by_task: bool = False,
) -> None:
    """Save enhanced healing data to disk."""
    output = Path(output_path)

    if split_by_task:
        output.mkdir(parents=True, exist_ok=True)
        for task_name, samples in healing_data.items():
            task_file = output / f"healing_enhanced_{task_name}.jsonl"
            with open(task_file, "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample) + "\n")
            logger.info(f"Saved {len(samples)} samples to {task_file}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)

        all_samples = []
        for samples in healing_data.values():
            all_samples.extend(samples)

        random.shuffle(all_samples)

        with open(output, "w") as f:
            for sample in all_samples:
                f.write(json.dumps(sample) + "\n")

        logger.info(f"Saved {len(all_samples)} samples to {output}")


def validate_enhanced_healing_data(output_path: str) -> bool:
    """Validate the created enhanced healing data."""
    output = Path(output_path)

    if output.is_dir():
        files = list(output.glob("healing_enhanced_*.jsonl"))
    else:
        files = [output]

    total_samples = 0
    task_counts = defaultdict(int)
    task_types = defaultdict(set)

    for file_path in files:
        with open(file_path) as f:
            for line in f:
                sample = json.loads(line)
                total_samples += 1
                task_counts[sample["task"]] += 1
                task_types[sample["task"]].add(sample["task_type"])

    logger.info("\n" + "=" * 50)
    logger.info("Enhanced Healing Data Validation")
    logger.info("=" * 50)
    logger.info(f"Total samples: {total_samples}")
    for task, count in sorted(task_counts.items()):
        types = ", ".join(task_types[task])
        logger.info(f"  {task}: {count} ({types})")

    # Verify all 5 tasks present
    expected_tasks = {"sentiment", "ner", "nli", "qa", "similarity"}
    actual_tasks = set(task_counts.keys())

    if expected_tasks != actual_tasks:
        missing = expected_tasks - actual_tasks
        logger.error(f"Missing tasks: {missing}")
        return False

    if total_samples < TOTAL_ENHANCED_SAMPLES * 0.95:
        logger.warning(f"Lower than expected: {total_samples} < {TOTAL_ENHANCED_SAMPLES}")
        return False

    logger.info("✅ Enhanced validation passed!")
    return True


def main() -> int:
    """Main entry point."""
    args = parse_args()

    print("\n" + "=" * 60)
    print("ModernBERT v3 ENHANCED Healing Data Preparation")
    print("=" * 60)
    print(f"Output: {args.output}")
    print(f"Tasks: SST-2, CoNLL, MNLI, SQuAD, STS-B")
    print(f"Total samples: ~{TOTAL_ENHANCED_SAMPLES}")
    print()

    healing_data = prepare_enhanced_healing_data(seed=args.seed)

    save_enhanced_healing_data(
        healing_data,
        args.output,
        split_by_task=args.split_by_task,
    )

    if args.validate:
        if not validate_enhanced_healing_data(args.output):
            return 1

    print("\n" + "=" * 60)
    print("✅ Enhanced healing data preparation complete!")
    print("=" * 60)

    total = sum(len(samples) for samples in healing_data.values())
    print(f"\nTotal samples: {total}")
    for task, samples in healing_data.items():
        purpose = ENHANCED_HEALING_CONFIG[task]["purpose"]
        print(f"  {task}: {len(samples)} - {purpose}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Acceptance Criteria:**

- [ ] Includes all 5 tasks: SST-2, CoNLL, MNLI, SQuAD, STS-B
- [ ] SQuAD samples have question/context/answer structure
- [ ] STS-B samples have similarity scores (0-5 normalized)
- [ ] Total ~12k samples (3k+3k+2k+2k+2k)
- [ ] Each sample includes `healing_purpose` field
- [ ] Validation checks all 5 task types present

**Tests:** `tests/v3/test_data_prep.py::test_enhanced_healing_script`

---

#### Issue 5.2.5: Create Basic Healing Dataset Configuration

**File:** `configs/data/multitask/healing_datasets.yaml`
**Effort:** 2 hours
**Dependencies:** Issue 5.2.3

**Description:**
Create YAML configuration for basic healing datasets (SST-2, CoNLL, MNLI) that can be loaded by the v3 training pipeline.

**Implementation:**

```yaml
# configs/data/multitask/healing_datasets.yaml
#
# Basic healing dataset configuration for ModernBERT v3 Phase 0.5
# Uses 3 standard benchmarks: SST-2, CoNLL-2003, MNLI
# Total: 10,000 samples

# Dataset metadata
dataset:
  name: "healing_basic"
  version: "1.0"
  description: "Basic healing data for v3 Phase 0.5 training"
  total_samples: 10000
  purpose: "Preserve v2 capabilities during layer healing"

# Data paths
paths:
  unified_file: "data/healing/healing_generic.jsonl"
  split_dir: "data/healing/"
  cache_dir: "data/cache/healing/"

# Task configurations
tasks:
  sentiment:
    source: "sst2"
    n_samples: 3000
    task_type: "classification"
    num_labels: 2
    label_names: ["negative", "positive"]
    weight: 1.0
    metrics:
      - accuracy
      - f1

  ner:
    source: "conll2003"
    n_samples: 3000
    task_type: "token_classification"
    num_labels: 9
    label_names:
      - "O"
      - "B-PER"
      - "I-PER"
      - "B-ORG"
      - "I-ORG"
      - "B-LOC"
      - "I-LOC"
      - "B-MISC"
      - "I-MISC"
    weight: 1.0
    metrics:
      - f1
      - precision
      - recall

  nli:
    source: "mnli"
    n_samples: 4000
    task_type: "classification"
    num_labels: 3
    label_names: ["entailment", "neutral", "contradiction"]
    weight: 1.0
    metrics:
      - accuracy

# Preprocessing
preprocessing:
  max_length: 512
  padding: "max_length"
  truncation: true
  add_hub_tokens: true
  hub_token_offset: 5  # [CLS] + 4 hub tokens

# Data loading
loading:
  batch_size: 32
  shuffle: true
  num_workers: 4
  pin_memory: true
  prefetch_factor: 2

# Sampling strategy
sampling:
  strategy: "proportional"  # or "balanced", "task_weighted"
  task_weights:
    sentiment: 1.0
    ner: 1.0
    nli: 1.0
  oversample_minority: false
  undersample_majority: false

# Validation split
validation:
  enabled: true
  split_ratio: 0.1
  stratify_by_task: true

# Augmentation (optional for healing)
augmentation:
  enabled: false
  techniques: []

# Quality filters
filters:
  min_length: 5
  max_length: 500
  remove_duplicates: true
  remove_empty: true
```

**Acceptance Criteria:**

- [ ] Defines all 3 basic tasks with correct settings
- [ ] Includes label mappings for each task
- [ ] Specifies hub token offset (5)
- [ ] Configures data loading parameters
- [ ] Supports proportional and balanced sampling
- [ ] Can be loaded by Hydra/OmegaConf

**Tests:** `tests/v3/test_configs.py::test_healing_config_basic`

---

#### Issue 5.2.6: Create Enhanced Healing Dataset Configuration

**File:** `configs/data/multitask/healing_enhanced.yaml`
**Effort:** 2 hours
**Dependencies:** Issue 5.2.4, Issue 5.2.5

**Description:**
Create YAML configuration for enhanced healing datasets including SQuAD and STS-B for better attention healing and embedding preservation.

**Implementation:**

```yaml
# configs/data/multitask/healing_enhanced.yaml
#
# Enhanced healing dataset configuration for ModernBERT v3 Phase 0.5
# Uses 5 tasks: SST-2, CoNLL, MNLI, SQuAD, STS-B
# Total: 12,000 samples
#
# Enhancement rationale:
#   - SQuAD: Heals attention patterns for long-range context understanding
#   - STS-B: Prevents embedding collapse, maintains semantic similarity capability

# Dataset metadata
dataset:
  name: "healing_enhanced"
  version: "1.0"
  description: "Enhanced healing data for v3 Phase 0.5 with attention/embedding repair"
  total_samples: 12000
  purpose: "Comprehensive capability preservation during layer healing"

# Data paths
paths:
  unified_file: "data/healing/healing_enhanced.jsonl"
  split_dir: "data/healing/"
  cache_dir: "data/cache/healing_enhanced/"

# Task configurations
tasks:
  # === BASIC TASKS (from healing_datasets.yaml) ===

  sentiment:
    source: "sst2"
    n_samples: 3000
    task_type: "classification"
    num_labels: 2
    label_names: ["negative", "positive"]
    weight: 1.0
    healing_target: "classification_head"
    metrics:
      - accuracy
      - f1

  ner:
    source: "conll2003"
    n_samples: 3000
    task_type: "token_classification"
    num_labels: 9
    label_names:
      - "O"
      - "B-PER"
      - "I-PER"
      - "B-ORG"
      - "I-ORG"
      - "B-LOC"
      - "I-LOC"
      - "B-MISC"
      - "I-MISC"
    weight: 1.0
    healing_target: "token_representations"
    metrics:
      - f1
      - precision
      - recall

  nli:
    source: "mnli"
    n_samples: 2000
    task_type: "classification"
    num_labels: 3
    label_names: ["entailment", "neutral", "contradiction"]
    weight: 1.0
    healing_target: "reasoning_capability"
    metrics:
      - accuracy

  # === ENHANCED TASKS ===

  qa:
    source: "squad"
    n_samples: 2000
    task_type: "span_extraction"
    healing_target: "attention_patterns"
    healing_purpose: "Heal long-range attention by requiring context-question alignment"
    fields:
      question: "question"
      context: "context"
      answer_text: "labels.answer_text"
      answer_start: "labels.answer_start"
    weight: 1.2  # Slightly higher weight for attention healing
    metrics:
      - exact_match
      - f1

  similarity:
    source: "stsb"
    n_samples: 2000
    task_type: "regression"
    healing_target: "embedding_space"
    healing_purpose: "Prevent embedding collapse, maintain semantic similarity"
    score_range: [0.0, 5.0]
    normalized_range: [0.0, 1.0]
    weight: 1.2  # Slightly higher weight for embedding preservation
    metrics:
      - pearson
      - spearman
      - mse

# Preprocessing
preprocessing:
  max_length: 512
  padding: "max_length"
  truncation: true
  add_hub_tokens: true
  hub_token_offset: 5

  # Task-specific preprocessing
  task_specific:
    qa:
      max_context_length: 384
      max_question_length: 64
      doc_stride: 128  # For sliding window
    similarity:
      normalize_scores: true

# Data loading
loading:
  batch_size: 32
  shuffle: true
  num_workers: 4
  pin_memory: true
  prefetch_factor: 2

# Enhanced sampling strategy
sampling:
  strategy: "healing_weighted"
  task_weights:
    sentiment: 1.0
    ner: 1.0
    nli: 1.0
    qa: 1.2       # Boost for attention healing
    similarity: 1.2  # Boost for embedding preservation
  healing_priority:
    - qa         # Process attention-healing samples first in epoch
    - similarity # Then embedding preservation
    - ner        # Then structural tasks
    - sentiment
    - nli

# Validation
validation:
  enabled: true
  split_ratio: 0.1
  stratify_by_task: true

# Healing effectiveness tracking
healing_metrics:
  track_attention_entropy: true
  track_embedding_similarity: true
  track_layer_activation_stats: true
  checkpoints:
    - 500
    - 1000
    - 1500
    - 2000
    - 2500

# Loss weighting for healing
loss_config:
  use_task_weights: true
  use_uncertainty_weighting: false
  normalize_losses: true
  healing_loss_scale: 1.0
```

**Acceptance Criteria:**

- [ ] Defines all 5 enhanced tasks
- [ ] SQuAD configured for span extraction
- [ ] STS-B configured for regression with normalization
- [ ] Higher weights for attention/embedding healing tasks
- [ ] Healing priority order defined
- [ ] Healing effectiveness metrics specified
- [ ] Compatible with healing_datasets.yaml structure

**Tests:** `tests/v3/test_configs.py::test_healing_config_enhanced`

---

#### Issue 5.2.7: Create Enhanced Phase 0.5 Training Configuration

**File:** `configs/training/multitask/stage_v3_phase0_5_enhanced.yaml`
**Effort:** 3 hours
**Dependencies:** Issues 5.1.1-5.1.8, Issue 5.2.6

**Description:**
Create comprehensive training configuration for Phase 0.5 that integrates enhanced healing data with the Zipper LR strategy, gradient clipping, and phase-aware freezing.

**Implementation:**

```yaml
# configs/training/multitask/stage_v3_phase0_5_enhanced.yaml
#
# Phase 0.5 Training Configuration for ModernBERT v3
# "Enhanced Healing" - Repairs cloned layers while preserving v2 capabilities
#
# Training Focus:
#   - Heal L23-28 (cloned from L15-20)
#   - Smooth L22→L23 interface
#   - Preserve L1-22 capabilities
#   - Use enhanced healing data (SST-2, CoNLL, MNLI, SQuAD, STS-B)

# ═══════════════════════════════════════════════════════════════
# Model Configuration
# ═══════════════════════════════════════════════════════════════
model:
  name: "modernbert_v3_ultra"
  pretrained_path: null  # Will be set by initialization script
  config:
    num_layers: 28
    hidden_size: 768
    num_attention_heads: 12
    num_hub_tokens: 4
    vocab_size: 50372

# ═══════════════════════════════════════════════════════════════
# Training Configuration
# ═══════════════════════════════════════════════════════════════
training:
  # Phase identification
  phase: "phase_0.5"
  phase_name: "Enhanced Healing"

  # Training duration
  max_steps: 2500
  warmup_steps: 500
  warmup_ratio: 0.2

  # Batch settings
  per_device_train_batch_size: 32
  per_device_eval_batch_size: 64
  gradient_accumulation_steps: 1
  effective_batch_size: 32  # per_device * gradient_accumulation

  # Evaluation
  eval_steps: 250
  eval_strategy: "steps"
  save_steps: 500
  save_total_limit: 3

  # Logging
  logging_steps: 50
  logging_first_step: true

  # Mixed precision
  fp16: false
  bf16: true

  # Reproducibility
  seed: 42
  data_seed: 42

# ═══════════════════════════════════════════════════════════════
# Layer Freezing Configuration (Phase 0.5)
# ═══════════════════════════════════════════════════════════════
layer_freezing:
  phase: "phase_0.5"

  # Frozen bands (L1-18)
  frozen_bands:
    - foundation  # L1-6
    - core        # L7-18

  # Trainable bands (L19-28)
  trainable_bands:
    - feeder      # L19-22: Interface preparation
    - family      # L23-28: New cloned layers

  # Component freezing
  freeze_embeddings: true
  freeze_hub_tokens: false  # Hub tokens are trainable

  # Expected freeze stats
  expected_frozen_params: "~100M"
  expected_trainable_params: "~50M"

# ═══════════════════════════════════════════════════════════════
# Zipper Learning Rate Strategy
# ═══════════════════════════════════════════════════════════════
learning_rate:
  strategy: "zipper"

  # Layer-specific rates
  base_lr: 3e-5
  layers_1_18: 0.0        # Frozen
  layers_19_22: 1e-5      # Feeder: gentle adaptation
  layer_23: 5e-5          # Interface: maximum plasticity
  layers_24_28: 3e-5      # Family: moderate adaptation

  # Decay in family band
  family_graduated: true
  family_decay: 0.85

  # Component rates
  embeddings_lr: 0.0      # Frozen
  hub_tokens_lr: 1e-5     # Careful with hub tokens
  task_heads_lr: 3e-5     # Same as family band

# ═══════════════════════════════════════════════════════════════
# Scheduler Configuration
# ═══════════════════════════════════════════════════════════════
scheduler:
  type: "cosine"
  warmup_steps: 500
  min_lr_ratio: 0.01

  # LR Profile:
  # Step 0:    lr = 0
  # Step 500:  lr = peak (warmup complete)
  # Step 2500: lr = 1% of peak

# ═══════════════════════════════════════════════════════════════
# Gradient Configuration
# ═══════════════════════════════════════════════════════════════
gradient:
  # Global clipping
  max_grad_norm: 1.0

  # Per-layer clipping (more fine-grained)
  per_layer_clip: true
  interface_clip: 0.5     # L23: tighter clip at interface
  feeder_clip: 1.0        # L19-22
  family_clip: 1.0        # L24-28

  # Gradient monitoring
  log_grad_norms: true
  log_every_n_steps: 100
  explosion_threshold: 10.0

  # NaN handling
  nan_check: true
  zero_nan_grads: true

# ═══════════════════════════════════════════════════════════════
# Hub Token Gradient Masking
# ═══════════════════════════════════════════════════════════════
hub_tokens:
  gradient_masking: true
  freeze_original_vocab: true
  train_hub_tokens:
    - "[EMO]"
    - "[MEM]"
    - "[REL]"
    - "[TASK]"
  hub_token_grad_scale: 1.0

# ═══════════════════════════════════════════════════════════════
# Data Configuration
# ═══════════════════════════════════════════════════════════════
data:
  # Use enhanced healing data
  config_file: "configs/data/multitask/healing_enhanced.yaml"

  # Replay sampling (for forgetting prevention)
  replay:
    enabled: true
    ratio: 0.15  # 15% replay samples
    task_balanced: true
    dynamic_ratio: true
    loss_threshold: 0.5
    max_ratio: 0.3

  # Data loading
  num_workers: 4
  pin_memory: true
  prefetch_factor: 2

# ═══════════════════════════════════════════════════════════════
# Optimizer Configuration
# ═══════════════════════════════════════════════════════════════
optimizer:
  type: "adamw"
  weight_decay: 0.01
  betas: [0.9, 0.999]
  eps: 1e-8

# ═══════════════════════════════════════════════════════════════
# Loss Configuration
# ═══════════════════════════════════════════════════════════════
loss:
  # Multi-task loss weighting
  use_task_weights: true
  normalize_losses: true

  task_weights:
    sentiment: 1.0
    ner: 1.0
    nli: 1.0
    qa: 1.2           # Higher for attention healing
    similarity: 1.2   # Higher for embedding preservation

  # Uncertainty weighting (optional)
  use_uncertainty_weighting: false

# ═══════════════════════════════════════════════════════════════
# LoRA Configuration (Optional for Phase 0.5)
# ═══════════════════════════════════════════════════════════════
lora:
  enabled: false  # LoRA typically used in Phase 1
  rank: 16
  alpha: 32.0
  dropout: 0.1
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
  target_layers: [22, 23, 24, 25, 26, 27]  # L23-28

# ═══════════════════════════════════════════════════════════════
# Checkpointing
# ═══════════════════════════════════════════════════════════════
checkpointing:
  output_dir: "outputs/v3_phase0_5_enhanced"
  save_strategy: "steps"
  save_steps: 500
  save_total_limit: 3
  load_best_model_at_end: true

  # What to save
  save_optimizer_state: true
  save_scheduler_state: true
  save_rng_state: true

# ═══════════════════════════════════════════════════════════════
# Logging & Monitoring
# ═══════════════════════════════════════════════════════════════
logging:
  # Weights & Biases
  use_wandb: true
  wandb_project: "modernbert-v3"
  wandb_run_name: "phase0_5_enhanced_healing"
  wandb_tags:
    - "phase_0.5"
    - "healing"
    - "enhanced"
    - "v3"

  # Console logging
  logging_steps: 50
  log_level: "info"

  # Metrics to track
  track_metrics:
    - train_loss
    - eval_loss
    - learning_rate
    - grad_norm
    - interface_grad_ratio  # L23/L22 gradient ratio

# ═══════════════════════════════════════════════════════════════
# Early Stopping
# ═══════════════════════════════════════════════════════════════
early_stopping:
  enabled: false  # Usually complete all 2500 steps
  patience: 5
  threshold: 0.001
  metric: "eval_loss"
  greater_is_better: false

# ═══════════════════════════════════════════════════════════════
# Healing Verification Gates
# ═══════════════════════════════════════════════════════════════
verification:
  # Run at these checkpoints
  checkpoints: [500, 1000, 1500, 2000, 2500]

  # Metrics to verify
  metrics:
    - name: "interface_activation_similarity"
      description: "Activation similarity between L22 and L23"
      threshold: 0.8
      direction: "higher_is_better"

    - name: "embedding_space_stability"
      description: "Cosine similarity of embeddings before/after"
      threshold: 0.95
      direction: "higher_is_better"

    - name: "attention_entropy"
      description: "Entropy of attention patterns in L23-28"
      threshold: 2.0
      direction: "bounded"  # Not too high, not too low

  # Actions on failure
  on_failure:
    - "log_warning"
    - "increase_replay_ratio"
```

**Acceptance Criteria:**

- [ ] Complete Phase 0.5 configuration
- [ ] Zipper LR strategy with all layer-specific rates
- [ ] Gradient clipping with interface-specific settings
- [ ] Hub token gradient masking configured
- [ ] Enhanced healing data config referenced
- [ ] Replay sampling enabled
- [ ] Verification gates at checkpoints
- [ ] Wandb logging configured
- [ ] Compatible with `ModernBERTv3Trainer`

**Tests:** `tests/v3/test_configs.py::test_phase_0_5_training_config`

---

### Epic 5.3: Unified FamilyOS Data Loading (for generated data)

#### Issue 5.3.1: Implement Unified FamilyOS Dataset Loader

**Effort:** 5 hours | **Priority:** P0 | **Depends On:** Epic 5.2

**File:** `src/modeling_studio/data/loaders_v3.py`
**Data:** `data/familyos/unified/output_healed/shard_*.jsonl`, `data/familyos/unified/output_synthetic_healed/shard_*.jsonl`

**Purpose:** Load the generated unified JSONL with hub_routing field for v3 multi-task training.

**Data Format Reference:**

```json
{
  "id": "fam_00005",
  "text": "Thanksgiving self-reflection: Gratitude list, but grief overshadows.",
  "tasks": {
    "emotions": ["gratitude", "grief", "sadness", "bittersweet"],
    "sentiment": "mixed",
    "ner_family": [{"start": 0, "end": 12, "label": "TRADITION", "token": "Thanksgiving"}],
    "safety_familyos": "AMBER",
    "intent": "reflect",
    "ingress": "DIARY",
    "relations": [{"subject": "Mike", "predicate": "parent_of", "object": "kids"}],
    "temporal": [{"start": 8, "end": 23, "label": "DATE_ABS", "token": "August 5th 2024"}]
  },
  "hub_routing": {"EMO": true, "REL": false, "MEM": true, "TASK": false}
}
```

**Implementation:**

```python
# src/modeling_studio/data/loaders_v3.py
"""
Unified FamilyOS Dataset Loader for v3 Multi-Task Training

Loads unified JSONL files with hub_routing and 8 task types:
- emotions (multi-label list)
- sentiment (single label)
- ner_family (span list)
- safety_familyos (single label)
- intent (single label)
- ingress (single label)
- relations (triple list)
- temporal (span list)
"""
from __future__ import annotations

import json
import glob
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterator, Any
from enum import Enum
import logging

import torch
from torch.utils.data import Dataset, IterableDataset

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Supported task types in unified FamilyOS data."""
    EMOTIONS = "emotions"           # Multi-label classification
    SENTIMENT = "sentiment"         # Single-label classification
    NER_FAMILY = "ner_family"       # Token classification (spans)
    SAFETY_FAMILYOS = "safety_familyos"  # Single-label classification
    INTENT = "intent"               # Single-label classification
    INGRESS = "ingress"             # Single-label classification
    RELATIONS = "relations"         # Relation extraction (triples)
    TEMPORAL = "temporal"           # Token classification (spans)


class HubType(Enum):
    """Hub token routing types."""
    EMO = "EMO"     # Emotion hub - emotions, sentiment, safety
    REL = "REL"     # Relation hub - relations
    MEM = "MEM"     # Memory hub - temporal, ner_family, diary entries
    TASK = "TASK"   # Task hub - intent, ingress, actionable items


@dataclass
class HubRouting:
    """Hub routing configuration parsed from sample."""
    emo: bool = False
    rel: bool = False
    mem: bool = False
    task: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, bool]) -> "HubRouting":
        """Parse hub_routing dict from JSON."""
        return cls(
            emo=data.get("EMO", False),
            rel=data.get("REL", False),
            mem=data.get("MEM", False),
            task=data.get("TASK", False)
        )

    def to_tensor(self) -> torch.Tensor:
        """Convert to float tensor [EMO, REL, MEM, TASK]."""
        return torch.tensor(
            [float(self.emo), float(self.rel), float(self.mem), float(self.task)],
            dtype=torch.float32
        )

    @property
    def active_hubs(self) -> List[str]:
        """Return list of active hub names."""
        result = []
        if self.emo:
            result.append("EMO")
        if self.rel:
            result.append("REL")
        if self.mem:
            result.append("MEM")
        if self.task:
            result.append("TASK")
        return result


@dataclass
class SpanAnnotation:
    """Span annotation for NER and temporal tasks."""
    start: int
    end: int
    label: str
    token: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpanAnnotation":
        return cls(
            start=data["start"],
            end=data["end"],
            label=data["label"],
            token=data["token"]
        )


@dataclass
class RelationTriple:
    """Relation triple annotation."""
    subject: str
    predicate: str
    object: str

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "RelationTriple":
        return cls(
            subject=data["subject"],
            predicate=data["predicate"],
            object=data["object"]
        )


@dataclass
class UnifiedSample:
    """
    Parsed sample from unified FamilyOS JSONL.

    Contains all 8 task labels and hub routing information.
    """
    id: str
    text: str

    # Classification tasks (single or multi-label)
    emotions: List[str] = field(default_factory=list)
    sentiment: Optional[str] = None
    safety_familyos: Optional[str] = None
    intent: Optional[str] = None
    ingress: Optional[str] = None

    # Token classification tasks (spans)
    ner_family: List[SpanAnnotation] = field(default_factory=list)
    temporal: List[SpanAnnotation] = field(default_factory=list)

    # Relation extraction
    relations: List[RelationTriple] = field(default_factory=list)

    # Hub routing
    hub_routing: HubRouting = field(default_factory=HubRouting)

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "UnifiedSample":
        """Parse sample from JSON dict."""
        tasks = data.get("tasks", {})

        return cls(
            id=data["id"],
            text=data["text"],
            # Classification
            emotions=tasks.get("emotions", []),
            sentiment=tasks.get("sentiment"),
            safety_familyos=tasks.get("safety_familyos"),
            intent=tasks.get("intent"),
            ingress=tasks.get("ingress"),
            # Spans
            ner_family=[
                SpanAnnotation.from_dict(s)
                for s in tasks.get("ner_family", [])
            ],
            temporal=[
                SpanAnnotation.from_dict(s)
                for s in tasks.get("temporal", [])
            ],
            # Relations
            relations=[
                RelationTriple.from_dict(r)
                for r in tasks.get("relations", [])
            ],
            # Hub routing
            hub_routing=HubRouting.from_dict(data.get("hub_routing", {}))
        )

    def has_task(self, task_type: TaskType) -> bool:
        """Check if sample has non-empty data for given task."""
        if task_type == TaskType.EMOTIONS:
            return len(self.emotions) > 0
        elif task_type == TaskType.SENTIMENT:
            return self.sentiment is not None
        elif task_type == TaskType.NER_FAMILY:
            return len(self.ner_family) > 0
        elif task_type == TaskType.SAFETY_FAMILYOS:
            return self.safety_familyos is not None
        elif task_type == TaskType.INTENT:
            return self.intent is not None
        elif task_type == TaskType.INGRESS:
            return self.ingress is not None
        elif task_type == TaskType.RELATIONS:
            return len(self.relations) > 0
        elif task_type == TaskType.TEMPORAL:
            return len(self.temporal) > 0
        return False


class UnifiedFamilyOSDataset(Dataset):
    """
    PyTorch Dataset for unified FamilyOS data.

    Loads all samples into memory for random access.
    Use IterableUnifiedFamilyOSDataset for streaming.
    """

    def __init__(
        self,
        data_dir: str,
        shard_pattern: str = "shard_*.jsonl",
        max_samples: Optional[int] = None,
        filter_tasks: Optional[List[TaskType]] = None,
        require_hub_routing: bool = False
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Directory containing shard files
            shard_pattern: Glob pattern for shard files
            max_samples: Maximum samples to load (None = all)
            filter_tasks: Only include samples with these tasks
            require_hub_routing: Only include samples with at least one active hub
        """
        self.data_dir = Path(data_dir)
        self.shard_pattern = shard_pattern
        self.max_samples = max_samples
        self.filter_tasks = filter_tasks
        self.require_hub_routing = require_hub_routing

        # Load samples
        self.samples: List[UnifiedSample] = []
        self._load_samples()

        logger.info(f"Loaded {len(self.samples)} samples from {data_dir}")

    def _load_samples(self) -> None:
        """Load all samples from shard files."""
        shard_files = sorted(glob.glob(str(self.data_dir / self.shard_pattern)))

        if not shard_files:
            raise FileNotFoundError(
                f"No shard files found matching {self.data_dir / self.shard_pattern}"
            )

        logger.info(f"Found {len(shard_files)} shard files")

        for shard_path in shard_files:
            with open(shard_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue

                    data = json.loads(line)
                    sample = UnifiedSample.from_json(data)

                    # Apply filters
                    if self._should_include(sample):
                        self.samples.append(sample)

                    # Check max samples limit
                    if self.max_samples and len(self.samples) >= self.max_samples:
                        return

    def _should_include(self, sample: UnifiedSample) -> bool:
        """Check if sample should be included based on filters."""
        # Filter by required tasks
        if self.filter_tasks:
            has_required_task = any(
                sample.has_task(task) for task in self.filter_tasks
            )
            if not has_required_task:
                return False

        # Filter by hub routing
        if self.require_hub_routing:
            if not sample.hub_routing.active_hubs:
                return False

        return True

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> UnifiedSample:
        return self.samples[idx]

    def get_task_distribution(self) -> Dict[str, int]:
        """Get count of samples per task type."""
        dist = {task.value: 0 for task in TaskType}

        for sample in self.samples:
            for task_type in TaskType:
                if sample.has_task(task_type):
                    dist[task_type.value] += 1

        return dist

    def get_hub_distribution(self) -> Dict[str, int]:
        """Get count of samples per hub routing."""
        dist = {"EMO": 0, "REL": 0, "MEM": 0, "TASK": 0, "none": 0}

        for sample in self.samples:
            routing = sample.hub_routing
            if routing.emo:
                dist["EMO"] += 1
            if routing.rel:
                dist["REL"] += 1
            if routing.mem:
                dist["MEM"] += 1
            if routing.task:
                dist["TASK"] += 1
            if not routing.active_hubs:
                dist["none"] += 1

        return dist


class IterableUnifiedFamilyOSDataset(IterableDataset):
    """
    Streaming/Iterable Dataset for unified FamilyOS data.

    Memory efficient - doesn't load all samples upfront.
    """

    def __init__(
        self,
        data_dir: str,
        shard_pattern: str = "shard_*.jsonl",
        shuffle_shards: bool = True,
        filter_tasks: Optional[List[TaskType]] = None
    ):
        """
        Initialize iterable dataset.

        Args:
            data_dir: Directory containing shard files
            shard_pattern: Glob pattern for shard files
            shuffle_shards: Whether to shuffle shard order
            filter_tasks: Only yield samples with these tasks
        """
        self.data_dir = Path(data_dir)
        self.shard_pattern = shard_pattern
        self.shuffle_shards = shuffle_shards
        self.filter_tasks = filter_tasks

        self.shard_files = sorted(glob.glob(str(self.data_dir / self.shard_pattern)))

        if not self.shard_files:
            raise FileNotFoundError(
                f"No shard files found matching {self.data_dir / self.shard_pattern}"
            )

    def __iter__(self) -> Iterator[UnifiedSample]:
        """Iterate over samples."""
        import random

        shard_files = self.shard_files.copy()
        if self.shuffle_shards:
            random.shuffle(shard_files)

        for shard_path in shard_files:
            with open(shard_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue

                    data = json.loads(line)
                    sample = UnifiedSample.from_json(data)

                    # Apply task filter
                    if self.filter_tasks:
                        has_task = any(
                            sample.has_task(task) for task in self.filter_tasks
                        )
                        if not has_task:
                            continue

                    yield sample
```

**Acceptance Criteria:**

- [ ] `UnifiedSample` dataclass parses all 8 task types correctly
- [ ] `HubRouting` dataclass parses EMO/REL/MEM/TASK booleans
- [ ] `SpanAnnotation` handles start/end/label/token format
- [ ] `RelationTriple` handles subject/predicate/object format
- [ ] `UnifiedFamilyOSDataset` loads from multiple shard files
- [ ] `IterableUnifiedFamilyOSDataset` provides streaming access
- [ ] Filter by task type works correctly
- [ ] Filter by hub routing works correctly
- [ ] Distribution statistics computed correctly

**Tests:** `tests/v3/test_loaders_v3.py::test_unified_familyos_loader`

---

#### Issue 5.3.2: Implement Hub-Routing-Aware Sample Parser

**Effort:** 4 hours | **Priority:** P0 | **Depends On:** Issue 5.3.1

**File:** `src/modeling_studio/data/loaders_v3.py` (extend)

**Purpose:** Parse `hub_routing` field to determine which hub tokens should receive gradient signal and which task heads should be trained for each sample.

**Hub-to-Task Mapping:**

```
EMO hub → emotions, sentiment, safety_familyos
REL hub → relations
MEM hub → temporal, ner_family (memory-related entities)
TASK hub → intent, ingress
```

**Implementation:**

```python
# src/modeling_studio/data/loaders_v3.py (continued)

@dataclass
class HubTaskMapping:
    """
    Maps hub routing to task activation.

    Defines which tasks are trained when each hub is active.
    """

    # Hub → Task mapping (which tasks train when hub is active)
    HUB_TO_TASKS: Dict[str, List[TaskType]] = field(default_factory=lambda: {
        "EMO": [TaskType.EMOTIONS, TaskType.SENTIMENT, TaskType.SAFETY_FAMILYOS],
        "REL": [TaskType.RELATIONS],
        "MEM": [TaskType.TEMPORAL, TaskType.NER_FAMILY],
        "TASK": [TaskType.INTENT, TaskType.INGRESS]
    })

    # Task → Hub mapping (which hub controls each task)
    TASK_TO_HUB: Dict[TaskType, str] = field(default_factory=lambda: {
        TaskType.EMOTIONS: "EMO",
        TaskType.SENTIMENT: "EMO",
        TaskType.SAFETY_FAMILYOS: "EMO",
        TaskType.RELATIONS: "REL",
        TaskType.TEMPORAL: "MEM",
        TaskType.NER_FAMILY: "MEM",
        TaskType.INTENT: "TASK",
        TaskType.INGRESS: "TASK"
    })


class HubRoutingParser:
    """
    Parses hub routing to determine task activation and gradient masking.

    Provides utilities for:
    1. Determining which tasks should train for a sample
    2. Computing hub token gradient masks
    3. Computing per-task loss weights
    """

    def __init__(
        self,
        hub_to_tasks: Optional[Dict[str, List[TaskType]]] = None,
        always_train_safety: bool = True,
        safety_weight_override: float = 2.0
    ):
        """
        Initialize parser.

        Args:
            hub_to_tasks: Custom hub → task mapping (uses default if None)
            always_train_safety: Always train safety regardless of hub routing
            safety_weight_override: Weight multiplier for safety task
        """
        self.hub_to_tasks = hub_to_tasks or {
            "EMO": [TaskType.EMOTIONS, TaskType.SENTIMENT, TaskType.SAFETY_FAMILYOS],
            "REL": [TaskType.RELATIONS],
            "MEM": [TaskType.TEMPORAL, TaskType.NER_FAMILY],
            "TASK": [TaskType.INTENT, TaskType.INGRESS]
        }

        # Reverse mapping
        self.task_to_hub: Dict[TaskType, str] = {}
        for hub, tasks in self.hub_to_tasks.items():
            for task in tasks:
                self.task_to_hub[task] = hub

        self.always_train_safety = always_train_safety
        self.safety_weight_override = safety_weight_override

    def get_active_tasks(
        self,
        hub_routing: HubRouting,
        sample: UnifiedSample
    ) -> List[TaskType]:
        """
        Get list of tasks that should be trained for this sample.

        Tasks are active if:
        1. The controlling hub is active in hub_routing
        2. The sample has non-empty data for the task

        Args:
            hub_routing: Hub routing for sample
            sample: The unified sample

        Returns:
            List of active task types
        """
        active_tasks = []

        # Check each hub
        hub_active = {
            "EMO": hub_routing.emo,
            "REL": hub_routing.rel,
            "MEM": hub_routing.mem,
            "TASK": hub_routing.task
        }

        for task_type in TaskType:
            # Check if controlling hub is active
            controlling_hub = self.task_to_hub.get(task_type)
            if controlling_hub and hub_active.get(controlling_hub, False):
                # Check if sample has data for this task
                if sample.has_task(task_type):
                    active_tasks.append(task_type)

        # Always include safety if configured (and sample has safety label)
        if self.always_train_safety and sample.has_task(TaskType.SAFETY_FAMILYOS):
            if TaskType.SAFETY_FAMILYOS not in active_tasks:
                active_tasks.append(TaskType.SAFETY_FAMILYOS)

        return active_tasks

    def get_hub_gradient_mask(
        self,
        hub_routing: HubRouting
    ) -> torch.Tensor:
        """
        Get gradient mask for hub tokens.

        Returns mask [EMO, REL, MEM, TASK] where:
        - 1.0 = hub receives gradient
        - 0.0 = hub is frozen for this sample

        Args:
            hub_routing: Hub routing for sample

        Returns:
            Float tensor of shape [4]
        """
        return hub_routing.to_tensor()

    def get_task_weights(
        self,
        hub_routing: HubRouting,
        active_tasks: List[TaskType]
    ) -> Dict[TaskType, float]:
        """
        Get loss weight for each active task.

        Weights are based on:
        1. Hub routing (active hubs get higher weight)
        2. Safety override (safety always gets higher weight)
        3. Number of active tasks (normalize to prevent loss explosion)

        Args:
            hub_routing: Hub routing for sample
            active_tasks: List of active tasks

        Returns:
            Dict mapping task type to weight
        """
        if not active_tasks:
            return {}

        weights = {}
        n_tasks = len(active_tasks)

        # Base weight per task (normalized)
        base_weight = 1.0 / n_tasks

        for task_type in active_tasks:
            weight = base_weight

            # Apply safety override
            if task_type == TaskType.SAFETY_FAMILYOS:
                weight *= self.safety_weight_override

            weights[task_type] = weight

        return weights

    def parse_batch(
        self,
        samples: List[UnifiedSample]
    ) -> Dict[str, Any]:
        """
        Parse a batch of samples for training.

        Returns:
            Dict with:
            - hub_masks: [batch, 4] hub gradient masks
            - task_active: Dict[task_type, List[int]] - sample indices with each task
            - task_weights: Dict[task_type, Tensor] - per-sample weights
        """
        batch_size = len(samples)

        # Hub gradient masks [batch, 4]
        hub_masks = torch.zeros(batch_size, 4)

        # Track active samples per task
        task_active: Dict[TaskType, List[int]] = {t: [] for t in TaskType}

        # Task weights per sample
        task_weights: Dict[TaskType, List[float]] = {t: [] for t in TaskType}

        for i, sample in enumerate(samples):
            # Hub mask
            hub_masks[i] = self.get_hub_gradient_mask(sample.hub_routing)

            # Active tasks
            active = self.get_active_tasks(sample.hub_routing, sample)
            weights = self.get_task_weights(sample.hub_routing, active)

            for task_type in TaskType:
                if task_type in active:
                    task_active[task_type].append(i)
                    task_weights[task_type].append(weights[task_type])
                else:
                    task_weights[task_type].append(0.0)

        # Convert weights to tensors
        task_weight_tensors = {
            t: torch.tensor(w) for t, w in task_weights.items()
        }

        return {
            "hub_masks": hub_masks,
            "task_active": task_active,
            "task_weights": task_weight_tensors
        }
```

**Acceptance Criteria:**

- [ ] Hub-to-task mapping correctly defined (EMO→emotions/sentiment/safety, etc.)
- [ ] `get_active_tasks()` returns only tasks where hub is active AND sample has data
- [ ] `get_hub_gradient_mask()` returns correct [4] tensor
- [ ] Safety always trained when `always_train_safety=True`
- [ ] Task weights normalize by number of active tasks
- [ ] Safety weight override applied correctly
- [ ] `parse_batch()` aggregates per-sample routing into batch tensors

**Tests:** `tests/v3/test_loaders_v3.py::test_hub_routing_parser`

---

#### Issue 5.3.3: Implement Multi-Task Sample Extractor

**Effort:** 5 hours | **Priority:** P0 | **Depends On:** Issue 5.3.2

**File:** `src/modeling_studio/data/extractors_v3.py`

**Purpose:** Extract all 8 task labels from unified `tasks` dict into training-ready tensors with proper tokenization alignment.

**Task Categories:**

1. **Classification**: emotions (multi-label), sentiment, safety, intent, ingress
2. **Token Classification**: ner_family, temporal (BIO tagging from char spans)
3. **Relation Extraction**: relations (subject-predicate-object triples)

**Implementation:**

```python
# src/modeling_studio/data/extractors_v3.py
"""
Multi-Task Sample Extractor for v3 Training

Extracts 8 task types from unified samples into training-ready tensors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import logging

import torch
from transformers import PreTrainedTokenizer

from .loaders_v3 import (
    UnifiedSample, SpanAnnotation, RelationTriple, TaskType
)

logger = logging.getLogger(__name__)


# ============================================================================
# Label Vocabularies
# ============================================================================

@dataclass
class LabelVocabulary:
    """Label vocabulary for a single task."""
    labels: List[str]
    label_to_id: Dict[str, int] = field(default_factory=dict)
    id_to_label: Dict[int, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.label_to_id:
            self.label_to_id = {label: i for i, label in enumerate(self.labels)}
            self.id_to_label = {i: label for label, i in self.label_to_id.items()}

    def encode(self, label: str) -> int:
        """Encode single label to id."""
        return self.label_to_id.get(label, -1)

    def encode_multi(self, labels: List[str]) -> List[int]:
        """Encode multi-label list to ids."""
        return [self.label_to_id.get(l, -1) for l in labels if l in self.label_to_id]

    def to_multi_hot(self, labels: List[str]) -> torch.Tensor:
        """Convert label list to multi-hot tensor."""
        vec = torch.zeros(len(self.labels))
        for label in labels:
            if label in self.label_to_id:
                vec[self.label_to_id[label]] = 1.0
        return vec

    @property
    def num_labels(self) -> int:
        return len(self.labels)


class V3LabelVocabularies:
    """
    All label vocabularies for v3 FamilyOS tasks.
    """

    # Emotions (28 FamilyOS emotions)
    EMOTIONS = LabelVocabulary(labels=[
        "neutral", "joy", "love", "gratitude", "hope", "excitement",
        "contentment", "pride", "amusement", "relief", "tenderness",
        "curiosity", "surprise", "sadness", "grief", "loneliness",
        "disappointment", "fear", "anxiety", "worry", "anger",
        "frustration", "annoyance", "disgust", "guilt", "shame",
        "remorse", "bittersweet"
    ])

    # Sentiment (5 levels)
    SENTIMENT = LabelVocabulary(labels=[
        "very_negative", "negative", "neutral", "positive", "very_positive", "mixed"
    ])

    # Safety levels
    SAFETY = LabelVocabulary(labels=[
        "GREEN", "AMBER", "RED", "CRISIS"
    ])

    # Intent categories
    INTENT = LabelVocabulary(labels=[
        "inform", "request", "confirm", "seek_advice", "express_emotion",
        "schedule", "remind", "plan", "reflect", "share", "ask", "command",
        "greet", "farewell", "thank", "apologize", "compliment", "complain",
        "joke", "other"
    ])

    # Ingress categories
    INGRESS = LabelVocabulary(labels=[
        "DIARY", "CHAT", "TODO", "CALENDAR", "MEMORY", "PLANNING",
        "RELATIONSHIP", "FINANCE", "HEALTH", "SHOPPING", "RECIPE",
        "TRAVEL", "KIDS", "PETS", "OTHER"
    ])

    # NER labels (BIO format)
    NER_FAMILY = LabelVocabulary(labels=[
        "O",
        "B-PERSON", "I-PERSON",
        "B-KINSHIP", "I-KINSHIP",
        "B-PET", "I-PET",
        "B-LOCATION", "I-LOCATION",
        "B-EVENT", "I-EVENT",
        "B-TRADITION", "I-TRADITION",
        "B-ORG", "I-ORG"
    ])

    # Temporal labels (BIO format)
    TEMPORAL = LabelVocabulary(labels=[
        "O",
        "B-DATE_ABS", "I-DATE_ABS",
        "B-DATE_REL", "I-DATE_REL",
        "B-TIME", "I-TIME",
        "B-DURATION", "I-DURATION",
        "B-RECURRENCE", "I-RECURRENCE"
    ])

    # Relation predicates
    RELATION_PREDICATES = LabelVocabulary(labels=[
        "parent_of", "child_of", "sibling_of", "spouse_of", "partner_of",
        "friend_of", "colleague_of", "pet_of", "owner_of", "lives_with",
        "works_at", "member_of", "attends", "related_to", "knows"
    ])


# ============================================================================
# Sample Extractor
# ============================================================================

@dataclass
class ExtractedLabels:
    """
    All extracted labels for a single sample.

    Each field is Optional - only present if sample has that task.
    """
    # Classification
    emotions: Optional[torch.Tensor] = None      # [n_emotions] multi-hot
    sentiment: Optional[int] = None              # single label id
    safety: Optional[int] = None                 # single label id
    intent: Optional[int] = None                 # single label id
    ingress: Optional[int] = None                # single label id

    # Token classification (aligned to tokens)
    ner_family_labels: Optional[torch.Tensor] = None  # [seq_len]
    temporal_labels: Optional[torch.Tensor] = None    # [seq_len]

    # Relations (list of triple ids)
    relation_triples: Optional[List[Tuple[int, int, int]]] = None

    # Original spans for debugging
    ner_spans: Optional[List[SpanAnnotation]] = None
    temporal_spans: Optional[List[SpanAnnotation]] = None


class MultiTaskExtractor:
    """
    Extracts all 8 task labels from unified samples.

    Handles:
    1. Classification label encoding
    2. Char-to-token span alignment for NER/temporal
    3. Relation triple encoding
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        vocabs: Optional[V3LabelVocabularies] = None,
        max_seq_length: int = 512
    ):
        """
        Initialize extractor.

        Args:
            tokenizer: Tokenizer for char-to-token alignment
            vocabs: Label vocabularies (uses defaults if None)
            max_seq_length: Maximum sequence length for token labels
        """
        self.tokenizer = tokenizer
        self.vocabs = vocabs or V3LabelVocabularies()
        self.max_seq_length = max_seq_length

    def extract(
        self,
        sample: UnifiedSample,
        encoding: Optional[Any] = None
    ) -> ExtractedLabels:
        """
        Extract all labels from a unified sample.

        Args:
            sample: The unified sample
            encoding: Optional tokenizer encoding for span alignment
                      If None, will tokenize sample.text

        Returns:
            ExtractedLabels with all tasks
        """
        # Tokenize if encoding not provided
        if encoding is None:
            encoding = self.tokenizer(
                sample.text,
                max_length=self.max_seq_length,
                truncation=True,
                return_offsets_mapping=True
            )

        labels = ExtractedLabels()

        # Classification tasks
        if sample.emotions:
            labels.emotions = self.vocabs.EMOTIONS.to_multi_hot(sample.emotions)

        if sample.sentiment:
            labels.sentiment = self.vocabs.SENTIMENT.encode(sample.sentiment)

        if sample.safety_familyos:
            labels.safety = self.vocabs.SAFETY.encode(sample.safety_familyos)

        if sample.intent:
            labels.intent = self.vocabs.INTENT.encode(sample.intent)

        if sample.ingress:
            labels.ingress = self.vocabs.INGRESS.encode(sample.ingress)

        # Token classification tasks (need offset mapping)
        if hasattr(encoding, 'offset_mapping') or 'offset_mapping' in encoding:
            offset_mapping = encoding.get('offset_mapping', getattr(encoding, 'offset_mapping', None))

            if offset_mapping is not None:
                if sample.ner_family:
                    labels.ner_family_labels = self._extract_bio_labels(
                        sample.ner_family,
                        offset_mapping,
                        self.vocabs.NER_FAMILY
                    )
                    labels.ner_spans = sample.ner_family

                if sample.temporal:
                    labels.temporal_labels = self._extract_bio_labels(
                        sample.temporal,
                        offset_mapping,
                        self.vocabs.TEMPORAL
                    )
                    labels.temporal_spans = sample.temporal

        # Relation extraction
        if sample.relations:
            labels.relation_triples = self._extract_relations(sample.relations)

        return labels

    def _extract_bio_labels(
        self,
        spans: List[SpanAnnotation],
        offset_mapping: List[Tuple[int, int]],
        vocab: LabelVocabulary
    ) -> torch.Tensor:
        """
        Convert character-level spans to BIO token labels.

        Args:
            spans: List of character-level span annotations
            offset_mapping: Tokenizer offset mapping [(start, end), ...]
            vocab: BIO label vocabulary

        Returns:
            Tensor of label ids [seq_len]
        """
        seq_len = len(offset_mapping)
        labels = torch.zeros(seq_len, dtype=torch.long)  # All O by default

        for span in spans:
            char_start = span.start
            char_end = span.end
            label_type = span.label

            # Find tokens overlapping with span
            is_first = True
            for token_idx, (tok_start, tok_end) in enumerate(offset_mapping):
                # Skip special tokens (offset 0,0)
                if tok_start == tok_end == 0:
                    continue

                # Check overlap
                if tok_start < char_end and tok_end > char_start:
                    # Token overlaps with span
                    if is_first:
                        bio_label = f"B-{label_type}"
                        is_first = False
                    else:
                        bio_label = f"I-{label_type}"

                    label_id = vocab.encode(bio_label)
                    if label_id >= 0:
                        labels[token_idx] = label_id

        return labels

    def _extract_relations(
        self,
        relations: List[RelationTriple]
    ) -> List[Tuple[int, int, int]]:
        """
        Extract relation triples as (subject_id, predicate_id, object_id).

        Note: For now, we encode the predicate. Subject/object are string refs
        that need entity linking at collation time.
        """
        triples = []

        for rel in relations:
            pred_id = self.vocabs.RELATION_PREDICATES.encode(rel.predicate)
            if pred_id >= 0:
                # Store as (subject_str, predicate_id, object_str)
                # Actual entity IDs resolved at batch time
                triples.append((rel.subject, pred_id, rel.object))

        return triples

    def extract_batch(
        self,
        samples: List[UnifiedSample],
        encodings: Optional[List[Any]] = None
    ) -> List[ExtractedLabels]:
        """
        Extract labels for a batch of samples.

        Args:
            samples: List of unified samples
            encodings: Optional list of tokenizer encodings

        Returns:
            List of ExtractedLabels
        """
        if encodings is None:
            encodings = [None] * len(samples)

        return [
            self.extract(sample, encoding)
            for sample, encoding in zip(samples, encodings)
        ]


# ============================================================================
# Batch Collation Helpers
# ============================================================================

def collate_classification_labels(
    labels: List[Optional[int]],
    ignore_index: int = -100
) -> torch.Tensor:
    """Collate single-label classification targets."""
    return torch.tensor([
        label if label is not None else ignore_index
        for label in labels
    ], dtype=torch.long)


def collate_multi_label(
    labels: List[Optional[torch.Tensor]],
    num_labels: int
) -> torch.Tensor:
    """Collate multi-label targets with padding."""
    batch = []
    for label in labels:
        if label is not None:
            batch.append(label)
        else:
            batch.append(torch.zeros(num_labels))
    return torch.stack(batch)


def collate_token_labels(
    labels: List[Optional[torch.Tensor]],
    max_len: int,
    ignore_index: int = -100
) -> torch.Tensor:
    """Collate token-level labels with padding."""
    batch = []
    for label in labels:
        if label is not None:
            # Pad or truncate
            if len(label) < max_len:
                padded = torch.full((max_len,), ignore_index, dtype=torch.long)
                padded[:len(label)] = label
                batch.append(padded)
            else:
                batch.append(label[:max_len])
        else:
            batch.append(torch.full((max_len,), ignore_index, dtype=torch.long))
    return torch.stack(batch)
```

**Acceptance Criteria:**

- [ ] All 8 label vocabularies correctly defined
- [ ] Multi-hot encoding for emotions
- [ ] Single-label encoding for sentiment, safety, intent, ingress
- [ ] BIO label extraction from char spans correctly aligns to tokens
- [ ] Special tokens (offset 0,0) skipped in BIO extraction
- [ ] Relation triples extracted with predicate encoding
- [ ] Batch collation helpers for all label types
- [ ] Ignore index (-100) used for padding/missing

**Tests:** `tests/v3/test_extractors_v3.py::test_multi_task_extractor`

---

#### Issue 5.3.4: Implement Hub-Weighted Loss Scaling

**Effort:** 4 hours | **Priority:** P0 | **Depends On:** Issue 5.3.2, 5.3.3

**File:** `src/modeling_studio/training/losses_v3.py`

**Purpose:** Use `hub_routing` to weight losses per task and per hub token. When `EMO=true`, emotion/sentiment/safety losses receive higher weight. This enables focused training on relevant tasks per sample.

**Loss Scaling Strategy:**

1. **Hub-active tasks**: Full loss weight (1.0)
2. **Hub-inactive but present**: Reduced weight (0.3) - maintains capability
3. **Safety override**: Always 2.0x weight regardless of hub routing
4. **Missing tasks**: Zero weight (loss masked)

**Implementation:**

```python
# src/modeling_studio/training/losses_v3.py
"""
Hub-Weighted Loss Scaling for v3 Multi-Task Training

Implements per-sample, per-task loss weighting based on hub_routing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.loaders_v3 import HubRouting, TaskType

logger = logging.getLogger(__name__)


# ============================================================================
# Loss Weight Configuration
# ============================================================================

@dataclass
class HubLossConfig:
    """
    Configuration for hub-weighted loss scaling.
    """
    # Weight when hub is active and task has data
    active_weight: float = 1.0

    # Weight when hub is inactive but task has data (maintains capability)
    inactive_weight: float = 0.3

    # Safety task always gets this multiplier (on top of active/inactive)
    safety_multiplier: float = 2.0

    # Whether to always train safety regardless of hub routing
    always_train_safety: bool = True

    # Per-task base weights (before hub scaling)
    task_base_weights: Dict[str, float] = field(default_factory=lambda: {
        "emotions": 1.0,
        "sentiment": 1.0,
        "safety_familyos": 1.0,
        "intent": 0.8,
        "ingress": 0.8,
        "ner_family": 1.0,
        "temporal": 1.0,
        "relations": 1.2,  # Relations are harder, slightly higher weight
    })

    # Hub → Task mapping
    hub_to_tasks: Dict[str, List[str]] = field(default_factory=lambda: {
        "EMO": ["emotions", "sentiment", "safety_familyos"],
        "REL": ["relations"],
        "MEM": ["temporal", "ner_family"],
        "TASK": ["intent", "ingress"]
    })


# ============================================================================
# Per-Sample Loss Weight Calculator
# ============================================================================

class HubLossWeightCalculator:
    """
    Calculates per-sample, per-task loss weights based on hub routing.
    """

    def __init__(self, config: Optional[HubLossConfig] = None):
        """
        Initialize calculator.

        Args:
            config: Loss weight configuration
        """
        self.config = config or HubLossConfig()

        # Build task → hub mapping (reverse of hub → tasks)
        self.task_to_hub: Dict[str, str] = {}
        for hub, tasks in self.config.hub_to_tasks.items():
            for task in tasks:
                self.task_to_hub[task] = hub

    def compute_weight(
        self,
        task_name: str,
        hub_routing: HubRouting,
        has_label: bool
    ) -> float:
        """
        Compute loss weight for a single task on a single sample.

        Args:
            task_name: Name of the task (e.g., "emotions", "safety_familyos")
            hub_routing: Hub routing for the sample
            has_label: Whether the sample has a label for this task

        Returns:
            Loss weight for this task on this sample
        """
        if not has_label:
            return 0.0

        # Get base weight
        base_weight = self.config.task_base_weights.get(task_name, 1.0)

        # Get controlling hub
        hub = self.task_to_hub.get(task_name)

        # Check if hub is active
        hub_active = False
        if hub == "EMO":
            hub_active = hub_routing.emo
        elif hub == "REL":
            hub_active = hub_routing.rel
        elif hub == "MEM":
            hub_active = hub_routing.mem
        elif hub == "TASK":
            hub_active = hub_routing.task

        # Apply hub scaling
        if hub_active:
            weight = base_weight * self.config.active_weight
        else:
            weight = base_weight * self.config.inactive_weight

        # Safety override
        if task_name == "safety_familyos":
            if self.config.always_train_safety:
                # Always use at least inactive weight for safety
                weight = max(weight, base_weight * self.config.inactive_weight)
            weight *= self.config.safety_multiplier

        return weight

    def compute_batch_weights(
        self,
        task_name: str,
        hub_routings: List[HubRouting],
        has_labels: List[bool]
    ) -> torch.Tensor:
        """
        Compute weights for a batch.

        Args:
            task_name: Name of the task
            hub_routings: Hub routing for each sample
            has_labels: Whether each sample has a label

        Returns:
            Tensor of weights [batch_size]
        """
        weights = [
            self.compute_weight(task_name, hr, hl)
            for hr, hl in zip(hub_routings, has_labels)
        ]
        return torch.tensor(weights, dtype=torch.float32)


# ============================================================================
# Hub-Weighted Multi-Task Loss
# ============================================================================

class HubWeightedMultiTaskLoss(nn.Module):
    """
    Multi-task loss with hub-based per-sample weighting.

    Supports:
    - Classification losses (single-label, multi-label)
    - Token classification losses (NER, temporal)
    - Per-sample task weighting based on hub_routing
    """

    def __init__(
        self,
        config: Optional[HubLossConfig] = None,
        label_smoothing: float = 0.1
    ):
        """
        Initialize loss module.

        Args:
            config: Hub loss configuration
            label_smoothing: Label smoothing for classification
        """
        super().__init__()

        self.config = config or HubLossConfig()
        self.weight_calculator = HubLossWeightCalculator(self.config)
        self.label_smoothing = label_smoothing

        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss(
            reduction='none',
            label_smoothing=label_smoothing
        )
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(
        self,
        task_logits: Dict[str, torch.Tensor],
        task_labels: Dict[str, torch.Tensor],
        hub_routings: List[HubRouting],
        task_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute hub-weighted multi-task loss.

        Args:
            task_logits: Dict[task_name, logits tensor]
            task_labels: Dict[task_name, labels tensor]
            hub_routings: List of hub routing per sample
            task_masks: Optional masks per task (1 = compute loss, 0 = ignore)

        Returns:
            Tuple of (total_loss, per_task_losses)
        """
        total_loss = torch.tensor(0.0, device=self._get_device(task_logits))
        task_losses: Dict[str, torch.Tensor] = {}

        batch_size = len(hub_routings)

        for task_name, logits in task_logits.items():
            if task_name not in task_labels:
                continue

            labels = task_labels[task_name]

            # Determine task type and compute loss
            if task_name == "emotions":
                # Multi-label BCE
                loss = self.bce_loss(logits, labels.float())
                loss = loss.mean(dim=-1)  # Average across labels
            elif task_name in ["ner_family_labels", "temporal_labels"]:
                # Token classification
                loss = self._compute_token_loss(logits, labels)
            else:
                # Single-label classification
                loss = self.ce_loss(logits, labels)

            # Compute hub-weighted sample weights
            has_labels = self._get_has_labels(labels, task_name)
            weights = self.weight_calculator.compute_batch_weights(
                task_name.replace("_labels", ""),  # Handle ner_family_labels → ner_family
                hub_routings,
                has_labels
            ).to(loss.device)

            # Apply optional mask
            if task_masks and task_name in task_masks:
                weights = weights * task_masks[task_name]

            # Weight and average
            weighted_loss = (loss * weights).sum()
            weight_sum = weights.sum()

            if weight_sum > 0:
                task_loss = weighted_loss / weight_sum
            else:
                task_loss = torch.tensor(0.0, device=loss.device)

            task_losses[task_name] = task_loss
            total_loss = total_loss + task_loss

        return total_loss, task_losses

    def _compute_token_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = -100
    ) -> torch.Tensor:
        """
        Compute token classification loss.

        Args:
            logits: [batch, seq_len, num_labels]
            labels: [batch, seq_len]
            ignore_index: Label index to ignore

        Returns:
            Per-sample loss [batch]
        """
        batch_size, seq_len, num_labels = logits.shape

        # Flatten for CE loss
        logits_flat = logits.view(-1, num_labels)
        labels_flat = labels.view(-1)

        # Compute loss (ignoring padding)
        loss = F.cross_entropy(
            logits_flat, labels_flat,
            ignore_index=ignore_index,
            reduction='none'
        )

        # Reshape and average per sample
        loss = loss.view(batch_size, seq_len)

        # Count valid tokens per sample
        valid_mask = (labels != ignore_index).float()
        valid_counts = valid_mask.sum(dim=1).clamp(min=1)

        # Average per sample
        sample_loss = (loss * valid_mask).sum(dim=1) / valid_counts

        return sample_loss

    def _get_has_labels(
        self,
        labels: torch.Tensor,
        task_name: str
    ) -> List[bool]:
        """Check which samples have valid labels."""
        if task_name == "emotions":
            # Multi-label: has label if any emotion is 1
            return (labels.sum(dim=-1) > 0).tolist()
        elif task_name in ["ner_family_labels", "temporal_labels"]:
            # Token classification: has label if any non-O, non-padding
            return ((labels > 0) & (labels != -100)).any(dim=-1).tolist()
        else:
            # Single label: has label if not -100
            return (labels != -100).tolist()

    def _get_device(self, task_logits: Dict[str, torch.Tensor]) -> torch.device:
        """Get device from logits."""
        for logits in task_logits.values():
            return logits.device
        return torch.device('cpu')


# ============================================================================
# Hub Gradient Mask Loss Wrapper
# ============================================================================

class HubGradientMaskedLoss(nn.Module):
    """
    Wrapper that applies hub token gradient masking.

    Uses hub_routing to mask gradients for inactive hub tokens
    during the backward pass.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        hub_token_positions: Optional[Dict[str, int]] = None
    ):
        """
        Initialize wrapper.

        Args:
            base_loss: The underlying loss module
            hub_token_positions: Position of each hub token in sequence
                                 Default: {"EMO": 1, "REL": 2, "MEM": 3, "TASK": 4}
        """
        super().__init__()

        self.base_loss = base_loss
        self.hub_token_positions = hub_token_positions or {
            "EMO": 1, "REL": 2, "MEM": 3, "TASK": 4
        }

    def get_hub_gradient_mask(
        self,
        hub_routings: List[HubRouting],
        seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create gradient mask for hub tokens.

        Args:
            hub_routings: Hub routing per sample
            seq_len: Sequence length
            device: Device for tensor

        Returns:
            Mask [batch, seq_len] where 0 = mask gradient, 1 = allow gradient
        """
        batch_size = len(hub_routings)
        mask = torch.ones(batch_size, seq_len, device=device)

        for i, routing in enumerate(hub_routings):
            # Mask inactive hub tokens
            if not routing.emo:
                mask[i, self.hub_token_positions["EMO"]] = 0.0
            if not routing.rel:
                mask[i, self.hub_token_positions["REL"]] = 0.0
            if not routing.mem:
                mask[i, self.hub_token_positions["MEM"]] = 0.0
            if not routing.task:
                mask[i, self.hub_token_positions["TASK"]] = 0.0

        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        hub_routings: List[HubRouting],
        **loss_kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with hub gradient masking.

        Note: The gradient masking is applied via a hook on hidden_states.
        This should be called before the task heads.
        """
        # Create gradient mask
        mask = self.get_hub_gradient_mask(
            hub_routings,
            hidden_states.shape[1],
            hidden_states.device
        )

        # Apply mask as a gradient hook
        if hidden_states.requires_grad:
            hidden_states.register_hook(
                lambda grad: grad * mask.unsqueeze(-1)
            )

        # Call base loss
        return self.base_loss(**loss_kwargs)


# ============================================================================
# Loss Aggregation Utilities
# ============================================================================

def aggregate_task_losses(
    task_losses: Dict[str, torch.Tensor],
    task_weights: Optional[Dict[str, float]] = None
) -> torch.Tensor:
    """
    Aggregate per-task losses into total loss.

    Args:
        task_losses: Dict[task_name, loss]
        task_weights: Optional per-task weights

    Returns:
        Total weighted loss
    """
    if task_weights is None:
        task_weights = {}

    total = torch.tensor(0.0)

    for task_name, loss in task_losses.items():
        weight = task_weights.get(task_name, 1.0)
        total = total + (loss * weight)

    return total


def log_task_losses(
    task_losses: Dict[str, torch.Tensor],
    prefix: str = "train"
) -> Dict[str, float]:
    """
    Convert task losses to logging format.

    Args:
        task_losses: Dict[task_name, loss]
        prefix: Prefix for log keys

    Returns:
        Dict for wandb/logging
    """
    return {
        f"{prefix}/loss_{task_name}": loss.item()
        for task_name, loss in task_losses.items()
    }
```

**Acceptance Criteria:**

- [ ] `HubLossConfig` configurable for active/inactive weights
- [ ] Safety always gets multiplier regardless of hub routing
- [ ] Per-sample weights computed based on hub_routing
- [ ] Multi-label BCE loss for emotions
- [ ] Token classification loss with ignore_index handling
- [ ] Single-label CE loss with label smoothing
- [ ] Hub gradient masking applied via hook
- [ ] Loss aggregation and logging utilities provided

**Tests:** `tests/v3/test_losses_v3.py::test_hub_weighted_loss`

---

#### Issue 5.3.5: Implement Shard-Based Data Loading

**Effort:** 4 hours | **Priority:** P0 | **Depends On:** Issue 5.3.1

**File:** `src/modeling_studio/data/shard_loader_v3.py`

**Purpose:** Support loading multiple `shard_*.jsonl` files with streaming/memory efficiency for large-scale FamilyOS training data.

**Requirements:**

- Memory-efficient streaming for datasets that don't fit in RAM
- Parallel shard loading for faster data ingestion
- Worker-aware shard distribution for distributed training
- Resume capability for interrupted training
- Shard statistics and validation

**Implementation:**

```python
# src/modeling_studio/data/shard_loader_v3.py
"""
Shard-Based Data Loading for v3 Multi-Task Training

Provides memory-efficient loading of large JSONL shard files
with support for:
- Streaming iteration
- Parallel shard loading
- Worker-aware distribution
- Resume from checkpoint
"""
from __future__ import annotations

import json
import glob
import os
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import (
    Dict, List, Optional, Iterator, Any,
    Tuple, Callable, Union
)
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import torch
from torch.utils.data import IterableDataset, get_worker_info

from .loaders_v3 import UnifiedSample, HubRouting, TaskType

logger = logging.getLogger(__name__)


# ============================================================================
# Shard Configuration
# ============================================================================

@dataclass
class ShardConfig:
    """Configuration for shard-based loading."""

    # Paths
    data_dir: str
    shard_pattern: str = "shard_*.jsonl"

    # Loading behavior
    shuffle_shards: bool = True
    shuffle_within_shard: bool = False  # Memory intensive if True

    # Memory management
    buffer_size: int = 10000  # Samples to buffer per worker
    prefetch_shards: int = 2  # Number of shards to prefetch

    # Parallel loading
    num_loading_threads: int = 2

    # Filtering
    min_text_length: int = 5
    max_text_length: int = 2000
    require_hub_routing: bool = False
    filter_tasks: Optional[List[str]] = None

    # Validation
    validate_samples: bool = True
    skip_invalid: bool = True

    # Resume support
    checkpoint_path: Optional[str] = None

    # Statistics
    collect_stats: bool = True


@dataclass
class ShardStats:
    """Statistics for a single shard."""
    shard_path: str
    num_samples: int = 0
    num_valid: int = 0
    num_skipped: int = 0
    task_counts: Dict[str, int] = field(default_factory=dict)
    hub_counts: Dict[str, int] = field(default_factory=dict)
    avg_text_length: float = 0.0

    def merge(self, other: "ShardStats") -> "ShardStats":
        """Merge statistics from another shard."""
        merged = ShardStats(shard_path=f"{self.shard_path}+{other.shard_path}")
        merged.num_samples = self.num_samples + other.num_samples
        merged.num_valid = self.num_valid + other.num_valid
        merged.num_skipped = self.num_skipped + other.num_skipped

        # Merge task counts
        for task, count in self.task_counts.items():
            merged.task_counts[task] = count
        for task, count in other.task_counts.items():
            merged.task_counts[task] = merged.task_counts.get(task, 0) + count

        # Merge hub counts
        for hub, count in self.hub_counts.items():
            merged.hub_counts[hub] = count
        for hub, count in other.hub_counts.items():
            merged.hub_counts[hub] = merged.hub_counts.get(hub, 0) + count

        # Weighted average text length
        if merged.num_valid > 0:
            total_length = (
                self.avg_text_length * self.num_valid +
                other.avg_text_length * other.num_valid
            )
            merged.avg_text_length = total_length / merged.num_valid

        return merged


# ============================================================================
# Shard Index
# ============================================================================

@dataclass
class ShardIndex:
    """
    Index of available shards with metadata.

    Enables efficient shard selection and distribution.
    """
    shards: List[Dict[str, Any]] = field(default_factory=list)
    total_samples: int = 0
    total_shards: int = 0

    @classmethod
    def build(cls, data_dir: str, shard_pattern: str = "shard_*.jsonl") -> "ShardIndex":
        """Build index by scanning shard files."""
        index = cls()

        shard_files = sorted(glob.glob(str(Path(data_dir) / shard_pattern)))
        index.total_shards = len(shard_files)

        logger.info(f"Indexing {len(shard_files)} shard files...")

        for shard_path in shard_files:
            # Count lines (samples) in shard
            with open(shard_path, 'r', encoding='utf-8') as f:
                num_samples = sum(1 for line in f if line.strip())

            index.shards.append({
                "path": shard_path,
                "num_samples": num_samples,
                "size_bytes": os.path.getsize(shard_path)
            })
            index.total_samples += num_samples

        logger.info(f"Index complete: {index.total_samples} samples across {index.total_shards} shards")
        return index

    def get_worker_shards(
        self,
        worker_id: int,
        num_workers: int
    ) -> List[Dict[str, Any]]:
        """Get shards assigned to a specific worker."""
        return [
            shard for i, shard in enumerate(self.shards)
            if i % num_workers == worker_id
        ]

    def save(self, path: str) -> None:
        """Save index to disk."""
        with open(path, 'w') as f:
            json.dump({
                "shards": self.shards,
                "total_samples": self.total_samples,
                "total_shards": self.total_shards
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ShardIndex":
        """Load index from disk."""
        with open(path, 'r') as f:
            data = json.load(f)

        index = cls()
        index.shards = data["shards"]
        index.total_samples = data["total_samples"]
        index.total_shards = data["total_shards"]
        return index


# ============================================================================
# Shard Reader
# ============================================================================

class ShardReader:
    """
    Reads samples from a single shard file.

    Supports:
    - Line-by-line streaming
    - Sample validation
    - Statistics collection
    """

    def __init__(
        self,
        shard_path: str,
        config: ShardConfig
    ):
        self.shard_path = shard_path
        self.config = config
        self.stats = ShardStats(shard_path=shard_path)

    def __iter__(self) -> Iterator[UnifiedSample]:
        """Iterate over samples in shard."""
        text_lengths = []

        with open(self.shard_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                self.stats.num_samples += 1

                try:
                    data = json.loads(line)
                    sample = UnifiedSample.from_json(data)

                    # Validate sample
                    if not self._validate_sample(sample):
                        self.stats.num_skipped += 1
                        if self.config.skip_invalid:
                            continue

                    self.stats.num_valid += 1

                    # Collect statistics
                    if self.config.collect_stats:
                        text_lengths.append(len(sample.text))
                        self._update_task_stats(sample)
                        self._update_hub_stats(sample)

                    yield sample

                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in {self.shard_path}: {e}")
                    self.stats.num_skipped += 1
                except Exception as e:
                    logger.warning(f"Error parsing sample: {e}")
                    self.stats.num_skipped += 1

        # Calculate average text length
        if text_lengths:
            self.stats.avg_text_length = sum(text_lengths) / len(text_lengths)

    def _validate_sample(self, sample: UnifiedSample) -> bool:
        """Validate a sample against config filters."""
        # Text length
        text_len = len(sample.text)
        if text_len < self.config.min_text_length:
            return False
        if text_len > self.config.max_text_length:
            return False

        # Hub routing requirement
        if self.config.require_hub_routing:
            if not sample.hub_routing.active_hubs:
                return False

        # Task filter
        if self.config.filter_tasks:
            has_task = any(
                sample.has_task(TaskType(task))
                for task in self.config.filter_tasks
                if task in [t.value for t in TaskType]
            )
            if not has_task:
                return False

        return True

    def _update_task_stats(self, sample: UnifiedSample) -> None:
        """Update task statistics."""
        for task_type in TaskType:
            if sample.has_task(task_type):
                task_name = task_type.value
                self.stats.task_counts[task_name] = (
                    self.stats.task_counts.get(task_name, 0) + 1
                )

    def _update_hub_stats(self, sample: UnifiedSample) -> None:
        """Update hub routing statistics."""
        for hub in sample.hub_routing.active_hubs:
            self.stats.hub_counts[hub] = self.stats.hub_counts.get(hub, 0) + 1


# ============================================================================
# Streaming Shard Dataset
# ============================================================================

class StreamingShardDataset(IterableDataset):
    """
    Memory-efficient streaming dataset over multiple shards.

    Features:
    - Worker-aware shard distribution
    - Prefetching for smooth iteration
    - Resume from checkpoint
    - Epoch shuffling
    """

    def __init__(
        self,
        config: ShardConfig,
        transform: Optional[Callable[[UnifiedSample], Any]] = None,
        epoch: int = 0
    ):
        """
        Initialize streaming dataset.

        Args:
            config: Shard configuration
            transform: Optional transform to apply to samples
            epoch: Current epoch (for shuffling seed)
        """
        self.config = config
        self.transform = transform
        self.epoch = epoch

        # Build or load shard index
        self.index = ShardIndex.build(config.data_dir, config.shard_pattern)

        # Aggregate statistics
        self.total_stats: Optional[ShardStats] = None

        logger.info(
            f"StreamingShardDataset initialized: "
            f"{self.index.total_samples} samples, "
            f"{self.index.total_shards} shards"
        )

    def __iter__(self) -> Iterator[Any]:
        """Iterate over samples from all shards."""
        worker_info = get_worker_info()

        if worker_info is None:
            # Single-process loading
            worker_id = 0
            num_workers = 1
        else:
            # Multi-process loading
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Get shards for this worker
        worker_shards = self.index.get_worker_shards(worker_id, num_workers)

        if not worker_shards:
            logger.warning(f"Worker {worker_id} has no assigned shards")
            return

        logger.debug(
            f"Worker {worker_id}/{num_workers}: "
            f"processing {len(worker_shards)} shards"
        )

        # Shuffle shard order per epoch
        if self.config.shuffle_shards:
            rng = random.Random(42 + self.epoch + worker_id)
            worker_shards = worker_shards.copy()
            rng.shuffle(worker_shards)

        # Track statistics
        all_stats: List[ShardStats] = []

        # Process shards
        for shard_info in worker_shards:
            shard_path = shard_info["path"]
            reader = ShardReader(shard_path, self.config)

            samples = list(reader)  # Read shard

            # Shuffle within shard if requested
            if self.config.shuffle_within_shard:
                rng = random.Random(42 + self.epoch + hash(shard_path))
                rng.shuffle(samples)

            # Yield samples
            for sample in samples:
                if self.transform:
                    yield self.transform(sample)
                else:
                    yield sample

            # Collect stats
            if self.config.collect_stats:
                all_stats.append(reader.stats)

        # Merge statistics
        if all_stats and self.config.collect_stats:
            merged = all_stats[0]
            for stats in all_stats[1:]:
                merged = merged.merge(stats)
            self.total_stats = merged

    def __len__(self) -> int:
        """Return approximate length (total samples across all shards)."""
        return self.index.total_samples

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for shuffling."""
        self.epoch = epoch

    def get_stats(self) -> Optional[ShardStats]:
        """Get aggregated statistics (available after full iteration)."""
        return self.total_stats


# ============================================================================
# Buffered Shard Dataset
# ============================================================================

class BufferedShardDataset(IterableDataset):
    """
    Buffered streaming dataset with prefetching.

    Maintains a buffer of samples for smoother iteration
    and supports parallel shard loading.
    """

    def __init__(
        self,
        config: ShardConfig,
        transform: Optional[Callable[[UnifiedSample], Any]] = None,
        epoch: int = 0
    ):
        self.config = config
        self.transform = transform
        self.epoch = epoch

        self.index = ShardIndex.build(config.data_dir, config.shard_pattern)
        self._buffer: List[UnifiedSample] = []
        self._buffer_lock = threading.Lock()

    def _prefetch_shard(self, shard_path: str) -> List[UnifiedSample]:
        """Load a shard in background thread."""
        reader = ShardReader(shard_path, self.config)
        return list(reader)

    def __iter__(self) -> Iterator[Any]:
        """Iterate with buffering and prefetching."""
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        worker_shards = self.index.get_worker_shards(worker_id, num_workers)

        if self.config.shuffle_shards:
            rng = random.Random(42 + self.epoch + worker_id)
            worker_shards = worker_shards.copy()
            rng.shuffle(worker_shards)

        # Use thread pool for prefetching
        with ThreadPoolExecutor(max_workers=self.config.num_loading_threads) as executor:
            # Submit first batch of shards
            futures = {}
            shard_iter = iter(worker_shards)

            for _ in range(min(self.config.prefetch_shards, len(worker_shards))):
                try:
                    shard_info = next(shard_iter)
                    future = executor.submit(self._prefetch_shard, shard_info["path"])
                    futures[future] = shard_info["path"]
                except StopIteration:
                    break

            # Process completed shards and submit new ones
            while futures:
                # Wait for any shard to complete
                done, _ = as_completed(futures, return_when="FIRST_COMPLETED")

                for future in done:
                    shard_path = futures.pop(future)
                    samples = future.result()

                    # Shuffle within buffer if requested
                    if self.config.shuffle_within_shard:
                        rng = random.Random(42 + self.epoch + hash(shard_path))
                        rng.shuffle(samples)

                    # Yield samples
                    for sample in samples:
                        if self.transform:
                            yield self.transform(sample)
                        else:
                            yield sample

                    # Submit next shard
                    try:
                        shard_info = next(shard_iter)
                        new_future = executor.submit(
                            self._prefetch_shard, shard_info["path"]
                        )
                        futures[new_future] = shard_info["path"]
                    except StopIteration:
                        pass

    def __len__(self) -> int:
        return self.index.total_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


# ============================================================================
# Factory Functions
# ============================================================================

def create_shard_dataset(
    data_dir: str,
    shard_pattern: str = "shard_*.jsonl",
    streaming: bool = True,
    buffered: bool = True,
    transform: Optional[Callable] = None,
    **config_kwargs
) -> Union[StreamingShardDataset, BufferedShardDataset]:
    """
    Create a shard-based dataset.

    Args:
        data_dir: Directory containing shard files
        shard_pattern: Glob pattern for shards
        streaming: Use streaming (True) vs loading all (False)
        buffered: Use buffered prefetching
        transform: Optional sample transform
        **config_kwargs: Additional ShardConfig options

    Returns:
        Appropriate dataset instance
    """
    config = ShardConfig(
        data_dir=data_dir,
        shard_pattern=shard_pattern,
        **config_kwargs
    )

    if buffered:
        return BufferedShardDataset(config, transform)
    else:
        return StreamingShardDataset(config, transform)


def get_shard_statistics(
    data_dir: str,
    shard_pattern: str = "shard_*.jsonl"
) -> ShardStats:
    """
    Compute aggregate statistics over all shards.

    Args:
        data_dir: Directory containing shard files
        shard_pattern: Glob pattern for shards

    Returns:
        Merged statistics from all shards
    """
    config = ShardConfig(
        data_dir=data_dir,
        shard_pattern=shard_pattern,
        collect_stats=True
    )

    shard_files = sorted(glob.glob(str(Path(data_dir) / shard_pattern)))

    all_stats: List[ShardStats] = []

    for shard_path in shard_files:
        reader = ShardReader(shard_path, config)
        # Consume iterator to collect stats
        for _ in reader:
            pass
        all_stats.append(reader.stats)

    # Merge all
    if not all_stats:
        return ShardStats(shard_path=data_dir)

    merged = all_stats[0]
    for stats in all_stats[1:]:
        merged = merged.merge(stats)

    return merged
```

**Acceptance Criteria:**

- [ ] `ShardConfig` provides comprehensive configuration options
- [ ] `ShardIndex` builds index of shards with sample counts
- [ ] `ShardReader` streams samples from individual shards
- [ ] `StreamingShardDataset` is memory-efficient for large datasets
- [ ] `BufferedShardDataset` uses prefetching for better throughput
- [ ] Worker-aware distribution for multi-worker DataLoader
- [ ] Per-epoch shuffling at shard level
- [ ] Statistics collection for task/hub distributions
- [ ] Invalid sample handling (skip or raise)

**Tests:** `tests/v3/test_shard_loader.py::test_streaming_shard_dataset`

---

#### Issue 5.3.6: Create Unified FamilyOS Dataset Configuration

**Effort:** 3 hours | **Priority:** P0 | **Depends On:** Issue 5.3.5

**File:** `configs/data/multitask/familyos_unified.yaml`

**Purpose:** Define comprehensive YAML configuration for loading unified FamilyOS data with hub routing, multi-task extraction, and shard-based loading.

**Implementation:**

```yaml
# configs/data/multitask/familyos_unified.yaml
#
# Unified FamilyOS Dataset Configuration for ModernBERT v3
#
# This configuration defines how to load and process the unified
# FamilyOS JSONL data with 8 task types and hub routing.
#
# Data Format:
#   {
#     "id": "fam_00005",
#     "text": "...",
#     "tasks": { emotions, sentiment, ner_family, safety_familyos,
#                intent, ingress, relations, temporal },
#     "hub_routing": { EMO, REL, MEM, TASK }
#   }

# ═══════════════════════════════════════════════════════════════
# Dataset Metadata
# ═══════════════════════════════════════════════════════════════
dataset:
  name: "familyos_unified"
  version: "1.0"
  description: "Unified FamilyOS multi-task data with hub routing"
  format: "jsonl"
  purpose: "Phase 1+ training with FamilyOS-specific tasks"

# ═══════════════════════════════════════════════════════════════
# Data Paths
# ═══════════════════════════════════════════════════════════════
paths:
  # Primary data directory
  data_dir: "data/familyos/unified/output"
  shard_pattern: "shard_*.jsonl"

  # Cache and index
  cache_dir: "data/cache/familyos_unified"
  index_file: "data/cache/familyos_unified/shard_index.json"

  # Validation split (optional pre-split)
  train_shards: "shard_000[0-7].jsonl"
  val_shards: "shard_0008.jsonl"
  test_shards: "shard_0009.jsonl"

# ═══════════════════════════════════════════════════════════════
# Task Configurations (8 Tasks)
# ═══════════════════════════════════════════════════════════════
tasks:
  # === CLASSIFICATION TASKS ===

  emotions:
    type: "multi_label"
    num_labels: 28
    labels:
      - neutral
      - joy
      - love
      - gratitude
      - hope
      - excitement
      - contentment
      - pride
      - amusement
      - relief
      - tenderness
      - curiosity
      - surprise
      - sadness
      - grief
      - loneliness
      - disappointment
      - fear
      - anxiety
      - worry
      - anger
      - frustration
      - annoyance
      - disgust
      - guilt
      - shame
      - remorse
      - bittersweet
    hub: "EMO"
    weight: 1.0
    loss: "bce_with_logits"
    metrics:
      - macro_f1
      - micro_f1
      - hamming_loss

  sentiment:
    type: "single_label"
    num_labels: 6
    labels:
      - very_negative
      - negative
      - neutral
      - positive
      - very_positive
      - mixed
    hub: "EMO"
    weight: 1.0
    loss: "cross_entropy"
    label_smoothing: 0.1
    metrics:
      - accuracy
      - macro_f1

  safety_familyos:
    type: "single_label"
    num_labels: 4
    labels:
      - GREEN
      - AMBER
      - RED
      - CRISIS
    hub: "EMO"
    weight: 2.0  # Safety always prioritized
    loss: "cross_entropy"
    label_smoothing: 0.05  # Less smoothing for safety
    class_weights:
      GREEN: 1.0
      AMBER: 1.5
      RED: 2.0
      CRISIS: 3.0  # Heavily weight CRISIS detection
    metrics:
      - accuracy
      - crisis_recall  # Critical metric
      - confusion_matrix

  intent:
    type: "single_label"
    num_labels: 20
    labels:
      - inform
      - request
      - confirm
      - seek_advice
      - express_emotion
      - schedule
      - remind
      - plan
      - reflect
      - share
      - ask
      - command
      - greet
      - farewell
      - thank
      - apologize
      - compliment
      - complain
      - joke
      - other
    hub: "TASK"
    weight: 0.8
    loss: "cross_entropy"
    label_smoothing: 0.1
    metrics:
      - accuracy
      - macro_f1

  ingress:
    type: "single_label"
    num_labels: 15
    labels:
      - DIARY
      - CHAT
      - TODO
      - CALENDAR
      - MEMORY
      - PLANNING
      - RELATIONSHIP
      - FINANCE
      - HEALTH
      - SHOPPING
      - RECIPE
      - TRAVEL
      - KIDS
      - PETS
      - OTHER
    hub: "TASK"
    weight: 0.8
    loss: "cross_entropy"
    label_smoothing: 0.1
    metrics:
      - accuracy
      - macro_f1

  # === TOKEN CLASSIFICATION TASKS ===

  ner_family:
    type: "token_classification"
    num_labels: 15  # BIO tags
    labels:
      - O
      - B-PERSON
      - I-PERSON
      - B-KINSHIP
      - I-KINSHIP
      - B-PET
      - I-PET
      - B-LOCATION
      - I-LOCATION
      - B-EVENT
      - I-EVENT
      - B-TRADITION
      - I-TRADITION
      - B-ORG
      - I-ORG
    hub: "MEM"
    weight: 1.0
    loss: "cross_entropy"
    ignore_index: -100
    span_format: "char"  # Spans are character offsets
    metrics:
      - entity_f1
      - entity_precision
      - entity_recall

  temporal:
    type: "token_classification"
    num_labels: 11  # BIO tags
    labels:
      - O
      - B-DATE_ABS
      - I-DATE_ABS
      - B-DATE_REL
      - I-DATE_REL
      - B-TIME
      - I-TIME
      - B-DURATION
      - I-DURATION
      - B-RECURRENCE
      - I-RECURRENCE
    hub: "MEM"
    weight: 1.0
    loss: "cross_entropy"
    ignore_index: -100
    span_format: "char"
    metrics:
      - entity_f1
      - temporal_precision
      - temporal_recall

  # === RELATION EXTRACTION ===

  relations:
    type: "relation_extraction"
    num_predicates: 15
    predicates:
      - parent_of
      - child_of
      - sibling_of
      - spouse_of
      - partner_of
      - friend_of
      - colleague_of
      - pet_of
      - owner_of
      - lives_with
      - works_at
      - member_of
      - attends
      - related_to
      - knows
    hub: "REL"
    weight: 1.2  # Relations are harder
    loss: "cross_entropy"
    metrics:
      - relation_f1
      - relation_precision
      - relation_recall

# ═══════════════════════════════════════════════════════════════
# Hub Routing Configuration
# ═══════════════════════════════════════════════════════════════
hub_routing:
  # Hub → Task mapping
  hub_to_tasks:
    EMO:
      - emotions
      - sentiment
      - safety_familyos
    REL:
      - relations
    MEM:
      - ner_family
      - temporal
    TASK:
      - intent
      - ingress

  # Loss weighting based on hub activation
  loss_weighting:
    active_weight: 1.0      # Weight when hub is active
    inactive_weight: 0.3    # Weight when hub is inactive but task has data
    safety_override: true   # Safety always trained regardless
    safety_multiplier: 2.0  # Extra weight for safety

# ═══════════════════════════════════════════════════════════════
# Preprocessing Configuration
# ═══════════════════════════════════════════════════════════════
preprocessing:
  # Tokenization
  max_length: 512
  padding: "max_length"
  truncation: true
  return_offsets_mapping: true  # Needed for span alignment

  # Hub tokens
  add_hub_tokens: true
  hub_token_offset: 5  # [CLS] + 4 hub tokens
  hub_tokens:
    - "[EMO]"
    - "[REL]"
    - "[MEM]"
    - "[TASK]"

  # Text cleaning
  lowercase: false
  strip_whitespace: true
  normalize_unicode: true

# ═══════════════════════════════════════════════════════════════
# Shard Loading Configuration
# ═══════════════════════════════════════════════════════════════
shard_loading:
  # Streaming behavior
  streaming: true
  buffered: true

  # Memory management
  buffer_size: 10000
  prefetch_shards: 2

  # Shuffling
  shuffle_shards: true
  shuffle_within_shard: false  # Too memory intensive

  # Filtering
  min_text_length: 5
  max_text_length: 2000
  require_hub_routing: false

  # Validation
  validate_samples: true
  skip_invalid: true

  # Parallel loading
  num_loading_threads: 2

# ═══════════════════════════════════════════════════════════════
# Data Loading Configuration
# ═══════════════════════════════════════════════════════════════
loading:
  batch_size: 32
  num_workers: 4
  pin_memory: true
  prefetch_factor: 2
  persistent_workers: true
  drop_last: false

# ═══════════════════════════════════════════════════════════════
# Sampling Strategy
# ═══════════════════════════════════════════════════════════════
sampling:
  strategy: "hub_weighted"  # Weight by hub routing

  # Task weights (on top of hub weighting)
  task_weights:
    emotions: 1.0
    sentiment: 1.0
    safety_familyos: 2.0  # Always prioritize safety
    intent: 0.8
    ingress: 0.8
    ner_family: 1.0
    temporal: 1.0
    relations: 1.2

  # Oversampling rare hubs
  oversample_rare_hubs: false
  hub_balance_factor: 0.5

  # Replay (for capability preservation)
  replay:
    enabled: true
    healing_data: "configs/data/multitask/healing_enhanced.yaml"
    ratio: 0.15
    task_balanced: true

# ═══════════════════════════════════════════════════════════════
# Validation Configuration
# ═══════════════════════════════════════════════════════════════
validation:
  enabled: true
  split_ratio: 0.1
  stratify_by: "hub_routing"  # or "task" or "none"

  # Validation frequency
  eval_steps: 500
  eval_strategy: "steps"

  # Metrics to compute
  compute_metrics:
    - per_task_loss
    - per_task_accuracy
    - hub_activation_stats
    - safety_metrics

# ═══════════════════════════════════════════════════════════════
# Collation Configuration
# ═══════════════════════════════════════════════════════════════
collation:
  collator_type: "v3_multitask"

  # Hub token handling
  insert_hub_tokens: true
  hub_token_positions:
    CLS: 0
    EMO: 1
    REL: 2
    MEM: 3
    TASK: 4

  # Label collation
  classification_ignore_index: -100
  token_classification_ignore_index: -100

  # Multi-task batching
  mixed_task_batches: true
  task_per_batch: null  # null = mix all tasks

# ═══════════════════════════════════════════════════════════════
# Statistics and Monitoring
# ═══════════════════════════════════════════════════════════════
statistics:
  collect_stats: true
  log_distribution_every_n_steps: 1000

  # Expected distributions (for validation)
  expected:
    min_samples_per_task: 1000
    min_hub_activation_rate: 0.1
    max_hub_activation_rate: 0.9

# ═══════════════════════════════════════════════════════════════
# Data Augmentation (Optional)
# ═══════════════════════════════════════════════════════════════
augmentation:
  enabled: false
  techniques:
    - name: "synonym_replacement"
      probability: 0.1
      max_replacements: 2
    - name: "random_insertion"
      probability: 0.05
      max_insertions: 1

  # Preserve safety-critical content
  safety_aware: true
  skip_crisis_samples: true

# ═══════════════════════════════════════════════════════════════
# Quality Filters
# ═══════════════════════════════════════════════════════════════
filters:
  # Basic filters
  min_length: 5
  max_length: 2000
  remove_duplicates: true
  remove_empty: true

  # Content filters
  min_tasks_per_sample: 1
  require_text_content: true

  # Hub-based filters
  require_any_hub: false
  filter_by_hub: null  # e.g., ["EMO", "MEM"]
```

**Acceptance Criteria:**

- [ ] All 8 tasks fully configured with labels, weights, metrics
- [ ] Hub routing configuration with hub-to-task mapping
- [ ] Shard loading parameters for streaming
- [ ] Hub-weighted sampling strategy defined
- [ ] Collation configuration for v3 token layout
- [ ] Validation and monitoring settings
- [ ] Safety-aware augmentation options
- [ ] Compatible with `UnifiedFamilyOSDataset` and `StreamingShardDataset`
- [ ] Can be loaded by Hydra/OmegaConf

**Tests:** `tests/v3/test_configs.py::test_familyos_unified_config`

---

### Epic 5.4: Training Scripts

#### Issue 5.4.1: Implement Phase 0.5 Healing Training Script

**Effort:** 6 hours | **Priority:** P0 | **Depends On:** Epic 5.1, Epic 5.2

**File:** `scripts/train_v3_phase0_5.py`

**Purpose:** Training script for Phase 0.5 "Enhanced Healing" that repairs cloned layers (L23-28) while preserving v2 capabilities using the Zipper LR strategy, gradient clipping, and enhanced healing data.

**Phase 0.5 Objectives:**

1. Heal L23-28 (cloned from L15-20) to work coherently
2. Smooth the L22→L23 interface transition
3. Preserve L1-22 frozen capabilities
4. Train hub tokens for routing semantics

**Implementation:**

```python
#!/usr/bin/env python3
"""
Phase 0.5 Enhanced Healing Training Script for ModernBERT v3

This script implements the "healing" phase that repairs the cloned layers
and establishes smooth activation flow across the L22→L23 interface.

Training Strategy:
    - Freeze: L1-18 (Foundation + Core bands)
    - Train: L19-22 (Feeder), L23-28 (Family), Hub tokens
    - LR: Zipper strategy with L23 at maximum plasticity
    - Data: Enhanced healing (SST-2, CoNLL, MNLI, SQuAD, STS-B)

Usage:
    python scripts/train_v3_phase0_5.py \
        --config configs/training/multitask/stage_v3_phase0_5_enhanced.yaml \
        --output-dir outputs/v3_phase0_5 \
        --resume-from checkpoints/v3_initialized

    # With overrides
    python scripts/train_v3_phase0_5.py \
        --config configs/training/multitask/stage_v3_phase0_5_enhanced.yaml \
        --learning-rate 3e-5 \
        --max-steps 3000 \
        --wandb-run-name "phase0_5_experiment_1"
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modeling_studio.models.modernbert_v3 import ModernBERTv3Model
from modeling_studio.trainers.trainer_v3 import (
    ModernBERTv3Trainer,
    TrainingConfig,
    TrainingPhase
)
from modeling_studio.trainers.freezing_v3 import LayerFreezer, LayerBand
from modeling_studio.trainers.zipper_lr_v3 import (
    ZipperLROptimizer,
    create_zipper_optimizer,
    ZIPPER_PRESETS
)
from modeling_studio.trainers.schedulers_v3 import (
    create_scheduler,
    PhaseAwareScheduler
)
from modeling_studio.trainers.gradient_utils_v3 import (
    GradientClipper,
    InterfaceGradientMonitor,
    GradientClipConfig
)
from modeling_studio.trainers.gradient_masking_v3 import (
    setup_hub_token_gradient_masking
)
from modeling_studio.data.collators_v3 import create_v3_collator
from modeling_studio.data.replay_sampler_v3 import create_replay_sampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Phase05Config:
    """Configuration specific to Phase 0.5 healing."""

    # Model
    model_path: str = ""
    model_config: str = "configs/model/encoder/modernbert_v3_ultra.yaml"

    # Training
    max_steps: int = 2500
    warmup_steps: int = 500
    eval_steps: int = 250
    save_steps: int = 500
    logging_steps: int = 50

    # Batch size
    train_batch_size: int = 32
    eval_batch_size: int = 64
    gradient_accumulation_steps: int = 1

    # Optimizer
    base_lr: float = 3e-5
    weight_decay: float = 0.01

    # Zipper LR (layer-specific)
    zipper_preset: str = "phase_0.5_healing"

    # Gradient
    max_grad_norm: float = 1.0
    interface_grad_clip: float = 0.5

    # Data
    healing_data_config: str = "configs/data/multitask/healing_enhanced.yaml"
    replay_ratio: float = 0.15

    # Output
    output_dir: str = "outputs/v3_phase0_5"

    # Logging
    use_wandb: bool = True
    wandb_project: str = "modernbert-v3"
    wandb_run_name: str = "phase0_5_healing"

    # Device
    device: str = "cuda"
    bf16: bool = True

    # Seed
    seed: int = 42


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 0.5 Enhanced Healing Training for ModernBERT v3"
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/multitask/stage_v3_phase0_5_enhanced.yaml",
        help="Path to training config YAML"
    )

    # Model
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to initialized v3 model checkpoint"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Resume training from checkpoint"
    )

    # Training overrides
    parser.add_argument("--max-steps", type=int, help="Maximum training steps")
    parser.add_argument("--warmup-steps", type=int, help="Warmup steps")
    parser.add_argument("--learning-rate", type=float, help="Base learning rate")
    parser.add_argument("--train-batch-size", type=int, help="Training batch size")

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/v3_phase0_5",
        help="Output directory"
    )

    # Logging
    parser.add_argument("--wandb-run-name", type=str, help="W&B run name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B")

    # Device
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--no-bf16", action="store_true", help="Disable bf16")

    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Phase05Config:
    """Load and merge configuration."""
    # Load base config from YAML
    if Path(args.config).exists():
        yaml_config = OmegaConf.load(args.config)
        config = Phase05Config(**OmegaConf.to_container(yaml_config, resolve=True))
    else:
        logger.warning(f"Config file not found: {args.config}, using defaults")
        config = Phase05Config()

    # Apply CLI overrides
    if args.model_path:
        config.model_path = args.model_path
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.warmup_steps:
        config.warmup_steps = args.warmup_steps
    if args.learning_rate:
        config.base_lr = args.learning_rate
    if args.train_batch_size:
        config.train_batch_size = args.train_batch_size
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.wandb_run_name:
        config.wandb_run_name = args.wandb_run_name
    if args.no_wandb:
        config.use_wandb = False
    if args.device:
        config.device = args.device
    if args.no_bf16:
        config.bf16 = False

    return config


# ============================================================================
# Data Loading
# ============================================================================

def load_healing_data(config: Phase05Config) -> tuple:
    """Load enhanced healing datasets."""
    from datasets import load_dataset
    import json

    healing_path = Path(config.healing_data_config).parent / "healing_enhanced.jsonl"

    if healing_path.exists():
        # Load from prepared file
        samples = []
        with open(healing_path, 'r') as f:
            for line in f:
                samples.append(json.loads(line))

        # Split into train/val
        val_size = int(len(samples) * 0.1)
        train_samples = samples[val_size:]
        val_samples = samples[:val_size]

        logger.info(f"Loaded {len(train_samples)} train, {len(val_samples)} val samples")
        return train_samples, val_samples

    else:
        logger.warning(f"Healing data not found at {healing_path}")
        logger.info("Run: python scripts/prepare_healing_data_enhanced.py first")
        raise FileNotFoundError(f"Healing data not found: {healing_path}")


def create_dataloaders(
    train_samples: list,
    val_samples: list,
    tokenizer,
    config: Phase05Config
) -> tuple:
    """Create train and validation dataloaders."""
    from torch.utils.data import Dataset

    class HealingDataset(Dataset):
        def __init__(self, samples, tokenizer, max_length=512):
            self.samples = samples
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]

            # Tokenize
            encoding = self.tokenizer(
                sample["text"],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "task": sample.get("task", "unknown"),
                "labels": sample.get("labels", {})
            }

    train_dataset = HealingDataset(train_samples, tokenizer)
    val_dataset = HealingDataset(val_samples, tokenizer)

    # Create collator
    collator = create_v3_collator(tokenizer, task_type="multitask")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader


# ============================================================================
# Training Loop
# ============================================================================

def train_phase_0_5(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Phase05Config
) -> Dict[str, Any]:
    """
    Execute Phase 0.5 training loop.

    Returns:
        Dict with training results and final metrics
    """
    device = torch.device(config.device)
    model = model.to(device)

    # =========================================
    # 1. Setup Layer Freezing
    # =========================================
    logger.info("Setting up layer freezing for Phase 0.5...")

    freezer = LayerFreezer(model)
    freezer.set_phase(TrainingPhase.PHASE_0_5)

    # Freeze L1-18 (Foundation + Core)
    freezer.freeze_bands([LayerBand.FOUNDATION, LayerBand.CORE])

    # Trainable: L19-22 (Feeder), L23-28 (Family)
    freezer.unfreeze_bands([LayerBand.FEEDER, LayerBand.FAMILY])

    # Freeze embeddings (except hub tokens)
    freezer.freeze_embeddings(except_hub_tokens=True)

    frozen_count, trainable_count = freezer.get_param_counts()
    logger.info(f"Frozen params: {frozen_count:,} | Trainable: {trainable_count:,}")

    # =========================================
    # 2. Setup Hub Token Gradient Masking
    # =========================================
    logger.info("Setting up hub token gradient masking...")

    hub_grad_manager = setup_hub_token_gradient_masking(
        model,
        freeze_original_vocab=True,
        train_hub_tokens=["[EMO]", "[MEM]", "[REL]", "[TASK]"],
        hub_token_grad_scale=1.0
    )

    # =========================================
    # 3. Setup Zipper LR Optimizer
    # =========================================
    logger.info(f"Creating Zipper LR optimizer (preset: {config.zipper_preset})...")

    optimizer = create_zipper_optimizer(
        model,
        preset=config.zipper_preset,
        base_lr=config.base_lr,
        weight_decay=config.weight_decay
    )

    # =========================================
    # 4. Setup Scheduler
    # =========================================
    logger.info("Creating warmup + cosine scheduler...")

    scheduler = create_scheduler(
        optimizer,
        scheduler_type="cosine",
        warmup_steps=config.warmup_steps,
        total_steps=config.max_steps,
        min_lr_ratio=0.01
    )

    # =========================================
    # 5. Setup Gradient Clipping & Monitoring
    # =========================================
    logger.info("Setting up gradient clipping and interface monitoring...")

    grad_config = GradientClipConfig(
        global_max_norm=config.max_grad_norm,
        per_layer_clip=True,
        interface_layer_clip=config.interface_grad_clip,
        nan_check=True
    )
    grad_clipper = GradientClipper(grad_config)

    interface_monitor = InterfaceGradientMonitor(
        interface_layer=23,
        log_every_n_steps=config.logging_steps
    )

    # =========================================
    # 6. Setup Mixed Precision
    # =========================================
    scaler = None
    if config.bf16:
        logger.info("Enabling bf16 mixed precision...")
        # For bf16, we don't need a scaler (only fp16 needs it)

    # =========================================
    # 7. Initialize W&B
    # =========================================
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config={
                "phase": "0.5",
                "max_steps": config.max_steps,
                "warmup_steps": config.warmup_steps,
                "base_lr": config.base_lr,
                "zipper_preset": config.zipper_preset,
                "train_batch_size": config.train_batch_size,
                "frozen_params": frozen_count,
                "trainable_params": trainable_count,
            },
            tags=["phase_0.5", "healing", "v3"]
        )

    # =========================================
    # 8. Training Loop
    # =========================================
    logger.info("=" * 60)
    logger.info("Starting Phase 0.5 Enhanced Healing Training")
    logger.info("=" * 60)

    model.train()
    global_step = 0
    best_val_loss = float("inf")
    train_losses = []

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    train_iter = iter(train_loader)
    pbar = tqdm(total=config.max_steps, desc="Phase 0.5 Training")

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

        # Forward pass
        with torch.cuda.amp.autocast(dtype=torch.bfloat16 if config.bf16 else torch.float32):
            outputs = model(**batch)
            loss = outputs.loss

        # Backward pass
        loss.backward()

        # Gradient clipping
        grad_norm = grad_clipper.clip(model)

        # Interface monitoring
        interface_stats = interface_monitor.log_step(model, global_step)

        # Optimizer step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Logging
        train_losses.append(loss.item())
        global_step += 1
        pbar.update(1)

        if global_step % config.logging_steps == 0:
            avg_loss = sum(train_losses[-100:]) / len(train_losses[-100:])
            current_lr = scheduler.get_last_lr()[0]

            log_dict = {
                "train/loss": avg_loss,
                "train/learning_rate": current_lr,
                "train/grad_norm": grad_norm,
                "train/step": global_step,
            }

            if interface_stats:
                log_dict.update(interface_stats)

            if config.use_wandb:
                wandb.log(log_dict, step=global_step)

            pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{current_lr:.2e}")

        # Evaluation
        if global_step % config.eval_steps == 0:
            val_loss = evaluate(model, val_loader, device, config)

            logger.info(f"Step {global_step}: val_loss={val_loss:.4f}")

            if config.use_wandb:
                wandb.log({"eval/loss": val_loss}, step=global_step)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, scheduler, global_step,
                    output_dir / "best_model"
                )
                logger.info(f"New best model saved (val_loss={val_loss:.4f})")

            model.train()

        # Save checkpoint
        if global_step % config.save_steps == 0:
            save_checkpoint(
                model, optimizer, scheduler, global_step,
                output_dir / f"checkpoint-{global_step}"
            )

    pbar.close()

    # =========================================
    # 9. Final Evaluation & Cleanup
    # =========================================
    logger.info("=" * 60)
    logger.info("Phase 0.5 Training Complete!")
    logger.info("=" * 60)

    final_val_loss = evaluate(model, val_loader, device, config)
    logger.info(f"Final validation loss: {final_val_loss:.4f}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")

    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, global_step,
        output_dir / "final_model"
    )

    if config.use_wandb:
        wandb.finish()

    return {
        "final_val_loss": final_val_loss,
        "best_val_loss": best_val_loss,
        "total_steps": global_step,
        "output_dir": str(output_dir)
    }


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    config: Phase05Config
) -> float:
    """Run evaluation on validation set."""
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

    return total_loss / max(num_batches, 1)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    path: Path
) -> None:
    """Save training checkpoint."""
    path.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "phase": "0.5"
    }, path / "checkpoint.pt")

    # Also save model config
    if hasattr(model, "config"):
        model.config.save_pretrained(path)

    logger.info(f"Checkpoint saved to {path}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main() -> int:
    """Main entry point."""
    args = parse_args()
    config = load_config(args)

    # Set random seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    print("\n" + "=" * 60)
    print("ModernBERT v3 - Phase 0.5 Enhanced Healing")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Model: {config.model_path or 'Will initialize new'}")
    print(f"Output: {config.output_dir}")
    print(f"Max steps: {config.max_steps}")
    print(f"Base LR: {config.base_lr}")
    print(f"Zipper preset: {config.zipper_preset}")
    print()

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    # Add hub tokens if not present
    hub_tokens = ["[EMO]", "[MEM]", "[REL]", "[TASK]"]
    tokenizer.add_special_tokens({"additional_special_tokens": hub_tokens})

    # Load or initialize model
    if config.model_path and Path(config.model_path).exists():
        logger.info(f"Loading model from {config.model_path}")
        model = ModernBERTv3Model.from_pretrained(config.model_path)
    else:
        logger.info("Initializing new v3 model...")
        model = ModernBERTv3Model.from_config(config.model_config)

    # Resize embeddings for hub tokens
    model.resize_token_embeddings(len(tokenizer))

    # Load data
    logger.info("Loading healing data...")
    train_samples, val_samples = load_healing_data(config)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_samples, val_samples, tokenizer, config
    )

    # Run training
    results = train_phase_0_5(model, train_loader, val_loader, config)

    print("\n" + "=" * 60)
    print("Training Results")
    print("=" * 60)
    print(f"Final val loss: {results['final_val_loss']:.4f}")
    print(f"Best val loss: {results['best_val_loss']:.4f}")
    print(f"Total steps: {results['total_steps']}")
    print(f"Output: {results['output_dir']}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Acceptance Criteria:**

- [ ] Layer freezing correctly freezes L1-18, trains L19-28
- [ ] Zipper LR strategy applied with L23 at maximum plasticity
- [ ] Hub token gradient masking preserves hub token gradients
- [ ] Gradient clipping with interface-specific settings
- [ ] Interface gradient monitoring logs L22→L23 gradient ratio
- [ ] Warmup + cosine decay scheduler
- [ ] W&B logging of all metrics
- [ ] Checkpoint saving at intervals
- [ ] Resume from checkpoint works
- [ ] Final model saved with config

**Tests:** `tests/v3/test_training_scripts.py::test_phase_0_5_script`

---

#### Issue 5.4.2: Implement Phase 1 Multi-Task Training Script

**Effort:** 7 hours | **Priority:** P0 | **Depends On:** Issue 5.4.1, Epic 5.3

**File:** `scripts/train_v3_phase1.py`

**Purpose:** Training script for Phase 1 "Multi-Task FamilyOS" that trains on unified FamilyOS data with hub routing, combining all 8 task types while preserving healing from Phase 0.5.

**Phase 1 Objectives:**

1. Train on all 8 FamilyOS task types simultaneously
2. Use hub routing for per-sample task activation
3. Apply hub-weighted loss scaling
4. Integrate replay sampling to prevent forgetting
5. Monitor per-task and per-hub metrics

**Implementation:**

```python
#!/usr/bin/env python3
"""
Phase 1 Multi-Task FamilyOS Training Script for ModernBERT v3

This script implements full multi-task training on unified FamilyOS data
with hub routing for per-sample task activation.

Training Strategy:
    - Freeze: L1-18 (Foundation + Core bands)
    - Train: L19-28, Hub tokens, All task heads
    - LR: Zipper strategy (phase_1_multitask preset)
    - Data: Unified FamilyOS shards with hub_routing
    - Loss: Hub-weighted multi-task loss
    - Replay: 15% healing data for forgetting prevention

Usage:
    python scripts/train_v3_phase1.py \
        --config configs/training/multitask/stage_v3_phase1.yaml \
        --model-path outputs/v3_phase0_5/best_model \
        --output-dir outputs/v3_phase1

    # Resume training
    python scripts/train_v3_phase1.py \
        --resume-from outputs/v3_phase1/checkpoint-5000
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
import json
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modeling_studio.models.modernbert_v3 import ModernBERTv3Model
from modeling_studio.trainers.trainer_v3 import (
    ModernBERTv3Trainer,
    TrainingConfig,
    TrainingPhase
)
from modeling_studio.trainers.freezing_v3 import LayerFreezer, LayerBand
from modeling_studio.trainers.zipper_lr_v3 import (
    create_zipper_optimizer,
    ZIPPER_PRESETS
)
from modeling_studio.trainers.schedulers_v3 import create_scheduler
from modeling_studio.trainers.gradient_utils_v3 import (
    GradientClipper,
    GradientClipConfig
)
from modeling_studio.trainers.gradient_masking_v3 import (
    setup_hub_token_gradient_masking
)
from modeling_studio.training.losses_v3 import (
    HubWeightedMultiTaskLoss,
    HubLossConfig,
    log_task_losses
)
from modeling_studio.data.loaders_v3 import (
    UnifiedFamilyOSDataset,
    TaskType,
    HubRouting
)
from modeling_studio.data.shard_loader_v3 import (
    create_shard_dataset,
    StreamingShardDataset
)
from modeling_studio.data.extractors_v3 import (
    MultiTaskExtractor,
    V3LabelVocabularies
)
from modeling_studio.data.collators_v3 import V3MultiTaskCollator
from modeling_studio.data.replay_sampler_v3 import create_replay_sampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Phase1Config:
    """Configuration specific to Phase 1 multi-task training."""

    # Model
    model_path: str = "outputs/v3_phase0_5/best_model"

    # Training
    max_steps: int = 10000
    warmup_steps: int = 1000
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100

    # Batch size
    train_batch_size: int = 32
    eval_batch_size: int = 64
    gradient_accumulation_steps: int = 2

    # Optimizer
    base_lr: float = 2e-5  # Lower than Phase 0.5
    weight_decay: float = 0.01

    # Zipper LR
    zipper_preset: str = "phase_1_multitask"

    # Gradient
    max_grad_norm: float = 1.0

    # Data
    familyos_data_dir: str = "data/familyos/unified/output"
    familyos_config: str = "configs/data/multitask/familyos_unified.yaml"
    healing_data_path: str = "data/healing/healing_enhanced.jsonl"
    replay_ratio: float = 0.15

    # Hub loss weighting
    hub_active_weight: float = 1.0
    hub_inactive_weight: float = 0.3
    safety_multiplier: float = 2.0

    # Output
    output_dir: str = "outputs/v3_phase1"

    # Logging
    use_wandb: bool = True
    wandb_project: str = "modernbert-v3"
    wandb_run_name: str = "phase1_multitask"

    # Device
    device: str = "cuda"
    bf16: bool = True

    # Seed
    seed: int = 42


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 1 Multi-Task FamilyOS Training for ModernBERT v3"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/multitask/stage_v3_phase1.yaml",
        help="Path to training config YAML"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to Phase 0.5 trained model"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Resume training from checkpoint"
    )

    # Training overrides
    parser.add_argument("--max-steps", type=int, help="Maximum training steps")
    parser.add_argument("--learning-rate", type=float, help="Base learning rate")
    parser.add_argument("--replay-ratio", type=float, help="Healing replay ratio")

    # Output
    parser.add_argument("--output-dir", type=str, help="Output directory")

    # Logging
    parser.add_argument("--wandb-run-name", type=str, help="W&B run name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B")

    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Phase1Config:
    """Load and merge configuration."""
    if Path(args.config).exists():
        yaml_config = OmegaConf.load(args.config)
        config = Phase1Config(**OmegaConf.to_container(yaml_config, resolve=True))
    else:
        config = Phase1Config()

    # Apply CLI overrides
    if args.model_path:
        config.model_path = args.model_path
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.learning_rate:
        config.base_lr = args.learning_rate
    if args.replay_ratio:
        config.replay_ratio = args.replay_ratio
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.wandb_run_name:
        config.wandb_run_name = args.wandb_run_name
    if args.no_wandb:
        config.use_wandb = False

    return config


# ============================================================================
# Data Loading
# ============================================================================

def load_familyos_dataset(
    config: Phase1Config,
    tokenizer
) -> StreamingShardDataset:
    """Load unified FamilyOS dataset from shards."""
    logger.info(f"Loading FamilyOS data from {config.familyos_data_dir}")

    # Create extractor for label encoding
    extractor = MultiTaskExtractor(
        tokenizer=tokenizer,
        vocabs=V3LabelVocabularies(),
        max_seq_length=512
    )

    # Transform function to extract labels
    def transform_sample(sample):
        encoding = tokenizer(
            sample.text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        labels = extractor.extract(sample, encoding)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "hub_routing": sample.hub_routing,
            "labels": labels,
            "sample_id": sample.id
        }

    # Create streaming dataset
    dataset = create_shard_dataset(
        data_dir=config.familyos_data_dir,
        shard_pattern="shard_*.jsonl",
        streaming=True,
        buffered=True,
        transform=transform_sample,
        shuffle_shards=True,
        prefetch_shards=2
    )

    logger.info(f"Created streaming dataset with ~{len(dataset)} samples")
    return dataset


def load_healing_replay_dataset(
    config: Phase1Config,
    tokenizer
) -> list:
    """Load healing data for replay sampling."""
    healing_path = Path(config.healing_data_path)

    if not healing_path.exists():
        logger.warning(f"Healing data not found: {healing_path}")
        return []

    samples = []
    with open(healing_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))

    logger.info(f"Loaded {len(samples)} healing samples for replay")
    return samples


def create_phase1_dataloader(
    familyos_dataset: StreamingShardDataset,
    healing_samples: list,
    tokenizer,
    config: Phase1Config
) -> DataLoader:
    """Create dataloader with replay sampling."""
    # Create collator
    collator = V3MultiTaskCollator(
        tokenizer=tokenizer,
        hub_token_positions={"EMO": 1, "REL": 2, "MEM": 3, "TASK": 4}
    )

    # If we have healing samples, set up replay
    if healing_samples and config.replay_ratio > 0:
        from torch.utils.data import Dataset

        class HealingReplayDataset(Dataset):
            def __init__(self, samples, tokenizer):
                self.samples = samples
                self.tokenizer = tokenizer

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                sample = self.samples[idx]
                encoding = self.tokenizer(
                    sample["text"],
                    max_length=512,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                return {
                    "input_ids": encoding["input_ids"].squeeze(0),
                    "attention_mask": encoding["attention_mask"].squeeze(0),
                    "task": sample.get("task", "healing"),
                    "labels": sample.get("labels", {}),
                    "is_replay": True
                }

        replay_dataset = HealingReplayDataset(healing_samples, tokenizer)

        # Create replay sampler
        combined_dataset, sampler = create_replay_sampler(
            primary_dataset=familyos_dataset,
            replay_dataset=replay_dataset,
            replay_ratio=config.replay_ratio,
            batch_size=config.train_batch_size,
            task_balanced=True
        )

        dataloader = DataLoader(
            combined_dataset,
            batch_size=config.train_batch_size,
            collate_fn=collator,
            num_workers=4,
            pin_memory=True
        )
    else:
        dataloader = DataLoader(
            familyos_dataset,
            batch_size=config.train_batch_size,
            collate_fn=collator,
            num_workers=4,
            pin_memory=True
        )

    return dataloader


# ============================================================================
# Training Loop
# ============================================================================

def train_phase_1(
    model: nn.Module,
    train_loader: DataLoader,
    config: Phase1Config
) -> Dict[str, Any]:
    """
    Execute Phase 1 multi-task training loop.

    Returns:
        Dict with training results and final metrics
    """
    device = torch.device(config.device)
    model = model.to(device)

    # =========================================
    # 1. Setup Layer Freezing (same as Phase 0.5)
    # =========================================
    logger.info("Setting up layer freezing for Phase 1...")

    freezer = LayerFreezer(model)
    freezer.set_phase(TrainingPhase.PHASE_1)

    # Freeze L1-18, train L19-28
    freezer.freeze_bands([LayerBand.FOUNDATION, LayerBand.CORE])
    freezer.unfreeze_bands([LayerBand.FEEDER, LayerBand.FAMILY])
    freezer.freeze_embeddings(except_hub_tokens=True)

    frozen_count, trainable_count = freezer.get_param_counts()
    logger.info(f"Frozen: {frozen_count:,} | Trainable: {trainable_count:,}")

    # =========================================
    # 2. Setup Hub Token Gradient Masking
    # =========================================
    hub_grad_manager = setup_hub_token_gradient_masking(
        model,
        freeze_original_vocab=True,
        train_hub_tokens=["[EMO]", "[MEM]", "[REL]", "[TASK]"]
    )

    # =========================================
    # 3. Setup Optimizer with Zipper LR
    # =========================================
    logger.info(f"Creating optimizer (preset: {config.zipper_preset})...")

    optimizer = create_zipper_optimizer(
        model,
        preset=config.zipper_preset,
        base_lr=config.base_lr,
        weight_decay=config.weight_decay
    )

    # =========================================
    # 4. Setup Scheduler
    # =========================================
    total_steps = config.max_steps
    scheduler = create_scheduler(
        optimizer,
        scheduler_type="cosine",
        warmup_steps=config.warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=0.01
    )

    # =========================================
    # 5. Setup Hub-Weighted Multi-Task Loss
    # =========================================
    logger.info("Setting up hub-weighted multi-task loss...")

    loss_config = HubLossConfig(
        active_weight=config.hub_active_weight,
        inactive_weight=config.hub_inactive_weight,
        safety_multiplier=config.safety_multiplier,
        always_train_safety=True
    )
    loss_fn = HubWeightedMultiTaskLoss(loss_config)

    # =========================================
    # 6. Setup Gradient Clipping
    # =========================================
    grad_config = GradientClipConfig(global_max_norm=config.max_grad_norm)
    grad_clipper = GradientClipper(grad_config)

    # =========================================
    # 7. Initialize W&B
    # =========================================
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config={
                "phase": "1",
                "max_steps": config.max_steps,
                "base_lr": config.base_lr,
                "replay_ratio": config.replay_ratio,
                "hub_active_weight": config.hub_active_weight,
                "hub_inactive_weight": config.hub_inactive_weight,
                "safety_multiplier": config.safety_multiplier,
            },
            tags=["phase_1", "multitask", "familyos", "v3"]
        )

    # =========================================
    # 8. Training Loop
    # =========================================
    logger.info("=" * 60)
    logger.info("Starting Phase 1 Multi-Task FamilyOS Training")
    logger.info("=" * 60)

    model.train()
    global_step = 0
    best_val_loss = float("inf")

    # Metrics tracking
    task_losses: Dict[str, List[float]] = defaultdict(list)
    hub_activation_counts: Dict[str, int] = defaultdict(int)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(total=config.max_steps, desc="Phase 1 Training")
    accumulation_step = 0

    for batch in train_loader:
        if global_step >= config.max_steps:
            break

        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Extract hub routings from batch
        hub_routings = batch.pop("hub_routing", None)

        # Forward pass
        with torch.cuda.amp.autocast(dtype=torch.bfloat16 if config.bf16 else torch.float32):
            outputs = model(**batch)

            # Compute hub-weighted multi-task loss
            if hub_routings is not None:
                total_loss, per_task_losses = loss_fn(
                    task_logits=outputs.task_logits,
                    task_labels=batch.get("labels", {}),
                    hub_routings=hub_routings
                )
            else:
                total_loss = outputs.loss
                per_task_losses = {}

        # Scale loss for gradient accumulation
        scaled_loss = total_loss / config.gradient_accumulation_steps
        scaled_loss.backward()

        accumulation_step += 1

        # Optimizer step after accumulation
        if accumulation_step >= config.gradient_accumulation_steps:
            grad_norm = grad_clipper.clip(model)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            accumulation_step = 0
            global_step += 1
            pbar.update(1)

            # Track per-task losses
            for task_name, task_loss in per_task_losses.items():
                task_losses[task_name].append(task_loss.item())

            # Track hub activations
            if hub_routings is not None:
                for hr in hub_routings:
                    for hub in hr.active_hubs:
                        hub_activation_counts[hub] += 1

            # Logging
            if global_step % config.logging_steps == 0:
                log_dict = {
                    "train/loss": total_loss.item(),
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/grad_norm": grad_norm,
                    "train/step": global_step,
                }

                # Add per-task losses
                task_log = log_task_losses(per_task_losses, prefix="train")
                log_dict.update(task_log)

                # Add hub activation stats
                total_activations = sum(hub_activation_counts.values())
                if total_activations > 0:
                    for hub, count in hub_activation_counts.items():
                        log_dict[f"hub/{hub}_ratio"] = count / total_activations

                if config.use_wandb:
                    wandb.log(log_dict, step=global_step)

                pbar.set_postfix(
                    loss=f"{total_loss.item():.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}"
                )

            # Save checkpoint
            if global_step % config.save_steps == 0:
                save_checkpoint(
                    model, optimizer, scheduler, global_step,
                    output_dir / f"checkpoint-{global_step}",
                    task_losses=dict(task_losses),
                    hub_counts=dict(hub_activation_counts)
                )

    pbar.close()

    # =========================================
    # 9. Final Save & Cleanup
    # =========================================
    logger.info("=" * 60)
    logger.info("Phase 1 Training Complete!")
    logger.info("=" * 60)

    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, global_step,
        output_dir / "final_model",
        task_losses=dict(task_losses),
        hub_counts=dict(hub_activation_counts)
    )

    # Compute final metrics
    final_metrics = compute_final_metrics(task_losses, hub_activation_counts)

    logger.info("Final Task Losses:")
    for task, losses in task_losses.items():
        avg_loss = sum(losses[-100:]) / len(losses[-100:]) if losses else 0
        logger.info(f"  {task}: {avg_loss:.4f}")

    logger.info("Hub Activation Distribution:")
    total = sum(hub_activation_counts.values())
    for hub, count in hub_activation_counts.items():
        logger.info(f"  {hub}: {count} ({100*count/total:.1f}%)")

    if config.use_wandb:
        wandb.log({"final": final_metrics})
        wandb.finish()

    return {
        "total_steps": global_step,
        "task_losses": dict(task_losses),
        "hub_activations": dict(hub_activation_counts),
        "output_dir": str(output_dir)
    }


def compute_final_metrics(
    task_losses: Dict[str, List[float]],
    hub_counts: Dict[str, int]
) -> Dict[str, float]:
    """Compute final summary metrics."""
    metrics = {}

    # Average loss per task
    for task, losses in task_losses.items():
        if losses:
            metrics[f"final_loss/{task}"] = sum(losses[-100:]) / len(losses[-100:])

    # Hub activation ratios
    total = sum(hub_counts.values())
    if total > 0:
        for hub, count in hub_counts.items():
            metrics[f"hub_ratio/{hub}"] = count / total

    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    path: Path,
    **extra_state
) -> None:
    """Save training checkpoint with extra state."""
    path.mkdir(parents=True, exist_ok=True)

    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "phase": "1"
    }
    state.update(extra_state)

    torch.save(state, path / "checkpoint.pt")

    if hasattr(model, "config"):
        model.config.save_pretrained(path)

    logger.info(f"Checkpoint saved to {path}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main() -> int:
    """Main entry point."""
    args = parse_args()
    config = load_config(args)

    # Set random seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    print("\n" + "=" * 60)
    print("ModernBERT v3 - Phase 1 Multi-Task FamilyOS Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Model: {config.model_path}")
    print(f"Data: {config.familyos_data_dir}")
    print(f"Output: {config.output_dir}")
    print(f"Max steps: {config.max_steps}")
    print(f"Replay ratio: {config.replay_ratio}")
    print()

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    hub_tokens = ["[EMO]", "[MEM]", "[REL]", "[TASK]"]
    tokenizer.add_special_tokens({"additional_special_tokens": hub_tokens})

    # Load model from Phase 0.5
    if not Path(config.model_path).exists():
        logger.error(f"Model not found: {config.model_path}")
        logger.error("Run Phase 0.5 training first!")
        return 1

    logger.info(f"Loading model from {config.model_path}")
    model = ModernBERTv3Model.from_pretrained(config.model_path)
    model.resize_token_embeddings(len(tokenizer))

    # Load datasets
    logger.info("Loading FamilyOS dataset...")
    familyos_dataset = load_familyos_dataset(config, tokenizer)

    logger.info("Loading healing replay dataset...")
    healing_samples = load_healing_replay_dataset(config, tokenizer)

    # Create dataloader
    train_loader = create_phase1_dataloader(
        familyos_dataset, healing_samples, tokenizer, config
    )

    # Run training
    results = train_phase_1(model, train_loader, config)

    print("\n" + "=" * 60)
    print("Phase 1 Training Results")
    print("=" * 60)
    print(f"Total steps: {results['total_steps']}")
    print(f"Output: {results['output_dir']}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Acceptance Criteria:**

- [ ] Loads model from Phase 0.5 checkpoint
- [ ] Uses streaming shard dataset for FamilyOS data
- [ ] Hub routing parsed and passed to loss function
- [ ] Hub-weighted multi-task loss applied correctly
- [ ] Per-task loss tracking and logging
- [ ] Hub activation distribution tracked
- [ ] Replay sampling integrates healing data at 15%
- [ ] Gradient accumulation for larger effective batch
- [ ] W&B logging with per-task and per-hub metrics
- [ ] Checkpoint saving with task/hub statistics

**Tests:** `tests/v3/test_training_scripts.py::test_phase_1_script`

---

#### Issue 5.4.3: Implement Multi-Phase Training Orchestrator

**Effort:** 5 hours | **Priority:** P1 | **Depends On:** Issue 5.4.1, 5.4.2

**File:** `scripts/train_v3_orchestrator.py`

**Purpose:** Orchestrate the full v3 training pipeline across all phases (0.5 → 1 → 1.5 → 2) with automatic phase transitions, forgetting gates, and checkpoint management.

**Implementation:**

```python
#!/usr/bin/env python3
"""
Multi-Phase Training Orchestrator for ModernBERT v3

Orchestrates the complete training pipeline:
    Phase 0.5: Enhanced Healing (2,500 steps)
    Phase 1:   Multi-Task FamilyOS (10,000 steps)
    Phase 1.5: Forgetting Evaluation Gate
    Phase 2:   Fine-Tuning (optional, 5,000 steps)

Usage:
    # Run full pipeline
    python scripts/train_v3_orchestrator.py \
        --start-phase 0.5 \
        --end-phase 2 \
        --output-dir outputs/v3_full

    # Resume from Phase 1
    python scripts/train_v3_orchestrator.py \
        --resume-from outputs/v3_full/phase_0.5/final_model \
        --start-phase 1
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Phase(Enum):
    """Training phases."""
    PHASE_0_5 = "0.5"
    PHASE_1 = "1"
    PHASE_1_5 = "1.5"  # Forgetting evaluation gate
    PHASE_2 = "2"


@dataclass
class PhaseConfig:
    """Configuration for a single phase."""
    name: str
    script: str
    config_file: str
    max_steps: int
    depends_on: Optional[str] = None
    skip_if_exists: bool = True
    forgetting_gate: bool = False
    max_forgetting: float = 0.02  # 2% max allowed drop


@dataclass
class OrchestratorConfig:
    """Full orchestrator configuration."""
    output_dir: str = "outputs/v3_full"
    start_phase: str = "0.5"
    end_phase: str = "2"

    # Phase configurations
    phases: Dict[str, PhaseConfig] = field(default_factory=lambda: {
        "0.5": PhaseConfig(
            name="Enhanced Healing",
            script="scripts/train_v3_phase0_5.py",
            config_file="configs/training/multitask/stage_v3_phase0_5_enhanced.yaml",
            max_steps=2500,
            depends_on=None
        ),
        "1": PhaseConfig(
            name="Multi-Task FamilyOS",
            script="scripts/train_v3_phase1.py",
            config_file="configs/training/multitask/stage_v3_phase1.yaml",
            max_steps=10000,
            depends_on="0.5"
        ),
        "1.5": PhaseConfig(
            name="Forgetting Evaluation",
            script="scripts/evaluate_forgetting.py",
            config_file="configs/evaluation/forgetting_gate.yaml",
            max_steps=0,  # Evaluation only
            depends_on="1",
            forgetting_gate=True,
            max_forgetting=0.02
        ),
        "2": PhaseConfig(
            name="Fine-Tuning",
            script="scripts/train_v3_phase2.py",
            config_file="configs/training/multitask/stage_v3_phase2.yaml",
            max_steps=5000,
            depends_on="1.5"
        )
    })

    # Wandb
    use_wandb: bool = True
    wandb_project: str = "modernbert-v3"

    # Device
    device: str = "cuda"
    bf16: bool = True


class TrainingOrchestrator:
    """
    Orchestrates multi-phase training with automatic transitions.
    """

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # State tracking
        self.completed_phases: List[str] = []
        self.phase_results: Dict[str, Dict] = {}

        # Load previous state if exists
        self._load_state()

    def _load_state(self) -> None:
        """Load orchestrator state from disk."""
        state_file = self.output_dir / "orchestrator_state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                state = json.load(f)
                self.completed_phases = state.get("completed_phases", [])
                self.phase_results = state.get("phase_results", {})
            logger.info(f"Loaded state: completed phases = {self.completed_phases}")

    def _save_state(self) -> None:
        """Save orchestrator state to disk."""
        state_file = self.output_dir / "orchestrator_state.json"
        with open(state_file, 'w') as f:
            json.dump({
                "completed_phases": self.completed_phases,
                "phase_results": self.phase_results,
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)

    def run(self) -> Dict[str, Any]:
        """
        Run the full training pipeline.

        Returns:
            Dict with results from all phases
        """
        logger.info("=" * 60)
        logger.info("ModernBERT v3 Multi-Phase Training Orchestrator")
        logger.info("=" * 60)

        # Determine phases to run
        all_phases = ["0.5", "1", "1.5", "2"]
        start_idx = all_phases.index(self.config.start_phase)
        end_idx = all_phases.index(self.config.end_phase)
        phases_to_run = all_phases[start_idx:end_idx + 1]

        logger.info(f"Phases to run: {phases_to_run}")

        # Track current model path
        current_model_path: Optional[str] = None

        # Run each phase
        for phase_id in phases_to_run:
            phase_config = self.config.phases[phase_id]

            # Check if already completed
            if phase_id in self.completed_phases and phase_config.skip_if_exists:
                logger.info(f"Phase {phase_id} already completed, skipping")
                # Get model path from previous run
                phase_output = self.output_dir / f"phase_{phase_id}"
                if (phase_output / "final_model").exists():
                    current_model_path = str(phase_output / "final_model")
                continue

            # Run phase
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting Phase {phase_id}: {phase_config.name}")
            logger.info(f"{'='*60}")

            result = self._run_phase(phase_id, phase_config, current_model_path)

            # Check forgetting gate
            if phase_config.forgetting_gate:
                if not self._check_forgetting_gate(result, phase_config):
                    logger.error(f"Forgetting gate failed at Phase {phase_id}!")
                    logger.error("Consider increasing replay ratio and re-running Phase 1")
                    return {
                        "status": "failed",
                        "failed_phase": phase_id,
                        "reason": "forgetting_gate",
                        "results": self.phase_results
                    }

            # Update state
            self.completed_phases.append(phase_id)
            self.phase_results[phase_id] = result
            self._save_state()

            # Update model path for next phase
            if "output_dir" in result:
                current_model_path = result["output_dir"] + "/final_model"

        logger.info("\n" + "=" * 60)
        logger.info("All phases completed successfully!")
        logger.info("=" * 60)

        return {
            "status": "success",
            "completed_phases": self.completed_phases,
            "results": self.phase_results
        }

    def _run_phase(
        self,
        phase_id: str,
        phase_config: PhaseConfig,
        model_path: Optional[str]
    ) -> Dict[str, Any]:
        """Run a single training phase."""
        phase_output = self.output_dir / f"phase_{phase_id}"
        phase_output.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = [
            sys.executable,
            phase_config.script,
            "--config", phase_config.config_file,
            "--output-dir", str(phase_output),
        ]

        if model_path:
            cmd.extend(["--model-path", model_path])

        if phase_config.max_steps > 0:
            cmd.extend(["--max-steps", str(phase_config.max_steps)])

        if self.config.use_wandb:
            cmd.extend(["--wandb-run-name", f"v3_phase_{phase_id}"])
        else:
            cmd.append("--no-wandb")

        logger.info(f"Running: {' '.join(cmd)}")

        # Execute
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time

        if result.returncode != 0:
            logger.error(f"Phase {phase_id} failed!")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            raise RuntimeError(f"Phase {phase_id} failed with code {result.returncode}")

        logger.info(f"Phase {phase_id} completed in {elapsed/60:.1f} minutes")

        # Load results
        results_file = phase_output / "results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                return json.load(f)

        return {
            "status": "completed",
            "elapsed_seconds": elapsed,
            "output_dir": str(phase_output)
        }

    def _check_forgetting_gate(
        self,
        result: Dict[str, Any],
        phase_config: PhaseConfig
    ) -> bool:
        """Check if forgetting is within acceptable bounds."""
        if "forgetting_metrics" not in result:
            logger.warning("No forgetting metrics found, skipping gate")
            return True

        metrics = result["forgetting_metrics"]
        max_drop = phase_config.max_forgetting

        passed = True
        for task, drop in metrics.items():
            if drop > max_drop:
                logger.error(f"Forgetting exceeded for {task}: {drop:.2%} > {max_drop:.2%}")
                passed = False
            else:
                logger.info(f"Forgetting OK for {task}: {drop:.2%}")

        return passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ModernBERT v3 Multi-Phase Training Orchestrator"
    )
    parser.add_argument("--output-dir", type=str, default="outputs/v3_full")
    parser.add_argument("--start-phase", type=str, default="0.5",
                        choices=["0.5", "1", "1.5", "2"])
    parser.add_argument("--end-phase", type=str, default="2",
                        choices=["0.5", "1", "1.5", "2"])
    parser.add_argument("--resume-from", type=str, help="Model path to resume from")
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = OrchestratorConfig(
        output_dir=args.output_dir,
        start_phase=args.start_phase,
        end_phase=args.end_phase,
        use_wandb=not args.no_wandb
    )

    orchestrator = TrainingOrchestrator(config)
    results = orchestrator.run()

    if results["status"] == "success":
        print("\n✅ Training pipeline completed successfully!")
        return 0
    else:
        print(f"\n❌ Training failed at Phase {results['failed_phase']}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

**Acceptance Criteria:**

- [ ] Orchestrates phases 0.5 → 1 → 1.5 → 2
- [ ] State persistence for resume capability
- [ ] Automatic phase transitions
- [ ] Forgetting gate at Phase 1.5 blocks if >2% drop
- [ ] Subprocess execution with stdout/stderr capture
- [ ] Phase skipping for already-completed phases
- [ ] Model path chaining between phases
- [ ] W&B run naming per phase

**Tests:** `tests/v3/test_training_scripts.py::test_orchestrator`

---

#### Issue 5.4.4: Create Phase-Specific Training Configurations

**Effort:** 3 hours | **Priority:** P1 | **Depends On:** Issues 5.4.1-5.4.3

**Files:**

- `configs/training/multitask/stage_v3_phase1.yaml`
- `configs/training/multitask/stage_v3_phase2.yaml`
- `configs/evaluation/forgetting_gate.yaml`

**Purpose:** Create YAML configurations for Phase 1, Phase 2, and the forgetting evaluation gate.

**Implementation:**

**Phase 1 Config (`stage_v3_phase1.yaml`):**

```yaml
# configs/training/multitask/stage_v3_phase1.yaml
#
# Phase 1 Multi-Task FamilyOS Training Configuration
# Trains on all 8 FamilyOS tasks with hub routing

# Model
model:
  pretrained_path: "outputs/v3_phase0_5/best_model"
  num_layers: 28
  hidden_size: 768
  num_hub_tokens: 4

# Training
training:
  phase: "phase_1"
  phase_name: "Multi-Task FamilyOS"

  max_steps: 10000
  warmup_steps: 1000
  eval_steps: 500
  save_steps: 1000
  logging_steps: 100

  per_device_train_batch_size: 32
  per_device_eval_batch_size: 64
  gradient_accumulation_steps: 2
  effective_batch_size: 64

  fp16: false
  bf16: true
  seed: 42

# Layer Freezing (same as Phase 0.5)
layer_freezing:
  phase: "phase_1"
  frozen_bands:
    - foundation
    - core
  trainable_bands:
    - feeder
    - family
  freeze_embeddings: true
  freeze_hub_tokens: false

# Zipper Learning Rate
learning_rate:
  strategy: "zipper"
  base_lr: 2e-5  # Lower than Phase 0.5
  layers_1_18: 0.0
  layers_19_22: 8e-6
  layer_23: 4e-5
  layers_24_28: 2e-5
  embeddings_lr: 0.0
  hub_tokens_lr: 8e-6
  task_heads_lr: 2e-5

# Scheduler
scheduler:
  type: "cosine"
  warmup_steps: 1000
  min_lr_ratio: 0.01

# Gradient
gradient:
  max_grad_norm: 1.0
  per_layer_clip: false
  nan_check: true

# Data
data:
  familyos_config: "configs/data/multitask/familyos_unified.yaml"
  data_dir: "data/familyos/unified/output"

  replay:
    enabled: true
    healing_data: "data/healing/healing_enhanced.jsonl"
    ratio: 0.15
    task_balanced: true
    dynamic_ratio: true
    loss_threshold: 0.5
    max_ratio: 0.25

# Hub-Weighted Loss
loss:
  hub_active_weight: 1.0
  hub_inactive_weight: 0.3
  safety_multiplier: 2.0
  always_train_safety: true
  label_smoothing: 0.1

  task_weights:
    emotions: 1.0
    sentiment: 1.0
    safety_familyos: 2.0
    intent: 0.8
    ingress: 0.8
    ner_family: 1.0
    temporal: 1.0
    relations: 1.2

# Optimizer
optimizer:
  type: "adamw"
  weight_decay: 0.01
  betas: [0.9, 0.999]

# Checkpointing
checkpointing:
  output_dir: "outputs/v3_phase1"
  save_strategy: "steps"
  save_steps: 1000
  save_total_limit: 3

# Logging
logging:
  use_wandb: true
  wandb_project: "modernbert-v3"
  wandb_run_name: "phase1_multitask"
  wandb_tags: ["phase_1", "multitask", "familyos"]
  logging_steps: 100
```

**Phase 2 Config (`stage_v3_phase2.yaml`):**

```yaml
# configs/training/multitask/stage_v3_phase2.yaml
#
# Phase 2 Fine-Tuning Configuration
# Optional fine-tuning phase with lower LR and focused training

# Model
model:
  pretrained_path: "outputs/v3_phase1/final_model"

# Training
training:
  phase: "phase_2"
  phase_name: "Fine-Tuning"

  max_steps: 5000
  warmup_steps: 500
  eval_steps: 250
  save_steps: 500
  logging_steps: 50

  per_device_train_batch_size: 32
  gradient_accumulation_steps: 1

  bf16: true
  seed: 42

# Layer Freezing - More aggressive
layer_freezing:
  phase: "phase_2"
  frozen_bands:
    - foundation
    - core
    - feeder  # Also freeze feeder in Phase 2
  trainable_bands:
    - family  # Only family band trainable
  freeze_embeddings: true
  freeze_hub_tokens: true  # Hub tokens frozen in Phase 2

# Learning Rate - Very low
learning_rate:
  strategy: "zipper"
  base_lr: 5e-6
  layers_1_22: 0.0
  layer_23: 1e-5
  layers_24_28: 5e-6
  task_heads_lr: 5e-6

# Scheduler
scheduler:
  type: "cosine"
  warmup_steps: 500
  min_lr_ratio: 0.1

# Data - Focus on hard samples
data:
  familyos_config: "configs/data/multitask/familyos_unified.yaml"
  sample_strategy: "hard_mining"
  focus_on_errors: true

  replay:
    enabled: true
    ratio: 0.20  # Higher replay for stability
    task_balanced: true

# Loss
loss:
  hub_active_weight: 1.0
  hub_inactive_weight: 0.5  # Higher inactive weight
  safety_multiplier: 2.0
  label_smoothing: 0.05

# Checkpointing
checkpointing:
  output_dir: "outputs/v3_phase2"
  save_total_limit: 2

# Logging
logging:
  use_wandb: true
  wandb_run_name: "phase2_finetuning"
  wandb_tags: ["phase_2", "finetuning"]
```

**Forgetting Gate Config (`forgetting_gate.yaml`):**

```yaml
# configs/evaluation/forgetting_gate.yaml
#
# Phase 1.5 Forgetting Evaluation Gate Configuration
# Evaluates model on healing benchmarks to detect capability loss

# Evaluation
evaluation:
  phase: "phase_1.5"
  phase_name: "Forgetting Evaluation"

  # Model to evaluate
  model_path: "outputs/v3_phase1/final_model"

  # Baseline for comparison
  baseline_path: "outputs/v3_phase0_5/best_model"

# Benchmarks
benchmarks:
  sst2:
    dataset: "glue"
    subset: "sst2"
    split: "validation"
    metric: "accuracy"
    baseline_score: null  # Will be computed from Phase 0.5
    max_drop: 0.02

  conll:
    dataset: "conll2003"
    split: "validation"
    metric: "f1"
    baseline_score: null
    max_drop: 0.02

  mnli:
    dataset: "glue"
    subset: "mnli"
    split: "validation_matched"
    metric: "accuracy"
    baseline_score: null
    max_drop: 0.02

  squad:
    dataset: "squad"
    split: "validation"
    metric: "f1"
    baseline_score: null
    max_drop: 0.03  # Slightly more lenient for QA

# Gate Configuration
gate:
  # All benchmarks must pass
  require_all: true

  # Maximum allowed drop
  max_drop: 0.02

  # Actions on failure
  on_failure:
    - action: "log_warning"
    - action: "suggest_remediation"
      remediation: "increase_replay_ratio"
      suggested_value: 0.25
    - action: "block_next_phase"

# Remediation Options
remediation:
  increase_replay:
    current_ratio: 0.15
    suggested_ratio: 0.25
    max_ratio: 0.40

  add_benchmark_samples:
    sample_count: 1000
    per_benchmark: true

# Output
output:
  results_file: "outputs/v3_phase1/forgetting_results.json"
  generate_report: true
  report_format: "markdown"
```

**Acceptance Criteria:**

- [ ] Phase 1 config inherits from Phase 0.5 style
- [ ] Phase 1 has lower base LR (2e-5 vs 3e-5)
- [ ] Phase 1 includes replay with dynamic ratio
- [ ] Phase 2 freezes feeder band additionally
- [ ] Phase 2 has very low LR (5e-6)
- [ ] Forgetting gate defines max 2% drop per benchmark
- [ ] Forgetting gate suggests remediation on failure
- [ ] All configs loadable by Hydra/OmegaConf

**Tests:** `tests/v3/test_configs.py::test_phase_training_configs`

---

## Milestone 6: Evaluation & Validation

**Goal:** Validate v3 model quality and forgetting prevention

### Epic 6.1: Forgetting Gates

**Goal:** Prevent catastrophic forgetting of Stage A knowledge during v3 training
**Total Estimated Hours:** 18 hours

#### Issue 6.1.1: Implement Phase 1.5 Forgetting Evaluation

**Priority:** 🔴 Critical
**Estimated Hours:** 8 hours
**Dependencies:** Epic 5.4 (Training Scripts complete)

**Description:**
Implement the Phase 1.5 evaluation checkpoint that runs after Phase 1 multi-task training to detect any catastrophic forgetting of Stage A benchmark performance. This is a NO-TRAINING phase that serves as a quality gate before production deployment.

**File:** `src/modeling_studio/evaluation/forgetting_eval_v3.py`

```python
"""
Phase 1.5 Forgetting Evaluation for ModernBERT v3

This module implements the forgetting detection pipeline specifically
designed for v3's Phase 1.5 checkpoint validation.

v3-Specific Considerations:
- Evaluates AFTER Phase 1 multi-task training (not Stage B like v2)
- Must account for hub token injection (positions 1-4)
- Compares against v2 baseline OR Phase 0 initialized model
- Integrates with orchestrator for automatic phase gating

Forgetting Gates (v3.3 Spec):
- CoNLL-2003 (NER): ≤ 2% F1 drop
- SST-2 (Sentiment): ≤ 2% Accuracy drop
- MNLI (NLI): ≤ 2% Accuracy drop
- FamilyOS Emotions: ≤ 3% Macro F1 drop (optional, family-specific)

Usage:
    from modeling_studio.evaluation.forgetting_eval_v3 import (
        Phase15ForgettingEvaluator,
        ForgettingGateResult,
        run_forgetting_gates,
    )

    # After Phase 1 training
    evaluator = Phase15ForgettingEvaluator(
        baseline_checkpoint="checkpoints/modernbert-v3-phase0",
        phase1_checkpoint="checkpoints/modernbert-v3-phase1",
        hub_tokenizer=hub_tokenizer,
    )

    result = evaluator.run_all_gates()
    if not result.all_passed:
        print(result.get_remediation_actions())
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

import torch
import torch.nn as nn
from tqdm import tqdm

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


# ============================================================================
# V3 FORGETTING THRESHOLDS (from enhanced_design_v3.md Section 9.3)
# ============================================================================

V3_FORGETTING_THRESHOLDS = {
    "ner_general": {
        "benchmark": "conll2003",
        "metric": "f1",
        "max_drop": 0.02,  # ≤ 2% F1 drop
        "priority": "critical",
        "remediation": "increase_replay_ratio",
    },
    "sentiment": {
        "benchmark": "sst2",
        "metric": "accuracy",
        "max_drop": 0.02,  # ≤ 2% Accuracy drop
        "priority": "critical",
        "remediation": "increase_replay_ratio",
    },
    "nli": {
        "benchmark": "mnli",
        "metric": "accuracy",
        "max_drop": 0.02,  # ≤ 2% Accuracy drop
        "priority": "critical",
        "remediation": "increase_replay_ratio",
    },
    "emotions": {
        "benchmark": "goemotions",
        "metric": "macro_f1",
        "max_drop": 0.03,  # ≤ 3% Macro F1 drop (more lenient for family-specific)
        "priority": "high",
        "remediation": "reduce_lora_r",
    },
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class BenchmarkScore:
    """Score for a single benchmark evaluation."""

    task: str
    benchmark: str
    metric_name: str
    score: float
    num_samples: int
    inference_time_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "benchmark": self.benchmark,
            "metric_name": self.metric_name,
            "score": self.score,
            "num_samples": self.num_samples,
            "inference_time_ms": self.inference_time_ms,
            "details": self.details,
        }


@dataclass
class ForgettingGateResult:
    """Result of a single forgetting gate check."""

    task: str
    benchmark: str
    metric_name: str
    baseline_score: float
    phase1_score: float
    drop: float  # Positive = regression
    max_allowed_drop: float
    passed: bool
    priority: str
    remediation_action: str

    def __repr__(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return (
            f"ForgettingGate({self.task}/{self.benchmark}): "
            f"{self.baseline_score:.4f} → {self.phase1_score:.4f} "
            f"(drop: {self.drop:+.4f}, max: {self.max_allowed_drop:.4f}) {status}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "benchmark": self.benchmark,
            "metric_name": self.metric_name,
            "baseline_score": self.baseline_score,
            "phase1_score": self.phase1_score,
            "drop": self.drop,
            "max_allowed_drop": self.max_allowed_drop,
            "passed": self.passed,
            "priority": self.priority,
            "remediation_action": self.remediation_action,
        }


@dataclass
class Phase15EvaluationReport:
    """Complete Phase 1.5 forgetting evaluation report."""

    gate_results: list[ForgettingGateResult]
    all_passed: bool
    critical_failures: list[str]
    high_priority_failures: list[str]
    recommended_actions: list[str]
    baseline_checkpoint: str
    phase1_checkpoint: str
    evaluation_timestamp: str = ""
    total_evaluation_time_s: float = 0.0

    def __post_init__(self):
        from datetime import datetime
        if not self.evaluation_timestamp:
            self.evaluation_timestamp = datetime.now().isoformat()

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 70,
            "PHASE 1.5 FORGETTING EVALUATION REPORT",
            "=" * 70,
            f"Baseline: {self.baseline_checkpoint}",
            f"Phase 1:  {self.phase1_checkpoint}",
            f"Time:     {self.evaluation_timestamp}",
            "",
            "GATE RESULTS:",
            "-" * 70,
        ]

        for gate in self.gate_results:
            status = "✅" if gate.passed else "❌"
            lines.append(
                f"  {status} {gate.task:15} | "
                f"{gate.metric_name:10} | "
                f"{gate.baseline_score:.4f} → {gate.phase1_score:.4f} | "
                f"drop: {gate.drop:+.4f} (max: {gate.max_allowed_drop:.4f})"
            )

        lines.append("-" * 70)

        if self.all_passed:
            lines.append("✅ ALL FORGETTING GATES PASSED - Ready for production")
        else:
            lines.append("❌ FORGETTING DETECTED - Remediation required")

            if self.critical_failures:
                lines.append(f"\n🔴 Critical Failures: {', '.join(self.critical_failures)}")
            if self.high_priority_failures:
                lines.append(f"🟡 High Priority Failures: {', '.join(self.high_priority_failures)}")

            lines.append("\nRecommended Actions:")
            for i, action in enumerate(self.recommended_actions, 1):
                lines.append(f"  {i}. {action}")

        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_results": [g.to_dict() for g in self.gate_results],
            "all_passed": self.all_passed,
            "critical_failures": self.critical_failures,
            "high_priority_failures": self.high_priority_failures,
            "recommended_actions": self.recommended_actions,
            "baseline_checkpoint": self.baseline_checkpoint,
            "phase1_checkpoint": self.phase1_checkpoint,
            "evaluation_timestamp": self.evaluation_timestamp,
            "total_evaluation_time_s": self.total_evaluation_time_s,
        }

    def save(self, path: str | Path) -> None:
        """Save report to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved Phase 1.5 evaluation report to {path}")


# ============================================================================
# PHASE 1.5 EVALUATOR
# ============================================================================

class Phase15ForgettingEvaluator:
    """
    Evaluator for Phase 1.5 forgetting detection in v3 training pipeline.

    This evaluator is specifically designed for ModernBERT v3's training
    phases and handles:
    - Hub token injection (tokens at positions 1-4)
    - Comparison between Phase 0 (initialized) and Phase 1 (trained)
    - Integration with v3 training orchestrator
    - Automatic remediation recommendations

    Args:
        baseline_checkpoint: Path to baseline model (Phase 0 or v2).
        phase1_checkpoint: Path to Phase 1 trained model.
        hub_tokenizer: v3 tokenizer with hub tokens.
        thresholds: Custom thresholds (default: V3_FORGETTING_THRESHOLDS).
        device: Evaluation device.
        batch_size: Batch size for evaluation.
    """

    # Hub token positions in v3 input sequence
    HUB_TOKEN_POSITIONS = [1, 2, 3, 4]  # [EMO], [MEM], [REL], [TASK]

    def __init__(
        self,
        baseline_checkpoint: str | Path,
        phase1_checkpoint: str | Path,
        hub_tokenizer: PreTrainedTokenizer | None = None,
        thresholds: dict[str, dict] | None = None,
        device: str | None = None,
        batch_size: int = 32,
    ):
        self.baseline_checkpoint = Path(baseline_checkpoint)
        self.phase1_checkpoint = Path(phase1_checkpoint)
        self.hub_tokenizer = hub_tokenizer
        self.thresholds = thresholds or V3_FORGETTING_THRESHOLDS
        self.batch_size = batch_size

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self._baseline_model: PreTrainedModel | None = None
        self._phase1_model: PreTrainedModel | None = None
        self._datasets_cache: dict[str, Dataset] = {}

    def load_models(self) -> None:
        """Load baseline and Phase 1 models."""
        from transformers import AutoModel, AutoTokenizer

        logger.info(f"Loading baseline model from {self.baseline_checkpoint}")
        self._baseline_model = AutoModel.from_pretrained(str(self.baseline_checkpoint))
        self._baseline_model.to(self.device)
        self._baseline_model.eval()

        logger.info(f"Loading Phase 1 model from {self.phase1_checkpoint}")
        self._phase1_model = AutoModel.from_pretrained(str(self.phase1_checkpoint))
        self._phase1_model.to(self.device)
        self._phase1_model.eval()

        if self.hub_tokenizer is None:
            self.hub_tokenizer = AutoTokenizer.from_pretrained(str(self.phase1_checkpoint))

    def _load_benchmark_dataset(self, benchmark: str) -> Dataset | None:
        """Load benchmark dataset with caching."""
        if benchmark in self._datasets_cache:
            return self._datasets_cache[benchmark]

        try:
            from datasets import load_dataset

            if benchmark == "conll2003":
                dataset = load_dataset("conll2003", split="test")
            elif benchmark == "sst2":
                dataset = load_dataset("glue", "sst2", split="validation")
            elif benchmark == "mnli":
                dataset = load_dataset("glue", "mnli", split="validation_matched")
            elif benchmark == "goemotions":
                dataset = load_dataset("go_emotions", "simplified", split="test")
            else:
                logger.warning(f"Unknown benchmark: {benchmark}")
                return None

            self._datasets_cache[benchmark] = dataset
            return dataset

        except Exception as e:
            logger.error(f"Failed to load benchmark {benchmark}: {e}")
            return None

    def _evaluate_on_benchmark(
        self,
        model: PreTrainedModel,
        task: str,
        benchmark: str,
        metric_name: str,
    ) -> BenchmarkScore:
        """Evaluate a model on a single benchmark."""
        import time
        from sklearn.metrics import accuracy_score, f1_score

        dataset = self._load_benchmark_dataset(benchmark)
        if dataset is None:
            return BenchmarkScore(
                task=task,
                benchmark=benchmark,
                metric_name=metric_name,
                score=0.0,
                num_samples=0,
                details={"error": "dataset_unavailable"},
            )

        # Prepare evaluation
        all_predictions = []
        all_labels = []
        start_time = time.time()

        # Process in batches
        for i in tqdm(range(0, len(dataset), self.batch_size), desc=f"Eval {task}"):
            batch = dataset[i:i + self.batch_size]

            # Get texts based on benchmark format
            if benchmark == "conll2003":
                texts = [" ".join(tokens) for tokens in batch["tokens"]]
                labels = batch["ner_tags"]
            elif benchmark == "sst2":
                texts = batch["sentence"]
                labels = batch["label"]
            elif benchmark == "mnli":
                texts = [
                    f"{p} [SEP] {h}"
                    for p, h in zip(batch["premise"], batch["hypothesis"])
                ]
                labels = batch["label"]
            elif benchmark == "goemotions":
                texts = batch["text"]
                labels = batch["labels"]
            else:
                continue

            # Tokenize with hub tokens
            if self.hub_tokenizer:
                encoded = self.hub_tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )

                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)

                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                # Get predictions
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs.last_hidden_state[:, 0, :]  # CLS

                preds = logits.argmax(dim=-1).cpu().numpy()

                if benchmark == "conll2003":
                    # Flatten token predictions
                    for pred_seq, label_seq in zip(preds, labels):
                        if isinstance(label_seq, list):
                            all_predictions.extend(pred_seq[:len(label_seq)].tolist())
                            all_labels.extend(label_seq)
                elif benchmark == "goemotions":
                    # Multi-label
                    for pred, label_list in zip(preds, labels):
                        all_predictions.append(pred)
                        all_labels.append(label_list[0] if label_list else 0)
                else:
                    all_predictions.extend(preds.tolist())
                    all_labels.extend(labels)

        elapsed_time = time.time() - start_time

        # Compute metrics
        if len(all_labels) == 0:
            score = 0.0
        elif metric_name == "accuracy":
            score = accuracy_score(all_labels, all_predictions)
        elif metric_name == "f1":
            score = f1_score(all_labels, all_predictions, average="weighted", zero_division=0)
        elif metric_name == "macro_f1":
            score = f1_score(all_labels, all_predictions, average="macro", zero_division=0)
        else:
            score = 0.0

        return BenchmarkScore(
            task=task,
            benchmark=benchmark,
            metric_name=metric_name,
            score=float(score),
            num_samples=len(all_labels),
            inference_time_ms=(elapsed_time * 1000) / max(len(all_labels), 1),
        )

    def evaluate_gate(self, task: str) -> ForgettingGateResult:
        """Evaluate a single forgetting gate."""
        if task not in self.thresholds:
            raise ValueError(f"Unknown task: {task}")

        config = self.thresholds[task]
        benchmark = config["benchmark"]
        metric_name = config["metric"]
        max_drop = config["max_drop"]
        priority = config.get("priority", "high")
        remediation = config.get("remediation", "increase_replay_ratio")

        # Evaluate baseline
        logger.info(f"Evaluating baseline on {task}/{benchmark}...")
        baseline_score = self._evaluate_on_benchmark(
            self._baseline_model, task, benchmark, metric_name
        )

        # Evaluate Phase 1
        logger.info(f"Evaluating Phase 1 on {task}/{benchmark}...")
        phase1_score = self._evaluate_on_benchmark(
            self._phase1_model, task, benchmark, metric_name
        )

        # Calculate drop
        drop = baseline_score.score - phase1_score.score
        passed = drop <= max_drop

        return ForgettingGateResult(
            task=task,
            benchmark=benchmark,
            metric_name=metric_name,
            baseline_score=baseline_score.score,
            phase1_score=phase1_score.score,
            drop=drop,
            max_allowed_drop=max_drop,
            passed=passed,
            priority=priority,
            remediation_action=remediation,
        )

    def run_all_gates(
        self,
        tasks: list[str] | None = None,
    ) -> Phase15EvaluationReport:
        """
        Run all forgetting gates and generate report.

        Args:
            tasks: Specific tasks to evaluate. Default: all configured tasks.

        Returns:
            Phase15EvaluationReport with all results and recommendations.
        """
        import time

        if self._baseline_model is None or self._phase1_model is None:
            self.load_models()

        if tasks is None:
            tasks = list(self.thresholds.keys())

        start_time = time.time()
        gate_results = []
        critical_failures = []
        high_failures = []

        for task in tasks:
            result = self.evaluate_gate(task)
            gate_results.append(result)

            if not result.passed:
                if result.priority == "critical":
                    critical_failures.append(task)
                else:
                    high_failures.append(task)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            gate_results, critical_failures, high_failures
        )

        elapsed = time.time() - start_time

        return Phase15EvaluationReport(
            gate_results=gate_results,
            all_passed=len(critical_failures) == 0 and len(high_failures) == 0,
            critical_failures=critical_failures,
            high_priority_failures=high_failures,
            recommended_actions=recommendations,
            baseline_checkpoint=str(self.baseline_checkpoint),
            phase1_checkpoint=str(self.phase1_checkpoint),
            total_evaluation_time_s=elapsed,
        )

    def _generate_recommendations(
        self,
        results: list[ForgettingGateResult],
        critical: list[str],
        high: list[str],
    ) -> list[str]:
        """Generate remediation recommendations based on failures."""
        recommendations = []

        if not critical and not high:
            return ["No remediation needed - all gates passed"]

        # Collect unique remediation actions
        actions = set()
        for result in results:
            if not result.passed:
                actions.add(result.remediation_action)

        # Map actions to specific recommendations
        action_map = {
            "increase_replay_ratio": (
                "Increase Stage A replay ratio from 15% to 25% in Phase 1 training"
            ),
            "reduce_lora_r": (
                "Reduce LoRA rank from r=16 to r=8 to limit parameter updates"
            ),
            "freeze_more_layers": (
                "Extend frozen layers from L1-18 to L1-20 to preserve more v2 knowledge"
            ),
            "reduce_lr": (
                "Reduce learning rate for trainable layers by 50%"
            ),
        }

        for action in actions:
            if action in action_map:
                recommendations.append(action_map[action])

        # Add general recommendations for critical failures
        if critical:
            recommendations.append(
                f"CRITICAL: Re-run Phase 1 with adjusted hyperparameters before deployment"
            )
            recommendations.append(
                "Consider adding task-specific replay samples for: " + ", ".join(critical)
            )

        return recommendations


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def run_forgetting_gates(
    baseline_checkpoint: str | Path,
    phase1_checkpoint: str | Path,
    output_path: str | Path | None = None,
    device: str | None = None,
) -> Phase15EvaluationReport:
    """
    Run Phase 1.5 forgetting evaluation (convenience function).

    Args:
        baseline_checkpoint: Path to baseline (Phase 0 or v2) checkpoint.
        phase1_checkpoint: Path to Phase 1 trained checkpoint.
        output_path: Optional path to save JSON report.
        device: Evaluation device.

    Returns:
        Phase15EvaluationReport with all gate results.

    Example:
        >>> report = run_forgetting_gates(
        ...     baseline_checkpoint="checkpoints/modernbert-v3-phase0",
        ...     phase1_checkpoint="checkpoints/modernbert-v3-phase1",
        ...     output_path="outputs/phase15_eval.json",
        ... )
        >>> print(report.summary())
        >>> assert report.all_passed, "Forgetting detected!"
    """
    evaluator = Phase15ForgettingEvaluator(
        baseline_checkpoint=baseline_checkpoint,
        phase1_checkpoint=phase1_checkpoint,
        device=device,
    )

    report = evaluator.run_all_gates()

    if output_path:
        report.save(output_path)

    print(report.summary())
    return report


def check_gate_passed(
    task: str,
    baseline_score: float,
    phase1_score: float,
    thresholds: dict[str, dict] | None = None,
) -> bool:
    """
    Quick check if a single gate passes.

    Args:
        task: Task name (e.g., "ner_general", "sentiment").
        baseline_score: Score from baseline model.
        phase1_score: Score from Phase 1 model.
        thresholds: Custom thresholds.

    Returns:
        True if gate passes (drop <= max_allowed).
    """
    thresholds = thresholds or V3_FORGETTING_THRESHOLDS
    if task not in thresholds:
        raise ValueError(f"Unknown task: {task}")

    max_drop = thresholds[task]["max_drop"]
    drop = baseline_score - phase1_score
    return drop <= max_drop
```

**Acceptance Criteria:**

- [ ] `Phase15ForgettingEvaluator` class loads baseline and Phase 1 checkpoints
- [ ] Evaluates on all 4 benchmark datasets (CoNLL, SST-2, MNLI, GoEmotions)
- [ ] Correctly computes performance drop (baseline - phase1)
- [ ] Generates `Phase15EvaluationReport` with pass/fail status per gate
- [ ] Hub tokenizer properly injects hub tokens for v3 model evaluation
- [ ] `run_forgetting_gates()` convenience function works end-to-end
- [ ] Report saved to JSON with all metrics and recommendations

**Tests:** `tests/v3/test_forgetting_eval_v3.py::test_phase15_evaluator`

---

#### Issue 6.1.2: Define Forgetting Thresholds (≤2% drop)

**Priority:** 🔴 Critical
**Estimated Hours:** 5 hours
**Dependencies:** Issue 6.1.1

**Description:**
Define and implement the forgetting threshold configuration system that specifies maximum allowed performance drops for each benchmark task. These thresholds are the core quality gates that determine whether Phase 1 training succeeded without catastrophic forgetting.

**File:** `src/modeling_studio/evaluation/forgetting_thresholds_v3.py`

```python
"""
Forgetting Threshold Configuration for ModernBERT v3

This module defines the forgetting thresholds that serve as quality gates
after Phase 1 multi-task training. Thresholds are derived from:

1. enhanced_design_v3.md Section 9.3 (Forgetting Gates)
2. v2 evaluation experience (what drops are acceptable)
3. Task criticality for FamilyOS use cases

Threshold Philosophy:
- Core NLP tasks (NER, NLI, Sentiment): Strict 2% max drop
- Family-specific tasks: Slightly relaxed 3% max drop
- Safety tasks: Zero tolerance (separate validation in Epic 6.3)

Usage:
    from modeling_studio.evaluation.forgetting_thresholds_v3 import (
        ForgettingThresholdConfig,
        get_default_thresholds,
        validate_threshold,
        ThresholdRegistry,
    )

    # Get default thresholds
    thresholds = get_default_thresholds()

    # Check a specific threshold
    passed = validate_threshold("ner_general", baseline=0.91, current=0.90)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Literal


class ThresholdPriority(Enum):
    """Priority levels for forgetting gates."""

    CRITICAL = "critical"    # Must pass - blocks deployment
    HIGH = "high"            # Should pass - triggers warning
    MEDIUM = "medium"        # Nice to pass - logged only

    def __str__(self) -> str:
        return self.value


class RemediationAction(Enum):
    """Possible remediation actions when threshold is exceeded."""

    INCREASE_REPLAY_RATIO = "increase_replay_ratio"
    REDUCE_LORA_R = "reduce_lora_r"
    FREEZE_MORE_LAYERS = "freeze_more_layers"
    REDUCE_LEARNING_RATE = "reduce_lr"
    ADD_TASK_REPLAY = "add_task_specific_replay"
    INCREASE_EPOCHS = "increase_training_epochs"

    def __str__(self) -> str:
        return self.value


@dataclass
class ForgettingThreshold:
    """Configuration for a single forgetting threshold."""

    task: str
    benchmark: str
    metric: str
    max_drop: float
    priority: ThresholdPriority = ThresholdPriority.HIGH
    remediation: RemediationAction = RemediationAction.INCREASE_REPLAY_RATIO
    description: str = ""
    min_baseline_score: float = 0.0  # Minimum expected baseline score

    def validate(self, baseline_score: float, current_score: float) -> bool:
        """Check if threshold is met."""
        drop = baseline_score - current_score
        return drop <= self.max_drop

    def get_drop(self, baseline_score: float, current_score: float) -> float:
        """Calculate performance drop (positive = regression)."""
        return baseline_score - current_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "benchmark": self.benchmark,
            "metric": self.metric,
            "max_drop": self.max_drop,
            "priority": str(self.priority),
            "remediation": str(self.remediation),
            "description": self.description,
            "min_baseline_score": self.min_baseline_score,
        }


@dataclass
class ForgettingThresholdConfig:
    """Complete configuration for all forgetting thresholds."""

    thresholds: dict[str, ForgettingThreshold] = field(default_factory=dict)
    version: str = "v3.3"
    strict_mode: bool = True  # If True, any critical failure blocks deployment

    def add_threshold(self, threshold: ForgettingThreshold) -> None:
        """Add a threshold to the configuration."""
        self.thresholds[threshold.task] = threshold

    def get_threshold(self, task: str) -> ForgettingThreshold | None:
        """Get threshold for a specific task."""
        return self.thresholds.get(task)

    def validate_all(
        self,
        scores: dict[str, tuple[float, float]],  # task -> (baseline, current)
    ) -> dict[str, bool]:
        """Validate all thresholds at once."""
        results = {}
        for task, (baseline, current) in scores.items():
            if task in self.thresholds:
                results[task] = self.thresholds[task].validate(baseline, current)
        return results

    def get_critical_tasks(self) -> list[str]:
        """Get list of tasks with critical priority."""
        return [
            task for task, thresh in self.thresholds.items()
            if thresh.priority == ThresholdPriority.CRITICAL
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "strict_mode": self.strict_mode,
            "thresholds": {
                task: thresh.to_dict()
                for task, thresh in self.thresholds.items()
            },
        }


# ============================================================================
# DEFAULT THRESHOLDS (from enhanced_design_v3.md Section 9.3)
# ============================================================================

def get_default_thresholds() -> ForgettingThresholdConfig:
    """
    Get default forgetting thresholds for v3 training.

    These thresholds are derived from enhanced_design_v3.md Section 9.3:

    | Benchmark       | Max Allowed Drop | Priority  | Remediation           |
    |-----------------|------------------|-----------|----------------------|
    | CoNLL-2003      | ≤ 2% F1          | Critical  | Increase replay ratio |
    | SST-2           | ≤ 2% Accuracy    | Critical  | Increase replay ratio |
    | MNLI            | ≤ 2% Accuracy    | Critical  | Increase replay ratio |
    | FamilyOS Emo    | ≤ 3% Macro F1    | High      | Reduce LoRA r         |

    Returns:
        ForgettingThresholdConfig with all default thresholds.
    """
    config = ForgettingThresholdConfig(version="v3.3", strict_mode=True)

    # Critical: Core NLP benchmarks (Stage A knowledge)
    config.add_threshold(ForgettingThreshold(
        task="ner_general",
        benchmark="conll2003",
        metric="f1",
        max_drop=0.02,  # ≤ 2% F1 drop
        priority=ThresholdPriority.CRITICAL,
        remediation=RemediationAction.INCREASE_REPLAY_RATIO,
        description="Named Entity Recognition on CoNLL-2003 test set",
        min_baseline_score=0.91,  # v2 target: 91% F1
    ))

    config.add_threshold(ForgettingThreshold(
        task="sentiment",
        benchmark="sst2",
        metric="accuracy",
        max_drop=0.02,  # ≤ 2% Accuracy drop
        priority=ThresholdPriority.CRITICAL,
        remediation=RemediationAction.INCREASE_REPLAY_RATIO,
        description="Binary sentiment classification on SST-2 validation",
        min_baseline_score=0.94,  # v2 target: 94% Accuracy
    ))

    config.add_threshold(ForgettingThreshold(
        task="nli",
        benchmark="mnli",
        metric="accuracy",
        max_drop=0.02,  # ≤ 2% Accuracy drop
        priority=ThresholdPriority.CRITICAL,
        remediation=RemediationAction.INCREASE_REPLAY_RATIO,
        description="Natural Language Inference on MNLI matched validation",
        min_baseline_score=0.88,  # v2 target: 88% Accuracy
    ))

    # High: Family-specific tasks (more tolerance for domain adaptation)
    config.add_threshold(ForgettingThreshold(
        task="emotions",
        benchmark="goemotions",
        metric="macro_f1",
        max_drop=0.03,  # ≤ 3% Macro F1 drop (more lenient)
        priority=ThresholdPriority.HIGH,
        remediation=RemediationAction.REDUCE_LORA_R,
        description="Multi-label emotion classification on GoEmotions",
        min_baseline_score=0.78,  # v2 target: 78% Macro F1
    ))

    # Medium: Embedding tasks (can regress slightly during domain adaptation)
    config.add_threshold(ForgettingThreshold(
        task="embedding",
        benchmark="stsb",
        metric="spearman",
        max_drop=0.03,  # ≤ 3% Spearman correlation drop
        priority=ThresholdPriority.MEDIUM,
        remediation=RemediationAction.FREEZE_MORE_LAYERS,
        description="Semantic similarity on STS-B validation",
        min_baseline_score=0.85,  # v2 target: 85% Spearman
    ))

    return config


# ============================================================================
# THRESHOLD REGISTRY (for custom configurations)
# ============================================================================

class ThresholdRegistry:
    """
    Registry for managing multiple threshold configurations.

    Allows switching between different threshold profiles:
    - "default": Standard v3.3 thresholds
    - "strict": Tighter thresholds for production deployment
    - "relaxed": Looser thresholds for experimental training
    - "custom": User-defined thresholds
    """

    _configs: dict[str, ForgettingThresholdConfig] = {}
    _default_profile: str = "default"

    @classmethod
    def register(cls, name: str, config: ForgettingThresholdConfig) -> None:
        """Register a threshold configuration."""
        cls._configs[name] = config

    @classmethod
    def get(cls, name: str | None = None) -> ForgettingThresholdConfig:
        """Get a threshold configuration by name."""
        name = name or cls._default_profile
        if name not in cls._configs:
            if name == "default":
                cls._configs["default"] = get_default_thresholds()
            else:
                raise KeyError(f"Unknown threshold profile: {name}")
        return cls._configs[name]

    @classmethod
    def set_default(cls, name: str) -> None:
        """Set the default profile name."""
        cls._default_profile = name

    @classmethod
    def list_profiles(cls) -> list[str]:
        """List all registered profiles."""
        return list(cls._configs.keys())


# Pre-register default profiles
def _init_registry():
    """Initialize the threshold registry with default profiles."""
    # Default profile
    ThresholdRegistry.register("default", get_default_thresholds())

    # Strict profile (tighter thresholds for production)
    strict_config = get_default_thresholds()
    strict_config.version = "v3.3-strict"
    for thresh in strict_config.thresholds.values():
        thresh.max_drop = thresh.max_drop * 0.5  # Halve all thresholds
    ThresholdRegistry.register("strict", strict_config)

    # Relaxed profile (for experimental training)
    relaxed_config = get_default_thresholds()
    relaxed_config.version = "v3.3-relaxed"
    relaxed_config.strict_mode = False
    for thresh in relaxed_config.thresholds.values():
        thresh.max_drop = thresh.max_drop * 2.0  # Double all thresholds
    ThresholdRegistry.register("relaxed", relaxed_config)


_init_registry()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def validate_threshold(
    task: str,
    baseline: float,
    current: float,
    profile: str = "default",
) -> bool:
    """
    Quick validation of a single threshold.

    Args:
        task: Task name (e.g., "ner_general").
        baseline: Baseline score (e.g., from Phase 0 or v2).
        current: Current score (e.g., from Phase 1).
        profile: Threshold profile to use.

    Returns:
        True if threshold is met (drop <= max_allowed).

    Example:
        >>> validate_threshold("sentiment", baseline=0.94, current=0.93)
        True  # 1% drop is within 2% threshold
        >>> validate_threshold("sentiment", baseline=0.94, current=0.90)
        False  # 4% drop exceeds 2% threshold
    """
    config = ThresholdRegistry.get(profile)
    threshold = config.get_threshold(task)
    if threshold is None:
        raise ValueError(f"Unknown task: {task}")
    return threshold.validate(baseline, current)


def get_remediation_for_task(task: str, profile: str = "default") -> str:
    """Get the recommended remediation action for a task."""
    config = ThresholdRegistry.get(profile)
    threshold = config.get_threshold(task)
    if threshold is None:
        return "unknown_task"
    return str(threshold.remediation)


def format_threshold_table(profile: str = "default") -> str:
    """Format thresholds as a markdown table for documentation."""
    config = ThresholdRegistry.get(profile)

    lines = [
        "| Task | Benchmark | Metric | Max Drop | Priority | Remediation |",
        "|------|-----------|--------|----------|----------|-------------|",
    ]

    for task, thresh in config.thresholds.items():
        lines.append(
            f"| {thresh.task} | {thresh.benchmark} | {thresh.metric} | "
            f"{thresh.max_drop:.0%} | {thresh.priority} | {thresh.remediation} |"
        )

    return "\n".join(lines)
```

**YAML Configuration Support:**

**File:** `configs/evaluation/forgetting_thresholds.yaml`

```yaml
# Forgetting Threshold Configuration for ModernBERT v3
# Based on enhanced_design_v3.md Section 9.3

version: "v3.3"
strict_mode: true  # Any critical failure blocks deployment

thresholds:
  # Critical: Core NLP benchmarks (Stage A knowledge preservation)
  ner_general:
    benchmark: conll2003
    metric: f1
    max_drop: 0.02  # ≤ 2% F1 drop
    priority: critical
    remediation: increase_replay_ratio
    min_baseline_score: 0.91
    description: "Named Entity Recognition on CoNLL-2003"

  sentiment:
    benchmark: sst2
    metric: accuracy
    max_drop: 0.02  # ≤ 2% Accuracy drop
    priority: critical
    remediation: increase_replay_ratio
    min_baseline_score: 0.94
    description: "Binary sentiment on SST-2"

  nli:
    benchmark: mnli
    metric: accuracy
    max_drop: 0.02  # ≤ 2% Accuracy drop
    priority: critical
    remediation: increase_replay_ratio
    min_baseline_score: 0.88
    description: "NLI on MNLI matched"

  # High: Family-specific (more tolerance for domain adaptation)
  emotions:
    benchmark: goemotions
    metric: macro_f1
    max_drop: 0.03  # ≤ 3% Macro F1 drop
    priority: high
    remediation: reduce_lora_r
    min_baseline_score: 0.78
    description: "Multi-label emotions on GoEmotions"

  # Medium: Embedding coherence
  embedding:
    benchmark: stsb
    metric: spearman
    max_drop: 0.03  # ≤ 3% Spearman drop
    priority: medium
    remediation: freeze_more_layers
    min_baseline_score: 0.85
    description: "Semantic similarity on STS-B"

# Remediation action descriptions
remediation_actions:
  increase_replay_ratio:
    description: "Increase Stage A replay from 15% to 25% in Phase 1"
    config_change: "training.phase_1.data_mix.stage_a_replay_ratio: 0.25"

  reduce_lora_r:
    description: "Reduce LoRA rank from r=16 to r=8"
    config_change: "training.phase_1.lora.r: 8"

  freeze_more_layers:
    description: "Freeze layers 1-20 instead of 1-18"
    config_change: "training.phase_1.frozen_layers: [1..20]"

  reduce_lr:
    description: "Reduce learning rates by 50%"
    config_change: "training.phase_1.learning_rate.*: *= 0.5"
```

**Acceptance Criteria:**

- [ ] `ForgettingThreshold` dataclass with task, benchmark, metric, max_drop fields
- [ ] `ForgettingThresholdConfig` aggregates all thresholds with validation methods
- [ ] Default thresholds match enhanced_design_v3.md Section 9.3 exactly
- [ ] `ThresholdRegistry` supports multiple profiles (default, strict, relaxed)
- [ ] YAML configuration file loadable by OmegaConf
- [ ] `validate_threshold()` convenience function for quick checks
- [ ] `format_threshold_table()` generates markdown documentation

**Tests:** `tests/v3/test_forgetting_thresholds.py::test_threshold_validation`

---

#### Issue 6.1.3: Implement Automatic Remediation Triggers

**Priority:** 🟡 High
**Estimated Hours:** 5 hours
**Dependencies:** Issues 6.1.1, 6.1.2

**Description:**
Implement the automatic remediation system that triggers corrective actions when forgetting thresholds are exceeded. This system integrates with the training orchestrator to automatically adjust hyperparameters and re-run Phase 1 training when needed.

**File:** `src/modeling_studio/evaluation/forgetting_remediation_v3.py`

```python
"""
Automatic Remediation for Forgetting Detection in ModernBERT v3

This module implements automatic remediation actions that are triggered
when Phase 1.5 forgetting evaluation detects threshold violations.

Remediation Philosophy:
1. Start with least invasive changes (increase replay ratio)
2. Escalate to more aggressive changes if needed (freeze layers)
3. Log all remediation attempts for debugging
4. Maximum 3 remediation cycles before manual intervention required

Remediation Actions (in order of preference):
1. increase_replay_ratio: 15% → 25% → 35%
2. reduce_lora_r: 16 → 8 → 4
3. freeze_more_layers: L1-18 → L1-20 → L1-22
4. reduce_lr: 50% reduction per cycle

Usage:
    from modeling_studio.evaluation.forgetting_remediation_v3 import (
        RemediationEngine,
        RemediationPlan,
        apply_remediation,
    )

    # After failed Phase 1.5 evaluation
    engine = RemediationEngine(
        failed_gates=["ner_general", "sentiment"],
        current_config=training_config,
    )

    plan = engine.generate_plan()
    new_config = engine.apply_plan(plan)
    # Re-run Phase 1 with new_config
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from .forgetting_eval_v3 import Phase15EvaluationReport

logger = logging.getLogger(__name__)


# ============================================================================
# REMEDIATION ACTIONS
# ============================================================================

@dataclass
class RemediationStep:
    """A single remediation step to apply."""

    action: str
    parameter: str
    old_value: Any
    new_value: Any
    rationale: str
    priority: int = 0  # Lower = apply first

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "parameter": self.parameter,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "rationale": self.rationale,
            "priority": self.priority,
        }


@dataclass
class RemediationPlan:
    """Complete remediation plan with all steps."""

    steps: list[RemediationStep]
    failed_gates: list[str]
    cycle_number: int
    timestamp: str = ""
    estimated_impact: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        # Sort steps by priority
        self.steps = sorted(self.steps, key=lambda s: s.priority)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"REMEDIATION PLAN (Cycle {self.cycle_number})",
            f"Failed Gates: {', '.join(self.failed_gates)}",
            f"Timestamp: {self.timestamp}",
            "",
            "Steps to Apply:",
        ]
        for i, step in enumerate(self.steps, 1):
            lines.append(
                f"  {i}. {step.action}: {step.parameter} "
                f"({step.old_value} → {step.new_value})"
            )
            lines.append(f"     Rationale: {step.rationale}")

        if self.estimated_impact:
            lines.append(f"\nEstimated Impact: {self.estimated_impact}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "failed_gates": self.failed_gates,
            "cycle_number": self.cycle_number,
            "timestamp": self.timestamp,
            "estimated_impact": self.estimated_impact,
        }

    def save(self, path: str | Path) -> None:
        """Save plan to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ============================================================================
# REMEDIATION STRATEGIES
# ============================================================================

class RemediationStrategy:
    """Base class for remediation strategies."""

    def __init__(self, max_cycles: int = 3):
        self.max_cycles = max_cycles

    def get_escalation_values(self, cycle: int) -> Any:
        """Get the value to apply at each escalation cycle."""
        raise NotImplementedError

    def generate_step(
        self,
        cycle: int,
        current_value: Any,
        failed_task: str,
    ) -> RemediationStep | None:
        """Generate a remediation step for this strategy."""
        raise NotImplementedError


class IncreaseReplayRatioStrategy(RemediationStrategy):
    """Increase Stage A replay ratio in Phase 1."""

    ESCALATION = [0.25, 0.35, 0.45]  # 25% → 35% → 45%

    def get_escalation_values(self, cycle: int) -> float:
        if cycle > len(self.ESCALATION):
            return self.ESCALATION[-1]
        return self.ESCALATION[cycle - 1]

    def generate_step(
        self,
        cycle: int,
        current_value: float,
        failed_task: str,
    ) -> RemediationStep:
        new_value = self.get_escalation_values(cycle)
        return RemediationStep(
            action="increase_replay_ratio",
            parameter="training.phase_1.data_mix.stage_a_replay_ratio",
            old_value=current_value,
            new_value=new_value,
            rationale=f"Increase Stage A data to prevent forgetting on {failed_task}",
            priority=1,
        )


class ReduceLoRARankStrategy(RemediationStrategy):
    """Reduce LoRA rank to limit parameter updates."""

    ESCALATION = [8, 4, 2]  # r=8 → r=4 → r=2

    def get_escalation_values(self, cycle: int) -> int:
        if cycle > len(self.ESCALATION):
            return self.ESCALATION[-1]
        return self.ESCALATION[cycle - 1]

    def generate_step(
        self,
        cycle: int,
        current_value: int,
        failed_task: str,
    ) -> RemediationStep:
        new_value = self.get_escalation_values(cycle)
        return RemediationStep(
            action="reduce_lora_r",
            parameter="training.phase_1.lora.r",
            old_value=current_value,
            new_value=new_value,
            rationale=f"Reduce LoRA rank to limit parameter drift for {failed_task}",
            priority=2,
        )


class FreezeMoreLayersStrategy(RemediationStrategy):
    """Freeze additional encoder layers."""

    ESCALATION = [20, 21, 22]  # Freeze up to L20 → L21 → L22

    def get_escalation_values(self, cycle: int) -> int:
        if cycle > len(self.ESCALATION):
            return self.ESCALATION[-1]
        return self.ESCALATION[cycle - 1]

    def generate_step(
        self,
        cycle: int,
        current_value: int,
        failed_task: str,
    ) -> RemediationStep:
        new_value = self.get_escalation_values(cycle)
        return RemediationStep(
            action="freeze_more_layers",
            parameter="training.phase_1.frozen_layers_end",
            old_value=current_value,
            new_value=new_value,
            rationale=f"Freeze more layers to preserve v2 knowledge for {failed_task}",
            priority=3,
        )


class ReduceLearningRateStrategy(RemediationStrategy):
    """Reduce learning rates by percentage."""

    REDUCTION_FACTOR = 0.5  # Halve LR each cycle

    def get_escalation_values(self, cycle: int) -> float:
        return self.REDUCTION_FACTOR ** cycle

    def generate_step(
        self,
        cycle: int,
        current_value: float,
        failed_task: str,
    ) -> RemediationStep:
        factor = self.get_escalation_values(cycle)
        new_value = current_value * factor
        return RemediationStep(
            action="reduce_lr",
            parameter="training.phase_1.learning_rate.layers_19_22",
            old_value=current_value,
            new_value=new_value,
            rationale=f"Reduce LR to slow adaptation and preserve {failed_task}",
            priority=4,
        )


# ============================================================================
# REMEDIATION ENGINE
# ============================================================================

# Map task remediation preferences to strategies
TASK_STRATEGY_MAP = {
    "ner_general": ["increase_replay_ratio", "freeze_more_layers"],
    "sentiment": ["increase_replay_ratio", "reduce_lr"],
    "nli": ["increase_replay_ratio", "freeze_more_layers"],
    "emotions": ["reduce_lora_r", "increase_replay_ratio"],
    "embedding": ["freeze_more_layers", "reduce_lr"],
}


class RemediationEngine:
    """
    Engine for generating and applying remediation plans.

    This engine:
    1. Analyzes failed forgetting gates
    2. Determines appropriate remediation strategies
    3. Generates a remediation plan with specific config changes
    4. Tracks remediation cycles to prevent infinite loops

    Args:
        failed_gates: List of task names that failed forgetting gates.
        current_config: Current training configuration (OmegaConf).
        max_cycles: Maximum remediation cycles before manual intervention.
        history_path: Path to save remediation history.
    """

    STRATEGIES = {
        "increase_replay_ratio": IncreaseReplayRatioStrategy(),
        "reduce_lora_r": ReduceLoRARankStrategy(),
        "freeze_more_layers": FreezeMoreLayersStrategy(),
        "reduce_lr": ReduceLearningRateStrategy(),
    }

    def __init__(
        self,
        failed_gates: list[str],
        current_config: DictConfig | dict[str, Any],
        max_cycles: int = 3,
        history_path: str | Path | None = None,
    ):
        self.failed_gates = failed_gates
        self.current_config = (
            OmegaConf.create(current_config)
            if isinstance(current_config, dict)
            else current_config
        )
        self.max_cycles = max_cycles
        self.history_path = Path(history_path) if history_path else None

        # Track cycle number from history
        self._cycle_number = self._get_current_cycle()

    def _get_current_cycle(self) -> int:
        """Get current remediation cycle from history."""
        if self.history_path and self.history_path.exists():
            with open(self.history_path) as f:
                history = json.load(f)
            return len(history.get("cycles", [])) + 1
        return 1

    def can_remediate(self) -> bool:
        """Check if remediation is possible (not exceeded max cycles)."""
        return self._cycle_number <= self.max_cycles

    def generate_plan(self) -> RemediationPlan:
        """
        Generate a remediation plan based on failed gates.

        Returns:
            RemediationPlan with all steps to apply.

        Raises:
            RuntimeError: If max remediation cycles exceeded.
        """
        if not self.can_remediate():
            raise RuntimeError(
                f"Maximum remediation cycles ({self.max_cycles}) exceeded. "
                "Manual intervention required."
            )

        steps = []
        applied_strategies = set()

        for task in self.failed_gates:
            preferred_strategies = TASK_STRATEGY_MAP.get(
                task, ["increase_replay_ratio"]
            )

            for strategy_name in preferred_strategies:
                if strategy_name in applied_strategies:
                    continue  # Don't apply same strategy twice

                strategy = self.STRATEGIES.get(strategy_name)
                if strategy is None:
                    continue

                # Get current value from config
                current_value = self._get_config_value(strategy_name)

                # Generate step
                step = strategy.generate_step(
                    cycle=self._cycle_number,
                    current_value=current_value,
                    failed_task=task,
                )

                if step:
                    steps.append(step)
                    applied_strategies.add(strategy_name)

        return RemediationPlan(
            steps=steps,
            failed_gates=self.failed_gates,
            cycle_number=self._cycle_number,
            estimated_impact=self._estimate_impact(steps),
        )

    def _get_config_value(self, strategy_name: str) -> Any:
        """Get current config value for a strategy."""
        defaults = {
            "increase_replay_ratio": 0.15,
            "reduce_lora_r": 16,
            "freeze_more_layers": 18,
            "reduce_lr": 2e-5,
        }

        try:
            if strategy_name == "increase_replay_ratio":
                return OmegaConf.select(
                    self.current_config,
                    "training.phase_1.data_mix.stage_a_replay_ratio",
                    default=defaults[strategy_name],
                )
            elif strategy_name == "reduce_lora_r":
                return OmegaConf.select(
                    self.current_config,
                    "training.phase_1.lora.r",
                    default=defaults[strategy_name],
                )
            elif strategy_name == "freeze_more_layers":
                frozen = OmegaConf.select(
                    self.current_config,
                    "training.phase_1.frozen_layers",
                    default=list(range(1, 19)),
                )
                return max(frozen) if frozen else defaults[strategy_name]
            elif strategy_name == "reduce_lr":
                return OmegaConf.select(
                    self.current_config,
                    "training.phase_1.learning_rate.layers_19_22",
                    default=defaults[strategy_name],
                )
        except Exception:
            pass

        return defaults.get(strategy_name, 0)

    def _estimate_impact(self, steps: list[RemediationStep]) -> str:
        """Estimate the impact of remediation steps."""
        impacts = []

        for step in steps:
            if step.action == "increase_replay_ratio":
                impacts.append(
                    f"Training data will include {step.new_value:.0%} Stage A samples"
                )
            elif step.action == "reduce_lora_r":
                impacts.append(
                    f"LoRA will have {step.new_value} rank (fewer parameters to update)"
                )
            elif step.action == "freeze_more_layers":
                impacts.append(
                    f"Layers 1-{step.new_value} will be frozen (more v2 preservation)"
                )
            elif step.action == "reduce_lr":
                impacts.append(
                    f"Learning rate reduced to {step.new_value:.2e} (slower adaptation)"
                )

        return "; ".join(impacts) if impacts else "Minimal expected impact"

    def apply_plan(
        self,
        plan: RemediationPlan,
        config: DictConfig | None = None,
    ) -> DictConfig:
        """
        Apply remediation plan to configuration.

        Args:
            plan: RemediationPlan to apply.
            config: Configuration to modify (default: self.current_config).

        Returns:
            Modified configuration with remediation applied.
        """
        config = config if config is not None else copy.deepcopy(self.current_config)

        for step in plan.steps:
            try:
                OmegaConf.update(config, step.parameter, step.new_value)
                logger.info(
                    f"Applied remediation: {step.parameter} = {step.new_value}"
                )
            except Exception as e:
                logger.warning(f"Failed to apply {step.action}: {e}")

        # Save history
        if self.history_path:
            self._save_history(plan)

        return config

    def _save_history(self, plan: RemediationPlan) -> None:
        """Save remediation history."""
        history = {"cycles": []}

        if self.history_path.exists():
            with open(self.history_path) as f:
                history = json.load(f)

        history["cycles"].append(plan.to_dict())

        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_path, "w") as f:
            json.dump(history, f, indent=2)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def apply_remediation(
    evaluation_report: Phase15EvaluationReport,
    current_config: DictConfig | dict[str, Any],
    output_config_path: str | Path | None = None,
    history_path: str | Path | None = None,
) -> tuple[DictConfig, RemediationPlan]:
    """
    Apply automatic remediation based on evaluation report.

    Args:
        evaluation_report: Phase15EvaluationReport with failed gates.
        current_config: Current training configuration.
        output_config_path: Path to save modified config.
        history_path: Path to save remediation history.

    Returns:
        Tuple of (modified_config, remediation_plan).

    Example:
        >>> report = run_forgetting_gates(baseline, phase1)
        >>> if not report.all_passed:
        ...     new_config, plan = apply_remediation(report, config)
        ...     print(plan.summary())
        ...     # Re-run Phase 1 with new_config
    """
    if evaluation_report.all_passed:
        raise ValueError("No remediation needed - all gates passed")

    failed_gates = (
        evaluation_report.critical_failures +
        evaluation_report.high_priority_failures
    )

    engine = RemediationEngine(
        failed_gates=failed_gates,
        current_config=current_config,
        history_path=history_path,
    )

    plan = engine.generate_plan()
    modified_config = engine.apply_plan(plan)

    if output_config_path:
        output_path = Path(output_config_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(modified_config, output_path)
        logger.info(f"Saved remediated config to {output_path}")

    return modified_config, plan


def get_remediation_summary(
    failed_gates: list[str],
    cycle: int = 1,
) -> str:
    """
    Get a summary of recommended remediation actions.

    Args:
        failed_gates: List of failed task names.
        cycle: Current remediation cycle.

    Returns:
        Human-readable summary of recommended actions.
    """
    lines = [f"Remediation Recommendations (Cycle {cycle}):", ""]

    seen_actions = set()
    for task in failed_gates:
        strategies = TASK_STRATEGY_MAP.get(task, ["increase_replay_ratio"])
        for strategy_name in strategies:
            if strategy_name not in seen_actions:
                strategy = RemediationEngine.STRATEGIES.get(strategy_name)
                if strategy:
                    value = strategy.get_escalation_values(cycle)
                    lines.append(f"  • {strategy_name}: Apply value {value} for {task}")
                    seen_actions.add(strategy_name)

    return "\n".join(lines)
```

**Acceptance Criteria:**

- [ ] `RemediationEngine` class generates plans based on failed gates
- [ ] Supports 4 remediation strategies: replay ratio, LoRA r, freeze layers, reduce LR
- [ ] Escalation values increase with each remediation cycle
- [ ] Maximum 3 cycles before requiring manual intervention
- [ ] `apply_remediation()` modifies OmegaConf config in-place
- [ ] History tracking prevents infinite remediation loops
- [ ] Integration with Phase15EvaluationReport from Issue 6.1.1
- [ ] Config changes persist to YAML file for next training run

**Tests:** `tests/v3/test_forgetting_remediation.py::test_remediation_engine`

---

### Epic 6.2: Quality Benchmarks

**Goal:** Validate v3 model quality across all capabilities and measure performance impact
**Total Estimated Hours:** 16 hours

#### Issue 6.2.1: Implement v3 Benchmark Suite

**Priority:** 🔴 Critical | **Hours:** 5 | **Depends On:** Epic 5.4

**File:** `src/modeling_studio/evaluation/benchmarks_v3.py`

**Description:**
Create unified benchmark suite for all 12 v3 capabilities with hub-token-aware evaluation.

```python
"""v3 Benchmark Suite with Hub Token Support."""

from dataclasses import dataclass
from typing import Dict, List, Any
import torch

@dataclass
class V3BenchmarkResult:
    """Result for a single benchmark."""
    task: str
    metric: str
    score: float
    num_samples: int
    hub_used: str  # Which hub token was active
    inference_ms: float

class V3BenchmarkSuite:
    """
    Unified benchmark suite for ModernBERT v3.

    Benchmarks:
        - NER: CoNLL-2003, OntoNotes (ner_general, ner_family)
        - Classification: SST-2, GoEmotions (sentiment, emotions)
        - NLI: MNLI, SNLI (nli)
        - Safety: BeaverTails, FamilyOS Safety (safety_familyos)
        - Embedding: STS-B, MTEB subset (embedding)
        - FamilyOS: Intent, Ingress, Relations, Temporal
    """

    BENCHMARKS = {
        "ner_general": {"dataset": "conll2003", "metric": "f1", "hub": "MEM"},
        "ner_family": {"dataset": "familyos_ner", "metric": "f1", "hub": "MEM"},
        "sentiment": {"dataset": "sst2", "metric": "accuracy", "hub": "EMO"},
        "emotions": {"dataset": "goemotions", "metric": "macro_f1", "hub": "EMO"},
        "safety_familyos": {"dataset": "familyos_safety", "metric": "crisis_recall", "hub": "EMO"},
        "nli": {"dataset": "mnli", "metric": "accuracy", "hub": "REL"},
        "embedding": {"dataset": "stsb", "metric": "spearman", "hub": "MEM"},
        "intent": {"dataset": "familyos_intent", "metric": "accuracy", "hub": "TASK"},
        "ingress": {"dataset": "familyos_ingress", "metric": "accuracy", "hub": "TASK"},
        "relations": {"dataset": "familyos_relations", "metric": "f1", "hub": "REL"},
        "temporal": {"dataset": "familyos_temporal", "metric": "f1", "hub": "MEM"},
    }

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def run_all(self, tasks: List[str] = None) -> Dict[str, V3BenchmarkResult]:
        """Run all benchmarks and return results."""
        tasks = tasks or list(self.BENCHMARKS.keys())
        results = {}
        for task in tasks:
            results[task] = self._run_benchmark(task)
        return results

    def _run_benchmark(self, task: str) -> V3BenchmarkResult:
        """Run single benchmark with hub token routing."""
        config = self.BENCHMARKS[task]
        dataset = self._load_dataset(config["dataset"])
        # Evaluate with hub token active
        score, num_samples, inference_ms = self._evaluate(dataset, task, config["hub"])
        return V3BenchmarkResult(
            task=task, metric=config["metric"], score=score,
            num_samples=num_samples, hub_used=config["hub"], inference_ms=inference_ms
        )

def run_v3_benchmarks(model_path: str, output_path: str = None) -> Dict[str, Any]:
    """Convenience function to run full benchmark suite."""
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    suite = V3BenchmarkSuite(model, tokenizer)
    results = suite.run_all()
    if output_path:
        import json
        with open(output_path, "w") as f:
            json.dump({k: v.__dict__ for k, v in results.items()}, f, indent=2)
    return results
```

**Acceptance Criteria:**

- [ ] Benchmarks all 12 capabilities with correct metrics
- [ ] Hub token routing tracked per benchmark
- [ ] Inference latency measured per task
- [ ] JSON report generation with all scores

**Tests:** `tests/v3/test_benchmarks_v3.py::test_benchmark_suite`

---

#### Issue 6.2.2: Compare v2 vs v3 Performance

**Priority:** 🔴 Critical | **Hours:** 4 | **Depends On:** Issue 6.2.1

**File:** `src/modeling_studio/evaluation/v2_v3_comparison.py`

**Description:**
Side-by-side comparison of v2 and v3 model performance with improvement analysis.

```python
"""v2 vs v3 Performance Comparison."""

from dataclasses import dataclass
from typing import Dict, Tuple
import json

@dataclass
class ComparisonResult:
    """Comparison between v2 and v3 for a task."""
    task: str
    v2_score: float
    v3_score: float
    delta: float  # v3 - v2 (positive = improvement)
    delta_pct: float  # Percentage improvement
    meets_target: bool  # Meets v3 target from enhanced_design_v3.md

# Target improvements from enhanced_design_v3.md Section 9.1
V3_TARGETS = {
    "ner_general": {"v2": 0.91, "v3": 0.93, "metric": "f1"},
    "ner_family": {"v2": 0.88, "v3": 0.91, "metric": "f1"},
    "sentiment": {"v2": 0.94, "v3": 0.96, "metric": "accuracy"},
    "emotions": {"v2": 0.78, "v3": 0.82, "metric": "macro_f1"},
    "safety_familyos": {"v2": 0.98, "v3": 0.99, "metric": "crisis_recall"},
    "nli": {"v2": 0.88, "v3": 0.91, "metric": "accuracy"},
    "embedding": {"v2": 0.85, "v3": 0.90, "metric": "recall@10"},
    "intent": {"v2": 0.90, "v3": 0.93, "metric": "accuracy"},
    "ingress": {"v2": 0.92, "v3": 0.95, "metric": "accuracy"},
    "relations": {"v2": 0.82, "v3": 0.87, "metric": "f1"},
    "temporal": {"v2": 0.85, "v3": 0.89, "metric": "f1"},
}

class V2V3Comparator:
    """Compare v2 and v3 model performance."""

    def __init__(self, v2_results: Dict, v3_results: Dict):
        self.v2_results = v2_results
        self.v3_results = v3_results

    def compare_all(self) -> Dict[str, ComparisonResult]:
        """Compare all tasks."""
        results = {}
        for task in V3_TARGETS:
            v2_score = self.v2_results.get(task, {}).get("score", V3_TARGETS[task]["v2"])
            v3_score = self.v3_results.get(task, {}).get("score", 0.0)
            delta = v3_score - v2_score
            delta_pct = (delta / v2_score) * 100 if v2_score > 0 else 0
            results[task] = ComparisonResult(
                task=task, v2_score=v2_score, v3_score=v3_score,
                delta=delta, delta_pct=delta_pct,
                meets_target=v3_score >= V3_TARGETS[task]["v3"]
            )
        return results

    def summary_table(self) -> str:
        """Generate markdown comparison table."""
        results = self.compare_all()
        lines = [
            "| Task | v2 | v3 | Δ | Target | Status |",
            "|------|----|----|---|--------|--------|",
        ]
        for task, r in results.items():
            status = "✅" if r.meets_target else "❌"
            lines.append(
                f"| {task} | {r.v2_score:.2%} | {r.v3_score:.2%} | "
                f"{r.delta:+.2%} | {V3_TARGETS[task]['v3']:.2%} | {status} |"
            )
        return "\n".join(lines)

def compare_models(v2_path: str, v3_path: str, output_path: str = None) -> str:
    """Run comparison and generate report."""
    from .benchmarks_v3 import run_v3_benchmarks
    v2_results = run_v3_benchmarks(v2_path)
    v3_results = run_v3_benchmarks(v3_path)
    comparator = V2V3Comparator(
        {k: v.__dict__ for k, v in v2_results.items()},
        {k: v.__dict__ for k, v in v3_results.items()}
    )
    report = comparator.summary_table()
    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
    return report
```

**Acceptance Criteria:**

- [ ] Loads and evaluates both v2 and v3 checkpoints
- [ ] Calculates delta and percentage improvement per task
- [ ] Validates against v3 targets from enhanced_design_v3.md
- [ ] Generates markdown comparison table

**Tests:** `tests/v3/test_benchmarks_v3.py::test_v2_v3_comparison`

---

#### Issue 6.2.3: Validate Hub Token Routing Effectiveness

**Priority:** 🟡 High | **Hours:** 4 | **Depends On:** Issue 6.2.1

**File:** `src/modeling_studio/evaluation/hub_routing_eval.py`

**Description:**
Validate that hub tokens correctly route information for their assigned capabilities.

```python
"""Hub Token Routing Effectiveness Evaluation."""

from dataclasses import dataclass
from typing import Dict, List
import torch
import torch.nn.functional as F

@dataclass
class HubRoutingMetrics:
    """Metrics for hub token routing effectiveness."""
    hub: str
    assigned_tasks: List[str]
    avg_attention_to_hub: float  # How much text attends to this hub
    hub_representation_similarity: float  # Cosine sim to task-relevant embeddings
    routing_accuracy: float  # Correct hub activation rate

# Hub → Task mapping from enhanced_design_v3.md Section 2.4
HUB_TASK_MAPPING = {
    "EMO": ["emotions", "sentiment", "safety_familyos"],
    "MEM": ["embedding", "ner_family", "temporal"],
    "REL": ["nli", "relations"],
    "TASK": ["intent", "ingress"],
}

class HubRoutingEvaluator:
    """Evaluate hub token routing effectiveness."""

    HUB_POSITIONS = {"EMO": 1, "MEM": 2, "REL": 3, "TASK": 4}

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def evaluate_attention_patterns(self, texts: List[str]) -> Dict[str, float]:
        """Measure attention from text tokens to each hub token."""
        hub_attention = {hub: [] for hub in self.HUB_POSITIONS}

        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)

            # Average attention to hub tokens across all layers/heads
            for layer_attn in outputs.attentions:
                attn = layer_attn.mean(dim=1)  # Average heads
                for hub, pos in self.HUB_POSITIONS.items():
                    # Attention from text tokens (5+) to hub token
                    hub_attention[hub].append(attn[0, 5:, pos].mean().item())

        return {hub: sum(v)/len(v) for hub, v in hub_attention.items()}

    def evaluate_routing_accuracy(
        self,
        samples: List[Dict],  # {"text": ..., "expected_hub": ...}
    ) -> Dict[str, float]:
        """Measure if correct hub is most activated for each task."""
        correct = {hub: 0 for hub in self.HUB_POSITIONS}
        total = {hub: 0 for hub in self.HUB_POSITIONS}

        for sample in samples:
            expected = sample["expected_hub"]
            # Get hub activations
            activations = self._get_hub_activations(sample["text"])
            predicted = max(activations, key=activations.get)

            total[expected] += 1
            if predicted == expected:
                correct[expected] += 1

        return {hub: correct[hub]/max(total[hub], 1) for hub in self.HUB_POSITIONS}

    def _get_hub_activations(self, text: str) -> Dict[str, float]:
        """Get activation magnitude for each hub token."""
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state[0]  # (seq_len, hidden)
        return {
            hub: hidden[pos].norm().item()
            for hub, pos in self.HUB_POSITIONS.items()
        }

    def generate_report(self, test_samples: List[Dict]) -> str:
        """Generate hub routing effectiveness report."""
        texts = [s["text"] for s in test_samples]
        attention = self.evaluate_attention_patterns(texts)
        accuracy = self.evaluate_routing_accuracy(test_samples)

        lines = [
            "# Hub Token Routing Effectiveness Report",
            "",
            "## Attention Patterns (text → hub)",
            *[f"- {hub}: {v:.4f}" for hub, v in attention.items()],
            "",
            "## Routing Accuracy",
            *[f"- {hub}: {v:.2%}" for hub, v in accuracy.items()],
        ]
        return "\n".join(lines)
```

**Acceptance Criteria:**

- [ ] Measures attention flow from text tokens to hub tokens
- [ ] Validates correct hub activation for assigned tasks
- [ ] Reports routing accuracy per hub (target: >90%)
- [ ] Generates effectiveness report

**Tests:** `tests/v3/test_benchmarks_v3.py::test_hub_routing_eval`

---

#### Issue 6.2.4: Measure Latency Impact of 6 Extra Layers

**Priority:** 🟡 High | **Hours:** 3 | **Depends On:** Issue 6.2.1

**File:** `src/modeling_studio/evaluation/latency_benchmark.py`

**Description:**
Benchmark v3 latency across platforms and validate against targets from enhanced_design_v3.md.

```python
"""Latency Benchmarking for ModernBERT v3."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import time
import torch

@dataclass
class LatencyResult:
    """Latency measurement result."""
    platform: str
    seq_length: int
    batch_size: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    meets_target: bool

# Latency targets from enhanced_design_v3.md Section 9.2
LATENCY_TARGETS = {
    "A100": {"v2": 15, "v3_target": 20},  # ms
    "RTX_4090": {"v2": 25, "v3_target": 35},
    "Ryzen_AI_NPU": {"v2": 60, "v3_target": 80},
    "Apple_M3": {"v2": 45, "v3_target": 60},
}

class LatencyBenchmark:
    """Benchmark inference latency for v3 model."""

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device

    def benchmark(
        self,
        seq_lengths: List[int] = [128, 256, 512],
        batch_sizes: List[int] = [1, 8, 32],
        warmup_runs: int = 10,
        benchmark_runs: int = 100,
    ) -> List[LatencyResult]:
        """Run latency benchmark across configurations."""
        results = []
        platform = self._detect_platform()
        target = LATENCY_TARGETS.get(platform, {}).get("v3_target", float("inf"))

        for seq_len in seq_lengths:
            for batch_size in batch_sizes:
                latencies = self._measure_latency(
                    seq_len, batch_size, warmup_runs, benchmark_runs
                )
                import numpy as np
                results.append(LatencyResult(
                    platform=platform,
                    seq_length=seq_len,
                    batch_size=batch_size,
                    mean_ms=np.mean(latencies),
                    p50_ms=np.percentile(latencies, 50),
                    p95_ms=np.percentile(latencies, 95),
                    p99_ms=np.percentile(latencies, 99),
                    meets_target=np.mean(latencies) <= target,
                ))
        return results

    def _measure_latency(
        self, seq_len: int, batch_size: int, warmup: int, runs: int
    ) -> List[float]:
        """Measure inference latency in milliseconds."""
        # Create dummy input
        input_ids = torch.randint(0, 50000, (batch_size, seq_len), device=self.device)
        attention_mask = torch.ones_like(input_ids)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(input_ids, attention_mask)

        # Benchmark
        if self.device == "cuda":
            torch.cuda.synchronize()

        latencies = []
        with torch.no_grad():
            for _ in range(runs):
                start = time.perf_counter()
                _ = self.model(input_ids, attention_mask)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - start) * 1000)

        return latencies

    def _detect_platform(self) -> str:
        """Detect current hardware platform."""
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            if "A100" in name:
                return "A100"
            elif "4090" in name:
                return "RTX_4090"
        return "Unknown"

    def summary_table(self, results: List[LatencyResult]) -> str:
        """Generate markdown latency table."""
        lines = [
            "| Platform | Seq | Batch | Mean | P95 | Target | Status |",
            "|----------|-----|-------|------|-----|--------|--------|",
        ]
        for r in results:
            status = "✅" if r.meets_target else "❌"
            target = LATENCY_TARGETS.get(r.platform, {}).get("v3_target", "N/A")
            lines.append(
                f"| {r.platform} | {r.seq_length} | {r.batch_size} | "
                f"{r.mean_ms:.1f}ms | {r.p95_ms:.1f}ms | {target}ms | {status} |"
            )
        return "\n".join(lines)

def run_latency_benchmark(model_path: str, output_path: str = None) -> str:
    """Run full latency benchmark and generate report."""
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    benchmark = LatencyBenchmark(model, tokenizer)
    results = benchmark.benchmark()
    report = benchmark.summary_table(results)
    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
    return report
```

**Acceptance Criteria:**

- [ ] Benchmarks across seq_lengths (128, 256, 512) and batch_sizes (1, 8, 32)
- [ ] Reports mean, P50, P95, P99 latencies
- [ ] Validates against platform-specific targets
- [ ] Generates comparison table with v2 baseline

**Tests:** `tests/v3/test_benchmarks_v3.py::test_latency_benchmark`

---

### Epic 6.3: Safety Validation

#### Issue 6.3.1: Validate CRISIS Recall ≥99%

**Priority:** P0-Critical | **Estimate:** 6 hours | **Status:** Ready

**Description:**
Validate that v3 model achieves ≥99% recall on CRISIS-class safety detection, exceeding v2's 98% target. This is the most critical safety metric.

**v3-Specific Additions:**

- Hub token [EMO] specialization for safety routing
- Multi-stage training preserves safety from Phase 1.5
- Enhanced attention to suicide, self-harm, abuse keywords

**File:** `src/modeling_studio/evaluation/safety_validation_v3.py`

```python
"""CRISIS Recall Validation for v3 - Target ≥99%."""

from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader

# Safety band indices
CRISIS_IDX = 3
CRISIS_KEYWORDS = [
    "kill myself", "end my life", "suicide", "want to die",
    "hurt myself", "cut myself", "overdose", "no point living",
    "abuse", "hitting me", "touching me", "harming me"
]

@dataclass
class CRISISRecallResult:
    """CRISIS recall validation result."""
    recall: float
    total_crisis_samples: int
    detected_count: int
    missed_samples: list[dict]
    confidence_distribution: dict[str, float]
    keyword_recall: dict[str, float]
    passes_threshold: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "recall": self.recall,
            "total_samples": self.total_crisis_samples,
            "detected": self.detected_count,
            "missed_count": len(self.missed_samples),
            "passes_threshold": self.passes_threshold,
            "keyword_recall": self.keyword_recall,
            "confidence_p50": self.confidence_distribution.get("p50", 0),
            "confidence_p95": self.confidence_distribution.get("p95", 0),
        }

class CRISISRecallValidator:
    """Validates CRISIS recall meets ≥99% target."""

    RECALL_TARGET = 0.99  # v3 target: 99% (up from v2's 98%)

    def __init__(self, model, tokenizer, device: str = "auto", batch_size: int = 32):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def validate(self, dataset, show_progress: bool = True) -> CRISISRecallResult:
        """Run CRISIS recall validation."""
        # Filter to CRISIS samples
        crisis_samples = dataset.filter(lambda x: x.get("label") == CRISIS_IDX)

        if len(crisis_samples) == 0:
            return CRISISRecallResult(
                recall=1.0, total_crisis_samples=0, detected_count=0,
                missed_samples=[], confidence_distribution={},
                keyword_recall={}, passes_threshold=True
            )

        predictions, confidences, texts = self._run_inference(crisis_samples, show_progress)

        # Calculate recall
        detected_mask = predictions == CRISIS_IDX
        detected_count = int(detected_mask.sum())
        recall = detected_count / len(crisis_samples)

        # Collect missed samples for analysis
        missed_samples = []
        for i, (pred, text) in enumerate(zip(predictions, texts)):
            if pred != CRISIS_IDX:
                missed_samples.append({
                    "text": text[:200],
                    "predicted": int(pred),
                    "confidence": float(confidences[i])
                })

        # Confidence distribution
        conf_dist = {
            "mean": float(np.mean(confidences)),
            "p50": float(np.percentile(confidences, 50)),
            "p95": float(np.percentile(confidences, 95)),
            "min": float(np.min(confidences))
        }

        # Keyword-specific recall
        keyword_recall = self._compute_keyword_recall(texts, predictions)

        return CRISISRecallResult(
            recall=recall,
            total_crisis_samples=len(crisis_samples),
            detected_count=detected_count,
            missed_samples=missed_samples[:10],  # Limit for report
            confidence_distribution=conf_dist,
            keyword_recall=keyword_recall,
            passes_threshold=recall >= self.RECALL_TARGET
        )

    def _run_inference(self, dataset, show_progress: bool):
        """Run model inference on CRISIS samples."""
        from tqdm import tqdm

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        all_preds, all_confs, all_texts = [], [], []

        iterator = tqdm(dataloader, desc="CRISIS Validation") if show_progress else dataloader

        with torch.no_grad():
            for batch in iterator:
                inputs = self.tokenizer(
                    batch["text"], padding=True, truncation=True,
                    max_length=512, return_tensors="pt"
                ).to(self.device)

                outputs = self.model(**inputs, capability="safety_familyos")
                logits = outputs.logits if hasattr(outputs, "logits") else outputs

                probs = torch.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1)
                confs = probs[:, CRISIS_IDX]  # CRISIS class confidence

                all_preds.extend(preds.cpu().numpy())
                all_confs.extend(confs.cpu().numpy())
                all_texts.extend(batch["text"])

        return np.array(all_preds), np.array(all_confs), all_texts

    def _compute_keyword_recall(self, texts: list[str], predictions: np.ndarray) -> dict[str, float]:
        """Compute recall for each CRISIS keyword category."""
        keyword_recall = {}

        for keyword in CRISIS_KEYWORDS[:6]:  # Top keywords
            mask = [keyword.lower() in t.lower() for t in texts]
            if sum(mask) > 0:
                keyword_preds = predictions[np.array(mask)]
                keyword_recall[keyword] = float((keyword_preds == CRISIS_IDX).mean())

        return keyword_recall

    def summary(self, result: CRISISRecallResult) -> str:
        """Generate validation summary."""
        status = "✅ PASS" if result.passes_threshold else "❌ FAIL"
        lines = [
            "=" * 60,
            f"CRISIS RECALL VALIDATION - {status}",
            "=" * 60,
            f"Target: ≥{self.RECALL_TARGET:.0%} | Achieved: {result.recall:.2%}",
            f"Samples: {result.total_crisis_samples} | Detected: {result.detected_count}",
            f"Missed: {len(result.missed_samples)}",
            "",
            "Confidence Distribution:",
            f"  Mean: {result.confidence_distribution.get('mean', 0):.3f}",
            f"  P50:  {result.confidence_distribution.get('p50', 0):.3f}",
            f"  P95:  {result.confidence_distribution.get('p95', 0):.3f}",
        ]

        if result.keyword_recall:
            lines.append("\nKeyword-Specific Recall:")
            for kw, rec in result.keyword_recall.items():
                lines.append(f"  '{kw}': {rec:.2%}")

        if result.missed_samples:
            lines.append("\nMissed Samples (first 3):")
            for ms in result.missed_samples[:3]:
                lines.append(f"  - \"{ms['text'][:80]}...\" → pred={ms['predicted']}")

        lines.append("=" * 60)
        return "\n".join(lines)
```

**Acceptance Criteria:**

- [ ] Validates CRISIS recall ≥99% (vs v2's 98%)
- [ ] Reports per-keyword recall for suicide, self-harm, abuse
- [ ] Logs all missed samples for remediation analysis
- [ ] Confidence distribution tracks model certainty

**Tests:** `tests/v3/test_safety_validation.py::test_crisis_recall`

---

#### Issue 6.3.2: Validate Cultural FP Rate ≤1%

**Priority:** P0-Critical | **Estimate:** 6 hours | **Status:** Ready

**Description:**
Validate that v3 model achieves ≤1% false positive rate for cultural expressions (e.g., Indian English hyperbole, venting). This improves on v2's ≤2% target.

**v3-Specific Additions:**

- Hub token [EMO] trained on diverse cultural expressions
- Phase 2 fine-tuning includes curated Indian English samples
- Better distinction between venting and concerning content

**File:** `src/modeling_studio/evaluation/cultural_fp_validation_v3.py`

```python
"""Cultural False Positive Rate Validation for v3 - Target ≤1%."""

from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader

# Cultural expression categories that should NOT escalate
CULTURAL_CATEGORIES = {
    "indian_hyperbole": [
        "I'll die if I don't get this", "killing me softly",
        "I'm dead tired", "this homework is killing me"
    ],
    "venting_expressions": [
        "I hate everything today", "worst day ever",
        "I could scream", "so frustrated right now"
    ],
    "dramatic_expressions": [
        "literally dying", "I can't even", "end me now",
        "life is pain", "I'm so done"
    ],
    "cultural_idioms": [
        "I'll kill for that", "drop dead gorgeous",
        "scared to death", "bored to death"
    ]
}

# Non-GREEN is considered escalation (false positive for safe content)
GREEN_IDX = 0

@dataclass
class CulturalFPResult:
    """Cultural false positive rate validation result."""
    fp_rate: float
    total_cultural_samples: int
    escalated_count: int
    per_category_fp: dict[str, float]
    false_positive_samples: list[dict]
    passes_threshold: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "fp_rate": self.fp_rate,
            "total_samples": self.total_cultural_samples,
            "escalated": self.escalated_count,
            "per_category_fp": self.per_category_fp,
            "passes_threshold": self.passes_threshold,
        }

class CulturalFPValidator:
    """Validates cultural expression false positive rate ≤1%."""

    FP_TARGET = 0.01  # v3 target: ≤1% (down from v2's ≤2%)

    def __init__(self, model, tokenizer, device: str = "auto", batch_size: int = 32):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def validate(self, dataset, show_progress: bool = True) -> CulturalFPResult:
        """Run cultural FP rate validation."""
        # Filter to GREEN-labeled cultural samples
        cultural_samples = dataset.filter(lambda x: x.get("label") == GREEN_IDX)

        if len(cultural_samples) == 0:
            return CulturalFPResult(
                fp_rate=0.0, total_cultural_samples=0, escalated_count=0,
                per_category_fp={}, false_positive_samples=[], passes_threshold=True
            )

        predictions, confidences, texts = self._run_inference(cultural_samples, show_progress)

        # Calculate FP rate (non-GREEN predictions for GREEN content)
        escalated_mask = predictions != GREEN_IDX
        escalated_count = int(escalated_mask.sum())
        fp_rate = escalated_count / len(cultural_samples)

        # Collect false positive samples
        fp_samples = []
        for i, (pred, text) in enumerate(zip(predictions, texts)):
            if pred != GREEN_IDX:
                fp_samples.append({
                    "text": text[:200],
                    "predicted": int(pred),
                    "confidence": float(confidences[i]),
                    "category": self._identify_category(text)
                })

        # Per-category FP rates
        per_category_fp = self._compute_category_fp(texts, predictions)

        return CulturalFPResult(
            fp_rate=fp_rate,
            total_cultural_samples=len(cultural_samples),
            escalated_count=escalated_count,
            per_category_fp=per_category_fp,
            false_positive_samples=fp_samples[:20],
            passes_threshold=fp_rate <= self.FP_TARGET
        )

    def _run_inference(self, dataset, show_progress: bool):
        """Run model inference on cultural samples."""
        from tqdm import tqdm

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        all_preds, all_confs, all_texts = [], [], []

        iterator = tqdm(dataloader, desc="Cultural FP Validation") if show_progress else dataloader

        with torch.no_grad():
            for batch in iterator:
                inputs = self.tokenizer(
                    batch["text"], padding=True, truncation=True,
                    max_length=512, return_tensors="pt"
                ).to(self.device)

                outputs = self.model(**inputs, capability="safety_familyos")
                logits = outputs.logits if hasattr(outputs, "logits") else outputs

                probs = torch.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1)
                confs = probs.max(dim=-1).values

                all_preds.extend(preds.cpu().numpy())
                all_confs.extend(confs.cpu().numpy())
                all_texts.extend(batch["text"])

        return np.array(all_preds), np.array(all_confs), all_texts

    def _identify_category(self, text: str) -> str:
        """Identify which cultural category a text belongs to."""
        text_lower = text.lower()
        for category, examples in CULTURAL_CATEGORIES.items():
            if any(ex.lower() in text_lower for ex in examples):
                return category
        return "other"

    def _compute_category_fp(self, texts: list[str], predictions: np.ndarray) -> dict[str, float]:
        """Compute FP rate per cultural category."""
        category_fp = {}

        for category in CULTURAL_CATEGORIES.keys():
            mask = [self._identify_category(t) == category for t in texts]
            if sum(mask) > 0:
                cat_preds = predictions[np.array(mask)]
                category_fp[category] = float((cat_preds != GREEN_IDX).mean())

        return category_fp

    def summary(self, result: CulturalFPResult) -> str:
        """Generate validation summary."""
        status = "✅ PASS" if result.passes_threshold else "❌ FAIL"
        lines = [
            "=" * 60,
            f"CULTURAL FP RATE VALIDATION - {status}",
            "=" * 60,
            f"Target: ≤{self.FP_TARGET:.0%} | Achieved: {result.fp_rate:.2%}",
            f"Samples: {result.total_cultural_samples} | Escalated: {result.escalated_count}",
        ]

        if result.per_category_fp:
            lines.append("\nPer-Category FP Rates:")
            for cat, rate in result.per_category_fp.items():
                status_icon = "✅" if rate <= self.FP_TARGET else "⚠️"
                lines.append(f"  {status_icon} {cat}: {rate:.2%}")

        if result.false_positive_samples:
            lines.append("\nFalse Positive Examples (first 5):")
            for fp in result.false_positive_samples[:5]:
                lines.append(f"  - \"{fp['text'][:60]}...\" → pred={fp['predicted']} ({fp['category']})")

        lines.append("=" * 60)
        return "\n".join(lines)
```

**Acceptance Criteria:**

- [ ] Validates cultural FP rate ≤1% (vs v2's ≤2%)
- [ ] Reports per-category FP for hyperbole, venting, idioms
- [ ] Identifies specific false positive samples for analysis
- [ ] Tests Indian English expressions specifically

**Tests:** `tests/v3/test_safety_validation.py::test_cultural_fp_rate`

---

#### Issue 6.3.3: Test Hub Token Safety Routing

**Priority:** P1-High | **Estimate:** 5 hours | **Status:** Ready

**Description:**
Validate that hub token [EMO] correctly routes safety-critical content to the safety_familyos head, ensuring proper attention patterns for CRISIS detection.

**v3-Specific Additions:**

- Hub token [EMO] (idx=1) handles emotional and safety routing
- Global bidirectional attention ensures hub sees all tokens
- Attention weight analysis for interpretability

**File:** `src/modeling_studio/evaluation/hub_safety_routing_v3.py`

```python
"""Hub Token Safety Routing Validation for v3."""

from dataclasses import dataclass
from typing import Any
import numpy as np
import torch

# Hub token indices (from enhanced_design_v3.md)
HUB_TOKENS = {"[EMO]": 1, "[MEM]": 2, "[REL]": 3, "[TASK]": 4}
EMO_HUB_IDX = 1
CRISIS_IDX = 3

@dataclass
class HubSafetyRoutingResult:
    """Hub token safety routing validation result."""
    emo_attention_to_crisis_keywords: float
    hub_routing_accuracy: float
    attention_concentration: float  # How focused attention is
    per_band_hub_activation: dict[str, float]
    routing_examples: list[dict]
    passes_validation: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "emo_attention_to_crisis": self.emo_attention_to_crisis_keywords,
            "routing_accuracy": self.hub_routing_accuracy,
            "attention_concentration": self.attention_concentration,
            "per_band_activation": self.per_band_hub_activation,
            "passes_validation": self.passes_validation,
        }

class HubSafetyRoutingValidator:
    """Validates hub token [EMO] routing for safety detection."""

    ATTENTION_THRESHOLD = 0.15  # [EMO] should attend ≥15% to crisis keywords
    ROUTING_ACCURACY_TARGET = 0.95  # 95% of safety samples use [EMO] hub

    def __init__(self, model, tokenizer, device: str = "auto"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

        # Crisis keywords to track attention
        self.crisis_keywords = ["kill", "suicide", "die", "hurt", "abuse", "harm", "dead", "end"]

    def validate(self, dataset, num_samples: int = 100) -> HubSafetyRoutingResult:
        """Validate hub token safety routing."""
        # Sample from dataset
        samples = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))

        emo_attentions = []
        routing_correct = []
        per_band_activation = {band: [] for band in ["green", "amber", "red", "crisis"]}
        routing_examples = []

        for sample in samples:
            text = sample["text"]
            label = sample.get("label", 0)
            band = ["green", "amber", "red", "crisis"][label]

            # Get attention weights with hook
            attention_data = self._get_hub_attention(text)

            # Track [EMO] attention to crisis keywords
            keyword_attention = self._compute_keyword_attention(
                text, attention_data["emo_attention"]
            )
            emo_attentions.append(keyword_attention)

            # Track routing correctness (safety uses [EMO] hub)
            if label >= 2:  # RED or CRISIS
                routing_correct.append(attention_data["emo_activation"] > 0.3)
            else:
                routing_correct.append(True)  # Non-safety can use any hub

            # Per-band hub activation
            per_band_activation[band].append(attention_data["emo_activation"])

            # Store examples
            if len(routing_examples) < 10:
                routing_examples.append({
                    "text": text[:100],
                    "band": band,
                    "emo_attention": float(keyword_attention),
                    "emo_activation": float(attention_data["emo_activation"]),
                })

        # Compute aggregates
        avg_emo_attention = np.mean(emo_attentions) if emo_attentions else 0.0
        routing_accuracy = np.mean(routing_correct) if routing_correct else 0.0

        # Attention concentration (entropy-based)
        attention_concentration = 1.0 - np.std(emo_attentions) if emo_attentions else 0.0

        # Per-band averages
        per_band_avg = {
            band: float(np.mean(vals)) if vals else 0.0
            for band, vals in per_band_activation.items()
        }

        passes = (
            avg_emo_attention >= self.ATTENTION_THRESHOLD and
            routing_accuracy >= self.ROUTING_ACCURACY_TARGET
        )

        return HubSafetyRoutingResult(
            emo_attention_to_crisis_keywords=float(avg_emo_attention),
            hub_routing_accuracy=float(routing_accuracy),
            attention_concentration=float(attention_concentration),
            per_band_hub_activation=per_band_avg,
            routing_examples=routing_examples,
            passes_validation=passes,
        )

    def _get_hub_attention(self, text: str) -> dict[str, float]:
        """Extract hub token attention weights."""
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True,
                capability="safety_familyos"
            )

        # Extract attention from last layer
        if hasattr(outputs, "attentions") and outputs.attentions:
            last_layer_attn = outputs.attentions[-1]  # [batch, heads, seq, seq]
            # Average over heads, get [EMO] hub attention (position 1)
            emo_attention = last_layer_attn[0, :, EMO_HUB_IDX, :].mean(dim=0).cpu().numpy()
            emo_activation = float(emo_attention.mean())
        else:
            emo_attention = np.zeros(inputs["input_ids"].shape[1])
            emo_activation = 0.0

        return {"emo_attention": emo_attention, "emo_activation": emo_activation}

    def _compute_keyword_attention(self, text: str, attention_weights: np.ndarray) -> float:
        """Compute attention to crisis keywords."""
        tokens = self.tokenizer.tokenize(text)

        keyword_attention = 0.0
        keyword_count = 0

        for i, token in enumerate(tokens[:len(attention_weights)-2]):  # Exclude special tokens
            token_text = token.replace("##", "").lower()
            if any(kw in token_text for kw in self.crisis_keywords):
                if i + 1 < len(attention_weights):  # +1 for [CLS]
                    keyword_attention += attention_weights[i + 1]
                    keyword_count += 1

        return keyword_attention / max(keyword_count, 1)

    def summary(self, result: HubSafetyRoutingResult) -> str:
        """Generate validation summary."""
        status = "✅ PASS" if result.passes_validation else "❌ FAIL"
        lines = [
            "=" * 60,
            f"HUB TOKEN SAFETY ROUTING VALIDATION - {status}",
            "=" * 60,
            f"[EMO] Attention to Crisis Keywords: {result.emo_attention_to_crisis_keywords:.2%}",
            f"  Target: ≥{self.ATTENTION_THRESHOLD:.0%}",
            f"Routing Accuracy: {result.hub_routing_accuracy:.2%}",
            f"  Target: ≥{self.ROUTING_ACCURACY_TARGET:.0%}",
            f"Attention Concentration: {result.attention_concentration:.3f}",
            "",
            "Per-Band [EMO] Activation:",
        ]

        for band, activation in result.per_band_hub_activation.items():
            lines.append(f"  {band.upper()}: {activation:.3f}")

        if result.routing_examples:
            lines.append("\nRouting Examples:")
            for ex in result.routing_examples[:3]:
                lines.append(
                    f"  [{ex['band'].upper()}] attn={ex['emo_attention']:.2f} "
                    f"act={ex['emo_activation']:.2f}: \"{ex['text'][:50]}...\""
                )

        lines.append("=" * 60)
        return "\n".join(lines)


def run_safety_validation(model, tokenizer, dataset, output_path: str = None) -> dict:
    """Run all safety validations."""
    results = {}

    # 1. CRISIS Recall
    crisis_validator = CRISISRecallValidator(model, tokenizer)
    crisis_result = crisis_validator.validate(dataset)
    results["crisis_recall"] = crisis_result.to_dict()
    print(crisis_validator.summary(crisis_result))

    # 2. Cultural FP Rate
    cultural_validator = CulturalFPValidator(model, tokenizer)
    cultural_result = cultural_validator.validate(dataset)
    results["cultural_fp"] = cultural_result.to_dict()
    print(cultural_validator.summary(cultural_result))

    # 3. Hub Safety Routing
    hub_validator = HubSafetyRoutingValidator(model, tokenizer)
    hub_result = hub_validator.validate(dataset)
    results["hub_routing"] = hub_result.to_dict()
    print(hub_validator.summary(hub_result))

    # Overall pass/fail
    results["all_passed"] = all([
        crisis_result.passes_threshold,
        cultural_result.passes_threshold,
        hub_result.passes_validation,
    ])

    if output_path:
        import json
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    return results
```

**Acceptance Criteria:**

- [ ] Validates [EMO] hub attention to crisis keywords ≥15%
- [ ] Validates safety routing accuracy ≥95%
- [ ] Reports per-band hub activation patterns
- [ ] Provides interpretable attention analysis

**Tests:** `tests/v3/test_safety_validation.py::test_hub_safety_routing`

---

## 🏁 Milestone 7: Production Export & Integration

**Goal:** Export production-ready v3 model and update integrations

### Epic 7.1: Model Export

**Goal:** Export production-ready v3 model with merged weights and calibration
**Total Estimated Hours:** 18 hours

#### Issue 7.1.1: Implement LoRA Weight Merging

**Priority:** P0-Critical | **Estimate:** 5 hours | **Status:** Ready

**Description:**
Merge LoRA adapter weights into base model for zero-overhead production inference. V3 uses LoRA adapters in Phase 1-2 training that must be merged before deployment.

**v3-Specific Considerations:**

- LoRA applied to q_proj, k_proj, v_proj, o_proj in layers 19-28
- Hub tokens have separate LoRA adapters
- Merge preserves function-preserving initialization guarantees

**File:** `export_utility/lora_merge_v3.py`

```python
"""LoRA Weight Merging for ModernBERT v3 Production Export."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

@dataclass
class LoRAMergeConfig:
    """Configuration for LoRA weight merging."""
    lora_alpha: float = 16.0
    lora_r: int = 16
    target_modules: list = None  # q_proj, k_proj, v_proj, o_proj
    merge_hub_lora: bool = True
    validate_after_merge: bool = True

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

@dataclass
class LoRAMergeResult:
    """Result of LoRA weight merging."""
    num_modules_merged: int
    total_lora_params: int
    params_removed: int
    validation_passed: bool
    merged_modules: list
    errors: list

class LoRAWeightMerger:
    """Merges LoRA adapter weights into base model."""

    def __init__(self, config: LoRAMergeConfig = None):
        self.config = config or LoRAMergeConfig()
        self.scaling = self.config.lora_alpha / self.config.lora_r

    def merge(self, model: nn.Module, validate: bool = True) -> LoRAMergeResult:
        """Merge all LoRA weights into base model."""
        merged_modules = []
        total_lora_params = 0
        errors = []

        for name, module in model.named_modules():
            if self._has_lora_weights(module):
                try:
                    params_merged = self._merge_module(module, name)
                    merged_modules.append(name)
                    total_lora_params += params_merged
                except Exception as e:
                    errors.append(f"{name}: {str(e)}")

        # Remove LoRA parameters after merging
        params_removed = self._remove_lora_params(model)

        # Validate merged model
        validation_passed = True
        if validate and self.config.validate_after_merge:
            validation_passed = self._validate_merge(model)

        result = LoRAMergeResult(
            num_modules_merged=len(merged_modules),
            total_lora_params=total_lora_params,
            params_removed=params_removed,
            validation_passed=validation_passed,
            merged_modules=merged_modules,
            errors=errors,
        )

        logger.info(f"Merged {len(merged_modules)} LoRA modules, removed {params_removed} params")
        return result

    def _has_lora_weights(self, module: nn.Module) -> bool:
        """Check if module has LoRA weights."""
        return hasattr(module, "lora_A") and hasattr(module, "lora_B")

    def _merge_module(self, module: nn.Module, name: str) -> int:
        """Merge LoRA weights into a single module."""
        lora_A = module.lora_A  # [r, in_features]
        lora_B = module.lora_B  # [out_features, r]

        # Compute LoRA delta: W_delta = B @ A * scaling
        delta = (lora_B @ lora_A) * self.scaling

        # Merge into base weight
        if hasattr(module, "weight"):
            module.weight.data += delta
        elif hasattr(module, "base_weight"):
            module.base_weight.data += delta

        params_count = lora_A.numel() + lora_B.numel()
        logger.debug(f"Merged {name}: {params_count} LoRA params")
        return params_count

    def _remove_lora_params(self, model: nn.Module) -> int:
        """Remove LoRA parameters after merging."""
        params_removed = 0

        for name, module in model.named_modules():
            attrs_to_remove = []
            for attr in ["lora_A", "lora_B", "lora_dropout", "scaling"]:
                if hasattr(module, attr):
                    param = getattr(module, attr)
                    if isinstance(param, (nn.Parameter, torch.Tensor)):
                        params_removed += param.numel()
                    attrs_to_remove.append(attr)

            for attr in attrs_to_remove:
                delattr(module, attr)

        return params_removed

    def _validate_merge(self, model: nn.Module) -> bool:
        """Validate merged model produces valid outputs."""
        try:
            model.eval()
            dummy_input = torch.randint(0, 1000, (1, 128))
            with torch.no_grad():
                output = model(dummy_input)
            return output is not None
        except Exception as e:
            logger.error(f"Merge validation failed: {e}")
            return False

def merge_lora_weights(
    model_path: str,
    output_path: str,
    config: LoRAMergeConfig = None
) -> LoRAMergeResult:
    """Convenience function to merge LoRA weights and save model."""
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    merger = LoRAWeightMerger(config)
    result = merger.merge(model)

    if result.validation_passed:
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        logger.info(f"Saved merged model to {output_path}")

    return result
```

**Acceptance Criteria:**

- [ ] Merges LoRA A/B matrices with correct scaling (alpha/r)
- [ ] Removes LoRA parameters after merge (reduces model size)
- [ ] Validates merged model produces valid outputs
- [ ] Supports hub token LoRA adapters

**Tests:** `tests/v3/test_export_v3.py::test_lora_merge`

---

#### Issue 7.1.2: Implement Temperature Calibration per Head

**Priority:** P1-High | **Estimate:** 4 hours | **Status:** Ready

**Description:**
Apply learned temperature scaling per task head to calibrate confidence outputs. Each head gets its own temperature based on validation set calibration.

**v3-Specific Considerations:**

- Safety heads require conservative temperatures (higher = less confident)
- Embedding head doesn't use temperature (cosine similarity)
- Store temperatures in `calibration_config.yaml`

**File:** `export_utility/temperature_calibration_v3.py`

```python
"""Temperature Calibration per Head for ModernBERT v3."""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)

@dataclass
class CalibrationResult:
    """Result for a single head calibration."""
    head_name: str
    optimal_temperature: float
    ece_before: float
    ece_after: float
    num_samples: int

@dataclass
class TemperatureConfig:
    """Temperature configuration for all heads."""
    temperatures: Dict[str, float]
    calibration_method: str = "platt"  # platt, histogram, isotonic

    def to_dict(self) -> Dict[str, Any]:
        return {
            "temperatures": self.temperatures,
            "method": self.calibration_method
        }

class TemperatureCalibrator:
    """Calibrates temperature scaling per task head."""

    DEFAULT_TEMPERATURES = {
        "safety_familyos": 1.5,  # Conservative (less confident)
        "safety_generic": 1.5,
        "emotions": 1.2,
        "sentiment": 1.0,
        "ner_general": 1.0,
        "ner_family": 1.0,
        "nli": 1.0,
        "intent": 1.0,
        "ingress": 1.0,
        "relations": 1.1,
        "temporal": 1.0,
    }

    def __init__(self, model, tokenizer, device: str = "auto"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def calibrate_all(
        self,
        validation_datasets: Dict[str, Any],
        heads_to_calibrate: list = None
    ) -> TemperatureConfig:
        """Calibrate temperature for all heads."""
        temperatures = {}

        heads = heads_to_calibrate or list(self.model.heads.keys())

        for head_name in heads:
            if head_name == "embedding":
                continue  # Skip embedding head

            if head_name not in validation_datasets:
                temperatures[head_name] = self.DEFAULT_TEMPERATURES.get(head_name, 1.0)
                logger.warning(f"No validation data for {head_name}, using default T={temperatures[head_name]}")
                continue

            result = self.calibrate_head(head_name, validation_datasets[head_name])
            temperatures[head_name] = result.optimal_temperature
            logger.info(f"{head_name}: T={result.optimal_temperature:.3f} (ECE: {result.ece_before:.4f} → {result.ece_after:.4f})")

        return TemperatureConfig(temperatures=temperatures)

    def calibrate_head(self, head_name: str, dataset) -> CalibrationResult:
        """Calibrate temperature for a single head using Platt scaling."""
        logits_list, labels_list = self._collect_logits(head_name, dataset)

        logits = torch.cat(logits_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        # ECE before calibration
        ece_before = self._compute_ece(logits, labels, temperature=1.0)

        # Find optimal temperature via grid search
        best_temp = 1.0
        best_ece = ece_before

        for temp in np.arange(0.5, 3.0, 0.1):
            ece = self._compute_ece(logits, labels, temperature=temp)
            if ece < best_ece:
                best_ece = ece
                best_temp = temp

        return CalibrationResult(
            head_name=head_name,
            optimal_temperature=best_temp,
            ece_before=ece_before,
            ece_after=best_ece,
            num_samples=len(labels)
        )

    def _collect_logits(self, head_name: str, dataset) -> tuple:
        """Collect logits and labels from dataset."""
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        logits_list, labels_list = [], []

        with torch.no_grad():
            for batch in dataloader:
                inputs = self.tokenizer(
                    batch["text"], padding=True, truncation=True,
                    max_length=512, return_tensors="pt"
                ).to(self.device)

                outputs = self.model(**inputs, capability=head_name)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs

                logits_list.append(logits.cpu())
                labels_list.append(torch.tensor(batch["label"]))

        return logits_list, labels_list

    def _compute_ece(self, logits: torch.Tensor, labels: torch.Tensor, temperature: float, n_bins: int = 15) -> float:
        """Compute Expected Calibration Error."""
        probs = torch.softmax(logits / temperature, dim=-1)
        confidences, predictions = probs.max(dim=-1)
        accuracies = predictions.eq(labels).float()

        ece = 0.0
        for i in range(n_bins):
            bin_lower = i / n_bins
            bin_upper = (i + 1) / n_bins
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

            if in_bin.sum() > 0:
                bin_accuracy = accuracies[in_bin].mean()
                bin_confidence = confidences[in_bin].mean()
                ece += (in_bin.sum() / len(confidences)) * abs(bin_accuracy - bin_confidence)

        return float(ece)

def apply_temperature_to_model(model, temperatures: Dict[str, float]) -> None:
    """Apply temperature scaling to model heads."""
    for head_name, temp in temperatures.items():
        if head_name in model.heads:
            model.heads[head_name].temperature = temp
            logger.info(f"Applied T={temp:.3f} to {head_name}")
```

**Acceptance Criteria:**

- [ ] Calibrates temperature per head using validation data
- [ ] Computes ECE before/after calibration
- [ ] Safety heads get conservative (higher) temperatures
- [ ] Stores temperatures in config for inference

**Tests:** `tests/v3/test_export_v3.py::test_temperature_calibration`

---

#### Issue 7.1.3: Export Unified v3 Model

**Priority:** P0-Critical | **Estimate:** 5 hours | **Status:** Ready

**Description:**
Export complete v3 model with merged LoRA, calibrated temperatures, and hub token configuration for production deployment.

**v3-Specific Additions:**

- Include hub token embeddings and routing config
- Export `capabilities_v3.json` with hub assignments
- Generate model card with v3 architecture details

**File:** `export_utility/export_v3_model.py`

```python
"""Unified v3 Model Export for Production Deployment."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional
import json
import yaml
import torch
import logging

logger = logging.getLogger(__name__)

@dataclass
class V3ExportConfig:
    """Configuration for v3 model export."""
    model_path: str
    output_path: str
    merge_lora: bool = True
    calibrate_temperatures: bool = True
    export_format: str = "safetensors"  # safetensors, pytorch
    include_onnx: bool = False
    model_name: str = "familyos-unified-v3"

@dataclass
class V3ExportResult:
    """Result of v3 model export."""
    output_path: str
    model_size_mb: float
    num_heads: int
    hub_tokens: list
    lora_merged: bool
    temperatures_applied: bool
    files_exported: list

# Hub token assignments from enhanced_design_v3.md
HUB_ASSIGNMENTS = {
    "[EMO]": ["emotions", "sentiment", "safety_familyos", "safety_generic"],
    "[MEM]": ["embedding", "ner_family", "temporal"],
    "[REL]": ["nli", "relations"],
    "[TASK]": ["intent", "ingress", "ner_general"],
}

class V3ModelExporter:
    """Exports unified v3 model for production."""

    def __init__(self, config: V3ExportConfig):
        self.config = config
        self.output_dir = Path(config.output_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(self) -> V3ExportResult:
        """Export complete v3 model."""
        from transformers import AutoTokenizer

        # Load model
        logger.info(f"Loading model from {self.config.model_path}")
        model, tokenizer = self._load_model()

        files_exported = []

        # 1. Merge LoRA weights
        lora_merged = False
        if self.config.merge_lora:
            from .lora_merge_v3 import LoRAWeightMerger
            merger = LoRAWeightMerger()
            result = merger.merge(model)
            lora_merged = result.validation_passed
            logger.info(f"LoRA merge: {result.num_modules_merged} modules merged")

        # 2. Apply temperature calibration
        temps_applied = False
        if self.config.calibrate_temperatures:
            from .temperature_calibration_v3 import TemperatureCalibrator
            # Use default temperatures if no validation data
            temps = TemperatureCalibrator.DEFAULT_TEMPERATURES
            for head_name, temp in temps.items():
                if head_name in model.heads:
                    model.heads[head_name].temperature = temp
            temps_applied = True

        # 3. Export model weights
        if self.config.export_format == "safetensors":
            self._export_safetensors(model, tokenizer)
            files_exported.append("model.safetensors")
        else:
            self._export_pytorch(model, tokenizer)
            files_exported.append("pytorch_model.bin")

        # 4. Export capabilities_v3.json
        self._export_capabilities(model)
        files_exported.append("capabilities_v3.json")

        # 5. Export hub routing config
        self._export_hub_config()
        files_exported.append("hub_routing.yaml")

        # 6. Export calibration config
        self._export_calibration_config(model)
        files_exported.append("calibration_config.yaml")

        # 7. Generate model card
        self._generate_model_card(model)
        files_exported.append("README.md")

        # 8. Optional ONNX export
        if self.config.include_onnx:
            self._export_onnx(model, tokenizer)
            files_exported.append("model.onnx")

        # Calculate model size
        model_size = sum(
            f.stat().st_size for f in self.output_dir.iterdir() if f.is_file()
        ) / (1024 * 1024)

        return V3ExportResult(
            output_path=str(self.output_dir),
            model_size_mb=model_size,
            num_heads=len(model.heads),
            hub_tokens=["[EMO]", "[MEM]", "[REL]", "[TASK]"],
            lora_merged=lora_merged,
            temperatures_applied=temps_applied,
            files_exported=files_exported,
        )

    def _load_model(self):
        """Load model and tokenizer."""
        from modeling_studio.models import ModernBertMultiTaskModel
        from transformers import AutoTokenizer

        model = ModernBertMultiTaskModel.from_pretrained(self.config.model_path)
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        model.eval()
        return model, tokenizer

    def _export_safetensors(self, model, tokenizer):
        """Export in safetensors format."""
        from safetensors.torch import save_file

        state_dict = {k: v.contiguous() for k, v in model.state_dict().items()
                      if isinstance(v, torch.Tensor)}
        save_file(state_dict, str(self.output_dir / "model.safetensors"))
        model.config.save_pretrained(str(self.output_dir))
        tokenizer.save_pretrained(str(self.output_dir))

    def _export_pytorch(self, model, tokenizer):
        """Export in PyTorch format."""
        torch.save(model.state_dict(), self.output_dir / "pytorch_model.bin")
        model.config.save_pretrained(str(self.output_dir))
        tokenizer.save_pretrained(str(self.output_dir))

    def _export_capabilities(self, model):
        """Export capabilities_v3.json with hub assignments."""
        capabilities = {}

        for head_name, head in model.heads.items():
            hub = self._get_hub_for_capability(head_name)
            capabilities[head_name] = {
                "type": head.__class__.__name__,
                "num_labels": getattr(head, "num_labels", None),
                "hub_token": hub,
                "temperature": getattr(head, "temperature", 1.0),
            }

        with open(self.output_dir / "capabilities_v3.json", "w") as f:
            json.dump(capabilities, f, indent=2)

    def _get_hub_for_capability(self, capability: str) -> str:
        """Get assigned hub token for a capability."""
        for hub, caps in HUB_ASSIGNMENTS.items():
            if capability in caps:
                return hub
        return "[TASK]"  # Default

    def _export_hub_config(self):
        """Export hub routing configuration."""
        config = {
            "hub_tokens": {
                "[EMO]": {"position": 1, "capabilities": HUB_ASSIGNMENTS["[EMO]"]},
                "[MEM]": {"position": 2, "capabilities": HUB_ASSIGNMENTS["[MEM]"]},
                "[REL]": {"position": 3, "capabilities": HUB_ASSIGNMENTS["[REL]"]},
                "[TASK]": {"position": 4, "capabilities": HUB_ASSIGNMENTS["[TASK]"]},
            },
            "global_attention_positions": [0, 1, 2, 3, 4],  # CLS + 4 hubs
            "version": "v3.3",
        }

        with open(self.output_dir / "hub_routing.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    def _export_calibration_config(self, model):
        """Export calibration configuration."""
        config = {"temperatures": {}, "thresholds": {}}

        for head_name, head in model.heads.items():
            config["temperatures"][head_name] = getattr(head, "temperature", 1.0)

        # Safety thresholds
        config["thresholds"]["safety_familyos"] = {
            "crisis_threshold": 0.5,
            "red_threshold": 0.4,
            "amber_threshold": 0.3,
        }

        with open(self.output_dir / "calibration_config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    def _generate_model_card(self, model):
        """Generate README.md model card."""
        capabilities_list = "\n".join(
            f"- **{cap}**: {head.__class__.__name__}"
            for cap, head in model.heads.items()
        )

        card = f"""# {self.config.model_name}

## Model Description
ModernBERT v3 Ultra - 28 layers, 768 hidden, 4 hub tokens for task routing.

## Capabilities
{capabilities_list}

## Hub Tokens
- **[EMO]**: Emotions, Sentiment, Safety
- **[MEM]**: Embeddings, NER Family, Temporal
- **[REL]**: NLI, Relations
- **[TASK]**: Intent, Ingress, NER General

## Usage
from modeling_studio.models import ModernBertMultiTaskModel
model = ModernBertMultiTaskModel.from_pretrained("{self.output_dir}")
"""
        with open(self.output_dir / "README.md", "w") as f:
            f.write(card)

def export_v3_model(model_path: str, output_path: str, **kwargs) -> V3ExportResult:
    """Convenience function to export v3 model."""
    config = V3ExportConfig(model_path=model_path, output_path=output_path, **kwargs)
    exporter = V3ModelExporter(config)
    return exporter.export()
```

**Acceptance Criteria:**

- [ ] Exports model with merged LoRA weights
- [ ] Includes `capabilities_v3.json` with hub assignments
- [ ] Generates `hub_routing.yaml` configuration
- [ ] Creates model card with v3 architecture details

**Tests:** `tests/v3/test_export_v3.py::test_unified_export`

---

#### Issue 7.1.4: Export ONNX for NPU Deployment

**Priority:** P1-High | **Estimate:** 4 hours | **Status:** Ready

**Description:**
Export v3 model to ONNX format optimized for NPU deployment (Ryzen AI, Apple M3 Neural Engine).

**v3-Specific Considerations:**

- Hub tokens require static positions in ONNX graph
- Per-capability ONNX export (each head gets own file)
- INT8 quantization for NPU efficiency

**File:** `export_utility/export_onnx_v3.py`

```python
"""ONNX Export for ModernBERT v3 NPU Deployment."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class ONNXExportConfig:
    """Configuration for v3 ONNX export."""
    model_path: str
    output_path: str
    capabilities: List[str] = None  # None = all
    opset_version: int = 17
    quantize: str = "none"  # none, dynamic, static
    optimize_for_npu: bool = True
    max_seq_length: int = 512

@dataclass
class ONNXExportResult:
    """Result of ONNX export."""
    capability: str
    onnx_path: str
    size_mb: float
    quantized: bool
    validated: bool

class V3ONNXWrapper(nn.Module):
    """Wrapper for v3 model ONNX export with hub tokens."""

    HUB_POSITIONS = {0: "[CLS]", 1: "[EMO]", 2: "[MEM]", 3: "[REL]", 4: "[TASK]"}

    def __init__(self, model, capability: str):
        super().__init__()
        self.model = model
        self.capability = capability
        self.is_token_level = capability in ["ner_general", "ner_family", "temporal"]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward with explicit hub token handling."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task=self.capability
        )
        return outputs["logits"]

class V3ONNXExporter:
    """Exports v3 model to ONNX with NPU optimizations."""

    def __init__(self, config: ONNXExportConfig):
        self.config = config
        self.output_dir = Path(config.output_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_all(self) -> List[ONNXExportResult]:
        """Export all requested capabilities to ONNX."""
        import onnx
        from transformers import AutoTokenizer
        from modeling_studio.models import ModernBertMultiTaskModel

        model = ModernBertMultiTaskModel.from_pretrained(self.config.model_path)
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        model.eval()

        capabilities = self.config.capabilities or list(model.heads.keys())
        results = []

        for cap in capabilities:
            if cap == "embedding":
                result = self._export_embedding(model, tokenizer, cap)
            else:
                result = self._export_classification(model, tokenizer, cap)
            results.append(result)

        # Export hub token embeddings separately
        self._export_hub_embeddings(model)

        return results

    def _export_classification(self, model, tokenizer, capability: str) -> ONNXExportResult:
        """Export classification head to ONNX."""
        import onnx

        wrapper = V3ONNXWrapper(model, capability)
        wrapper.eval()

        # Create dummy input
        dummy = tokenizer("Sample text", return_tensors="pt", max_length=128, padding="max_length")

        onnx_path = self.output_dir / f"{capability}.onnx"

        dynamic_axes = {
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "output": {0: "batch"},
        }

        if wrapper.is_token_level:
            dynamic_axes["output"][1] = "seq"

        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (dummy["input_ids"], dummy["attention_mask"]),
                str(onnx_path),
                opset_version=self.config.opset_version,
                input_names=["input_ids", "attention_mask"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
            )

        # Validate
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)

        # Apply quantization
        quantized = False
        if self.config.quantize == "dynamic":
            onnx_path = self._quantize_dynamic(onnx_path)
            quantized = True

        size_mb = onnx_path.stat().st_size / (1024 * 1024)

        logger.info(f"Exported {capability} to ONNX ({size_mb:.2f} MB)")

        return ONNXExportResult(
            capability=capability,
            onnx_path=str(onnx_path),
            size_mb=size_mb,
            quantized=quantized,
            validated=True,
        )

    def _export_embedding(self, model, tokenizer, capability: str) -> ONNXExportResult:
        """Export embedding head to ONNX."""
        # Similar to classification but output is [batch, hidden_size]
        return self._export_classification(model, tokenizer, capability)

    def _quantize_dynamic(self, onnx_path: Path) -> Path:
        """Apply dynamic INT8 quantization."""
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantized_path = onnx_path.with_name(onnx_path.stem + "_int8.onnx")

        quantize_dynamic(
            model_input=str(onnx_path),
            model_output=str(quantized_path),
            weight_type=QuantType.QInt8,
        )

        logger.info(f"Quantized to INT8: {quantized_path.name}")
        return quantized_path

    def _export_hub_embeddings(self, model):
        """Export hub token embeddings for NPU initialization."""
        hub_embeddings = {}

        for name, param in model.named_parameters():
            if "hub_token" in name or "hub_embedding" in name:
                hub_embeddings[name] = param.detach().cpu().numpy()

        if hub_embeddings:
            np.savez(self.output_dir / "hub_embeddings.npz", **hub_embeddings)
            logger.info(f"Exported {len(hub_embeddings)} hub embeddings")

def export_onnx_for_npu(model_path: str, output_path: str, **kwargs) -> List[ONNXExportResult]:
    """Convenience function to export v3 to ONNX for NPU."""
    config = ONNXExportConfig(model_path=model_path, output_path=output_path, **kwargs)
    exporter = V3ONNXExporter(config)
    return exporter.export_all()
```

**Acceptance Criteria:**

- [ ] Exports per-capability ONNX files
- [ ] Hub token positions handled correctly in ONNX graph
- [ ] Dynamic INT8 quantization for NPU efficiency
- [ ] Validates ONNX models after export

**Tests:** `tests/v3/test_export_v3.py::test_onnx_npu_export`

---

### Epic 7.2: K0 Integration Updates

#### Issue 7.2.1: Update Model Registry for v3

#### Issue 7.2.2: Update Unified Output API for Hub Routing

#### Issue 7.2.3: Update K0 Module Migration Guide for v3

---

### Epic 7.3: Documentation

#### Issue 7.3.1: Document v3 Architecture Changes

#### Issue 7.3.2: Document Training Phase Procedures

#### Issue 7.3.3: Create v3 Deployment Guide

---

## 📊 Quality Targets

### Capability Targets (v2 → v3)

| Capability | Metric | v2 Target | v3 Target | Improvement |
|------------|--------|-----------|-----------|-------------|
| ner_general | F1 | 91% | 93% | +2% |
| ner_family | F1 | 88% | 91% | +3% |
| sentiment | Accuracy | 94% | 96% | +2% |
| emotions | Macro F1 | 78% | 82% | +4% |
| safety_familyos | CRISIS Recall | 98% | **99%** | +1% |
| safety_familyos | Cultural FP | ≤2% | **≤1%** | Better |
| ingress | Accuracy | 92% | 95% | +3% |
| relation | F1 | 82% | 87% | +5% |
| intent | Accuracy | 90% | 93% | +3% |
| temporal | F1 | 85% | 89% | +4% |
| nli | Accuracy | 88% | 91% | +3% |
| embedding | Recall@10 | 85% | 90% | +5% |

### Latency Targets

| Platform | v2 Latency | v3 Target | Notes |
|----------|------------|-----------|-------|
| A100 GPU | ~15ms | <20ms | +6 layers overhead |
| RTX 4090 | ~25ms | <35ms | Consumer GPU |
| Ryzen AI NPU | ~60ms | <80ms | Edge deployment |
| Apple M3 Neural | ~45ms | <60ms | macOS edge |

### Forgetting Gates

| Benchmark | Max Allowed Drop | Action if Exceeded |
|-----------|------------------|--------------------|
| CoNLL-2003 (NER) | ≤ 2% F1 | Increase replay ratio |
| SST-2 (Sentiment) | ≤ 2% Acc | Increase replay ratio |
| MNLI (NLI) | ≤ 2% Acc | Increase replay ratio |

---

## 🗓️ Implementation Timeline

| Week | Milestone | Key Deliverables |
|------|-----------|------------------|
| 1 | M1: Configuration & Hub Tokens | config_v3.py, hub_tokens.py, hub_initialization_v3.py |
| 2 | M2: Attention & Layers | attention_v3.py, layers_v3.py, lora_v3.py |
| 3 | M3: Model Assembly | modernbert_v3.py, pair_encoder_v3.py |
| 4 | M4: Initialization | initialization_v3.py, verification scripts |
| 5 | M5: Training Infrastructure | trainer_v3.py, collators_v3.py, training scripts |
| 6 | M6: Evaluation | Benchmark runs, forgetting checks |
| 7 | M7: Export & Integration | ONNX export, K0 updates, documentation |

**Total: ~7 weeks**

---

## 📋 Test File Inventory

| File | Status | Purpose |
|------|--------|---------|
| `tests/v3/test_config_v3.py` | 📝 NEW | v3 configuration tests |
| `tests/v3/test_hub_tokens.py` | 📝 NEW | Hub token system tests |
| `tests/v3/test_attention_v3.py` | 📝 NEW | Sliding window + global attention tests |
| `tests/v3/test_layers_v3.py` | 📝 NEW | v3 layer tests |
| `tests/v3/test_modernbert_v3.py` | 📝 NEW | Full model integration tests |
| `tests/v3/test_initialization_v3.py` | 📝 NEW | Function preserving growth tests |
| `tests/v3/test_trainer_v3.py` | 📝 NEW | Phase-based training tests |

---

**Document Version:** 3.0
**Created:** December 2025
**Status:** Skeleton - Awaiting Issue Population
**Based On:** enhanced_design_v3.md + implementation_plan.md (v2)
