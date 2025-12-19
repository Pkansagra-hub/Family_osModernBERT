# Decoder Token Embedding Analysis Report

**Date**: December 18, 2025
**Author**: GitHub Copilot (Claude Opus 4.5)
**Purpose**: Document findings from decoder quality investigation and token embedding analysis

---

## Executive Summary

Investigation into decoder output quality issues revealed that **special tokens (BOS/EOS/PAD) have approximately half the embedding magnitude** of standard GPT-2 vocabulary tokens. This stems from a missing initialization method that was called but never implemented in the codebase.

---

## 1. Training Regime Documentation

### 1.1 Stage A+B: ModernBERT with 12 Classification Heads

**Checkpoint**: `outputs/modernbert-v2-for-v3-transfer/checkpoint-18000`

| Property | Value |
|----------|-------|
| Architecture | ModernBERT (22 layers, 768 hidden, 12 heads) |
| Total Parameters | ~155M |
| Capabilities | 12 classification heads (sentiment, NER, safety, etc.) |
| Training Data | Multi-task family domain data |
| Status | **FROZEN** for decoder training |

### 1.2 Stage C v1: Initial Decoder Training

**Checkpoint**: `outputs/ultrabert-gen-decoder-v1`

| Property | Value |
|----------|-------|
| Config | `configs/training/multitask/stage_c_gpt2.yaml` |
| Script | `scripts/train_stage_c.py` |
| Base Decoder | GPT-2 Medium (355M params) |
| Encoder Source | `checkpoint-18000` (frozen, embeddings precomputed) |
| Training Data | `data/counterfactual/merged/` (217,086 samples) |
| Embedding Mode | Precomputed full-sequence embeddings |
| Output Format | `pytorch_model.bin` |

**Key Training Parameters (v1)**:

```yaml
learning_rate: 1.0e-4
num_train_epochs: 10
per_device_train_batch_size: 64
gradient_accumulation_steps: 2
# Effective batch: 128
```

### 1.3 Stage C v3: Subdomain Rebalancing Fine-tune

**Checkpoint**: `outputs/ultrabert-gen-decoder-v3`

| Property | Value |
|----------|-------|
| Config | `configs/training/multitask/stage_c_gpt2_v3.yaml` |
| Script | `scripts/train_stage_c.py` |
| Base Model | `outputs/ultrabert-gen-decoder-v1` (continued training) |
| Encoder Source | `checkpoint-18000` (same as v1) |
| Training Data | `data/counterfactual/training_v2/samples.jsonl` (86K samples) |
| Output Format | `model.safetensors` |

**Key Training Parameters (v3)**:

```yaml
learning_rate: 7.0e-5      # Lower than v1 (was 1e-4)
num_train_epochs: 5        # Fewer epochs
per_device_train_batch_size: 32
gradient_accumulation_steps: 4
dropout: 0.15              # Higher regularization
weight_decay: 0.02         # Stronger L2
warmup_ratio: 0.05         # Faster ramp
```

**Training Stats (from trainer_state.json)**:

- Best checkpoint: step 2000
- Final loss: Started at 52.49, converged to ~2.43
- Epochs completed: ~3.13

---

## 2. Current Status: Decoder Quality Issues

### 2.1 Observed Problems

During quality testing with 18 diverse family scenarios, the decoder exhibited:

| Issue | Example |
|-------|---------|
| **Context drift** | Input about daughter, output mentions son |
| **Sentence fragmentation** | "If you had taken a moment..." ends abruptly |
| **Generic suggestions** | Same "take a deep breath" advice across different contexts |
| **Wrong domain** | Career advice when asked about parenting |

### 2.2 Sample Output Analysis

**Input**: "I'm so frustrated with my daughter's science project - I just did most of it myself last night"

**Output**: "If you had guided your son through the scientific method step-by-step..."

**Problems**:

- ❌ Gender confusion (daughter → son)
- ❌ Generic advice (not specific to the situation)

---

## 3. Code Exploration Findings

### 3.1 File: `src/modeling_studio/models/decoder_gpt2.py`

**Lines 150-169** - Token embedding resize:

```python
# Resize token embeddings to match target vocab size
# GPT-2 has 50257 tokens, ModernBERT tokenizer has 50368
original_vocab_size = self.gpt2.config.vocab_size
if config.vocab_size != original_vocab_size:
    logger.info(
        f"Resizing GPT-2 embeddings: {original_vocab_size} -> {config.vocab_size}"
    )
    self.gpt2.resize_token_embeddings(config.vocab_size)

    # Initialize new token embeddings properly
    # New tokens (50257-50367) are random after resize - fix them
    self._initialize_new_token_embeddings(
        original_vocab_size=original_vocab_size,
        new_vocab_size=config.vocab_size,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        pad_token_id=config.pad_token_id,
    )
```

### 3.2 Critical Bug: Missing Method Implementation

**The method `_initialize_new_token_embeddings()` is CALLED but NEVER DEFINED.**

A grep search confirms:

```
grep "_initialize_new_token_embeddings" src/**/*.py
  → Only 1 match: the call site at line 161
  → No "def _initialize_new_token_embeddings" found anywhere
```

When `resize_token_embeddings()` is called, HuggingFace initializes new tokens with:

- Random normal distribution
- Small std (~0.02)
- Mean ~0.0

These tokens never get proper initialization, so they remain "quiet" compared to GPT-2's learned vocabulary.

### 3.3 Token Configuration

From `configs/training/multitask/stage_c_gpt2_v3.yaml`:

```yaml
decoder:
  vocab_size: 50368
  pad_token_id: 50283
  bos_token_id: 50281
  eos_token_id: 50282
```

**Token ID Mapping**:

| Token | ID | Purpose |
|-------|-----|---------|
| Original GPT-2 vocab | 0-50256 | Standard vocabulary |
| `<\|endoftext\|>` | 50256 | GPT-2's original EOS |
| New tokens | 50257-50280 | ModernBERT additions |
| `[CLS]` → BOS | 50281 | Start of generation |
| `[SEP]` → EOS | 50282 | End of generation |
| `[PAD]` | 50283 | Padding token |
| Reserved | 50284-50367 | Unused |

---

## 4. Empirical Evidence: Embedding Analysis

### 4.1 Embedding Magnitude Comparison

**V3 Checkpoint Analysis** (`outputs/ultrabert-gen-decoder-v3/model.safetensors`):

| Token Range | Norm (mean) | Std | Description |
|-------------|-------------|-----|-------------|
| 0-50256 | **3.69** | 0.40 | Original GPT-2 vocabulary |
| 50257-50367 | **2.09** | 0.01 | New tokens (including special) |
| 50281 (BOS) | **2.04** | 0.064 | Start token |
| 50282 (EOS) | **2.03** | 0.064 | End token |
| 50283 (PAD) | **2.09** | 0.065 | Padding token |
| 50256 (old EOS) | **2.55** | 0.080 | GPT-2's original end token |

**Key Finding**: New tokens have **56% the magnitude** of original GPT-2 embeddings (2.09 vs 3.69).

### 4.2 V1 vs V3 Comparison

| Metric | V1 | V3 | Change |
|--------|-----|-----|--------|
| GPT-2 tokens norm | 3.71 | 3.69 | -0.02 |
| New tokens norm | 2.18 | 2.09 | -0.09 |
| BOS norm | 2.02 | 2.04 | +0.02 |
| V1→V3 BOS diff | - | - | 0.35 |
| V1→V3 EOS diff | - | - | 0.41 |
| V1→V3 random token diff | - | - | 0.45 |

**Interpretation**:

- Special tokens DID update during training (diff ~0.35-0.41)
- But they started from random small values and never caught up to GPT-2 scale
- The magnitude gap persisted through both v1 and v3 training

### 4.3 Impact on Generation

When the decoder generates:

1. **BOS token (50281)** starts with a "quiet" embedding (norm 2.04)
   - GPT-2 attention sees this as a weak signal
   - Less informative context for first generated token

2. **EOS token (50282)** has low magnitude (norm 2.03)
   - Model may not recognize it as a strong stopping signal
   - Could lead to runaway generation or premature stops

3. **Prefix injection** projects encoder outputs (norm ~1.0 after LayerNorm)
   - Combined with weak BOS, the "start" signal is diluted

---

## 5. Root Cause Analysis

### 5.1 The Bug Timeline

```
1. GPT-2 loaded with 50,257 tokens (well-trained embeddings, norm ~3.7)
           ↓
2. resize_token_embeddings(50368) called
           ↓
3. HuggingFace adds 111 new tokens with random init (norm ~2.1)
           ↓
4. _initialize_new_token_embeddings() CALLED but NOT IMPLEMENTED
           ↓
5. Training proceeds with mismatched embedding scales
           ↓
6. Gradients update all tokens, but new tokens stay relatively "quiet"
           ↓
7. Decoder struggles with start/stop signals
```

### 5.2 Why Training Didn't Fix It

- **Learning rate**: At 1e-4 to 7e-5, gradients would need many epochs to scale up embeddings by 76%
- **Gradient flow**: Loss primarily from language modeling, not embedding magnitude
- **Implicit regularization**: Weight decay (0.01-0.02) slightly shrinks all embeddings
- **Frozen encoder**: No gradient signal from encoder to fix token alignment

---

## 6. Remediation Options

### Option A: Retrain V4 with Proper Initialization (Recommended)

**Effort**: 2-3 hours on A100, 4-6 hours on T4/V100

1. Implement `_initialize_new_token_embeddings()`:

   ```python
   def _initialize_new_token_embeddings(
       self,
       original_vocab_size: int,
       new_vocab_size: int,
       bos_token_id: int,
       eos_token_id: int,
       pad_token_id: int,
   ) -> None:
       """Initialize new token embeddings to match GPT-2 scale."""
       with torch.no_grad():
           wte = self.gpt2.transformer.wte.weight

           # Compute mean embedding from original GPT-2 vocab
           old_mean = wte[:original_vocab_size].mean(dim=0)

           # Initialize all new tokens to this mean
           num_new = new_vocab_size - original_vocab_size
           wte[original_vocab_size:].copy_(
               old_mean.unsqueeze(0).expand(num_new, -1)
           )

           # Special handling for BOS/EOS/PAD
           # Copy from GPT-2's endoftext token (similar function)
           endoftext_embed = wte[50256].clone()
           wte[bos_token_id].copy_(endoftext_embed)
           wte[eos_token_id].copy_(endoftext_embed)
           wte[pad_token_id].zero_()  # PAD should be zero

       logger.info(f"Initialized {num_new} new token embeddings")
   ```

2. Create `stage_c_gpt2_v4.yaml` starting from **fresh GPT-2**, not v1
3. Train on merged dataset (217K samples)

**Pros**: Clean fix, proper initialization from start
**Cons**: Full retraining required

### Option B: Post-hoc Embedding Surgery (Experimental)

**Effort**: 30 minutes, no retraining

1. Load V3 checkpoint
2. Scale up new token embeddings:

   ```python
   scale_factor = 3.69 / 2.09  # ~1.76
   wte[50257:] *= scale_factor
   ```

3. Save modified checkpoint

**Pros**: Immediate, no GPU time
**Cons**:

- Model learned with wrong scale, may behave unexpectedly
- Untested approach
- Attention patterns may be misaligned

### Option C: Fine-tune V3 with Higher LR on Special Tokens Only

**Effort**: 1 hour on A100

1. Freeze all embeddings except 50257-50367
2. Fine-tune with higher LR (1e-3) on same data
3. Let special tokens "catch up"

**Pros**: Faster than full retrain
**Cons**: May not fully resolve attention pattern issues

---

## 7. Recommendation

**Primary**: Implement Option A (Retrain V4)

The initialization bug affects the fundamental representation of start/stop signals. A clean retrain with proper initialization will:

- Give special tokens correct scale from epoch 0
- Allow attention patterns to form around properly-scaled tokens
- Produce higher quality generations

**Secondary**: If GPU time is limited, try Option B as a quick experiment to validate the diagnosis before committing to full retraining.

---

## 8. Files to Modify

| File | Change Required |
|------|-----------------|
| `src/modeling_studio/models/decoder_gpt2.py` | Add `_initialize_new_token_embeddings()` method |
| `familyos_ultrabert/models/decoder_gpt2.py` | Same fix for package version |
| `configs/training/multitask/stage_c_gpt2_v4.yaml` | New config starting from fresh GPT-2 |

---

## Appendix: Raw Data

### A.1 V3 Embedding Statistics (Full Output)

```
Token embeddings shape: torch.Size([50368, 1024])
Token 50281 (BOS) mean: 0.0017, std: 0.0639
Token 50282 (EOS) mean: 0.0021, std: 0.0636
Token 50283 (PAD) mean: 0.0014, std: 0.0654
Original GPT-2 tokens (0-50256) mean: 0.0015, std: 0.1160
New tokens (50257-50367) mean: 0.0014, std: 0.0654
```

### A.2 V1 vs V3 Diff Output

```
=== V1 Checkpoint ===
GPT-2 tokens (0-50256) norm: mean=3.71
New tokens (50257-50367) norm: mean=2.18
BOS (50281) norm: 2.02

=== V3 Checkpoint ===
GPT-2 tokens (0-50256) norm: mean=3.69
New tokens (50257-50367) norm: mean=2.09
BOS (50281) norm: 2.04

=== V1 vs V3 difference ===
BOS diff norm: 0.3522
EOS diff norm: 0.4070
PAD diff norm: 0.3357
Random GPT-2 token (1000) diff: 0.4510
```

### A.3 Training Config Comparison

| Parameter | V1 | V3 |
|-----------|-----|-----|
| Base model | Fresh GPT-2 Medium | ultrabert-gen-decoder-v1 |
| Learning rate | 1.0e-4 | 7.0e-5 |
| Epochs | 10 | 5 |
| Batch size | 64 | 32 |
| Grad accum | 2 | 4 |
| Dropout | 0.1 | 0.15 |
| Weight decay | 0.01 | 0.02 |
| Warmup | 10% | 5% |
| Data samples | 217K (merged) | 86K (training_v2) |

---

# Project Plan: Decoder V4 Training

**Target**: Train a production-quality counterfactual decoder with proper token initialization

---

## Milestone 1: Data Integrity and Generation

**Goal**: Prepare a high-quality, balanced dataset that addresses identified gaps

### Epic 1.1: Data Audit and Gap Analysis

#### Issue 1.1.1: Analyze Current Data Distribution

**Status**: DONE (findings below)

| Dataset | Samples | Location |
|---------|---------|----------|
| Synthetic | 67,517 | `data/counterfactual/synthetic/` |
| Merged | 217,086 | `data/counterfactual/merged/` |
| Training V2 | 86,000 | `data/counterfactual/training_v2/` |

**Domain Distribution (Merged)**:

| Domain | % | Count |
|--------|---|-------|
| health | 11.2% | 24,292 |
| relationship | 11.1% | 24,062 |
| parenting | 10.1% | 22,005 |
| emotions | 9.1% | 19,670 |
| communication | 9.1% | 19,656 |
| caregiving | 8.8% | 19,147 |
| time_management | 8.8% | 19,010 |
| life_events | 3.2% | 6,952 |

#### Issue 1.1.2: Identify Critical Gaps

**Status**: DONE

| Scenario | Training Samples | Status |
|----------|-----------------|--------|
| Career/promotion | 3,788 | OK |
| Moving/relocation | 5,436 | OK |
| Science project | 5,123 | OK |
| Playground injury | 1,392 | OK |
| School choice | 51 | LOW |
| Driving test | 7 | CRITICAL GAP |

**Valence Distribution Problem**:

- 99.7% negative valence
- 0.3% neutral valence
- 0% positive valence

This creates a model that only knows how to respond to negative scenarios.

---

### Epic 1.2: Generate Missing Data

#### Issue 1.2.1: Generate Driving Test Scenarios

**Priority**: P0 (CRITICAL)
**Target**: 500 samples

**Files to modify**:

- `scripts/agents/counterfactual_data_generator.py`

**Action**: Add driving test subdomain to life_events domain with examples:

```python
DOMAINS = {
    "life_events": {
        "subdomains": ["driving_test", "graduations", "promotions", ...],
        "examples": {
            "driving_test": [
                "My son failed his driving test for the third time...",
                "I panicked when my daughter almost hit a curb during practice...",
            ]
        }
    }
}
```

**Command**:

```bash
python scripts/agents/counterfactual_data_generator.py generate \
    --count 500 \
    --domain life_events \
    --subdomain driving_test \
    --vertex-ai
```

#### Issue 1.2.2: Generate School Choice Scenarios

**Priority**: P1
**Target**: 500 samples

**Command**:

```bash
python scripts/agents/counterfactual_data_generator.py generate \
    --count 500 \
    --domain parenting \
    --subdomain school_choice \
    --vertex-ai
```

#### Issue 1.2.3: Generate Neutral/Positive Valence Scenarios

**Priority**: P1
**Target**: 10,000 samples (5% of total)

Currently 99.7% negative. Need neutral scenarios like:

- "We're considering whether to move to a new city..."
- "My daughter is choosing between two summer camps..."

**Files to modify**:

- `scripts/agents/counterfactual_data_generator.py` - Add `outcome_valence` parameter
- Update `SYSTEM_PROMPT` to include neutral scenario generation

---

### Epic 1.3: Data Preparation Pipeline

#### Issue 1.3.1: Merge New Data with Existing

**Files involved**:

- `scripts/agents/clean_synthetic_data.py`
- `scripts/agents/remove_duplicates.py`

**Steps**:

1. Run deduplication: `python scripts/agents/remove_duplicates.py --input-dir data/counterfactual/synthetic`
2. Validate schema: `python scripts/agents/validate_unified_schema.py`
3. Merge: Combine all shards into unified training set

#### Issue 1.3.2: Regenerate Encoder Embeddings

**Files involved**:

- `scripts/agents/prepare_decoder_training_data.py`

After adding new data, regenerate embeddings:

```bash
python scripts/agents/prepare_decoder_training_data.py \
    --input-dir data/counterfactual/merged \
    --output-dir data/counterfactual/training_v3 \
    --full-sequence \
    --batch-size 32 \
    --model-path outputs/modernbert-v2-for-v3-transfer/checkpoint-18000
```

**Output structure**:

```
data/counterfactual/training_v3/
├── samples.jsonl              # Text data
├── sequence_embeddings.h5     # Full sequence embeddings (768-dim)
├── manifest.json              # Metadata
└── train_val_split.json       # 95/5 split
```

#### Issue 1.3.3: Validate Training Data Quality

**Create new script**: `scripts/validate_training_data.py`

Checks:

- [ ] All samples have embeddings
- [ ] No NaN/Inf in embeddings
- [ ] Embedding norms are in expected range
- [ ] Output text matches expected format ("If you had...")
- [ ] Domain/subdomain coverage report

---

*Milestone 1 Deliverables*:

- [ ] 500+ driving test samples
- [ ] 500+ school choice samples
- [ ] 10,000+ neutral/positive valence samples
- [ ] `data/counterfactual/training_v3/` with embeddings
- [ ] Data validation report

---

## Milestone 2: Model Codebase Check and Updates

**Goal**: Fix the token embedding initialization bug and ensure SOTA training/eval setup

### Epic 2.1: Fix Token Embedding Initialization

#### Issue 2.1.1: Implement Missing `_initialize_new_token_embeddings()` Method

**Priority**: P0 (CRITICAL - ROOT CAUSE OF QUALITY ISSUES)

**Files to modify**:

1. `src/modeling_studio/models/decoder_gpt2.py`
2. `familyos_ultrabert/models/decoder_gpt2.py`

**Current state** (line 161 of `src/modeling_studio/models/decoder_gpt2.py`):

```python
# Method is CALLED but NOT IMPLEMENTED
self._initialize_new_token_embeddings(
    original_vocab_size=original_vocab_size,
    new_vocab_size=config.vocab_size,
    bos_token_id=config.bos_token_id,
    eos_token_id=config.eos_token_id,
    pad_token_id=config.pad_token_id,
)
```

**Implementation to add**:

```python
def _initialize_new_token_embeddings(
    self,
    original_vocab_size: int,
    new_vocab_size: int,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
) -> None:
    """
    Initialize new token embeddings to match GPT-2's learned embedding scale.

    Problem: resize_token_embeddings() adds new tokens with random normal init
    (std ~0.02, norm ~2.1) while GPT-2's learned tokens have norm ~3.7.
    This 56% magnitude gap causes weak BOS/EOS signals.

    Solution: Initialize new tokens to mean of existing GPT-2 embeddings,
    with special handling for BOS (copy from endoftext) and PAD (zero).
    """
    with torch.no_grad():
        wte = self.gpt2.transformer.wte.weight

        # Compute statistics from original GPT-2 vocabulary
        original_embeddings = wte[:original_vocab_size]
        old_mean = original_embeddings.mean(dim=0)
        old_std = original_embeddings.std()

        logger.info(f"Original GPT-2 embeddings: mean norm={original_embeddings.norm(dim=1).mean():.2f}")

        # Initialize all new tokens to the mean embedding
        # This gives them the correct magnitude from the start
        num_new = new_vocab_size - original_vocab_size
        wte[original_vocab_size:].copy_(
            old_mean.unsqueeze(0).expand(num_new, -1)
        )

        # Special token initialization:
        # BOS/EOS: Copy from GPT-2's <|endoftext|> token (ID 50256)
        # This token has learned start/stop semantics
        endoftext_embed = wte[50256].clone()
        wte[bos_token_id].copy_(endoftext_embed)
        wte[eos_token_id].copy_(endoftext_embed)

        # PAD: Zero vector (should not contribute to attention)
        wte[pad_token_id].zero_()

        logger.info(
            f"Initialized {num_new} new token embeddings: "
            f"BOS={bos_token_id}, EOS={eos_token_id}, PAD={pad_token_id}"
        )
        logger.info(f"New embeddings norm: {wte[original_vocab_size:].norm(dim=1).mean():.2f}")
```

**Insert location**: After `_freeze_layers()` method, before `forward()` method (~line 220)

---

### Epic 2.2: Update GPT-2 Config with Special Tokens

#### Issue 2.2.1: Ensure GPT-2 Config Has Correct Token IDs

**Status**: ALREADY IMPLEMENTED (lines 170-173)

```python
# Update GPT-2 config with our special token IDs
self.gpt2.config.bos_token_id = config.bos_token_id
self.gpt2.config.eos_token_id = config.eos_token_id
self.gpt2.config.pad_token_id = config.pad_token_id
```

---

### Epic 2.3: Sync Package Code with Source

#### Issue 2.3.1: Update `familyos_ultrabert/models/decoder_gpt2.py`

**Priority**: P1

The package version must match the source version. After implementing Issue 2.1.1:

1. Copy the new `_initialize_new_token_embeddings()` method
2. Ensure same token IDs are used
3. Run package tests

---

### Epic 2.4: SOTA Evaluation Setup

#### Issue 2.4.1: Add Generation Quality Metrics

**Files to create/modify**:

- `src/modeling_studio/evaluation/counterfactual_metrics.py`

**Metrics to implement**:

1. **BLEU/ROUGE**: Compare generated vs reference counterfactuals
2. **Diversity**: Unique n-grams / total n-grams
3. **Completion Rate**: % of outputs that don't cut off mid-sentence
4. **Context Fidelity**: Does output mention same entities as input?
5. **Format Adherence**: Does output start with "If you had..."?

#### Issue 2.4.2: Add Eval Callbacks to Training

**Files to modify**:

- `scripts/train_stage_c.py`
- `src/modeling_studio/trainers/decoder_trainer.py`

Add generation-based evaluation during training:

```python
class GenerationEvalCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        # Generate on 100 samples
        # Compute quality metrics
        # Log to wandb/tensorboard
```

---

### Epic 2.5: Training Script Validation

#### Issue 2.5.1: Verify Freeze Logic

**File**: `scripts/train_stage_c.py`

Ensure encoder and 12 heads are frozen:

```python
# Expected in config:
model:
  freeze_encoder: true
  freeze_existing_heads: true
```

Verify in code that only decoder parameters have `requires_grad=True`.

#### Issue 2.5.2: Verify Loss Computation

**File**: `src/modeling_studio/trainers/decoder_collator.py`

Ensure:

- Labels are shifted correctly for causal LM
- Padding tokens have label -100 (IGNORE_INDEX)
- BOS token is not included in loss

---

*Milestone 2 Deliverables*:

- [ ] `_initialize_new_token_embeddings()` implemented in `src/`
- [ ] Same method implemented in `familyos_ultrabert/`
- [ ] Generation quality metrics added
- [ ] Freeze logic verified
- [ ] Loss computation verified

---

## Milestone 3: End-to-End Testing

**Goal**: Verify all components work together before committing to full training

### Epic 3.1: Unit Tests for New Code

#### Issue 3.1.1: Test Token Embedding Initialization

**File to create**: `tests/test_decoder_token_init.py`

```python
def test_new_token_embeddings_have_correct_scale():
    """New tokens should have same norm as original GPT-2 tokens."""
    config = GPT2DecoderConfig(vocab_size=50368)
    decoder = GPT2DecoderHead(config, encoder_hidden_size=768)

    wte = decoder.gpt2.transformer.wte.weight

    # Original GPT-2 tokens (0-50256)
    original_norm = wte[:50257].norm(dim=1).mean()

    # New tokens (50257-50367)
    new_norm = wte[50257:].norm(dim=1).mean()

    # Should be within 10% of original
    assert abs(new_norm - original_norm) / original_norm < 0.1

def test_bos_eos_initialized_from_endoftext():
    """BOS/EOS should be copies of GPT-2's endoftext token."""
    config = GPT2DecoderConfig(vocab_size=50368, bos_token_id=50281, eos_token_id=50282)
    decoder = GPT2DecoderHead(config, encoder_hidden_size=768)

    wte = decoder.gpt2.transformer.wte.weight

    endoftext = wte[50256]
    bos = wte[50281]
    eos = wte[50282]

    assert torch.allclose(bos, endoftext)
    assert torch.allclose(eos, endoftext)

def test_pad_token_is_zero():
    """PAD token should be zero vector."""
    config = GPT2DecoderConfig(vocab_size=50368, pad_token_id=50283)
    decoder = GPT2DecoderHead(config, encoder_hidden_size=768)

    pad = decoder.gpt2.transformer.wte.weight[50283]
    assert pad.abs().max() < 1e-6
```

---

### Epic 3.2: Data Pipeline Tests

#### Issue 3.2.1: Test Dataset Loading

**File to create**: `tests/test_counterfactual_dataset.py`

```python
def test_dataset_loads_embeddings():
    """Verify dataset can load precomputed embeddings."""
    dataset = CounterfactualDataset(
        data_dir="data/counterfactual/training_v3",
        tokenizer=tokenizer,
        mode="precomputed",
        split="train",
        full_sequence=True,
    )

    assert len(dataset) > 0
    sample = dataset[0]

    assert "encoder_embeddings" in sample
    assert "labels" in sample
    assert sample["encoder_embeddings"].shape[-1] == 768

def test_dataset_embeddings_not_nan():
    """No NaN values in embeddings."""
    dataset = CounterfactualDataset(...)

    for i in range(min(100, len(dataset))):
        sample = dataset[i]
        assert not torch.isnan(sample["encoder_embeddings"]).any()
```

#### Issue 3.2.2: Test Data Collator

**File**: `tests/test_decoder_collator.py`

```python
def test_collator_creates_correct_batch():
    """Collator should create properly padded batches."""
    collator = DecoderDataCollator(tokenizer, full_sequence=True)

    batch = collator([dataset[0], dataset[1]])

    assert "encoder_hidden_states" in batch
    assert "labels" in batch
    assert batch["labels"].shape[0] == 2  # batch size
```

---

### Epic 3.3: Training Dry Run

#### Issue 3.3.1: Create V4 Config File

**File to create**: `configs/training/multitask/stage_c_gpt2_v4.yaml`

Key differences from V3:

```yaml
# =============================================================================
# Stage C v4: Fresh GPT-2 with Proper Token Initialization
# =============================================================================
# PURPOSE: Train from fresh GPT-2 (not from V1) with properly initialized
#          BOS/EOS/PAD tokens. This fixes the 56% embedding magnitude gap.
#
# CRITICAL CHANGE: Starting from fresh GPT-2 Medium, NOT from V1 checkpoint
# =============================================================================

model:
  # Load ENCODER from Stage B (12 heads)
  checkpoint_path: "outputs/modernbert-v2-for-v3-transfer/checkpoint-18000"

  freeze_encoder: true
  freeze_existing_heads: true

  torch_dtype: bfloat16
  use_flash_attention_2: true

decoder:
  type: "gpt2"
  gpt2_model_name: "gpt2-medium"  # Fresh GPT-2, not from V1

  projection_hidden_size: 1024
  prefix_projection_layers: 1
  use_prefix_injection: true

  freeze_layers: 0

  max_position_embeddings: 512
  vocab_size: 50368

  pad_token_id: 50283
  bos_token_id: 50281
  eos_token_id: 50282

  dropout: 0.1

  generation_max_length: 128
  temperature: 1.0
  top_k: 50
  top_p: 0.9
  repetition_penalty: 1.2

data:
  train_path: "data/counterfactual/training_v3"  # New balanced dataset

  embeddings_mode: "precomputed"
  full_sequence: true

  max_input_length: 256
  max_output_length: 256

  num_workers: 8
  pin_memory: true
  prefetch_factor: 8
  persistent_workers: true
  load_to_ram: true

training:
  output_dir: "outputs/ultrabert-gen-decoder-v4"
  checkpoint_dir: "checkpoints/ultrabert-gen-decoder-v4"

  num_train_epochs: 7
  max_steps: -1

  per_device_train_batch_size: 64
  per_device_eval_batch_size: 128
  gradient_accumulation_steps: 2

  learning_rate: 1.0e-4
  weight_decay: 0.01
  max_grad_norm: 1.0

  lr_scheduler_type: "cosine"
  warmup_ratio: 0.05

  optim: "adamw_torch_fused"

  bf16: true
  fp16: false

  gradient_checkpointing: false
  dataloader_pin_memory: true
  torch_compile: false
  tf32: true

  save_strategy: "steps"
  save_steps: 500
  save_total_limit: 5
  save_safetensors: true

  eval_strategy: "steps"
  eval_steps: 500

  logging_strategy: "steps"
  logging_steps: 10
  logging_first_step: true

  load_best_model_at_end: false
  metric_for_best_model: "eval_loss"
  greater_is_better: false

  report_to:
    - "tensorboard"

  seed: 42

  dataloader_num_workers: 8
  dataloader_drop_last: true
  dataloader_prefetch_factor: 8
  remove_unused_columns: false

  resume_from_checkpoint: null
```

#### Issue 3.3.2: Run Debug Training (10 steps)

**Command**:

```bash
python scripts/train_stage_c.py \
    --config configs/training/multitask/stage_c_gpt2_v4.yaml \
    --debug
```

**Verify**:

- [ ] Model loads without errors
- [ ] Data loads without errors
- [ ] Forward pass succeeds
- [ ] Loss decreases
- [ ] Checkpoint saves correctly

#### Issue 3.3.3: Verify Token Embeddings After Init

**Command** (add to debug output):

```python
# After model init, before training
wte = model.heads["counterfactual"].gpt2.transformer.wte.weight
print(f"GPT-2 original tokens norm: {wte[:50257].norm(dim=1).mean():.2f}")
print(f"New tokens norm: {wte[50257:].norm(dim=1).mean():.2f}")
print(f"BOS norm: {wte[50281].norm():.2f}")
print(f"EOS norm: {wte[50282].norm():.2f}")
print(f"PAD norm: {wte[50283].norm():.2f}")

# Expected: All norms should be ~3.7 (except PAD which should be ~0)
```

---

### Epic 3.4: Generation Quality Test

#### Issue 3.4.1: Test Generation on Sample Inputs

After debug training, run generation:

```python
test_inputs = [
    "I yelled at my daughter for spilling milk",
    "I forgot my son's school play",
    "My daughter failed her driving test for the third time",
]

for text in test_inputs:
    output = model.generate(text)
    print(f"Input: {text}")
    print(f"Output: {output}")
    print(f"Starts with 'If you had': {output.startswith('If you had')}")
    print(f"Ends with punctuation: {output[-1] in '.!?'}")
    print()
```

**Success criteria**:

- Output starts with "If you had..."
- Output ends with complete sentence (., !, ?)
- Output mentions same subject as input (daughter/son)
- No mid-sentence cutoff

---

*Milestone 3 Deliverables*:

- [ ] Unit tests for token initialization
- [ ] Data pipeline tests
- [ ] `stage_c_gpt2_v4.yaml` config created
- [ ] Debug training runs successfully
- [ ] Token embeddings verified after init
- [ ] Generation quality verified

---

## Milestone 4: Train Model V4

**Goal**: Train production decoder using existing Colab infrastructure

### Epic 4.1: Colab Notebook Update

#### Issue 4.1.1: Update Config Path in Colab Notebook

**File**: `notebooks/train_stage_c_colab.ipynb`

The notebook already has the training infrastructure. Only need to change config:

**Current** (in cell that sets config):

```python
CONFIG_FILE = "configs/training/multitask/stage_c_gpt2.yaml"
```

**Change to**:

```python
CONFIG_FILE = "configs/training/multitask/stage_c_gpt2_v4.yaml"
```

#### Issue 4.1.2: Verify Colab Environment Setup

**File**: `notebooks/train_stage_c_colab.ipynb`

Cells to verify:

1. **Cell 3**: Flash Attention install (pre-built wheel)
2. **Cell 5**: GPU check (`nvidia-smi`)
3. **Cell 7**: Repository clone
4. **Cell 8**: Dependencies install
5. **Cell 9**: Google Drive mount for persistence

#### Issue 4.1.3: Update Package Wheel Version

**Current**:

```python
!pip install -q https://github.com/Pkansagra-hub/Family_osModernBERT/releases/download/v2.2.1/familyos_ultrabert-2.2.1-py3-none-any.whl
```

**After fixing token initialization**, build and release new wheel (v3.0.2 or v3.1.0):

```python
!pip install -q https://github.com/Pkansagra-hub/Family_osModernBERT/releases/download/v3.1.0/familyos_ultrabert-3.1.0-py3-none-any.whl
```

---

### Epic 4.2: Pre-Training Checklist

#### Issue 4.2.1: Upload Training Data to Google Drive

**Steps**:

1. Zip `data/counterfactual/training_v3/` locally
2. Upload to Google Drive: `MyDrive/ultrabert_training/data/`
3. In Colab, extract to local path

```python
# In Colab
!cp /content/drive/MyDrive/ultrabert_training/data/training_v3.zip .
!unzip -q training_v3.zip -d data/counterfactual/
```

#### Issue 4.2.2: Upload Encoder Checkpoint

**Steps**:

1. Zip `outputs/modernbert-v2-for-v3-transfer/checkpoint-18000/`
2. Upload to Google Drive
3. Extract in Colab

```python
!cp /content/drive/MyDrive/ultrabert_training/checkpoints/checkpoint-18000.zip .
!unzip -q checkpoint-18000.zip -d outputs/modernbert-v2-for-v3-transfer/
```

#### Issue 4.2.3: Verify All Files Present

```python
# Pre-flight check
import os
from pathlib import Path

required_files = [
    "data/counterfactual/training_v3/samples.jsonl",
    "data/counterfactual/training_v3/sequence_embeddings.h5",
    "data/counterfactual/training_v3/train_val_split.json",
    "outputs/modernbert-v2-for-v3-transfer/checkpoint-18000/model.safetensors",
    "configs/training/multitask/stage_c_gpt2_v4.yaml",
]

for f in required_files:
    exists = Path(f).exists()
    print(f"{'[OK]' if exists else '[MISSING]'} {f}")
```

---

### Epic 4.3: Execute Training

#### Issue 4.3.1: Run Training

**Command** (in Colab cell):

```python
!python scripts/train_stage_c.py \
    --config configs/training/multitask/stage_c_gpt2_v4.yaml \
    --auto_resume
```

**Expected duration**:

- A100 80GB: 2-3 hours
- T4/V100 16GB: 4-6 hours

**Checkpoint saves**:

- Every 500 steps (~30 min)
- Saved to Google Drive for persistence

#### Issue 4.3.2: Monitor Training

**Metrics to watch**:

- `train_loss`: Should decrease from ~50 to ~2-3
- `eval_loss`: Should track train_loss closely
- `grad_norm`: Should stay < 10 after warmup

**TensorBoard** (in separate Colab cell):

```python
%load_ext tensorboard
%tensorboard --logdir checkpoints/ultrabert-gen-decoder-v4
```

---

### Epic 4.4: Post-Training Validation

#### Issue 4.4.1: Verify Token Embeddings in Trained Model

```python
from safetensors.torch import load_file

state_dict = load_file("outputs/ultrabert-gen-decoder-v4/model.safetensors")
wte = state_dict[[k for k in state_dict if 'wte.weight' in k][0]]

print(f"GPT-2 tokens (0-50256) norm: {wte[:50257].norm(dim=1).mean():.2f}")
print(f"New tokens (50257-50367) norm: {wte[50257:].norm(dim=1).mean():.2f}")
print(f"BOS norm: {wte[50281].norm():.2f}")

# V4 should show:
# - GPT-2 tokens: ~3.7 (same as before)
# - New tokens: ~3.7 (NOT 2.1 like V3!)
# - BOS: ~3.7 (properly initialized)
```

#### Issue 4.4.2: Run Quality Tests

```python
from familyos_ultrabert import Client

client = Client(load_decoder=True)

test_cases = [
    "I yelled at my daughter for spilling milk this morning",
    "My son failed his driving test for the third time",
    "I forgot to pick up my daughter from school",
    "We chose the prestigious school over staying with friends",
]

for text in test_cases:
    result = client.generate_counterfactual(text)
    print(f"Input: {text}")
    print(f"Output: {result}")
    print(f"Format OK: {result.startswith('If you had')}")
    print(f"Complete: {result[-1] in '.!?'}")
    print()
```

**Success criteria**:

- [ ] All outputs start with "If you had..."
- [ ] All outputs end with complete sentences
- [ ] No gender/entity confusion
- [ ] Driving test scenario produces relevant output
- [ ] School choice scenario produces relevant output

#### Issue 4.4.3: Build and Release New Package

**Steps**:

1. Copy trained decoder weights to package:

   ```bash
   cp outputs/ultrabert-gen-decoder-v4/model.safetensors familyos_ultrabert/weights/decoder/
   ```

2. Update version in `familyos_ultrabert/pyproject.toml`:

   ```toml
   version = "3.1.0"
   ```

3. Build wheel:

   ```bash
   cd familyos_ultrabert
   python -m build
   ```

4. Create GitHub release with new wheel

---

### Epic 4.5: Documentation Update

#### Issue 4.5.1: Update RELEASE_NOTES.md

```markdown
## v3.1.0 (December 2025)

### Bug Fixes
- **CRITICAL**: Fixed token embedding initialization bug where BOS/EOS/PAD tokens
  had 56% of the magnitude of normal GPT-2 tokens, causing weak start/stop signals
  and degraded generation quality.

### Improvements
- Added balanced training data covering driving tests, school choice scenarios
- Added neutral/positive valence scenarios (previously 99.7% negative)
- Improved generation completion rate (no more mid-sentence cutoffs)

### Training
- V4 decoder trained from fresh GPT-2 Medium with proper token initialization
- 7 epochs on ~230K balanced samples
- Final loss: ~2.3
```

#### Issue 4.5.2: Update README.md

Add note about v3.1.0 fixing generation quality issues.

---

*Milestone 4 Deliverables*:

- [ ] Colab notebook updated with V4 config
- [ ] Training data uploaded to Google Drive
- [ ] V4 model trained successfully
- [ ] Token embeddings verified (norm ~3.7 for all)
- [ ] Quality tests pass
- [ ] Package v3.1.0 built and released
- [ ] Documentation updated

---

## Summary: Complete Issue List

| # | Issue | Priority | Status |
|---|-------|----------|--------|
| 1.1.1 | Analyze current data distribution | P1 | DONE |
| 1.1.2 | Identify critical gaps | P0 | DONE |
| 1.2.1 | Generate driving test scenarios | P0 | TODO |
| 1.2.2 | Generate school choice scenarios | P1 | TODO |
| 1.2.3 | Generate neutral/positive scenarios | P1 | TODO |
| 1.3.1 | Merge new data | P1 | TODO |
| 1.3.2 | Regenerate embeddings | P1 | TODO |
| 1.3.3 | Validate training data | P1 | TODO |
| 2.1.1 | Implement `_initialize_new_token_embeddings()` | P0 | TODO |
| 2.3.1 | Sync package code | P1 | TODO |
| 2.4.1 | Add generation quality metrics | P2 | TODO |
| 2.4.2 | Add eval callbacks | P2 | TODO |
| 2.5.1 | Verify freeze logic | P1 | TODO |
| 2.5.2 | Verify loss computation | P1 | TODO |
| 3.1.1 | Test token initialization | P1 | TODO |
| 3.2.1 | Test dataset loading | P1 | TODO |
| 3.2.2 | Test data collator | P1 | TODO |
| 3.3.1 | Create V4 config | P0 | TODO |
| 3.3.2 | Run debug training | P0 | TODO |
| 3.3.3 | Verify token embeddings | P0 | TODO |
| 3.4.1 | Test generation quality | P1 | TODO |
| 4.1.1 | Update Colab config path | P0 | TODO |
| 4.1.2 | Verify Colab environment | P1 | TODO |
| 4.1.3 | Update package wheel | P1 | TODO |
| 4.2.1 | Upload training data | P0 | TODO |
| 4.2.2 | Upload encoder checkpoint | P0 | TODO |
| 4.2.3 | Verify all files present | P0 | TODO |
| 4.3.1 | Run training | P0 | TODO |
| 4.3.2 | Monitor training | P1 | TODO |
| 4.4.1 | Verify trained embeddings | P0 | TODO |
| 4.4.2 | Run quality tests | P0 | TODO |
| 4.4.3 | Build and release package | P1 | TODO |
| 4.5.1 | Update RELEASE_NOTES | P2 | TODO |
| 4.5.2 | Update README | P2 | TODO |

---

## Estimated Timeline

| Milestone | Effort | Dependencies |
|-----------|--------|--------------|
| M1: Data | 4-6 hours | Vertex AI/OpenRouter API |
| M2: Code | 2-3 hours | None |
| M3: Testing | 2-3 hours | M1, M2 |
| M4: Training | 4-8 hours | M1, M2, M3 + GPU |

**Total**: 12-20 hours

---

*End of Project Plan*
