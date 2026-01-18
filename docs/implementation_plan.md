---

# 🎯 Constitutional Training Enhancement - Implementation Plan

## Executive Summary

**Current State:** Constitution/cultural context exists only in metadata, not used during training. Constitution steering is only applied at inference time via logits processors.

**Goal:** Make constitutional context a first-class citizen throughout the training pipeline, enabling explicit conditioning and better scenario grounding.

---

## 📁 Files That Need Updates

### Data Layer
| File | Purpose | Changes Needed |
|------|---------|----------------|
| prepare_decoder_training_data.py | Preprocessing script | Add explicit constitution field extraction |
| counterfactual_dataset.py | Dataset class | Add constitution field to `__getitem__` output |
| decoder_collator.py | Batch collation | Handle constitution tensors/IDs |

### Model Layer
| File | Purpose | Changes Needed |
|------|---------|----------------|
| decoder_gpt2.py | GPT-2 decoder head | Add constitutional conditioning (prefix/embedding) |
| decoder_gpt2_config.py | Decoder config | Add constitution config options |

### Training Layer
| File | Purpose | Changes Needed |
|------|---------|----------------|
| train_stage_c.py | Training script | Pass constitution to model, add aux loss |
| decoder_trainer.py | Trainer class | Handle constitution loss, logging |
| stage_c_gpt2_v4.yaml | Training config | Add constitution training options |

### Evaluation Layer
| File | Purpose | Changes Needed |
|------|---------|----------------|
| decoder_metrics.py | Metrics | Add Constitution Adherence Score (CAS) |
| train_v3_colab.ipynb | Colab notebook | Update for constitution training |

### Data Files
| File | Purpose | Changes Needed |
|------|---------|----------------|
| `data/counterfactual/training_jsonl/shard_*.jsonl` | Training data | Augment with explicit constitution field |
| constitution_schemas_v2.json | Constitution schemas | Add training-compatible embeddings/IDs |

---

## 🏔️ Milestone 1: Data Pipeline Enhancement

### Epic 1.1: Explicit Constitution Field in Data

#### Issue 1.1.1: Add Constitution Extraction to Preprocessing
**File:** prepare_decoder_training_data.py
**Changes:**
```python
# In load_counterfactual_samples(), add:
- Extract constitution from metadata.cultural_context
- Map to canonical constitution ID (gentle_parenting, traditional_strict, indian_joint_family, etc.)
- Add "constitution" field to each sample
- Add "constitution_id" (integer) for embedding lookup
- Validate: log warning if constitution is missing
```

#### Issue 1.1.2: Constitution ID Mapping Registry
**New File:** `src/modeling_studio/data/constitution_registry.py`
**Content:**
```python
CONSTITUTION_TO_ID = {
    "universal": 0,
    "gentle_parenting": 1,
    "traditional_strict": 2,
    "indian_joint_family": 3,
    "western_nuclear": 4,
    # ... (load from constitution_schemas_v2.json)
}
```

#### Issue 1.1.3: Update Dataset to Include Constitution
**File:** counterfactual_dataset.py
**Changes:**
```python
# In __getitem__(), add:
return {
    ...existing fields...,
    "constitution_id": sample.get("constitution_id", 0),
    "constitution_text": sample.get("constitution", ""),  # For prefix injection
}
```

#### Issue 1.1.4: Update Collator for Constitution Batching
**File:** decoder_collator.py
**Changes:**
```python
# In __call__(), add:
- Extract constitution_ids from features
- Stack into tensor: constitution_ids = torch.stack([f["constitution_id"] for f in features])
- Return in batch dict
```

---

## 🏔️ Milestone 2: Model Architecture Enhancement

### Epic 2.1: Constitutional Conditioning in Decoder

#### Issue 2.1.1: Add Constitution Embedding Layer
**File:** decoder_gpt2.py
**Changes:**
```python
# In GPT2DecoderHead.__init__(), add:
num_constitutions = getattr(config, 'num_constitutions', 10)
self.constitution_embeddings = nn.Embedding(num_constitutions, self.gpt2_hidden_size)

# Initialization: learn distinct constitution representations
```

#### Issue 2.1.2: Constitution Prefix Injection
**File:** decoder_gpt2.py
**Changes:**
```python
# In forward(), modify prefix injection:
def _prepare_prefix_with_constitution(
    self,
    encoder_hidden_states,
    constitution_ids,
):
    # Get constitution embedding
    const_emb = self.constitution_embeddings(constitution_ids)  # [B, hidden]
    const_emb = const_emb.unsqueeze(1)  # [B, 1, hidden]

    # Prepend constitution to encoder prefix
    prefix = torch.cat([const_emb, encoder_hidden_states], dim=1)
    return prefix
```

#### Issue 2.1.3: Update Config for Constitution Training
**File:** decoder_gpt2_config.py
**Changes:**
```python
@dataclass
class GPT2DecoderConfig:
    ...existing fields...
    # Constitution conditioning
    num_constitutions: int = 10
    use_constitution_conditioning: bool = True
    constitution_embedding_dim: int = None  # Defaults to gpt2_hidden_size
    constitution_conditioning_type: str = "prefix"  # "prefix", "additive", "gated"
```

---

## 🏔️ Milestone 3: Training Pipeline Updates

### Epic 3.1: Constitution-Aware Training

#### Issue 3.1.1: Update Training Script for Constitution
**File:** train_stage_c.py
**Changes:**
```python
# In _initialize_gpt2_decoder(), add:
gpt2_config = GPT2DecoderConfig(
    ...existing params...,
    num_constitutions=len(CONSTITUTION_TO_ID),
    use_constitution_conditioning=decoder_config_dict.get("use_constitution_conditioning", True),
)
```

#### Issue 3.1.2: Update DecoderTrainer.compute_loss
**File:** decoder_trainer.py
**Changes:**
```python
# In compute_loss(), add:
constitution_ids = inputs.get("constitution_ids")

outputs = decoder_head(
    encoder_hidden_states=encoder_embeddings,
    ...existing params...,
    constitution_ids=constitution_ids,  # NEW
)
```

#### Issue 3.1.3: Add Constitution Auxiliary Loss (Optional)
**File:** decoder_trainer.py
**Changes:**
```python
# Optional: Add constitution prediction auxiliary loss
# This encourages the model to encode constitution info

if self.config.use_constitution_aux_loss:
    const_logits = self.constitution_classifier(decoder_hidden_states[:, 0, :])
    const_loss = F.cross_entropy(const_logits, constitution_ids)
    total_loss = lm_loss + self.const_aux_weight * const_loss
```

#### Issue 3.1.4: Update Training Config
**File:** stage_c_gpt2_v4.yaml
**Changes:**
```yaml
decoder:
  type: "gpt2"
  gpt2_model_name: "gpt2-medium"
  # Constitution conditioning (NEW)
  use_constitution_conditioning: true
  num_constitutions: 10
  constitution_conditioning_type: "prefix"  # or "gated"
  constitution_aux_loss_weight: 0.1  # Optional auxiliary loss
```

---

## 🏔️ Milestone 4: Evaluation & Metrics

### Epic 4.1: Constitution Adherence Metrics

#### Issue 4.1.1: Implement Constitution Adherence Score (CAS)
**File:** decoder_metrics.py
**New Function:**
```python
def compute_constitution_adherence_score(
    predictions: list[str],
    constitution_schemas: dict,
    target_constitution: str,
) -> dict[str, float]:
    """
    Compute how well generated text adheres to target constitution.

    Returns:
        - cas_score: Overall adherence (0-1)
        - positive_token_rate: Rate of constitution-positive tokens
        - negative_token_rate: Rate of constitution-negative tokens
        - principle_alignment: Semantic alignment with core principles
    """
```

#### Issue 4.1.2: Add CAS to Training Callbacks
**File:** decoder_trainer.py
**Changes:**
```python
# In GenerationEvalCallback._run_generation_eval(), add:
from modeling_studio.evaluation.decoder_metrics import compute_constitution_adherence_score

cas_results = compute_constitution_adherence_score(
    predictions=predictions,
    constitution_schemas=self.constitution_schemas,
    target_constitution=self.target_constitution,
)
logger.info(f"[GenEval] CAS: {cas_results['cas_score']:.4f}")
```

#### Issue 4.1.3: Constitution Confusion Matrix
**File:** decoder_metrics.py
**New Function:**
```python
def compute_constitution_confusion(
    predictions: list[str],
    target_constitutions: list[str],
    all_constitutions: list[str],
) -> dict:
    """
    Compute confusion matrix showing if model produces wrong constitution style.
    Useful for detecting "constitution leakage".
    """
```

---

## 🏔️ Milestone 5: Data Augmentation

### Epic 5.1: Constitution-Aware Data Augmentation

#### Issue 5.1.1: Constitution Swapping Augmentation
**New File:** `scripts/augment_constitution_data.py`
**Purpose:** For each scenario, generate variations with different constitutions
```python
def augment_with_constitution_swap(
    samples: list[dict],
    constitution_schemas: dict,
) -> list[dict]:
    """
    For each sample, create variants with different constitutions.
    This teaches the model to differentiate between constitution styles.
    """
```

#### Issue 5.1.2: Constitution-Balanced Sampling
**File:** counterfactual_dataset.py
**New Class:**
```python
class ConstitutionBalancedSampler(Sampler):
    """
    Ensures each batch contains balanced representation of constitutions.
    Prevents the model from defaulting to majority constitution.
    """
```

---

## 🏔️ Milestone 6: Notebook & Colab Updates

### Epic 6.1: Update Training Notebooks

#### Issue 6.1.1: Update Colab Notebook for Constitution Training
**File:** train_v3_colab.ipynb
**Changes:**
- Add cell to verify constitution field in data
- Add constitution distribution visualization
- Update training config cell with constitution options
- Add CAS evaluation cell

---

## 📊 Dependency Graph

```
Milestone 1 (Data Pipeline)
    ├── Issue 1.1.1 → Issue 1.1.2 (registry needed for extraction)
    ├── Issue 1.1.3 (depends on 1.1.1)
    └── Issue 1.1.4 (depends on 1.1.3)

Milestone 2 (Model Architecture)
    ├── Issue 2.1.1 → Issue 2.1.2 (embedding before injection)
    └── Issue 2.1.3 (can be done in parallel)

Milestone 3 (Training Pipeline)
    ├── Issue 3.1.1 (depends on M2)
    ├── Issue 3.1.2 (depends on M1, M2)
    ├── Issue 3.1.3 (optional, depends on 3.1.2)
    └── Issue 3.1.4 (depends on 2.1.3)

Milestone 4 (Evaluation)
    ├── Issue 4.1.1 (can start early)
    ├── Issue 4.1.2 (depends on 4.1.1, 3.1.2)
    └── Issue 4.1.3 (depends on 4.1.1)

Milestone 5 (Augmentation)
    └── Depends on M1 completion

Milestone 6 (Notebooks)
    └── Depends on M1-M4 completion
```

---

## ✅ Implementation Order (Critical Path)

1. **Week 1:** Milestone 1 (Data Pipeline) - Foundation for everything
2. **Week 2:** Milestone 2 (Model Architecture) - Enable constitutional conditioning
3. **Week 3:** Milestone 3 (Training) + Milestone 4 (Evaluation) - In parallel
4. **Week 4:** Milestone 5 (Augmentation) + Milestone 6 (Notebooks)

---

Now let me review this plan for alignment and gaps:

Starting: *Review plan for alignment* (3/4)

---

## 🔍 Plan Alignment Review

### ✅ Alignment Check

| Requirement | Addressed By | Status |
|-------------|--------------|--------|
| Data preprocessing for constitution | Issue 1.1.1, 1.1.2 | ✅ |
| Dataset returns constitution | Issue 1.1.3 | ✅ |
| Collator handles constitution | Issue 1.1.4 | ✅ |
| Model accepts constitution | Issue 2.1.1, 2.1.2 | ✅ |
| Config supports constitution | Issue 2.1.3 | ✅ |
| Training script passes constitution | Issue 3.1.1, 3.1.2 | ✅ |
| Trainer computes loss with constitution | Issue 3.1.2, 3.1.3 | ✅ |
| YAML config updated | Issue 3.1.4 | ✅ |
| Evaluation metrics include CAS | Issue 4.1.1-4.1.3 | ✅ |
| Data augmentation for robustness | Issue 5.1.1, 5.1.2 | ✅ |
| Colab notebook updated | Issue 6.1.1 | ✅ |

---

Starting: *Identify gaps in plan* (4/4)

## 🔴 Gaps Identified & Missing Items

After reviewing the full plan against the codebase, I identified these **gaps**:

### Gap 1: Inference Script Updates (CRITICAL)
**Missing:** infer_decoder_fp16.py uses constitution at inference but not from model's learned embeddings.

**Add Issue 3.2.1:** Update inference script to use model's constitution embeddings
```python
# The model now has learned constitution embeddings
# Inference should use these instead of just logits bias
constitution_embedding = model.constitution_embeddings(constitution_id)
```

### Gap 2: Backward Compatibility (CRITICAL)
**Missing:** Existing checkpoints don't have constitution embeddings.

**Add Issue 2.1.4:** Backward compatibility for loading old checkpoints
```python
# In GPT2DecoderHead.from_pretrained() or load_checkpoint():
if "constitution_embeddings" not in state_dict:
    # Initialize constitution embeddings from scratch
    self.constitution_embeddings.weight.data.normal_(0, 0.02)
    logger.info("Constitution embeddings not found in checkpoint, initialized fresh")
```

### Gap 3: Constitution Schema Embedding Initialization (IMPORTANT)
**Missing:** How to initialize constitution embeddings meaningfully from schema text.

**Add Issue 2.1.5:** Initialize constitution embeddings from schema descriptions
```python
def initialize_constitution_embeddings_from_text(
    model,
    tokenizer,
    encoder,
    constitution_schemas: dict,
):
    """
    Initialize constitution embeddings using encoded schema descriptions.
    This gives meaningful starting points rather than random init.
    """
    for const_name, schema in constitution_schemas.items():
        desc = schema.get("description", "")
        enc_output = encoder.encode(desc)
        model.constitution_embeddings.weight.data[const_id] = project(enc_output)
```

### Gap 4: Contrastive Constitution Loss (OPTIONAL BUT VALUABLE)
**Missing:** The model could benefit from contrastive learning to distinguish constitutions.

**Add Issue 3.1.5:** Contrastive constitution loss
```python
# For each sample, create a negative pair with wrong constitution
# Loss encourages model to produce different outputs for different constitutions
def constitution_contrastive_loss(
    model,
    batch,
    negative_constitution_ids,
):
    # Generate with correct constitution
    correct_hidden = model.generate_hidden(constitution_ids=batch["constitution_ids"])
    # Generate with wrong constitution
    wrong_hidden = model.generate_hidden(constitution_ids=negative_constitution_ids)
    # InfoNCE loss
    return info_nce_loss(correct_hidden, wrong_hidden)
```

### Gap 5: Test Files (QUALITY ASSURANCE)
**Missing:** No test file updates mentioned.

**Add Issue 7.1.1:** Add tests for constitution training
```
tests/test_constitution_training.py
- TestConstitutionDataset
- TestConstitutionCollator
- TestConstitutionConditioning
- TestConstitutionAdherenceScore
```

### Gap 6: Data Validation Script (DATA QUALITY)
**Missing:** No validation that existing data has extractable constitution info.

**Add Issue 1.1.5:** Validate existing data for constitution coverage
```python
# scripts/validate_constitution_coverage.py
# Check all training shards for:
# - % samples with metadata.cultural_context
# - Distribution of cultural contexts
# - Samples with missing/invalid constitution
```

### Gap 7: Generation Prompt Engineering (INFERENCE)
**Missing:** How constitution affects the generation prompt at inference.

**Add Issue 3.2.2:** Constitution-aware prompt templates
```python
CONSTITUTION_PROMPTS = {
    "gentle_parenting": "Respond with empathy, focusing on understanding feelings and connection...",
    "traditional_strict": "Provide clear guidance with consistent expectations and boundaries...",
    # ...
}
```

---

## 📋 Updated Complete Milestone List

| Milestone | Epic | Issues |
|-----------|------|--------|
| **M1: Data Pipeline** | 1.1: Explicit Constitution | 1.1.1, 1.1.2, 1.1.3, 1.1.4, **1.1.5 (new)** |
| **M2: Model Architecture** | 2.1: Constitutional Conditioning | 2.1.1, 2.1.2, 2.1.3, **2.1.4 (new)**, **2.1.5 (new)** |
| **M3: Training Pipeline** | 3.1: Constitution-Aware Training | 3.1.1, 3.1.2, 3.1.3, 3.1.4, **3.1.5 (new)** |
| **M3: Training Pipeline** | **3.2: Inference Updates (new)** | **3.2.1 (new)**, **3.2.2 (new)** |
| **M4: Evaluation** | 4.1: CAS Metrics | 4.1.1, 4.1.2, 4.1.3 |
| **M5: Augmentation** | 5.1: Data Augmentation | 5.1.1, 5.1.2 |
| **M6: Notebooks** | 6.1: Colab Updates | 6.1.1 |
| **M7: Testing (new)** | **7.1: Test Coverage (new)** | **7.1.1 (new)** |

---

Completed: *Identify gaps in plan* (4/4)

---

## 📊 Final Summary: Files to Update

| Priority | File | Issues | Effort |
|----------|------|--------|--------|
| 🔴 **P0** | prepare_decoder_training_data.py | 1.1.1, 1.1.5 | Medium |
| 🔴 **P0** | counterfactual_dataset.py | 1.1.3 | Small |
| 🔴 **P0** | decoder_collator.py | 1.1.4 | Small |
| 🔴 **P0** | decoder_gpt2.py | 2.1.1, 2.1.2, 2.1.4, 2.1.5 | Large |
| 🔴 **P0** | decoder_gpt2_config.py | 2.1.3 | Small |
| 🟡 **P1** | train_stage_c.py | 3.1.1, 3.1.2 | Medium |
| 🟡 **P1** | decoder_trainer.py | 3.1.2, 3.1.3 | Medium |
| 🟡 **P1** | stage_c_gpt2_v4.yaml | 3.1.4 | Small |
| 🟢 **P2** | decoder_metrics.py | 4.1.1, 4.1.2, 4.1.3 | Medium |
| 🟢 **P2** | infer_decoder_fp16.py | 3.2.1, 3.2.2 | Medium |
| 🟢 **P2** | train_v3_colab.ipynb | 6.1.1 | Small |
| 🔵 **P3** | **NEW:** `constitution_registry.py` | 1.1.2 | Small |
| 🔵 **P3** | **NEW:** `augment_constitution_data.py` | 5.1.1 | Medium |
| 🔵 **P3** | **NEW:** `test_constitution_training.py` | 7.1.1 | Medium |

---

## 🚀 Quick Start (Minimum Viable Path)

If you want to start immediately with minimum changes:

1. **Issue 1.1.1** - Add constitution to preprocessing (30 min)
2. **Issue 1.1.3** - Update dataset `__getitem__` (15 min)
3. **Issue 1.1.4** - Update collator (15 min)
4. **Issue 2.1.1** - Add constitution embedding layer (30 min)
5. **Issue 2.1.2** - Constitution prefix injection (30 min)
6. **Issue 3.1.2** - Pass constitution_ids in trainer (15 min)

This gives you a working constitutional training pipeline in ~2-3 hours of coding.

---
