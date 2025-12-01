# Stage B: FamilyOS Domain Adaptation

Stage B fine-tunes the Stage A model on FamilyOS-specific data using LoRA adapters to add domain capabilities while preserving generic knowledge.

## Overview

| Property | Value |
|----------|-------|
| Base Model | Stage A checkpoint (`modernbert-multitask-v0`) |
| New Capabilities | 5 (NER Family, Ingress, Safety FamilyOS, Relations, Intent) |
| Adaptation Method | LoRA (r=32, alpha=64) |
| Training Duration | 3-5 epochs |
| Output | `outputs/familyos-modernbert-unified-v1` |

---

## 1. Datasets Used

### 1.1 FamilyOS Domain Data

#### NER Family (Family-specific Named Entities)

| Dataset | Location | Train | Val | Description |
|---------|----------|-------|-----|-------------|
| Gold | `data/familyos/ner_family/gold/` | 87 | 5 | Human-annotated |
| Silver | `data/familyos/ner_family/silver/` | 12,703 | - | LLM-generated |
| **Total** | | **12,790** | **5** | |

**Label Schema (15 labels):**

- `O`, `B-PERSON`, `I-PERSON`, `B-KINSHIP`, `I-KINSHIP`, `B-NICKNAME`, `I-NICKNAME`, `B-PET`, `I-PET`, `B-HOME_LOCATION`, `I-HOME_LOCATION`, `B-FAMILY_EVENT`, `I-FAMILY_EVENT`, `B-ROUTINE`, `I-ROUTINE`

#### Ingress Classification (Domain Routing)

| Dataset | Location | Train | Val | Description |
|---------|----------|-------|-----|-------------|
| Gold | `data/familyos/ingress/` | 36 | 12 | Human-annotated |
| Silver | `data/familyos/ingress/silver/` | 12,623 | - | LLM-generated |
| **Total** | | **12,659** | **12** | |

**Label Schema (7 domains):**

- `DIARY` (0): Personal reflections, journaling
- `TASK` (1): To-dos, reminders, action items
- `HEALTH` (2): Medical, wellness, fitness
- `FINANCE` (3): Money, bills, budgets
- `RELATIONSHIP` (4): Family dynamics, social
- `WORK` (5): Job, career, professional
- `META` (6): System commands, queries about FamilyOS

#### Safety FamilyOS (Policy Bands)

| Dataset | Location | Train | Val | Description |
|---------|----------|-------|-----|-------------|
| Gold | `data/familyos/safety/` | 30 | 8 | Human-annotated |
| Silver | `data/familyos/safety/silver/` | 14,333 | - | LLM-generated |
| **Total** | | **14,363** | **8** | |

**Label Schema (4 bands):**

- `GREEN` (0): Safe, routine content
- `AMBER` (1): Needs attention, mild concern
- `RED` (2): Serious concern, escalate to K1
- `CRISIS` (3): Immediate intervention needed

#### Relations (Family Relationship Extraction)

| Dataset | Location | Train | Val | Description |
|---------|----------|-------|-----|-------------|
| Gold | `data/familyos/relations/` | 75 | 17 | Human-annotated |
| Silver | `data/familyos/relations/silver/` | 12,037 | - | LLM-generated |
| **Total** | | **12,112** | **17** | |

**Relation Types:**

- parent_of, child_of, spouse_of, sibling_of, grandparent_of, etc.

#### Intent Classification

| Dataset | Location | Train | Val | Description |
|---------|----------|-------|-----|-------------|
| Gold | `data/familyos/intents/` | 70 | 16 | Human-annotated |
| Silver | `data/familyos/intents/silver/` | 15,972 | - | LLM-generated |
| **Total** | | **16,042** | **16** | |

**Intent Categories:**

- diary_entry, task_create, health_log, query_info, reminder_set, etc.

#### Embeddings (Similarity Learning)

| Dataset | Location | Samples | Description |
|---------|----------|---------|-------------|
| Clusters | `data/familyos/embeddings/clusters.jsonl` | 40 | Semantic clusters |
| Pairs | `data/familyos/embeddings/pairs.jsonl` | 14 | Positive/negative pairs |
| Triplets | `data/familyos/embeddings/triplets.jsonl` | 10 | Anchor/positive/negative |

#### Temporal (Time Expression Extraction)

| Dataset | Location | Train | Val | Description |
|---------|----------|-------|-----|-------------|
| Gold | `data/familyos/temporal/` | 61 | 15 | Human-annotated |
| Silver | `data/familyos/temporal/silver/` | 13,980 | - | LLM-generated |
| **Total** | | **14,041** | **15** | |

### 1.2 Replay Data (Prevent Forgetting)

| Dataset | Task | Max Samples | Purpose |
|---------|------|-------------|---------|
| CoNLL-2003 | `ner_general` | 5,000 | Preserve generic NER |
| SNLI | `nli` | 10,000 | Preserve NLI ability |
| GoEmotions | `emotions` | 5,000 | Preserve emotion detection |

### 1.3 Data Summary

| Category | Total Samples | Gold | Silver |
|----------|--------------|------|--------|
| **FamilyOS Domain** | ~82,000 | ~359 | ~81,641 |
| **Replay** | ~20,000 | N/A | N/A |
| **Total Stage B** | ~102,000 | | |

---

## 2. Training Approach

### 2.1 Architecture

```
┌─────────────────────────────────────┐
│     ModernBERT-base Encoder         │
│   (Stage A weights + LoRA adapters) │
│         LoRA: r=32, α=64            │
└──────────────┬──────────────────────┘
               │
    ┌──────────┴──────────┐
    │  Stage A Heads      │  Stage B Heads (NEW)
    │  (some frozen)      │
    ▼                     ▼
┌───────┐             ┌───────┐
│NER Gen│ (frozen)    │NER Fam│
│Safety │ (frozen)    │Ingress│
│  NLI  │ (frozen)    │SafetyF│
│Sentim.│ (trainable) │Relat. │
│Emot.  │ (trainable) │Intent │
│Embed. │ (trainable) └───────┘
└───────┘
```

### 2.2 LoRA Configuration

```yaml
peft:
  method: lora
  lora:
    r: 32                 # Rank
    lora_alpha: 64        # Scaling factor
    lora_dropout: 0.05
    bias: none
    target_modules:       # Apply to attention + MLP
      - q_proj
      - k_proj
      - v_proj
      - o_proj
      - gate_proj
      - up_proj
      - down_proj
    task_type: FEATURE_EXTRACTION
```

### 2.3 Head Freezing Strategy

| Head | Status | Rationale |
|------|--------|-----------|
| `ner_general` | **Frozen** | Preserve generic NER |
| `safety_generic` | **Frozen** | Keep as baseline |
| `nli` | **Frozen** | Preserve inference ability |
| `sentiment` | Trainable | Adapt to family context |
| `emotions` | Trainable | Adapt to family emotions |
| `embedding` | Trainable | Learn FamilyOS similarity |

### 2.4 Safety Oversampling

Critical safety classes are oversampled during training:

| Class | Oversampling Factor | Rationale |
|-------|---------------------|-----------|
| `CRISIS` | **20x** | Must not miss any crisis |
| `RED` | **5x** | High-priority escalation |
| `AMBER` | 1x | Standard sampling |
| `GREEN` | 1x | Standard sampling |

### 2.5 Task Weights

```yaml
task_weights:
  # Replay tasks (lower weight)
  ner_general: 0.2
  sentiment: 0.3
  emotions: 0.3
  safety_generic: 0.2
  nli: 0.1
  embedding: 0.3

  # FamilyOS tasks (main focus)
  ner_family: 1.0
  ingress: 1.0
  safety_familyos: 1.5  # Emphasized
  relation: 1.0
  intent: 1.0
```

### 2.6 Configuration Files

| File | Purpose |
|------|---------|
| `configs/training/multitask/stage_b_familyos.yaml` | Main training config |
| `configs/data/multitask/stage_b_datasets.yaml` | Dataset definitions |
| `configs/training/lora.yaml` | LoRA base config |

---

## 3. Evaluation and Scores

### 3.1 Target Metrics (V2 Spec)

#### FamilyOS Tasks

| Task | Metric | Target | Priority |
|------|--------|--------|----------|
| Safety CRISIS Recall | Recall | **≥ 98%** | P0 (non-negotiable) |
| Safety RED Recall | Recall | ≥ 90% | P0 |
| Safety Overall | Accuracy | ≥ 80% | P1 |
| NER Family | F1 | ≥ 85% | P1 |
| Ingress | F1 | ≥ 85% | P1 |
| Relations | F1 | ≥ 80% | P2 |
| Intent | Accuracy | ≥ 85% | P1 |

#### Forgetting Gates (Must Pass)

| Task | Metric | Max Drop | Status |
|------|--------|----------|--------|
| NER (CoNLL) | F1 | ≤ 2% | Gate |
| Sentiment (SST-2) | Accuracy | ≤ 2% | Gate |
| NLI (MNLI) | Accuracy | ≤ 2% | Gate |
| Emotions | Macro F1 | ≤ 3% | Gate |

### 3.2 Evaluation Scripts

```bash
# Evaluate Stage B model on FamilyOS tasks
python scripts/evaluate_stage_b.py \
    --model outputs/familyos-modernbert-unified-v1

# Check forgetting (compare Stage A vs Stage B)
python scripts/forgetting_eval.py \
    --stage-a outputs/modernbert-multitask-v0 \
    --stage-b outputs/familyos-modernbert-unified-v1

# Calibrate safety thresholds
python scripts/calibrate_safety.py \
    --model outputs/familyos-modernbert-unified-v1 \
    --data data/familyos/safety/validation.jsonl
```

### 3.3 Sample Forgetting Report

```
======================================================================
CATASTROPHIC FORGETTING EVALUATION REPORT
======================================================================
Stage A: outputs/modernbert-multitask-v0
Stage B: outputs/familyos-modernbert-unified-v1

----------------------------------------------------------------------
✅ ner_general:     F1 89.2% → 88.5% (drop: 0.7%, max: 2.0%) PASS
✅ sentiment:       Acc 93.1% → 92.4% (drop: 0.7%, max: 2.0%) PASS
✅ nli:             Acc 85.4% → 84.9% (drop: 0.5%, max: 2.0%) PASS
✅ emotions:        F1 47.3% → 46.1% (drop: 1.2%, max: 3.0%) PASS
----------------------------------------------------------------------
✅ ALL FORGETTING GATES PASSED
======================================================================
```

### 3.4 Sample Calibration Report

```
======================================================================
SAFETY THRESHOLD CALIBRATION REPORT
======================================================================
Temperature: 1.15

THRESHOLDS
----------------------------------------------------------------------
  GREEN_AMBER: 0.35
  AMBER_RED: 0.45
  RED_CRISIS: 0.60

PER-CLASS METRICS
----------------------------------------------------------------------
  GREEN:  Recall=0.95  Precision=0.92  FNR=0.05
  AMBER:  Recall=0.91  Precision=0.85  FNR=0.09
  RED:    Recall=0.96  Precision=0.88  FNR=0.04
  CRISIS: Recall=0.99  Precision=0.94  FNR=0.01  ← TARGET MET

CULTURAL ROBUSTNESS
----------------------------------------------------------------------
  Pass rate: 10/10 (100%)  ← No false CRISIS on Indian hyperbole
======================================================================
```

---

## 4. How to Run Scripts

### 4.1 Training

```bash
# Standard training
python scripts/train_stage_b.py \
    --config configs/training/multitask/stage_b_familyos.yaml

# Debug mode (smaller batches, data subset)
python scripts/train_stage_b.py \
    --config configs/training/multitask/stage_b_familyos.yaml \
    --debug

# Custom Stage A checkpoint
python scripts/train_stage_b.py \
    --config configs/training/multitask/stage_b_familyos.yaml \
    --model.name_or_path checkpoints/modernbert-multitask-v0/best

# Adjust LoRA parameters
python scripts/train_stage_b.py \
    --config configs/training/multitask/stage_b_familyos.yaml \
    --peft.lora.r 64 \
    --peft.lora.lora_alpha 128

# Resume training
python scripts/train_stage_b.py \
    --config configs/training/multitask/stage_b_familyos.yaml \
    --resume-from-checkpoint checkpoints/familyos-modernbert-unified-v1/checkpoint-1000
```

### 4.2 Evaluation

```bash
# Full Stage B evaluation
python scripts/evaluate_stage_b.py \
    --model outputs/familyos-modernbert-unified-v1

# Forgetting evaluation
python scripts/forgetting_eval.py \
    --stage-a outputs/modernbert-multitask-v0 \
    --stage-b outputs/familyos-modernbert-unified-v1 \
    --output outputs/forgetting_report.json

# Specific tasks only
python scripts/forgetting_eval.py \
    --stage-a outputs/modernbert-multitask-v0 \
    --stage-b outputs/familyos-modernbert-unified-v1 \
    --tasks ner_general sentiment nli
```

### 4.3 Safety Calibration

```bash
# Standard calibration
python scripts/calibrate_safety.py \
    --model outputs/familyos-modernbert-unified-v1 \
    --data data/familyos/safety/validation.jsonl

# Custom FNR targets
python scripts/calibrate_safety.py \
    --model outputs/familyos-modernbert-unified-v1 \
    --data data/familyos/safety/validation.jsonl \
    --crisis-fnr 0.005 \
    --red-fnr 0.02

# Save to custom location
python scripts/calibrate_safety.py \
    --model outputs/familyos-modernbert-unified-v1 \
    --data data/familyos/safety/validation.jsonl \
    --output configs/calibration
```

### 4.4 Inference

```bash
# Single text with all capabilities
python scripts/infer.py \
    --model outputs/familyos-modernbert-unified-v1 \
    --text "Mum said Panda has a doctor appointment tomorrow" \
    --capabilities ner_family safety_familyos ingress

# FamilyOS-specific inference
python scripts/infer.py \
    --model outputs/familyos-modernbert-unified-v1 \
    --text "I've been feeling really down lately, nothing seems to matter" \
    --capabilities safety_familyos
```

---

## 5. Benchmarking Guide

### 5.1 FamilyOS Benchmarks

| Benchmark | Dataset | Metric | Script |
|-----------|---------|--------|--------|
| **Safety** | FamilyOS Safety | CRISIS Recall | `evaluate_stage_b.py` |
| **NER Family** | FamilyOS NER | F1 | `evaluate_stage_b.py` |
| **Ingress** | FamilyOS Ingress | F1 | `evaluate_stage_b.py` |
| **Forgetting** | Stage A benchmarks | Max Drop | `forgetting_eval.py` |
| **Cultural** | Indian patterns | FP Rate | `calibrate_safety.py` |

### 5.2 Adding New FamilyOS Benchmarks

1. **Create test dataset** in `data/familyos/{task}/test.jsonl`:

```jsonl
{"text": "...", "label": 0, ...}
{"text": "...", "label": 1, ...}
```

2. **Add loader** in `src/modeling_studio/data/loaders.py`:

```python
def load_familyos_new_task(split: str = "test") -> Dataset:
    """Load new FamilyOS benchmark."""
    path = f"data/familyos/new_task/{split}.jsonl"
    return load_dataset("json", data_files=path, split="train")
```

3. **Add evaluation metrics** in `src/modeling_studio/evaluation/`:

```python
def compute_new_task_metrics(predictions, labels) -> dict:
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="macro"),
    }
```

### 5.3 Safety Stress Testing

Test edge cases for safety classification:

```bash
# Run safety stress test
python scripts/safety_stress_test.py \
    --model outputs/familyos-modernbert-unified-v1 \
    --test-suite data/familyos/safety/stress_tests.jsonl

# Test cultural robustness specifically
python scripts/safety_stress_test.py \
    --model outputs/familyos-modernbert-unified-v1 \
    --test-suite indian_hyperbole
```

**Stress Test Categories:**

| Category | Purpose | Expected Behavior |
|----------|---------|-------------------|
| Indian Hyperbole | Cultural expressions | Never CRISIS |
| Genuine Crisis | Real self-harm | Always CRISIS |
| Edge Cases | Ambiguous text | Conservative (escalate) |
| Code-switching | Mixed language | Correct classification |

### 5.4 A/B Testing New Models

```bash
# Compare two Stage B models
python scripts/compare_models.py \
    --model-a outputs/familyos-v1 \
    --model-b outputs/familyos-v2 \
    --benchmark familyos_safety \
    --output comparison_report.json
```

### 5.5 Performance Tracking

Track metrics over time:

```bash
# Log evaluation to tracking system
python scripts/evaluate_stage_b.py \
    --model outputs/familyos-modernbert-unified-v1 \
    --track \
    --run-name "v1.0.0-release"
```

---

## 6. Deployment Checklist

Before deploying a Stage B model:

### 6.1 Quality Gates

- [ ] All forgetting gates pass (≤2% drop on Stage A tasks)
- [ ] CRISIS recall ≥ 98%
- [ ] RED recall ≥ 90%
- [ ] Cultural robustness ≥ 95% (no false CRISIS on hyperbole)
- [ ] Safety thresholds calibrated and saved

### 6.2 Files to Deploy

```
outputs/familyos-modernbert-unified-v1/
├── config.json                    # Model config
├── model.safetensors              # Model weights (merged)
├── tokenizer.json                 # Tokenizer
├── tokenizer_config.json
├── special_tokens_map.json
├── capabilities.json              # List of capabilities
├── calibration.json               # Safety thresholds
└── safety_thresholds.yaml         # Deployment config

outputs/familyos-modernbert-unified-v1-lora/
├── adapter_config.json            # LoRA config (optional)
└── adapter_model.safetensors      # LoRA weights only
```

### 6.3 Integration Example

```python
from modeling_studio.inference import UnifiedInference

# Load model with calibration
model = UnifiedInference(
    model_path="outputs/familyos-modernbert-unified-v1",
    calibration_path="outputs/familyos-modernbert-unified-v1/calibration.json",
)

# Run inference
result = model.infer(
    text="Panda has been crying all day, says nobody loves her",
    capabilities=["safety_familyos", "ner_family", "emotions"],
)

print(result.safety_familyos)  # → "AMBER" or "RED"
print(result.ner_family)       # → [("Panda", "NICKNAME")]
print(result.emotions)         # → ["sadness", "fear"]
```

---

## 7. Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Forgetting gates fail | LoRA too aggressive | Reduce r, increase replay ratio |
| CRISIS recall low | Insufficient CRISIS samples | Increase CRISIS oversampling |
| Training diverges | LR too high | Reduce learning_rate to 5e-5 |
| OOM on GPU | Batch too large | Reduce batch_size, enable gradient_checkpointing |

### Debugging Tips

```bash
# Check model capabilities
python -c "
from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel
model = ModernBertMultiTaskModel.load_checkpoint('outputs/familyos-modernbert-unified-v1')
print('Capabilities:', [c.value for c in model.capabilities])
print('Heads:', list(model.heads.keys()))
"

# Verify LoRA was merged
python -c "
import json
with open('outputs/familyos-modernbert-unified-v1/config.json') as f:
    config = json.load(f)
print('Model type:', config.get('model_type'))
print('Hidden size:', config.get('hidden_size'))
"
```

---

## 8. Next Steps

After Stage B training:

1. **Run full evaluation** on FamilyOS benchmarks
2. **Check forgetting** against Stage A baselines
3. **Calibrate safety** thresholds
4. **Deploy** to K0 runtime
5. **Monitor** safety metrics in production

```bash
# Complete post-training workflow
python scripts/evaluate_stage_b.py --model outputs/familyos-modernbert-unified-v1
python scripts/forgetting_eval.py --stage-a outputs/modernbert-multitask-v0 --stage-b outputs/familyos-modernbert-unified-v1
python scripts/calibrate_safety.py --model outputs/familyos-modernbert-unified-v1
```
