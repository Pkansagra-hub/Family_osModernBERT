# FamilyOS Model Training Strategy

## Curriculum Learning: From Foundation to Ultra

This document outlines the progressive training strategy for FamilyOS emotion understanding models.

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CURRICULUM LEARNING PIPELINE                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Stage A (Foundation)     Stage B (Specialization)     v3 (Ultra)      │
│   ┌──────────────────┐     ┌──────────────────────┐    ┌─────────────┐  │
│   │   7 Super-Labels │ ──► │   44 FamilyOS Labels │ ─► │  28 Layers  │  │
│   │   ModernBERT-base│     │   Fine-tuned Model   │    │  + Reasoning│  │
│   └──────────────────┘     └──────────────────────┘    └─────────────┘  │
│                                                                          │
│   Easy ──────────────────► Hard ────────────────────► Complex           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Stage A: Foundation (7 Super-Labels)

### Goal

Build a robust base model that understands the **broad strokes** of emotion. It learns the difference between "Good" and "Bad" and "Active" vs "Passive" very well.

### Super-Label Taxonomy

| Super-Label | Constituent Emotions |
|-------------|---------------------|
| **JOY** | joy, excitement, celebration, pride, relief, happiness, amusement |
| **AFFECTION** | love, warmth, caring, gratitude, tenderness, admiration, affection |
| **SADNESS** | sadness, grief, disappointment, longing, melancholy, sorrow |
| **ANXIETY** | worry, overwhelmed, frustration, annoyance, stress, nervousness, fear |
| **NOSTALGIA** | nostalgia, bittersweet, wistfulness, reminiscence |
| **CONTENTMENT** | contentment, belonging, togetherness, peace, serenity, satisfaction |
| **NEUTRAL** | neutral, patience, calm, indifference |

### Training Configuration

```yaml
model: answerdotai/ModernBERT-base
num_labels: 7
task: multi-label classification
loss: BCEWithLogitsLoss
epochs: 3-5
learning_rate: 2e-5
batch_size: 32
```

### Data Source

- Use healed unified dataset: `data/familyos/unified/output_healed/` + `output_synthetic_healed/`
- Map 44 emotions → 7 super-labels during data loading

### Expected Outcome

- Model understands emotional valence (positive/negative)
- Model distinguishes arousal levels (calm/excited)
- Strong separation between super-label clusters
- F1 Score Target: > 0.85 on super-labels

---

## Stage B: Specialization (44 FamilyOS Labels)

### Goal

Teach the model **nuance** by fine-tuning on the full 44-label dataset. Because the base model already knows `JOY` vs `SADNESS`, it converges much faster and more accurately on the subtypes.

### Full Label Set (44 Emotions)

**Positive Spectrum:**

- joy, excitement, love, pride, gratitude, happiness, celebration, relief
- warmth, caring, tenderness, admiration, affection, contentment
- belonging, togetherness, hope, optimism, anticipation, amusement

**Negative Spectrum:**

- sadness, grief, disappointment, longing, worry, overwhelmed
- frustration, annoyance, anger, resentment, fear, anxiety
- stress, nervousness, guilt, shame, embarrassment, jealousy

**Complex/Mixed:**

- nostalgia, bittersweet, wistfulness, melancholy
- parental_pride, parental_guilt, protective, empathy
- neutral, patience, calm, curiosity

### Training Configuration

```yaml
base_model: outputs/stage-a-foundation/best_model  # Load Stage A weights
num_labels: 44
task: multi-label classification
loss: BCEWithLogitsLoss (with class weights for imbalance)
epochs: 5-8
learning_rate: 1e-5  # Lower LR for fine-tuning
batch_size: 16
warmup_ratio: 0.1
```

### Data Source

- Same healed dataset with original 44-label annotations
- Apply class balancing/oversampling for rare emotions

### Expected Outcome

- Model distinguishes subtle emotional differences
- `Joy` vs `Parental Pride` correctly separated
- `Caring` appropriately detected in implicit contexts
- F1 Score Target: > 0.70 on 44 labels (multi-label is harder)

---

## v3: Ultra (28-Layer Expansion + Reasoning)

### Goal

Use the mature v2 weights as a **"seed"** to grow a larger, smarter model that can handle complex reasoning (NLI, Relations) alongside the 44 emotions.

### Architecture Expansion

```
ModernBERT-base (22 layers)
├── Layers 1-14: Keep frozen (foundational representations)
├── Layers 15-22: Fine-tuned emotion understanding
│
v3 Ultra (28 layers)
├── Layers 1-22: Initialize from v2 weights
├── Layers 23-28: CLONE from Layers 15-20 (warm start)
│
New Heads:
├── Emotion Head (44 labels) - from v2
├── NLI Head (entailment/contradiction/neutral)
├── Relation Head (family relationships)
├── Safety Head (GREEN/AMBER/RED)
└── Intent Head (8 intents)
```

### Layer Cloning Strategy

```python
# Pseudocode for weight initialization
v3_model.layers[1:22] = v2_model.layers[1:22]  # Copy all v2 weights
v3_model.layers[23:28] = v2_model.layers[15:20].clone()  # Clone mature layers

# The cloned layers already understand emotion - they just need to
# learn to apply that understanding to new tasks (NLI, Relations)
```

### Multi-Task Training

```yaml
base_model: outputs/stage-b-v2/best_model
architecture: ModernBERT-28L (custom)
heads:
  emotion:
    num_labels: 44
    weight: 1.0
  nli:
    num_labels: 3
    weight: 0.5
  relations:
    num_labels: 15
    weight: 0.3
  safety:
    num_labels: 3
    weight: 0.5
  intent:
    num_labels: 8
    weight: 0.3

training:
  epochs: 10
  learning_rate: 5e-6
  batch_size: 8
  gradient_accumulation: 4
  freeze_layers: [1-14]  # Keep foundational layers frozen
```

### Expected Outcome

- Single model handles ALL FamilyOS tasks
- Emotion understanding enhanced by reasoning capabilities
- Relations inform emotion (knowing "mom" adds "caring" context)
- Safety-aware emotional responses

---

## Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                           DATA SOURCES                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Healed Dataset (419,501 samples)                                   │
│  ├── output_healed/ (69,371 real)                                   │
│  └── output_synthetic_healed/ (350,130 synthetic)                   │
│                                                                      │
│  ┌──────────────────┐                                               │
│  │   Stage A        │  Map 44 → 7 super-labels                      │
│  │   (Foundation)   │  Train on broad categories                    │
│  └────────┬─────────┘                                               │
│           │                                                          │
│           ▼                                                          │
│  ┌──────────────────┐                                               │
│  │   Stage B        │  Use original 44 labels                       │
│  │   (v2 - 44)      │  Fine-tune from Stage A                       │
│  └────────┬─────────┘                                               │
│           │                                                          │
│           ▼                                                          │
│  ┌──────────────────┐                                               │
│  │   v3 Ultra       │  Add NLI, Relations, Safety, Intent           │
│  │   (28 layers)    │  Multi-task from v2 weights                   │
│  └──────────────────┘                                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Training Timeline

| Stage | Duration | GPU Hours | Output |
|-------|----------|-----------|--------|
| Stage A | 2-3 hours | ~3 hrs (A100) | `outputs/stage-a-foundation/` |
| Stage B | 4-6 hours | ~6 hrs (A100) | `outputs/stage-b-v2/` |
| v3 Ultra | 8-12 hours | ~12 hrs (A100) | `outputs/v3-ultra/` |

---

## Evaluation Checkpoints

### Stage A Metrics

- [ ] Super-label F1 > 0.85
- [ ] Confusion matrix shows clean separation
- [ ] No super-label has < 0.75 F1

### Stage B Metrics

- [ ] 44-label macro F1 > 0.65
- [ ] Rare emotions (< 1% samples) F1 > 0.50
- [ ] `Caring` correctly detected on task sentences
- [ ] No emotion collapse (all predicting same label)

### v3 Ultra Metrics

- [ ] Emotion F1 maintained from v2
- [ ] NLI accuracy > 0.85
- [ ] Relation extraction F1 > 0.70
- [ ] Safety classification F1 > 0.90
- [ ] Intent classification F1 > 0.85

---

## Files & Scripts

| Stage | Script | Config |
|-------|--------|--------|
| Stage A | `scripts/train_stage_a.py` | `configs/training/stage_a.yaml` |
| Stage B | `scripts/train_stage_b.py` | `configs/training/stage_b.yaml` |
| v3 Ultra | `scripts/train_v3_ultra.py` | `configs/training/v3_ultra.yaml` |

---

## Next Steps

1. **Create super-label mapping** for Stage A data loader
2. **Run Stage A training** on healed dataset
3. **Evaluate Stage A** and verify super-label separation
4. **Run Stage B** fine-tuning from Stage A checkpoint
5. **Implement v3 architecture** with layer cloning
6. **Multi-task training** for v3 Ultra

---

## References

- Curriculum Learning: Bengio et al., 2009
- Progressive Training: Karras et al. (StyleGAN)
- Multi-Task Learning: Caruana, 1997
- ModernBERT: <https://huggingface.co/answerdotai/ModernBERT-base>
