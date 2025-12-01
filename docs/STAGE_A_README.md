# Stage A: Generic Multi-Task Training

Stage A trains a unified encoder on public datasets to build strong generic capabilities before domain adaptation.

## Overview

| Property | Value |
|----------|-------|
| Base Model | `answerdotai/ModernBERT-base` (149M params) |
| Capabilities | 7 (NER, Sentiment, Emotions, Safety, NLI, Embedding, Temporal) |
| Training Duration | 10 epochs |
| Output | `outputs/modernbert-multitask-v0` |

---

## 1. Datasets Used

### 1.1 Named Entity Recognition (NER)

| Dataset | Source | Train Size | Description |
|---------|--------|-----------|-------------|
| **CoNLL-2003** | `conll2003` | 14,041 | Gold standard NER benchmark |
| **WikiNeural** | `tner/wikineural` (en) | 92,720 | Large-scale Wikipedia NER |

**Label Schema (9 labels):**
- `O`, `B-PER`, `I-PER`, `B-ORG`, `I-ORG`, `B-LOC`, `I-LOC`, `B-MISC`, `I-MISC`

### 1.2 Sentiment Classification

| Dataset | Source | Train Size | Description |
|---------|--------|-----------|-------------|
| **SST-2** | `stanfordnlp/sst2` | 67,349 | Stanford Sentiment Treebank |

**Label Schema (5 labels):**
- Very Negative, Negative, Neutral, Positive, Very Positive

### 1.3 Emotion Classification

| Dataset | Source | Train Size | Description |
|---------|--------|-----------|-------------|
| **GoEmotions** | `google-research-datasets/go_emotions` | 43,410 | Reddit emotions (multi-label) |

**Label Schema (28 emotions + neutral):**
- admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral

### 1.4 Natural Language Inference (NLI)

| Dataset | Source | Train Size | Description |
|---------|--------|-----------|-------------|
| **SNLI** | `stanfordnlp/snli` | 550,152 | Stanford NLI corpus |
| **MNLI** | `multi_nli` | 392,702 | Multi-genre NLI |

**Label Schema (3 labels):**
- Entailment, Neutral, Contradiction

### 1.5 Safety Classification

| Dataset | Source | Train Size | Description |
|---------|--------|-----------|-------------|
| **Jigsaw Toxicity** | `jigsaw_toxicity_pred` | ~160K | Toxic comment classification |

**Label Schema (8 labels, multi-label):**
- toxic, severe_toxic, obscene, threat, insult, identity_hate, sexual_explicit, unknown

### 1.6 Embedding (Sentence Similarity)

| Dataset | Source | Train Size | Description |
|---------|--------|-----------|-------------|
| **STS-B** | `glue/stsb` | 5,749 | Semantic textual similarity |
| **AllNLI** | Multiple | ~1M | Combined NLI for contrastive learning |

### 1.7 Temporal Expression Extraction

| Dataset | Source | Train Size | Description |
|---------|--------|-----------|-------------|
| **TimeBank** | Custom | ~10K | Temporal expressions in text |

**Label Schema (13 labels):**
- `O`, `B-DATE`, `I-DATE`, `B-TIME`, `I-TIME`, `B-DURATION`, `I-DURATION`, `B-SET`, `I-SET`, `B-RELATIVE`, `I-RELATIVE`, `B-FUZZY`, `I-FUZZY`

---

## 2. Training Approach

### 2.1 Architecture

```
┌─────────────────────────────────────┐
│     ModernBERT-base Encoder         │
│         (149M params)               │
└──────────────┬──────────────────────┘
               │
    ┌──────────┼──────────┬──────────┬──────────┬──────────┬──────────┐
    ▼          ▼          ▼          ▼          ▼          ▼          ▼
┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
│  NER  │ │ Sent. │ │ Emot. │ │ Safety│ │  NLI  │ │ Embed │ │ Temp. │
│ Head  │ │ Head  │ │ Head  │ │ Head  │ │ Head  │ │ Head  │ │ Head  │
└───────┘ └───────┘ └───────┘ └───────┘ └───────┘ └───────┘ └───────┘
```

### 2.2 Key Training Features (V2 Compliant)

| Feature | Value | Purpose |
|---------|-------|---------|
| **Epochs** | 10 | Full convergence |
| **Batch Size** | 128 (effective ~256) | A100 optimized |
| **Head-wise LR** | Encoder: 2e-5, Heads: 1e-4 | Preserve pretrained knowledge |
| **Layer Decay** | 0.95 | Lower LR for earlier layers |
| **EMA** | decay=0.999 | Smoothed model averaging |
| **Uncertainty Weighting** | Enabled | Automatic task balancing |
| **Temperature Sampling** | T=4.0 | Balanced task sampling |
| **Flash Attention 2** | Enabled | Speed optimization |

### 2.3 Task Loss Weights

```yaml
task_weights:
  ner_general: 1.0
  sentiment: 1.0
  emotions: 1.5       # Emphasized
  safety_generic: 2.0 # Emphasized (critical)
  nli: 1.0
  embedding: 0.5      # Reduced (dominates otherwise)
  temporal: 1.0
```

### 2.4 Configuration Files

| File | Purpose |
|------|---------|
| `configs/training/multitask/stage_a_a100_fast.yaml` | A100 hyperfast config |
| `configs/training/multitask/stage_a_v2_compliant.yaml` | V2 spec compliant config |
| `configs/data/multitask/stage_a_datasets.yaml` | Dataset definitions |

---

## 3. Evaluation and Scores

### 3.1 Target Metrics (V2 Spec)

| Task | Metric | Target | Priority |
|------|--------|--------|----------|
| NER (CoNLL) | F1 | ≥ 88% | P0 |
| Sentiment (SST-2) | Accuracy | ≥ 92% | P0 |
| Emotions (GoEmotions) | Macro F1 | ≥ 45% | P1 |
| NLI (MNLI) | Accuracy | ≥ 84% | P0 |
| Safety (Jigsaw) | Macro F1 | ≥ 70% | P0 |
| Embedding (STS-B) | Spearman | ≥ 0.80 | P1 |

### 3.2 Evaluation Script

```bash
# Evaluate a checkpoint
python scripts/evaluate_stage_a.py \
    --checkpoint outputs/modernbert-multitask-v0/best \
    --tasks ner_general sentiment emotions nli safety_generic

# Evaluate specific tasks
python scripts/evaluate_stage_a.py \
    --checkpoint outputs/modernbert-multitask-v0/best \
    --tasks sentiment nli \
    --output eval_results.json
```

### 3.3 Sample Output

```
================================================================================
                        STAGE A EVALUATION RESULTS
================================================================================
Model: outputs/modernbert-multitask-v0/best

Task Results:
--------------------------------------------------------------------------------
✅ ner_general:     F1=89.2%  Precision=88.5%  Recall=89.9%
✅ sentiment:       Accuracy=93.1%  F1=92.8%
✅ emotions:        Macro F1=47.3%  Accuracy=52.1%
✅ nli:             Accuracy=85.4%  F1=85.1%
✅ safety_generic:  Macro F1=72.4%  Accuracy=91.2%
✅ embedding:       Spearman=0.823  Pearson=0.819

Overall Score: 79.9%
================================================================================
```

---

## 4. How to Run Scripts

### 4.1 Training

```bash
# Standard training (A100 recommended)
python scripts/train_stage_a.py \
    --config configs/training/multitask/stage_a_a100_fast.yaml

# Debug mode (smaller batches, subset data)
python scripts/train_stage_a.py \
    --config configs/training/multitask/stage_a_a100_fast.yaml \
    --debug

# Resume from checkpoint
python scripts/train_stage_a.py \
    --config configs/training/multitask/stage_a_a100_fast.yaml \
    --resume-from-checkpoint checkpoints/modernbert-multitask-v0/checkpoint-5000

# Override config values
python scripts/train_stage_a.py \
    --config configs/training/multitask/stage_a_a100_fast.yaml \
    --training.num_train_epochs=5 \
    --training.per_device_train_batch_size=64
```

### 4.2 Evaluation

```bash
# Full evaluation
python scripts/evaluate_stage_a.py \
    --checkpoint outputs/modernbert-multitask-v0/best

# Quick evaluation (specific tasks)
python scripts/evaluate_stage_a.py \
    --checkpoint outputs/modernbert-multitask-v0/best \
    --tasks sentiment ner_general

# Save results to file
python scripts/evaluate_stage_a.py \
    --checkpoint outputs/modernbert-multitask-v0/best \
    --output outputs/eval_results.json
```

### 4.3 Inference

```bash
# Single text inference
python scripts/infer.py \
    --model outputs/modernbert-multitask-v0/best \
    --text "Apple Inc. reported strong Q4 earnings" \
    --capabilities ner_general sentiment

# Batch inference
python scripts/infer.py \
    --model outputs/modernbert-multitask-v0/best \
    --input data/test_samples.jsonl \
    --output predictions.jsonl
```

---

## 5. Benchmarking Guide

### 5.1 Standard Benchmarks

| Benchmark | Dataset | Metric | How to Run |
|-----------|---------|--------|------------|
| **NER** | CoNLL-2003 test | F1 | `--tasks ner_general` |
| **Sentiment** | SST-2 validation | Accuracy | `--tasks sentiment` |
| **Emotions** | GoEmotions test | Macro F1 | `--tasks emotions` |
| **NLI** | MNLI matched | Accuracy | `--tasks nli` |
| **Similarity** | STS-B | Spearman | `--tasks embedding` |

### 5.2 Adding New Benchmarks

1. **Create benchmark dataset loader** in `src/modeling_studio/data/loaders.py`:

```python
def load_new_benchmark(split: str = "test") -> Dataset:
    """Load your new benchmark dataset."""
    ds = load_dataset("your/benchmark", split=split)
    # Preprocess to match expected format
    return ds
```

2. **Add evaluation logic** in `src/modeling_studio/evaluation/evaluator.py`:

```python
def evaluate_new_benchmark(model, dataset) -> dict:
    """Evaluate on new benchmark."""
    # Run inference
    # Compute metrics
    return {"accuracy": ..., "f1": ...}
```

3. **Register in evaluate script**:

```python
BENCHMARK_REGISTRY["new_benchmark"] = {
    "loader": load_new_benchmark,
    "evaluator": evaluate_new_benchmark,
    "metrics": ["accuracy", "f1"],
}
```

### 5.3 Cross-lingual Benchmarking

For multilingual evaluation:

```bash
# XNLI (cross-lingual NLI)
python scripts/evaluate_stage_a.py \
    --checkpoint outputs/modernbert-multitask-v0/best \
    --benchmark xnli \
    --languages en de fr es

# WikiANN (multilingual NER)
python scripts/evaluate_stage_a.py \
    --checkpoint outputs/modernbert-multitask-v0/best \
    --benchmark wikiann \
    --languages en de fr
```

### 5.4 Performance Profiling

```bash
# Measure inference speed
python scripts/benchmark_speed.py \
    --model outputs/modernbert-multitask-v0/best \
    --batch-sizes 1 8 32 64 \
    --sequence-lengths 64 128 256 512

# Memory profiling
python scripts/benchmark_speed.py \
    --model outputs/modernbert-multitask-v0/best \
    --profile-memory
```

### 5.5 Comparison with Baselines

| Model | NER F1 | Sent Acc | NLI Acc | Params |
|-------|--------|----------|---------|--------|
| BERT-base | 91.0% | 92.5% | 84.6% | 110M |
| RoBERTa-base | 91.5% | 94.8% | 87.6% | 125M |
| **ModernBERT-MT (Ours)** | 89.2% | 93.1% | 85.4% | 149M |

---

## 6. Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| OOM on A100-40GB | Reduce batch size to 64, enable gradient checkpointing |
| NaN loss | Disable Flash Attention 2, reduce learning rate |
| Poor NER performance | Check label alignment, ensure BIO format |
| Slow training | Increase dataloader workers, enable pin_memory |

### Logging & Monitoring

```bash
# View TensorBoard logs
tensorboard --logdir outputs/modernbert-multitask-v0/logs

# Check training progress
tail -f outputs/modernbert-multitask-v0/logs/trainer_state.json
```

---

## 7. Next Steps

After Stage A training completes:

1. **Evaluate** the final model on all benchmarks
2. **Export** the best checkpoint for Stage B
3. **Run Stage B** for FamilyOS domain adaptation

```bash
# Export best model
cp -r checkpoints/modernbert-multitask-v0/best outputs/modernbert-multitask-v0/best

# Start Stage B
python scripts/train_stage_b.py \
    --config configs/training/multitask/stage_b_familyos.yaml \
    --model.name_or_path outputs/modernbert-multitask-v0/best
```
