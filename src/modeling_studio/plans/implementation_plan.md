# FamilyOS Unified Encoder - Implementation Plan (Enhanced v2)

> **Base Model:** `answerdotai/ModernBERT-base` (Apache 2.0)
> **Goal:** Multi-task encoder with 12 capabilities replacing the current model zoo
> **Reference:** `enhanced_design_v2.md` and `unified_encoder_solution.md`
> **Updated:** November 2025 - Incorporated 2024-2025 SOTA training practices

---

## 🎯 Training Strategy Overview (Updated from Expert Review)

### Accepted Best Practices (2024-2025 SOTA)

| Technique | Implementation | Expected Gain |
|-----------|----------------|---------------|
| **EMA Model** | Decay 0.999, use for checkpointing | +0.8-1.5 pt consistent |
| **Head-wise Learning Rates** | Encoder 2e-5, heads 1e-4, token heads 5e-5 | +1-3 pt |
| **Uncertainty Weighting** | Already in plan, add log-var regularization | +2-4 pt |
| **Hard-negative Mining** | For embedding head, 15 negatives per batch | Embedding Spearman +0.06 |
| **Safety Oversampling** | CRISIS 20×, RED 5×, safety weight 10-20× | CRISIS recall ≥98% |
| **Catastrophic Forgetting Check** | Re-eval CoNLL, SST-2, MNLI after Stage B | ≤2% drop allowed |

### Rejected/Adapted Recommendations

| Recommendation | Status | Reason |
|----------------|--------|--------|
| 750B-1T token CPT | ❌ Skip | Don't have enough family data (need billions) |
| T5-style span corruption | ❌ Skip | ModernBERT is encoder-only, not compatible |
| Add 500 vocab tokens | ❌ Skip | Risky, BPE handles family terms fine |
| PCGrad/GradVac | ⚠️ Optional | Complex, add only if head conflicts observed |
| Sleep-style replay | ⚠️ Optional | Only needed for 100k+ step training |
| Separate Safety Phase C | ❌ Skip | Keep safety in Stage B with high weight |

---

## 📁 Project File Inventory

### Core Model Files (✅ Implemented)

| File | Status | Purpose |
|------|--------|---------|
| `src/modeling_studio/data/labels.py` | ✅ Done | Label schemas for all 12 capabilities (v2 enhanced) |
| `src/modeling_studio/models/heads.py` | ✅ Done | 9 head types: Seq, Token, NLI, Embedding, Safety, MultiLabel, Relation, Intent, Temporal |
| `src/modeling_studio/models/modernbert_multitask.py` | ✅ Done | Main multi-task model with 12 capability→head mappings |
| `src/modeling_studio/models/poolers.py` | 📝 Stub | Pooling strategies (CLS, Mean, CLSMean, Max, Weighted) |
| `src/modeling_studio/models/losses.py` | 📝 Stub | Custom loss functions |
| `src/modeling_studio/models/adapters.py` | 📝 **NEW (Stage B)** | Task-specific bottleneck adapters |
| `src/modeling_studio/models/pair_encoder.py` | 📝 **NEW (Stage B)** | Cross-attention pair encoder for NLI/Relation |

### Data Pipeline Files (✅ Done / 📝 Stub)

| File | Status | Purpose |
|------|--------|---------|
| `src/modeling_studio/data/loaders.py` | ✅ Done | All 12 dataset loaders implemented |
| `src/modeling_studio/data/tokenization.py` | ✅ Done | 7 tokenization functions + subword alignment |
| `src/modeling_studio/data/multitask_dataset.py` | ✅ Done | Combined multi-task dataset |
| `src/modeling_studio/data/preprocessing.py` | 📝 Stub | Text cleaning/normalization |
| `src/modeling_studio/data/augmentation.py` | 📝 **NEW** | Family-specific data augmentation |

### Training Files (📝 Stub - NEW FILES ADDED)

| File | Status | Purpose |
|------|--------|---------|
| `src/modeling_studio/trainers/multitask_trainer.py` | ✅ Done | Multi-task trainer extending HF Trainer |
| `src/modeling_studio/trainers/collators.py` | ✅ Done | Task-specific data collators (6 collators + MultiTaskCollator) |
| `src/modeling_studio/trainers/callbacks.py` | ✅ Done | Training callbacks (logging, early stop) |
| `src/modeling_studio/trainers/task_sampler.py` | ✅ Done | Task sampling strategies (5 samplers + factory) |
| `src/modeling_studio/trainers/ema.py` | ✅ Done | EMA model for smoother training |
| `src/modeling_studio/trainers/optimizer.py` | ✅ Done | Head-wise LR optimizer creation |
| `src/modeling_studio/trainers/task_weighting.py` | ✅ Done | Uncertainty-based task weighting |
| `src/modeling_studio/trainers/curriculum.py` | 📝 **NEW** | Curriculum learning scheduler |

### Evaluation Files (📝 Stub - NEW FILES ADDED)

| File | Status | Purpose |
|------|--------|---------|
| `src/modeling_studio/evaluation/evaluator.py` | ✅ Done | Evaluation runner |
| `src/modeling_studio/evaluation/metrics.py` | ✅ Done | Per-task metric computation (12 functions) |
| `src/modeling_studio/evaluation/benchmarks.py` | ✅ Done | Benchmark suite runner (LatencyBenchmark, BenchmarkSuite, GLUEBenchmark, NERBenchmark, EmbeddingBenchmark, FamilyOSBenchmark) |
| `src/modeling_studio/evaluation/safety_eval.py` | 📝 Stub | Safety-specific evaluation |
| `src/modeling_studio/evaluation/forgetting_eval.py` | 📝 **NEW** | Catastrophic forgetting checks |
| `src/modeling_studio/evaluation/cultural_robustness.py` | 📝 **NEW** | Indian hyperbole false-positive tests |

### Scripts (📝 Stub)

| File | Status | Purpose |
|------|--------|---------|
| `scripts/train_stage_a.py` | 📝 Stub | Stage A: Generic multi-task training |
| `scripts/train_stage_b.py` | 📝 Stub | Stage B: FamilyOS domain adaptation |
| `scripts/evaluate.py` | 📝 Stub | Run evaluation on checkpoints |
| `scripts/calibrate_safety.py` | 📝 Stub | Calibrate safety thresholds |
| `scripts/export_model.py` | 📝 Stub | Export to ONNX/production format |
| `scripts/infer.py` | 📝 Stub | CLI inference tool |

### Config Files (✅ Created)

| File | Status | Purpose |
|------|--------|---------|
| `configs/model/encoder/modernbert_base.yaml` | ✅ Created | ModernBERT model config |
| `configs/training/multitask/stage_a_generic.yaml` | ✅ Created | Stage A training config |
| `configs/training/multitask/stage_b_familyos.yaml` | ✅ Created | Stage B training config |
| `configs/data/multitask/stage_a_datasets.yaml` | ✅ Created | Public dataset configs |
| `configs/data/multitask/stage_b_datasets.yaml` | ✅ Created | FamilyOS dataset configs |

### Test Files (📝 Stub)

| File | Status | Purpose |
|------|--------|---------|
| `tests/test_models.py` | 📝 Stub | Model unit tests |
| `tests/test_trainers.py` | 📝 Stub | Trainer unit tests |
| `tests/test_data.py` | 📝 Stub | Data pipeline tests |
| `tests/test_evaluation.py` | 📝 Stub | Evaluation tests |

### Data Directories

| Directory | Purpose |
|-----------|---------|
| `data/public/` | Public datasets (CoNLL, SST-2, GoEmotions, etc.) |
| `data/familyos/ner_family/` | Family NER annotations |
| `data/familyos/ingress/` | Ingress classification data |
| `data/familyos/safety/` | Safety policy band labels |
| `data/familyos/embeddings/` | Embedding sanity clusters |

---

## 🏁 Milestone 1: Data Pipeline Foundation

**Goal:** Load, preprocess, and tokenize all datasets for training

### Epic 1.1: Public Dataset Loaders

#### Issue 1.1.1: Implement NER Dataset Loader ✅

**File:** `src/modeling_studio/data/loaders.py`

**Tasks:**

- [x] Implement `load_ner_dataset()` function
- [x] Support HuggingFace datasets: `conll2003`, `ontonotes_5`
- [x] Support local JSONL with BIO tags
- [x] Apply label mapping from `labels.py` → `NER_GENERAL_LABELS`
- [x] Return standardized HF Dataset with columns: `tokens`, `ner_tags`

**Acceptance Criteria:**

```python
# Must pass before moving to next issue
from modeling_studio.data.loaders import load_ner_dataset
from modeling_studio.data.labels import NER_GENERAL_LABELS

ds = load_ner_dataset("conll2003", split="train")
assert "tokens" in ds.column_names
assert "ner_tags" in ds.column_names
assert all(tag in range(NER_GENERAL_LABELS.num_labels) for tag in ds[0]["ner_tags"])
print(f"✅ Loaded {len(ds)} NER samples")
```

---

#### Issue 1.1.2: Implement Classification Dataset Loader ✅

**File:** `src/modeling_studio/data/loaders.py`

**Tasks:**

- [x] Implement `load_classification_dataset()` function
- [x] Support: `sst2`, `imdb`, local CSV/JSONL
- [x] Handle binary vs multi-class labels (maps binary→5-class)
- [x] Column mapping: `text`, `label`

**Acceptance Criteria:**

```python
from modeling_studio.data.loaders import load_classification_dataset
from modeling_studio.data.labels import SENTIMENT_LABELS

ds = load_classification_dataset("sst2", split="train")
assert "text" in ds.column_names
assert "label" in ds.column_names
assert all(label in range(SENTIMENT_LABELS.num_labels) for label in ds["label"][:100])
print(f"✅ Loaded {len(ds)} classification samples")
```

---

#### Issue 1.1.3: Implement Multi-Label Dataset Loader ✅

**File:** `src/modeling_studio/data/loaders.py`

**Tasks:**

- [x] Implement `load_multilabel_dataset()` function
- [x] Support: `go_emotions`, `jigsaw_toxicity_pred`
- [x] Convert to multi-hot encoding (32-element vector for EMOTIONS_LABELS)
- [x] Handle label remapping for reduced label sets
- [x] Support local CSV/JSONL with string or integer labels

**Acceptance Criteria:**

```python
from modeling_studio.data.loaders import load_multilabel_dataset
from modeling_studio.data.labels import EMOTIONS_LABELS

ds = load_multilabel_dataset("go_emotions", split="train")
assert "text" in ds.column_names
assert "labels" in ds.column_names  # multi-hot vector
assert len(ds[0]["labels"]) == EMOTIONS_LABELS.num_labels
print(f"✅ Loaded {len(ds)} multi-label samples")
```

---

#### Issue 1.1.4: Implement NLI Dataset Loader ✅

**File:** `src/modeling_studio/data/loaders.py`

**Tasks:**

- [x] Implement `load_nli_dataset()` function
- [x] Support: `multi_nli`, `snli`, `anli`, `xnli`
- [x] Return columns: `premise`, `hypothesis`, `label`
- [x] Map labels to `NLI_LABELS` schema (entailment=0, neutral=1, contradiction=2)
- [x] Filter invalid labels (e.g., -1 for unlabeled in SNLI)
- [x] Support local JSONL files

**Acceptance Criteria:**

```python
from modeling_studio.data.loaders import load_nli_dataset
from modeling_studio.data.labels import NLI_LABELS

ds = load_nli_dataset("multi_nli", split="train")
assert all(col in ds.column_names for col in ["premise", "hypothesis", "label"])
assert all(label in range(NLI_LABELS.num_labels) for label in ds["label"][:100])
print(f"✅ Loaded {len(ds)} NLI pairs")
```

---

#### Issue 1.1.5: Implement Embedding Dataset Loader ✅

**File:** `src/modeling_studio/data/loaders.py`

**Tasks:**

- [x] Implement `load_embedding_dataset()` function
- [x] Support: `stsb`, `sickr`, `sts12-16` HuggingFace datasets
- [x] Support triplets format with `anchor`, `positive`, `negative` columns
- [x] Return columns: `sentence1`, `sentence2`, `score` (pairs) or `anchor`, `positive`, `negative` (triplets)
- [x] Normalize scores to 0.0-1.0 range
- [x] Support local CSV/JSONL files for both formats

**Acceptance Criteria:**

```python
from modeling_studio.data.loaders import load_embedding_dataset

ds = load_embedding_dataset("stsb", split="train")
assert "sentence1" in ds.column_names
assert "sentence2" in ds.column_names
assert "score" in ds.column_names
print(f"✅ Loaded {len(ds)} embedding pairs")
```

---

### Epic 1.2: FamilyOS Dataset Loaders

#### Issue 1.2.1: Implement Family NER Loader ✅

**File:** `src/modeling_studio/data/loaders.py`
**Data Dir:** `data/familyos/ner_family/`

**Tasks:**

- [x] Implement `load_familyos_ner()` function
- [x] Load from local JSONL in `data/familyos/ner_family/`
- [x] Apply `NER_FAMILY_LABELS` schema (21 BIO tags v2)
- [x] Validate BIO tag consistency
- [x] Support new entity types: TRADITION, MILESTONE, HEIRLOOM

**Acceptance Criteria:**

```python
from modeling_studio.data.loaders import load_familyos_ner
from modeling_studio.data.labels import NER_FAMILY_LABELS

ds = load_familyos_ner(split="train")
assert "tokens" in ds.column_names
assert "ner_tags" in ds.column_names
# Check for family-specific tags including new v2 types
sample_tags = [NER_FAMILY_LABELS.decode(t) for t in ds[0]["ner_tags"]]
assert NER_FAMILY_LABELS.num_labels == 21  # v2 enhanced
print(f"✅ Loaded {len(ds)} family NER samples, tags: {set(sample_tags)}")
```

---

#### Issue 1.2.2: Implement Ingress Classification Loader ✅

**File:** `src/modeling_studio/data/loaders.py`
**Data Dir:** `data/familyos/ingress/`

**Tasks:**

- [x] Implement `load_familyos_ingress()` function
- [x] Load from local JSONL with domain labels
- [x] Apply `INGRESS_LABELS` schema (12 domains v2)
- [x] Support new domains: SHOPPING, TRAVEL, EDUCATION, SOCIAL, AUTOMATION

**Acceptance Criteria:**

```python
from modeling_studio.data.loaders import load_familyos_ingress
from modeling_studio.data.labels import INGRESS_LABELS

ds = load_familyos_ingress(split="train")
assert "text" in ds.column_names
assert "label" in ds.column_names
assert INGRESS_LABELS.num_labels == 12  # v2 enhanced
unique_labels = set(ds["label"])
print(f"✅ Loaded {len(ds)} ingress samples, domains: {[INGRESS_LABELS.decode(l) for l in unique_labels]}")
```

---

#### Issue 1.2.3: Implement FamilyOS Safety Loader ✅

**File:** `src/modeling_studio/data/loaders.py`
**Data Dir:** `data/familyos/safety/`

**Tasks:**

- [x] Implement `load_familyos_safety()` function
- [x] Load policy band labels: GREEN, AMBER, RED, CRISIS
- [x] Apply `SAFETY_FAMILYOS_LABELS` schema (4 bands)

**Acceptance Criteria:**

```python
from modeling_studio.data.loaders import load_familyos_safety
from modeling_studio.data.labels import SAFETY_FAMILYOS_LABELS

ds = load_familyos_safety(split="train")
assert "text" in ds.column_names
assert "label" in ds.column_names
band_dist = {SAFETY_FAMILYOS_LABELS.decode(l): ds["label"].count(l) for l in range(4)}
print(f"✅ Loaded {len(ds)} safety samples, distribution: {band_dist}")
```

---

#### Issue 1.2.4: Implement Relation Extraction Loader (NEW v2) ✅

**File:** `src/modeling_studio/data/loaders.py`
**Data Dir:** `data/familyos/relations/`

**Tasks:**

- [x] Implement `load_familyos_relations()` function
- [x] Load entity pair relationship data from JSONL
- [x] Apply `RELATION_LABELS` schema (15 relation types)
- [x] Support new relation types: parent_of, child_of, sibling_of, spouse_of, etc.
- [x] Return columns: `text`, `entity1`, `entity2`, `relation`

**Acceptance Criteria:**

```python
from modeling_studio.data.loaders import load_familyos_relations
from modeling_studio.data.labels import RELATION_LABELS

ds = load_familyos_relations(split="train")
assert all(col in ds.column_names for col in ["text", "entity1", "entity2", "relation"])
assert RELATION_LABELS.num_labels == 15  # v2 NEW
unique_relations = set(ds["relation"])
print(f"✅ Loaded {len(ds)} relation samples, relations: {[RELATION_LABELS.decode(r) for r in unique_relations]}")
```

---

#### Issue 1.2.5: Implement Intent Classification Loader (NEW v2) ✅

**File:** `src/modeling_studio/data/loaders.py`
**Data Dir:** `data/familyos/intents/`

**Tasks:**

- [x] Implement `load_familyos_intents()` function
- [x] Load user intent classification data from JSONL
- [x] Apply `INTENT_LABELS` schema (8 intent types)
- [x] Support intent types: query, command, share, request, inform, schedule, remind, other

**Acceptance Criteria:**

```python
from modeling_studio.data.loaders import load_familyos_intents
from modeling_studio.data.labels import INTENT_LABELS

ds = load_familyos_intents(split="train")
assert "text" in ds.column_names
assert "label" in ds.column_names
assert INTENT_LABELS.num_labels == 8  # v2 NEW
unique_intents = set(ds["label"])
print(f"✅ Loaded {len(ds)} intent samples, intents: {[INTENT_LABELS.decode(i) for i in unique_intents]}")
```

---

#### Issue 1.2.6: Implement Temporal Extraction Loader (NEW v2) ✅

**File:** `src/modeling_studio/data/loaders.py`
**Data Dir:** `data/familyos/temporal/`

**Tasks:**

- [x] Implement `load_familyos_temporal()` function
- [x] Load temporal expression data (token classification)
- [x] Apply `TEMPORAL_LABELS` schema (13 BIO tags)
- [x] Support temporal types: DATE, TIME, DURATION, RELATIVE_DATE, RELATIVE_TIME, RECURRING

**Acceptance Criteria:**

```python
from modeling_studio.data.loaders import load_familyos_temporal
from modeling_studio.data.labels import TEMPORAL_LABELS

ds = load_familyos_temporal(split="train")
assert "tokens" in ds.column_names
assert "temporal_tags" in ds.column_names
assert TEMPORAL_LABELS.num_labels == 13  # v2 NEW
sample_tags = [TEMPORAL_LABELS.decode(t) for t in ds[0]["temporal_tags"]]
print(f"✅ Loaded {len(ds)} temporal samples, tags: {set(sample_tags)}")
```

---

### Epic 1.3: Tokenization Pipeline

#### Issue 1.3.1: Implement Tokenization Utilities ✅

**File:** `src/modeling_studio/data/tokenization.py`

**Tasks:**

- [x] Implement `load_tokenizer()` - loads ModernBERT tokenizer
- [x] Implement `tokenize_for_classification()` - sequence classification tokenization
- [x] Implement `tokenize_for_token_classification()` - NER with label alignment
- [x] Implement `tokenize_for_nli()` - premise/hypothesis pair tokenization
- [x] Handle subword alignment for NER (propagate labels to subwords)
- [x] Implement `tokenize_for_embedding()` - embedding generation
- [x] Implement `tokenize_for_relation()` - relation extraction with entity markers
- [x] Implement `align_labels_with_tokens()` - word-to-subword label alignment
- [x] Implement `get_tokenize_function()` - task-based tokenize function factory

**Acceptance Criteria:**

```python
from modeling_studio.data.tokenization import (
    load_tokenizer,
    tokenize_for_classification,
    tokenize_for_token_classification,
    tokenize_for_nli,
)

tokenizer = load_tokenizer("answerdotai/ModernBERT-base")

# Classification
result = tokenize_for_classification(tokenizer, "This is a test", max_length=128)
assert "input_ids" in result
assert "attention_mask" in result

# Token classification (NER)
result = tokenize_for_token_classification(
    tokenizer,
    tokens=["John", "lives", "in", "New", "York"],
    ner_tags=[1, 0, 0, 5, 6],  # B-PER, O, O, B-LOC, I-LOC
    max_length=128,
)
assert len(result["labels"]) == len(result["input_ids"])

# NLI
result = tokenize_for_nli(tokenizer, "The sky is blue", "It is daytime", max_length=128)
assert "input_ids" in result

print("✅ All tokenization tests passed")
```

---

### Epic 1.4: Multi-Task Dataset

#### Issue 1.4.1: Implement MultiTaskDataset ✅

**File:** `src/modeling_studio/data/multitask_dataset.py`

**Tasks:**

- [x] Implement `TaskDataset` wrapper class
- [x] Implement `MultiTaskDataset` combining multiple tasks
- [x] Store task name with each sample
- [x] Support task-specific preprocessing

**Acceptance Criteria:**

```python
from modeling_studio.data.multitask_dataset import MultiTaskDataset, TaskDataset

# Create task datasets
ner_ds = TaskDataset(name="ner_general", dataset=ner_hf_dataset)
sent_ds = TaskDataset(name="sentiment", dataset=sentiment_hf_dataset)

# Combine
multitask_ds = MultiTaskDataset([ner_ds, sent_ds])

sample = multitask_ds[0]
assert "task" in sample  # Which task this sample belongs to
assert sample["task"] in ["ner_general", "sentiment"]
print(f"✅ MultiTaskDataset with {len(multitask_ds)} total samples")
```

---

#### Issue 1.4.2: Implement Config-Based Dataset Loading ✅

**File:** `src/modeling_studio/data/loaders.py`
**Config:** `configs/data/multitask/stage_a_datasets.yaml`

**Tasks:**

- [x] Implement `load_from_config()` function
- [x] Parse dataset YAML config
- [x] Route to appropriate loader based on task type
- [x] Apply preprocessing and tokenization
- [x] Return dict of datasets (can be wrapped with `MultiTaskDataset`)

**Acceptance Criteria:**

```python
from modeling_studio.data.loaders import load_from_config

# Load all Stage A datasets from config (7 generic capabilities)
datasets = load_from_config("configs/data/multitask/stage_a_datasets.yaml")

assert "ner_general" in datasets
assert "sentiment" in datasets
assert "emotions" in datasets
assert "safety_generic" in datasets
assert "nli" in datasets
assert "embedding" in datasets
assert "temporal" in datasets  # NEW v2

for name, ds in datasets.items():
    print(f"  {name}: {len(ds)} samples")
print("✅ All Stage A datasets loaded from config (7 generic capabilities)")
```

---

## 🏁 Milestone 2: Training Infrastructure

**Goal:** Multi-task trainer with proper sampling, evaluation, and checkpointing

### Epic 2.1: Data Collation

#### Issue 2.1.1: Implement Task-Specific Collators ✅

**File:** `src/modeling_studio/trainers/collators.py`

**Tasks:**

- [x] Implement `SequenceClassificationCollator`
- [x] Implement `TokenClassificationCollator` (with padding for NER)
- [x] Implement `NLICollator`
- [x] Implement `MultiTaskCollator` (routes to task-specific collator)
- [x] Implement `MultiLabelCollator` (for emotions, safety_generic)
- [x] Implement `EmbeddingCollator` (triplets, pairs, simple format)
- [x] Implement `RelationCollator` (for relation extraction)

**Acceptance Criteria:**

```python
from modeling_studio.trainers.collators import MultiTaskCollator

collator = MultiTaskCollator(tokenizer=tokenizer)

# Batch with mixed tasks
batch = collator([
    {"task": "sentiment", "input_ids": [...], "labels": 1},
    {"task": "sentiment", "input_ids": [...], "labels": 0},
])

assert "input_ids" in batch
assert "attention_mask" in batch
assert "labels" in batch
assert batch["input_ids"].shape[0] == 2
print("✅ MultiTaskCollator works correctly")
```

---

### Epic 2.2: Task Sampling

#### Issue 2.2.1: Implement Task Sampler ✅

**File:** `src/modeling_studio/trainers/task_sampler.py`

**Tasks:**

- [x] Implement `TaskSampler` base class
- [x] Implement `ProportionalSampler` - sample proportional to dataset size
- [x] Implement `TemperatureSampler` - softmax with temperature
- [x] Implement `UniformSampler` - equal probability per task
- [x] Implement `SequentialSampler` - round-robin cycling
- [x] Implement `CurriculumSampler` - curriculum learning support
- [x] Implement `create_sampler()` factory function

**Acceptance Criteria:**

```python
from modeling_studio.trainers.task_sampler import (
    ProportionalSampler,
    TemperatureSampler,
    UniformSampler,
)

task_sizes = {"ner": 1000, "sentiment": 5000, "emotions": 2000}

# Proportional
sampler = ProportionalSampler(task_sizes)
samples = [sampler.sample() for _ in range(1000)]
# Sentiment should be sampled ~5x more than NER
assert samples.count("sentiment") > samples.count("ner") * 3

# Temperature
sampler = TemperatureSampler(task_sizes, temperature=2.0)
samples = [sampler.sample() for _ in range(1000)]
# More balanced with higher temperature

# Uniform
sampler = UniformSampler(list(task_sizes.keys()))
samples = [sampler.sample() for _ in range(1000)]
assert abs(samples.count("ner") - 333) < 50  # ~equal

print("✅ All samplers work correctly")
```

---

### Epic 2.3: Multi-Task Trainer

#### Issue 2.3.1: Implement MultiTaskTrainer Core ✅

**File:** `src/modeling_studio/trainers/multitask_trainer.py`

**Tasks:**

- [x] Extend `transformers.Trainer`
- [x] Override `get_train_dataloader()` for multi-task sampling
- [x] Override `compute_loss()` for task routing to correct head
- [x] Add `task_weights` parameter for loss weighting
- [x] Store `current_task` for proper head selection

**Acceptance Criteria:**

```python
from modeling_studio.trainers.multitask_trainer import MultiTaskTrainer
from modeling_studio.models import ModernBertMultiTaskModel

model = ModernBertMultiTaskModel.from_pretrained(
    "answerdotai/ModernBERT-base",
    capabilities=["ner_general", "sentiment"],
)

trainer = MultiTaskTrainer(
    model=model,
    args=training_args,
    train_datasets={"ner_general": ner_ds, "sentiment": sent_ds},
    task_weights={"ner_general": 1.0, "sentiment": 1.0},
)

# Verify dataloader yields batches with task info
dataloader = trainer.get_train_dataloader()
batch = next(iter(dataloader))
assert "task" in batch or hasattr(batch, "task")
print("✅ MultiTaskTrainer initializes correctly")
```

---

#### Issue 2.3.2: Implement Multi-Task Evaluation ✅

**File:** `src/modeling_studio/trainers/multitask_trainer.py`

**Tasks:**

- [x] Override `evaluation_loop()` for per-task evaluation
- [x] Compute metrics per task using `evaluation/metrics.py`
- [x] Aggregate into single metric for model selection
- [x] Log per-task metrics with task prefix

**Acceptance Criteria:**

```python
trainer = MultiTaskTrainer(...)
trainer.args.do_eval = True

# Run evaluation
metrics = trainer.evaluate()

assert "eval_ner_general_f1" in metrics
assert "eval_sentiment_accuracy" in metrics
assert "eval_avg_score" in metrics  # Aggregated
print(f"✅ Per-task metrics: {metrics}")
```

---

### Epic 2.4: Training Callbacks

#### Issue 2.4.1: Implement Training Callbacks ✅

**File:** `src/modeling_studio/trainers/callbacks.py`

**Tasks:**

- [x] Implement `TaskMetricsCallback` - logs per-task metrics
- [x] Implement `GradientMonitorCallback` - tracks gradient norms per task
- [x] Implement `EarlyStoppingCallback` - stop on no improvement
- [x] Implement `ModelCheckpointCallback` - save best per metric

**Acceptance Criteria:**

```python
from modeling_studio.trainers.callbacks import (
    TaskMetricsCallback,
    EarlyStoppingCallback,
)

callbacks = [
    TaskMetricsCallback(log_every=100),
    EarlyStoppingCallback(patience=3, metric="eval_avg_score"),
]

trainer = MultiTaskTrainer(..., callbacks=callbacks)
# Training should log task metrics and stop early if needed
print("✅ Callbacks configured correctly")
```

---

## 🏁 Milestone 3: Evaluation Pipeline

**Goal:** Comprehensive evaluation with per-task metrics and benchmarking

### Epic 3.1: Metrics Implementation

#### Issue 3.1.1: Implement Per-Task Metrics ✅

**File:** `src/modeling_studio/evaluation/metrics.py`

**Tasks:**

- [x] Implement `compute_ner_metrics()` - seqeval F1, precision, recall
- [x] Implement `compute_classification_metrics()` - accuracy, F1, precision, recall
- [x] Implement `compute_multilabel_metrics()` - micro/macro F1, hamming loss
- [x] Implement `compute_nli_metrics()` - accuracy, per-class F1
- [x] Implement `compute_embedding_metrics()` - Spearman correlation (STS)
- [x] Implement `compute_relation_metrics()` - F1 for relation classification (NEW v2)
- [x] Implement `compute_intent_metrics()` - accuracy, F1, confidence calibration (NEW v2)
- [x] Implement `compute_temporal_metrics()` - seqeval F1 for temporal spans (NEW v2)

**Acceptance Criteria:**

```python
from modeling_studio.evaluation.metrics import (
    compute_ner_metrics,
    compute_classification_metrics,
    compute_multilabel_metrics,
    compute_relation_metrics,
    compute_intent_metrics,
    compute_temporal_metrics,
)

# NER
ner_metrics = compute_ner_metrics(
    predictions=[[1, 2, 0, 0]],  # B-PER, I-PER, O, O
    references=[[1, 2, 0, 0]],
    label_list=["O", "B-PER", "I-PER", ...]
)
assert "f1" in ner_metrics
assert "precision" in ner_metrics

# Classification
cls_metrics = compute_classification_metrics(
    predictions=[0, 1, 2, 1],
    references=[0, 1, 2, 0],
)
assert "accuracy" in cls_metrics
assert "f1" in cls_metrics

# Relation (NEW v2)
rel_metrics = compute_relation_metrics(
    predictions=[0, 1, 2],
    references=[0, 1, 2],
)
assert "f1" in rel_metrics

# Intent (NEW v2)
intent_metrics = compute_intent_metrics(
    predictions=[0, 1],
    references=[0, 1],
    confidence_scores=[0.9, 0.85],
)
assert "accuracy" in intent_metrics
assert "calibration_error" in intent_metrics

# Temporal (NEW v2)
temporal_metrics = compute_temporal_metrics(
    predictions=[[9, 10, 0]],
    references=[[9, 10, 0]],
    label_list=["O", "B-DATE", "I-DATE", ...]
)
assert "f1" in temporal_metrics

print("✅ All metric functions work correctly (12 capabilities)")
```

---

### Epic 3.2: Evaluator Implementation

#### Issue 3.2.1: Implement Evaluator Class ✅

**File:** `src/modeling_studio/evaluation/evaluator.py`

**Tasks:**

- [x] Implement `Evaluator` class with batch inference
- [x] Implement `evaluate_task()` for single task evaluation
- [x] Implement `evaluate_all()` for all tasks
- [x] Support GPU/CPU inference
- [x] Return structured `EvalResults` object

**Acceptance Criteria:**

```python
from modeling_studio.evaluation.evaluator import Evaluator

evaluator = Evaluator(
    model=model,
    tokenizer=tokenizer,
    capabilities=["ner_general", "sentiment", "emotions"],
)

results = evaluator.evaluate_all(
    datasets={"ner_general": ner_test, "sentiment": sent_test, "emotions": emo_test},
    batch_size=32,
)

assert "ner_general" in results.per_task
assert "sentiment" in results.per_task
assert results.per_task["ner_general"]["f1"] > 0
print(f"✅ Evaluation results: {results.summary()}")
```

---

#### Issue 3.2.2: Implement Latency Benchmarking ✅

**File:** `src/modeling_studio/evaluation/benchmarks.py`

**Tasks:**

- [x] Implement `LatencyBenchmark` class
- [x] Warmup runs before measurement
- [x] Measure per-sample latency
- [x] Report P50, P95, P99
- [x] Memory profiling

**Acceptance Criteria:**

```python
from modeling_studio.evaluation.benchmarks import LatencyBenchmark

benchmark = LatencyBenchmark(model=model, tokenizer=tokenizer)

results = benchmark.run(
    texts=["Sample text " * 10] * 100,  # 100 samples
    batch_size=1,
    warmup=10,
    capability="sentiment",
)

assert "p50_ms" in results
assert "p95_ms" in results
assert "p99_ms" in results
assert "memory_mb" in results
print(f"✅ Latency: P50={results['p50_ms']:.1f}ms, P95={results['p95_ms']:.1f}ms")
```

---

#### Issue 3.2.3: Implement Benchmark Suite Framework ✅

**File:** `src/modeling_studio/evaluation/benchmarks.py`

**Tasks:**

- [x] Implement `BenchmarkSuite` class to orchestrate multiple benchmarks
- [x] Support parallel benchmark execution
- [x] Implement result aggregation and reporting
- [x] Support configuration-based benchmark selection

**Acceptance Criteria:**

```python
from modeling_studio.evaluation.benchmarks import BenchmarkSuite

suite = BenchmarkSuite(model=model, tokenizer=tokenizer)
suite.add_benchmark("latency", LatencyBenchmark(...))
suite.add_benchmark("glue", GLUEBenchmark(...))

results = suite.run_all()
assert "latency" in results
assert "glue" in results
print(f"✅ BenchmarkSuite completed all benchmarks")
```

---

#### Issue 3.2.4: Implement GLUE Benchmarks ✅

**File:** `src/modeling_studio/evaluation/benchmarks.py`

**Tasks:**

- [x] Implement `GLUEBenchmark` class
- [x] Support SST-2, CoLA, MRPC, QQP, STS-B, MNLI, QNLI, RTE, WNLI
- [x] Implement standardized metric reporting
- [x] Support subset selection for quick validation

**Acceptance Criteria:**

```python
from modeling_studio.evaluation.benchmarks import GLUEBenchmark

benchmark = GLUEBenchmark(model=model, tokenizer=tokenizer, tasks=["sst2", "mnli"])
results = benchmark.run()
assert "sst2_accuracy" in results
assert "mnli_accuracy" in results
```

---

#### Issue 3.2.5: Implement NER Benchmarks ✅

**File:** `src/modeling_studio/evaluation/benchmarks.py`

**Tasks:**

- [x] Implement `NERBenchmark` class
- [x] Support CoNLL-2003, OntoNotes 5.0, FamilyOS NER
- [x] Report per-entity-type F1 scores
- [x] Support custom label mappings

**Acceptance Criteria:**

```python
from modeling_studio.evaluation.benchmarks import NERBenchmark

benchmark = NERBenchmark(model=model, tokenizer=tokenizer)
results = benchmark.run(datasets=["conll2003", "familyos_ner"])
assert "conll2003_f1" in results
assert "familyos_ner_f1" in results
```

---

#### Issue 3.2.6: Implement Embedding Benchmarks ✅

**File:** `src/modeling_studio/evaluation/benchmarks.py`

**Tasks:**

- [x] Implement `EmbeddingBenchmark` class
- [x] Support STS-B, STS12-16, SICK-R
- [x] Report Spearman and Pearson correlation
- [x] Support triplet evaluation (accuracy@1, MRR)

**Acceptance Criteria:**

```python
from modeling_studio.evaluation.benchmarks import EmbeddingBenchmark

benchmark = EmbeddingBenchmark(model=model, tokenizer=tokenizer)
results = benchmark.run(datasets=["stsb", "sts12"])
assert "stsb_spearman" in results
assert "stsb_pearson" in results
```

---

#### Issue 3.2.7: Implement FamilyOS Domain Benchmarks ✅

**File:** `src/modeling_studio/evaluation/benchmarks.py`

**Tasks:**

- [x] Implement `FamilyOSBenchmark` class
- [x] Evaluate all 5 FamilyOS-specific capabilities
- [x] Support family-specific test scenarios
- [x] Report per-capability metrics with detailed breakdown

**Acceptance Criteria:**

```python
from modeling_studio.evaluation.benchmarks import FamilyOSBenchmark

benchmark = FamilyOSBenchmark(model=model, tokenizer=tokenizer)
results = benchmark.run()
assert "ner_family_f1" in results
assert "ingress_accuracy" in results
assert "safety_familyos_macro_f1" in results
```

---

#### Issue 3.2.8: Implement Baseline Comparison ✅

**File:** `src/modeling_studio/evaluation/benchmarks.py`

**Tasks:**

- [x] Implement `BaselineComparison` class
- [x] Compare against individual specialist models
- [x] Calculate relative improvement/regression
- [x] Generate comparison reports

**Acceptance Criteria:**

```python
from modeling_studio.evaluation.benchmarks import BaselineComparison

comparison = BaselineComparison(
    unified_model=model,
    baselines={"ner": ner_model, "sentiment": sent_model},
)
results = comparison.compare()
assert "ner_vs_baseline" in results
assert "overall_improvement" in results
```

---

#### Issue 3.2.9: Implement Result Tracking ✅

**File:** `src/modeling_studio/evaluation/benchmarks.py`

**Tasks:**

- [x] Implement `BenchmarkResultTracker` class
- [x] Store results with timestamps and model versions
- [x] Support result comparison across runs
- [x] Export to JSON/CSV for analysis

**Acceptance Criteria:**

```python
from modeling_studio.evaluation.benchmarks import BenchmarkResultTracker

tracker = BenchmarkResultTracker(output_dir="./benchmark_results")
tracker.log_result("latency", results, model_version="v1")
history = tracker.get_history("latency")
assert len(history) > 0
```

---

### Epic 3.3: Safety Evaluation

#### Issue 3.3.1: Implement Safety Evaluation Suite ✅

**File:** `src/modeling_studio/evaluation/safety_eval.py`

**Tasks:**

- [x] Implement `SafetyEvaluator` class
- [x] Test sets: self-harm, abuse, harassment, medical risk
- [x] Compute confusion matrix for safety bands
- [x] Calculate threshold-based metrics (at different operating points)
- [x] Compare against baseline safety model

**Acceptance Criteria:**

```python
from modeling_studio.evaluation.safety_eval import SafetyEvaluator

evaluator = SafetyEvaluator(
    model=model,
    tokenizer=tokenizer,
    capability="safety_familyos",
)

results = evaluator.evaluate(test_dataset)

assert "confusion_matrix" in results
assert "per_band_precision" in results
assert "per_band_recall" in results
assert "crisis_recall" in results  # Critical: must be high
assert results["crisis_recall"] > 0.95  # 95%+ recall on CRISIS
print(f"✅ Safety evaluation: CRISIS recall = {results['crisis_recall']:.2%}")
```

---

#### Issue 3.3.2: Implement Safety-Specific Metrics ✅

**File:** `src/modeling_studio/evaluation/safety_eval.py`

**Tasks:**

- [x] Implement `compute_safety_metrics()` function
- [x] Calculate per-band precision, recall, F1
- [x] Implement CRISIS recall with threshold sweep
- [x] Implement false positive rate at specific recall targets

**Acceptance Criteria:**

```python
from modeling_studio.evaluation.safety_eval import compute_safety_metrics

metrics = compute_safety_metrics(predictions, labels, logits)
assert "crisis_recall_at_95_precision" in metrics
assert "fpr_at_98_recall" in metrics
```

---

#### Issue 3.3.3: Implement Calibration Evaluation ✅

**File:** `src/modeling_studio/evaluation/safety_eval.py`

**Tasks:**

- [x] Implement `evaluate_calibration()` function
- [x] Calculate Expected Calibration Error (ECE)
- [x] Generate reliability diagrams
- [x] Support temperature scaling evaluation

**Acceptance Criteria:**

```python
from modeling_studio.evaluation.safety_eval import evaluate_calibration

calibration = evaluate_calibration(model, test_dataset)
assert "ece" in calibration
assert "reliability_diagram" in calibration
```

---

#### Issue 3.3.4: Implement Threshold Selection ✅

**File:** `src/modeling_studio/evaluation/safety_eval.py`

**Tasks:**

- [x] Implement `find_optimal_thresholds()` function
- [x] Support multiple operating points (high precision vs high recall)
- [x] Implement cost-sensitive threshold selection
- [x] Generate threshold recommendation report

**Acceptance Criteria:**

```python
from modeling_studio.evaluation.safety_eval import find_optimal_thresholds

thresholds = find_optimal_thresholds(logits, labels, cost_matrix=crisis_cost_matrix)
assert "crisis_threshold" in thresholds
assert "red_threshold" in thresholds
```

---

#### Issue 3.3.5: Implement Scenario Evaluation ✅

**File:** `src/modeling_studio/evaluation/safety_eval.py`

**Tasks:**

- [x] Implement `evaluate_scenarios()` function
- [x] Support predefined safety scenarios (self-harm, medical, abuse)
- [x] Calculate per-scenario metrics
- [x] Identify systematic failures

**Acceptance Criteria:**

```python
from modeling_studio.evaluation.safety_eval import evaluate_scenarios

scenario_results = evaluate_scenarios(model, scenario_datasets)
assert "self_harm_recall" in scenario_results
assert "medical_risk_f1" in scenario_results
```

---

#### Issue 3.3.6: Implement FamilyOS Safety Evaluation ✅

**File:** `src/modeling_studio/evaluation/safety_eval.py`

**Tasks:**

- [x] Implement `evaluate_familyos_safety()` function
- [x] Test family-specific safety scenarios
- [x] Evaluate child safety content handling
- [x] Test cultural/religious sensitivity

**Acceptance Criteria:**

```python
from modeling_studio.evaluation.safety_eval import evaluate_familyos_safety

familyos_results = evaluate_familyos_safety(model, familyos_test_data)
assert "child_safety_recall" in familyos_results
assert "cultural_sensitivity_score" in familyos_results
```

---

### Epic 3.4: Robustness Evaluation

#### Issue 3.4.1: Implement Catastrophic Forgetting Evaluation ✅

**File:** `src/modeling_studio/evaluation/forgetting_eval.py`

**Tasks:**

- [x] Implement `ForgettingEvaluator` class
- [x] Load and evaluate checkpoints from different training stages
- [x] Compare Stage A vs Stage B performance on generic tasks
- [x] Report per-task regression/improvement

**Acceptance Criteria:**

```python
from modeling_studio.evaluation.forgetting_eval import ForgettingEvaluator

evaluator = ForgettingEvaluator()
results = evaluator.compare(
    stage_a_model="checkpoints/stage_a",
    stage_b_model="checkpoints/stage_b",
    tasks=["ner_general", "sentiment", "nli"],
)
assert all(results[task]["regression"] <= 0.02 for task in results)  # ≤2% drop
```

---

#### Issue 3.4.2: Implement Cultural Robustness Evaluation ✅

**File:** `src/modeling_studio/evaluation/cultural_robustness.py`

**Tasks:**

- [x] Implement `CulturalRobustnessEvaluator` class
- [x] Create Indian hyperbole test cases
- [x] Evaluate false positive rates on culturally-specific expressions
- [x] Test multi-lingual/transliterated inputs

**Acceptance Criteria:**

```python
from modeling_studio.evaluation.cultural_robustness import CulturalRobustnessEvaluator

evaluator = CulturalRobustnessEvaluator(model=model)
results = evaluator.evaluate(indian_hyperbole_testset)
assert results["false_positive_rate"] < 0.05  # <5% FPR on hyperbole
```

---

### Epic 3.5: Model Components (Pre-Training Prerequisites)

> **Purpose:** These components are REQUIRED before any training can begin. They were originally in Milestone 5 but must be implemented earlier.

#### Issue 3.5.1: Implement Custom Loss Functions ✅

**File:** `src/modeling_studio/models/losses.py`

**Tasks:**

- [x] Implement `FocalLoss` - class imbalance handling for safety
- [x] Implement `LabelSmoothingCrossEntropy` - regularization
- [x] Implement `MultipleNegativesRankingLoss` - contrastive embedding loss
- [x] Implement `CosineSimilarityLoss` - embedding similarity training
- [x] Implement `TripletLoss` - triplet margin loss for embeddings
- [x] Implement `CRFLoss` - CRF layer for NER sequences
- [x] Implement `MultiTaskLoss` - combine losses across tasks
- [x] Implement `UncertaintyWeightedLoss` - Kendall uncertainty weighting (CRITICAL)

**Acceptance Criteria:**

```python
from modeling_studio.models.losses import (
    FocalLoss, UncertaintyWeightedLoss, MultipleNegativesRankingLoss
)

# Focal loss for imbalanced safety classification
focal = FocalLoss(alpha=0.25, gamma=2.0)
loss = focal(logits, targets)
assert loss.requires_grad

# Uncertainty weighting (CRITICAL for multi-task)
uw_loss = UncertaintyWeightedLoss(num_tasks=5)
combined_loss = uw_loss(task_losses=[loss1, loss2, loss3, loss4, loss5])
assert combined_loss.requires_grad
print("✅ All 8 loss functions implemented")
```

---

#### Issue 3.5.2: Implement Pooling Strategies (Moved from 5.0.1) ✅

**File:** `src/modeling_studio/models/poolers.py`

**Tasks:**

- [x] Implement `BasePooler` abstract class
- [x] Implement `CLSPooler` - extract [CLS] token representation
- [x] Implement `MeanPooler` - masked mean over sequence
- [x] Implement `MaxPooler` - masked max pooling
- [x] Implement `WeightedMeanPooler` - attention-weighted mean
- [x] Implement `LastTokenPooler` - for causal models

**Acceptance Criteria:**

```python
from modeling_studio.models.poolers import CLSPooler, MeanPooler, MaxPooler

hidden_states = torch.randn(2, 128, 768)
attention_mask = torch.ones(2, 128)

for Pooler in [CLSPooler, MeanPooler, MaxPooler]:
    pooler = Pooler(hidden_size=768)
    pooled = pooler(hidden_states, attention_mask)
    assert pooled.shape == (2, 768)
print("✅ All 6 poolers implemented")
```

---

#### Issue 3.5.3: Implement Data Preprocessing ✅

**File:** `src/modeling_studio/data/preprocessing.py`

**Tasks:**

- [x] Implement `TextPreprocessor` class with configurable pipeline
- [x] Implement `clean_text()` - normalize unicode, remove control chars
- [x] Implement task-specific preprocessors (NER entity alignment, etc.)
- [x] Implement FamilyOS-specific preprocessors (kinship term normalization)

**Acceptance Criteria:**

```python
from modeling_studio.data.preprocessing import TextPreprocessor

preprocessor = TextPreprocessor(
    lowercase=False,
    normalize_unicode=True,
    clean_whitespace=True,
)
clean = preprocessor("  Hello   World!!! ")
assert clean == "Hello World!!!"
print("✅ TextPreprocessor implemented")
```

---

#### Issue 3.5.4: Complete FamilyOS Data Loaders ✅

**File:** `src/modeling_studio/data/loaders.py`

**Tasks:**

- [x] Complete `load_familyos_relations()` implementation (line 2973)
- [x] Complete `load_familyos_intents()` implementation (line 3126)
- [x] Verify all 12 loaders fully functional

**Acceptance Criteria:**

```python
from modeling_studio.data.loaders import load_familyos_relations, load_familyos_intents

# Relations
rel_ds = load_familyos_relations(split="train")
assert all(col in rel_ds.column_names for col in ["text", "entity1", "entity2", "relation"])

# Intents
intent_ds = load_familyos_intents(split="train")
assert all(col in intent_ds.column_names for col in ["text", "label"])
print("✅ All FamilyOS loaders complete")
```

---

#### Issue 3.5.5: Implement Data Augmentation ✅

**File:** `src/modeling_studio/data/augmentation.py` (NEW)

**Tasks:**

- [x] Implement `FamilyAugmenter` class with kinship term variations
- [x] Support Indian + Western kinship variants (mom→mum/mummy/amma/aai)
- [x] Implement nickname pattern augmentation
- [x] Implement `back_translate()` for paraphrase generation
- [x] Support multiple languages (Hindi, Spanish, French)

**Acceptance Criteria:**

```python
from modeling_studio.data.augmentation import FamilyAugmenter, back_translate

augmenter = FamilyAugmenter()
augmented = augmenter.augment_kinship("Mom made dinner")
assert "Mummy made dinner" in augmented
assert "Amma made dinner" in augmented

# Back-translation
paraphrases = back_translate("Had a great day with family")
assert len(paraphrases) > 0
print("✅ Data augmentation working")
```

---

#### Issue 3.5.6: Implement Curriculum Learning Strategy ✅

**File:** `src/modeling_studio/trainers/curriculum.py` (NEW)

**Tasks:**

- [x] Implement `CurriculumScheduler` class
- [x] Support staged task introduction (easy→medium→hard)
- [x] Implement task difficulty scoring
- [x] Integration with MultiTaskTrainer

**Acceptance Criteria:**

```python
from modeling_studio.trainers.curriculum import CurriculumScheduler

scheduler = CurriculumScheduler(
    stages=[
        {"tasks": ["sentiment", "ner_general"], "epochs": 3},
        {"tasks": ["sentiment", "ner_general", "emotions", "nli"], "epochs": 3},
        {"tasks": "all", "epochs": 4},
    ]
)
current_tasks = scheduler.get_active_tasks(epoch=2)
assert current_tasks == ["sentiment", "ner_general"]
```

---

#### Issue 3.5.7: Implement FamilyContrastiveLoss ✅ COMPLETE

**File:** `src/modeling_studio/models/losses.py`

**Tasks:**

- [x] Implement `FamilyContrastiveLoss` for family memory retrieval
- [x] Support hard negative mining (same person different event, temporal neighbors)
- [x] Temperature-scaled InfoNCE loss

**Implementation Notes:**

- ~400 lines added to `losses.py`
- Temperature-scaled InfoNCE loss with configurable temperature (default 0.07)
- Multiple forward variants: standard, in-batch negatives, memory bank
- Static methods for hard negative mining: `mine_hard_negatives()` with strategies (hardest, semi-hard, random-hard)
- Family-specific hard negative creation: `create_family_hard_negatives()` supporting SPDE (Same Person Different Event) and temporal neighbors
- Learnable temperature option for fine-tuning

**Acceptance Criteria:**

```python
from modeling_studio.models.losses import FamilyContrastiveLoss

loss_fn = FamilyContrastiveLoss(temperature=0.07)
anchor = torch.randn(32, 768)
positive = torch.randn(32, 768)
negatives = torch.randn(32, 15, 768)
loss = loss_fn(anchor, positive, negatives)
assert loss.requires_grad
print("✅ FamilyContrastiveLoss working")
```

---

#### Issue 3.5.8: Implement Enhanced Safety Head ✅ COMPLETE

**File:** `src/modeling_studio/models/heads.py` (New class EnhancedSafetyHead)

**Tasks:**

- [x] Add keyword override detection (CRISIS keywords always trigger)
- [x] Add subcategory classification (12 subcategories)
- [x] Implement confidence calibration with learnable temperature
- [x] Support hierarchical classification (GREEN→AMBER→RED→CRISIS)

**Implementation Notes:**

- ~450 lines added to `heads.py`
- 4 safety bands: GREEN (0), AMBER (1), RED (2), CRISIS (3)
- 12 subcategories mapped to parent bands:
  - GREEN: general_safe, positive_interaction
  - AMBER: mild_profanity, sensitive_topic, boundary_test, emotional_distress
  - RED: harassment, explicit_content, misinformation, hate_speech
  - CRISIS: self_harm, violence_threat
- 25+ CRISIS keywords for override detection (e.g., "suicide", "kill myself")
- Hierarchical classification with subcategory masking based on predicted band
- Learnable temperature calibration
- Severity score computation utility

**Acceptance Criteria:**

```python
from modeling_studio.models.heads import EnhancedSafetyHead

head = EnhancedSafetyHead(hidden_size=768)
output = head(hidden_states, attention_mask, text="I want to kill myself")
assert output["band"] == "CRISIS"  # Keyword override
assert "subcategory" in output
```

---

### Epic 3.6: Production Readiness

#### Issue 3.6.1: Implement Unified NLP Output API ✅ COMPLETE

**File:** `src/modeling_studio/inference/unified_output.py` (NEW)

**Tasks:**

- [x] Implement `UnifiedNLPOutput` dataclass for all 12 capabilities
- [x] Implement `sys_nlp_infer()` function for batch inference
- [x] Support selective capability inference
- [x] Return structured output for K0 module integration

**Implementation Notes:**

- Created new `inference/` module with `unified_output.py` (~1200 lines)
- `Entity` dataclass: Named entity with text, label, start/end, confidence
- `Relation` dataclass: Extracted relation with subject, relation type, object
- `UnifiedNLPOutput` dataclass: Contains all 12 capability outputs
  - Token-level: ner_general, ner_family, temporal
  - Emotions: emotions (32 scores), primary_emotion, secondary_emotions
  - Sentiment: sentiment (5 levels), valence (0-1 score)
  - Safety: safety_generic (8 types), safety_familyos (4 bands), safety_score
  - Activity/Intent: ingress, intent with confidences
  - Relations: list of Relation objects
  - Embeddings: 768-dim vectors
  - NLI: nli_label, nli_confidence
- `get_unified_model()`: Model factory with caching
- `sys_nlp_infer()`: Main batch inference function
- Convenience functions: infer_entities(), infer_safety(), infer_sentiment(), infer_embeddings()

**Acceptance Criteria:**

```python
from modeling_studio.inference.unified_output import UnifiedNLPOutput, sys_nlp_infer

outputs = sys_nlp_infer(
    texts=["Mom took Panda to the park"],
    capabilities=["ner_family", "sentiment", "safety_familyos"],
)
assert outputs[0].ner_family is not None
assert outputs[0].sentiment is not None
assert outputs[0].safety_familyos in ["GREEN", "AMBER", "RED", "CRISIS"]
```

---

#### Issue 3.6.2: Document Rollout Plan ✅ COMPLETE

**File:** `docs/rollout_plan.md` (NEW)

**Tasks:**

- [x] Document Phase 1: Shadow Mode (Week 1-2)
- [x] Document Phase 2: Gradual Migration (Week 3-4)
- [x] Document Phase 3: Full Rollout (Week 5+)
- [x] Define rollback criteria (CRISIS recall <95%, FP >5%, latency >100ms, error >1%)
- [x] Document K0 module integration (M02, M04, M10, P08)

**Implementation Notes:**

- Created comprehensive rollout plan (~500 lines)
- Phase 1: Shadow mode with parallel inference, metric collection
- Phase 2: Gradual traffic shifting 10%→25%→50%→100%
- Phase 3: Full rollout with legacy model deprecation
- Detailed K0 integration for M02, M04, M10, P08
- Prometheus/Grafana monitoring dashboards specified

**Acceptance Criteria:**

- Rollout plan document exists and is reviewed
- Rollback criteria are clearly defined
- Integration steps for each K0 module are documented

---

#### Issue 3.6.3: Implement K0 Model Registry Integration ✅ COMPLETE

**File:** `src/modeling_studio/k0/runtime/model_registry.py` (NEW)

**Tasks:**

- [x] Add `familyos_unified_v2` entry to `MODEL_REGISTRY`
- [x] Implement `resolve_capability()` function to route capabilities to unified model
- [x] Implement `get_unified_model()` factory function
- [x] Support capability-based inference routing

**Implementation Notes:**

- Created new `k0/runtime/` module (~740 lines)
- `Capability` enum with all 12 capabilities
- `ModelInfo` and `HeadInfo` dataclasses for registry entries
- `MODEL_REGISTRY` with `familyos_unified_v2` entry
- `HEAD_REGISTRY` with all task-specific heads
- `CAPABILITY_ALIASES` for backward compatibility (e.g., "ner" → NER_GENERAL)
- `LEGACY_MODEL_MAPPING` for migration support
- `resolve_capability()`: Maps capability to (model_name, head_name)
- `get_unified_model()`: Model factory with caching
- `get_capability_for_module()`: K0 module to capability mapping
- Module init files for clean imports

**Acceptance Criteria:**

```python
from k0.runtime.model_registry import MODEL_REGISTRY, resolve_capability, get_unified_model

# Registry entry exists
assert "familyos_unified_v2" in MODEL_REGISTRY
assert MODEL_REGISTRY["familyos_unified_v2"]["capabilities"] == [
    "ner_general", "ner_family", "sentiment", "emotions", "safety_generic",
    "safety_familyos", "ingress", "embedding", "nli", "relation", "intent", "temporal"
]

# Capability routing
model_name, head_name = resolve_capability("ner_family")
assert model_name == "familyos_unified_v2"

# Model factory
model = get_unified_model()
assert hasattr(model, "infer")
print("✅ K0 Model Registry integration complete")
```

---

#### Issue 3.6.4: Implement K0 Module Migration Guide ✅ COMPLETE

**File:** `docs/k0_module_migration.md` (NEW)

**Tasks:**

- [x] Document M02 (hippocampus.semantic_project) migration
  - Before: `ner_transformer`, `spacy_nlp`
  - After: `model.infer(capabilities=["ner_general", "ner_family", "temporal"])`
- [x] Document M04 (affect.analyze) migration
  - Before: `vader_analyzer`, `go_emotions`, `sentiment_transformer`, `clinical_safety`
  - After: `model.infer(capabilities=["sentiment", "emotions", "safety_generic", "safety_familyos", "intent"])`
- [x] Document M10 (context.ingress_classify) migration
  - Before: `zero_shot_classifier` (1.5GB BART)
  - After: `model.infer(capabilities=["ingress"])`
- [x] Document P08 (embedding pipeline) migration
  - Before: `sentence_transformer.encode()` (384-dim)
  - After: `model.infer(capabilities=["embedding"])` (768-dim)

**Implementation Notes:**

- Comprehensive migration guide (~700 lines)
- Complete before/after code for all 4 modules
- Resource savings: 4,350 MB → 650 MB (85% reduction)
- Latency improvement: 150ms → 35ms (4.3x speedup)
- Shadow mode implementation with comparison wrapper
- Rollback procedures with automatic triggers
- Unit test update examples
- Dependency cleanup guide

**Acceptance Criteria:**

- Migration guide document exists
- Code examples for each module provided
- Memory savings documented (4350MB → 650MB)
- Latency improvements documented (150ms → 35ms)

---

#### Issue 3.6.5: Implement TemporalSafetyMonitor ✅ COMPLETE

**File:** `src/modeling_studio/evaluation/temporal_safety.py` (NEW)

**Tasks:**

- [x] Implement `SafetySignal` dataclass (band, text, timestamp, indicators)
- [x] Implement `SafetyEscalation` dataclass (from_band, to_band, reason)
- [x] Implement `TemporalSafetyMonitor` class
- [x] Track safety signals over time (configurable window, default 7 days)
- [x] Implement escalation rules:
  - 3+ AMBERs in window → escalate to RED
  - RED + isolation keywords ("alone", "nobody") → escalate to CRISIS
- [x] Integration with SafetyEvaluator

**Implementation Notes:**

- Created comprehensive temporal safety module (~860 lines)
- `SafetyBand` IntEnum: GREEN (0), AMBER (1), RED (2), CRISIS (3)
- `SafetySignal` dataclass: band, text, timestamp, indicators, confidence, user_id
- `SafetyEscalation` dataclass: from_band, to_band, reason, trigger_signal, contributing_signals
- `EscalationRule` dataclass for configurable rules with check functions
- `TemporalSafetyMonitor` class features:
  - Configurable window (default 7 days)
  - AMBER accumulation detection (3+ → RED)
  - Isolation pattern detection (RED + keywords → CRISIS)
  - Rapid escalation detection (GREEN→AMBER→RED in 48h)
  - Risk score computation
  - Custom rule support
- `SafetyMonitorIntegration` helper for SafetyEvaluator integration
- 30+ isolation keywords and 15+ escalation keywords
- Exported from evaluation module **init**.py

**Acceptance Criteria:**

```python
from modeling_studio.evaluation.temporal_safety import (
    TemporalSafetyMonitor, SafetySignal
)

monitor = TemporalSafetyMonitor(window_days=7)

# Track multiple AMBER signals
monitor.add_signal(SafetySignal(band="AMBER", text="Stressed about work", timestamp=day1))
monitor.add_signal(SafetySignal(band="AMBER", text="Feeling down", timestamp=day3))
escalation = monitor.add_signal(SafetySignal(band="AMBER", text="Can't sleep", timestamp=day5))

assert escalation is not None
assert escalation.to_band == "RED"
assert "3 AMBER signals" in escalation.reason
print("✅ TemporalSafetyMonitor escalation working")
```

---

#### Issue 3.6.6: Implement HierarchicalEmotionHead

**File:** `src/modeling_studio/models/heads.py` (UPDATE)

**Tasks:**

- [ ] Implement `HierarchicalEmotionHead` class
- [ ] Support primary emotion (single strongest emotion)
- [ ] Support secondary emotions (top-k additional emotions)
- [ ] Support emotion intensity scoring (0-1 scale)
- [ ] Integrate with existing emotions capability

**Acceptance Criteria:**

```python
from modeling_studio.models.heads import HierarchicalEmotionHead

head = HierarchicalEmotionHead(hidden_size=768, num_emotions=32)
output = head(hidden_states, attention_mask)

assert "primary_emotion" in output
assert "secondary_emotions" in output
assert "emotion_scores" in output
assert len(output["secondary_emotions"]) <= 3  # Top-3 secondary
print("✅ HierarchicalEmotionHead working")
```

---

#### Issue 3.6.7: Implement Indian English Support

**File:** `src/modeling_studio/data/cultural_mappings.py` (NEW)

**Tasks:**

- [ ] Define `INDIAN_ENGLISH_MAPPINGS` dictionary for expression normalization
- [ ] Define `INDIAN_VENTING_PATTERNS` for safety false-positive prevention
- [ ] Define `KINSHIP_VARIANTS` for Indian family terms (amma, appa, didi, bhai, etc.)
- [ ] Define `FAMILY_STRUCTURE_TYPES` (nuclear, joint_family, blended, etc.)
- [ ] Implement `IndianEnglishNormalizer` class for preprocessing

**Acceptance Criteria:**

```python
from modeling_studio.data.cultural_mappings import (
    INDIAN_ENGLISH_MAPPINGS,
    INDIAN_VENTING_PATTERNS,
    KINSHIP_VARIANTS,
    IndianEnglishNormalizer,
)

normalizer = IndianEnglishNormalizer()

# Expression mapping
assert normalizer.normalize("doing the needful") == "doing what's needed"
assert normalizer.normalize("I passed out from college") == "I graduated from college"

# Kinship variants
assert "amma" in KINSHIP_VARIANTS["mom"]
assert "bhai" in KINSHIP_VARIANTS["brother"]

# Venting patterns (should NOT trigger CRISIS)
assert "I'll die of embarrassment" in INDIAN_VENTING_PATTERNS
print("✅ Indian English support complete")
```

---

#### Issue 3.6.8: Implement Safety Subcategory Classification

**File:** `src/modeling_studio/models/heads.py` (UPDATE SafetyHead)

**Tasks:**

- [ ] Add `subcategory` output to SafetyHead
- [ ] Support 12 subcategories:
  - AMBER: stress, mild_sadness, frustration, health_mention
  - RED: persistent_sadness, isolation, hopelessness, substance
  - CRISIS: self_harm_ideation, suicide_ideation, harm_to_others, abuse_disclosure
- [ ] Implement hierarchical loss (band → subcategory)
- [ ] Return subcategory with band prediction

**Acceptance Criteria:**

```python
from modeling_studio.models.heads import SafetyHead
from modeling_studio.data.labels import SAFETY_SUBCATEGORIES

head = SafetyHead(hidden_size=768, num_bands=4, num_subcategories=12)
output = head(hidden_states, attention_mask)

assert "band" in output  # GREEN, AMBER, RED, CRISIS
assert "subcategory" in output  # e.g., "stress", "hopelessness"
assert "band_confidence" in output
assert "subcategory_confidence" in output
print("✅ Safety subcategory classification working")
```

---

## 🏁 Milestone 4: Stage A Training (Generic Multi-Task)

**Goal:** Train `modernbert-multitask-v0` on public datasets

### Epic 4.1: Training Script

#### Issue 4.1.1: Implement Stage A Training Script

**File:** `scripts/train_stage_a.py`
**Config:** `configs/training/multitask/stage_a_generic.yaml`

**Tasks:**

- [ ] Argument parsing (config file, overrides, resume)
- [ ] Load config and initialize model
- [ ] Load all Stage A datasets
- [ ] Initialize MultiTaskTrainer
- [ ] Run training with proper logging
- [ ] Save checkpoints and final model

**Acceptance Criteria:**

```bash
# Must complete without errors
python scripts/train_stage_a.py \
    --config configs/training/multitask/stage_a_generic.yaml \
    --training.max_steps 100  # Quick test

# Check outputs
ls checkpoints/modernbert-multitask-v0/
# Should contain: checkpoint-100/, config.json, capabilities.json

ls outputs/modernbert-multitask-v0/
# Should contain: model files, training_args.json, eval_results.json
```

---

#### Issue 4.1.2: Validate Stage A Model Quality

**File:** `scripts/evaluate.py`

**Tasks:**

- [ ] Run evaluation on all Stage A tasks
- [ ] Compare against individual baseline models
- [ ] Generate evaluation report

**Acceptance Criteria:**

```bash
python scripts/evaluate.py \
    --model checkpoints/modernbert-multitask-v0 \
    --config configs/data/multitask/stage_a_datasets.yaml \
    --output outputs/eval_stage_a.json

# Check quality thresholds
# NER F1 >= 88% (CoNLL-2003)
# Sentiment Acc >= 92% (SST-2)
# NLI Acc >= 85% (MNLI)
```

**Quality Gates (must pass to proceed to Milestone 5):**

| Task | Metric | Threshold |
|------|--------|-----------|
| ner_general | F1 | >= 88% |
| sentiment | Accuracy | >= 92% |
| emotions | Macro F1 | >= 45% |
| safety_generic | Macro F1 | >= 70% |
| nli | Accuracy | >= 84% |
| embedding | Spearman | >= 0.82 |
| temporal | F1 | >= 80% |

---

## 🏁 Milestone 5: Stage B Training (FamilyOS Domain Adaptation)

**Goal:** Adapt to FamilyOS domain with LoRA + 5 FamilyOS-specific heads + architecture enhancements

### Epic 5.0: Model Architecture Enhancements (Pre-Stage B)

> **Purpose:** Update `ModernBertMultiTaskModel` to match the enhanced architecture diagram before domain adaptation training.

#### Issue 5.0.1: Implement Shared Pooler Module

**File:** `src/modeling_studio/models/poolers.py`

**Tasks:**

- [ ] Implement `BasePooler` abstract class
- [ ] Implement `CLSPooler` - extract [CLS] token representation
- [ ] Implement `MeanPooler` - masked mean over sequence
- [ ] Implement `CLSMeanPooler` - combine CLS + Mean (as in diagram)
- [ ] Implement `MaxPooler` - masked max pooling
- [ ] Implement `WeightedMeanPooler` - attention-weighted mean

**Acceptance Criteria:**

```python
from modeling_studio.models.poolers import CLSMeanPooler, MeanPooler

pooler = CLSMeanPooler(hidden_size=768)
hidden_states = torch.randn(2, 128, 768)  # batch=2, seq=128
attention_mask = torch.ones(2, 128)

pooled = pooler(hidden_states, attention_mask)
assert pooled.shape == (2, 768)  # Combined CLS + Mean
print("✅ CLSMeanPooler works correctly")
```

---

#### Issue 5.0.2: Implement Task-Specific Adapters

**File:** `src/modeling_studio/models/adapters.py`

**Tasks:**

- [ ] Implement `BottleneckAdapter` - lightweight adapter layer
- [ ] Implement `TaskGroupAdapter` - adapter per task group (token, sequence, pair)
- [ ] Implement `AdapterConfig` dataclass for adapter hyperparameters
- [ ] Support freezing/unfreezing adapters independently
- [ ] Integrate with PEFT library for LoRA compatibility

**Acceptance Criteria:**

```python
from modeling_studio.models.adapters import BottleneckAdapter, TaskGroupAdapter

# Single adapter
adapter = BottleneckAdapter(hidden_size=768, bottleneck_size=64)
x = torch.randn(2, 128, 768)
out = adapter(x)
assert out.shape == x.shape
print(f"✅ BottleneckAdapter: {sum(p.numel() for p in adapter.parameters())} params")

# Task group adapter
group_adapter = TaskGroupAdapter(
    hidden_size=768,
    task_groups=["token_tasks", "sequence_tasks", "pair_tasks"],
    bottleneck_size=64,
)
out = group_adapter(x, task_group="sequence_tasks")
assert out.shape == x.shape
print("✅ TaskGroupAdapter works correctly")
```

---

#### Issue 5.0.3: Implement Cross-Attention Pair Encoder

**File:** `src/modeling_studio/models/pair_encoder.py`

**Tasks:**

- [ ] Implement `CrossAttentionPairEncoder` for NLI/Relation tasks
- [ ] Support premise-hypothesis cross-attention
- [ ] Support entity pair cross-attention for relations
- [ ] Add optional residual connections
- [ ] Make backward compatible (fallback to concatenation)

**Acceptance Criteria:**

```python
from modeling_studio.models.pair_encoder import CrossAttentionPairEncoder

encoder = CrossAttentionPairEncoder(hidden_size=768, num_heads=8)

# For NLI: premise and hypothesis representations
premise = torch.randn(2, 64, 768)
hypothesis = torch.randn(2, 32, 768)
premise_mask = torch.ones(2, 64)
hypothesis_mask = torch.ones(2, 32)

pair_repr = encoder(premise, hypothesis, premise_mask, hypothesis_mask)
assert pair_repr.shape == (2, 768)
print("✅ CrossAttentionPairEncoder works correctly")
```

---

#### Issue 5.0.4: Update ModernBertMultiTaskModel with Enhancements

**File:** `src/modeling_studio/models/modernbert_multitask.py`

**Tasks:**

- [ ] Add `shared_pooler` parameter (use CLSMeanPooler by default)
- [ ] Add `use_adapters` parameter for task-group adapters
- [ ] Add `pair_encoder` for NLI/Relation heads
- [ ] Refactor heads to use shared pooler instead of internal pooling
- [ ] Maintain backward compatibility (adapters disabled by default)
- [ ] Update `from_pretrained()` to support new architecture options

**Acceptance Criteria:**

```python
from modeling_studio.models import ModernBertMultiTaskModel

# Enhanced model with adapters and shared pooler
model = ModernBertMultiTaskModel.from_pretrained(
    "answerdotai/ModernBERT-base",
    capabilities=["ner_general", "sentiment", "nli", "relation"],
    use_adapters=True,  # Enable task-group adapters
    adapter_bottleneck_size=64,
    use_shared_pooler=True,  # Use CLSMeanPooler
    use_cross_attention_pair_encoder=True,  # For NLI/Relation
)

# Verify architecture
assert hasattr(model, "shared_pooler")
assert hasattr(model, "adapters")
assert hasattr(model, "pair_encoder")

# Forward pass still works
outputs = model(input_ids, attention_mask, capability="sentiment")
print("✅ Enhanced ModernBertMultiTaskModel works correctly")
```

---

#### Issue 5.0.5: Update Heads to Use Shared Components

**File:** `src/modeling_studio/models/heads.py`

**Tasks:**

- [ ] Refactor `SequenceClassificationHead` to accept external pooler
- [ ] Refactor `NLIHead` to use `CrossAttentionPairEncoder`
- [ ] Refactor `RelationHead` to use `CrossAttentionPairEncoder`
- [ ] Add `use_external_pooler` flag for backward compatibility
- [ ] Update head initialization in model

**Acceptance Criteria:**

```python
from modeling_studio.models.heads import SequenceClassificationHead
from modeling_studio.models.poolers import CLSMeanPooler

# Head with external pooler
pooler = CLSMeanPooler(hidden_size=768)
head = SequenceClassificationHead(
    hidden_size=768,
    num_labels=5,
    external_pooler=pooler,  # Use shared pooler
)

hidden_states = torch.randn(2, 128, 768)
attention_mask = torch.ones(2, 128)
output = head(hidden_states, attention_mask)
assert output["logits"].shape == (2, 5)
print("✅ Head with external pooler works correctly")
```

---

### Epic 5.1: Domain Adaptation

#### Issue 5.1.1: Implement Stage B Training Script

**File:** `scripts/train_stage_b.py`
**Config:** `configs/training/multitask/stage_b_familyos.yaml`

**Tasks:**

- [ ] Load `modernbert-multitask-v0` as base (7 generic capabilities)
- [ ] Add FamilyOS-specific heads (ner_family, ingress, safety_familyos, relation, intent)
- [ ] Apply LoRA to encoder layers (rank=16, alpha=32)
- [ ] Mix FamilyOS data with public data (prevent forgetting)
- [ ] Train and save as `modernbert-unified-v2`

**Acceptance Criteria:**

```bash
python scripts/train_stage_b.py \
    --config configs/training/multitask/stage_b_familyos.yaml \
    --base_model checkpoints/modernbert-multitask-v0

# Check outputs
ls checkpoints/modernbert-unified-v2/
# Should contain: adapter_model.safetensors, capabilities.json (with 12 capabilities)
```

---

### Epic 5.2: Safety Calibration

#### Issue 5.2.1: Implement Safety Threshold Calibration

**File:** `scripts/calibrate_safety.py`

**Tasks:**

- [ ] Run inference on held-out safety data
- [ ] Compute optimal thresholds per band
- [ ] Apply temperature scaling
- [ ] Save calibration config

**Acceptance Criteria:**

```bash
python scripts/calibrate_safety.py \
    --model checkpoints/modernbert-unified-v1 \
    --data data/familyos/safety/calibration.jsonl \
    --output configs/calibration/safety_thresholds.yaml

# Verify calibration
cat configs/calibration/safety_thresholds.yaml
# Should contain thresholds for GREEN/AMBER/RED/CRISIS
```

---

#### Issue 5.2.2: Validate FamilyOS Model Quality

**File:** `scripts/evaluate.py`

**Tasks:**

- [ ] Evaluate all 12 capabilities (7 generic + 5 FamilyOS)
- [ ] Verify no regression on Stage A tasks
- [ ] Verify FamilyOS task quality
- [ ] Verify new v2 capabilities (relation, intent, temporal)

**Acceptance Criteria:**

**Quality Gates (must pass):**

| Task | Metric | Threshold |
|------|--------|-----------|
| ner_family | F1 | >= 85% |
| ingress | Accuracy | >= 90% |
| safety_familyos | Macro F1 | >= 80% |
| safety_familyos (CRISIS) | Recall | >= 95% |
| relation | F1 | >= 75% |
| intent | Accuracy | >= 85% |
| ner_general | F1 | >= 86% (≤2% drop) |
| sentiment | Accuracy | >= 90% (≤2% drop) |
| temporal | F1 | >= 78% (≤2% drop) |

---

## 🏁 Milestone 6: Export & Inference

**Goal:** Production-ready model export and inference API

### Epic 6.1: Model Export

#### Issue 6.1.1: Implement Model Export Script

**File:** `scripts/export_model.py`

**Tasks:**

- [ ] Export to HuggingFace format (safetensors)
- [ ] Export capabilities.json
- [ ] Export calibration config
- [ ] Optional: ONNX export

**Acceptance Criteria:**

```bash
python scripts/export_model.py \
    --model checkpoints/modernbert-unified-v1 \
    --output outputs/familyos-modernbert-unified-v1 \
    --format safetensors

# Verify export
ls outputs/familyos-modernbert-unified-v1/
# config.json, model.safetensors, capabilities.json, tokenizer files
```

---

### Epic 6.2: Inference API

#### Issue 6.2.1: Implement Inference Script

**File:** `scripts/infer.py`

**Tasks:**

- [ ] CLI for single/batch inference
- [ ] Support all 12 capabilities (v2)
- [ ] Output JSON results with UnifiedNLPOutput format
- [ ] Latency reporting
- [ ] Support new v2 outputs: relation, intent, temporal

**Acceptance Criteria:**

```bash
# Multiple capabilities (v2 enhanced)
python scripts/infer.py \
    --model outputs/familyos-modernbert-unified-v2 \
    --text "Mummy said we should pick up Grandma at 3pm tomorrow" \
    --capabilities ner_family,sentiment,safety_familyos,relation,intent,temporal

# Output (v2 format):
# {
#   "text": "...",
#   "ner_family": [{"entity": "KINSHIP", "word": "Mummy", "start": 0, "end": 5}, {"entity": "KINSHIP", "word": "Grandma", "start": 30, "end": 37}],
#   "sentiment": {"label": "neutral", "confidence": 0.85},
#   "safety_familyos": {"band": "GREEN", "confidence": 0.98},
#   "relation": [{"entity1": "Mummy", "entity2": "Grandma", "relation": "parent_of", "confidence": 0.72}],
#   "intent": {"label": "inform", "confidence": 0.91},
#   "temporal": [{"type": "TIME", "text": "3pm", "normalized": "15:00"}, {"type": "RELATIVE_DATE", "text": "tomorrow"}]
# }
```

---

## 🏁 Milestone 7: Testing & Quality Assurance

**Goal:** Comprehensive tests ensuring reliability

### Epic 7.1: Unit Tests

#### Issue 7.1.1: Model Unit Tests

**File:** `tests/test_models.py`

**Tasks:**

- [ ] Test model initialization
- [ ] Test forward pass for each capability
- [ ] Test head freezing/unfreezing
- [ ] Test save/load roundtrip

**Acceptance Criteria:**

```bash
pytest tests/test_models.py -v
# All tests pass
```

---

#### Issue 7.1.2: Data Pipeline Tests

**File:** `tests/test_data.py`

**Tasks:**

- [ ] Test each loader function
- [ ] Test tokenization functions
- [ ] Test MultiTaskDataset
- [ ] Test config loading

**Acceptance Criteria:**

```bash
pytest tests/test_data.py -v
# All tests pass
```

---

#### Issue 7.1.3: Trainer Tests

**File:** `tests/test_trainers.py`

**Tasks:**

- [ ] Test MultiTaskTrainer initialization
- [ ] Test task sampling
- [ ] Test collators
- [ ] Test training step (mock)

**Acceptance Criteria:**

```bash
pytest tests/test_trainers.py -v
# All tests pass
```

---

#### Issue 7.1.4: Evaluation Tests

**File:** `tests/test_evaluation.py`

**Tasks:**

- [ ] Test metric computation
- [ ] Test Evaluator class
- [ ] Test safety evaluation

**Acceptance Criteria:**

```bash
pytest tests/test_evaluation.py -v
# All tests pass
```

---

## 📋 Summary: Issue Dependency Graph (Enhanced v2 - Updated)

```
Milestone 1: Data Pipeline (12 capabilities)
├── Epic 1.1: Public Dataset Loaders (7 generic) ✅ ALL DONE
│   ├── Issue 1.1.1: NER Loader ✅ ─────────────┐
│   ├── Issue 1.1.2: Classification Loader ✅ ──┤
│   ├── Issue 1.1.3: Multi-Label Loader ✅ ─────┼──► Epic 1.4
│   ├── Issue 1.1.4: NLI Loader ✅ ─────────────┤
│   └── Issue 1.1.5: Embedding Loader ✅ ───────┘
├── Epic 1.2: FamilyOS Dataset Loaders (5 FamilyOS + temporal)
│   ├── Issue 1.2.1: Family NER Loader ✅ ──────┐
│   ├── Issue 1.2.2: Ingress Loader ✅ ─────────┤
│   ├── Issue 1.2.3: FamilyOS Safety Loader ✅ ─┤
│   ├── Issue 1.2.4: Relation Loader ⚠️ STUB ───┼──► Epic 1.4
│   ├── Issue 1.2.5: Intent Loader ⚠️ STUB ─────┤
│   └── Issue 1.2.6: Temporal Loader ✅ ────────┘
├── Epic 1.3: Tokenization
│   └── Issue 1.3.1: Tokenization Utilities ✅ ─► Epic 1.4
└── Epic 1.4: Multi-Task Dataset
    ├── Issue 1.4.1: MultiTaskDataset ──────────► Milestone 2
    └── Issue 1.4.2: Config-Based Loading ──────► Milestone 2

Milestone 2: Training Infrastructure
├── Epic 2.1: Data Collation
│   └── Issue 2.1.1: Task-Specific Collators ───► Epic 2.3
├── Epic 2.2: Task Sampling
│   └── Issue 2.2.1: Task Sampler ──────────────► Epic 2.3
├── Epic 2.3: Multi-Task Trainer
│   ├── Issue 2.3.1: MultiTaskTrainer Core ✅ ──► Milestone 4
│   └── Issue 2.3.2: Multi-Task Evaluation ✅ ──► Milestone 4
└── Epic 2.4: Training Callbacks
    └── Issue 2.4.1: Training Callbacks ✅ ─────► Milestone 4

Milestone 3: Evaluation Pipeline (12 capabilities) ◄── CURRENT
├── Epic 3.1: Metrics (12 metric functions)
│   └── Issue 3.1.1: Per-Task Metrics ✅ ────────► Epic 3.2
├── Epic 3.2: Evaluator & Benchmarks
│   ├── Issue 3.2.1: Evaluator Class ✅ ─────────► Milestone 4
│   ├── Issue 3.2.2: Latency Benchmarking ✅ ────► Milestone 4
│   ├── Issue 3.2.3: Benchmark Suite Framework ──► Milestone 4 ◄── NEW
│   ├── Issue 3.2.4: GLUE Benchmarks ────────────► Milestone 4 ◄── NEW
│   ├── Issue 3.2.5: NER Benchmarks ─────────────► Milestone 4 ◄── NEW
│   ├── Issue 3.2.6: Embedding Benchmarks ───────► Milestone 4 ◄── NEW
│   ├── Issue 3.2.7: FamilyOS Domain Benchmarks ─► Milestone 4 ◄── NEW
│   ├── Issue 3.2.8: Baseline Comparison ────────► Milestone 4 ◄── NEW
│   └── Issue 3.2.9: Result Tracking ────────────► Milestone 4 ◄── NEW
├── Epic 3.3: Safety Evaluation
│   ├── Issue 3.3.1: Safety Evaluation Suite ────► Milestone 5
│   ├── Issue 3.3.2: Safety-Specific Metrics ────► Milestone 5 ◄── NEW
│   ├── Issue 3.3.3: Calibration Evaluation ─────► Milestone 5 ◄── NEW
│   ├── Issue 3.3.4: Threshold Selection ────────► Milestone 5 ◄── NEW
│   ├── Issue 3.3.5: Scenario Evaluation ────────► Milestone 5 ◄── NEW
│   └── Issue 3.3.6: FamilyOS Safety Eval ───────► Milestone 5 ◄── NEW
├── Epic 3.4: Robustness Evaluation ◄── NEW
│   ├── Issue 3.4.1: Forgetting Evaluation ──────► Milestone 5 ◄── NEW
│   └── Issue 3.4.2: Cultural Robustness ────────► Milestone 5 ◄── NEW
└── Epic 3.5: Model Components (Pre-Training) ◄── CRITICAL NEW
    ├── Issue 3.5.1: Custom Loss Functions ──────► Milestone 4 ◄── CRITICAL
    ├── Issue 3.5.2: Pooling Strategies ─────────► Milestone 4 ◄── NEW (moved from 5.0.1)
    ├── Issue 3.5.3: Data Preprocessing ─────────► Milestone 4 ◄── NEW
    ├── Issue 3.5.4: Complete FamilyOS Loaders ──► Milestone 4 ◄── NEW
    ├── Issue 3.5.5: Data Augmentation ──────────► Milestone 4 ◄── NEW
    ├── Issue 3.5.6: Curriculum Learning ────────► Milestone 4 ◄── NEW (optional)
    ├── Issue 3.5.7: FamilyContrastiveLoss ──────► Milestone 4 ◄── NEW
    └── Issue 3.5.8: Enhanced Safety Head ───────► Milestone 5 ◄── NEW
└── Epic 3.6: Production Readiness ◄── NEW
    ├── Issue 3.6.1: UnifiedNLPOutput API ───────► Milestone 6 ◄── NEW
    ├── Issue 3.6.2: Rollout Plan Document ──────► Milestone 6 ◄── NEW
    ├── Issue 3.6.3: K0 Model Registry ──────────► Milestone 6 ◄── NEW (GAP FIX)
    ├── Issue 3.6.4: K0 Module Migration ────────► Milestone 6 ◄── NEW (GAP FIX)
    ├── Issue 3.6.5: TemporalSafetyMonitor ──────► Milestone 5 ◄── NEW (GAP FIX)
    ├── Issue 3.6.6: HierarchicalEmotionHead ────► Milestone 5 ◄── NEW (GAP FIX)
    ├── Issue 3.6.7: Indian English Support ─────► Milestone 4 ◄── NEW (GAP FIX)
    └── Issue 3.6.8: Safety Subcategories ───────► Milestone 5 ◄── NEW (GAP FIX)

Milestone 4: Stage A Training (7 generic) ────────► Milestone 5
├── Epic 4.1: Training Script
│   ├── Issue 4.1.1: Stage A Training Script
│   └── Issue 4.1.2: Validate Stage A Quality (GATE: 7 metrics)

Milestone 5: Stage B Training (12 total) ─────────► Milestone 6
├── Epic 5.0: Model Architecture Enhancements (Pre-Stage B)
│   ├── Issue 5.0.1: Shared Pooler Module (MOVED to 3.5.2)
│   ├── Issue 5.0.2: Task-Specific Adapters ─────┐
│   ├── Issue 5.0.3: Cross-Attention Pair ───────┼──► Issue 5.0.4
│   ├── Issue 5.0.4: Update Model ───────────────┘
│   └── Issue 5.0.5: Update Heads ───────────────► Epic 5.1
├── Epic 5.1: Domain Adaptation (LoRA + 5 FamilyOS heads)
│   └── Issue 5.1.1: Stage B Training Script
└── Epic 5.2: Safety Calibration
    ├── Issue 5.2.1: Safety Threshold Calibration
    └── Issue 5.2.2: Validate FamilyOS Quality (GATE: 9 metrics)

Milestone 6: Export & Inference (v2)
├── Epic 6.1: Model Export
│   └── Issue 6.1.1: Model Export Script
└── Epic 6.2: Inference API (UnifiedNLPOutput format)
    └── Issue 6.2.1: Inference Script (12 capabilities)

Milestone 7: Testing (Parallel with M1-M6)
├── Issue 7.1.1: Model Unit Tests (9 head types)
├── Issue 7.1.2: Data Pipeline Tests (12 loaders)
├── Issue 7.1.3: Trainer Tests
└── Issue 7.1.4: Evaluation Tests (8 metric functions)
```

---

## 📊 Progress Summary

### Completed Issues (✅)

| Issue | Description |
|-------|-------------|
| 1.1.1-1.1.5 | All public dataset loaders |
| 1.2.1-1.2.3, 1.2.6 | FamilyOS NER, Ingress, Safety, Temporal loaders |
| 1.3.1 | Tokenization utilities |
| 2.3.1 | MultiTaskTrainer Core |
| 2.3.2 | Multi-Task Evaluation |
| 2.4.1 | Training Callbacks |
| 3.1.1 | Per-Task Metrics (12 functions) |
| 3.2.1 | Evaluator Class |
| 3.2.2 | Latency Benchmarking |

### Next Priority (🔴 Blocking)

| Issue | Description | Why Blocking |
|-------|-------------|--------------|
| **3.5.1** | Custom Loss Functions | **UncertaintyWeightedLoss needed for multi-task training** |
| **3.5.2** | Pooling Strategies | Required by all sequence heads |
| **3.5.4** | Complete FamilyOS Loaders | Relations & Intents loaders are stubs |
| 3.2.3-3.2.9 | Benchmark Suite | Model comparison infrastructure |
| 3.3.1-3.3.6 | Safety Evaluation | CRISIS recall validation |

---

## 🎯 Quick Reference: Files to Implement (Enhanced v2)

### Priority 1 (Milestone 1-2): Foundation

1. `src/modeling_studio/data/loaders.py` - Complete FamilyOS loaders (relations, intents)
2. `src/modeling_studio/data/preprocessing.py` - TextPreprocessor ◄── NEW
3. `src/modeling_studio/data/multitask_dataset.py` - Dataset classes
4. `src/modeling_studio/trainers/collators.py` - Data collators
5. `src/modeling_studio/trainers/task_sampler.py` - Task sampling
6. `src/modeling_studio/trainers/multitask_trainer.py` - ✅ Done

### Priority 2 (Milestone 3): Evaluation & Components

7. `src/modeling_studio/evaluation/metrics.py` - ✅ Done (12 functions)
8. `src/modeling_studio/evaluation/evaluator.py` - ✅ Done
9. `src/modeling_studio/evaluation/benchmarks.py` - 8 benchmark types remaining
10. `src/modeling_studio/evaluation/safety_eval.py` - 6 implementations needed
11. `src/modeling_studio/evaluation/forgetting_eval.py` - ◄── NEW
12. `src/modeling_studio/evaluation/cultural_robustness.py` - ◄── NEW
13. `src/modeling_studio/models/losses.py` - **8 loss functions (CRITICAL)** ◄── NEW
14. `src/modeling_studio/models/poolers.py` - 6 pooler implementations ◄── NEW

### Priority 3 (Milestone 4-6): Scripts

15. `scripts/train_stage_a.py` - Stage A training (7 generic)
16. `scripts/train_stage_b.py` - Stage B training (12 total)
17. `scripts/evaluate.py` - Evaluation runner
18. `scripts/calibrate_safety.py` - Safety calibration
19. `scripts/export_model.py` - Model export
20. `scripts/infer.py` - Inference CLI (UnifiedNLPOutput v2)

### Priority 4: Tests (Continuous)

21. `tests/test_models.py` - 9 head types
22. `tests/test_data.py` - 12 loaders
23. `tests/test_trainers.py` - ✅ 62 tests passing
24. `tests/test_evaluation.py` - 8 metrics
