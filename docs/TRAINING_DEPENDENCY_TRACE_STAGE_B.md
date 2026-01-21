# Stage B Training Pipeline Dependency Trace

**Purpose**: Complete architecture trace for Stage B (FamilyOS domain adaptation) training to guide improvements for ALL 12 task heads.

**Root Entry Point**: `scripts/train_stage_b.py`

**Stage B Goal**: Train v2 encoder layers 15-20 with FamilyOS data → These become v3 layers 23-28

---

## Level 0: Entry Point

### `scripts/train_stage_b.py`

- FamilyOS domain adaptation training
- Full fine-tuning mode (no LoRA) for v3 preparation
- **12 capabilities total**: 5 new FamilyOS + 7 Stage A replay
- Layer-wise learning rates (L15-20 get 5e-5, highest)
- Output: `modernbert-v2-for-v3-transfer`

---

## Level 1: Configuration Files

### `configs/training/multitask/stage_b_for_v3_prep.yaml`

- **Training mode**: Full fine-tuning (no LoRA)
- **Layer-wise LR strategy**: Strongly train layers 15-20 (future v3 L23-28)
- **Task weights**: Emphasize FamilyOS tasks (safety: 5.0, relation: 2.0, ner_family: 2.0)
- **Batch config**: 128 per device (A100 optimized for short sequences)
- **Epochs**: 8 (full fine-tuning needs more)
- **12 heads configuration** (all enabled)

### `configs/data/multitask/stage_b_datasets.yaml`

- **FamilyOS datasets**: Gold (curated) + Silver (LLM-generated)
- **Stage A replay**: 15% to prevent catastrophic forgetting
- **Data mix**:
  - ner_family: 12.7K silver
  - ingress: 22.7K silver
  - safety_familyos: 14.3K silver
  - intent: 16K silver
  - relation: 21.3K silver
  - Replay: NER (5K), NLI (10K), emotions (5K), sentiment (5K)

### `configs/model/encoder/modernbert_base.yaml`

- Base encoder (same as Stage A)
- 768-dim, 22 layers, 12 attention heads

---

## Level 2: Core Modules (from train_stage_b.py imports)

### `src/modeling_studio/data/__init__.py`

- **Imports**: load_stage_b_datasets
- Stage B data loading orchestration

### `src/modeling_studio/data/labels.py`

- **Imports**: Capability enum
- All 12 task label schemas
- Stage B adds 5 new schemas

### `src/modeling_studio/data/loaders.py`

- **Imports**: load_familyos_unified, load_embedding_triplets, load_familyos_unified_for_training
- Unified FamilyOS data format (420K synthetic samples)
- Handles all tasks in single JSONL files
- Embedding triplet loader (200K+ triplets)

### `src/modeling_studio/models/modernbert_multitask.py`

- **Imports**: ModernBertMultiTaskModel
- Multi-task model with 12 heads
- Head initialization for all capabilities
- Layer-wise LR assignment (Epic 5.0)

### `src/modeling_studio/trainers/multitask_trainer.py`

- **Imports**: MultiTaskTrainer, MultiTaskTrainingArguments
- Training loop with task sampling
- Layer-wise optimizer groups
- Safety oversampling (CRISIS: 20x, RED: 5x)

### `src/modeling_studio/evaluation/evaluator.py`

- **Imports**: Evaluator
- Forgetting evaluation (Stage A metrics comparison)
- Per-task evaluation

---

## Level 3: All 12 Task Heads Architecture

### `src/modeling_studio/models/heads.py`

**CRITICAL FILE**: Contains all 12 head implementations

#### Token Classification Heads (3)

1. **TokenClassificationHead** (ner_general, ner_family)
   - Architecture: `Dropout → Linear(768 → num_labels)`
   - ner_general: 17 BIO tags
   - ner_family: 21 BIO tags
   - **Issues**: No POS, no CRF, no span extraction

2. **TemporalHead** (extends TokenClassificationHead)
   - 13 BIO tags for time expressions
   - Same architecture issues

#### Sequence Classification Heads (6)

3. **SequenceClassificationHead** (sentiment, ingress, generic base)
   - Architecture: `Pooler → Dropout → Classifier → [Dense] → Output`
   - sentiment: 5 classes
   - ingress: 12 domains

4. **IntentHead** (extends SequenceClassificationHead)
   - 8 user intent classes
   - Same base architecture

5. **NLIHead** (extends SequenceClassificationHead)
   - 3-way classification (entailment, neutral, contradiction)
   - Optional CrossAttentionPairEncoder integration

6. **SafetyHead** (4-band hierarchical)
   - Architecture: `Pooler → Hierarchical bands → Subcategory`
   - 4 bands (GREEN, AMBER, RED, CRISIS)
   - 13 subcategories
   - Keyword override logic

7. **EnhancedSafetyHead** (safety_generic)
   - Multi-label toxicity (8 classes)
   - ASL (Asymmetric Loss) for imbalance
   - More complex than SafetyHead

8. **HierarchicalEmotionHead** (emotions)
   - 44 FamilyOS emotion classes
   - Multi-label classification
   - Intensity prediction
   - Valence/arousal (optional)
   - **Most complex head**

#### Relation Classification Head (1)

9. **RelationHead**
   - Architecture: `Entity pooling → Concatenate → Classifier`
   - 15 family relationship types
   - Optional CrossAttentionPairEncoder

#### Embedding Head (1)

10. **EmbeddingHead**
    - Architecture: `Pooling → [Projection] → Normalize`
    - 768-dim dense vectors
    - Pooling: cls, mean, or max
    - Triplet margin loss

---

## Level 4: Head-Specific Dependencies

### Pooling Strategies (Epic 5.0)

#### `src/modeling_studio/models/poolers.py`

- **CLSMeanPooler**: Shared pooler for sequence heads
- **get_pooler()**: Factory function
- Used by: SequenceClassificationHead, SafetyHead

### Pair Encoding (Epic 5.0)

#### `src/modeling_studio/models/pair_encoder.py`

- **CrossAttentionPairEncoder**: Cross-attention for pairs
- Used by: NLIHead, RelationHead
- Improves over simple concatenation

### Loss Functions

#### `src/modeling_studio/models/losses.py`

- **Imports by multitask_trainer.py**:
  - FGM (Fast Gradient Method adversarial)
  - PGD (Projected Gradient Descent)
  - EmbeddingMixup
  - RDropLoss
- Custom loss modules for robustness

### Data Collation

#### `src/modeling_studio/trainers/collators.py`

- **MultiTaskCollator**: Routes to task-specific collators
- **TokenClassificationCollator**: Label alignment for NER
- **SequenceClassificationCollator**: Padding for classification
- **PairCollator**: For NLI/relation pairs
- **TripletCollator**: For embedding triplets
- **CounterfactualCollator**: For decoder (Stage C)

---

## Level 5: Training Infrastructure

### Task Sampling & Weighting

#### `src/modeling_studio/trainers/task_sampler.py`

- **TaskSampler**: Samples next task based on weights
- **create_sampler()**: Factory with strategies
- Strategies: proportional, temperature, uncertainty

#### `src/modeling_studio/trainers/task_weighting.py`

- **UncertaintyWeighting**: Dynamic loss balancing
- Learns per-task uncertainty parameters

### Layer-wise Optimization

#### `src/modeling_studio/trainers/optimizer.py`

- **create_layer_wise_optimizer()**: Assigns different LRs per layer group
- Critical for v3 prep: L15-20 get 5e-5 (highest)

### Model Checkpointing

#### `src/modeling_studio/trainers/ema.py`

- **EMAModel**: Exponential moving average
- Smooths model weights for better generalization

---

## Complete 12-Head Architecture Map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ModernBERT Encoder (22 layers)                       │
│                         Output: hidden_states (B, L, 768)                   │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         │                                                       │
         ▼                                                       ▼
┌─────────────────────┐                               ┌─────────────────────┐
│  TOKEN-LEVEL HEADS  │                               │  SEQUENCE-LEVEL    │
│     (3 heads)       │                               │      HEADS         │
└─────────────────────┘                               │    (7 heads)       │
                                                      └─────────────────────┘
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. ner_general (TokenClassificationHead)                                    │
│    Dropout → Linear(768 → 17)                                               │
│    Issues: ❌ No POS, ❌ No CRF, ❌ Partial entities                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. ner_family (TokenClassificationHead)                                     │
│    Dropout → Linear(768 → 21)                                               │
│    Issues: ❌ Tags verbs as MILESTONE, ❌ Tags "the" as PET                │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. temporal (TemporalHead extends TokenClassificationHead)                  │
│    Dropout → Linear(768 → 13)                                               │
│    Issues: Same as base TokenClassificationHead                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. sentiment (SequenceClassificationHead)                                   │
│    Pooler → Dropout → Linear(768 → 5)                                       │
│    5-point scale: very_negative → very_positive                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ 5. ingress (SequenceClassificationHead)                                     │
│    Pooler → Dropout → Linear(768 → 12)                                      │
│    12 domains: DIARY, TASK, HEALTH, MEMORY, etc.                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ 6. intent (IntentHead extends SequenceClassificationHead)                   │
│    Pooler → Dropout → Linear(768 → 8)                                       │
│    8 intents: log_memory, query_memory, set_reminder, etc.                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ 7. nli (NLIHead extends SequenceClassificationHead)                         │
│    Option A: Pooler → Dropout → Linear(768 → 3)                             │
│    Option B: CrossAttentionPairEncoder → Classifier                         │
│    3 classes: entailment, neutral, contradiction                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ 8. safety_familyos (SafetyHead)                                             │
│    Architecture:                                                            │
│      Pooler → band_classifier(768 → 4)  [GREEN/AMBER/RED/CRISIS]            │
│            → subcat_classifier(768 → 13) [stress, isolation, self_harm, ...]│
│    Hierarchical: Band first, then subcategory within band                   │
│    Keyword override: Explicit crisis detection bypasses model               │
├─────────────────────────────────────────────────────────────────────────────┤
│ 9. safety_generic (EnhancedSafetyHead - more complex)                       │
│    Multi-label toxicity (8 classes)                                         │
│    Uses ASL (Asymmetric Loss) for class imbalance                           │
│    Calibration support for threshold tuning                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ 10. emotions (HierarchicalEmotionHead - MOST COMPLEX)                       │
│     Architecture:                                                           │
│       Pooler → primary_head(768 → 44)  [Multi-label BCE]                    │
│            → secondary_head(768 → 3)   [Top-3 selection]                    │
│            → intensity_head(768 → 1)   [Strength prediction]                │
│            → valence_arousal(768 → 2)  [Optional V-A space]                 │
│     44 FamilyOS emotions (8 core + 12 positive + 10 negative + 14 family)   │
│     Issues: Prone to collapse with ASL/Focal loss (now uses plain BCE)      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 11. relation (RelationHead)                                                 │
│     Architecture:                                                           │
│       Entity1 pooling + Entity2 pooling → Concatenate(1536) → Classifier(15)│
│     Option: CrossAttentionPairEncoder for better entity interaction         │
│     15 relations: parent_of, sibling_of, spouse_of, pet_of, etc.           │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 12. embedding (EmbeddingHead)                                               │
│     Architecture:                                                           │
│       Pooling (mean/cls/max) → [Projection(768 → 768)] → L2 Normalize      │
│     Output: 768-dim dense vectors                                           │
│     Loss: TripletMarginLoss with hard negatives (margin=0.3)                │
│     Use case: Semantic similarity, retrieval, clustering                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Critical Path for v3 Transfer

```
train_stage_b.py
    ↓
stage_b_for_v3_prep.yaml
    ↓ (defines layer-wise LRs)
Layer-wise optimizer creation
    ↓
L1-6:   1e-5 (preserve foundation)
L7-14:  2e-5 (light tuning)
L15-20: 5e-5 ⭐ CRITICAL - strongly train for family context
L21-22: 4e-5 (semantic band)
Heads:  1e-4 (train heads but they'll be discarded)
    ↓
Training loop with FamilyOS data
    ↓
Encoder layers learn family-specific patterns
    ↓
Output: modernbert-v2-for-v3-transfer
    ↓
┌─────────────────────────────────────────────────────────────────┐
│  v3 Weight Transfer (happens in initialize_v3_from_v2.py)       │
│                                                                 │
│  v3 L1-22   ← Direct copy from v2 L1-22                         │
│  v3 L23-28  ← CLONE from v2 L15-20 (these trained layers!) ⭐  │
│  v3 heads   ← New random init (v2 heads discarded)              │
│  Hub tokens ← Semantic centroid init                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Head Complexity Ranking (Simple → Complex)

### Tier 1: Simplest (Single Linear Layer)

1. **TokenClassificationHead** (ner_general, ner_family, temporal)
   - `Dropout → Linear`
   - **Most problematic** due to simplicity

### Tier 2: Basic Classification

2. **SequenceClassificationHead** (sentiment, ingress)
   - `Pooler → Dropout → Linear`
3. **IntentHead** (extends SequenceClassificationHead)
   - Same as base

### Tier 3: Structured Prediction

4. **NLIHead** (with optional CrossAttentionPairEncoder)
   - Can use pair encoder for better representations
5. **RelationHead** (entity concatenation + optional pair encoder)
   - Entity-aware pooling

### Tier 4: Hierarchical/Multi-output

6. **SafetyHead** (4-band hierarchical + 13 subcategories)
   - Two-stage classification
   - Keyword override logic
7. **EnhancedSafetyHead** (multi-label with ASL)
   - Complex loss function

### Tier 5: Most Complex

8. **HierarchicalEmotionHead** (44-class multi-label + intensity + V-A)
   - Multiple output heads
   - Prone to training instability
9. **EmbeddingHead** (metric learning)
   - Triplet loss, hard negative mining

---

## Known Issues by Head

### TokenClassificationHead (ner_general, ner_family)

**Garbage Rate**: 66%+ before filtering

**Issues**:

- No POS tagging → Tags verbs as entities
- No character features → Misses capitalization patterns
- No CRF → Invalid BIO transitions
- No span extraction → Partial entities ("Lincoln" not "Lincoln School")

**Mitigation**: 10-stage post-processing filter pipeline

### HierarchicalEmotionHead (emotions)

**Issues**:

- Training collapse with ASL/Focal loss
- Current fix: Plain BCE loss (expert recommended)
- Multi-label thresholding needed

### SafetyHead (safety_familyos)

**Issues**:

- Indian English false positives ("dying of laughter" → CRISIS)
- Cultural awareness needed
- Current fix: Keyword override + calibration

### RelationHead (relation)

**Issues**:

- Simple concatenation may miss entity interactions
- CrossAttentionPairEncoder improves but adds complexity

---

## Data Flow (Full 12-Task Training)

```
1. YAML Config Loading
   ├─ stage_b_for_v3_prep.yaml: Training hyperparameters
   ├─ stage_b_datasets.yaml: Data sources (gold + silver)
   └─ modernbert_base.yaml: Model architecture

2. Data Loading (Unified Format)
   ├─ load_familyos_unified() reads 420K synthetic samples
   ├─ Extracts 11 tasks from unified JSONL
   ├─ load_embedding_triplets() loads 200K+ triplets separately
   └─ Stage A replay datasets (15%): CoNLL, SST-2, MNLI

3. Tokenization & Collation
   ├─ MultiTaskCollator routes to task-specific collators
   ├─ TokenClassificationCollator: Aligns NER labels with subwords
   ├─ SequenceClassificationCollator: Pads classification tasks
   ├─ PairCollator: Handles NLI/relation entity pairs
   └─ TripletCollator: Prepares anchor/pos/neg for embeddings

4. Model Initialization
   ├─ Load Stage A checkpoint: modernbert-multitask-v0-stage-a-fast
   ├─ Initialize 12 heads (or load existing)
   ├─ Create layer-wise optimizer groups:
   │   L1-6: 1e-5, L7-14: 2e-5, L15-20: 5e-5, L21-22: 4e-5, Heads: 1e-4
   └─ Setup EMA model (decay=0.999)

5. Training Loop (MultiTaskTrainer)
   ├─ Task sampling with temperature (temp=4 for speed)
   ├─ Safety oversampling: CRISIS 20x, RED 5x
   ├─ Forward pass through encoder + selected head
   ├─ Compute task-specific loss (weighted)
   ├─ Backward pass (gradient accumulation)
   └─ Update layer-wise optimizer groups

6. Per-Head Forward Pass Examples

   NER (token-level):
   encoder(input_ids) → hidden_states (B, L, 768)
                      → TokenClassificationHead
                      → logits (B, L, num_labels)
                      → CrossEntropyLoss with ignore_index=-100

   Sentiment (sequence-level):
   encoder(input_ids) → hidden_states (B, L, 768)
                      → SequenceClassificationHead.pooler(hidden_states)
                      → pooled (B, 768)
                      → Dropout → Linear → logits (B, 5)
                      → CrossEntropyLoss

   Emotions (multi-label):
   encoder(input_ids) → hidden_states (B, L, 768)
                      → HierarchicalEmotionHead.pooler
                      → primary_head → logits (B, 44)
                      → BCEWithLogitsLoss (multi-label)

   Relation (pairs):
   encoder(input_ids) → hidden_states (B, L, 768)
                      → RelationHead.pool_entities(e1, e2)
                      → concat([e1_pooled, e2_pooled]) (B, 1536)
                      → Linear(1536 → 15) → logits (B, 15)
                      → CrossEntropyLoss

   Embedding (triplets):
   encoder(anchor) → hidden_anchor (B, L, 768)
   encoder(pos)    → hidden_pos (B, L, 768)
   encoder(neg)    → hidden_neg (B, L, 768)
                   → EmbeddingHead.pool + normalize
                   → embeddings (B, 768) each
                   → TripletMarginLoss(anchor, pos, neg, margin=0.3)

7. Evaluation & Checkpointing
   ├─ Per-task metrics (F1, accuracy, etc.)
   ├─ Forgetting evaluation: Compare Stage A baseline metrics
   ├─ Save checkpoints every N steps
   └─ Save final model: modernbert-v2-for-v3-transfer
```

---

## Files to Modify for Head Improvements

### Primary Targets (All Heads)

**`src/modeling_studio/models/heads.py`** - All 12 head implementations

#### Token Classification Heads (ner_general, ner_family, temporal)

- Replace TokenClassificationHead with:
  - **ContextAwareNERHead**: Add POS features, char features
  - **NERHeadWithCRF**: Add CRF layer for valid transitions
  - **SpanNERHead**: Span-based extraction (no more partial entities)

#### Emotion Head (emotions)

- **HierarchicalEmotionHead** improvements:
  - Better threshold calibration
  - Curriculum learning (easier emotions first)
  - Label correlation loss (optional, currently disabled)

#### Safety Heads (safety_familyos, safety_generic)

- **SafetyHead** improvements:
  - Better cultural awareness features
  - Expanded keyword override dictionary
  - Confidence calibration

#### Relation Head (relation)

- **RelationHead** improvements:
  - Better entity representation (span pooling)
  - CrossAttentionPairEncoder as default
  - Multi-hop reasoning for complex relations

### Secondary Targets

**`src/modeling_studio/models/modernbert_multitask.py`**

- Update CAPABILITY_TO_HEAD_TYPE mapping
- Add new head type configurations

**`configs/training/multitask/stage_b_for_v3_prep.yaml`**

- Add head-specific hyperparameters
- Tune per-head dropout rates
- Add per-head learning rates

### Supporting Changes

**`src/modeling_studio/trainers/collators.py`**

- Add span-based NER collator
- Improve relation pair collation

**`src/modeling_studio/evaluation/metrics.py`**

- Add span-level F1 for NER
- Add per-band metrics for safety
- Add per-emotion-category metrics

---

## Stage B Capabilities Overview

### Stage A Replay (7 tasks - prevent forgetting)

1. **ner_general** (TokenClassificationHead) - 17 BIO tags
2. **sentiment** (SequenceClassificationHead) - 5 classes
3. **emotions** (HierarchicalEmotionHead) - 44 classes
4. **safety_generic** (EnhancedSafetyHead) - 8 toxicity types
5. **nli** (NLIHead) - 3 classes
6. **embedding** (EmbeddingHead) - 768-dim vectors
7. **temporal** (TemporalHead) - 13 BIO tags

### Stage B New (5 tasks - FamilyOS domain)

8. **ner_family** (TokenClassificationHead) - 21 BIO tags
9. **ingress** (SequenceClassificationHead) - 12 domains
10. **safety_familyos** (SafetyHead) - 4 bands + 13 subcategories
11. **intent** (IntentHead) - 8 user intents
12. **relation** (RelationHead) - 15 family relationship types

---

## Training Statistics

### Data Volumes

| Task | Gold Samples | Silver Samples | Total |
|------|-------------|----------------|-------|
| ner_family | 250 | 12,703 | 12,953 |
| ingress | 360 | 22,747 | 23,107 |
| safety_familyos | 200 | 14,333 | 14,533 |
| intent | 200 | 15,972 | 16,172 |
| relation | 40 | 21,340 | 21,380 |
| embedding | - | 200,000+ | 200,000+ |
| **FamilyOS Total** | - | - | **~288K** |
| **Stage A Replay** | - | - | **~25K** |
| **Grand Total** | - | - | **~313K** |

### Training Configuration

- **Epochs**: 8
- **Batch size**: 128 per device (A100 optimized)
- **Effective batch**: 512 (with grad accumulation)
- **Max sequence length**: 96 tokens (FamilyOS data is short)
- **Learning rates**:
  - L1-6: 1e-5 (foundation)
  - L7-14: 2e-5 (context)
  - L15-20: 5e-5 ⭐ (target for v3 transfer)
  - L21-22: 4e-5 (semantic)
  - Heads: 1e-4

---

## Next Steps for All-Head Improvements

### Phase 1: Token Classification (Highest Priority)

1. Implement **ContextAwareNERHead** with POS features
2. Implement **NERHeadWithCRF** for valid transitions
3. Test on CoNLL-2003 (Stage A) and FamilyOS NER (Stage B)
4. Measure garbage entity reduction

### Phase 2: Hierarchical Heads

5. Improve **HierarchicalEmotionHead** threshold calibration
6. Enhance **SafetyHead** cultural awareness
7. Add confidence calibration for both

### Phase 3: Relation & Pair Heads

8. Make **CrossAttentionPairEncoder** default for RelationHead
9. Improve **NLIHead** with better pair encoding
10. Add span-level entity pooling

### Phase 4: Embedding & Loss Functions

11. Experiment with **hard negative mining** for EmbeddingHead
12. Add **focal loss** alternatives for safety heads
13. Implement **label smoothing** for sequence heads

---

## Key Metrics to Track

### Before Enhancement (Current Baselines)

- **NER garbage rate**: 66%+ (requires 10-stage filtering)
- **Emotion collapse**: Fixed with plain BCE (was ASL/Focal)
- **Safety cultural FP**: ~2% (Indian English patterns)
- **Relation accuracy**: ~75% (simple concatenation)

### After Enhancement (Targets)

- **NER garbage rate**: <10% (reduce filtering to 2-3 stages)
- **Span-level NER F1**: +5-10% improvement
- **Emotion calibration**: Better thresholds, no collapse
- **Safety cultural FP**: <1% with expanded awareness
- **Relation accuracy**: +10% with CrossAttention

---

## References

- **Stage A Trace**: [TRAINING_DEPENDENCY_TRACE.md](./TRAINING_DEPENDENCY_TRACE.md)
- **NER Quality Issues**: See separate NER quality catalog
- **ADR K023**: Entity Filtering Consolidation
- **v3 Architecture**: [RELEASE_ARCHITECTURE_v3.md](./RELEASE_ARCHITECTURE_v3.md)
- **Training Logs**: `outputs/modernbert-v2-for-v3-transfer/`
- **Checkpoints**: `checkpoints/modernbert-v2-for-v3-transfer/`
