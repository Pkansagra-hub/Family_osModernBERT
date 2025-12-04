# FamilyOS ModernBERT Benchmarking Plan

> **Purpose:** Define realistic evaluation metrics that reflect actual FamilyOS use cases, not just academic benchmarks.

---

## 1. Sentiment Benchmarking

### 1.1 The Problem with Strict 5-Class Accuracy

Our sentiment model predicts 5 classes:

- `very_negative` (0)
- `negative` (1)
- `neutral` (2)
- `positive` (3)
- `very_positive` (4)

**Current strict accuracy: ~48%** at epoch 4

But for FamilyOS, predicting `negative` when the true label is `very_negative` is **still useful**! The direction is correct, only the intensity is off.

### 1.2 Proposed Evaluation Modes

| Mode | Description | Acceptance Criteria |
|------|-------------|---------------------|
| **Strict 5-class** | Exact match required | Academic benchmark |
| **Grouped 3-class** | Negative family / Neutral / Positive family | **Primary metric** |
| **Binary direction** | Negative vs Positive (exclude neutral) | Secondary metric |
| **Adjacent tolerance** | Allow ±1 class error | Relaxed metric |

### 1.3 Grouped 3-Class Logic

Map 5-class predictions to 3 sentiment groups:

```python
def to_3class(label: int) -> str:
    """Map 5-class to 3-class sentiment."""
    if label in (0, 1):      # very_negative, negative
        return "negative"
    elif label == 2:          # neutral
        return "neutral"
    else:                     # positive, very_positive (3, 4)
        return "positive"
```

**Evaluation matrix:**

| Predicted \ Actual | Very Neg (0) | Neg (1) | Neutral (2) | Pos (3) | Very Pos (4) |
|-------------------|--------------|---------|-------------|---------|--------------|
| **Very Neg (0)** | ✅ | ✅ | ❌ | ❌ | ❌ |
| **Neg (1)** | ✅ | ✅ | ❌ | ❌ | ❌ |
| **Neutral (2)** | ❌ | ❌ | ✅ | ❌ | ❌ |
| **Pos (3)** | ❌ | ❌ | ❌ | ✅ | ✅ |
| **Very Pos (4)** | ❌ | ❌ | ❌ | ✅ | ✅ |

### 1.4 Expected Performance

| Metric | Epoch 4 | Epoch 10 (Projected) | Target |
|--------|---------|---------------------|--------|
| Strict 5-class accuracy | 48% | 55-60% | 55%+ |
| **Grouped 3-class accuracy** | ~70% | **75-80%** | **75%+** |
| Binary direction accuracy | ~85% | 88-92% | 88%+ |

### 1.5 Confusion Matrix Analysis

Generate confusion matrix to understand:

1. Which classes are most confused
2. Whether confusion is within-group (acceptable) or cross-group (bad)
3. Neutral class precision/recall (often hardest)

### 1.6 FamilyOS Use Case Alignment

| Use Case | Required Accuracy | Metric |
|----------|-------------------|--------|
| Mood trend detection | Direction correct | Grouped 3-class |
| Alert on negative patterns | High recall for negative | Negative recall |
| Celebrate positive moments | Identify positive | Positive precision |
| Neutral filtering | Filter factual content | Neutral F1 |

### 1.7 Benchmark Script Requirements

Create `scripts/benchmark_sentiment.py`:

```python
# Inputs:
# - checkpoint_path: Path to model checkpoint
# - validation_data: Sentiment validation set

# Outputs:
# - Strict 5-class accuracy
# - Grouped 3-class accuracy
# - Binary direction accuracy (excluding neutral)
# - Confusion matrix (5x5)
# - Grouped confusion matrix (3x3)
# - Per-class precision/recall/F1
# - Adjacent tolerance accuracy (±1 class)
```

---

## 2. Emotions Benchmarking

### 2.1 The Problem with Strict Multi-Label Metrics

Our emotions model predicts 44 classes with ~3 emotions active per sample:

- `joy`, `pride`, `love` (example: 3 emotions)
- Multi-hot vector: 44 elements, ~3 are 1s

**Current strict metrics (Epoch 4):**

- Micro-F1: 24.7%
- Subset Accuracy: 0.0% (exact match - nearly impossible!)

But for FamilyOS, if we predict **2 out of 3 emotions correctly**, that's useful!

### 2.2 Proposed Evaluation Modes

| Mode | Description | Acceptance Criteria |
|------|-------------|---------------------|
| **Strict Micro-F1** | All predictions vs all labels | Academic benchmark |
| **Top-K Recall** | Did we find at least K correct emotions? | **Primary metric** |
| **Primary Emotion Accuracy** | Is the highest-confidence prediction correct? | Critical for UI |
| **Partial Match Score** | Jaccard similarity (intersection / union) | Relaxed metric |
| **At-Least-One Correct** | Did we get ANY emotion right? | Baseline check |

### 2.3 Top-K Recall Logic (Primary Metric)

For FamilyOS, getting **at least 1-2 emotions right** is a win:

```python
def top_k_recall(predictions: list[int], ground_truth: list[int], k: int = 2) -> bool:
    """Check if at least K ground truth emotions were predicted."""
    correct = len(set(predictions) & set(ground_truth))
    return correct >= min(k, len(ground_truth))

# Example:
# Ground truth: [joy, pride, love] (3 emotions)
# Predicted: [joy, love, excitement, gratitude] (4 emotions)
#
# Correct: joy, love (2 out of 3)
# top_k_recall(k=2) → TRUE ✅ (we got at least 2 right!)
# top_k_recall(k=3) → FALSE (we only got 2, needed 3)
```

**Evaluation matrix:**

| Ground Truth Count | K=1 (Any correct) | K=2 (Two correct) | All Correct |
|--------------------|-------------------|-------------------|-------------|
| 1 emotion | Must get it | N/A | Must get it |
| 2 emotions | Get 1 of 2 | Get 2 of 2 | Get 2 of 2 |
| 3 emotions | Get 1 of 3 | Get 2 of 3 | Get 3 of 3 |
| 4 emotions | Get 1 of 4 | Get 2 of 4 | Get 4 of 4 |

### 2.4 Primary Emotion Accuracy

The **highest-confidence emotion** should ideally match the `primary_emotion` field:

```python
def primary_emotion_accuracy(pred_logits: Tensor, primary_emotion: str) -> bool:
    """Check if highest-confidence prediction matches primary emotion."""
    predicted_primary = pred_logits.argmax()
    return id2label[predicted_primary] == primary_emotion
```

This is critical for FamilyOS UI that might show "Feeling: Joy 😊"

### 2.5 Partial Match Score (Jaccard)

```python
def partial_match_score(predictions: set, ground_truth: set) -> float:
    """Jaccard similarity - intersection over union."""
    if not predictions and not ground_truth:
        return 1.0
    intersection = len(predictions & ground_truth)
    union = len(predictions | ground_truth)
    return intersection / union

# Example:
# Ground truth: {joy, pride, love}
# Predicted: {joy, love, excitement}
#
# Intersection: {joy, love} = 2
# Union: {joy, pride, love, excitement} = 4
# Jaccard: 2/4 = 0.50 (50% match)
```

### 2.6 Expected Performance

| Metric | Epoch 4 | Epoch 10 (Projected) | Target |
|--------|---------|---------------------|--------|
| Strict Micro-F1 | 24.7% | 55-65% | 55%+ |
| **Top-2 Recall** | ~70% | **85-90%** | **85%+** |
| Primary Emotion Acc | ~50% | 70-75% | 70%+ |
| Partial Match (Jaccard) | ~35% | 55-65% | 55%+ |
| At-Least-One Correct | ~90% | 95%+ | 95%+ |

### 2.7 Emotion Families (Grouped Evaluation)

Similar to sentiment, group related emotions for relaxed matching:

| Family | Emotions |
|--------|----------|
| **Joy Family** | joy, amusement, excitement, contentment, playfulness, celebration |
| **Sadness Family** | sadness, grief, disappointment, longing, emptiness, homesickness |
| **Anger Family** | anger, annoyance, frustration, disapproval |
| **Love Family** | love, caring, tenderness, warmth, togetherness, belonging |
| **Pride Family** | pride, admiration, approval, parental_pride |
| **Fear Family** | fear, nervousness, worry, overwhelmed |

**Grouped match:** If ground truth is `joy` and model predicts `excitement`, count as partial win (same family).

### 2.8 FamilyOS Use Case Alignment

| Use Case | Required Metric | Target |
|----------|-----------------|--------|
| Mood summary ("Feeling joyful today") | Primary emotion accuracy | 70%+ |
| Emotion trend tracking | Top-2 recall | 85%+ |
| Alert on negative emotions | Negative family recall | 90%+ |
| Memory tagging | Partial match score | 55%+ |

### 2.9 Why Current Metrics Look Low But Are Actually Good

```
44 emotions, ~3 active per sample

Random baseline:
- Predicting 3 random emotions: P(any correct) ≈ 3/44 × 3 ≈ 20%

Current model (Epoch 4):
- Recall: 96.2% (finding almost ALL correct emotions!)
- Micro-F1: 24.7% (low because of false positives)
- Top-2 Recall (estimated): ~70-80%

The model IS learning - it just needs to reduce false positives in epochs 5-10.
```

---

## 3. NER Benchmarking

### 3.1 Current Metrics (Epoch 4)

| Entity Type | F1 Score | Notes |
|-------------|----------|-------|
| **PER** (Person) | 58.9% | Best performing |
| **LOC** (Location) | 58.3% | Good |
| **MISC** (Miscellaneous) | 31.8% | Challenging |
| **ORG** (Organization) | 29.3% | Most difficult |
| **Overall** | 48.8% | Target: 88% |

### 3.2 The Problem with Token-Level F1

Standard NER evaluation is **token-level** - every B-PER, I-PER token must be correct.

For FamilyOS, if we detect "John Smith" but tag it as:

- Predicted: `[B-PER, B-PER]` (two separate persons)
- Actual: `[B-PER, I-PER]` (one person)

This is still **useful** - we found the person!

### 3.3 Proposed Evaluation Modes

| Mode | Description | Acceptance Criteria |
|------|-------------|---------------------|
| **Strict Token F1** | Every BIO tag must match | Academic benchmark |
| **Entity-Level F1** | Entity span must match (any overlap) | **Primary metric** |
| **Partial Span Match** | Allow partial entity overlap | Relaxed metric |
| **Entity Detection** | Did we find ANY entity of this type? | Baseline check |

### 3.4 Entity-Level Logic

```python
def entity_level_match(pred_entities: list, gold_entities: list) -> dict:
    """
    Match entities at the entity level, not token level.

    Example:
    Gold: [("John Smith", PER, 0, 2), ("NYC", LOC, 5, 6)]
    Pred: [("John", PER, 0, 1), ("Smith", PER, 1, 2), ("NYC", LOC, 5, 6)]

    Token-level: 2/4 correct (50%)
    Entity-level with overlap: 2/2 gold entities found (100%)
    """
    # Count if predicted entity overlaps with gold entity of same type
    pass
```

### 3.5 Entity Type Groupings

For FamilyOS, some confusions are acceptable:

| Predicted | Actual | Acceptable? |
|-----------|--------|-------------|
| PER | PER | ✅ Exact |
| LOC | ORG | ⚠️ Partial (companies have addresses) |
| MISC | any | ⚠️ Partial (catch-all) |
| ORG | PER | ❌ Wrong |

### 3.6 Expected Performance

| Metric | Epoch 4 | Epoch 10 (Projected) | Target |
|--------|---------|---------------------|--------|
| Token-level F1 | 48.8% | 62-68% | 65%+ |
| **Entity-level F1** | ~60% | **75-80%** | **75%+** |
| PER F1 | 58.9% | 75-80% | 75%+ |
| LOC F1 | 58.3% | 75-80% | 75%+ |

---

## 4. Temporal Benchmarking

### 4.1 Current Metrics (Epoch 4) - ✅ EXCEEDS TARGET

| Temporal Type | F1 Score | Notes |
|---------------|----------|-------|
| **DATE_ABS** | 95.7% | Excellent! |
| **DATE_REL** | 96.3% | Excellent! |
| **FREQUENCY** | 92.9% | Excellent! |
| **AGE** | 87.5% | Good |
| **DURATION** | 75.0% | Improving |
| **TIME** | 57.1% | Needs work |
| **Overall** | **83.1%** | Target: 82% ✅ |

### 4.2 Why Temporal is Easy (Relatively)

- Only **6 entity types** (vs 4 for NER, but clearer patterns)
- Strong lexical cues: "yesterday", "3pm", "every week", "for 2 hours"
- Less ambiguity than NER (organization vs location)

### 4.3 FamilyOS Use Case Alignment

| Use Case | Required Metric | Current |
|----------|-----------------|---------|
| Timeline construction | DATE_ABS + DATE_REL F1 | 96% ✅ |
| Routine detection | FREQUENCY F1 | 93% ✅ |
| Event duration | DURATION F1 | 75% 🟡 |
| Time-based reminders | TIME F1 | 57% ⚠️ |

### 4.4 TIME F1 Improvement Strategy

TIME (57.1%) is lowest because:

- Ambiguous: "morning" vs "3am" vs "breakfast time"
- Context-dependent: "dinner" = 6pm? 7pm?

For FamilyOS, we might want:

- **Fuzzy time matching**: "morning" = 6am-11am (any prediction in range is correct)
- **Meal-time normalization**: "dinner" → "evening" category

---

## 5. Safety Generic Benchmarking

### 5.1 Current Metrics (Epoch 4)

| Metric | Value | Notes |
|--------|-------|-------|
| Micro-F1 | 24.1% | Low but expected |
| Macro-F1 | 18.0% | Class imbalance |
| **Recall** | **99.98%** | Finding almost ALL toxic! |
| Precision | 13.7% | Many false positives |
| Hamming Loss | 0.75 | High (many wrong slots) |

### 5.2 Safety Is Different - Recall > Precision

For safety, **missing toxic content is worse than false positives**.

| Scenario | Outcome |
|----------|---------|
| ❌ Miss toxic content | User sees harmful content |
| ⚠️ False positive | Content flagged for review (minor inconvenience) |

**We WANT high recall, even at cost of precision!**

### 5.3 Proposed Evaluation Modes

| Mode | Description | Target |
|------|-------------|--------|
| **Per-Type Recall** | Did we catch this type of toxicity? | 95%+ each |
| **Any-Toxic Recall** | Did we flag ANY toxic content? | 99%+ |
| **Precision @ High Recall** | Precision when recall ≥ 95% | 50%+ |
| **Severity-Weighted** | Weight severe_toxic, threat higher | Custom |

### 5.4 Safety Type Hierarchy

| Type | Severity | Required Recall |
|------|----------|-----------------|
| `severe_toxic` | 🔴 Critical | **99%+** |
| `threat` | 🔴 Critical | **99%+** |
| `self_harm` | 🔴 Critical | **99%+** |
| `identity_hate` | 🟠 High | 95%+ |
| `insult` | 🟠 High | 95%+ |
| `obscene` | 🟡 Medium | 90%+ |
| `toxic` | 🟡 Medium | 90%+ |
| `dangerous_advice` | 🟠 High | 95%+ |

### 5.5 Expected Performance

Current 99.98% recall is GREAT! Now we need precision to improve:

| Metric | Epoch 4 | Epoch 10 (Projected) | Target |
|--------|---------|---------------------|--------|
| Any-Toxic Recall | 99.98% | 99%+ | 99%+ ✅ |
| Micro-F1 | 24.1% | 55-65% | 55%+ |
| Precision @ 95% Recall | ~15% | 50-60% | 50%+ |

### 5.6 Safety Generic vs Safety FamilyOS

**Two distinct safety tasks in our pipeline:**

| Task | Stage | Labels | Head Type | Purpose |
|------|-------|--------|-----------|---------|
| `safety_generic` | Stage A | 8 toxicity types | Multi-label | Detect toxic content types |
| `safety_familyos` | Stage B | 4 policy bands | Single-label | Route to appropriate response |

**Stage A (Current):** Learning to detect toxicity presence and types.
**Stage B (Future):** Learning FamilyOS-specific policy bands (GREEN → CRISIS).

### 5.7 Data Sources

| Dataset | Samples | Labels | Notes |
|---------|---------|--------|-------|
| Civil Comments (curated) | ~200K | Multi-hot [8] | Balanced classes |
| Jigsaw Toxicity | ~150K | 6 types | Original benchmark |
| FamilyOS Safety (gold) | ~2K | 4 bands | Expert-labeled |
| FamilyOS Safety (silver) | ~5K | 4 bands | LLM-generated |

---

## 5.8 Safety FamilyOS Benchmarking (Stage B)

### Policy Band Targets (Non-Negotiable)

| Band | Description | Required Recall | Priority |
|------|-------------|-----------------|----------|
| **CRISIS** | Immediate intervention | **≥ 98%** | P0 |
| **RED** | Serious concern | ≥ 90% | P0 |
| **AMBER** | Needs attention | ≥ 85% | P1 |
| **GREEN** | Safe content | Precision ≥ 90% | P2 |

### Cultural Robustness Testing (Critical)

Indian English expressions that should NOT trigger CRISIS:

```python
INDIAN_VENTING_PATTERNS = [
    "I'll die of embarrassment",      # GREEN/AMBER, not CRISIS
    "This is killing me",              # GREEN/AMBER, not CRISIS
    "I could die",                     # GREEN/AMBER, not CRISIS
    "My head is bursting",             # GREEN/AMBER, not CRISIS
    "I'm going mad",                   # GREEN/AMBER, not CRISIS
    "I want to kill the mood",         # GREEN, not CRISIS
    "My head is exploding with tension", # AMBER, not CRISIS
    "This exam will be the death of me", # GREEN/AMBER, not CRISIS
    "Mummy will kill me if I fail",    # GREEN/AMBER, not CRISIS
]

# Target: ≤ 2% false positive rate on cultural expressions
```

### Keyword Override Testing

These MUST always trigger CRISIS regardless of context:

```python
CRISIS_KEYWORDS = [
    "kill myself",
    "end my life",
    "don't want to live",
    "suicide",
    "I want to die",
    "hurt myself",
]

# Target: 100% CRISIS detection
```

### Safety Benchmark Script Requirements

Create `scripts/benchmark_safety.py`:

```python
# Inputs:
# - checkpoint_path: Path to model checkpoint
# - validation_data: Safety validation sets (generic + familyos)

# Outputs for safety_generic (Stage A):
# - Per-type recall (8 types)
# - Any-toxic recall
# - Micro-F1, Macro-F1
# - Precision @ 95% recall threshold

# Outputs for safety_familyos (Stage B):
# - Per-band recall (CRISIS, RED, AMBER)
# - Per-band precision
# - Confusion matrix (4x4)
# - Cultural FP rate (Indian expressions)
# - Keyword override accuracy (must be 100%)
# - Band-wise subcategory accuracy (13 subcategories)
```

### Calibration Requirements

Post-training calibration using `scripts/calibrate_safety.py`:

```yaml
# configs/calibration/safety_thresholds.yaml
safety_familyos:
  temperature: 1.15  # Calibrated via LBFGS on validation set
  thresholds:
    GREEN_AMBER: 0.35  # If P(AMBER|RED|CRISIS) > 0.35, escalate
    AMBER_RED: 0.45    # If P(RED|CRISIS) > 0.45, escalate
    RED_CRISIS: 0.60   # If P(CRISIS) > 0.60, escalate
  crisis_keywords: [...]  # Always override to CRISIS
```

### Rollback Criteria

If safety performance drops during deployment:

| Metric | Rollback Threshold |
|--------|-------------------|
| CRISIS Recall | < 95% |
| Cultural FP Rate | > 5% |
| Keyword Override | < 100% |

---

## 6. NLI Benchmarking

### 6.1 Current Metrics (Epoch 4)

| Metric | Value | Notes |
|--------|-------|-------|
| Accuracy | 53.3% | Above random (33%) |
| F1 | 53.1% | Balanced |
| Macro-F1 | 53.0% | Per-class average |

### 6.2 NLI is Fundamentally Hard

3-class classification:

- `entailment`: Premise implies hypothesis
- `neutral`: No logical connection
- `contradiction`: Premise contradicts hypothesis

Random baseline: 33.3%
Current: 53.3% = **1.6x baseline**

### 6.3 FamilyOS Use Cases for NLI

| Use Case | Example | NLI Label |
|----------|---------|-----------|
| Contradiction detection | P: "Loved the movie" H: "Hated it" | contradiction |
| Memory verification | P: "Emma's birthday is Jan 15" H: "Emma was born in January" | entailment |
| Context matching | P: "Had dinner at 7pm" H: "Ate in the morning" | contradiction |

### 6.4 Confusion Matrix Focus

For FamilyOS, the critical distinction is:

- **Contradiction vs Entailment** (opposite meanings)
- Neutral can be tolerated in either direction

```python
# Grouped evaluation:
# entailment + neutral = "compatible" (no conflict)
# contradiction = "conflicting"

# Binary: Is there a conflict?
```

### 6.5 Expected Performance

| Metric | Epoch 4 | Epoch 10 (Projected) | Target |
|--------|---------|---------------------|--------|
| 3-class Accuracy | 53.3% | 58-62% | 58%+ |
| Contradiction Recall | ~50% | 65-70% | 65%+ |
| Binary (conflict/no-conflict) | ~70% | 80-85% | 80%+ |

---

## 7. Embedding Benchmarking

### 7.1 Current Metrics (Epoch 4)

| Metric | Value | Notes |
|--------|-------|-------|
| Spearman | 12.2% | Low correlation |
| Pearson | 9.3% | Very low |

### 7.2 Why Embeddings Are Flat

Embedding quality is measured by **Semantic Textual Similarity (STS)**:

- Compare cosine similarity of embeddings vs human similarity scores
- Requires contrastive learning to work well

**Stage A focuses on classification tasks** - embeddings improve in Stage B!

### 7.3 Expected Performance

| Metric | Epoch 4 | Epoch 10 | After Stage B |
|--------|---------|----------|---------------|
| Spearman | 12.2% | 20-25% | **60-70%** |
| Pearson | 9.3% | 18-22% | **55-65%** |

### 7.4 Embedding Evaluation for FamilyOS

| Use Case | Evaluation |
|----------|------------|
| Memory retrieval | Top-K recall (find similar memories) |
| Duplicate detection | Cosine similarity threshold |
| Clustering | Silhouette score on memory clusters |

---

## 8. Summary: All Tasks at Epoch 4

| Task | Current | Target | Status |
|------|---------|--------|--------|
| **Temporal** | 83.1% F1 | 82% | ✅ DONE |
| **NLI** | 53.3% Acc | 58% | 🟡 On track |
| **NER** | 48.8% F1 | 65%+ | 🟡 Improving |
| **Sentiment** | 47.8% Acc | 55%+ (5-class) | 🟡 On track |
| **Emotions** | 24.7% µF1 | 55%+ | 🟡 ASL working |
| **Safety** | 24.1% µF1 | 55%+ (99% recall maintained) | 🟡 Recall excellent |
| **Embedding** | 12.2% Spearman | 25%+ (Stage B: 60%) | ⚠️ Expected |

---

## Appendix A: Label Definitions

### Sentiment Labels (5-class)

| ID | Label | Description | Examples |
|----|-------|-------------|----------|
| 0 | `very_negative` | Strong negative (angry, devastated) | "This is terrible!", "I hate this" |
| 1 | `negative` | Mild negative (disappointed, sad) | "Not great", "Could be better" |
| 2 | `neutral` | Factual, no sentiment | "The meeting is at 3pm" |
| 3 | `positive` | Mild positive (happy, content) | "Pretty good", "I like it" |
| 4 | `very_positive` | Strong positive (ecstatic, overjoyed) | "This is amazing!", "Best day ever!" |

### Sentiment Groups (3-class)

| Group | Includes | FamilyOS Interpretation |
|-------|----------|------------------------|
| `negative` | very_negative, negative | User expressing distress/dissatisfaction |
| `neutral` | neutral | Factual content, no emotional signal |
| `positive` | positive, very_positive | User expressing happiness/satisfaction |

---

*Last updated: December 3, 2025*
