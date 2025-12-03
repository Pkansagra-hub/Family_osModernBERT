# Data Curation Master Plan

> **Objective:** Curate all datasets to achieve Stage A & Stage B training targets
>
> **Reference Docs:**
>
> - `unified_encoder_solution.md` - Architecture & targets
> - `enhanced_design_v2.md` - Full system design
>
> **Total Raw Data:** 5.2M samples across 19 datasets
> **Target Curated:** ~2M high-quality, balanced samples

---

## Executive Summary

### Current Data State

| Source | Raw Samples | Status |
|--------|-------------|--------|
| **Public (data/public/)** | 4,750,104 | ✅ Downloaded |
| **FamilyOS (data/familyos/)** | 486,682 | ✅ Local |
| **Total** | 5,236,786 | Needs curation |

### Key Curation Challenges

1. **Class Imbalance** - Civil Comments severe_toxic at 0.03%, CRISIS samples rare
2. **Format Inconsistency** - Different column names, encoding schemes
3. **Quality Variance** - Silver data (LLM-generated) vs Gold data (human-curated)

### Training Stage Clarification

| Task | Stage | Dataset Source |
|------|-------|----------------|
| `safety_generic` (8 toxicity labels) | **Stage A** | civil_comments_curated |
| `safety_familyos` (4 bands: GREEN/AMBER/RED/CRISIS) | **Stage B** | familyos_safety silver/gold |
| `emotions` (44 FamilyOS emotions) | **Stage A** | familyos_emotions silver/gold |
| All other generic tasks | **Stage A** | Public datasets |
| All FamilyOS domain tasks | **Stage B** | familyos/* local data |

**CRITICAL:**

- CRISIS recall ≥98% is a **Stage B** target (safety_familyos head)
- Cultural FP rate ≤2% is a **Stage B** target (safety_familyos head)
- Stage A trains generic toxicity detection (safety_generic)
- Stage B trains family-specific policy bands with cultural awareness

### Target Metrics (from unified_encoder_solution.md)

| Task | Target | Priority |
|------|--------|----------|
| NER (general) F1 | 88%+ | P1 |
| NER (family) F1 | 85%+ | P1 |
| Sentiment Acc | 92%+ | P1 |
| Emotions macro F1 | 75%+ | P1 |
| Safety CRISIS Recall | **98%+** | **P0** |
| Cultural FP Rate | ≤2% | P0 |
| Ingress Acc | 90%+ | P1 |
| Temporal F1 | 82%+ | P1 |
| Relation F1 | 80%+ | P1 |
| Intent Acc | 88%+ | P1 |

---

## Milestone 1: Stage A Data Curation (Public Datasets)

**Goal:** Prepare all public datasets for generic multi-task training

### Epic 1.1: NER Data Curation

**Datasets:** conll2003 (20K), wikineural (116K)

#### Issue 1.1.1: Validate CoNLL-2003 Format

- **Status:** 🟢 Ready
- **Task:** Verify BIO tag consistency (9 tags: O, B/I-PER/ORG/LOC/MISC)
- **Action:** Map to NER_GENERAL_LABELS (17 tags) - unused tags stay 0
- **Output:** `data/curated/ner/conll2003/`

#### Issue 1.1.2: Validate WikiNeural Format

- **Status:** 🟢 Ready
- **Task:** Verify same BIO scheme as CoNLL-2003
- **Note:** Uses 'tags' column, not 'ner_tags' - already handled in download
- **Output:** `data/curated/ner/wikineural/`

#### Issue 1.1.3: Create Combined NER Dataset

- **Task:** Merge CoNLL + WikiNeural with consistent format
- **Target:** ~130K samples
- **Output:** `data/curated/ner/combined/train.jsonl`

---

### Epic 1.2: Sentiment Data Curation

**Datasets:** sst2 (68K), amazon_polarity (optional)

#### Issue 1.2.1: Map SST-2 Binary → 5-Class

- **Status:** 🟡 Needs mapping
- **Problem:** SST-2 is binary (0=negative, 1=positive)
- **Solution:**
  - Use confidence scores if available, else:
  - 0 → "negative" (label 1)
  - 1 → "positive" (label 3)
  - Add neutral samples from other sources
- **Output:** `data/curated/sentiment/sst2_5class/`

#### Issue 1.2.2: Create 5-Class Sentiment Dataset

- **Task:** Combine SST-2 + find neutral/extreme samples
- **Options:**
  1. DynaSent dataset (HuggingFace) - has 5-class labels
  2. Amazon Reviews with star ratings → sentiment
  3. SemEval sentiment datasets
- **Target:** ~100K samples with balanced 5-class distribution
- **Output:** `data/curated/sentiment/combined/`

---

### Epic 1.3: Emotions Data Curation

**Datasets:** familyos_emotions (247K silver + 600 gold)

> **NOTE:** GoEmotions (54K) downloaded but **SKIPPED** - FamilyOS emotions dataset is 5x larger with correct 44-label schema. No mapping needed.

#### Issue 1.3.1: ⏭️ SKIP - GoEmotions Mapping

- **Status:** ⏭️ SKIPPED
- **Reason:** FamilyOS emotions silver (246K) already uses correct 44-label schema
- **Decision:** Use FamilyOS emotions directly, no GoEmotions mapping needed

#### Issue 1.3.2: Validate FamilyOS Emotions Silver Data

- **Status:** 🟡 Needs validation
- **Task:**
  1. Verify all 44 emotion labels present
  2. Check label distribution balance
  3. Validate text quality (no template leakage)
- **Data:** 246K silver + 600 gold
- **Output:** Validation report + `data/curated/emotions/familyos/`

#### Issue 1.3.3: Balance Emotions Dataset

- **Task:** Address class imbalance in emotions
- **Strategy:**
  1. Undersample dominant classes (neutral, joy)
  2. Keep all samples of rare family emotions (nostalgia, bittersweet, etc.)
  3. Combine GoEmotions (mapped) + FamilyOS silver/gold
- **Target:** ~200K balanced samples
- **Output:** `data/curated/emotions/combined/`

---

### Epic 1.4: Safety Data Curation (Stage A - safety_generic)

**Datasets:** civil_comments_curated (192K)

> **NOTE:** BeaverTails (300K) and suicide_prediction (232K) downloaded but **NOT USED FOR STAGE A**
>
> - civil_comments_curated already has balanced 8-label safety_generic data
> - BeaverTails/suicide data could be used to augment if needed later

#### Issue 1.4.1: ✅ Civil Comments Curated (DONE)

- **Status:** 🟢 Complete
- **Result:** 172K train + 19K val, all 8 labels balanced 12-21%
- **Location:** `data/public/civil_comments_curated/`

#### Issue 1.4.2: ⏭️ SKIP - BeaverTails Mapping

- **Status:** ⏭️ SKIPPED
- **Reason:** civil_comments_curated (192K) is sufficient for Stage A safety_generic
- **Downloaded:** Yes (300K samples in `data/public/beavertails/`)
- **Future Use:** Can augment if safety_generic performance is insufficient

#### Issue 1.4.3: ⏭️ SKIP - Suicide Dataset for Stage A

- **Status:** ⏭️ SKIPPED
- **Reason:** civil_comments_curated already has self_harm samples (12.5%)
- **Downloaded:** Yes (232K samples in `data/public/suicide_prediction/`)
- **Future Use:** May use for Stage B safety_familyos CRISIS augmentation

#### Issue 1.4.4: ⏭️ SKIP - Master Safety Dataset

- **Status:** ⏭️ SKIPPED
- **Reason:** civil_comments_curated is already the master safety dataset for Stage A
- **Note:** No merging needed - single source is cleaner

#### Issue 1.4.5: ⏭️ MOVED TO STAGE B - CRISIS Calibration

- **Status:** ⏭️ MOVED
- **Reason:** CRISIS is a safety_familyos label (Stage B), not safety_generic (Stage A)
- **See:** Epic 2.3 for Stage B safety_familyos curation

---

### Epic 1.5: NLI Data Curation

**Datasets:** mnli (402K), snli (569K)

#### Issue 1.5.1: Validate MNLI/SNLI Format

- **Status:** 🟢 Ready
- **Task:** Verify premise/hypothesis/label format
- **Labels:** 0=entailment, 1=neutral, 2=contradiction
- **Note:** SNLI has -1 labels (no gold) - already filtered in download

#### Issue 1.5.2: Balance NLI Dataset

- **Problem:** Combined ~970K samples - too large
- **Solution:**
  1. Sample 200K from combined (balanced across 3 labels)
  2. Prefer shorter sequences for efficiency
  3. Ensure genre diversity (MNLI has 5 genres)
- **Target:** 200K samples, ~67K per class
- **Output:** `data/curated/nli/combined/`

---

### Epic 1.6: Embedding Data Curation

**Datasets:** stsb (8.6K), allnli (981K)

#### Issue 1.6.1: Validate STS-B Format

- **Status:** 🟢 Ready
- **Format:** sentence1, sentence2, score (0-5)
- **Task:** Normalize scores to 0-1 range
- **Output:** `data/curated/embedding/stsb/`

#### Issue 1.6.2: Curate AllNLI for Embedding Training

- **Problem:** 942K train - too large, may not be optimal format
- **Current Format:** sentence1, sentence2, score
- **Solution:**
  1. Sample 100K high-quality pairs
  2. Ensure score distribution is balanced
  3. Add hard negatives from same batch
- **Target:** 100K pairs
- **Output:** `data/curated/embedding/allnli_pairs/`

#### Issue 1.6.3: Create Triplet Dataset

- **Task:** Convert pairs → triplets for contrastive learning
- **Method:**
  - anchor = sentence1
  - positive = sentence2 (high score)
  - negative = random sentence2 (low score)
- **Target:** 50K triplets
- **Output:** `data/curated/embedding/triplets/`

---

### Epic 1.7: Temporal Data Curation

**Datasets:** familyos_temporal (28K)

#### Issue 1.7.1: Validate Temporal Silver Data

- **Status:** 🟡 Needs validation
- **Task:**
  1. Verify 13 BIO tags present (6 entity types)
  2. Check temporal expression quality
  3. Validate date/time parsing accuracy
- **Labels:** DATE_ABS, DATE_REL, TIME, DURATION, FREQUENCY, AGE
- **Output:** Validation report

#### Issue 1.7.2: Augment Temporal with Public Data

- **Problem:** Only 28K samples
- **Solution:** Find public temporal datasets
- **Options:**
  1. TempEval-3 (TimeBank) - ~6K samples
  2. WikiWars - temporal expressions in news
  3. Generate synthetic temporal data
- **Target:** 50K total
- **Output:** `data/curated/temporal/combined/`

---

## Milestone 2: Stage B Data Curation (FamilyOS Domain)

**Goal:** Prepare FamilyOS-specific datasets for domain adaptation

### Epic 2.1: NER Family Data Curation

**Datasets:** familyos_ner_family (26K silver + 250 gold)

#### Issue 2.1.1: Validate NER Family Silver Data

- **Task:**
  1. Verify 21 BIO tags (10 entity types)
  2. Check entity boundary accuracy
  3. Identify annotation errors
- **Entities:** PERSON, KINSHIP, NICKNAME, PET, HOME_LOC, FAMILY_EVENT, ROUTINE, TRADITION, MILESTONE, HEIRLOOM

#### Issue 2.1.2: Expand Gold NER Data

- **Problem:** Only 250 gold samples
- **Target:** 500+ gold samples
- **Action:** Manual annotation or gold curation from silver
- **Priority:** Focus on rare entities (HEIRLOOM, TRADITION, MILESTONE)

#### Issue 2.1.3: Balance NER Family Classes

- **Task:** Analyze entity distribution, balance if needed
- **Strategy:** Oversample rare entity types
- **Output:** `data/curated/ner_family/`

---

### Epic 2.2: Ingress Data Curation

**Datasets:** familyos_ingress (45K silver + 360 gold)

#### Issue 2.2.1: Validate Ingress 12-Class Distribution

- **Task:**
  1. Verify all 12 domain labels present
  2. Check class balance
  3. Identify confusing samples (DIARY vs MEMORY, TASK vs PLANNING)
- **Labels:** DIARY, TASK, HEALTH, FINANCE, RELATIONSHIP, WORK, META, MEMORY, PLANNING, CELEBRATION, CONCERN, GRATITUDE

#### Issue 2.2.2: Create Confusion Matrix Analysis

- **Task:** Identify commonly confused label pairs
- **Action:**
  1. Run preliminary classifier
  2. Analyze error patterns
  3. Add clarifying samples for confused pairs

#### Issue 2.2.3: Balance Ingress Classes

- **Problem:** META, FINANCE may be underrepresented
- **Target:** Each class ≥ 3K samples
- **Output:** `data/curated/ingress/`

---

### Epic 2.3: Safety FamilyOS (4-Band) Curation

**Status:** 🔴 Critical for Stage B
**Datasets:** familyos_safety (28K silver + 386 gold)

**Current Distribution (Silver):**

| Band | Count | Percentage |
|------|-------|------------|
| GREEN | 12,502 | 43.6% |
| AMBER | 8,388 | 29.3% |
| RED | 6,090 | 21.2% |
| CRISIS | 1,686 | 5.9% |

#### Issue 2.3.1: ✅ Validate 4-Band Distribution

- **Status:** ✅ DONE (analyzed above)
- **Assessment:** Distribution is reasonable but CRISIS is underrepresented for 98% recall target
- **Next:** Need to oversample CRISIS and add cultural robustness samples

#### Issue 2.3.2: 🔴 Create Indian Cultural Robustness Set

- **Status:** 🔴 GAP IDENTIFIED - Only ~4 hyperbole samples in current data!
- **Problem:** Current data lacks Indian English hyperbole expressions that should NOT trigger CRISIS
- **Impact:** Without this, model will have Cultural FP > 2% target

**Required Samples (Target: 500+):**

| Phrase Pattern | Example | Correct Label | Count Needed |
|----------------|---------|---------------|--------------|
| "die of [emotion]" | "I'll die of embarrassment" | GREEN | 50 |
| "killing me" | "This heat is killing me" | GREEN/AMBER | 50 |
| "head is bursting" | "Work stress, my head is bursting" | AMBER | 50 |
| "going mad" | "Kids driving me mad" | AMBER | 50 |
| "could just die" | "So tired I could just die" | GREEN | 50 |
| "मर जाऊंगी" | "मर जाऊंगी इस शर्म से" | GREEN | 30 |
| "tension" | "Diwali shopping tension" | AMBER | 50 |
| "will be death of me" | "MIL will be the death of me" | AMBER | 50 |
| "heart attack ho jayega" | "Price dekh ke heart attack" | GREEN | 30 |
| "pagal ho jaungi" | "Bachche pagal kar denge" | AMBER | 40 |

**Implementation Options:**

1. **Option A: Agent-Generated (Recommended)**
   - Use LLM to generate 500+ samples following patterns above
   - Human review required for each sample
   - Script: `scripts/agents/indian_hyperbole_generator.py`

2. **Option B: Manual Curation**
   - Create 500+ samples manually
   - High quality but time-intensive
   - Requires cultural consultant

3. **Option C: Extract from Reddit/Quora India**
   - Scrape Indian subreddits (r/india, r/indiasocial)
   - Extract hyperbolic expressions
   - Requires NDA review for licensing

**Deliverable:** `data/curated/safety_familyos/cultural_robustness/`

#### Issue 2.3.3: 🔴 Oversample CRISIS Samples

- **Status:** 🔴 Critical for 98% recall
- **Problem:** Only 1,686 CRISIS samples (5.9%)
- **Target:** 98% recall on CRISIS
- **Solution:**
  1. Create held-out CRISIS test set (200 samples)
  2. Upsample remaining to 20x for training (~30K effective)
  3. Add suicide_prediction dataset samples (mapped to CRISIS)
- **Sources:**
  1. familyos_safety CRISIS samples (1,686)
  2. suicide_prediction dataset (positive cases → CRISIS)
- **Output:** `data/curated/safety_familyos/crisis_augmented/`

---

### Epic 2.4: Intent Data Curation

**Datasets:** familyos_intents (32K silver + 600 gold)

#### Issue 2.4.1: Validate 8-Intent Distribution

- **Labels:** log_memory, query_memory, set_reminder, express_feeling, seek_advice, share_news, reflect, other

#### Issue 2.4.2: Balance Intent Classes

- **Problem:** "other" may dominate
- **Target:** Each intent ≥ 3K samples
- **Output:** `data/curated/intent/`

---

### Epic 2.5: Relation Data Curation

**Datasets:** familyos_relations (43K silver + 313 gold)

#### Issue 2.5.1: Validate 15-Relation Distribution

- **Labels:** no_relation, parent_of, child_of, spouse_of, sibling_of, grandparent_of, grandchild_of, aunt_uncle_of, niece_nephew_of, cousin_of, pet_of, friend_of, colleague_of, lives_at, owns

#### Issue 2.5.2: Address no_relation Imbalance

- **Problem:** no_relation typically dominates (~50%)
- **Solution:** Undersample no_relation to 20%
- **Output:** `data/curated/relation/`

---

### Epic 2.6: Embedding FamilyOS Curation

**Datasets:** familyos_embeddings (34K silver + 1K gold)

#### Issue 2.6.1: Validate Triplet Quality

- **Task:** Check anchor/positive/negative coherence
- **Verify:** Positives are semantically similar, negatives are different

#### Issue 2.6.2: Add Family-Specific Hard Negatives

- **Task:** Create challenging negatives for family context
- **Example:**
  - anchor: "dinner with mom"
  - positive: "lunch with mother"
  - hard_negative: "meeting with manager" (similar structure, different context)
- **Output:** `data/curated/embedding_familyos/`

---

## Milestone 3: Data Pipeline & Validation

### Epic 3.1: Unified Data Format

#### Issue 3.1.1: Define Standard JSONL Schema

```json
// Token Classification (NER, Temporal)
{
  "id": "uuid",
  "text": "raw text",
  "tokens": ["token", "list"],
  "labels": [0, 1, 2, 0, 0],  // BIO tag indices
  "task": "ner_general"
}

// Sequence Classification
{
  "id": "uuid",
  "text": "raw text",
  "label": 3,  // single-label
  "labels": [0, 0, 1, 0, 1, 0, 0, 0],  // multi-label (multi-hot)
  "task": "emotions"
}

// Pair Classification (NLI, Relation)
{
  "id": "uuid",
  "text1": "premise",
  "text2": "hypothesis",
  "label": 0,  // entailment
  "task": "nli"
}

// Embedding
{
  "id": "uuid",
  "anchor": "anchor text",
  "positive": "positive text",
  "negative": "negative text",  // optional
  "score": 0.85,  // optional
  "task": "embedding"
}
```

#### Issue 3.1.2: Create Format Conversion Scripts

- **Task:** Scripts to convert each source format to standard
- **Output:** `scripts/curation/convert_*.py`

#### Issue 3.1.3: Create Data Validation Suite

- **Task:** Automated checks for:
  1. Schema compliance
  2. Label range validity
  3. Text length limits
  4. Duplicate detection
  5. Class distribution balance
- **Output:** `scripts/curation/validate_dataset.py`

---

### Epic 3.2: Data Quality Metrics

#### Issue 3.2.1: Implement Quality Scorers

- **Metrics:**
  1. Text quality (perplexity, grammar)
  2. Label confidence (for silver data)
  3. Annotation agreement (for gold data)
  4. Class balance score

#### Issue 3.2.2: Generate Quality Reports

- **Task:** Per-dataset quality dashboard
- **Output:** `data/curated/*/quality_report.json`

---

### Epic 3.3: Train/Val/Test Splits

#### Issue 3.3.1: Create Stratified Splits

- **Strategy:**
  1. 80/10/10 train/val/test split
  2. Stratified by label distribution
  3. No data leakage (same text in multiple splits)

#### Issue 3.3.2: Create Held-Out Test Sets

- **Critical:** Safety test sets must be held out completely
- **Output:** `data/curated/*/test_holdout.jsonl`

---

## Milestone 4: Curation Scripts & Automation

### Epic 4.1: Curation Script Development

#### Issue 4.1.1: Create Master Curation Script

```bash
python scripts/curate_all.py \
    --stage a \
    --output_dir data/curated \
    --validate
```

#### Issue 4.1.2: Create Per-Task Curation Scripts

- `scripts/curation/curate_ner.py`
- `scripts/curation/curate_sentiment.py`
- `scripts/curation/curate_emotions.py`
- `scripts/curation/curate_safety.py`
- `scripts/curation/curate_nli.py`
- `scripts/curation/curate_embedding.py`
- `scripts/curation/curate_temporal.py`

#### Issue 4.1.3: Create FamilyOS Curation Scripts

- `scripts/curation/curate_ner_family.py`
- `scripts/curation/curate_ingress.py`
- `scripts/curation/curate_safety_familyos.py`
- `scripts/curation/curate_intent.py`
- `scripts/curation/curate_relation.py`

---

## Summary: Curation Priorities

### P0 (Must Complete Before Training)

| Issue | Task | Blocker For |
|-------|------|-------------|
| 1.4.2 | Map BeaverTails → safety_generic | Stage A safety |
| 1.4.3 | Curate suicide → self_harm | Stage A safety |
| 1.4.4 | Create master safety dataset | Stage A training |
| 1.3.1 | Map GoEmotions → FamilyOS 44 | Stage A emotions |
| 2.3.2 | Indian cultural robustness set | Safety FP target |
| 2.3.3 | CRISIS oversampling | Safety recall target |

### P1 (Important for Quality)

| Issue | Task | Impact |
|-------|------|--------|
| 1.2.1 | SST-2 binary → 5-class | Sentiment accuracy |
| 1.3.3 | Balance emotions | Emotion F1 |
| 1.5.2 | Balance NLI | NLI accuracy |
| 2.1.2 | Expand gold NER | NER family F1 |

### P2 (Nice to Have)

| Issue | Task | Impact |
|-------|------|--------|
| 1.7.2 | Augment temporal | Temporal F1 |
| 2.6.2 | Family hard negatives | Embedding quality |

---

## Appendix: Dataset → Task Mapping

| Dataset | Task | Labels | Format | Target Size |
|---------|------|--------|--------|-------------|
| conll2003 + wikineural | ner_general | 17 BIO | token | 130K |
| sst2 + dynasent | sentiment | 5 class | sequence | 100K |
| goemotions + familyos | emotions | 44 multi | sequence | 200K |
| civil_comments + beavertails + suicide | safety_generic | 8 multi | sequence | 400K |
| mnli + snli | nli | 3 class | pair | 200K |
| stsb + allnli | embedding | score | pair | 100K |
| familyos_temporal | temporal | 13 BIO | token | 50K |
| familyos_ner | ner_family | 21 BIO | token | 26K |
| familyos_ingress | ingress | 12 class | sequence | 45K |
| familyos_safety | safety_familyos | 4 class | sequence | 30K |
| familyos_intent | intent | 8 class | sequence | 32K |
| familyos_relation | relation | 15 class | pair | 40K |

**Total Curated Target: ~1.35M samples**

---

**Document Version:** 1.0
**Created:** December 3, 2025
**Author:** FamilyOS ML Team
