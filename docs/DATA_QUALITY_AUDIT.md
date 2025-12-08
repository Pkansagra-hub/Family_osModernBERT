# FamilyOS Data Quality Audit Rules

## Overview

This document defines all validation rules for the FamilyOS dataset. These rules ensure consistency between labels, prevent training signal conflicts, and validate data integrity.

**Dataset Locations:**
- `data/familyos/unified/output/` - Real data (~69k samples)
- `data/familyos/unified/output_synthetic/` - Synthetic data (~350k samples)

---

## Category 1: Hub Routing vs Task Fields Consistency

### Rule 1.1: EMO Routing vs Emotions
- **Condition:** `hub_routing.EMO = True` should correlate with non-neutral emotions
- **Conflict:** `hub_routing.EMO = False` with emotions other than `["neutral"]`
- **Severity:** CRITICAL
- **Impact:** Model learns to predict emotions when routing says "no emotion needed"

**Example Conflict:**
```json
{
  "text": "Remind me to give Mittens her flea medication...",
  "emotions": ["caring"],
  "hub_routing": {"EMO": false}
}
```

### Rule 1.2: REL Routing vs Relations
- **Condition:** `hub_routing.REL = True` should have `relations[]` populated
- **Conflict:** `hub_routing.REL = True` with `relations = []`
- **Severity:** HIGH
- **Impact:** Model learns REL routing without relation extraction targets

### Rule 1.3: REL Routing False but Relations Exist
- **Condition:** `hub_routing.REL = False` should have empty `relations[]`
- **Conflict:** `hub_routing.REL = False` with populated `relations[]`
- **Severity:** HIGH

### Rule 1.4: MEM Routing vs Memory Intents
- **Condition:** `hub_routing.MEM = True` should correlate with memory intents
- **Valid intents for MEM=True:** `query_memory`, `log_memory`, `reflect`
- **Conflict:** `MEM = True` with non-memory intent
- **Severity:** MEDIUM

### Rule 1.5: TASK Routing vs Task Intents
- **Condition:** `hub_routing.TASK = True` should correlate with task intents
- **Valid intents for TASK=True:** `set_reminder`, `seek_advice`
- **Conflict:** `TASK = False` with `intent = set_reminder`
- **Severity:** MEDIUM

---

## Category 2: Sentiment vs Emotions Consistency

### Rule 2.1: Positive Sentiment with Negative Emotions
- **Conflict:** `sentiment = "very_positive"` with emotions like `["sadness", "frustration", "worry"]`
- **Exception:** Bittersweet contexts may have mixed signals
- **Severity:** HIGH

### Rule 2.2: Negative Sentiment with Positive Emotions
- **Conflict:** `sentiment = "very_negative"` with emotions like `["joy", "excitement", "love"]`
- **Severity:** HIGH

### Rule 2.3: Neutral Sentiment with Strong Emotions
- **Condition:** `sentiment = "neutral"` should not have intense emotions
- **Suspicious:** `sentiment = "neutral"` with `["joy", "sadness", "love", "anger"]`
- **Severity:** MEDIUM
- **Note:** Some samples legitimately have neutral sentiment with mild emotions

### Rule 2.4: Sentiment Strength Mismatch
- **Conflict:** `sentiment = "positive"` vs `"very_positive"` inconsistency with emotion intensity
- **Severity:** LOW

---

## Category 3: Intent vs Hub Routing Consistency

### Rule 3.1: Memory Intent but MEM=False
- **Conflict:** `intent = "query_memory"` but `hub_routing.MEM = False`
- **Severity:** CRITICAL

### Rule 3.2: Memory Logging but MEM=False
- **Conflict:** `intent = "log_memory"` but `hub_routing.MEM = False`
- **Severity:** CRITICAL

### Rule 3.3: Reminder Intent but TASK=False
- **Conflict:** `intent = "set_reminder"` but `hub_routing.TASK = False`
- **Severity:** HIGH

### Rule 3.4: Feeling Expression but EMO=False
- **Conflict:** `intent = "express_feeling"` but `hub_routing.EMO = False`
- **Severity:** CRITICAL

### Rule 3.5: Share News but REL=False
- **Condition:** `intent = "share_news"` often involves relationships
- **Suspicious:** News about family members with `REL = False`
- **Severity:** MEDIUM

---

## Category 4: NER vs Relations Consistency

### Rule 4.1: KINSHIP Token but No Relation
- **Conflict:** `ner_family` contains `label = "KINSHIP"` but `relations = []`
- **Exception:** Generic family references ("family trip")
- **Severity:** MEDIUM

### Rule 4.2: PERSON Token with Family Context but No Relation
- **Condition:** PERSON tokens in family context should generate relations
- **Conflict:** "my wife, Priya" with no `spouse_of` relation
- **Severity:** HIGH

### Rule 4.3: REL=True but Empty Relations and No KINSHIP
- **Conflict:** Routing says relationships but no evidence in NER or relations
- **Severity:** HIGH

### Rule 4.4: Relation Object Mismatch with NER
- **Conflict:** `relation.object = "Maya"` but no "Maya" in `ner_family`
- **Severity:** MEDIUM

---

## Category 5: Safety Label Consistency

### Rule 5.1: RED Safety without Crisis Content
- **Condition:** `safety_familyos = "RED"` should have crisis-related content
- **Keywords:** self-harm, abuse, violence, emergency
- **Conflict:** RED label on benign content
- **Severity:** CRITICAL

### Rule 5.2: GREEN Safety with Concerning Content
- **Conflict:** `safety_familyos = "GREEN"` with worry/health/concern emotions
- **Suspicious:** Health concerns marked as GREEN
- **Severity:** MEDIUM

### Rule 5.3: AMBER Safety Consistency
- **Condition:** AMBER should match worry/concern/health content
- **Valid:** Parenting concerns, health questions, stress
- **Severity:** MEDIUM

### Rule 5.4: Safety vs Emotion Alignment
- **Conflict:** `safety = "GREEN"` with `emotions = ["overwhelmed", "worry"]` and `intent = "seek_advice"`
- **Expected:** AMBER for stress/overwhelm seeking advice
- **Severity:** MEDIUM

---

## Category 6: Temporal Field Consistency

### Rule 6.1: Missing Temporal Annotations
- **Conflict:** Text contains time references but `temporal = []`
- **Keywords:** "yesterday", "tomorrow", "next week", "7 PM", "Friday"
- **Severity:** MEDIUM

### Rule 6.2: Temporal Span Validation
- **Condition:** `text[start:end]` must equal `token`
- **Conflict:** Off-by-one errors, whitespace issues
- **Severity:** HIGH (see Rule 11)

### Rule 6.3: Temporal Label Appropriateness
- **Valid Labels:** `DATE_REL`, `DATE_ABS`, `TIME`, `FREQUENCY`, `DURATION`, `AGE`
- **Conflict:** Wrong label type for the temporal expression
- **Severity:** LOW

---

## Category 7: ID and Deduplication

### Rule 7.1: Duplicate IDs
- **Conflict:** Same `id` appears multiple times (different content)
- **Severity:** CRITICAL
- **Note:** OBSERVED - multiple `syn_00021`, `syn_00022` in data

### Rule 7.2: ID Format Consistency
- **Format:** `syn_XXXXX` for synthetic, `fam_XXXXX` for real
- **Conflict:** Mixed formats or malformed IDs
- **Severity:** LOW

### Rule 7.3: Near-Duplicate Text
- **Conflict:** Semantically identical text with different labels
- **Detection:** Fuzzy matching or embedding similarity
- **Severity:** HIGH

---

## Category 8: Multi-Label Emotion Conflicts

### Rule 8.1: Contradictory Emotions
- **Conflict:** `["joy", "sadness"]` without `"bittersweet"` context
- **Conflict:** `["excitement", "boredom"]`
- **Severity:** MEDIUM

### Rule 8.2: Redundant Emotion Labels
- **Conflict:** `["love", "affection", "warmth"]` - overlapping semantics
- **Impact:** Inflated label counts, model confusion
- **Severity:** LOW

### Rule 8.3: Neutral Infection (CRITICAL)
- **Rule:** `neutral` should be MUTUALLY EXCLUSIVE with all other emotions
- **Conflict:** `emotions = ["caring", "neutral"]`
- **Rationale:** You cannot be "Joyful and Neutral" simultaneously
- **Severity:** CRITICAL

**Example Conflict:**
```json
{
  "text": "Remind me to call my sister, Priya, after work today.",
  "emotions": ["caring", "neutral"]  // INVALID
}
```

---

## Category 9: Text Quality

### Rule 9.1: Text Length vs Label Complexity
- **Suspicious:** Text < 20 characters with > 3 emotion labels
- **Suspicious:** Text > 200 characters with 0-1 labels
- **Severity:** LOW

### Rule 9.2: Empty or Whitespace Text
- **Conflict:** `text = ""` or `text = "   "`
- **Severity:** CRITICAL

### Rule 9.3: Encoding Issues
- **Conflict:** Garbled characters, broken Unicode
- **Severity:** HIGH

---

## Category 10: Span Validation (NER/Temporal)

### Rule 10.1: Start/End Index Validation
- **Condition:** `start < end` and both within text bounds
- **Conflict:** Negative indices, end > len(text)
- **Severity:** CRITICAL

### Rule 10.2: Token Match Validation
- **Condition:** `text[start:end] == token`
- **Conflict:** Extracted token doesn't match annotated token
- **Severity:** CRITICAL

### Rule 10.3: Overlapping Spans
- **Conflict:** Two NER entities with overlapping character ranges
- **Severity:** MEDIUM

---

## Category 11: Whitespace/Token Mismatch (CRITICAL)

### Rule 11.1: Exact Token Alignment
- **Risk:** Transformers are extremely sensitive to token alignment
- **Check:** `text[start:end]` must EXACTLY equal `token`
- **Conflict:** Any trailing/leading whitespace, off-by-one errors
- **Action:** Discard or fix misaligned samples

**Example Conflict:**
```json
{
  "text": "Can you tell me when my niece Maya's ballet recital is?",
  "ner_family": [
    {"start": 16, "end": 20, "label": "KINSHIP", "token": "niece"}
  ]
}
// text[16:20] = "when" NOT "niece" - OFF BY SEVERAL CHARACTERS!
```

### Rule 11.2: Unicode Normalization
- **Risk:** Different Unicode representations of same character
- **Check:** Normalize both text and token before comparison
- **Severity:** HIGH

### Rule 11.3: Tokenizer Boundary Alignment
- **Risk:** Annotation boundaries may not align with subword tokenizer
- **Impact:** Training signal bleeds across tokens
- **Severity:** MEDIUM

---

## Category 12: The "Caring" Label Problem

### Rule 12.1: Caring vs Task Context
- **Issue:** "Caring" is applied to task/command sentences
- **Conflict:** Utilitarian commands labeled as emotional
- **Example:** "Buy cat food" labeled as `["caring"]`
- **Expected:** `["neutral"]` for commands

### Rule 12.2: Caring with EMO=False
- **Conflict:** `emotions = ["caring"]` but `hub_routing.EMO = False`
- **Rate:** 43.6% in synthetic data (CRITICAL)
- **Action:** Force label to `neutral` when `EMO = False`

### Rule 12.3: Intent-Based Caring Filter
- **Rule:** If `intent = "set_reminder"` AND no explicit emotional language, emotion should be `neutral`
- **Severity:** HIGH

---

## Category 13: Synthetic Data Skew

### Rule 13.1: Emotion Distribution Skew
- **Observation:** Synthetic over-represents:
  - `caring`: 10.14x vs real data
  - `worry`: 8.30x vs real data
  - `nostalgia`: 8.07x vs real data
- **Impact:** Model biased toward these emotions
- **Severity:** HIGH

### Rule 13.2: Pattern Rigidity
- **Issue:** Synthetic data may have repetitive patterns
- **Check:** Template detection, n-gram analysis
- **Severity:** MEDIUM

### Rule 13.3: Family Entity Over-Tagging
- **Issue:** Mentioning "Mom" auto-triggers emotional labels
- **Example:** "Mom is late" -> `["worry", "caring"]` (should be neutral factual)
- **Severity:** HIGH

---

## Audit Execution Plan

### Phase 1: Critical Issues (Block Training)
1. Rule 8.3: Neutral Infection
2. Rule 11.1: Token Alignment
3. Rule 1.1: EMO Routing vs Emotions
4. Rule 7.1: Duplicate IDs
5. Rule 3.1-3.4: Intent/Routing Mismatches

### Phase 2: High Priority (Fix Before Training)
1. Rule 12.1-12.3: Caring Label Problem
2. Rule 2.1-2.2: Sentiment/Emotion Conflicts
3. Rule 4.1-4.2: NER/Relations Consistency
4. Rule 10.1-10.2: Span Validation

### Phase 3: Medium Priority (Monitor)
1. Rule 5.1-5.4: Safety Label Consistency
2. Rule 6.1-6.3: Temporal Field Consistency
3. Rule 13.1-13.3: Synthetic Data Skew

### Phase 4: Low Priority (Nice to Have)
1. Rule 9.1: Text Length Analysis
2. Rule 7.2: ID Format Consistency
3. Rule 8.2: Redundant Labels

---

## Recommended Fixes

### Fix 1: Gated Training Strategy
Use `hub_routing` as a loss mask:
- When `EMO = False`, force emotion label to `neutral`
- When `REL = False`, skip relation extraction head
- When `MEM = False`, skip memory head

### Fix 2: Super-Label Grouping
Reduce 44 emotion labels to 8 categories:
- JOY: joy, excitement, celebration, pride, relief
- AFFECTION: love, warmth, caring, gratitude
- ANXIETY: worry, overwhelmed, frustration, annoyance
- SADNESS: sadness, grief, disappointment, longing
- NOSTALGIA: nostalgia, bittersweet
- NEUTRAL: neutral, patience
- ANTICIPATION: excitement, hope, optimism
- CONTENTMENT: contentment, belonging, togetherness

### Fix 3: Neutral Infection Cleanup
```python
# Remove neutral from multi-label emotions
if len(emotions) > 1 and "neutral" in emotions:
    emotions.remove("neutral")
```

### Fix 4: Token Alignment Validation
```python
# Validate all NER and temporal spans
for entity in ner_family:
    extracted = text[entity["start"]:entity["end"]]
    if extracted != entity["token"]:
        flag_sample_for_review(sample)
```

### Fix 5: Add Auxiliary Routing Head
Train a separate head to predict the 4 routing booleans:
- EMO, REL, MEM, TASK
- Use this to gate the task-specific heads

---

## Metrics to Track

| Metric | Target | Current |
|--------|--------|---------|
| EMO/Emotion Conflict Rate | < 1% | 14.76% |
| Caring/EMO=False Rate | < 5% | 40% |
| Neutral Infection Rate | 0% | TBD |
| Token Alignment Errors | 0% | TBD |
| Duplicate ID Rate | 0% | TBD |
| Sentiment/Emotion Conflict Rate | < 2% | TBD |

---

## Appendix: Audit Script Location

Run the comprehensive audit:
```bash
python scripts/analyze_hub_routing_conflict.py
```

Full audit script (to be created):
```bash
python scripts/comprehensive_data_audit.py
```
