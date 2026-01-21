# UltraBERT v2 Head Redesign - Working Memory

**Started**: January 21, 2026
**Goal**: Keep proven encoder from checkpoint-18000, redesign heads for better quality

---

## Proven Foundation

### Encoder: checkpoint-18000 (KEEP)

**Location**: `outputs/modernbert-v2-for-v3-transfer/checkpoint-18000`

**Probe Results** (verified Jan 21, 2026):

| Test | Similarity | Verdict |
|------|------------|---------|
| verb_vs_milestone | -0.03 | EXCELLENT |
| determiner_vs_entity | 0.06 | EXCELLENT |
| adjective_vs_entity | 0.01 | EXCELLENT |
| common_noun_vs_org | -0.02 | EXCELLENT |
| same_word_diff_pos | 0.33 | GOOD |
| span_coherence | -0.02 | WEAK (needs head fix) |

**Conclusion**: Encoder discriminates perfectly. HEAD IS THE BOTTLENECK.

### Training Data Available

**Location**: `data/familyos/unified/output_healed_merged/`
**Format**: 113 shards, JSONL
**Total Samples**: 562,156

**Tasks in unified data**:

- emotions, sentiment, ner_family, safety_familyos, intent, ingress, relations, temporal
- **NOTE: NO ner_general** - only ner_family

**Sample Format**:

```json
{
  "id": "fam_00005",
  "text": "Exam on August 5th 2024.",
  "tasks": {
    "emotions": ["neutral"],
    "sentiment": "neutral",
    "ner_family": [],
    "safety_familyos": "GREEN",
    "intent": "other",
    "ingress": "PLANNING",
    "relations": [],
    "temporal": [{"start": 8, "end": 23, "label": "DATE_ABS", "token": "August 5th 2024"}]
  }
}
```

**ner_family format** (already span-based):

```json
{
  "text": "Hope sparked seeing Nani walk farther.",
  "ner_family": [
    {"start": 20, "end": 24, "label": "KINSHIP", "token": "Nani"}
  ]
}
```

---

## Data Gap Analysis

### ner_family: DATA EXISTS

| Aspect | Status |
|--------|--------|
| Location | `output_healed_merged/*.jsonl` |
| Format | Span-based (start, end, label) - PERFECT for GlobalPointer |
| Labels | KINSHIP, MILESTONE, HEIRLOOM, PET, NICKNAME, etc. |
| Volume | 562K samples (many with empty ner_family) |
| Action | Use directly |

### ner_general: DATA GAP

| Aspect | Status |
|--------|--------|
| Location | NOT in unified data |
| Labels | PERSON, ORG, LOC, MISC |
| Current training | CoNLL-2003, WikiNeural (public datasets) |
| Problem | Public data is news/Wikipedia, not family narratives |

**Options for ner_general**:

1. **Use CoNLL-2003 only** (current approach)
   - Pro: Standard benchmark, easy
   - Con: Domain mismatch (news vs family diaries)

2. **Generate synthetic FamilyOS data with ner_general labels**
   - Pro: Domain-matched data
   - Con: Need to generate ~50-100K samples

3. **Hybrid: CoNLL + synthetic FamilyOS**
   - Pro: Best of both worlds
   - Con: More work

**RECOMMENDATION**: Generate FamilyOS-style synthetic data with ner_general labels

- Same diary/family narrative style
- PERSON (family names), LOC (places visited), ORG (schools, companies)
- Can use LLM to generate, similar to how ner_family data was created

---

## Head Status Assessment

### SOLID (Keep Architecture, Maybe Retrain)

| Head | Current Architecture | Status | Notes |
|------|---------------------|--------|-------|
| **embedding** | EmbeddingHead (pooling + normalize) | SOLID | Works well |
| **emotions** | HierarchicalEmotionHead (44-class) | SOLID | Plain BCE fixed collapse |
| **intent** | IntentHead (8 classes) | SOLID | Simple task, works |
| **ingress** | SequenceClassificationHead (12 domains) | SOLID | Simple task, works |

### NEEDS REDESIGN

| Head | Current Problem | Garbage Rate | Priority |
|------|-----------------|--------------|----------|
| **ner_family** | Tags verbs as MILESTONE, "the" as PET | 66%+ | P0 |
| **ner_general** | Tags verbs as PERSON, partial entities | 66%+ | P0 |
| **temporal** | Same issues as NER | ~40%? | P1 |
| **safety_familyos** | Cultural FP ("dying of laughter" = CRISIS) | ~2% | P2 |
| **safety_generic** | Less critical for FamilyOS | Low | P3 |
| **nli** | May benefit from CrossAttention | Unknown | P3 |
| **relation** | Simple concat, may miss interactions | ~25% error | P2 |
| **sentiment** | Probably fine | Low | P4 |

---

## Architecture Decisions

### Decision 1: NER Heads (ner_general, ner_family, temporal)

**Current**:

```python
# TokenClassificationHead - Line 494
def forward(self, hidden_states):
    x = self.dropout(hidden_states)     # (B, L, 768)
    logits = self.classifier(x)         # (B, L, num_labels)
```

**Problem**: Per-token classification with no context aggregation or transition constraints.

**Proposed New Architecture - Options**:

#### Option A: CRF Layer

```
hidden_states → Dropout → Linear → CRF → valid BIO sequence
```

- Enforces: B-PER must be followed by I-PER or O, not I-ORG
- Prevents: Invalid transitions that cause garbage
- Con: Slower inference (Viterbi decoding)

#### Option B: Context Window

```
hidden_states → Conv1D(k=3) or BiLSTM → Dropout → Linear → logits
```

- Aggregates neighboring tokens before classifying
- Helps with span coherence
- Con: Adds parameters

#### Option C: Span-Based (SpanNER)

```
hidden_states → predict(start, end, label) instead of BIO
```

- No more partial entities
- Naturally handles multi-word entities
- Con: Different training objective, more complex

#### Option D: Hybrid (RECOMMENDED)

```
hidden_states → BiLSTM(1 layer) → Dropout → Linear → CRF
```

- BiLSTM adds context (fixes span coherence)
- CRF enforces valid transitions (fixes garbage BIO)
- Best of both worlds

**DECISION**: SUPERSEDED - See SOTA options below

---

## SOTA NER Architectures (World-Class Options)

### Tier 1: SOTA Leaders (2023-2025)

| Architecture | F1 CoNLL-03 | Key Innovation | Latency |
|--------------|-------------|----------------|---------|
| **W2NER** | 93.4+ | Word-Word relation matrix | +15-20ms |
| **Global Pointer** | 93.2+ | Efficient span detection | +5-10ms |
| **Biaffine NER** | 93.0+ | Biaffine attention for spans | +10-15ms |
| **UIE** | 93.0+ | Unified extraction (T5) | +50ms (too slow) |
| **MRC-NER** | 92.8+ | Query-based extraction | +30ms |

### Tier 2: Production Proven

| Architecture | F1 CoNLL-03 | Key Innovation | Latency |
|--------------|-------------|----------------|---------|
| **BERT-BiLSTM-CRF** | 92.4+ | Context + transitions | +8-12ms |
| **SpanNER** | 92.2+ | Span classification | +5-8ms |
| **Nested NER** | 92.0+ | Handles overlapping | +10ms |

---

## RECOMMENDED: Global Pointer

**Why Global Pointer?**

- SOTA on Chinese NER benchmarks (94%+ F1)
- Efficient: O(n) instead of O(n^2) for span enumeration
- Naturally handles nested entities
- Clean output: (start, end, label) tuples - NO BIO!
- No invalid transitions possible (BIO garbage eliminated by design)

**Architecture**:

```python
class GlobalPointerNERHead(nn.Module):
    """
    Global Pointer for efficient span-based NER.

    Instead of BIO tagging, directly predicts:
    - P(token_i is START of entity_type_k)
    - P(token_j is END of entity_type_k)
    - Combined: P(span[i:j] is entity_type_k)

    Key insight: Use RoPE-style relative position encoding
    to capture "i <= j" constraint naturally.
    """

    def __init__(self, hidden_size, num_labels, head_size=64):
        super().__init__()
        self.num_labels = num_labels
        self.head_size = head_size

        # Project to query/key for each label type
        self.q_proj = nn.Linear(hidden_size, num_labels * head_size * 2)
        self.k_proj = nn.Linear(hidden_size, num_labels * head_size * 2)

        # RoPE for relative position
        self.rope = RotaryPositionEncoding(head_size)

    def forward(self, hidden_states, attention_mask=None):
        batch, seq_len, _ = hidden_states.shape

        # Project to Q, K for all label types
        q = self.q_proj(hidden_states)  # (B, L, num_labels * head_size * 2)
        k = self.k_proj(hidden_states)

        # Reshape and apply RoPE
        q = q.view(batch, seq_len, self.num_labels, self.head_size, 2)
        k = k.view(batch, seq_len, self.num_labels, self.head_size, 2)
        q, k = self.rope(q, k)

        # Compute span scores: score[i,j,k] = q[i,k] @ k[j,k]
        scores = torch.einsum('bmnh,blnh->bmnl', q, k)  # (B, L, L, num_labels)

        # Mask: only allow i <= j (upper triangular)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device))
        scores = scores * mask.unsqueeze(0).unsqueeze(-1)

        return scores  # (B, L, L, num_labels)
```

**Decoding** (simple threshold, no Viterbi needed):

```python
def decode_global_pointer(scores, threshold=0.0):
    """Extract spans - just find positive scores in upper triangle."""
    entities = []
    probs = torch.sigmoid(scores)  # Convert to probabilities

    for label_id in range(num_labels):
        # Find all (i, j) where prob > threshold
        starts, ends = torch.where(probs[:, :, label_id] > threshold)
        for s, e in zip(starts, ends):
            if s <= e:  # Valid span (guaranteed by upper triangle)
                entities.append({"start": s, "end": e+1, "label": label_id})

    return entities
```

**Why This Kills BIO Garbage**:

- No B/I/O transitions to get wrong
- Directly outputs (start, end, label)
- "Lincoln School" = one span prediction, not 2 BIO tokens
- No "the" tagged as PET - it's never part of a span start/end

---

## GlobalPointer Data Format Requirements

### The Big Picture

Moving from BIO to GlobalPointer requires format changes:

| Component | BIO Format (OLD) | Span Format (NEW) |
|-----------|------------------|-------------------|
| **Data storage** | `{"tokens": [...], "ner_tags": [...]}` | `{"text": "...", "entities": [{start, end, label}]}` |
| **Collator** | TokenClassificationCollator | GlobalPointerCollator |
| **Head output** | `(B, L, num_labels)` logits per token | `(B, L, L, num_labels)` span scores |
| **Decoding** | BIO→spans (post-process) | Already spans |

### What Exists (Ready for GlobalPointer)

| Data Source | Format | Ready? |
|-------------|--------|--------|
| **ner_family unified** | Span: `{"start": 20, "end": 24, "label": "KINSHIP"}` | YES |
| **temporal unified** | Span: `{"start": 8, "end": 23, "label": "DATE_ABS"}` | YES |

### What Needs Conversion

| Data Source | Current Format | Action Needed |
|-------------|----------------|---------------|
| **CoNLL-2003** | BIO: `{"tokens": [...], "ner_tags": [...]}` | Create BIO→Span converter |
| **WikiNeural** | BIO: same as CoNLL | Use same converter |
| **ner_family generator** | Outputs BIO | Adapt for Span output |

### Converter: BIO → Span

```python
def bio_to_spans(tokens: list[str], bio_tags: list[int], label_map: dict) -> dict:
    """
    Convert BIO format to span format.

    Input:
        tokens: ["Emma", "lives", "in", "New", "York"]
        bio_tags: [1, 0, 0, 5, 6]  # B-PER, O, O, B-LOC, I-LOC
        label_map: {1: "PER", 2: "PER", 5: "LOC", 6: "LOC"}

    Output:
        {
            "text": "Emma lives in New York",
            "entities": [
                {"start": 0, "end": 4, "label": "PER"},
                {"start": 14, "end": 22, "label": "LOC"}
            ]
        }
    """
    text = " ".join(tokens)
    entities = []

    char_pos = 0
    i = 0
    while i < len(tokens):
        tag = bio_tags[i]
        if tag == 0:  # O
            char_pos += len(tokens[i]) + 1
            i += 1
            continue

        # Start of entity (B-tag)
        if tag % 2 == 1:  # B-tags are odd
            label = label_map[tag]
            start_char = char_pos
            end_char = char_pos + len(tokens[i])

            # Consume I-tags
            i += 1
            while i < len(tokens) and bio_tags[i] == tag + 1:
                end_char += 1 + len(tokens[i])  # +1 for space
                i += 1

            entities.append({
                "start": start_char,
                "end": end_char,
                "label": label
            })
            char_pos = end_char + 1
        else:
            # Orphan I-tag - skip
            char_pos += len(tokens[i]) + 1
            i += 1

    return {"text": text, "entities": entities}
```

### Data Generator Adaptation (ner_general for FamilyOS)

**Existing**: `scripts/agents/ner_data_generator.py` outputs BIO

**New**: Create `scripts/agents/ner_general_span_generator.py` that:

1. Uses same OpenRouterClient, SilverDataManager
2. Simplified SYSTEM_PROMPT (no BIO rules needed!)
3. Output format: `{"text": "...", "entities": [...]}`
4. Labels: PER, ORG, LOC, MISC (4 types vs ner_family's 10)
5. FamilyOS context: diary entries, family narratives, reminders

**Simpler prompt for LLM**:

```
Generate family diary entries with named entities annotated as:
- PER: People names (Emma, Uncle John, Dr. Smith)
- ORG: Organizations (Lincoln School, Google, St. Mary's Hospital)
- LOC: Locations (New York, backyard, kitchen)
- MISC: Other proper nouns (iPhone, Christmas, COVID-19)

Output format:
{"text": "Emma got accepted to Lincoln School today!", "entities": [{"start": 0, "end": 4, "label": "PER"}, {"start": 21, "end": 35, "label": "ORG"}]}
```

### GlobalPointer Collator (New)

```python
class GlobalPointerCollator:
    """
    Collates span-format data for GlobalPointer training.

    Input per sample:
        {"text": "...", "entities": [{"start": 0, "end": 4, "label": "PER"}]}

    Output batch:
        input_ids: (B, L)
        attention_mask: (B, L)
        span_labels: (B, L, L, num_labels) - 1 where span exists, 0 otherwise
    """

    def __call__(self, features):
        texts = [f["text"] for f in features]

        # Tokenize
        encoding = self.tokenizer(texts, padding=True, truncation=True,
                                   return_offsets_mapping=True)

        batch_size = len(texts)
        seq_len = len(encoding["input_ids"][0])

        # Build span labels: (B, L, L, num_labels)
        span_labels = torch.zeros(batch_size, seq_len, seq_len, self.num_labels)

        for b, feature in enumerate(features):
            offset_mapping = encoding["offset_mapping"][b]

            for entity in feature.get("entities", []):
                char_start, char_end = entity["start"], entity["end"]
                label_id = self.label_to_id[entity["label"]]

                # Find token indices for char span
                tok_start = tok_end = None
                for i, (cs, ce) in enumerate(offset_mapping):
                    if cs <= char_start < ce:
                        tok_start = i
                    if cs < char_end <= ce:
                        tok_end = i
                        break

                if tok_start is not None and tok_end is not None:
                    span_labels[b, tok_start, tok_end, label_id] = 1.0

        return {
            "input_ids": torch.tensor(encoding["input_ids"]),
            "attention_mask": torch.tensor(encoding["attention_mask"]),
            "span_labels": span_labels
        }
```

### Implementation Order

1. **BIO→Span converter** - Convert CoNLL-2003 for training
2. **GlobalPointerCollator** - Training data pipeline
3. **GlobalPointerNERHead** - The actual head
4. **ner_general_span_generator.py** - Generate FamilyOS ner_general data
5. **Training script** - Train on converted CoNLL + FamilyOS data

---

## Parallel Head Execution

**Current Architecture** (sequential):

```python
# modernbert_multitask.py - Line 519
def forward(self, capability: Capability):
    encoder_outputs = self.encoder(input_ids)  # 40ms
    head = self.heads[capability.value]        # Pick ONE head
    return head(encoder_outputs)               # Run ONE head
```

**Inference currently**: Call forward() 12 times = 12x head latency

**Proposed: Single Forward, All Heads**:

```python
def forward_all(self, input_ids, attention_mask, capabilities=None):
    """Encode once, run all heads in parallel."""

    # 1. Encode ONCE (40ms)
    hidden = self.encoder(input_ids, attention_mask).last_hidden_state

    # 2. Run all heads in parallel
    if capabilities is None:
        capabilities = list(self.heads.keys())

    # Option A: torch.jit.fork for GPU parallelism
    futures = {cap: torch.jit.fork(self.heads[cap], hidden)
               for cap in capabilities}
    results = {cap: torch.jit.wait(fut) for cap, fut in futures.items()}

    return results  # All 12 outputs in one call
```

**Latency Impact**:

- Current: encoder(40ms) + 12*head(5ms) = 100ms sequential
- Parallel: encoder(40ms) + max(heads)(10ms) = 50ms
- Your 52ms suggests you might already be parallel or batching somehow

---

### Decision 2: Safety Head (safety_familyos)

**Current Problem**: Indian English idioms cause false positives

**Examples**:

- "dying of laughter" → CRISIS (should be GREEN)
- "I could kill for a pizza" → RED (should be GREEN)

**Proposed Improvements**:

1. Expand keyword override dictionary with cultural idioms
2. Add confidence calibration per subcategory
3. Consider: Add "idiom detector" auxiliary head

**DECISION**: TBD

---

### Decision 3: Relation Head

**Current**:

```python
concat([e1_pooled, e2_pooled]) → Linear(1536 → 15)
```

**Problem**: Simple concat may miss entity interactions

**Proposed**: Make CrossAttentionPairEncoder default

**DECISION**: TBD

---

## Training Strategy

### Phase 1: Freeze Encoder, Train New Heads

```yaml
encoder:
  freeze: true  # OR lr: 1e-6 for light adaptation

heads:
  lr: 1e-4
  epochs: 3-5  # Should converge fast with frozen encoder
```

### Phase 2: Fine-tune Both (Optional)

```yaml
encoder:
  lr: 1e-5  # Very low

heads:
  lr: 5e-5
  epochs: 2
```

---

## Implementation Checklist

- [ ] Implement GlobalPointerNERHead
- [ ] Add RoPE (Rotary Position Encoding) support
- [ ] Create training script for head-only training
- [ ] Test on subset of data first
- [ ] Measure garbage rate reduction
- [ ] Apply to ner_general, ner_family, temporal
- [ ] Add parallel head execution (forward_all)
- [ ] Address safety_familyos cultural FP
- [ ] Address relation head

---

## Open Questions

1. ~~CRF vs Span-based~~: **DECIDED - Global Pointer (span-based, no BIO)**
2. **Training time**: How long with frozen encoder?
3. **Output format**: Global Pointer outputs (start, end, label) - need to update downstream

---

## Session Log

### Session 1: January 21, 2026

**Attended**: User + Copilot

**Actions**:

1. Ran encoder probe on checkpoint-18000
2. Confirmed encoder is EXCELLENT (near-zero similarity between different word types)
3. Confirmed HEAD IS THE BOTTLENECK
4. Inventoried training data: 562K samples available
5. Assessed 12 heads: 4 solid, 8 need work (NER highest priority)

**Decisions Made**:

- Use checkpoint-18000 as base (proven encoder)
- Freeze encoder, retrain heads only
- Priority: NER heads first (ner_family, ner_general, temporal)
- Solid heads: embedding, emotions, intent, ingress

**Next Steps**:

- ~~Decide on CRF vs Span-based vs Hybrid architecture for NER~~
- ~~Design new head implementation~~
- Create training script

---

### Session 1 (continued): January 21, 2026

**Discussion**: SOTA NER architecture selection

**User Requirements**:

- World-best NER that beats SOTA
- Latency not a concern (52ms current, can add more)
- Can heads run in parallel?

**Analysis Completed**:

1. Reviewed SOTA NER architectures (W2NER, Global Pointer, Biaffine, etc.)
2. Analyzed current forward() - runs ONE head per call (sequential)
3. Proposed parallel execution via torch.jit.fork

**DECISION: Global Pointer for NER**

Rationale:

- 93%+ F1 on benchmarks (SOTA tier)
- NO BIO tagging = NO invalid transitions = NO garbage
- Outputs (start, end, label) directly - cleaner than BIO
- Efficient: +5-10ms latency only
- Handles nested entities naturally

**DECISION: Parallel Head Execution**

Current architecture allows it - all heads take same encoder output.
Will implement `forward_all()` method.

**Updated Priority**:

| Head | Architecture | Status |
|------|--------------|--------|
| ner_family | GlobalPointerNERHead | TO IMPLEMENT |
| ner_general | GlobalPointerNERHead | TO IMPLEMENT |
| temporal | GlobalPointerNERHead | TO IMPLEMENT |
| safety_familyos | Keep + cultural fixes | P2 |
| relation | Add CrossAttention | P2 |
| embedding | KEEP | SOLID |
| emotions | KEEP | SOLID |
| intent | KEEP | SOLID |
| ingress | KEEP | SOLID |
| sentiment | KEEP | SOLID |
| safety_generic | KEEP | Low priority |
| nli | Consider CrossAttention | P3 |

**Next Steps**:

1. Implement GlobalPointerNERHead class
2. Implement RotaryPositionEncoding helper
3. Create training script with frozen encoder
4. Test on 10K samples first
5. Measure garbage rate before/after

---

# IMPLEMENTATION PLAN: GlobalPointer Head Training

**Project**: UltraBERT v2 Head Redesign
**Start Date**: January 2026
**Strategy**: Freeze checkpoint-18000 encoder, train new GlobalPointer heads

---

## Milestone 1: Data Pipeline

**Goal**: Prepare span-format data for GlobalPointer training

### Epic 1.1: BIO-to-Span Conversion

**Architecture Decision**: OFFLINE CONVERSION

| Approach | Decision |
|----------|----------|
| Offline (pre-convert, save to disk) | YES |
| Online (convert during training) | NO |

**Rationale**:

- CoNLL-2003: ~20K samples, WikiNeural: ~90K samples (small)
- One-time conversion cost vs repeated overhead
- Enables data validation before training
- Faster training iteration
- Easier debugging

**Output Location**: `data/ner_general_span/`

#### Issue 1.1.1: Create bio_to_spans() utility - COMPLETED

**File**: `src/modeling_studio/data/span_utils.py` (CREATED)

**Status**: DONE (Jan 21, 2026)

**Created Functions**:

- `bio_to_spans()` - Convert BIO-tagged sequence to span format
- `flat_to_spans()` - Convert flat-tagged sequence (for Few-NERD)
- `validate_spans()` - Validate span format correctness
- `spans_to_bio()` - Reverse conversion for debugging

**Function Signature**:

```python
def bio_to_spans(
    tokens: list[str],
    bio_tags: list[int],
    label_names: list[str],  # ["O", "B-PER", "I-PER", "B-ORG", ...]
) -> dict:
    """
    Convert BIO-tagged token sequence to span format.

    Args:
        tokens: Word tokens ["Emma", "lives", "in", "NYC"]
        bio_tags: BIO tag indices [1, 0, 0, 5]
        label_names: Mapping from index to label name

    Returns:
        {
            "text": "Emma lives in NYC",
            "entities": [
                {"start": 0, "end": 4, "label": "PER", "text": "Emma"},
                {"start": 14, "end": 17, "label": "LOC", "text": "NYC"}
            ]
        }
    """
```

**Edge Cases to Handle**:

- Orphan I-tags (no preceding B-tag) - skip or treat as B
- Empty token sequences
- Special tokens ([CLS], [SEP]) - skip
- Multi-word entities with I-continuation
- Punctuation attached to tokens

**Tests Required**:

- Single-token entity
- Multi-token entity
- Multiple entities in sequence
- No entities (all O)
- Orphan I-tag handling
- Edge: first/last token is entity

#### Issue 1.1.2: Convert CoNLL-2003 dataset - READY

**Script**: `scripts/convert_ner_to_spans.py` (CREATED)

**Status**: Script created, ready to run

**Command**:

```bash
python scripts/convert_ner_to_spans.py --dataset conll2003
```

**Steps**:

1. Load CoNLL-2003 from HuggingFace
2. Apply `bio_to_spans()` to each sample
3. Save as JSONL shards to `data/ner_general_span/conll2003/`
4. Generate statistics (entity counts per type)

**Output Format**:

```jsonl
{"text": "EU rejects German call to boycott British lamb.", "entities": [{"start": 0, "end": 2, "label": "ORG"}, {"start": 11, "end": 17, "label": "MISC"}, {"start": 36, "end": 43, "label": "MISC"}]}
```

**Expected Volume**:

- train: ~14K samples
- validation: ~3.5K samples
- test: ~3.5K samples

#### Issue 1.1.3: Convert WikiNeural dataset

**Command**: `python scripts/convert_ner_to_spans.py --dataset wikineural`

**Special Handling**:

- WikiNeural has 16 entity types, map to 4 (PER, ORG, LOC, MISC)
- Column is "tags" not "ner_tags" (handle in loader)
- Uses tner/wikineural config "en"

**Output Location**: `data/ner_general_span/wikineural/`

**Expected Volume**:

- train: ~92K samples
- validation: ~10K samples
- test: ~10K samples

#### Issue 1.1.4: Convert Few-NERD dataset

**Command**: `python scripts/convert_ner_to_spans.py --dataset fewnerd`

**Special Handling**:

- Few-NERD uses FLAT labels (not BIO) - need special conversion
- 66 fine-grained types → map to 4 (PER, ORG, LOC, MISC)
- Consecutive same-type tokens = one entity span

**Label Mapping**:

```python
FEWNERD_TO_CONLL = {
    "person": "PER",
    "organization": "ORG",
    "location": "LOC",
    "building": "LOC",
    "event": "MISC",
    "product": "MISC",
    "art": "MISC",
    "other": "MISC",
}
```

**Output Location**: `data/ner_general_span/fewnerd/`

**Expected Volume**:

- train: ~132K samples

#### Issue 1.1.5: Convert OntoNotes 5 dataset

**Command**: `python scripts/convert_ner_to_spans.py --dataset ontonotes`

**Special Handling**:

- 18 entity types → map to 4 (PER, ORG, LOC, MISC)
- Uses tner/ontonotes5 from HuggingFace

**Label Mapping**:

```python
ONTONOTES_TO_CONLL = {
    "PERSON": "PER",
    "ORG": "ORG", "GPE": "LOC", "LOC": "LOC", "FAC": "LOC",
    "NORP": "MISC", "EVENT": "MISC", "WORK_OF_ART": "MISC",
    "LAW": "MISC", "LANGUAGE": "MISC", "PRODUCT": "MISC",
    "DATE": "MISC", "TIME": "MISC", "PERCENT": "MISC",
    "MONEY": "MISC", "QUANTITY": "MISC", "ORDINAL": "MISC", "CARDINAL": "MISC",
}
```

**Output Location**: `data/ner_general_span/ontonotes/`

**Expected Volume**:

- train: ~75K samples

#### Issue 1.1.6: Sample WikiANN dataset

**Command**: `python scripts/convert_ner_to_spans.py --dataset wikiann --sample 100000`

**Special Handling**:

- WikiANN has 2M+ samples - sample 100K for balance
- Already has PER, ORG, LOC labels (no MISC)
- Random sampling with seed for reproducibility

**Output Location**: `data/ner_general_span/wikiann/`

**Expected Volume**:

- train: 100K samples (sampled from 2M+)

#### Issue 1.1.7: Validate all converted data - READY

**Script**: `scripts/validate_span_data.py` (CREATED)

**Status**: Script created, ready to run

**Command**:

```bash
python scripts/validate_span_data.py --data-dir data/ner_general_span
```

**Validation Checks**:

1. All spans have valid character offsets (start < end)
2. Span text matches text[start:end]
3. Labels are in expected set (PER, ORG, LOC, MISC)
4. No overlapping spans (unless nested NER desired)
5. Statistics: entity counts, avg entities per sample, label distribution

**Output**: `data/ner_general_span/validation_report.json`

**Target Volume Summary**:

| Dataset | Samples | Status |
|---------|---------|--------|
| CoNLL-2003 | 20K | Convert |
| WikiNeural | 92K | Convert |
| Few-NERD | 132K | Convert (flat→span) |
| OntoNotes 5 | 75K | Convert |
| WikiANN | 100K | Sample + Convert |
| FamilyOS synthetic | 150K | Generate (Epic 1.3) |
| **TOTAL** | **~570K** | Matches other heads! |

### Epic 1.2: GlobalPointer Collator - COMPLETED

**Status**: DONE (Jan 21, 2026)

**Deliverables**:

- `src/modeling_studio/data/globalpointer_collator.py` (CREATED)
- `tests/unit/test_globalpointer_collator.py` (CREATED - 25 tests, all passing)

**Goal**: Create a data collator that converts span-format data to GlobalPointer training tensors

**Reusable Components from Codebase**:

| Component | Location | Reuse? |
|-----------|----------|--------|
| `BaseCollator` | `trainers/collators.py` | YES - inherit for padding logic |
| `_pad_sequence()` | `trainers/collators.py` | YES - padding utility |
| `validate_spans()` | `data/span_utils.py` | YES - input validation |
| `V3CollatorConfig` pattern | `data/collators_v3.py` | PATTERN - dataclass config style |
| `IGNORE_INDEX = -100` | `trainers/collators.py` | YES - for masking |

**Output Tensor Shape**: `(B, num_labels, L, L)` - NOT `(B, L, L, num_labels)`

Rationale: GlobalPointer computes per-label span scores. Having label as second dim
allows efficient batched matrix operations per label type.

#### Issue 1.2.1: Design GlobalPointerCollator class

**File**: `src/modeling_studio/data/globalpointer_collator.py` (NEW)

**Class Signature**:

```python
@dataclass
class GlobalPointerCollator(BaseCollator):
    """
    Data collator for GlobalPointer NER training.

    Converts span-format data:
        {"text": "...", "entities": [{"start": 0, "end": 4, "label": "PER"}]}

    To batched tensors:
        input_ids: (B, L)
        attention_mask: (B, L)
        span_labels: (B, num_labels, L, L)
    """

    tokenizer: PreTrainedTokenizerBase
    label_to_id: dict[str, int]  # {"PER": 0, "ORG": 1, "LOC": 2, "MISC": 3}
    max_length: int = 512
    padding: str | bool = True

    # Computed
    num_labels: int = field(init=False)
```

**Key Design Decisions**:

1. **Inherit from BaseCollator**: Reuse `_pad_sequence()` and `pad_token_id` property
2. **Use dataclass**: Consistent with existing collators
3. **label_to_id as required arg**: Different tasks have different label sets
4. **Factory functions**: `create_ner_general_collator()`, `create_ner_family_collator()`

**Acceptance Criteria**:

- [ ] Inherits from BaseCollator
- [ ] Accepts span-format input
- [ ] Returns properly shaped tensors
- [ ] Has factory functions for each NER task

#### Issue 1.2.2: Implement char-to-token span alignment

**Method**: `_char_to_token_span()`

**Input**:

- `offset_mapping`: `list[tuple[int, int]]` from tokenizer
- `char_start`: Character start position
- `char_end`: Character end position (exclusive)

**Output**: `(tok_start, tok_end)` inclusive token indices

**Algorithm**:

```python
def _char_to_token_span(
    self,
    offset_mapping: list[tuple[int, int]],
    char_start: int,
    char_end: int,
) -> tuple[int | None, int | None]:
    """
    Convert character span to token span using offset_mapping.

    Handles:
        - Partial token overlaps (expand to containing token)
        - Special tokens with (0,0) offset (skip)
        - Spans crossing token boundaries

    Example:
        text: "New York City"
        tokens: ["New", "York", "City"]
        offset_mapping: [(0,3), (4,8), (9,13)]

        char_span (0, 8) -> "New York" -> token_span (0, 1)
        char_span (4, 13) -> "York City" -> token_span (1, 2)
    """
    tok_start = tok_end = None

    for i, (cs, ce) in enumerate(offset_mapping):
        # Skip special tokens (CLS, SEP, PAD have offset (0,0))
        if cs == 0 and ce == 0:
            continue

        # Token overlaps with character span if:
        # token_start < char_end AND token_end > char_start
        if cs < char_end and ce > char_start:
            if tok_start is None:
                tok_start = i
            tok_end = i  # Keep updating for last overlapping token

    return tok_start, tok_end
```

**Edge Cases**:

- Span exactly matches one token: `tok_start == tok_end`
- Span covers partial token: Expand to full token
- Span in special token region: Return `(None, None)` (skip)
- Span after truncation: Return `(None, None)` (skip)

**Acceptance Criteria**:

- [ ] Handles single-token spans
- [ ] Handles multi-token spans
- [ ] Skips special tokens
- [ ] Handles partial overlaps

#### Issue 1.2.3: Build span label matrix (B, num_labels, L, L)

**Method**: `__call__()` main collation logic

**Algorithm**:

```python
def __call__(self, features: list[dict[str, Any]]) -> dict[str, Tensor]:
    texts = [f["text"] for f in features]

    # 1. Tokenize with offset mapping
    encoding = self.tokenizer(
        texts,
        padding=self.padding,
        truncation=True,
        max_length=self.max_length,
        return_offsets_mapping=True,
        return_tensors=None,  # Get lists first
    )

    batch_size = len(texts)
    seq_len = len(encoding["input_ids"][0])

    # 2. Initialize span labels: (B, num_labels, L, L)
    # Upper triangular only (i <= j), but initialize full for simplicity
    span_labels = torch.zeros(batch_size, self.num_labels, seq_len, seq_len)

    # 3. Fill in entity spans
    for b, feature in enumerate(features):
        offset_mapping = encoding["offset_mapping"][b]

        for entity in feature.get("entities", []):
            char_start = entity["start"]
            char_end = entity["end"]
            label = entity.get("label", entity.get("type"))

            if label not in self.label_to_id:
                continue  # Unknown label

            label_id = self.label_to_id[label]

            # Map char span to token span
            tok_start, tok_end = self._char_to_token_span(
                offset_mapping, char_start, char_end
            )

            if tok_start is not None and tok_end is not None:
                # Set label at (tok_start, tok_end) position
                span_labels[b, label_id, tok_start, tok_end] = 1.0

    # 4. Return batch
    return {
        "input_ids": torch.tensor(encoding["input_ids"]),
        "attention_mask": torch.tensor(encoding["attention_mask"]),
        "span_labels": span_labels,
    }
```

**Memory Consideration**:

- Span labels: `(B=32, num_labels=4, L=512, L=512)` = 32 *4* 512 *512* 4 bytes = 128 MB
- May need sparse representation for very long sequences
- For training, 512 max_length is reasonable

**Acceptance Criteria**:

- [ ] Correct tensor shapes
- [ ] Labels at correct (tok_start, tok_end) positions
- [ ] Unknown labels skipped gracefully
- [ ] Truncated spans handled

#### Issue 1.2.4: Add unit tests

**File**: `tests/unit/test_globalpointer_collator.py` (NEW)

**Test Cases**:

```python
class TestGlobalPointerCollator:

    def test_single_entity(self):
        """Single entity correctly placed in span matrix."""
        sample = {
            "text": "Emma lives in NYC",
            "entities": [{"start": 0, "end": 4, "label": "PER", "text": "Emma"}]
        }
        batch = collator([sample])
        assert batch["span_labels"][0, 0, tok_emma, tok_emma] == 1.0

    def test_multi_token_entity(self):
        """Multi-token entity spans correct range."""
        sample = {
            "text": "New York is great",
            "entities": [{"start": 0, "end": 8, "label": "LOC", "text": "New York"}]
        }
        batch = collator([sample])
        assert batch["span_labels"][0, 2, tok_new, tok_york] == 1.0

    def test_multiple_entities(self):
        """Multiple entities in same sample."""
        sample = {
            "text": "Emma met John in NYC",
            "entities": [
                {"start": 0, "end": 4, "label": "PER"},
                {"start": 9, "end": 13, "label": "PER"},
                {"start": 17, "end": 20, "label": "LOC"},
            ]
        }
        batch = collator([sample])
        # Check all three entities are set

    def test_no_entities(self):
        """Sample with no entities has all-zero span labels."""
        sample = {"text": "Hello world", "entities": []}
        batch = collator([sample])
        assert batch["span_labels"].sum() == 0

    def test_unknown_label_skipped(self):
        """Unknown label is ignored, not error."""
        sample = {
            "text": "Test",
            "entities": [{"start": 0, "end": 4, "label": "UNKNOWN"}]
        }
        batch = collator([sample])
        assert batch["span_labels"].sum() == 0

    def test_truncated_span(self):
        """Entity beyond max_length is skipped."""
        # Create long text where entity is past truncation point

    def test_batch_padding(self):
        """Batch with different lengths pads correctly."""

    def test_char_to_token_alignment(self):
        """Character offsets align to correct tokens."""
```

**Acceptance Criteria**:

- [ ] All test cases pass
- [ ] Edge cases covered
- [ ] No regressions in existing collators

### Epic 1.3: ner_general Data Generation - IMPLEMENTED

**Status**: DONE (Jan 21, 2026)

**Deliverable**: `scripts/agents/ner_general_span_generator.py`

**Features**:

- Google Vertex AI (Gemini 2.5 Flash) with explicit caching (90% discount)
- Span-format output: `{"text": "...", "entities": [{"start", "end", "label", "token"}]}`
- Labels: PER, ORG, LOC, MISC (CoNLL-2003 compatible)
- Shard-based storage with deduplication
- Progress persistence for resume
- Validation with exact offset checking

**Usage**:

```bash
# Generate 50K samples
python scripts/agents/ner_general_span_generator.py generate --samples 50000

# Check stats
python scripts/agents/ner_general_span_generator.py stats

# Validate all samples
python scripts/agents/ner_general_span_generator.py validate
```

**Output Location**: `data/ner_general_span/familyos_synthetic/`

**Issues Completed**:

- [x] Issue 1.3.1: Adapted for span output (not BIO)
- [x] Issue 1.3.2: SYSTEM_PROMPT for PER/ORG/LOC/MISC with FamilyOS context
- [x] Issue 1.3.3: Generator script with Vertex AI
- [x] Issue 1.3.4: Validation and deduplication built-in

---

## Milestone 2: Model Architecture

**Goal**: Implement SOTA GlobalPointer NER head

### Epic 2.1: GlobalPointerNERHead Implementation - COMPLETED

**Status**: DONE (Jan 21, 2026)

**Deliverables**:

- `src/modeling_studio/models/heads.py`: Added GlobalPointerNERHead class (~470 lines)
- `tests/unit/test_globalpointer_head.py`: 33 unit tests, all passing

**File Location Decision**: ADD TO `src/modeling_studio/models/heads.py`

**Rationale**:

- `heads.py` already contains all head implementations (TokenClassificationHead, TemporalHead, etc.)
- NO new files - add GlobalPointerNERHead class after TemporalHead (line ~1969)
- Import RotaryEmbedding from `attention.py` (already exists)
- Reuse BaseHead pattern for consistency

**Deprecation Plan**:

| Current | Action | Timeline |
|---------|--------|----------|
| `TokenClassificationHead` for NER | KEEP for backward compat | Indefinite |
| `HubAwareTokenClassificationHead` | KEEP for v3 | Indefinite |
| `TemporalHead` (extends TokenClassificationHead) | REPLACE with GlobalPointer internally | After validation |

**Reusable Components**:

| Component | Location | Reuse |
|-----------|----------|-------|
| `RotaryEmbedding` | `models/attention.py:52` | YES - import for RoPE |
| `apply_rotary_pos_emb` | `models/attention.py:168` | YES - apply rotation |
| `BaseHead` | `models/heads.py:48` | NO - GlobalPointer has different loss signature |
| `nn.Module` | PyTorch | YES - inherit directly |
| `FocalLoss` | `models/losses.py:54` | PATTERN - for span loss |

**Output Tensor Convention**:

- Match collator output: `(B, num_labels, L, L)`
- Label dimension second for efficient per-label operations
- Upper triangular only (i <= j constraint)

#### Issue 2.1.1: Create GlobalPointerNERHead class skeleton

**File**: `src/modeling_studio/models/heads.py` (ADD after line ~1969, before HierarchicalEmotionHead)

**Class Signature**:

```python
class GlobalPointerNERHead(nn.Module):
    """
    Global Pointer head for span-based NER.

    Instead of BIO tagging, directly predicts span (start, end, label) tuples.
    Uses RoPE-style relative position encoding to enforce i <= j constraint.

    Architecture:
        hidden_states → Linear(q) → RoPE
        hidden_states → Linear(k) → RoPE
        scores = q @ k.T → upper-triangular mask → (B, num_labels, L, L)

    Args:
        hidden_size: Encoder hidden dimension (768 for ModernBERT)
        num_labels: Number of entity types (4 for ner_general, 10 for ner_family)
        head_size: Dimension per label head (default: 64)
        dropout: Dropout probability (default: 0.1)
        use_rope: Whether to use Rotary Position Encoding (default: True)

    Reference:
        "Global Pointer: Novel Efficient Span-based Approach for NER"
        https://arxiv.org/abs/2208.03054
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_labels: int = 4,
        head_size: int = 64,
        dropout: float = 0.1,
        use_rope: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.head_size = head_size
        self.use_rope = use_rope

        # Per-label Q/K projections
        self.q_proj = nn.Linear(hidden_size, num_labels * head_size * 2)
        self.k_proj = nn.Linear(hidden_size, num_labels * head_size * 2)

        self.dropout = nn.Dropout(dropout)

        # RoPE for relative position encoding (import from attention.py)
        if use_rope:
            from modeling_studio.models.attention import RotaryEmbedding
            self.rope = RotaryEmbedding(dim=head_size, max_seq_len=512)
        else:
            self.rope = None

        self._init_weights()
```

**Methods to Implement**:

- `__init__()` - constructor with Q/K projections and RoPE
- `_init_weights()` - Xavier initialization
- `forward()` - compute span scores (B, num_labels, L, L)
- `compute_loss()` - multi-label BCE or circle loss
- `decode()` - extract (start, end, label) tuples from scores
- `extra_repr()` - for print(model)

**Acceptance Criteria**:

- [ ] Class inherits from nn.Module (not BaseHead - different loss signature)
- [ ] Q/K projections output `num_labels * head_size * 2` (for RoPE split)
- [ ] RoPE imported from existing attention.py
- [ ] Docstring with paper reference

#### Issue 2.1.2: Implement RoPE (Rotary Position Encoding)

**Action**: REUSE existing implementation

**Existing Code** (`src/modeling_studio/models/attention.py`):

- `RotaryEmbedding` class (line 52-142)
- `apply_rotary_pos_emb` function (line 168-186)
- `rotate_half` helper (line 155-166)

**Integration**:

```python
# In GlobalPointerNERHead.__init__:
from modeling_studio.models.attention import RotaryEmbedding, apply_rotary_pos_emb

if use_rope:
    self.rope = RotaryEmbedding(dim=head_size, max_seq_len=512)

# In forward():
if self.rope is not None:
    cos, sin = self.rope(q, seq_len=seq_len)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
```

**NO new RoPE implementation needed** - reuse attention.py

**Acceptance Criteria**:

- [ ] Import RotaryEmbedding from attention.py
- [ ] Import apply_rotary_pos_emb from attention.py
- [ ] Works with head_size=64 dimension

#### Issue 2.1.3: Implement Q/K projections per label type

**Design**: Single linear layer, reshape to per-label heads

```python
def forward(self, hidden_states, attention_mask=None):
    batch, seq_len, _ = hidden_states.shape

    # Project to Q and K
    # Output: (B, L, num_labels * head_size * 2)
    q = self.q_proj(self.dropout(hidden_states))
    k = self.k_proj(self.dropout(hidden_states))

    # Reshape: (B, L, num_labels, head_size, 2) -> split for RoPE
    q = q.view(batch, seq_len, self.num_labels, self.head_size, 2)
    k = k.view(batch, seq_len, self.num_labels, self.head_size, 2)

    # Rearrange for attention: (B, num_labels, L, head_size, 2)
    q = q.permute(0, 2, 1, 3, 4)
    k = k.permute(0, 2, 1, 3, 4)

    # Split for RoPE (real/imag or cos/sin parts)
    q_cos, q_sin = q[..., 0], q[..., 1]
    k_cos, k_sin = k[..., 0], k[..., 1]
```

**Memory Consideration**:

- Q/K projection: `768 → num_labels * 64 * 2 = 512` for 4 labels
- Total params: 768 *512* 2 = 786K per head (acceptable)

**Acceptance Criteria**:

- [ ] Single Q projection, single K projection
- [ ] Reshape to (B, num_labels, L, head_size, 2)
- [ ] Dropout applied before projection

#### Issue 2.1.4: Implement span score computation (einsum)

**Core Computation**:

```python
def forward(self, hidden_states, attention_mask=None, span_labels=None):
    # ... Q/K projections and RoPE ...

    # Span scores: q_i dot k_j for all i, j
    # Shape: (B, num_labels, L, L)
    scores = torch.einsum("bnlh,bnmh->bnlm", q, k)

    # Scale by sqrt(head_size)
    scores = scores / (self.head_size ** 0.5)

    # Apply attention mask (mask padded positions)
    if attention_mask is not None:
        # Expand mask: (B, 1, 1, L) and (B, 1, L, 1)
        mask_i = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
        mask_j = attention_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, L, 1)
        pad_mask = mask_i * mask_j  # (B, 1, L, L)
        scores = scores.masked_fill(pad_mask == 0, -1e9)

    # Apply upper-triangular mask (i <= j constraint)
    # See Issue 2.1.5

    output = {"logits": scores}

    if span_labels is not None:
        loss = self.compute_loss(scores, span_labels, attention_mask)
        output["loss"] = loss

    return output
```

**Einsum Explanation**:

- `bnlh`: (Batch, Num_labels, seq_Len, Head_size) for query
- `bnmh`: (Batch, Num_labels, seq_len_M, Head_size) for key
- `bnlm`: (Batch, Num_labels, L, M) - span score matrix

**Acceptance Criteria**:

- [ ] Einsum produces (B, num_labels, L, L) scores
- [ ] Scaled by sqrt(head_size)
- [ ] Padding positions masked to -inf
- [ ] Returns dict with "logits" key (and "loss" if labels provided)

#### Issue 2.1.5: Implement upper-triangular masking

**Purpose**: Enforce start <= end constraint (valid spans only)

```python
def _get_triu_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Get upper-triangular mask for i <= j constraint.

    Returns:
        Boolean mask of shape (1, 1, L, L) where True = valid position
    """
    # Create lower-triangular mask and invert
    # triu includes diagonal (start == end for single-token entities)
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)

# In forward():
triu_mask = self._get_triu_mask(seq_len, scores.device)
scores = scores.masked_fill(~triu_mask, -1e9)
```

**Why Upper-Triangular?**:

- Matrix position [i, j] represents span from token i to token j
- Only valid when i <= j (start before or equal to end)
- Diagonal = single-token entities (i == j)
- Above diagonal = multi-token entities
- Below diagonal = invalid (would mean end before start)

**Acceptance Criteria**:

- [ ] Mask is upper-triangular (includes diagonal)
- [ ] Invalid positions set to -inf before sigmoid
- [ ] Works with batched computation

#### Issue 2.1.6: Add to heads.py exports and registry

**Updates Required**:

1. **Add to `__all__`** in `heads.py`:

```python
__all__ = [
    "BaseHead",
    "SequenceClassificationHead",
    "TokenClassificationHead",
    "GlobalPointerNERHead",  # NEW
    "EmbeddingHead",
    # ... rest
]
```

1. **Update `CAPABILITY_TO_HEAD_TYPE`** in `modernbert_multitask.py`:

```python
# For training mode - user configurable
CAPABILITY_TO_HEAD_TYPE: dict[Capability, type[nn.Module]] = {
    Capability.NER_GENERAL: TokenClassificationHead,  # DEFAULT (backward compat)
    Capability.NER_FAMILY: TokenClassificationHead,   # DEFAULT (backward compat)
    # New training uses GlobalPointerNERHead via config flag
}
```

1. **Add factory function** in `heads.py`:

```python
def create_globalpointer_head(
    capability: str,
    hidden_size: int = 768,
    head_size: int = 64,
    **kwargs,
) -> GlobalPointerNERHead:
    """
    Factory to create GlobalPointerNERHead for a capability.

    Args:
        capability: "ner_general", "ner_family", or "temporal"
        hidden_size: Encoder hidden size
        head_size: Per-label head dimension

    Returns:
        Configured GlobalPointerNERHead
    """
    from modeling_studio.data.globalpointer_collator import (
        NER_GENERAL_LABELS,
        NER_FAMILY_LABELS,
        TEMPORAL_LABELS,
    )

    label_configs = {
        "ner_general": NER_GENERAL_LABELS,
        "ner_family": NER_FAMILY_LABELS,
        "temporal": TEMPORAL_LABELS,
    }

    labels = label_configs.get(capability)
    if labels is None:
        raise ValueError(f"Unknown capability: {capability}")

    return GlobalPointerNERHead(
        hidden_size=hidden_size,
        num_labels=len(labels),
        head_size=head_size,
        **kwargs,
    )
```

**Acceptance Criteria**:

- [ ] GlobalPointerNERHead in **all**
- [ ] Factory function uses label configs from collator
- [ ] Backward compatible (old TokenClassificationHead still works)

#### Issue 2.1.7: Add unit tests

**File**: `tests/unit/test_globalpointer_head.py` (NEW)

**Test Cases**:

```python
class TestGlobalPointerNERHead:
    def test_init_default_params(self):
        """Head initializes with default params."""

    def test_init_custom_labels(self):
        """Head respects num_labels parameter."""

    def test_forward_shape(self):
        """Output shape is (B, num_labels, L, L)."""

    def test_forward_with_mask(self):
        """Padding positions are masked."""

    def test_upper_triangular_constraint(self):
        """Only upper triangle has valid scores."""

    def test_compute_loss_shape(self):
        """Loss is scalar tensor."""

    def test_decode_basic(self):
        """Decode extracts correct spans."""

    def test_decode_threshold(self):
        """Decode respects threshold parameter."""

    def test_rope_applied(self):
        """RoPE modifies Q/K correctly."""

    def test_factory_ner_general(self):
        """Factory creates correct head for ner_general."""

    def test_factory_ner_family(self):
        """Factory creates correct head for ner_family."""

    def test_backward_pass(self):
        """Gradients flow correctly."""
```

**Acceptance Criteria**:

- [ ] All tests pass
- [ ] Coverage for all public methods
- [ ] Edge cases covered (empty input, single token, max length)

### Epic 2.2: Loss Function

**Status**: COMPLETED (65 tests pass)
**Goal**: Implement SOTA GlobalPointer loss function (Multi-Label Categorical Cross-Entropy)

#### Code Discovery Findings

**Reference Implementation** (from `xhw205/Efficient-GlobalPointer-torch`):

```python
def multilabel_categorical_crossentropy(y_pred, y_true):
    """
    Multi-label categorical cross-entropy for GlobalPointer.

    Key insight: Treats span detection as multi-label classification where:
    - Each position (i,j) can have multiple entity types
    - Uses logsumexp trick for stable computation
    - Handles class imbalance naturally (most positions are negative)

    Args:
        y_pred: (batch_size * num_labels, seq_len * seq_len) - logits
        y_true: (batch_size * num_labels, seq_len * seq_len) - binary labels
    """
    y_pred = (1 - 2 * y_true) * y_pred  # Flip sign for positive classes
    y_pred_neg = y_pred - y_true * 1e12  # Mask positive class predictions
    y_pred_pos = y_pred - (1 - y_true) * 1e12  # Mask negative class predictions
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()
```

**Current Implementation** (in GlobalPointerNERHead.compute_loss):

- Simple BCE with logits: `F.binary_cross_entropy_with_logits(scores_flat, labels_flat, reduction="mean")`
- Masks out padding and lower-triangular positions correctly
- BUT: Does not handle extreme class imbalance (99%+ negative positions)

**Why Multi-Label Categorical CE is Better**:

1. **Circle Loss Principle**: Separates positive and negative samples in logit space
2. **LogSumExp Stability**: Numerically stable gradient computation
3. **Natural Imbalance Handling**: No need for explicit pos_weight tuning
4. **SOTA Results**: Used in original GlobalPointer paper achieving 93%+ F1

#### Issue 2.2.1: Implement GlobalPointerLoss class

**File**: `src/modeling_studio/models/losses.py`

**Implementation**:

```python
class GlobalPointerLoss(nn.Module):
    """
    Multi-Label Categorical Cross-Entropy Loss for GlobalPointer.

    Based on the original GlobalPointer paper (Su et al., 2022).
    Uses logsumexp trick for stable computation of circle-loss style
    separation between positive and negative span predictions.

    Args:
        reduction: 'mean', 'sum', or 'none'
        mask_diagonal: Whether to mask diagonal (single-token spans)
    """

    def __init__(
        self,
        reduction: str = "mean",
        mask_diagonal: bool = False
    ):
        super().__init__()
        self.reduction = reduction
        self.mask_diagonal = mask_diagonal

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute GlobalPointer loss.

        Args:
            y_pred: Span logits (B, num_labels, L, L)
            y_true: Binary span labels (B, num_labels, L, L)
            attention_mask: Padding mask (B, L)

        Returns:
            Scalar loss tensor
        """
        batch_size, num_labels, seq_len, _ = y_pred.shape

        # Create valid position mask (upper triangular, non-padding)
        triu_mask = torch.triu(torch.ones(seq_len, seq_len, device=y_pred.device), diagonal=0 if not self.mask_diagonal else 1)

        if attention_mask is not None:
            mask_i = attention_mask.unsqueeze(1).unsqueeze(-1)
            mask_j = attention_mask.unsqueeze(1).unsqueeze(2)
            pad_mask = (mask_i * mask_j).bool()
            valid_mask = triu_mask.bool() & pad_mask
        else:
            valid_mask = triu_mask.bool().expand(batch_size, 1, seq_len, seq_len)

        valid_mask = valid_mask.expand(batch_size, num_labels, seq_len, seq_len)

        # Reshape for multi-label categorical CE
        y_pred_flat = y_pred[valid_mask].view(batch_size * num_labels, -1)
        y_true_flat = y_true[valid_mask].view(batch_size * num_labels, -1).float()

        return self.multilabel_categorical_crossentropy(y_pred_flat, y_true_flat)

    def multilabel_categorical_crossentropy(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Core loss computation using logsumexp trick.

        Intuition:
        - Positive samples should have HIGH scores
        - Negative samples should have LOW scores
        - This loss pushes them apart in logit space
        """
        # Flip sign: positive classes get negative pred, negative get positive
        y_pred = (1 - 2 * y_true) * y_pred

        # Mask out opposite class predictions with -inf
        y_pred_neg = y_pred - y_true * 1e12  # Keep negative, mask positive
        y_pred_pos = y_pred - (1 - y_true) * 1e12  # Keep positive, mask negative

        # Add zero option for stability when no positives/negatives
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)

        # LogSumExp for soft-max-like aggregation
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

        loss = neg_loss + pos_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
```

**Acceptance Criteria**:

- [ ] Class added to losses.py
- [ ] Added to `__all__` exports
- [ ] Type hints and docstrings complete
- [ ] Handles batched input correctly
- [ ] Masks padding positions
- [ ] Masks lower-triangular (invalid spans)

#### Issue 2.2.2: Add loss_type parameter to GlobalPointerNERHead

**File**: `src/modeling_studio/models/heads.py`

**Changes**:

1. Add `loss_type` parameter to `__init__`: `"globalpointer"` (default) or `"bce"`
2. Instantiate appropriate loss function
3. Update `compute_loss()` to use selected loss

**Implementation**:

```python
class GlobalPointerNERHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        inner_dim: int = 64,
        use_rope: bool = True,
        dropout: float = 0.1,
        loss_type: str = "globalpointer",  # NEW
        **kwargs,
    ):
        # ... existing init ...

        # Loss function selection
        self.loss_type = loss_type
        if loss_type == "globalpointer":
            self.loss_fn = GlobalPointerLoss(reduction="mean")
        else:
            self.loss_fn = None  # Use BCE in compute_loss

    def compute_loss(self, scores, span_labels, attention_mask=None):
        if self.loss_type == "globalpointer":
            return self.loss_fn(scores, span_labels, attention_mask)
        else:
            # Existing BCE implementation
            ...
```

**Acceptance Criteria**:

- [ ] `loss_type` parameter added
- [ ] Default is `"globalpointer"` (SOTA)
- [ ] Backward compatible with `"bce"` option
- [ ] Tests pass for both loss types

#### Issue 2.2.3: Implement FocalGlobalPointerLoss variant

**File**: `src/modeling_studio/models/losses.py`

**Purpose**: For extreme imbalance cases, add focal loss weighting

**Implementation**:

```python
class FocalGlobalPointerLoss(GlobalPointerLoss):
    """
    GlobalPointer loss with focal loss weighting.

    Adds (1-p)^gamma weighting to down-weight easy negatives.
    Useful when positive spans are extremely rare.

    Args:
        gamma: Focal loss focusing parameter (default=2.0)
        alpha: Class balance weight (default=0.25)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
    ):
        super().__init__(reduction=reduction)
        self.gamma = gamma
        self.alpha = alpha

    def multilabel_categorical_crossentropy(self, y_pred, y_true):
        # Standard GlobalPointer loss
        base_loss = super().multilabel_categorical_crossentropy(y_pred, y_true)

        # Apply focal weighting
        probs = torch.sigmoid(y_pred)
        focal_weight = (1 - probs) ** self.gamma

        return (focal_weight * base_loss).mean()
```

**Acceptance Criteria**:

- [ ] Extends GlobalPointerLoss
- [ ] Gamma parameter configurable
- [ ] Tests show improvement on imbalanced data

#### Issue 2.2.4: Write unit tests for loss functions

**File**: `tests/unit/test_globalpointer_loss.py`

**Test Cases**:

```python
class TestGlobalPointerLoss:
    def test_output_shape_scalar(self):
        """Loss returns scalar tensor."""

    def test_zero_loss_perfect_prediction(self):
        """Loss is minimal when predictions match labels exactly."""

    def test_high_loss_inverted_prediction(self):
        """Loss is high when predictions are inverted."""

    def test_masking_padding(self):
        """Padding positions do not contribute to loss."""

    def test_masking_lower_triangular(self):
        """Lower triangular positions (invalid spans) are masked."""

    def test_gradient_flow(self):
        """Gradients flow through loss computation."""

    def test_batch_consistency(self):
        """Same loss for single sample vs batch of same sample."""

    def test_numerical_stability(self):
        """No NaN/Inf with extreme logit values."""

    def test_reduction_modes(self):
        """Mean, sum, none reductions work correctly."""

class TestFocalGlobalPointerLoss:
    def test_gamma_zero_equals_base(self):
        """Gamma=0 reduces to standard GlobalPointerLoss."""

    def test_higher_gamma_lower_easy_weight(self):
        """Higher gamma down-weights easy examples more."""
```

**Acceptance Criteria**:

- [ ] 10+ test cases
- [ ] All tests pass
- [ ] Edge cases covered (empty input, all positive, all negative)

#### Issue 2.2.5: Update GlobalPointerNERHead.compute_loss to use GlobalPointerLoss

**File**: `src/modeling_studio/models/heads.py`

**Current** (line 2190-2250):

```python
def compute_loss(self, scores, span_labels, attention_mask=None):
    # ... masking logic ...
    loss = F.binary_cross_entropy_with_logits(scores_flat, labels_flat, reduction="mean")
    return loss
```

**New**:

```python
def compute_loss(self, scores, span_labels, attention_mask=None):
    return self.loss_fn(scores, span_labels, attention_mask)
```

**Acceptance Criteria**:

- [ ] compute_loss delegates to loss_fn
- [ ] Masking logic moved to GlobalPointerLoss
- [ ] All existing tests still pass
- [x] New loss tests pass

### Epic 2.3: Decoding Logic - COMPLETED

**Status**: DONE (Jan 21, 2026)

**Goal**: Production-ready span decoding with overlap handling

**Implementation Summary**:

| Method | Location | Status | Notes |
| ------ | -------- | ------ | ----- |
| `decode()` | heads.py:2261-2310 | EXISTS | Loop-based, threshold, returns score |
| `decode_batch_efficient()` | heads.py:2312-2375 | EXISTS | Vectorized, torch.where |
| `_spans_overlap()` | heads.py:2380 | DONE | Inclusive-end overlap detection |
| `_calculate_iou()` | heads.py:2387 | DONE | Intersection over Union |
| `nms_spans()` | heads.py:2398 | DONE | Greedy NMS with IoU threshold |
| `_token_to_char_span()` | heads.py:2445 | DONE | Token to character mapping |
| `decode_with_nms()` | heads.py:2462 | DONE | Full production pipeline |

**Test Coverage**: 24 new tests added (64 total for GlobalPointerNERHead)

**Reference Implementation Analysis** (Efficient-GlobalPointer-torch):

- Uses simple threshold: `scores > 0` (logit space)
- No explicit NMS - relies on GlobalPointer's training to avoid overlaps
- Decoding: `np.where(scores > 0)` then map back to char spans

**Key Insight**: GlobalPointer naturally handles overlaps during training (multi-label
categorical CE loss encourages mutually exclusive spans). NMS is optional but useful for:

1. Inference-time threshold tuning
2. Handling edge cases where model is uncertain
3. Production robustness

#### Issue 2.3.1: Implement threshold-based span extraction - COMPLETED (EXISTS)

**Status**: ALREADY EXISTS

**Location**: `src/modeling_studio/models/heads.py:2261-2310`

**What Exists**:

```python
def decode(
    self,
    scores: torch.Tensor,         # (B, num_labels, L, L)
    attention_mask: torch.Tensor | None = None,
    threshold: float = 0.0,       # Logit threshold (0.0 = prob > 0.5)
    id2label: dict[int, str] | None = None,
) -> list[list[dict]]:
    """Decode span scores to entity predictions."""
    # Returns: [{"start": int, "end": int, "label": str, "score": float}]
```

**What's Missing**:

- [ ] Token-to-char span mapping (for final output)
- [ ] Integration with tokenizer offset_mapping

**Decision**: Keep existing, add token-to-char mapping in Issue 2.3.4

#### Issue 2.3.2: Implement NMS for overlapping spans - TO IMPLEMENT

**Status**: NOT STARTED

**Priority**: P1 (nice-to-have for production robustness)

**Design Decisions**:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| NMS strategy | Greedy by score | Simple, effective, O(n log n) |
| Overlap definition | Any character overlap | Most conservative |
| Cross-type NMS | Optional (default off) | Same text can be PER and LOC |
| Implementation | Standalone function | Reusable, testable |

**Algorithm: Greedy NMS**:

```python
def nms_spans(
    entities: list[dict],
    iou_threshold: float = 0.0,  # 0.0 = any overlap
    cross_type: bool = False,     # NMS across different label types
) -> list[dict]:
    """
    Non-maximum suppression for overlapping spans.

    Args:
        entities: List of {"start", "end", "label", "score"}
        iou_threshold: IoU threshold (0.0 = suppress any overlap)
        cross_type: If True, suppress across label types

    Returns:
        Filtered list with overlapping lower-score spans removed
    """
    if not entities:
        return []

    # Sort by score descending
    sorted_entities = sorted(entities, key=lambda x: x["score"], reverse=True)

    kept = []
    for entity in sorted_entities:
        # Check overlap with already-kept entities
        overlaps = False
        for kept_entity in kept:
            # Skip cross-type check if not enabled
            if not cross_type and entity["label"] != kept_entity["label"]:
                continue

            # Check character overlap
            if _spans_overlap(entity, kept_entity):
                # Calculate IoU if threshold > 0
                if iou_threshold > 0:
                    iou = _calculate_iou(entity, kept_entity)
                    if iou >= iou_threshold:
                        overlaps = True
                        break
                else:
                    overlaps = True
                    break

        if not overlaps:
            kept.append(entity)

    return kept


def _spans_overlap(a: dict, b: dict) -> bool:
    """Check if two spans have any character overlap."""
    return a["start"] < b["end"] and b["start"] < a["end"]


def _calculate_iou(a: dict, b: dict) -> float:
    """Calculate Intersection over Union for two spans."""
    intersection_start = max(a["start"], b["start"])
    intersection_end = min(a["end"], b["end"])
    intersection = max(0, intersection_end - intersection_start)

    union = (a["end"] - a["start"]) + (b["end"] - b["start"]) - intersection

    return intersection / union if union > 0 else 0.0
```

**File**: `src/modeling_studio/models/heads.py` (add after decode methods)

**Acceptance Criteria**:

- [ ] `nms_spans()` function implemented
- [ ] `_spans_overlap()` helper implemented
- [ ] `_calculate_iou()` helper implemented
- [ ] Same-type NMS by default
- [ ] Optional cross-type NMS
- [ ] IoU threshold configurable
- [ ] Unit tests for all edge cases

**Test Cases**:

1. No overlaps - all entities kept
2. Two overlaps, same type - higher score wins
3. Two overlaps, different types - both kept (default)
4. Cross-type NMS enabled - higher score wins
5. Three-way overlap - greedy selection
6. IoU threshold = 0.5 - partial overlaps allowed
7. Empty input - empty output
8. Single entity - kept

#### Issue 2.3.3: Add confidence scores to output - TO IMPLEMENT

**Status**: NOT STARTED

**Current State**: `decode()` returns raw logit as `score` field

**Problem**: Raw logits are not calibrated probabilities

**Proposed Enhancement**:

```python
def decode(
    self,
    scores: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    threshold: float = 0.0,
    id2label: dict[int, str] | None = None,
    return_probabilities: bool = False,  # NEW
) -> list[list[dict]]:
    """
    ...
    Args:
        return_probabilities: If True, return sigmoid(score) as confidence

    Returns:
        List of entities. If return_probabilities=True:
            {"start", "end", "label", "score", "confidence"}
        Else:
            {"start", "end", "label", "score"}
    """
    ...
    if return_probabilities:
        entity["confidence"] = torch.sigmoid(torch.tensor(score)).item()
```

**Additional Enhancement**: Temperature scaling for calibration

```python
def decode(
    ...
    temperature: float = 1.0,  # NEW - for calibrated probabilities
) -> list[list[dict]]:
    """
    Args:
        temperature: Temperature for softmax calibration (1.0 = no change)
                     < 1.0 = sharper, > 1.0 = smoother
    """
    if return_probabilities:
        calibrated_score = score / temperature
        entity["confidence"] = torch.sigmoid(torch.tensor(calibrated_score)).item()
```

**File**: `src/modeling_studio/models/heads.py` (modify decode methods)

**Acceptance Criteria**:

- [ ] `return_probabilities` parameter added to decode()
- [ ] `temperature` parameter added for calibration
- [ ] `confidence` field added to output when enabled
- [ ] Backward compatible (default behavior unchanged)
- [ ] Unit tests for probability conversion

#### Issue 2.3.4: Add token-to-char span mapping - TO IMPLEMENT

**Status**: NOT STARTED

**Problem**: Current decode() returns token indices, not character offsets

**Solution**: Add offset_mapping parameter to convert back

```python
def decode(
    self,
    scores: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    threshold: float = 0.0,
    id2label: dict[int, str] | None = None,
    offset_mapping: torch.Tensor | list | None = None,  # NEW
    return_probabilities: bool = False,
) -> list[list[dict]]:
    """
    Args:
        offset_mapping: Token-to-char mapping from tokenizer (B, L, 2)
                        If provided, returns char spans instead of token spans

    Returns:
        If offset_mapping provided:
            {"char_start", "char_end", "token_start", "token_end", "label", "score"}
        Else:
            {"start", "end", "label", "score"}  # Token indices
    """
```

**Implementation**:

```python
def _token_to_char_span(
    self,
    offset_mapping: list[tuple[int, int]],
    tok_start: int,
    tok_end: int,
) -> tuple[int, int]:
    """Convert token span to character span."""
    char_start = offset_mapping[tok_start][0]
    char_end = offset_mapping[tok_end][1]
    return char_start, char_end
```

**File**: `src/modeling_studio/models/heads.py` (modify decode methods)

**Acceptance Criteria**:

- [ ] `offset_mapping` parameter added to decode()
- [ ] `_token_to_char_span()` helper implemented
- [ ] Output includes both token and char spans when offset_mapping provided
- [ ] Backward compatible (token indices when no offset_mapping)
- [ ] Unit tests for char span conversion

#### Issue 2.3.5: Create decode_with_nms() convenience method - TO IMPLEMENT

**Status**: NOT STARTED

**Purpose**: All-in-one method for production inference

```python
def decode_with_nms(
    self,
    scores: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    offset_mapping: torch.Tensor | list | None = None,
    threshold: float = 0.0,
    id2label: dict[int, str] | None = None,
    nms_threshold: float = 0.0,  # IoU threshold for NMS
    cross_type_nms: bool = False,
    return_probabilities: bool = True,
    temperature: float = 1.0,
) -> list[list[dict]]:
    """
    Full decoding pipeline: threshold -> NMS -> char mapping.

    Returns:
        List of entities per batch:
        {
            "char_start": int,
            "char_end": int,
            "token_start": int,
            "token_end": int,
            "label": str,
            "score": float,       # Raw logit
            "confidence": float,  # Calibrated probability
        }
    """
    # 1. Basic threshold decode
    entities = self.decode_batch_efficient(
        scores, attention_mask, threshold, id2label
    )

    # 2. Apply NMS per batch item
    entities = [nms_spans(e, nms_threshold, cross_type_nms) for e in entities]

    # 3. Add char spans and confidence
    for b, batch_entities in enumerate(entities):
        for entity in batch_entities:
            if offset_mapping is not None:
                char_start, char_end = self._token_to_char_span(
                    offset_mapping[b], entity["start"], entity["end"]
                )
                entity["char_start"] = char_start
                entity["char_end"] = char_end
                entity["token_start"] = entity.pop("start")
                entity["token_end"] = entity.pop("end")

            if return_probabilities:
                calibrated = entity["score"] / temperature
                entity["confidence"] = torch.sigmoid(torch.tensor(calibrated)).item()

    return entities
```

**File**: `src/modeling_studio/models/heads.py`

**Acceptance Criteria**:

- [ ] `decode_with_nms()` method implemented
- [ ] Combines all decoding features
- [ ] Returns full entity info (char + token spans, confidence)
- [ ] Easy to use for production inference
- [ ] Unit tests for full pipeline

---

## Milestone 3: Training Infrastructure

**Goal**: Set up frozen-encoder training pipeline with frozen encoder from checkpoint-18000

### Epic 3.1: Training Script - COMPLETED

**Status**: DONE (Jan 21, 2026)

**Deliverable**: `scripts/training/train_globalpointer_heads.py`

**Purpose**: Train ONLY the GlobalPointer heads with frozen encoder. The encoder from checkpoint-18000
has proven excellent performance - we keep it frozen and only replace/train the NER heads.

**Training Strategy**:

| Component | Action | Learning Rate |
|-----------|--------|---------------|
| ModernBERT Encoder | FROZEN | 0.0 |
| GlobalPointerNERHead (ner_general) | TRAIN | 1e-4 |
| GlobalPointerNERHead (ner_family) | TRAIN | 1e-4 |
| GlobalPointerNERHead (temporal) | TRAIN | 1e-4 |

**Architecture**:

```
checkpoint-18000 (ModernBERT encoder)
        |
        v [FROZEN - no gradients]
   hidden_states
        |
        +---> GlobalPointerNERHead (ner_general, 4 labels: PER/ORG/LOC/MISC)
        |
        +---> GlobalPointerNERHead (ner_family, 10 labels: KINSHIP/MILESTONE/etc)
        |
        +---> GlobalPointerNERHead (temporal, 6 labels: DATE_ABS/DATE_REL/etc)
```

**Data Sources**:

| Task | Source | Format | Samples |
|------|--------|--------|---------|
| ner_general | CoNLL-2003 (converted) + FamilyOS synthetic | Span | ~365K + 100K |
| ner_family | data/curated/unified/ | Span | ~50K |
| temporal | data/curated/unified/ | Span | ~30K |

**Dependencies**:

- [x] Epic 1.1: BIO-to-Span converter (DONE)
- [x] Epic 1.2: GlobalPointerCollator (DONE)
- [x] Epic 2.1: GlobalPointerNERHead (DONE)
- [x] Epic 2.2: GlobalPointerLoss (DONE)
- [x] Epic 2.3: Decoding with NMS (DONE)
- [~] Epic 1.3: ner_general data generation (RUNNING - 100K target)

#### Issue 3.1.1: Create train_globalpointer_heads.py - TO IMPLEMENT

**File**: `scripts/training/train_globalpointer_heads.py`

**Structure**:

```python
#!/usr/bin/env python
"""
GlobalPointer Head Training Script

Train span-based NER heads with frozen ModernBERT encoder.
The encoder from checkpoint-18000 is proven excellent - we only train heads.

Usage:
    # Train all heads
    python scripts/training/train_globalpointer_heads.py \
        --checkpoint outputs/modernbert-v2-for-v3-transfer/checkpoint-18000 \
        --output_dir outputs/globalpointer-heads-v1

    # Train specific head only
    python scripts/training/train_globalpointer_heads.py \
        --checkpoint outputs/modernbert-v2-for-v3-transfer/checkpoint-18000 \
        --heads ner_family \
        --output_dir outputs/globalpointer-heads-v1

    # Resume from checkpoint
    python scripts/training/train_globalpointer_heads.py \
        --resume_from outputs/globalpointer-heads-v1/checkpoint-1000
"""
```

**Key Components**:

1. **Argument Parser**: checkpoint path, heads to train, hyperparameters
2. **Model Loader**: Load encoder from checkpoint-18000, freeze it
3. **Head Factory**: Create GlobalPointerNERHead for each capability
4. **Data Loaders**: Use GlobalPointerCollator with span-format data
5. **Training Loop**: Standard PyTorch training with validation
6. **Checkpointing**: Save heads only (encoder unchanged)

**Acceptance Criteria**:

- [ ] Script runs without errors
- [ ] Encoder parameters frozen (requires_grad=False)
- [ ] Only head parameters updated
- [ ] Checkpoint saves head weights
- [ ] Validation F1 computed correctly

#### Issue 3.1.2: Load checkpoint-18000 with freeze_encoder=True - TO IMPLEMENT

**Checkpoint Location**: `outputs/modernbert-v2-for-v3-transfer/checkpoint-18000`

**Implementation**:

```python
def load_frozen_encoder(checkpoint_path: str) -> ModernBertMultiTaskModel:
    """
    Load encoder from checkpoint and freeze all encoder parameters.

    The encoder has proven excellent - we only train new heads.
    """
    from transformers import AutoModel, AutoTokenizer
    from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

    # Load the full model from checkpoint
    model = ModernBertMultiTaskModel.from_pretrained(
        checkpoint_path,
        capabilities=[],  # No default heads - we'll add GlobalPointer
        torch_dtype=torch.bfloat16,
    )

    # Freeze encoder parameters
    for param in model.encoder.parameters():
        param.requires_grad = False

    logger.info(f"Loaded encoder from {checkpoint_path}")
    logger.info(f"Encoder parameters: {sum(p.numel() for p in model.encoder.parameters()):,} (frozen)")

    return model
```

**Verification**:

```python
# After freezing, verify no encoder gradients
encoder_trainable = sum(p.requires_grad for p in model.encoder.parameters())
assert encoder_trainable == 0, f"Expected 0 trainable encoder params, got {encoder_trainable}"
```

**Acceptance Criteria**:

- [ ] Encoder loaded from checkpoint-18000
- [ ] All encoder.parameters() have requires_grad=False
- [ ] Model can still forward() correctly
- [ ] Hidden states flow to heads

#### Issue 3.1.3: Replace TokenClassificationHead with GlobalPointerNERHead - TO IMPLEMENT

**Implementation**:

```python
def attach_globalpointer_heads(
    model: ModernBertMultiTaskModel,
    capabilities: list[str],
    hidden_size: int = 768,
    head_size: int = 64,
) -> None:
    """
    Attach GlobalPointerNERHead instances to the model.

    Replaces any existing NER heads with span-based GlobalPointer heads.
    """
    from modeling_studio.models.heads import GlobalPointerNERHead
    from modeling_studio.data.globalpointer_collator import (
        NER_GENERAL_LABELS,
        NER_FAMILY_LABELS,
        TEMPORAL_LABELS,
    )

    label_configs = {
        "ner_general": NER_GENERAL_LABELS,
        "ner_family": NER_FAMILY_LABELS,
        "temporal": TEMPORAL_LABELS,
    }

    for cap in capabilities:
        if cap not in label_configs:
            raise ValueError(f"Unknown capability: {cap}")

        labels = label_configs[cap]
        head = GlobalPointerNERHead(
            hidden_size=hidden_size,
            num_labels=len(labels),
            head_size=head_size,
            use_rope=True,
            dropout=0.1,
            loss_type="globalpointer",
        )

        # Register head in model
        model.heads[cap] = head

        logger.info(f"Attached GlobalPointerNERHead for {cap}: {len(labels)} labels")
```

**Label Configurations**:

| Capability | Labels | Count |
|------------|--------|-------|
| ner_general | PER, ORG, LOC, MISC | 4 |
| ner_family | KINSHIP, MILESTONE, HEIRLOOM, PET, NICKNAME, FAMILY_UNIT, CAREGIVING, TRADITION, HOME_BASE, FAMILY_ROLE | 10 |
| temporal | DATE_ABS, DATE_REL, TIME, DURATION, FREQUENCY, AGE | 6 |

**Acceptance Criteria**:

- [ ] GlobalPointerNERHead attached for each capability
- [ ] Correct num_labels for each
- [ ] Heads registered in model.heads dict
- [ ] Head parameters are trainable (requires_grad=True)

#### Issue 3.1.4: Configure optimizer (heads only) - TO IMPLEMENT

**Implementation**:

```python
def get_head_optimizer(
    model: ModernBertMultiTaskModel,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    """
    Create optimizer for head parameters only.

    The encoder is frozen, so we only optimize head parameters.
    """
    # Collect all head parameters
    head_params = []
    for name, head in model.heads.items():
        head_params.extend(list(head.parameters()))

    # Verify we have trainable params
    trainable = [p for p in head_params if p.requires_grad]
    logger.info(f"Trainable head parameters: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(
        trainable,
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )

    return optimizer
```

**Hyperparameters**:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 1e-4 | Higher than encoder fine-tuning (heads train from scratch) |
| Weight decay | 0.01 | Standard regularization |
| Warmup steps | 500 | 5% of expected steps |
| Scheduler | Linear | Simple, effective |

**Verification**:

```python
# Only head params in optimizer
optimizer_params = set()
for group in optimizer.param_groups:
    for p in group["params"]:
        optimizer_params.add(id(p))

encoder_params = set(id(p) for p in model.encoder.parameters())
assert optimizer_params.isdisjoint(encoder_params), "Encoder params should not be in optimizer"
```

**Acceptance Criteria**:

- [ ] Optimizer contains only head parameters
- [ ] No encoder parameters in optimizer
- [ ] Learning rate = 1e-4
- [ ] Weight decay applied
- [ ] Scheduler configured for warmup

### Epic 3.2: Configuration [COMPLETED 2026-01-21]

#### Issue 3.2.1: Create globalpointer_heads.yaml config [DONE]

**File**: `configs/training/globalpointer_heads.yaml`

```yaml
# Key settings:
encoder:
  checkpoint: outputs/modernbert-v2-for-v3-transfer/checkpoint-18000
  freeze: true  # CRITICAL: encoder stays frozen

heads:
  enabled: [ner_general, ner_family, temporal]
  architecture:
    head_size: 64
    use_rope: true
    dropout: 0.1

training:
  learning_rate: 1.0e-4  # Head LR
  encoder_lr: 0.0        # Encoder LR = 0 (frozen)
  num_epochs: 3
  batch_size: 16

data:
  ner_general:
    train: data/ner_general_span/
  ner_family:
    train: data/familyos/unified/output_healed_merged/
  temporal:
    train: data/familyos/unified/output_healed_merged/
```

#### Issue 3.2.2: Set encoder_lr=0 or freeze_encoder=true [DONE]

- Config: `encoder.freeze: true`
- Config: `training.encoder_lr: 0.0`
- Script: `freeze_encoder()` called in `load_encoder()`
- Verified: `Encoder params: 149,014,272 total, 0 trainable`

#### Issue 3.2.3: Set head_lr=1e-4 [DONE]

- Config: `training.learning_rate: 1.0e-4`
- Verified: Progress bar shows `lr=2e-7` during warmup (as expected)

#### Issue 3.2.4: Configure data loaders for span format [DONE]

- Script updated to support YAML config via `--config` flag
- `SpanNERDataset` handles both direct and unified formats
- `GlobalPointerCollator` handles span→tensor conversion
- Data paths configurable per head in config

**Usage**:

```bash
# Full training
python scripts/training/train_globalpointer_heads.py \
    --config configs/training/globalpointer_heads.yaml

# Debug mode
python scripts/training/train_globalpointer_heads.py \
    --config configs/training/globalpointer_heads.yaml \
    --debug --max_samples 100

# Single head
python scripts/training/train_globalpointer_heads.py \
    --config configs/training/globalpointer_heads.yaml \
    --heads ner_family
```

### Epic 3.3: Model Registration

#### Issue 3.3.1: Add GlobalPointerNERHead to heads.py

#### Issue 3.3.2: Update CAPABILITY_TO_HEAD_TYPE mapping

#### Issue 3.3.3: Add GlobalPointerCollator to collators.py

#### Issue 3.3.4: Update collator routing

---

## Milestone 4: Training Execution

**Goal**: Train and validate new heads

### Epic 4.1: ner_family Training

#### Issue 4.1.1: Load unified data (already span format)

#### Issue 4.1.2: Train GlobalPointer head (3-5 epochs)

#### Issue 4.1.3: Evaluate on held-out set

#### Issue 4.1.4: Measure garbage rate vs baseline

### Epic 4.2: ner_general Training

#### Issue 4.2.1: Load converted CoNLL + generated FamilyOS

#### Issue 4.2.2: Train GlobalPointer head

#### Issue 4.2.3: Evaluate on CoNLL test set

#### Issue 4.2.4: Evaluate on FamilyOS samples

### Epic 4.3: temporal Training

#### Issue 4.3.1: Load unified temporal data (already span format)

#### Issue 4.3.2: Train GlobalPointer head

#### Issue 4.3.3: Evaluate accuracy

### Epic 4.4: Validation & Comparison

#### Issue 4.4.1: Run full garbage rate audit (before/after)

#### Issue 4.4.2: Compare F1 scores vs BIO baseline

#### Issue 4.4.3: Measure latency impact

#### Issue 4.4.4: Document results in working_mreemory.md

---

## Milestone 5: Integration

**Goal**: Integrate new heads into production pipeline

### Epic 5.1: Inference Pipeline

#### Issue 5.1.1: Update forward() to handle GlobalPointer output

#### Issue 5.1.2: Update unified_output.py for span format

#### Issue 5.1.3: Remove BIO decoding for NER heads

### Epic 5.2: Export & Deployment

#### Issue 5.2.1: Update ONNX export for new head architecture

#### Issue 5.2.2: Benchmark ONNX latency

#### Issue 5.2.3: Update familyos_ultrabert package

### Epic 5.3: Documentation

#### Issue 5.3.1: Update API documentation

#### Issue 5.3.2: Update TRAINING_DEPENDENCY_TRACE.md

#### Issue 5.3.3: Archive working_mreemory.md decisions

---

## Dependencies & Blockers

### Critical Path

```
M1.Epic1.1 (BIO converter) ──┐
                              ├──> M4.Epic4.2 (ner_general training)
M1.Epic1.3 (data gen) ───────┘

M1.Epic1.2 (collator) ────────┐
                               ├──> M4.Epic4.1 (ner_family training)
M2.Epic2.1 (head impl) ───────┘

M3 (infrastructure) ──────────> M4 (training) ──────────> M5 (integration)
```

### No Blockers For

- ner_family: Data already in span format
- temporal: Data already in span format

### Blockers

- ner_general: Needs BIO conversion (M1.E1.1) + data generation (M1.E1.3)

---

## Success Criteria

| Metric | Baseline (BIO) | Target (GlobalPointer) |
|--------|----------------|------------------------|
| ner_family garbage rate | 66%+ | < 10% |
| ner_general F1 (CoNLL) | ~85% | > 90% |
| temporal garbage rate | ~40% | < 15% |
| Training time | N/A | < 4 hours (heads only) |
| Inference latency | 52ms | < 60ms |

---
