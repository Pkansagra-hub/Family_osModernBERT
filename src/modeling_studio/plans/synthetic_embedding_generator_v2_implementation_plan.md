# Synthetic Embedding Generator V2 Implementation Plan

> Scope: Extend the current LLM-backed embedding data generator with LLM-powered mining, new CLI modes, and structured output schemas for retrieval-first FamilyOS data creation.
> Updated: March 10, 2026
> Source baseline: `scripts/agents/synthetic_embedding_generator.py`

---

## 1. Goal

Create a new generator implementation that keeps the current script stable while adding production-grade support for:

- retrieval-native query-document pair generation
- wrong-person negatives
- wrong-time negatives
- safety-sensitive hard negatives
- emotion-sensitive hard negatives
- gold evaluation regeneration candidates
- failure-case benchmark candidates
- deterministic validation on generated/mined examples

The new file should be developed separately first and only replace or merge back into the original script after validation.

---

## 2. New working file

Recommended new script path:

- `scripts/agents/synthetic_embedding_generator_v2.py`

Rationale:

- preserves the current stable generator
- makes it easy to run A/B comparisons
- allows incremental migration of CLI modes and output schemas

---

## 3. Implementation principles

### 3.1 LLM-first, not heuristic-only

Heuristics should only do:

- candidate narrowing
- schema validation
- deduplication
- split control
- bucket balancing

LLMs should do:

- semantic rewriting
- positive/negative generation
- retrieval pair generation
- semantic consistency via strong generator prompts and deterministic validation
- rationale generation for auditability

### 3.2 Single-pass generation with deterministic validation

Every advanced example should use:

1. **Generator pass**

- produce candidate JSON record

1. **Deterministic validation**

- enforce schema, metadata, deduplication, and split checks

Only valid examples should be written to output.

### 3.3 Strict JSON contracts

All modes must emit schema-constrained JSONL.

Every output record should carry:

- generated content
- provenance
- mode
- tags / type labels
- generation rationale when applicable

---

## 4. Planned CLI modes

The v2 script should preserve existing modes and add new ones.

### 4.1 Existing modes to keep

- `cross_cluster`
- `hard_negative`

### 4.2 New modes to add

#### `query_doc`

Purpose:

- generate retrieval-native query-document pairs

Primary inputs:

- `data/familyos/unified/output_synthetic`

Primary outputs:

- `query_doc_pairs.jsonl`

#### `wrong_person_negative`

Purpose:

- generate hard negatives where the event is similar but the main person or kinship target changes

Primary outputs:

- `wrong_person_negatives.jsonl`

#### `wrong_time_negative`

Purpose:

- generate hard negatives where the event is similar but temporal interpretation changes

Primary outputs:

- `wrong_time_negatives.jsonl`

#### `safety_emotion_negative`

Purpose:

- generate lexically/plausibly similar but semantically important safety/emotion near misses

Primary outputs:

- `safety_emotion_negatives.jsonl`

#### `gold_regeneration`

Purpose:

- regenerate candidate gold evaluation examples for human review

Primary outputs:

- `gold_candidates.jsonl`

#### `failure_case_benchmark`

Purpose:

- generate adversarial or regression-oriented benchmark candidates from known failure patterns

Primary outputs:

- `failure_case_candidates.jsonl`

---

## 5. Proposed command shape

```text
python scripts/agents/synthetic_embedding_generator_v2.py generate --mode cross_cluster
python scripts/agents/synthetic_embedding_generator_v2.py generate --mode hard_negative
python scripts/agents/synthetic_embedding_generator_v2.py generate --mode query_doc --input-source unified_output_synthetic
python scripts/agents/synthetic_embedding_generator_v2.py generate --mode wrong_person_negative --input-source unified_output_synthetic
python scripts/agents/synthetic_embedding_generator_v2.py generate --mode wrong_time_negative --input-source unified_output_synthetic
python scripts/agents/synthetic_embedding_generator_v2.py generate --mode safety_emotion_negative --input-source unified_output_synthetic
python scripts/agents/synthetic_embedding_generator_v2.py generate --mode gold_regeneration --input-source unified_output_synthetic
python scripts/agents/synthetic_embedding_generator_v2.py generate --mode failure_case_benchmark --input-source unified_output_synthetic
python scripts/agents/synthetic_embedding_generator_v2.py stats
python scripts/agents/synthetic_embedding_generator_v2.py audit --input <path>
```

---

## 6. New top-level components to add

### 6.1 `GenerationMode` expansion

Add enum members:

- `QUERY_DOC`
- `WRONG_PERSON_NEGATIVE`
- `WRONG_TIME_NEGATIVE`
- `SAFETY_EMOTION_NEGATIVE`
- `GOLD_REGENERATION`
- `FAILURE_CASE_BENCHMARK`

### 6.2 Prompt registry

Replace large single-purpose prompt constants with a prompt registry:

- `SYSTEM_PROMPTS[mode]`

This makes mode-specific generation/judging much easier to maintain.

### 6.3 Output schema registry

Add per-mode schema validation definitions, e.g.:

- required fields by mode
- optional metadata fields by mode
- output destination by mode

### 6.4 Candidate source adapters

Add a structured way to read source candidates from:

- unified synthetic JSONL shards
- existing embedding triplets
- benchmark/failure-case seed files

Suggested abstraction:

- `CandidateSourceAdapter`
- `UnifiedOutputAdapter`
- `TripletSeedAdapter`

### 6.5 Audit / manifest writer

Every generation run should produce a manifest with:

- mode
- model used
- record counts
- rejection reasons
- source coverage
- timestamp

---

## 7. Per-mode JSON schemas

### 7.1 Base common fields

All records should include:

| Field | Required | Notes |
| --- | --- | --- |
| `mode` | Yes | Generation mode |
| `source_ids` | Yes | Source row ids |
| `source_files` | Yes | Source shard paths |
| `generator_model` | Yes | Model used for generation |
| `slice_tags` | Yes | Retrieval slice labels |
| `generator_rationale` | Optional | Why the example was generated this way |

### 7.2 `query_doc` schema

| Field | Required |
| --- | --- |
| `query` | Yes |
| `document` | Yes |
| `query_id` | Yes |
| `document_id` | Yes |
| `pair_type` | Yes |
| `shared_features` | Yes |
| `difficulty` | Yes |

### 7.3 `wrong_person_negative` / `wrong_time_negative` schema

| Field | Required |
| --- | --- |
| `anchor` | Yes |
| `positive` | Yes |
| `negative` | Yes |
| `hard_negative_type` | Yes |
| `mismatch_features` | Yes |
| `difficulty` | Yes |

### 7.4 `safety_emotion_negative` schema

| Field | Required |
| --- | --- |
| `anchor` | Yes |
| `positive` | Yes |
| `negative` | Yes |
| `hard_negative_type` | Yes |
| `safety_label_anchor` | Optional |
| `safety_label_negative` | Optional |
| `emotion_label_anchor` | Optional |
| `emotion_label_negative` | Optional |
| `mismatch_features` | Yes |

### 7.5 `gold_regeneration` schema

| Field | Required |
| --- | --- |
| `query` | Yes |
| `candidate_text` | Yes |
| `label` | Yes |
| `label_rationale` | Yes |
| `slice_tags` | Yes |
| `difficulty` | Yes |
| `review_status` | Yes |

### 7.6 `failure_case_benchmark` schema

| Field | Required |
| --- | --- |
| `query` | Yes |
| `positive` | Yes |
| `negative` | Yes |
| `failure_type` | Yes |
| `why_models_fail` | Yes |
| `slice_tags` | Yes |

---

## 8. Prompt strategy

### 8.1 Generator prompts

Each mode needs a dedicated generator prompt that:

- describes the task clearly
- constrains the mismatch dimension
- requires exact JSON output
- requires realistic FamilyOS language
- asks for provenance/rationale when needed

### 8.2 Model usage strategy

Recommended default:

- generator model: stronger creative semantic generation model

If cost-sensitive:

- generation on Gemini / Vertex Flash

---

## 9. Output folder layout

Recommended output layout:

```text
data/
  familyos/
    embeddings/
      silver_synthetic/
      hard_negatives/
      mined_v2/
        query_doc/
          query_doc_pairs_0000.jsonl
          manifest.json
        wrong_person/
          wrong_person_negatives_0000.jsonl
          manifest.json
        wrong_time/
          wrong_time_negatives_0000.jsonl
          manifest.json
        safety_emotion/
          safety_emotion_negatives_0000.jsonl
          manifest.json
      gold_candidates/
        gold_candidates_0000.jsonl
        manifest.json
    benchmarks/
      failure_cases_candidates/
        failure_case_candidates_0000.jsonl
        manifest.json
```

---

## 10. Validation and rejection pipeline

For every generated record:

1. parse JSON
2. validate required fields
3. validate mode-specific schema
4. reject malformed or semantically inconsistent examples using deterministic checks
5. deduplicate against existing outputs
6. write accepted examples only
7. record rejection reason histogram in manifest

Common rejection reasons:

- malformed JSON
- wrong mismatch type
- positive not truly positive
- negative too easy
- negative not retrieval-relevant
- duplicate / near-duplicate
- not FamilyOS-style enough

---

## 11. Phased implementation plan

### Phase 1: isolate the new script

- copy current script to `synthetic_embedding_generator_v2.py`
- keep existing CLI modes working unchanged
- verify parity for `cross_cluster` and `hard_negative`

### Phase 2: add architecture scaffolding

- add expanded `GenerationMode`
- add prompt registry
- add output schema registry
- add output-path routing by mode
- add manifest writer

### Phase 3: strengthen deterministic validation

- add accepted/rejected counters
- add rejection reason categories
- persist generator rationale when available

### Phase 4: implement first new modes

Order:

1. `wrong_person_negative`
2. `wrong_time_negative`
3. `query_doc`
4. `safety_emotion_negative`

### Phase 5: gold and benchmark generation

- implement `gold_regeneration`
- implement `failure_case_benchmark`
- add human-review-ready outputs

### Phase 6: audit tooling

- add `audit` command
- summarize slice balance, difficulty balance, rejection reasons, duplicates, and schema issues

---

## 12. Success criteria

The v2 generator is successful if it can:

- preserve current generation behavior for existing modes
- generate structured new retrieval data modes
- use deterministic validation instead of extra LLM verification passes
- produce auditable manifests and provenance
- feed training/eval pipelines with mode-specific JSONL outputs
- regenerate candidate gold data for human review

---

## 13. Immediate next coding step

Immediate next step after this plan:

1. copy `synthetic_embedding_generator.py` to `synthetic_embedding_generator_v2.py`
2. add new enum modes and prompt registries without changing behavior yet
3. keep old generation paths intact
4. then begin adding new modes one at a time
