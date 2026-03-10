# Tool Retrieval Improvement Playbook (UltraBERT Constraint)

## 1) Problem Definition (Current State)

We are solving the `find_capability` retrieval problem for tool selection.

Current reality from latest `POC/report.txt` (UltraBERT run):

- Baseline exact tool presence (`TPR@10`) is low (`0.190`).
- With rerank/domain/hybrid logic we improved to `0.300`, but this still means 70% exact-tool miss rate.
- Domain and operation coverage can be pushed very high (`DMR=1.000`, `OMR=1.000` under hybrid), but exact-tool precision remains the bottleneck.

### Core diagnosis

1. LLM selection is not the bottleneck once right candidates are present.
2. Retrieval candidate set quality is the bottleneck (especially exact-target recall).
3. Tool contract text quality and naming consistency in YAML strongly influence retrieval.

---

## 2) What We Can Change (and Should Change)

## 2.1 Editable surface in tool YAML contracts

These fields are highest leverage for retrieval quality:

- `tool_contract.name`
  - Must encode operation + object clearly.
  - Prefer stable action-object format.
  - Example pattern: `tool.<type>.<domain>.<provider>.<action_object>`

- `tool_contract.description`
  - First sentence should be operation-specific, not generic.
  - Include exact business object and constraints.
  - Avoid vague text like "handles workflow operations".

- `tool_contract.capabilities`
  - Include canonical operation terms and close synonyms.
  - Keep concise and retrieval-oriented.

- `tool_contract.required_inputs`
  - Use meaningful parameter names, not generic placeholders.
  - Include identifiers that disambiguate tools.

- `tool_contract.domain`
  - Must be normalized and accurate (single canonical taxonomy).

- `tool_contract.tags` / `limitations`
  - Add discriminative tags (entity type, channel, integration surface).

## 2.2 What NOT to change blindly

- Do not stuff unrelated keywords into descriptions/tags.
- Do not duplicate near-identical tools without clear discriminator fields.
- Do not use inconsistent domain naming (`calendar`, `cal`, `scheduling`) without normalization.

---

## 3) Vectorization Strategy (UltraBERT is Fixed Constraint)

UltraBERT remains the embedding engine. We optimize text shaping and retrieval stack around it.

## 3.1 Canonical embedding text (tool side)

Use template with deterministic field order:

1. `domain`
2. `operation canonical`
3. `tool name (humanized + raw operation)`
4. `description` (first sentence + key disambiguator sentence)
5. `capabilities`
6. `required_inputs`
7. `tags`

## 3.2 Query side normalization

Before embedding query:

1. normalize intent tokens (`create/add/new -> create`, `get/fetch/list -> retrieve`, etc.)
2. expand operation ontology synonyms
3. keep domain hint explicit if known
4. remove non-informative prompt wrappers

## 3.3 Retrieval pipeline (recommended production shape)

1. Domain routing (hard filter when caller domain is known).
2. Vector candidate retrieval from domain sub-index.
3. Hybrid fusion rerank:
   - vector similarity
   - BM25/token score over contract text
   - operation ontology overlap bonus
4. Return top-10 diverse candidates for LLM selection.

---

## 4) Target Metrics and Gates

We should optimize for tool-selection success, not just geometric cluster scores.

## 4.1 Primary KPI gates

- `TPR@10` (exact tool present):
  - Phase gate 1: `>= 0.50`
  - Phase gate 2: `>= 0.70`
  - Release gate: `>= 0.90`

- `DMR@10` (domain present):
  - Minimum gate: `>= 0.98`

- `OMR@10` (operation present):
  - Minimum gate: `>= 0.95`

## 4.2 Secondary quality metrics

- `Exact MRR`: trending upward each phase (no regression tolerated)
- `P@5` usable relevance: `>= 0.90`
- `Schema Quality Score (SQS)`: `>= 0.90`

## 4.3 Regression policy

Any change is rejected if:

- `TPR@10` drops by more than `0.02`, or
- `DMR@10` drops below `0.98`, or
- `OMR@10` drops below `0.95`

---

## 5) Step-by-Step Execution Plan

## Step 0: Baseline freeze

- Freeze current golden dataset and reports.
- Save baseline metrics for:
  - UltraBERT baseline
  - UltraBERT hybrid
  - standard model baseline/hybrid

Deliverable:

- `baseline_snapshot.md` with metric table.

## Step 1: YAML normalization pass

- Normalize domain labels and operation naming patterns.
- Rewrite low-quality descriptions (first sentence must be discriminative).
- Standardize capability vocabulary using ontology.

Deliverable:

- `yaml_normalization_changes.md`
- diff summary of changed contracts.

Acceptance gate:

- `SQS >= 0.95`
- no schema validation errors.

## Step 2: Ontology expansion and mapping QA

- Expand operation ontology to cover real contract verbs.
- Add canonical mapping tests for major intent families.

Deliverable:

- `operation_ontology_v1.md`
- mapping test report.

Acceptance gate:

- `OMR@10 >= 0.90` (intermediate).

## Step 3: Retrieval fusion calibration

- Tune hybrid weights (`vector`, `bm25`, domain/op bonuses).
- Run grid search on golden set.

Deliverable:

- `fusion_tuning_results.md` with best config and rationale.

Acceptance gate:

- `TPR@10 >= 0.50`
- `DMR@10 >= 0.98`

## Step 4: Miss-case diagnostics loop

- Export all misses where exact target not in top-10.
- Categorize miss reason:
  - naming ambiguity
  - description ambiguity
  - domain mismatch
  - ontology gap
  - duplicate/overlapping tools

Deliverable:

- `top10_miss_analysis.md`
- prioritized fix backlog.

Acceptance gate:

- each miss category mapped to concrete remediation action.

## Step 5: Iterative hardening to release threshold

- Apply highest-impact fixes in batches.
- Re-run benchmark each batch.

Release gate:

- `TPR@10 >= 0.90`
- `DMR@10 >= 0.98`
- `OMR@10 >= 0.95`
- no regression policy violations.

---

## 6) Immediate Next Action (Start Here)

Start with **Step 1 (YAML normalization pass)** on a limited slice (first 100 tools), then rerun benchmark and compare against baseline.

Why this first:

- It directly addresses exact-tool miss root causes.
- It improves all retrieval modes without changing UltraBERT.
- It gives durable gains across models.

---

## 7) Decision Rules (for day-to-day work)

- If exact tool is absent but operation is present -> improve disambiguation fields (`name`, first sentence, capabilities).
- If domain absent -> fix domain taxonomy or routing logic first.
- If both present but wrong ranking -> adjust hybrid fusion weights and lexical signal weighting.
- If two tools are semantically duplicate -> merge, split, or add explicit discriminator fields.

---

## 8) Summary

- Constraint: keep UltraBERT.
- Lever: improve contract text quality + retrieval architecture.
- Objective: move from "usable candidates" to reliable exact-tool presence in top-10.
- Execution: phase gates with hard numeric targets and regression policy.
