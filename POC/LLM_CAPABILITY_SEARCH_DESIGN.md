# K1 LLM Capability Search Design (Architecture-Aligned)

## Purpose

Design a production-grade capability search system for K1 where:

- Planner and Concierge discover capabilities semantically (`discover_capabilities`, `find_relevant_prompts`)
- Domain identity is embedded into vector geometry (not index partitioning)
- Retrieval remains flat, low-latency, and policy-safe
- Results are ready for Stage 1/2 planning and direct low-tier execution

This design aligns with:

- `L2.5 Fabric Retrieval Engine` in `k1_cognitive_architecture_skeleton.mmd`
- `Capability Fabric` and `EmbeddingIndex` in `fabric_new.mmd`
- Bridge/IFL adapter reality in bridge architecture diagrams

---

## Where it plugs in (exact architecture path)

### Caller path

1. `PlannerAgent` Stage 1/2 calls:
   - `discover_capabilities(domain?, intent?)`
   - `find_relevant_prompts(intent?, domain?)`
2. `FabricRetrieval` receives query
3. `SemanticSearchEngine` embeds query
4. `RetrievalHardFilter` enforces safety/availability/satisfiability
5. `RetrievalSoftRanker` computes composite score
6. `TopKRanker` returns Top-K to Planner

### Data source path

1. `ModuleLoader` loads contracts (tool/agent/workflow/prompt)
2. `ContractValidator` approves
3. `CapabilityRegistry` registers
4. `EmbeddingIndex` computes and stores vectors

No structural partitioning changes required.

---

## Embedding document model

## Canonical embedding text

Use a single canonical contract-to-text projection:

`domain:{domains_joined} | category:{category} | provider:{provider_type} | company:{company} | adapter:{adapter_id} | capability:{name} | description:{description} | capabilities:{capability_terms} | tags:{tags} | io:{input_keys}->{output_keys}`

### Rules

- `domains_joined`: stable sorted list (`health finance` for multi-domain)
- `name`: full capability name (`tool.read.finance.chase.check_balance`)
- Include `company` and `adapter` when available (IFL realism)
- Never repeat domain tokens excessively (avoid over-dominance)
- Keep deterministic formatting to stabilize vector drift

---

## Query model

## Query text templates

Generate two query embeddings per request:

1. **Hinted** (if ingress/domain hints exist):
   `domain:{hint_domains} | intent:{intent} | {user_text}`
2. **Blind**:
   `{user_text}`

Use fused retrieval score:

$$
S_{sem} = 0.65 \cdot \cos(q_{hint}, d) + 0.35 \cdot \cos(q_{blind}, d)
$$

Fallback to blind-only when no domain signal exists.

---

## Ranking pipeline

## Hard filters (must pass)

- Safety band access (`user_band >= capability_band_min`)
- Availability (`OFFLINE` excluded, `DEGRADED` allowed)
- Required input satisfiability from current planning context
- Optional: tenant/family policy constraints

## Soft rank score

Use architecture weights, but with explicit feature definitions:

$$
S = 0.40 S_{sem} + 0.30 S_{domain} + 0.15 S_{success} + 0.15 S_{costlat}
$$

Where:

- $S_{sem}$: fused cosine score above
- $S_{domain}$:
  - 1.0 exact primary domain
  - 0.75 multi-domain overlap
  - 0.4 inferred related domain
  - 0.0 mismatch
- $S_{success}$: normalized 30-day success SLI per capability/provider
- $S_{costlat}$: normalized inverse of expected latency+cost

Apply degraded penalty:

$$
S \leftarrow 0.7S \quad \text{if availability = DEGRADED}
$$

---

## Index strategy

## Flat + IVF hybrid

- Ground-truth index: `IndexFlatIP` for evaluation and small catalogs
- Production index: `IndexIVFFlat` (or HNSW later)
  - start: `nlist=100`, `nprobe=10`
  - adapt by corpus size and SLA

### Dynamic policy

- `< 10K`: Flat only
- `10K–250K`: IVF + periodic centroid refresh
- `> 250K`: IVF/HNSW + shard-aware cache layer

---

## Contract schema additions (minimal)

Add optional metadata fields to contracts (tool/agent/workflow/prompt):

- `domain[]` (already present in most)
- `company` (string)
- `adapter_id` (string)
- `tags[]` (string list)
- `retrieval_hints`:
  - `synonyms[]`
  - `anti_domains[]` (for adversarial suppression)

These are retrieval-only fields; execution path unchanged.

---

## Multi-domain behavior

No duplication required.

- Multi-domain capabilities keep `domain:"a b"` in embedding text
- Ranker gives overlap bonus when query domain intersects either domain
- Centroid overlap is expected and desirable

Monitor with domain representation ratio in nearest-neighbor set.

---

## LLM-facing output contract

Return a planner-safe retrieval payload:

- `capability_name`
- `provider_type`
- `score`
- `domain_match`
- `safety_band_min`
- `required_inputs`
- `output_schema_ref`
- `estimated_latency_ms`
- `availability`
- `explain` (1-line reason)

This supports Stage 2 expansion and Stage 3 validation without another lookup.

---

## Guardrails against false semantic collisions

1. Add domain token once in canonical prefix
2. Add company+adapter discriminators
3. Add anti-domain downweight in ranker when `anti_domains` hit
4. Keep adversarial pair monitoring as a release gate

Release gate suggestion:

- discrimination_gain >= 0.10
- P@5 no-hint >= 0.75 on realistic corpus slices

---

## Observability and SLOs

Emit retrieval metrics per call:

- `retrieval.duration_ms`
- `retrieval.candidates_pre_filter`
- `retrieval.candidates_post_filter`
- `retrieval.top1_domain_match`
- `retrieval.topk_overlap_with_groundtruth` (offline)
- `retrieval.ivf_recall_at_10` (shadow tests)

Track by domain and provider class (MCP/WASM/BRIDGE/AGENT/WORKFLOW).

---

## Rollout plan

1. **Phase A (shadow):** build embeddings with canonical text, run dual-ranker offline
2. **Phase B (read-only canary):** planner sees new ranking but execution unchanged
3. **Phase C (production):** make new ranker default, keep rollback flag
4. **Phase D:** tune domain weight and IVF parameters by telemetry

Feature flags:

- `retrieval.embedding_template=v2`
- `retrieval.ranker=v2`
- `retrieval.query_fusion=on`

---

## What to implement next in codebase

1. `ModuleLoader`: canonical embedding text builder (single utility)
2. `EmbeddingIndex`: versioned vector refresh on contract changes
3. `RetrievalEngine`: dual-query fusion + explicit score breakdown
4. `TopKRanker`: include explainability payload
5. `Metrics`: add retrieval observability fields above

This gives architecture-consistent LLM capability search without changing Orchestrator/Concierge execution semantics.
