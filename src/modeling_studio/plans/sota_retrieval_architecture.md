# SOTA Retrieval Architecture for FamilyOS

> Scope: Define the target retrieval architecture and the plan to reach a production-grade FamilyOS embedding system.
> Updated: March 10, 2026

---

## 0. Executive summary

The current bake-off results indicate that `agreement_gated` is the strongest retrieval head candidate so far. The next step is not to treat it as "done," but to define a rigorous path from promising bake-off winner to a world-class FamilyOS retriever.

For FamilyOS, "world class" means the embedding system must do more than generic semantic similarity. It must preserve:

- family-role distinctions
- temporal distinctions
- emotional nuance and intensity
- safety-critical differences
- retrieval correctness for memory-style queries
- robustness on messy, real-world user phrasing

This document defines:

1. architecture
2. data needed vs data already available
3. training strategy and loss functions
4. evaluation plan
5. benchmarking plan, including golden data creation

---

## 1. Architecture

### 1.1 Design goals

The retrieval architecture must satisfy five constraints:

1. **Retrieval-first**: optimize for nearest-neighbor retrieval quality, not just STS.
2. **Stable geometry**: preserve the good properties of encoder mean pooling as the safe anchor.
3. **Nuance-aware**: represent family role, time, emotion, intent, and safety distinctions.
4. **Deployable**: support fast single-vector retrieval for FamilyOS runtime.
5. **Auditable**: expose enough metadata and diagnostics to explain failures.

### 1.2 Base system definition

The target system is a two-stage retrieval stack:

1. **Dense retriever**: ModernBERT encoder + FamilyOS retrieval head
2. **Optional reranker**: cross-encoder or pairwise reranker for top-$k$ reordering

The dense retriever is the primary scope of this document.

### 1.3 Backbone

- **Encoder**: `answerdotai/ModernBERT-base`
- **Checkpoint base**: `outputs/globalpointer-unified-v1/checkpoint-8000` (or best checkpoint equivalent per environment)
- **Embedding size**: 768 by default
- **Runtime contract**: one normalized embedding vector per input text

### 1.4 Recommended head family

The architectural winner candidate is `AgreementGatedHead`, evolving into **AgreementGatedHeadV2**.

#### Current good idea to preserve

The key principle that should remain unchanged is:

> The mean-pooled encoder representation is the anchor. Learned refinement is allowed only when auxiliary views agree enough to justify deviation.

This makes the head safer than unconstrained projection heads.

### 1.5 AgreementGatedHeadV2 implementation spec

This section is intentionally written as a build-ready spec for `heads_embedding.py`.

#### Class contract

```python
class AgreementGatedHeadV2(nn.Module):
   def __init__(
      self,
      hidden_size: int,
      output_dim: int | None = None,
      normalize: bool = True,
      num_latents: int = 4,
      num_attn_heads: int = 4,
      gate_hidden: int = 128,
      gate_rank: int = 4,
      use_mode_prompts: bool = True,
      use_confidence_head: bool = True,
      dropout: float = 0.1,
      eps: float = 1.0e-8,
      **kwargs: Any,
   ):
      ...

   def forward(
      self,
      hidden_states: torch.Tensor,
      attention_mask: torch.Tensor | None = None,
      mode: str = "document",
      return_aux: bool = False,
   ) -> torch.Tensor | dict[str, torch.Tensor]:
      ...
```

#### Input / output contract

Inputs:

- `hidden_states`: $[B, L, D]$
- `attention_mask`: $[B, L]$ or `None`
- `mode`: one of `"query"` or `"document"`

Outputs:

- default: normalized embedding $[B, O]$
- optional auxiliary dictionary when `return_aux=True`

Definitions:

- $B$: batch size
- $L$: sequence length
- $D$: encoder hidden size
- $O$: output dimension, where $O = output\_dim$ if provided else $D$

#### Constructor parameters

| Parameter | Type | Default | Purpose |
| --- | --- | --- | --- |
| `hidden_size` | `int` | required | Encoder hidden width $D$ |
| `output_dim` | `int \| None` | `None` | Final embedding width $O$ |
| `normalize` | `bool` | `True` | L2-normalize final embedding |
| `num_latents` | `int` | `4` | Number of learnable latent queries |
| `num_attn_heads` | `int` | `4` | Heads in latent cross-attention |
| `gate_hidden` | `int` | `128` | Hidden size in gate and fusion MLPs |
| `gate_rank` | `int` | `4` | Number of semantic gate blocks before expansion to $O$ |
| `use_mode_prompts` | `bool` | `True` | Enable lightweight query/document asymmetry |
| `use_confidence_head` | `bool` | `True` | Produce confidence / ambiguity side outputs |
| `dropout` | `float` | `0.1` | Dropout for fusion/gating sublayers |
| `eps` | `float` | `1e-8` | Numerical stability constant |

#### Persistent attributes

The module should expose at least these attributes for checkpoint metadata and reload:

- `self.hidden_size`
- `self.output_dim`
- `self.normalize`
- `self.pooling = "agreement_gated_v2"`
- `self.num_latents`
- `self.num_attn_heads`
- `self.gate_hidden`
- `self.gate_rank`
- `self.use_mode_prompts`
- `self.use_confidence_head`

#### Submodules

##### 1. Base projections and norms

- `self.input_norm = nn.LayerNorm(hidden_size)`
- `self.base_proj = nn.Linear(hidden_size, output_dim)` if `output_dim != hidden_size`, else `None`
- `self.fusion_norm = nn.LayerNorm(output_dim)`

##### 2. Latent-view path

- `self.latent_queries = nn.Parameter(torch.randn(1, num_latents, hidden_size) * 0.02)`
- `self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_attn_heads, dropout=dropout, batch_first=True)`
- `self.attn_norm = nn.LayerNorm(hidden_size)`
- output latent view shape: $[B, D]$

##### 3. Mode prompts for query/document asymmetry

If `use_mode_prompts=True`:

- `self.query_prompt = nn.Parameter(torch.zeros(1, 1, hidden_size))`
- `self.document_prompt = nn.Parameter(torch.zeros(1, 1, hidden_size))`

Prompt application rule:

$$
H' = H + P_{mode}
$$

where `P_mode` is broadcast to $[B, L, D]$.

##### 4. Salience scorers

These replace hand-coded lexical features with differentiable token salience signals.

- `self.role_scorer = nn.Sequential(nn.Linear(hidden_size, gate_hidden), nn.GELU(), nn.Linear(gate_hidden, 1))`
- `self.temporal_scorer = nn.Sequential(nn.Linear(hidden_size, gate_hidden), nn.GELU(), nn.Linear(gate_hidden, 1))`
- `self.safety_scorer = nn.Sequential(nn.Linear(hidden_size, gate_hidden), nn.GELU(), nn.Linear(gate_hidden, 1))`

Each scorer produces token logits of shape $[B, L, 1]$.

##### 5. View projection layers

Project every pooled view into the common output space:

- `self.cls_proj = nn.Linear(hidden_size, output_dim, bias=False)`
- `self.latent_proj = nn.Linear(hidden_size, output_dim, bias=False)`
- `self.max_proj = nn.Linear(hidden_size, output_dim, bias=False)`
- `self.role_proj = nn.Linear(hidden_size, output_dim, bias=False)`
- `self.temporal_proj = nn.Linear(hidden_size, output_dim, bias=False)`

The mean view uses `self.base_proj` or identity.

##### 6. Refinement fusion MLP

Input is the concatenation of five auxiliary views in output space:

$$
[e_{cls}; e_{lat}; e_{max}; e_{role}; e_{temp}] \in \mathbb{R}^{B \times 5O}
$$

Module:

- `self.refine_mlp = nn.Sequential(nn.Linear(5 * output_dim, gate_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(gate_hidden, output_dim))`

Output:

- `e_refined_delta`: $[B, O]$

##### 7. Agreement feature MLP

Use compact numeric agreement features rather than raw view concatenation for gating.

Feature families per batch item:

1. pairwise cosine similarities among six views
2. pairwise norm ratios among six views
3. salience entropy for role/temporal/safety scorers
4. salience concentration stats for role/temporal/safety scorers
5. mode bit: query or document

Recommended exact feature count:

- 15 pairwise cosines for 6 views
- 15 pairwise norm ratios for 6 views
- 3 entropy values
- 3 concentration values
- 1 mode flag

Total gate feature width:

$$
F = 15 + 15 + 3 + 3 + 1 = 37
$$

Module:

- `self.gate_mlp = nn.Sequential(nn.Linear(37, gate_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(gate_hidden, gate_rank))`

##### 8. Gate expansion layer

Expand low-rank gate blocks to output width:

- `self.gate_expand = nn.Linear(gate_rank, output_dim)`

Final vector gate:

$$
g = \sigma(W_{expand}(h_{gate})) \in \mathbb{R}^{B \times O}
$$

##### 9. Confidence head

If `use_confidence_head=True`:

- `self.confidence_head = nn.Sequential(nn.Linear(37 + output_dim, gate_hidden), nn.GELU(), nn.Linear(gate_hidden, 3))`

Outputs three logits or bounded scores for:

- `embedding_confidence`
- `retrieval_confidence`
- `ambiguity_score`

#### View definitions and tensor shapes

Let $H \in \mathbb{R}^{B \times L \times D}$ after optional prompt injection.

1. **Mean view**
   - `e_mean_raw = masked_mean_pool(H, attention_mask)`
   - shape: $[B, D]$
   - project to output space: $e_{mean} \in [B, O]$

2. **CLS view**
   - `e_cls_raw = cls_pool(H)`
   - shape: $[B, D]$
   - `e_cls = cls_proj(e_cls_raw)`

3. **Latent view**
   - latent queries: $[B, N_{lat}, D]$
   - cross attention output: $[B, N_{lat}, D]$
   - mean pool latents to $[B, D]$
   - `e_lat = latent_proj(e_lat_raw)`

4. **Max view**
   - `e_max_raw = masked_max_pool(H, attention_mask)`
   - shape: $[B, D]$
   - `e_max = max_proj(e_max_raw)`

5. **Role view**
   - token logits from `role_scorer(H)` -> $[B, L, 1]$
   - masked softmax over sequence length -> role weights $[B, L, 1]$
   - weighted sum over tokens -> role pooled vector $[B, D]$
   - `e_role = role_proj(e_role_raw)`

6. **Temporal view**
   - token logits from `temporal_scorer(H)` -> $[B, L, 1]$
   - masked softmax over sequence length -> temporal weights $[B, L, 1]$
   - weighted sum over tokens -> temporal pooled vector $[B, D]$
   - `e_temp = temporal_proj(e_temp_raw)`

#### Agreement feature computation

The six views used for agreement are:

- `e_mean`
- `e_cls`
- `e_lat`
- `e_max`
- `e_role`
- `e_temp`

For each pair $(i, j)$ with $i < j$:

1. cosine similarity

$$
\cos(e_i, e_j)
$$

1. norm ratio

$$
\frac{\|e_i\|}{\|e_j\| + \epsilon}
$$

For role/temporal/safety salience distributions, compute:

1. entropy
2. max probability
3. top-k mass, where `k=3`

Mode flag encoding:

- query -> `1.0`
- document -> `0.0`

Concatenate all features into $[B, 37]$.

#### Forward pass algorithm

1. validate `mode in {"query", "document"}`
1. optionally add mode prompt to token states
1. compute six pooled views
1. project views to output space
1. build refinement vector:

$$
e_{aux} = \text{refine\_mlp}([e_{cls}; e_{lat}; e_{max}; e_{role}; e_{temp}])
$$

1. define refined target:

$$
e_{refined} = e_{mean} + e_{aux}
$$

1. compute gate features $f \in [B, 37]$
1. compute low-rank gate state:

$$
h_{gate} = \text{gate\_mlp}(f) \in [B, R]
$$

1. expand to vector gate:

$$
g = \sigma(\text{gate\_expand}(h_{gate})) \in [B, O]
$$

1. apply residual refinement:

$$
e_{out} = e_{mean} + g \odot (e_{refined} - e_{mean})
$$

1. apply `fusion_norm`
1. L2 normalize if `normalize=True`
1. if `return_aux=True`, also return diagnostics and confidence outputs

#### Initialization rules

To preserve the current safe behavior, initialization must bias toward mean-pool equivalence.

Required initialization:

- `query_prompt`, `document_prompt`: zeros
- `latent_queries`: `Normal(0, 0.02)`
- `gate_mlp` last-layer bias: negative, e.g. `-2.0`
- `gate_expand.weight`: small init
- `gate_expand.bias`: negative so gate starts mostly closed
- `refine_mlp` final layer: small init

Effect:

- early training behaves close to `mean_view`
- model learns to open dimensions only where agreement supports it

#### Optional auxiliary outputs

When `return_aux=True`, return:

```python
{
   "embedding": e_out,
   "confidence": confidence_scores,            # [B, 3] if enabled
   "gate": g,                                 # [B, O]
   "gate_features": f,                        # [B, 37]
   "views": {
      "mean": e_mean,
      "cls": e_cls,
      "latent": e_lat,
      "max": e_max,
      "role": e_role,
      "temporal": e_temp,
   },
   "salience": {
      "role_weights": role_weights,          # [B, L]
      "temporal_weights": temp_weights,      # [B, L]
      "safety_weights": safety_weights,      # [B, L]
   },
}
```

Note: training code can ignore this and use default embedding-only return; diagnostics are primarily for ablations and offline analysis.

#### Checkpoint / metadata requirements

`get_head_constructor_params()` should be extended to serialize:

- `gate_rank`
- `use_mode_prompts`
- `use_confidence_head`
- `dropout`

Registry entry to add:

```python
"agreement_gated_v2": AgreementGatedHeadV2
```

#### Expected parameter regime

Target parameter budget should remain modest relative to the encoder:

- expected range: roughly 2M to 5M parameters depending on `gate_hidden`, `gate_rank`, and projections
- must remain lightweight enough for joint head bake-offs on A100-class hardware

#### Non-goals for V2

Do not add the following in V2:

- full cross-encoder interaction between query and document
- symbolic reasoning modules
- external parsers as hard dependencies in forward pass
- multi-vector late interaction retrieval

Those can be layered on later, but V2 must remain a drop-in single-vector head for `heads_embedding.py`.

### 1.6 Production retrieval design

#### Phase 1

- single-vector dense retrieval only
- `agreement_gated` or `agreement_gated_v2`
- ANN index over memory/document embeddings

#### Phase 2

- dense retrieval top-$k$
- rerank top-$k$ with a cross-encoder or lightweight pair scorer
- use confidence score to decide when reranking is required

### 1.7 Promotion criteria for architecture

Promote a head only if it beats baseline on:

- full-eval retrieval margin
- hard-negative accuracy
- FamilyOS role/temporal/safety evaluations
- retrieval usefulness on golden benchmark

---

## 2. Data needed vs available

### 2.1 Data already available now

#### Core FamilyOS retrieval training data

Available and immediately usable:

- `data/familyos/embeddings/silver_synthetic`
- `data/familyos/embeddings/hard_negatives`

Observed current working total from bake-off run:

- silver synthetic: 261,805 triplets
- hard negatives: 42,945 triplets
- total: **304,750 triplets**

These are sufficient for current retrieval-head training and bake-off selection.

#### Additional documented sources

From `embedding_training_data_inventory.md`:

- FamilyOS gold triplets: desirable if restored or regenerated
- mined retrieval data from `data/familyos/unified/output_synthetic`
- open-source semantic regularization datasets such as STS Benchmark, AllNLI, SICK-R, STS12/13/14

### 2.2 Data we still need

To make the system genuinely world class, the current triplet pool is not enough. We still need curated data for the failure modes that matter most in FamilyOS.

#### Needed but not yet mature enough

1. **Golden FamilyOS retrieval set**
   - human-validated query → relevant memory/document labels
   - required for model selection and promotion

2. **Role-confusion hard negatives**
   - same event, wrong person
   - same people, wrong relation direction

3. **Temporal-confusion hard negatives**
   - same event frame, wrong date/time/frequency
   - past vs ongoing vs recurring vs planned

4. **Safety-confusion hard negatives**
   - similar wording, different safety implication
   - prevents dangerous semantic collapse

5. **Emotion nuance sets**
   - sadness vs anger vs anxiety vs shame
   - mild vs severe intensity

6. **Query-document pairs**
   - short, messy user query → richer memory/document match
   - needed for asymmetric retrieval training

### 2.3 Data gap table

| Category | Available now | Needed next | Why it matters |
| --- | --- | --- | --- |
| Generic contrastive retrieval | Yes | More variety later | Current triplets are enough for bake-off |
| Family role distinctions | Partial | Stronger gold + hard negatives | Prevent wrong-person/wrong-relation retrieval |
| Temporal distinctions | Partial | Stronger gold + hard negatives | Prevent wrong-time retrieval |
| Emotion nuance | Partial | Curated near-neighbor negatives | Needed for semantic nuance |
| Safety distinctions | Weak | Dedicated benchmark + negatives | Must avoid unsafe nearest neighbors |
| Query-document asymmetry | Weak | Needed | Current training is mostly symmetric triplets |
| Gold validation data | Weak / absent | Needed urgently | Required for trustworthy promotion |

### 2.4 Recommended data roadmap

#### Immediate

- continue using the existing 304,750 triplets as the core training set
- add mined wrong-person and wrong-time negatives
- restore or regenerate `gold` evaluation data if possible

#### Next

- mine retrieval-native query-document pairs from unified FamilyOS synthetic data
- add safety-sensitive and emotion-sensitive hard negatives

#### Later

- add production-derived anonymized retrieval judgments if policy permits
- add human adjudicated failure-case benchmark slices

### 2.5 LLM-powered data creation strategy

The key requirement is to avoid a brittle heuristics-only mining pipeline. The right design is **LLM-first with structured constraints**, not pure regex/overlap rules and not unconstrained free-form generation either.

The recommended pattern is:

1. **candidate proposal** from structured FamilyOS data
2. **LLM generation or transformation** to create retrieval pairs / negatives / gold candidates
3. **LLM judge verification** to validate semantic intent
4. **schema validation + dedup + balance controls** as final guardrails

This keeps the semantic heavy lifting in the LLM, while keeping data quality auditable.

### 2.6 Use of `scripts/agents/synthetic_embedding_generator.py`

The existing script `scripts/agents/synthetic_embedding_generator.py` should be treated as the backbone for LLM-powered data creation.

It already provides:

- OpenRouter / Vertex-backed LLM generation
- parallel generation workers
- cross-cluster triplet generation
- same-cluster hard-negative generation
- validation hooks
- shard writing and deduplication

Rather than replacing it, we should extend it with new LLM-powered generation modes.

### 2.7 Recommended generator modes to add

The script currently supports:

- `cross_cluster`
- `hard_negative`

Recommended new modes:

1. `query_doc`
   - generate retrieval-native query-document pairs
   - input source: `data/familyos/unified/output_synthetic`

2. `wrong_person_negative`
   - create same-event, wrong-person near misses
   - LLM must preserve topic and event frame while changing the main entity or kinship role

3. `wrong_time_negative`
   - create same-event, wrong-time near misses
   - LLM must preserve event content while changing date, timing, recurrence, or temporal status

4. `safety_emotion_negative`
   - create safety-sensitive and emotion-sensitive near misses
   - LLM must preserve lexical similarity while changing safety implication or emotional interpretation

5. `gold_regeneration`
   - regenerate candidate gold examples and rationales for human review

6. `failure_case_benchmark`
   - generate adversarial or regression-style benchmark candidates from known model failure types

### 2.8 LLM-powered mining pipeline

The mining pipeline should not be "heuristic or LLM"; it should be **heuristics for candidate narrowing, LLM for semantic judgment**.

#### Stage 1: candidate harvesting

Use lightweight rules only to construct candidate pools from `data/familyos/unified/output_synthetic`:

- same intent / ingress neighborhood
- shared entity family
- shared relation predicate family
- shared temporal family
- same safety band where appropriate

This stage should not decide truth. It should only reduce search space.

#### Stage 2: LLM semantic transformation

For each candidate pool, ask the LLM to perform one of these controlled tasks:

- rewrite a query-like utterance into a memory/document form
- identify the best positive memory/document candidate
- generate a wrong-person version of the same event
- generate a wrong-time version of the same event
- generate a safety-sensitive near miss
- generate an emotion-shifted near miss

The LLM prompt should require strict JSON output with provenance and rationale fields.

#### Stage 3: LLM judge verification

Use a second LLM pass as a **judge**, not the same prompt as the generator.

Judge tasks:

- verify that anchor and positive refer to the same event or retrieval target
- verify that negative differs in the intended dimension only
- classify negative type
- reject examples that are too easy, too ambiguous, or semantically wrong

This is the main way to avoid low-quality heuristic artifacts.

#### Stage 4: deterministic filters

After LLM generation/judging, apply deterministic checks only for:

- JSON schema validity
- duplicate removal
- minimum/maximum length
- forbidden empty fields
- split leakage prevention
- class balance and slice quotas

### 2.9 How to create the requested data types

#### A. Wrong-person negatives

Goal:

- preserve event and topical frame
- change the principal person / kinship target

LLM task:

- given a source text and entity metadata, produce a near-miss version that remains plausible but changes who the event is about

Examples of valid transformations:

- child -> sibling
- mother -> father
- self -> child
- parent-of relation -> sibling-of relation

Judge criteria:

- same topical frame
- different main target entity or family role
- semantic error if retrieved as a positive

#### B. Wrong-time negatives

Goal:

- preserve event and participants
- change time, frequency, or temporal status

LLM task:

- transform the same event into a past / present / recurring / planned / cancelled alternative while keeping language highly similar

Judge criteria:

- same event family
- different retrieval-relevant temporal interpretation
- not merely a paraphrase

#### C. Query-document pairs

Goal:

- produce asymmetric retrieval supervision for short user-style queries against fuller memory-style documents

LLM task:

- from unified synthetic rows, generate:
  - a query-like user utterance
  - a matching memory/document text
  - one or more hard distractors

Preferred source families:

- `query_memory` ↔ `log_memory`
- `reflect` ↔ memory-like summaries
- `set_reminder` ↔ reminder/memory document text
- `seek_advice` ↔ relevant declarative memory text

Judge criteria:

- query would realistically retrieve the paired document
- document is not merely lexical overlap; it must be semantically useful

#### D. Safety-sensitive hard negatives

Goal:

- prevent dangerous semantic collapse in retrieval space

LLM task:

- generate near-neighbor negatives that are lexically or emotionally similar but differ in safety implication

Examples:

- needing space vs wanting to disappear
- anger venting vs intent to harm
- sadness vs self-harm implication

Judge criteria:

- high lexical or topical similarity
- materially different safety interpretation
- retrieval should treat them as distinct

#### E. Emotion-sensitive hard negatives

Goal:

- teach the retriever that emotional nuance matters, not just topic overlap

LLM task:

- keep the same family situation but alter the emotional reading:
  - worried -> angry
  - ashamed -> sad
  - grateful -> relieved

Judge criteria:

- same broad situation
- different emotional meaning relevant to retrieval

#### F. Gold evaluation regeneration

Goal:

- rebuild `data/familyos/embeddings/gold` if the older gold set is missing or incomplete

LLM task:

- produce candidate evaluation triplets or query-document labels from trusted seeds
- attach rationale for relevance and negative-type choice

Human role:

- final acceptance must remain human-reviewed
- LLM proposes; human approves or edits

### 2.10 Generator prompt design principles

To avoid shallow or repetitive outputs, prompts should require:

- explicit event identity preservation when generating positives
- explicit mismatch dimension when generating negatives
- rationale fields explaining why positive/negative is correct
- natural FamilyOS-style language rather than template spam
- balanced persona and domain coverage

Recommended prompt fields per generated record:

- `anchor`
- `positive` or `document`
- `negative` if applicable
- `negative_type`
- `source_ids`
- `source_file`
- `generator_rationale`
- `judge_verdict`
- `judge_rationale`
- `slice_tags`

### 2.11 Recommended output artifacts

Use separate output folders so mining products remain auditable:

| Artifact family | Recommended location |
| --- | --- |
| LLM-mined query-document pairs | `data/familyos/embeddings/mined_v2/query_doc/` |
| Wrong-person negatives | `data/familyos/embeddings/mined_v2/wrong_person/` |
| Wrong-time negatives | `data/familyos/embeddings/mined_v2/wrong_time/` |
| Safety/emotion negatives | `data/familyos/embeddings/mined_v2/safety_emotion/` |
| Gold regeneration candidates | `data/familyos/embeddings/gold_candidates/` |
| Failure-case benchmark candidates | `data/familyos/benchmarks/failure_cases_candidates/` |

### 2.12 Quality control for LLM-generated data

Every LLM-created record should pass all of the following before training use:

1. generator output passes schema validation
2. judge model marks it valid
3. duplicate and near-duplicate checks pass
4. slice balance quotas are not exceeded
5. spot-check human audit passes sampled review

For gold or benchmark data, require an additional human validation pass.

### 2.13 Later-stage data sources

#### Production-derived anonymized retrieval judgments

If policy allows, this is the most valuable later-stage signal.

Recommended use:

- mine successful retrievals
- mine failed retrievals
- collect implicit or explicit relevance labels
- anonymize and strip sensitive user content per policy requirements

These judgments should become the highest-priority real-world evaluation set after policy clearance.

#### Human adjudicated failure-case benchmark slices

Maintain a curated benchmark of:

- known false positives
- known false negatives
- safety collisions
- role confusions
- temporal confusions

These should be stored as a versioned regression suite and expanded continuously.

---

## 3. Training and loss functions

### 3.1 Current working training recipe

Current bake-off recipe is strong enough as a baseline:

- frozen encoder
- train retrieval head only
- `FamilyContrastiveLoss`
- hard negatives enabled
- learnable temperature
- curriculum on hard-negative weight
- matryoshka dimensions `[768, 512, 256, 128]`

This is a solid head-selection setup.

### 3.2 Recommended final training strategy

Use a staged training plan.

#### Stage A: head bake-off and selection

Goal:

- choose the best architectural family under identical conditions

Recipe:

- frozen encoder
- joint multi-head bake-off
- core triplet data only
- select by retrieval eval, not train margin

#### Stage B: world-class retriever specialization

Goal:

- improve FamilyOS nuance without destabilizing geometry

Recipe:

- start from the winning bake-off head
- add query/document asymmetry
- add richer FamilyOS hard negatives
- add auxiliary objectives

#### Stage C: optional partial unfreezing

Goal:

- recover additional performance if head-only tuning saturates

Recipe:

- unfreeze only top encoder blocks or adapters
- keep lower layers frozen
- use much smaller encoder LR than head LR

### 3.3 Recommended loss design

Do not rely only on plain InfoNCE long-term.

Recommended loss family:

$$
L = \lambda_1 L_{retrieval} + \lambda_2 L_{semantic} + \lambda_3 L_{role} + \lambda_4 L_{temporal} + \lambda_5 L_{safety} + \lambda_6 L_{confidence}
$$

#### Loss terms

1. **Retrieval loss**
   - `FamilyContrastiveLoss`
   - main objective
   - uses hard negatives and in-batch negatives

2. **Semantic regularization loss**
   - pairwise similarity or ranking loss on STS/NLI-style data
   - prevents overly narrow domain overfitting

3. **Role consistency loss**
   - penalize collapse between wrong-person or wrong-relation pairs
   - can be implemented as contrastive sub-objective on curated role negatives

4. **Temporal separation loss**
   - same topic, wrong time should not collapse
   - use curated temporal negatives

5. **Safety separation loss**
   - keep unsafe/benign near-miss phrases separable where necessary
   - especially important for high-risk phrasings

6. **Confidence calibration loss**
   - optional auxiliary loss for confidence head
   - teaches uncertainty estimation from ambiguity/hardness labels

### 3.4 Recommended sampling strategy

Each batch should mix:

- standard FamilyOS triplets
- hard negatives
- role confusion slices
- temporal confusion slices
- safety confusion slices
- optional generic semantic pairs

Avoid letting one slice dominate the whole batch distribution.

### 3.5 Curriculum

Recommended curriculum:

1. easy semantic positives + standard negatives
2. hard negatives
3. role and temporal confusions
4. safety confusions
5. asymmetric query-document training

This gives cleaner optimization than starting with every hard case at once.

### 3.6 Hyperparameter guidance

#### Head-only phase

- head LR: around current bake-off scale (`2e-4` range)
- temperature LR: low (`1e-3` scale or below)
- encoder LR: `0`
- matryoshka retained

#### Partial unfreeze phase

- encoder LR: at least 10x to 50x lower than head LR
- strong early stopping
- gold benchmark required before promotion

### 3.7 Model-selection rule

Select checkpoints by this priority order:

1. best FamilyOS retrieval margin on full eval
2. best hard-negative accuracy
3. best golden retrieval metrics
4. no unacceptable regression on safety/role/temporal slices

Do **not** select by train margin.

---

## 4. Evals

### 4.1 Evaluation philosophy

The system should be evaluated as a retrieval model, not just as a sentence similarity model.

That means we need both:

- generic embedding evals
- FamilyOS-specific retrieval evals

### 4.2 Required evaluation tracks

#### Track A: generic semantic quality

Purpose:

- ensure the embedding space is not semantically broken

Examples:

- STS Benchmark
- SICK-R
- STS12/13/14

Metrics:

- Spearman / Pearson
- pair ranking quality

#### Track B: FamilyOS retrieval

Purpose:

- measure actual memory/document retrieval usefulness

Metrics:

- Recall@1
- Recall@5
- Recall@10
- MRR
- nDCG@10

#### Track C: role sensitivity

Purpose:

- detect wrong-person or wrong-relation retrieval collisions

Metrics:

- role confusion@1
- role confusion@10
- wrong-person nearest-neighbor rate

#### Track D: temporal sensitivity

Purpose:

- detect same-event-but-wrong-time collapse

Metrics:

- temporal confusion@1
- temporal confusion@10
- wrong-time retrieval rate

#### Track E: safety distinction

Purpose:

- ensure benign and dangerous near-neighbors are not collapsed inappropriately

Metrics:

- safety collision rate
- unsafe nearest-neighbor rate
- red/amber confusion rate

#### Track F: confidence and calibration

Purpose:

- measure whether uncertainty signal is useful

Metrics:

- ECE or bucketed calibration error
- confidence-vs-accuracy curve
- retrieval abstention/rerank trigger quality

### 4.3 Evaluation slices that must exist

Every full evaluation should report separate slices for:

- kinship / family role
- self vs other
- emotion type
- emotion intensity
- temporal framing
- safety level
- short query vs long document

### 4.4 Promotion gates

Promote a model only if it satisfies all of the following:

- beats `mean_baseline` on retrieval margin
- beats `mean_baseline` on hard-negative accuracy
- no regression on safety collision rate
- no regression on role confusion metrics
- no regression on temporal confusion metrics
- improves golden retrieval benchmark

---

## 5. Benchmarking and golden data

### 5.1 Benchmarking philosophy

We need a benchmark that actually matches FamilyOS retrieval behavior.

The benchmark should measure whether the model retrieves the right memory or document when the query is:

- short
- vague
- emotionally loaded
- family-specific
- temporally specific
- safety-relevant

### 5.2 Benchmark suite definition

The benchmark suite should have five parts.

#### Suite 1: retrieval gold benchmark

Format:

- query
- candidate pool
- one or more relevant documents
- graded relevance when possible

Use for:

- Recall@k
- MRR
- nDCG

#### Suite 2: hard-negative challenge set

Format:

- query
- positive
- one or more near-miss negatives
- negative type labels

Negative types:

- wrong person
- wrong relation
- wrong time
- same topic different event
- safety mismatch
- emotion mismatch

#### Suite 3: semantic nuance set

Format:

- short pairs or triplets focusing on subtle differences

Examples:

- same event, different emotional framing
- same entity, different relation direction
- same wording, different safety implication

#### Suite 4: production-style retrieval set

Format:

- messy user query
- realistic memory/document candidates
- human relevance judgments

#### Suite 5: adversarial regression set

Format:

- known failure cases from prior models
- manually curated edge cases

Purpose:

- prevent regressions after architecture or data changes

### 5.3 Golden data creation process

#### Step 1: define schema

Each golden retrieval example should contain at least:

| Field | Description |
| --- | --- |
| `query_id` | Unique query id |
| `query` | Query text |
| `candidate_id` | Candidate memory/document id |
| `candidate_text` | Candidate text |
| `label` | Relevant / not relevant / graded relevance |
| `slice_tags` | Tags such as `role`, `temporal`, `safety`, `emotion`, `query_doc` |
| `difficulty` | Easy / medium / hard |
| `negative_type` | Optional, for hard negatives |
| `notes` | Annotation rationale |

#### Step 2: define annotation slices

Create balanced golden sets for:

- role confusion
- temporal confusion
- emotion nuance
- safety nuance
- general memory retrieval
- short-query to long-memory retrieval

#### Step 3: seed with mined candidates

Start from:

- existing silver synthetic triplets
- existing hard negatives
- mined query-document pairs
- unified synthetic outputs

Then manually validate and correct.

#### Step 4: human review

Every golden example should be reviewed for:

- correctness of positive label
- correctness of hard negative label
- realism for FamilyOS use case
- absence of duplicate or contradictory entries

#### Step 5: split benchmark properly

Create separate:

- train-support mining pool
- dev / model-selection benchmark
- final holdout benchmark

Never leak the holdout golden set into training.

### 5.4 Initial golden data volume targets

Recommended minimum targets:

- 300 retrieval queries for dev
- 300 retrieval queries for holdout
- 100 examples each for role, temporal, safety, and emotion hard slices
- 100 adversarial regression examples

This is enough to start making trustworthy promotion decisions.

### 5.5 Benchmark outputs to persist

For every evaluated model version, save:

- overall leaderboard metrics
- per-slice metrics
- confusion/error buckets
- top failure examples
- nearest-neighbor audit samples
- model version + data version + config version

### 5.6 Golden benchmark acceptance criteria

A new model should beat the current production candidate on:

- overall retrieval Recall@k / MRR / nDCG
- role slice
- temporal slice
- safety slice

and must not introduce unacceptable new failure patterns.

---

## 6. Recommended phased plan

### Phase 1: finish bake-off and lock winner

- complete current joint bake-off
- choose winner by full eval and hard-negative performance
- current likely winner: `agreement_gated`

### Phase 2: build AgreementGatedHeadV2

- vector gate
- token salience features
- role-aware and temporal-aware views
- query/document asymmetry
- confidence head

### Phase 3: upgrade data

- mine wrong-person and wrong-time negatives
- create golden retrieval set
- create safety and emotion nuance benchmark slices

### Phase 4: upgrade training objectives

- add multi-objective loss mix
- add semantic regularization
- add role/temporal/safety auxiliary losses

### Phase 5: production benchmark and promotion

- benchmark on holdout golden set
- compare against baseline and prior winning head
- promote only if all gates pass

---

## 7. Concrete next actions

1. Finalize current bake-off leaderboard and declare winner.
2. Write `AgreementGatedHeadV2` technical spec from Section 1.5.
3. Create a golden benchmark schema and annotation template.
4. Build role / temporal / safety hard-negative slices.
5. Add benchmark reporting to the training/eval pipeline.

---

## 8. Bottom line

The right way to define the FamilyOS retrieval plan is:

- **Architecture**: mean-anchor gated retriever, evolving to `AgreementGatedHeadV2`
- **Data**: use current 304,750 triplets now, but add gold retrieval data and targeted FamilyOS hard negatives
- **Training**: start with `FamilyContrastiveLoss`, then move to a multi-objective retrieval loss stack
- **Evals**: measure retrieval, role, temporal, safety, and calibration — not just cosine margin
- **Benchmarking**: create a FamilyOS golden retrieval benchmark with curated hard slices and strict holdout discipline

That is the shortest credible path from a good bake-off result to a production-grade, world-class FamilyOS retrieval system.
