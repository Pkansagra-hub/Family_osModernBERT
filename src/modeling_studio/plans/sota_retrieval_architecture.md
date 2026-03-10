# FamilyOS Retrieval Embedding Architecture Draft

> Base model: `answerdotai/ModernBERT-base`
> Scope: Retrieval-native single-output embedding architecture with a unified design that does not break existing FamilyOS interfaces
> Updated: March 10, 2026
> Status: Draft for architecture review before implementation

---

## 1. Problem Statement

The current `EmbeddingHead` in `src/modeling_studio/models/heads.py` mixes together two separate concerns:

1. sequence pooling
2. representation transformation

In `pooling="mean"`, the head behaves like a lightweight readout over the encoder manifold.
In `pooling="attentive"`, the head becomes a heavy learned embedding model with:

- latent queries
- cross-attention
- FFN refinement
- residual blending
- LayerNorm
- SwiGLU projection

This architecture improved pairwise semantic similarity metrics but degraded retrieval ranking quality.

The benchmark evidence shows that the encoder manifold is already strong for retrieval, while the current attentive head is better for STS-style semantic similarity.

---

## 2. Evidence From Current Benchmarks

### 2.1 STS side-by-side benchmark

Source artifact: `outputs/mteb_sts_encoder_mean_vs_head.json`

| Method | Avg Spearman | Avg Pearson |
|--------|--------------|-------------|
| `encoder_mean` | 0.5279 | 0.5105 |
| `embedding_head` | 0.5527 | 0.5389 |

Observations:

- `embedding_head` wins on average across STSBenchmark, STS12, STS13, STS14
- `encoder_mean` still wins on STS15, STS16, and SICK-R
- `encoder_mean` cosine values are tightly compressed near 0.90-0.94, indicating anisotropic geometry
- `embedding_head` produces a much broader cosine spread, helping pairwise similarity correlation

### 2.2 Retrieval side-by-side benchmark

Source artifact: `outputs/retrieval_probe_encoder_mean_vs_head.json`

| Method | Recall@1 / 10d | Recall@1 / 100d | Recall@5 / 100d |
|--------|----------------|-----------------|-----------------|
| `encoder_mean` | 0.8450 | 0.7500 | 1.0000 |
| `embedding_head` | 0.5500 | 0.3833 | 0.8000 |

Observations:

- `encoder_mean` is materially better for retrieval ranking
- `embedding_head` has larger similarity margins but worse nearest-neighbor ranking
- current attentive refinement is likely improving pairwise calibration while distorting neighborhood structure

### 2.3 Architectural conclusion from the evidence

The current encoder is not the bottleneck for retrieval.
The current embedding head is not retrieval-native.
A true retrieval-first design should preserve the encoder manifold and add learned refinement in a controlled residual way.

---

## 3. Hard Compatibility Constraint

The embedding redesign must not break the existing FamilyOS system.

This is a strict requirement, not a preference.

The new design must preserve all of the following:

- one embedding output tensor only
- same `capability="embedding"` path
- same `Client.get_embedding()` contract
- same shape and normalized dense vector behavior expected by downstream FamilyOS code
- checkpoint loading semantics that remain backward-compatible

This removes several otherwise-attractive research directions from the first implementation phase:

- dual embedding outputs
- token-level retrieval outputs in the production embedding API
- separate retrieval and similarity APIs
- late-interaction-only production scoring

Therefore, the target is not a multi-output research architecture.
The target is the strongest possible single-output retrieval-native embedding head that remains a drop-in replacement.

---

## 4. Design Goals

The new architecture should satisfy all of the following:

1. Preserve the strong retrieval geometry already present in mean-pooled encoder representations
2. Support STS-style semantic similarity without collapsing retrieval quality
3. Remain backward-compatible with the current FamilyOS single-output embedding interface
4. Support matryoshka dimensions and export-friendly inference paths
5. Allow multiple candidate heads to be trained under the same script and compared fairly
6. Avoid forcing one heavy nonlinear projection path for every embedding use case

---

## 5. Non-Goals

This draft does not attempt to:

- maximize only STS correlation at the cost of retrieval
- rely solely on a more complex pooling block to achieve SOTA
- claim world-best performance without retrieval-native training data and hard-negative mining
- break the FamilyOS embedding API to introduce multiple embedding outputs
- replace the production embedding stack with a reranker-only design

---

## 6. Proposed SOTA-Oriented Architecture

## 6.1 High-level recommendation

Adopt a retrieval-native single-output head that is anchored on `encoder_mean` and allows only controlled residual deviation from that manifold.

The head should keep the existing external interface of `EmbeddingHead`, but internally follow a retrieval-first design.

The core principle is:

> The encoder mean representation is the anchor. Every learned improvement must be residual, bounded, and ablatable.

This is the best path toward a strong single-vector retrieval model without breaking production behavior.

---

## 6.2 Retrieval-safe base embedding

The new head should always compute a masked mean base embedding first.

### Base representation

$$
e_{mean} = \operatorname{normalize}(\operatorname{MeanPool}(H, M))
$$

Where:

- $H \in \mathbb{R}^{B \times L \times D}$ are encoder hidden states
- $M$ is the attention mask

This base representation is not a fallback. It is the canonical retrieval manifold.

---

## 6.3 Residual refinement rule

Every learned component must modify the base only through a residual path.

Let a learned auxiliary pooling or refiner produce:

$$
e_{aux} = f_{latent}(H, M)
$$

$$
\Delta = e_{aux} - e_{mean}
$$

Then the mixed representation is:

$$
e_{mix} = \operatorname{normalize}(e_{mean} + \alpha \cdot \Delta)
$$

Where:

- $f_{latent}$ is a lightweight latent attention or learned pooling function
- $\alpha$ is a learnable scalar or vector initialized near $0$

This formulation is safer than directly replacing mean pooling because the model learns how far to move away from the known-good retrieval manifold.

---

## 6.4 Residual projection refinement

If a learned projection is needed, it must also be residual:

$$
e_{dense} = \operatorname{normalize}(e_{mix} + \beta g(\operatorname{LN}(e_{mix})))
$$

Where:

- $g$ is a small residual MLP, gated projection, or diagonal re-scaling block
- $\beta$ is initialized near $0$

This keeps the encoder manifold as the primary geometry and allows learned improvements only when they actually help.

---

## 6.5 Novel idea: agreement-gated residual refinement

The novel part of this draft is not "more attention". It is controlled refinement based on whether learned views agree with the encoder anchor.

Define multiple lightweight views:

- $e_{mean}$: masked mean pool
- $e_{cls}$: CLS pooled view
- $e_{lat}$: latent attention pooled view
- $e_{max}$: masked max or top-k mean pooled view

Compute agreement features such as:

- cosine$(e_{mean}, e_{lat})$
- cosine$(e_{mean}, e_{cls})$
- token salience entropy from the latent pooler
- embedding norm statistics before normalization

Use these to predict a bounded gate:

$$
\alpha = \sigma(\text{GateMLP}([e_{mean}; e_{lat}; e_{cls}; s]))
$$

Where $s$ is a compact vector of agreement statistics.

Then apply only a bounded residual update:

$$
e_{out} = \operatorname{normalize}(e_{mean} + \alpha \odot (e_{lat} - e_{mean}))
$$

This is novel relative to the current head because:

- the model learns when the refined view is trustworthy
- the update is relative to the mean anchor, not a free replacement
- the gate can be scalar, vector, or low-rank without changing the output contract

This gives the head room to become more expressive without paying the usual retrieval penalty of unbounded nonlinear transformation.

---

## 6.6 Single output contract

The head must still emit exactly one embedding tensor.

Required output contract:

```python
embedding: Tensor  # shape [batch_size, output_dim]
```

Internally the head may compute multiple views, but externally it must return a single normalized vector exactly as the current FamilyOS system expects.

---

## 7. Candidate heads to train under the same script

Rather than betting on one idea immediately, the best research strategy is to implement 5-6 candidate heads behind a common config surface and train them under the same script.

All candidates must share:

- same trainer
- same data mixture
- same losses
- same optimizer schedule
- same batch size and hard-negative settings
- same evaluation gates

This turns the problem into a controlled head bake-off instead of architecture guesswork.

### Head A: `MeanBaselineHead`

Definition:

- masked mean pool
- optional output projection only if output dim differs
- normalize

Purpose:

- baseline retrieval anchor
- must always be included in experiments

### Head B: `ResidualMLPMeanHead`

Definition:

- base = mean pool
- residual MLP refiner with zero-init residual scale
- normalize

Purpose:

- simplest backward-compatible learned improvement over `encoder_mean`

### Head C: `LatentResidualHead`

Definition:

- base = mean pool
- auxiliary = latent attention pooling
- scalar residual gate initialized near zero
- optional tiny residual MLP

Purpose:

- test whether latent attention helps when forced to stay close to the retrieval anchor

### Head D: `AgreementGatedHead`

Definition:

- base = mean pool
- auxiliary views = mean, CLS, latent attention
- gate network uses agreement statistics to decide residual update strength
- single final embedding only

Purpose:

- novel retrieval-first design
- allows conditional refinement instead of always-on refinement

### Head E: `MultiPoolLowRankHead`

Definition:

- base = mean pool
- combine mean, CLS, max, and latent views using low-rank learned mixing coefficients
- residual update only

Purpose:

- test whether multiple cheap views provide complementary signal without heavy nonlinear overwrite

### Head F: `AnisotropyCorrectedHead`

Definition:

- base = mean pool
- learned centering, diagonal scaling, and optional low-rank whitening-style correction
- residual formulation

Purpose:

- explicitly target the compressed cosine geometry observed in `encoder_mean`
- improve retrieval ranking without over-warping the manifold

---

## 8. Why the Current `EmbeddingHead` Falls Short

The current attentive head is elegant as a semantic pooling module, but not as a retrieval-native embedding architecture.

Specific issues:

1. `attentive` is treated as a pooling strategy, but it is actually a separate representation model
2. attentive mode always adds `LayerNorm + GatedProjection`, while simple modes do not
3. projection behavior is tied to pooling choice rather than task need
4. the model has no explicit mechanism to preserve the mean-pooled encoder manifold
5. the model has no explicit agreement mechanism to decide when refinement should be trusted

The result is a head that can improve pairwise STS behavior while damaging retrieval neighborhoods.

---

## 9. Recommended class structure

A cleaner implementation should separate the following responsibilities.

### 7.1 Pooling

- `MaskedMeanPooler`
- `ClsPooler`
- `LatentAttentionPooler`

### 9.2 Refinement

- `IdentityRefiner`
- `ResidualMLPRefiner`
- `ResidualGatedRefiner`
- `AgreementGateRefiner`
- `AnisotropyCorrectionRefiner`

### 9.3 Retrieval head family

- `EmbeddingHead` (backward-compatible external API)
- internal implementations selected by config:
  - `MeanBaselineHead`
  - `ResidualMLPMeanHead`
  - `LatentResidualHead`
  - `AgreementGatedHead`
  - `MultiPoolLowRankHead`
  - `AnisotropyCorrectedHead`

### 9.4 Compatibility wrapper

- keep `EmbeddingHead` as the public class name and config entry point
- swap internals via an `embedding_architecture` config field

---

## 10. Training strategy required for SOTA ambition

A better head alone will not produce world-class retrieval. The training recipe is equally important.

### 10.1 Shared training recipe across candidate heads

All 5-6 candidate heads should be trained with the same script and same experiment harness.

Required shared controls:

- same seed set
- same training dataset mixture
- same hard negatives per batch
- same loss weights
- same optimizer and scheduler
- same evaluation cadence
- same checkpoint export path format

This is necessary so differences can be attributed to head design rather than run configuration drift.

### 10.2 Single-output dense training

Use a combination of:

- in-batch contrastive loss
- hard negative ranking loss
- margin-based retrieval loss
- optional matryoshka loss
- optional teacher distillation from a strong retrieval model

### 10.3 Data requirements

Need a mixture of:

- public retrieval pairs
- FamilyOS memory retrieval pairs
- synthetic query reformulations
- hard negatives sampled from semantically adjacent family/work/health/planning domains

### 10.4 Key training principle

Do not optimize only for STS or only for pairwise cosine regression.
Retrieval ranking must be a first-class objective.

---

## 11. Experimental protocol: 6-head bake-off

The recommended research workflow is:

1. add `embedding_architecture` to the training config
2. instantiate one of 5-6 candidate heads from the same script
3. train all candidates under identical settings
4. compare them on the same benchmark suite
5. promote only the best head to the default production architecture

Suggested config values:

```yaml
heads:
  embedding:
    enabled: true
    type: embedding
    embedding_architecture: agreement_gated  # one of 6 candidates
    pooling: mean
    normalize: true
```

Suggested experiment matrix:

| ID | Head | Purpose |
|----|------|---------|
| E0 | `mean_baseline` | must-have control |
| E1 | `residual_mlp_mean` | simplest learned residual |
| E2 | `latent_residual` | bounded attentive refinement |
| E3 | `agreement_gated` | novel candidate |
| E4 | `multi_pool_low_rank` | multi-view residual fusion |
| E5 | `anisotropy_corrected` | cosine-geometry repair |

For each candidate, record:

- Recall@1 / 10 distractors
- Recall@1 / 100 distractors
- Recall@5 / 100 distractors
- MTEB STS average
- latency and throughput
- embedding norm distribution
- cosine similarity histogram statistics

---

## 12. Evaluation plan

The architecture should not be accepted until it improves the right metrics.

### 12.1 Retrieval gates

Must compare against `encoder_mean` and current `embedding_head` on:

- Recall@1 / 10 distractors
- Recall@1 / 100 distractors
- Recall@5 / 100 distractors
- MTEB STS subset
- latency and throughput

### 12.2 Acceptance criteria for phase 1

For the first implementation phase, a candidate head should achieve at least:

- retrieval recall no worse than current `encoder_mean`
- STS average no worse than 90% of current `embedding_head`
- no large collapse in cosine geometry or matryoshka truncation quality
- no API or output shape breakage for FamilyOS consumers

---

## 13. Migration strategy

### Phase 0

Create the new architecture doc and benchmark baselines.

### Phase 1

Implement the 5-6 candidate single-output heads behind a shared config entry.

### Phase 2

Train all candidate heads under the same script and collect benchmark results.

### Phase 3

Select the best candidate and harden checkpoint compatibility.

### Phase 4

Retrain the selected head on retrieval-native data with hard negatives.

### Phase 5

Make the selected single-output head the default only after it surpasses `encoder_mean` on retrieval benchmarks and preserves FamilyOS compatibility.

---

## 14. Proposed API direction

No public API break is proposed in the first architecture phase.

The external API should remain:

```python
client.get_embedding(text)
```

For lower-level model code, `capability="embedding"` should still yield one embedding vector.

Any new internal head choices must be hidden behind config and checkpoint metadata, not a breaking API change.

---

## 15. Risks and trade-offs

### Risk: architecture complexity grows too fast

Mitigation: use the 6-head bake-off to identify the smallest head that wins.

### Risk: dense branch becomes too similar to current mean pooling

Mitigation: add only residual refinements that beat the baseline in controlled evaluation.

### Risk: overfitting to STS again

Mitigation: treat retrieval ranking metrics as primary acceptance gates.

### Risk: a novel head looks elegant but is unstable to train

Mitigation: require zero-init or near-identity initialization for all residual gates and compare against the baseline under the same script.

---

## 16. Bottom-line recommendation

If the goal is an elegant unified architecture with true SOTA headroom:

- do not keep evolving the current attentive pooling path as the sole embedding strategy
- redesign around a retrieval-safe single-output head anchored on `encoder_mean`
- add learned refinement only as residual improvement
- test 5-6 candidate heads under the same script instead of betting on one idea upfront

In short:

1. preserve the encoder manifold
2. make refinement residual
3. separate pooling from transformation
4. compare multiple candidate heads under identical training
5. keep the FamilyOS single-output contract intact

---

## 17. Immediate next step

Create an implementation plan for the single-output candidate-head experiment covering:

- class refactor in `src/modeling_studio/models/heads.py`
- head registry/config support for 5-6 embedding architectures
- output contract changes in `modernbert_multitask.py`
- retrieval-native training losses
- benchmark gates
- backward compatibility with current checkpoints
