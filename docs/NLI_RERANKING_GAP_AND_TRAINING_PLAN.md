# UltraBERT NLI Re-Ranking Gap Analysis and Training Plan

**Status**: Research finding + training roadmap
**Context**: Episode retrieval research Phase 7 proved NLI re-ranking is BLOCKED
**Model**: FamilyOS UltraBERT v4.0.7 (ModernBERT-base, 22 layers, 149M params, 768-dim)

---

## 1. The Problem: What Is Lacking

### 1.1 Current NLI Head Architecture

UltraBERT v4 has an `NLIHead` (in `src/modeling_studio/models/heads.py:801`) that
classifies into 3 labels: `entailment`, `neutral`, `contradiction`. It was trained
during Stage A on MNLI + SNLI (generic NLI corpora) and replayed at 0.3 weight
during Stage B (FamilyOS domain adaptation).

The head inherits from `SequenceClassificationHead` and supports an optional
`CrossAttentionPairEncoder` (Epic 5.0) for bidirectional cross-attention between
premise and hypothesis. The pair encoder exists in code (`pair_encoder.py`) but
is NOT wired through the inference client.

### 1.2 What the Client Actually Exposes

The production `Client` class exposes:

| Method | Type | Pair Input? | Cross-Attention? |
|-|-|-|-|
| `analyze(text)` | Single-text | NO | NO |
| `get_embedding(text)` | Single-text | NO | N/A |
| `get_query_embedding(text)` | Single-text | NO | N/A |
| `get_document_embedding(text)` | Single-text | NO | N/A |
| `similarity(text1, text2)` | Two texts | YES (bi-encoder cosine) | NO |
| `find_similar(query, corpus)` | One-to-many | YES (bi-encoder cosine) | NO |

**Critical finding from S7-1 probe:**

- `client.analyze(text, hypothesis="...")` raises `TypeError: unexpected keyword argument`
- There is NO `client.nli(premise, hypothesis)` method
- There is NO `client.get_nli_score(premise, hypothesis)` method
- The `nli` field in `analyze()` output is a **single-text classification** (always returns "neutral" for standalone text)

### 1.3 Why similarity() Cannot Substitute for NLI Re-Ranking

The S7-2 experiment tested `similarity()` (bi-encoder cosine) as a fallback:

| Metric | Value | Required | Verdict |
|-|-|-|-|
| Spearman correlation (sim vs relevance) | 0.2047 | > 0.50 | FAIL |
| Pearson correlation | 0.2170 | > 0.50 | FAIL |
| AUC-ROC | 0.6919 | > 0.70 | FAIL (marginal) |
| Mean similarity (relevant episodes) | 0.4584 | -- | -- |
| Mean similarity (irrelevant episodes) | 0.3478 | -- | -- |
| Separation (relevant - irrelevant) | 0.1106 | > 0.20 | FAIL |

Per-type separation reveals the core issue:

| Query Type | Separation | Assessment |
|-|-|-|
| temporal | 0.0064 | Useless -- bi-encoder cosine cannot distinguish temporal relevance |
| emotional | 0.0935 | Weak |
| entity | 0.0952 | Weak |
| causal | 0.1001 | Weak |
| cross_episode | 0.1116 | Weak |
| thematic | 0.1142 | Marginal |

The fundamental issue: **bi-encoder cosine compares topic similarity, not relevance**.
A cross-encoder (or NLI head) reads `[query + episode]` jointly with full
bidirectional attention, enabling it to reason about whether the episode ANSWERS
the query -- not just whether they share topic words.

### 1.4 Specific Failures Observed in S7-1

The sanity test revealed a critical flaw:

```
Pair 5 (keyword match, semantically wrong):
  Query:   "Maya eating at a restaurant"
  Episode: "Maya watched a cooking show about restaurants on TV."
  Expected: LOW
  Actual:   0.8537 (HIGHEST of all pairs)
```

The bi-encoder gave the HIGHEST score to a keyword-matching but semantically
irrelevant pair. A cross-encoder NLI head would read both texts jointly and
recognize that watching a cooking show is NOT eating at a restaurant.

### 1.5 The Three Missing Capabilities

| # | Missing Capability | Why It Matters |
|-|-|-|
| 1 | **Pair-input NLI inference API** | The NLI head exists in the model but is unreachable from the client. No method accepts (premise, hypothesis) and returns entailment/neutral/contradiction scores. |
| 2 | **Cross-encoder relevance scoring** | The NLI head uses the standard `[CLS] premise [SEP] hypothesis [SEP]` concatenation, which gives cross-attention over the joint sequence. But without a client method to invoke it, and without fine-tuning on relevance data, it only classifies NLI -- not retrieval relevance. |
| 3 | **Relevance-trained NLI weights** | Even if the API were wired, the NLI head was trained on MNLI/SNLI (generic textual entailment). "Does this episode entail this query?" is not the same as "Is this episode relevant to this query?". The head needs fine-tuning on retrieval relevance data. |

---

## 2. Architecture: What Needs to Change

### 2.1 Current Model Flow (Single-Text)

```
text -> Tokenizer -> [CLS] tokens [SEP] -> ModernBERT (22 layers) -> NLIHead -> 3 logits
                                                                         |
                                                               (no pair cross-attention,
                                                                just CLS classification)
```

### 2.2 Target Model Flow (Cross-Encoder Pair Scoring)

```
(query, episode_text) -> Tokenizer -> [CLS] query [SEP] episode_text [SEP]
                                              |
                                     ModernBERT (22 layers)
                                     (FULL cross-attention between query and episode tokens)
                                              |
                                     CrossAttentionPairEncoder (already in pair_encoder.py)
                                     (bidirectional cross-attention, residual, feedforward)
                                              |
                                     RelevanceHead (new)
                                     (binary: relevant/irrelevant, or regression: 0.0-1.0)
                                              |
                                     relevance_score
```

### 2.3 Components That Already Exist

| Component | File | Status |
|-|-|-|
| ModernBERT encoder (22 layers) | `modernbert_multitask.py` | Production |
| CrossAttentionPairEncoder | `pair_encoder.py` | Code exists, not production-tested for relevance |
| NLIHead with pair_encoder support | `heads.py:801` | Code accepts pair_encoder arg, never invoked with one |
| Pair tokenization in inference | `unified_output.py:996` | Exists for NLI: `tokenizer(premises, hypotheses, ...)` |
| AgreementGatedHeadV2 | `heads_embedding.py:579` | Production embedding head; query/document mode prompts |

### 2.4 Components That Need to Be Built

| Component | Description | Estimated Effort |
|-|-|-|
| `RelevanceHead` | New head, replaces 3-class NLI with relevance regression or binary | Small (inherits SequenceClassificationHead) |
| `client.score_relevance(query, episode_text)` | New client method, returns float 0.0-1.0 | Small |
| `client.rerank(query, candidates)` | Batch re-ranking method, returns sorted list | Small |
| Relevance training data | (query, episode, relevance_label) triples from family data | Medium |
| Cross-encoder fine-tuning pipeline | Training loop for relevance head | Medium |
| Latency optimization | Batch pair encoding with padding | Small |

---

## 3. Training Plan: NLI to Relevance Head

### 3.1 Training Strategy Overview

Two approaches, ordered by complexity:

**Approach A: Fine-Tune Existing NLI Head on Relevance Data (Recommended)**

Reuse the existing NLI head architecture but retrain it with relevance labels.
The 3-class output (entailment/neutral/contradiction) maps naturally:

- entailment -> relevant
- neutral -> partially relevant
- contradiction -> irrelevant

This preserves the pre-trained NLI weights as initialization, which already
encode some notion of "does text B follow from text A."

**Approach B: Add a New RelevanceHead (Clean Separation)**

Add a 14th task head specifically for relevance scoring. This avoids disturbing
the existing NLI head (which may be used elsewhere in K0/K1).

### 3.2 Training Data: What Already Exists in Modeling Studio

The following datasets were found and assessed in
`D:\Modeling_studio\data\familyos\embeddings\`:

#### Dataset Inventory

| Dataset | Count | Format | Usable for Re-Ranking? |
|---|---|---|---|
| `hard_negatives/` | 42,945 | `(anchor, positive, negative)` | PARTIAL — right hard-negative types, but designed for embedding training |
| `silver_synthetic/` | 261,805 | `(anchor, positive, negative)` | NO — cross-cluster easy negatives, too easy |
| `mined_v2/query_doc` | 10,019 | `(query, document)` | YES — Gemini-generated positive pairs, use as grade=1 positives |
| `mined_v2/wrong_time` | 10,019 | `(anchor, positive, negative)` | YES HIGH VALUE — temporal shift hard negatives |
| `mined_v2/wrong_person` | 12,041 | `(anchor, positive, negative)` | YES HIGH VALUE — entity-swap hard negatives |
| `mined_v2/safety_emotion` | 10,037 | `(anchor, positive, negative)` | PARTIAL — safety/emotion near-misses |

#### Why silver_synthetic Must Be Excluded

silver_synthetic (261K) uses cross-cluster negatives: anchor = memory about grandma,
negative = memory about household chores. The bi-encoder already handles cross-cluster
discrimination — that is exactly what produced AUC=0.6919 in S7-2. Re-training the
cross-encoder on easy cross-cluster negatives wastes capacity and teaches nothing new.

#### Why mined_v2/wrong_time and wrong_person Are the Most Valuable

These map directly to the Phase 7 failure modes:

- `wrong_time`: temporal shift negatives ("every Sunday" vs "next Sunday",
  "today" vs "yesterday") — targets temporal separation=0.0064 (worst query type)
- `wrong_person`: entity-swap negatives ("Dadi" vs "Nani", "Bhai" vs "Papa") —
  targets entity confusion at kinship level
- Both are rated `difficulty: hard` by the Gemini generator

#### Format Gap: No Listwise Data for LambdaRank

ALL existing datasets are binary pairwise triplets (positive/negative), not graded
listwise groups. LambdaRank requires per-query groups with grades 0-3:

```
Query → [(ep1, grade=3), (ep2, grade=1), (ep3, grade=0), ...]
```

The human benchmark (50 queries × 88 episodes, grades 0-3) is the **only** source
of graded listwise labels. LambdaRank applies only to those ~4,400 pairs.
For all mined/hard-negative data, the loss must be **pairwise margin** instead.

#### Recommended Training Mix

| Source | Count | Loss | Weight | Notes |
|---|---|---|---|---|
| Human benchmark (listwise) | 4,400 | LambdaRank | 1.0x | Gold graded labels — primary signal |
| mined_v2/query_doc + wrong_time + wrong_person | ~32K | Pairwise margin | 0.8x | Real domain, hard negatives, matched format |
| hard_negatives/ entity_swap + temporal_shift + same_topic | ~7K | Pairwise margin | 0.6x | Good variety, embedding-format origin |
| silver_synthetic/ | 0 | SKIP | — | Too easy, wastes training capacity |
| **Total usable** | **~43K** | — | — | No MS MARCO needed — domain data is sufficient |

#### hard_negatives/ Type Distribution (shard 0, n=10K)

| Type | Count | Value for Re-Ranking |
|---|---|---|
| sentiment_flip | 2,480 | Useful |
| entity_swap | 2,320 | HIGH — participant errors |
| same_topic_different_event | 2,298 | HIGH — canonical failure mode |
| temporal_shift | 2,293 | HIGH — temporal errors |
| causality_flip | 214 | Moderate |
| negation | 208 | Moderate |
| quantifier_change | 187 | Moderate |

### 3.3 Training Configuration

```yaml
# configs/training/multitask/relevance_head.yaml

training:
  stage: "relevance_finetune"
  base_checkpoint: "checkpoints/distil_stage_b_bestema"

  # Freeze everything except the new head + pair encoder
  freeze_encoder: true
  freeze_existing_heads: true
  trainable_modules:
    - "relevance_head"
    - "pair_encoder"

  optimizer:
    type: adamw
    lr_head: 1e-4
    lr_pair_encoder: 5e-5
    weight_decay: 0.01

  scheduler:
    type: cosine_with_warmup
    warmup_ratio: 0.1
    num_epochs: 10

  data:
    # Source 1: Human benchmark — graded listwise labels for LambdaRank
    human_benchmark:
      source: local
      path: "data/familyos/embeddings/human_benchmark_relevance.jsonl"
      format: "listwise"  # (query, [(episode_text, grade), ...])
      loss: "lambda_rank"
      weight: 1.0

    # Source 2: mined_v2 positive pairs + hard negatives — pairwise margin
    mined_v2_query_doc:
      source: local
      path: "data/familyos/embeddings/mined_v2/query_doc/"
      format: "positive_pair"   # (query, document) -> grade=1
      loss: "pairwise_margin"
      weight: 0.8

    mined_v2_wrong_time:
      source: local
      path: "data/familyos/embeddings/mined_v2/wrong_time/"
      format: "triplet"         # (anchor, positive, negative)
      loss: "pairwise_margin"
      hard_negative_weight: 2.0
      weight: 0.8

    mined_v2_wrong_person:
      source: local
      path: "data/familyos/embeddings/mined_v2/wrong_person/"
      format: "triplet"
      loss: "pairwise_margin"
      hard_negative_weight: 2.0
      weight: 0.8

    hard_negatives:
      source: local
      path: "data/familyos/embeddings/hard_negatives/"
      format: "triplet"
      include_types:
        - entity_swap
        - temporal_shift
        - same_topic_different_event
        - causality_flip
      exclude_types:
        - sentiment_flip  # Overlaps with safety head, less relevant for retrieval
      loss: "pairwise_margin"
      weight: 0.6

    # silver_synthetic: EXCLUDED — cross-cluster easy negatives, not useful for re-ranking

    # MS MARCO warm-start: NOT needed — 43K FamilyOS domain triples is sufficient

  batch_size: 32
  max_length: 512  # query (~50 tokens) + episode (~400 tokens) + special tokens
  gradient_accumulation_steps: 4
```

### 3.4 Training Pipeline Steps

```
Step 1: Prepare relevance training data
  - Export human benchmark as listwise (query, [(episode_text, grade), ...]) JSONL
  - Convert mined_v2 triplets to (anchor, positive, grade=1) + (anchor, negative, grade=0)
  - Filter hard_negatives/ by include_types list
  - Split: 80% train, 10% dev, 10% holdout (stratified by source)
  - Total: ~43K pairs — no external data needed

Step 2: Initialize relevance head
  - Option A: Clone NLI head weights -> RelevanceHead
  - Option B: Fresh RelevanceHead (768 -> 256 -> 1) from scratch
  - Initialize pair_encoder from existing weights (if Epic 5.0 trained them)

Step 3: Warm-start on MS MARCO (optional, 2-3 epochs)
  - Input: [CLS] query [SEP] passage [SEP]
  - Label: binary relevant/irrelevant
  - Freeze encoder, train head + pair_encoder only
  - Monitor: AUC-ROC on MS MARCO dev

Step 4: Fine-tune on FamilyOS relevance data (5-10 epochs)
  - Input: [CLS] query [SEP] episode_text [SEP]
  - Label: graded (0-3) or binary
  - Freeze encoder, train head + pair_encoder
  - Monitor: Spearman correlation on dev set, AUC-ROC
  - EMA checkpointing for stability

Step 5: Evaluate re-ranking quality
  - Run v4 pipeline to get top-20 candidates per query
  - Re-rank using new relevance head
  - Measure: MRR lift, nDCG lift, per-type improvement
  - Target: Spearman > 0.50, AUC-ROC > 0.80

Step 6: Integrate into Client
  - Add client.score_relevance(query, episode_text) -> float
  - Add client.rerank(query, candidates, top_k) -> sorted list
  - Benchmark latency: target < 50ms for re-ranking top-20
```

### 3.5 Head Architecture Options

**Option A: Reuse NLIHead with Relevance Labels**

```python
# Minimal change: reinterpret NLI labels
# entailment -> relevant, neutral -> partial, contradiction -> irrelevant
# Use entailment logit as relevance score

class RelevanceHead(NLIHead):
    """NLI head repurposed for retrieval relevance scoring."""

    def relevance_score(self, logits: torch.Tensor) -> torch.Tensor:
        """Extract relevance score from NLI logits."""
        probs = F.softmax(logits, dim=-1)
        # Entailment probability = relevance score
        return probs[:, 0]  # entailment index
```

**Option B: Dedicated Regression Head**

```python
class RelevanceRegressionHead(BaseHead):
    """Dedicated head for relevance scoring (0.0 to 1.0)."""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__(hidden_size)
        self.dense = nn.Linear(hidden_size, 256)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(256, 1)

    def forward(self, hidden_states, attention_mask=None, labels=None,
                pair_encoder=None, text_a_hidden=None, text_b_hidden=None,
                text_a_mask=None, text_b_mask=None):

        if pair_encoder and text_a_hidden is not None and text_b_hidden is not None:
            pooled = pair_encoder(text_a_hidden, text_b_hidden, text_a_mask, text_b_mask)
        else:
            # CLS pooling from concatenated input
            pooled = hidden_states[:, 0]

        x = self.dropout(self.activation(self.dense(pooled)))
        score = torch.sigmoid(self.out(x))  # 0.0 to 1.0

        output = {"logits": score.squeeze(-1)}
        if labels is not None:
            output["loss"] = F.mse_loss(score.squeeze(-1), labels.float())
        return output
```

**Option C: Leverage AgreementGatedHeadV2 Asymmetry**

The embedding head already has query/document mode prompts. A lightweight scorer
on top of the asymmetric embeddings could work:

```python
class EmbeddingRelevanceScorer(nn.Module):
    """Score relevance using AgreementGatedHeadV2 query/document embeddings."""

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        # Input: concat(q_embed, d_embed, q_embed * d_embed, |q_embed - d_embed|)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size * 4, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, query_embed: torch.Tensor, doc_embed: torch.Tensor):
        features = torch.cat([
            query_embed, doc_embed,
            query_embed * doc_embed,
            torch.abs(query_embed - doc_embed),
        ], dim=-1)
        return self.scorer(features).squeeze(-1)
```

This avoids full cross-attention (cheaper) but may not match true cross-encoder
quality. It does leverage the learned query/document asymmetry from the
AgreementGatedHeadV2.

### 3.6 Recommendation

**Start with Option A** (reuse NLI head, cheapest):

1. Wire the existing NLI head through a new client method
2. Fine-tune on FamilyOS relevance data (no MS MARCO needed initially)
3. Measure Spearman correlation and re-ranking MRR lift
4. If Spearman > 0.50: ship it
5. If Spearman < 0.50: escalate to Option B (dedicated regression head)

**Why Option A first**: The NLI head is already trained on 392K MNLI + 550K SNLI
examples to understand textual entailment. The cross-attention pair encoder code
exists. The only missing piece is the client wiring and fine-tuning on relevance
labels. This is a 1-2 day effort.

---

### 3.7 General NLI Datasets for MGRH Pre-Training

The MGRH (Multi-Granularity Relevance Head) requires **general NLI capability**
before FamilyOS domain fine-tuning. The CrossAttentionPairEncoder has had minimal
cross-encoder specific training; it needs exposure to diverse textual entailment
patterns first.

The following datasets were assessed via their Hugging Face dataset cards.

#### Dataset Overview

| Dataset | HF ID | Size | License | Role | Status in UltraBERT |
|-|-|-|-|-|-|
| MNLI | nyu-mll/multi_nli | 392K | CC-BY-3.0 | NLI backbone | Stage A (enabled) |
| SNLI | stanfordnlp/snli | 570K | CC-BY-SA-4.0 | NLI supplemental | Stage A + B replay |
| ANLI | facebook/anli | 169K | CC-BY-NC-4.0 | Adversarial NLI | Stage A (**disabled**) |
| WANLI | alisawuffles/WANLI | 108K | CC-BY-4.0 | GPT-3+human, OOD-robust | Not in training |
| FEVER-NLI | fever/fever | 185K (v1.0) | CC-BY-SA-3.0 | Fact verification NLI | Not in training |
| MS MARCO | microsoft/ms_marco | 808K (v2.1) | MS Research | Retrieval relevance | Not in training |

---

#### ANLI — Enable Immediately (facebook/anli)

**Priority: URGENT — already in codebase config but disabled.**

ANLI was built by iteratively fooling BERT/RoBERTa-based NLI models using a
human annotation loop, producing 3 rounds of increasing difficulty:

| Round | Train | Val | Test | Notes |
|-|-|-|-|-|
| R1 | 16,946 | 1,000 | 1,000 | Hard — early NLI models fooled |
| R2 | 45,460 | 1,000 | 1,000 | Harder — stronger models fooled |
| R3 | 100,459 | 1,200 | 1,200 | Hardest — state-of-the-art models fooled |
| **Total** | **162,865** | **3,200** | **3,200** | — |

Fields: `uid`, `premise`, `hypothesis`, `label` (entailment/neutral/contradiction),
`reason` (annotator explanation — rare, useful for debugging)

ANLI is specifically designed to expose failure modes where standard MNLI/SNLI
training produces overconfident wrong predictions. MGRH will encounter FamilyOS
episodes with subtle temporal and entity traps ("same person, different day" vs
"same day, different person") — exactly the class ANLI was designed to stress.

**Action**: In `D:\Modeling_studio\configs\datasets\stage_a_datasets.yaml`, change:

```yaml
nli_anli:
  enabled: false   # CHANGE THIS
```

to:

```yaml
nli_anli:
  enabled: true
  rounds:
    - r1
    - r2
    - r3   # R3 is most valuable (100K hardest adversarial pairs)
```

**License note**: CC-BY-NC-4.0 — non-commercial use only. Verify that FamilyOS
MGRH training is not for a commercial product before enabling.

---

#### WANLI — Add to Training (alisawuffles/WANLI)

**Priority: HIGH — not in any UltraBERT stage, high return per sample.**

WANLI (Worker-AI Collaborative NLI) was constructed by:

1. Identifying "hard pockets" in MNLI using dataset cartography (high training loss variability)
2. Using GPT-3 Curie to generate new hypotheses targeting those pockets
3. Human annotators re-annotating and revising the generated pairs

Size: 107,885 rows (102,885 train / 5,000 test)
License: CC-BY-4.0 (permissive — no commercial restriction)
Inter-annotator agreement: Cohen Kappa 0.60

**Why WANLI outperforms MNLI despite being 4x smaller:**

| Evaluation Benchmark | MNLI-trained model | WANLI-trained model |
|-|-|-|
| HANS (syntactic heuristics test) | baseline | +11 points |
| ANLI (adversarial) | baseline | +9 points |
| SNLI test | competitive | competitive |

WANLI forces the model to reason about subtle semantic distinctions rather than
surface-form shortcuts. For MGRH, where distinguishing episode relevance from
keyword overlap is the core challenge, this is the right generalization pressure.

**Action**: Add a new entry to `stage_a_datasets.yaml`:

```yaml
nli_wanli:
  enabled: true
  source: huggingface
  hf_path: "alisawuffles/WANLI"
  split_train: "train"    # 102,885 rows
  split_val: "test"       # 5,000 rows
  label_map:
    entailment: 0
    neutral: 1
    contradiction: 2
  weight: 1.0
```

---

#### FEVER-NLI — Add to Training (fever/fever)

**Priority: MEDIUM — structurally most similar to episode-query re-ranking.**

FEVER (Fact Extraction and VERification) contains 185,445 claims derived from
Wikipedia sentences. Claims are labeled Supported, Refuted, or NotEnoughInfo
against Wikipedia evidence sentences.

Size: 311K train rows with `(evidence_sentence, claim, label)` triples (v1.0)
License: CC-BY-SA-3.0

**Why FEVER matters for MGRH**: FEVER trains on `(evidence_sentence, claim)` pairs,
which is structurally identical to `(episode_text, query)` pairs in re-ranking.
The reasoning pattern — "does this evidence support this claim?" — is the closest
general-domain proxy to "does this episode answer this query?".

This explains why `DeBERTa-v3-base-mnli-fever-anli` (MoritzLaurer, 122K downloads)
achieves SOTA zero-shot NLI by combining exactly MNLI + FEVER-NLI + ANLI. That
triplet is the proven general NLI pre-training recipe.

**Note**: Use the FEVER-NLI preprocessed split (`easonnie/combine-FEVER-NSMN`)
which already strips the retrieval component and formats the data as standard
3-class NLI. The raw `fever/fever` dataset requires retrieval preprocessing.

**Action**: Add as `nli_fever` in `stage_a_datasets.yaml` using the preprocessed form.

---

#### SNLI — Already in Training, Reduce Relative Weight

**Priority: LOW — present, no action needed except awareness of limitations.**

SNLI (570K pairs, Flickr photo captions) is the foundational NLI dataset but
has documented weaknesses that are actively harmful for cross-encoder training:

| Property | Value | Implication for MGRH |
|-|-|-|
| Premise source | Flickr captions | Avg 14.1 tokens — far shorter than episodes (200-2000 chars) |
| Hypothesis source | Crowd workers | 69% of labels predictable from hypothesis alone (no premise needed) |
| Artifact rate | High | Model can ignore premise entirely and still score well |

The 69% premise-bypass rate (Gururangan et al., 2018) means a cross-encoder can
achieve high SNLI accuracy without actually reading premise + hypothesis jointly.
This is the opposite of what MGRH needs.

**Action**: No change needed — Stage B replay weight is already 0.3 (appropriately low).
SNLI in Stage A is acceptable as an initializer; ANLI + WANLI + FEVER provide the
corrective pressure that forces genuine cross-attention reasoning.

---

#### MS MARCO — Optional Warm-Start (microsoft/ms_marco)

**Priority: LOW — domain data is sufficient; use only if FamilyOS training stalls.**

MS MARCO v2.1 contains 808K real Bing queries with passage-level relevance
annotations (`is_selected: 0/1`). The `Tevatron/msmarco-passage` variant (407K
triplets, Apache-2.0 license) reformats these as `(query, positives, negatives)`
ready for cross-encoder pairwise training.

MS MARCO teaches "does this passage answer this query" — the same relevance signal
MGRH needs. However, the domain gap is significant:

- FamilyOS episodes: episodic memory narratives (200-2000 chars, first-person, kinship entities)
- MS MARCO passages: web documents (~100-200 words, third-person, general topics)

The 43K FamilyOS-domain hard negatives are likely more informative than 407K
out-of-domain web passages. Use MS MARCO only as a fallback if Spearman < 0.35
after Stage C fine-tuning:

```yaml
# configs/training/multitask/relevance_head.yaml
# Optional MS MARCO warm-start (2-3 epochs before Stage C)
ms_marco_warmstart:
  enabled: false    # Enable only if FamilyOS training insufficient
  source: huggingface
  hf_path: "Tevatron/msmarco-passage"    # Apache-2.0, 407K triplets
  format: triplet
  loss: pairwise_margin
  weight: 0.3        # Low — significant domain gap
  max_steps: 5000    # Cap to avoid overfitting to web search patterns
```

---

#### Recommended Pre-Training Order for MGRH

The proven SOTA recipe (validated by `DeBERTa-v3-base-mnli-fever-anli`, 122K downloads)
is MNLI + FEVER + ANLI. Adding WANLI extends this to 4 complementary datasets
totalling ~854K general NLI pairs:

```
Stage A — General NLI backbone (encoder stays frozen for MGRH, head trains):
  MNLI (392K) + FEVER-NLI (185K) + ANLI (169K) + WANLI (108K)
  = ~854K pairs total
  Teaches: entailment reasoning, fact-claim verification, adversarial cases, OOD generalization

Stage B — Domain adaptation replay:
  SNLI replay (weight 0.3) + FamilyOS positives/negatives (~43K)
  Teaches: FamilyOS episode-query relevance, kinship entity patterns, temporal shifts

Stage C — Relevance fine-tuning:
  Human benchmark listwise (LambdaRank) + mined_v2 hard negatives (pairwise margin)
  Teaches: graded relevance scoring for final re-ranking quality
```

**Summary of config changes required**:

| Action | File | Change |
|-|-|-|
| Enable ANLI | `stage_a_datasets.yaml` | `nli_anli.enabled: false -> true` |
| Add WANLI | `stage_a_datasets.yaml` | New `nli_wanli` entry (CC-BY-4.0) |
| Add FEVER-NLI | `stage_a_datasets.yaml` | New `nli_fever` entry (CC-BY-SA-3.0) |
| MS MARCO warm-start | `relevance_head.yaml` | New entry, `enabled: false` (opt-in fallback) |

Total new training pairs added to Stage A: **462K** (ANLI 169K + WANLI 108K + FEVER 185K)

---

## 4. Expected Impact on Retrieval

### 4.1 Conservative Estimate

Based on cross-encoder literature (Nogueira et al., 2019; Thakur et al., 2021):

- Cross-encoder re-ranking typically adds 5-15% nDCG over bi-encoder retrieval
- On our current v4 (MRR=0.81, nDCG=0.79), a 5-10% lift would give:
  - MRR: 0.81 -> 0.85-0.89
  - nDCG: 0.79 -> 0.83-0.87

### 4.2 Per-Type Impact Prediction

| Query Type | Current MRR | Expected Improvement | Reason |
|-|-|-|-|
| temporal | 0.615 | HIGH | Cross-encoder can reason about time expressions jointly |
| emotional | 0.635 | MEDIUM | Cross-encoder can match emotional tone in context |
| causal | 0.833 | MEDIUM | Cross-encoder can follow causal chains |
| entity | 0.833 | LOW | Already strong; entity matching is handled by structural scoring |
| thematic | 0.950 | MINIMAL | Already near-perfect |
| cross_episode | 1.000 | NONE | Already perfect |

### 4.3 Latency Budget

From S7-1 probe:

- Single `similarity()` call: 5.25ms mean
- Cross-encoder pair scoring (full forward pass): ~10-15ms estimated
- Re-ranking top-20: ~200-300ms (serial) or ~30-50ms (batched)

Batched pair encoding is essential:

```python
# Serial: 20 * 15ms = 300ms (too slow)
# Batched: 1 forward pass with batch_size=20 = ~30ms (acceptable)
```

---

## 5. Client API Design

### 5.1 New Methods

```python
class Client:
    # ... existing methods ...

    def score_relevance(
        self,
        query: str,
        passage: str,
    ) -> float:
        """Score how relevant a passage is to a query (0.0 to 1.0).

        Uses cross-encoder NLI/relevance head for joint reasoning
        over query and passage tokens.

        Args:
            query: The search query text.
            passage: The candidate passage/episode text.

        Returns:
            Relevance score between 0.0 (irrelevant) and 1.0 (highly relevant).
        """
        ...

    def rerank(
        self,
        query: str,
        candidates: list[str],
        top_k: int = 10,
    ) -> list[dict]:
        """Re-rank candidate passages by relevance to query.

        Uses batched cross-encoder scoring for efficiency.

        Args:
            query: The search query text.
            candidates: List of candidate passage texts.
            top_k: Number of top results to return.

        Returns:
            List of dicts with 'text', 'score', 'original_index', sorted by score desc.
        """
        ...
```

### 5.2 Integration with Episode Retrieval

```python
# In the v4 pipeline, after Stage 2 (structural fusion):

# Stage 3: Cross-Encoder Re-Ranking (when relevance head is trained)
top_20_episodes = results[:20]
episode_texts = [prepare_text_v1(store, ep.episode_id) for ep in top_20_episodes]
reranked = client.rerank(query_text, episode_texts, top_k=10)

# Interpolate with v4 scores
mu = 0.7  # weight for v4 fusion score
for item in reranked:
    orig_idx = item["original_index"]
    v4_score = top_20_episodes[orig_idx].score
    item["final_score"] = mu * v4_score + (1 - mu) * item["score"]

# Sort by final interpolated score
results = sorted(reranked, key=lambda x: -x["final_score"])[:10]
```

---

## 6. SOTA NLI Head Design

### 6.1 Architecture: Multi-Granularity Relevance Head (MGRH)

SOTA cross-encoder re-ranking (2024-2026) combines three complementary signals
rather than relying on a single CLS token bottleneck. The design below is the
recommended target for UltraBERT.

**Three-signal architecture:**

```
                     ┌─────────────────────────────────────────────────┐
query text ──────────┤                                                   │
                     │  [CLS] query [SEP] episode [SEP]                  │
episode text ─────── ┤  → ModernBERT (22 layers, frozen)                │
                     │  → h_joint  (768-dim per token)                   │
                     └───────────────┬─────────────────────────────────┘
                                     │
                     ┌───────────────▼─────────────────────────────────┐
                     │  Signal 1: CLS Token                              │
                     │  h_cls = h_joint[:, 0]  (768-dim)                │
                     └───────────────┬─────────────────────────────────┘
                                     │
query tokens ──────┐                 │
                   ├── CrossAttentionPairEncoder ──► h_cross (768-dim)  │
episode tokens ────┘ (ESIM bidirectional, 1 layer)   Signal 2          │
                                     │                                   │
query → mode("query") → q_emb ──┐   │                                   │
                                  ├──┤ interaction features (4*768)      │
episode → mode("doc") → d_emb ──┘   │ Signal 3: asymmetric agreement    │
                                     │                                   │
                     ┌───────────────▼─────────────────────────────────┐
                     │  Feature fusion MLP                               │
                     │  in: [h_cls | h_cross | q_emb | d_emb |          │
                     │       q_emb*d_emb | |q_emb-d_emb|]  (5*768)     │
                     │  → Linear(3840, 1024) → GELU → Dropout(0.1)     │
                     │  → Linear(1024, 256)  → GELU → Dropout(0.1)     │
                     │  → Linear(256, 1)     → Sigmoid                  │
                     └───────────────┬─────────────────────────────────┘
                                     │
                              relevance_score ∈ [0.0, 1.0]
```

### 6.2 Why Three Signals

| Signal | Captures | Weakness filled |
|-|-|-|
| CLS (joint cross-encoder) | Global semantic coherence; whether the episode as a whole answers the query | The current bi-encoder cosine (Spearman=0.2047) misses this entirely |
| CrossAttentionPairEncoder | Token-level alignment; which query spans match which episode spans | Catches keyword-match-but-semantically-wrong failures (S7-1 finding) |
| Asymmetric embeddings (q_emb, d_emb) | Query intent vs document role distinction | AgreementGatedHeadV2 mode prompts already encode this — reuse for free |

No single signal achieves SOTA on its own. The MLP fusion learns which signal to
trust per query type. Expected: temporal queries rely more on cross-attention;
thematic queries rely more on CLS coherence; entity queries rely on embedding
agreement.

### 6.3 SOTA Training Objective: LambdaRank + Pairwise Margin

Simple pointwise BCE trains each pair independently. It does not see the relative
ordering — a model could score all episodes at 0.5 and incur zero pointwise loss
while destroying the ranking.

**LambdaRank** (Burges et al., 2006; refined 2010) directly optimizes nDCG:

```
λᵢⱼ = |ΔnDCG(i,j)| · σ(sⱼ - sᵢ)

where:
  sᵢ, sⱼ  = model scores for episode i, j
  |ΔnDCG|  = absolute change in nDCG from swapping i and j
  σ(·)     = sigmoid
```

Episodes with high `|ΔnDCG|` (i.e., swapping them would most hurt the ranking)
get the largest gradient signal. This is why LambdaRank consistently outperforms
pointwise/pairwise objectives on ranking tasks.

**Combined objective:**

```python
loss = lambda_loss(scores, grades) + alpha * pairwise_margin_loss(scores, grades)

# Pairwise margin: for every (positive, hard_negative) pair:
# loss += max(0, margin - (s_pos - s_neg))
# margin = 0.2, alpha = 0.3

# This combines:
# - Global ranking signal (LambdaRank on all docs per query)
# - Local hard negative discrimination (margin on mined negatives)
```

Our human benchmark has **graded labels 0-3** which is exactly what LambdaRank
needs (it computes ΔnDCG from grade differences). This is not available in MNLI/SNLI.

### 6.4 Hard Negative Mining Strategy

Random negatives are too easy — the model learns to distinguish completely unrelated
episodes from relevant ones, but fails at the hard cases that actually appear in
retrieval (top-20 candidates all share surface similarity with the query).

Three hard negative types targeting the specific failure modes observed in Phase 7:

```
Type A: Keyword-overlap negatives (addresses S7-1 failure)
  Query:   "Maya eating at a restaurant"
  Positive: episode where Maya actually had a meal at a restaurant
  Hard neg: episode where Maya watched a cooking show about restaurants
  Construction: retrieve top-20 by BM25, take non-relevant among top-10

Type B: Temporal proximity negatives (addresses temporal separation=0.0064)
  Query:   "What happened on Sunday afternoon?"
  Positive: episode from Sunday afternoon that matches
  Hard neg: episode from Sunday morning or evening (same day, wrong time)
  Construction: same temporal window, different activity

Type C: Participant negatives (addresses entity type failures)
  Query:   "Maya and dad playing"
  Positive: episode of Maya and dad at the park
  Hard neg: episode of Maya and mom playing (same activity, wrong participants)
  Construction: same activity verb, different participant subset
```

Hard negatives are not the majority of training data — they are 2x weighted:

```
Total training triples target:
  Human benchmark:           4,400 (50 queries × 88 episodes, all grades)
  Type A hard negatives:     2,000 (BM25 top-10 non-relevant)
  Type B temporal negatives: 1,500 (time-proximity mining)
  Type C participant negatives: 1,000 (entity swap)
  ─────────────────────────────────
  Total:                     ~8,900 FamilyOS-domain triples

  Optional MS MARCO warm-start: 50,000 triples (subsample)
  Purpose: teach cross-encoder what "passage answers query" means generically
  before FamilyOS fine-tuning
```

### 6.5 Full Head Implementation

```python
class MultiGranularityRelevanceHead(BaseHead):
    """
    SOTA cross-encoder re-ranking head for episode retrieval.

    Combines three signals:
      1. Joint CLS representation (full cross-attention over [query SEP episode])
      2. ESIM cross-attention between separated query/episode token streams
      3. Asymmetric embedding features from AgreementGatedHeadV2 mode prompts

    Trained with LambdaRank + pairwise margin loss on graded relevance labels.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        dropout: float = 0.1,
        pair_encoder: CrossAttentionPairEncoder | None = None,
    ):
        super().__init__(hidden_size)
        self.pair_encoder = pair_encoder

        # Signal 3 interaction features: q_emb, d_emb, q*d, |q-d| = 4 * hidden_size
        # Signal 1 CLS: hidden_size
        # Signal 2 cross: hidden_size
        # Total: 6 * hidden_size = 4608
        in_features = hidden_size * 6

        self.fusion = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Residual projection to stabilize training when pair_encoder is absent
        self.cls_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,          # [B, seq_len, H] from full cross-encoder
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,   # [B] relevance grades 0-3 or 0.0-1.0
        # Pair encoder inputs (optional — from separate query/episode encoding)
        text_a_hidden: torch.Tensor | None = None,   # [B, q_len, H]
        text_b_hidden: torch.Tensor | None = None,   # [B, d_len, H]
        text_a_mask: torch.Tensor | None = None,
        text_b_mask: torch.Tensor | None = None,
        # Asymmetric embedding inputs (from AgreementGatedHeadV2)
        query_embed: torch.Tensor | None = None,     # [B, H] L2-normalized
        doc_embed: torch.Tensor | None = None,       # [B, H] L2-normalized
    ) -> dict:

        # Signal 1: CLS token from joint cross-encoder sequence
        h_cls = self.cls_proj(hidden_states[:, 0])   # [B, H]

        # Signal 2: ESIM cross-attention (use if separate streams available)
        if (self.pair_encoder is not None
                and text_a_hidden is not None
                and text_b_hidden is not None):
            h_cross = self.pair_encoder(
                text_a_hidden, text_b_hidden, text_a_mask, text_b_mask
            )  # [B, H]
        else:
            # Fallback: mean pool of full sequence (excluding CLS)
            mask = attention_mask.unsqueeze(-1).float() if attention_mask is not None \
                   else torch.ones_like(hidden_states[:, :, :1])
            h_cross = (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1)

        # Signal 3: Asymmetric embedding interaction features
        if query_embed is not None and doc_embed is not None:
            q = query_embed
            d = doc_embed
        else:
            # Fallback: split CLS representation (no mode asymmetry)
            q = hidden_states[:, 0]
            d = hidden_states[:, 0]

        interaction = torch.cat([q, d, q * d, torch.abs(q - d)], dim=-1)  # [B, 4H]

        # Fuse all signals
        features = torch.cat([h_cls, h_cross, interaction], dim=-1)  # [B, 6H]
        score = self.fusion(features).squeeze(-1)                      # [B]

        output = {"logits": score}

        if labels is not None:
            # Normalize grades 0-3 to 0.0-1.0 if integer grades provided
            if labels.dtype in (torch.long, torch.int):
                labels_f = labels.float() / 3.0
            else:
                labels_f = labels.float()
            output["loss"] = F.binary_cross_entropy(score, labels_f)

        return output

    def relevance_score(self, logits: torch.Tensor) -> torch.Tensor:
        """Direct relevance score accessor. Logits are already sigmoid-activated."""
        return logits
```

### 6.6 LambdaRank Loss Implementation

```python
class LambdaRankLoss(nn.Module):
    """
    LambdaRank loss for listwise relevance training.

    Directly optimizes nDCG by weighting pairwise gradients by |ΔnDCG|.
    Requires queries with multiple documents and graded relevance labels.

    Reference: Burges et al., "Learning to Rank using Gradient Descent" (2006),
               Burges, "From RankNet to LambdaRank to LambdaMART" (2010)
    """

    def __init__(self, sigma: float = 1.0, ndcg_at: int = 10):
        super().__init__()
        self.sigma = sigma
        self.ndcg_at = ndcg_at

    def forward(
        self,
        scores: torch.Tensor,    # [N] model scores for all docs in query group
        grades: torch.Tensor,    # [N] relevance grades (0-3)
    ) -> torch.Tensor:

        N = scores.shape[0]
        gains = (2.0 ** grades.float()) - 1.0  # standard DCG gain

        # Ideal DCG for normalization
        ideal_sorted = grades.float().sort(descending=True).values[:self.ndcg_at]
        ideal_dcg = (((2.0 ** ideal_sorted) - 1.0) /
                     torch.log2(torch.arange(2, ideal_sorted.shape[0] + 2,
                                             device=grades.device).float())).sum()

        # Pairwise score differences
        si = scores.unsqueeze(1).expand(N, N)   # [N, N]
        sj = scores.unsqueeze(0).expand(N, N)   # [N, N]
        diff = si - sj                           # sᵢ - sⱼ

        # Pairwise gain differences (positive when i is more relevant than j)
        gi = gains.unsqueeze(1).expand(N, N)
        gj = gains.unsqueeze(0).expand(N, N)
        relevant_pairs = (gi > gj).float()       # mask: i should rank above j

        # |ΔnDCG| weighting
        # Approximate: |ΔnDCG(i,j)| ≈ |1/log(rank_i+1) - 1/log(rank_j+1)| * |gain_i - gain_j|
        ranks = torch.arange(1, N + 1, device=scores.device).float()
        discount_i = (1.0 / torch.log2(ranks + 1)).unsqueeze(1).expand(N, N)
        discount_j = (1.0 / torch.log2(ranks + 1)).unsqueeze(0).expand(N, N)
        delta_ndcg = torch.abs(discount_i - discount_j) * torch.abs(gi - gj)
        if ideal_dcg > 0:
            delta_ndcg = delta_ndcg / ideal_dcg

        # LambdaRank gradient weighting
        lambda_ij = delta_ndcg * relevant_pairs * torch.sigmoid(-self.sigma * diff)

        # Loss: negative log-likelihood weighted by lambda
        loss = -(lambda_ij * F.logsigmoid(self.sigma * diff)).sum()
        return loss / (N * (N - 1) + 1e-8)


class CombinedRankingLoss(nn.Module):
    """
    Combined LambdaRank + pairwise margin loss.

    LambdaRank: global list-level nDCG optimization
    Pairwise margin: local hard-negative discrimination
    """

    def __init__(self, margin: float = 0.2, alpha: float = 0.3, ndcg_at: int = 10):
        super().__init__()
        self.lambda_loss = LambdaRankLoss(ndcg_at=ndcg_at)
        self.margin = margin
        self.alpha = alpha

    def forward(
        self,
        scores: torch.Tensor,    # [N] per query group
        grades: torch.Tensor,    # [N] relevance grades
        is_hard_negative: torch.Tensor | None = None,  # [N] bool mask
    ) -> torch.Tensor:

        ll = self.lambda_loss(scores, grades)

        # Pairwise margin on hard negatives
        pos_mask = grades > 1  # grades 2,3 = relevant
        neg_mask = (grades == 0)
        if is_hard_negative is not None:
            neg_mask = neg_mask & is_hard_negative

        pl = torch.tensor(0.0, device=scores.device)
        if pos_mask.any() and neg_mask.any():
            s_pos = scores[pos_mask].unsqueeze(1)   # [P, 1]
            s_neg = scores[neg_mask].unsqueeze(0)   # [1, N]
            pl = F.relu(self.margin - (s_pos - s_neg)).mean()

        return ll + self.alpha * pl
```

### 6.7 Training Loop Outline

```python
def train_relevance_head(
    model: ModernBERTMultiTask,
    train_data: list[QueryGroup],     # list of (query, [(episode, grade), ...])
    dev_data: list[QueryGroup],
    epochs: int = 10,
    lr_head: float = 1e-4,
    lr_pair_encoder: float = 5e-5,
):
    # Freeze everything except the new head and pair encoder
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.relevance_head.named_parameters():
        param.requires_grad = True
    if model.relevance_head.pair_encoder:
        for param in model.relevance_head.pair_encoder.parameters():
            param.requires_grad = True

    optimizer = AdamW([
        {"params": model.relevance_head.fusion.parameters(), "lr": lr_head},
        {"params": model.relevance_head.pair_encoder.parameters(), "lr": lr_pair_encoder}
        if model.relevance_head.pair_encoder else [],
    ], weight_decay=0.01)

    criterion = CombinedRankingLoss(margin=0.2, alpha=0.3, ndcg_at=10)
    ema = ExponentialMovingAverage(model.relevance_head.parameters(), decay=0.995)
    best_ndcg = 0.0

    for epoch in range(epochs):
        model.train()
        for group in train_data:
            query_text = group.query
            episodes = group.episodes   # list of (text, grade, is_hard_negative)

            # Forward pass: batch all (query, episode) pairs together
            scores = []
            for episode_text, grade, _ in episodes:
                score = model(
                    query=query_text,
                    passage=episode_text,
                    capability="relevance",
                )["logits"]
                scores.append(score)

            scores = torch.stack(scores)
            grades = torch.tensor([e[1] for e in episodes], device=scores.device)
            is_hard = torch.tensor([e[2] for e in episodes], device=scores.device)

            loss = criterion(scores, grades, is_hard)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            ema.update()

        # Dev evaluation with EMA weights
        with ema.average_parameters():
            ndcg = evaluate_ndcg(model, dev_data, at=10)
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            save_checkpoint(model.relevance_head, "best_relevance_head.pt")

    return model
```

### 6.8 Expected Quality vs Architecture Alternatives

| Architecture | Spearman (estimated) | nDCG Lift | Latency (top-20 batch) | Notes |
|-|-|-|-|-|
| Current bi-encoder (similarity()) | 0.2047 (measured) | 0 | 105ms | FAIL, abort flag |
| Standard cross-encoder (CLS only) | 0.45-0.55 | +4-8pp | 200-300ms serial | Good baseline |
| ESIM cross-attention head only | 0.50-0.60 | +6-10pp | 80ms batched | Better |
| **MGRH (this design)** | **0.60-0.70** | **+8-14pp** | **50ms batched** | **Recommended** |
| Generative re-ranker (monoT5) | 0.65-0.75 | +10-15pp | 500ms+ | Too slow for prod |

Generative re-rankers (RankGPT, monoT5) achieve top scores on BEIR but are 10x
slower. MGRH sits at the Pareto frontier of quality vs latency for this use case.

---

## 7. Implementation Checklist

### Phase 1: Wire Existing NLI Head (1-2 days)

- [ ] Add `client.score_relevance(query, passage)` using NLI head with pair tokenization
- [ ] Add `client.rerank(query, candidates, top_k)` with batched inference
- [ ] Test on S7-1 sanity pairs: verify ordering improves over `similarity()`
- [ ] Benchmark latency: target < 50ms for batch of 20

### Phase 2: Prepare Relevance Training Data (2-3 days)

- [ ] Export human benchmark as (query, episode_text, grade) JSONL
- [ ] Generate hard negatives: topic-similar, keyword-overlap, temporal-adjacent
- [ ] Train/dev/holdout split (80/10/10)
- [ ] Validate data quality: check label distributions, text lengths

### Phase 3: Fine-Tune Relevance Head (2-3 days)

- [ ] Set up training config (freeze encoder, train head + pair_encoder)
- [ ] Train on FamilyOS relevance data (5-10 epochs)
- [ ] Monitor Spearman correlation, AUC-ROC on dev set
- [ ] EMA checkpoint selection
- [ ] Evaluate on holdout: target Spearman > 0.50, AUC-ROC > 0.80

### Phase 4: Validate Re-Ranking Quality (1 day)

- [ ] Run v4 pipeline + re-ranking on full human benchmark
- [ ] Measure MRR lift, nDCG lift, per-query-type changes
- [ ] Error analysis: promoted vs demoted queries
- [ ] Determine optimal interpolation weight (mu)

### Phase 5: Production Integration (1-2 days)

- [ ] Add `RelevanceHead` or retrained `NLIHead` to production model export
- [ ] Update `unified_output.py` with re-ranking capability
- [ ] Wire into K1 retrieval syscall pipeline
- [ ] End-to-end latency validation

---

## 7. Summary

| Aspect | Current State | Target State |
|-|-|-|
| NLI head | Trained on MNLI/SNLI, unreachable from client | Fine-tuned on relevance, wired through `score_relevance()` |
| Pair encoding | Code exists (`pair_encoder.py`), never invoked | Activated for relevance scoring with cross-attention |
| Re-ranking | BLOCKED (Spearman=0.20, bi-encoder only) | Enabled (target Spearman>0.50, cross-encoder) |
| Client API | `similarity()` only (bi-encoder cosine) | `score_relevance()` + `rerank()` (cross-encoder) |
| Training data | MNLI + SNLI (generic NLI) | FamilyOS relevance triples + hard negatives |
| Expected MRR lift | 0 (GATE FAIL) | +5-10pp (0.81 -> 0.85-0.89) |
| Latency overhead | N/A | ~30-50ms for batched re-ranking of top-20 |
