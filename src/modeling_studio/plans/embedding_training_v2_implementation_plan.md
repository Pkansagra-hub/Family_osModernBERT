I’m turning your Colab-first constraints into a repo-specific rollout plan so we can agree on the shape before touching the two existing files.

Created 3 todos

I’ve got the shape: your approach is mostly right, but the cache and distillation need to be added as new modes inside the existing script rather than assumed to already exist.

Completed: *Validate existing-file approach* (1/3)

Yes — **mostly correct**, with one important correction.

## Short answer

Your high-level flow is **right**:

1. data already loaded in Colab
2. load Qwen teacher
3. create teacher cache
4. load `v2-best-ema`
5. reuse existing training script/config and distill

But this part needs to be said clearly:

## The correction

train_embedding_heads_bakeoff.py and embedding_heads_bakeoff.yaml do **not currently know anything** about:

- Qwen teacher loading
- teacher cache creation
- teacher cache reading
- distillation loss
- a “cache-only” mode
- a “distill-from-teacher” mode

So:

- **you are right** that we can stay inside the existing files
- but **we still need to edit those files** to add these capabilities

So the correct framing is:

> **No new files. No new standalone scripts.**
>
> **But yes, we must extend the existing script and config.**

That is the cleanest version of your constraint.

---

## Proposed epic

## Epic: Qwen teacher distillation using existing bakeoff pipeline

Goal:
- keep all work Colab-first
- reuse existing repo pipeline
- avoid creating new files
- extend only:
  - embedding_heads_bakeoff.yaml
  - train_embedding_heads_bakeoff.py
- use teacher:
  - `Qwen/Qwen3-Embedding-4B`
- use student checkpoint:
  - v2-best-ema

---

## Milestone 1 — Extend config for teacher + cache + distillation

This is the “teach the YAML new words” milestone.

### Why
Your current config knows about:
- encoder checkpoint
- embedding head
- loss
- training
- data
- stage_b

It does **not** know about distillation.

### Files
- embedding_heads_bakeoff.yaml

### Issues

#### Issue 1.1 — Add teacher section
Add config section for:
- teacher model id
- dtype
- max length
- query instruction behavior
- cache path
- batch size for cache generation

Example conceptually:
- `teacher.model_name`
- `teacher.max_length`
- `teacher.batch_size`
- `teacher.dtype`
- `teacher.cache_dir`

#### Issue 1.2 — Add distillation section
Add config section for:
- enable/disable distillation
- vector loss weight
- ranking loss weight
- student contrastive loss weight
- temperature for soft similarity matching

#### Issue 1.3 — Add run-mode section
Add config or CLI-compatible mode support for:
- cache build mode
- distill mode
- optionally plain old bakeoff mode remains unchanged

### Deliverable
A backward-compatible config:
- old runs still work
- new teacher/distill runs become possible

---

## Milestone 2 — Add teacher cache mode to existing script

This is the “Colab step 2” milestone.

### Why
You said:
> 2nd load qwen and then create cache

Correct — but the script must gain a mode for that.

### Files
- train_embedding_heads_bakeoff.py

### Issues

#### Issue 2.1 — Add teacher text collection from existing data sources
Reuse current dataset/source loading logic already in the script.

Important:
- do **not** invent a second data-loading system
- reuse the existing source resolution path from config

Need to collect texts from:
- broad corpus sources you choose
- embedding sources you choose

Text roles:
- query texts
- document texts
- plain semantic texts

#### Issue 2.2 — Add Qwen teacher loader
Inside the same script, add support to load:
- `Qwen/Qwen3-Embedding-4B`

Colab-side assumptions:
- model is downloaded there
- GPU is A100 80GB
- use bf16
- use attention acceleration if available

#### Issue 2.3 — Add cache generation mode
Add CLI mode like conceptually:
- `--build_teacher_cache`

This mode should:
- load config
- gather texts
- dedupe texts
- encode them with teacher
- store embeddings/cache metadata

#### Issue 2.4 — Store query/document-aware cache
For retrieval-aware data:
- query texts should be teacher-encoded in query mode
- document texts in document mode

This matters especially for:
- `query_doc`

### Deliverable
One existing script can now:
- build teacher cache in Colab
- without creating a new script file

---

## Milestone 3 — Add distillation training mode into the same script

This is the “Colab step 3” milestone.

### Why
You said:
> then we distill it

Yes — but current training code only knows:
- student contrastive training
- no teacher guidance

So we must extend the same script.

### Files
- train_embedding_heads_bakeoff.py

### Issues

#### Issue 3.1 — Load student from `v2-best-ema`
Use:
- v2-best-ema
- Colab equivalent mounted path there

This should become the student init checkpoint.

#### Issue 3.2 — Add teacher-cache loading path
Instead of live teacher forward every batch:
- load teacher vectors from cache

That keeps Colab training efficient.

#### Issue 3.3 — Add distillation losses
Add support for:
- existing contrastive loss
- teacher vector alignment loss
- teacher ranking/similarity loss

Recommended priority:
- keep current contrastive as main objective
- teacher acts as guidance, not replacement

#### Issue 3.4 — Keep current freeze behavior initially
First pass:
- encoder frozen
- only embedding head trainable

This matches your current bakeoff design and reduces risk.

#### Issue 3.5 — Add distillation run mode
Conceptually:
- `--distill_teacher_cache`

So same file can do:
- normal bakeoff
- stage_b
- teacher cache build
- teacher distillation

### Deliverable
Same script can now fine-tune `v2-best-ema` under teacher guidance.

---

## Milestone 4 — Preserve old behavior and avoid pipeline breakage

This is boring but crucial. Boring things save projects.

### Why
You do **not** want the old bakeoff flow broken while adding distillation.

### Issues

#### Issue 4.1 — Backward compatibility
If distillation config is absent:
- current bakeoff should still run exactly as before

#### Issue 4.2 — Keep stage_b intact
Current:
- `--stage_b`
- query/document routing
- aux objectives

These should still work.

#### Issue 4.3 — Keep current evaluation path
Continue using:
- slice eval
- retrieval eval
- composite score

Because those are already your trusted metrics.

### Deliverable
Distillation becomes an extension, not a rewrite.

---

## Milestone 5 — Colab execution sequence

This is the “how we actually run it there” milestone.

You said all of this is happening in Google Colab. Good. Then the execution plan should look like this:

### Issue 5.1 — Data already mounted
No work needed here, as you said.

### Issue 5.2 — Run teacher cache build
In Colab:
- load repo
- open modified config
- run existing script in cache-build mode

Output:
- teacher cache artifact on Drive / Colab storage

### Issue 5.3 — Run distillation from `v2-best-ema`
Then run same script in distill mode:
- load teacher cache
- load `v2-best-ema`
- train embedding head with teacher guidance

### Issue 5.4 — Evaluate with existing metrics
Use the same current evaluation path:
- retrieval benchmark
- slice metrics
- later PoC comparison if desired

---

## Issue breakdown as GitHub-style tickets

If you want this issue-wise, I’d break it down like this:

### Epic
**Qwen teacher distillation inside existing embedding bakeoff pipeline**

### Milestone A
**Config support**

- Issue A1: Add `teacher` section to embedding_heads_bakeoff.yaml
- Issue A2: Add `distillation` section to embedding_heads_bakeoff.yaml
- Issue A3: Add mode/config flags for cache-build and distillation runs

### Milestone B
**Teacher cache inside existing script**

- Issue B1: Reuse existing source loader to collect distillation texts
- Issue B2: Add Qwen teacher model loading in existing training script
- Issue B3: Add dedupe + cache write path
- Issue B4: Add query/document-aware teacher embedding cache handling

### Milestone C
**Distillation training inside existing script**

- Issue C1: Add teacher cache read path
- Issue C2: Add vector alignment loss
- Issue C3: Add teacher similarity/ranking distillation loss
- Issue C4: Add distillation run mode using `v2-best-ema`
- Issue C5: Keep encoder frozen in first implementation

### Milestone D
**Validation and Colab runbook**

- Issue D1: Verify old bakeoff still works without distillation config
- Issue D2: Verify cache-build mode in Colab
- Issue D3: Verify distill mode in Colab from `v2-best-ema`
- Issue D4: Compare results with existing retrieval metrics

---

## My verdict on your proposal

## What is correct
- Colab-first workflow: **correct**
- data loading already solved: **correct**
- Qwen teacher cache first, then distill: **correct**
- reuse existing script/config: **correct**
- use `v2-best-ema` as student: **correct**

## What needs correction
- “we will not be writing new codes” → **not fully correct**
  - we should say:
  - **we will not create new files**
  - **but we will edit existing code in the script/config**
  - because teacher cache + distillation logic does not exist yet

That is the exact accurate version.

## Recommended final framing

I would phrase the plan like this:

> We will not create new files.
> We will only extend:
> - embedding_heads_bakeoff.yaml
> - train_embedding_heads_bakeoff.py
>
> We will add two new capabilities into the existing pipeline:
> 1. teacher cache build mode
> 2. teacher distillation training mode
>
> Then in Colab we will:
> - load data
> - build Qwen cache
> - load `v2-best-ema`
> - distill using the same existing script

That is the plan I would stand behind.

## What I think we should do next

If you want to proceed, the next concrete step should be:

### Step next
I draft the **exact config schema changes** first for embedding_heads_bakeoff.yaml, before touching the Python.

That is the safest place to lock the design.

## Updated checklist

- [x] Validated the “existing files only” approach
- [x] Corrected the unrealistic part of the proposal
- [x] Broke the work into epic / milestones / issues
- [x] Anchored the plan to Colab execution flow

Summary: Your approach is mostly correct, but the exact right version is “no new files, only edits to the existing config and training script,” and I broke that into a concrete epic with milestones for config support, teacher-cache mode, distillation mode, backward compatibility, and Colab execution.I finished the planning work, clarified the “reuse existing files only” constraint, and turned your Colab workflow into a concrete epic/milestone/issue structure with the one necessary correction: no new files is fine, but the existing config and training script still need to be extended for teacher cache and distillation modes.
