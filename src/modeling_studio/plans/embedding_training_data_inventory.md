# Embedding Training Data Inventory

> Scope: Only datasets we plan to use in the embedding-head training mix
> Updated: March 10, 2026

---

## Readily available FamilyOS data

| Dataset | Path / Source | Type | Status | Why use it in mix |
| --- | --- | --- | --- | --- |
| FamilyOS Silver Synthetic Triplets | `data/familyos/embeddings/silver_synthetic` | Triplets (`anchor`, `positive`, `negative`) | Ready now | Main FamilyOS embedding volume for contrastive training |
| FamilyOS Hard Negatives | `data/familyos/embeddings/hard_negatives` | Triplets with hard-negative metadata | Ready now | Retrieval-critical near-miss negatives such as entity swap, temporal shift, negation, causality flip |
| FamilyOS Gold Triplets | `data/familyos/embeddings/gold` | Triplets | Not currently visible in `data/familyos/embeddings`; use if restored or generated locally | Validation / model selection / sanity-check on higher-quality FamilyOS examples |

---

## FamilyOS data to use after mining

| Dataset | Path / Source | Type after mining | Status | Why use it in mix |
| --- | --- | --- | --- | --- |
| Unified Synthetic Memory Data | `data/familyos/unified/output_synthetic` | Query-document pairs, pseudo-triplets, asymmetric retrieval pairs | Mine first | Best local source for `query_memory` ↔ memory-style retrieval supervision |
| Unified Synthetic Intent / Ingress Data | `data/familyos/unified/output_synthetic` | Topical positives and same-domain hard negatives | Mine first | Helps produce realistic retrieval queries with same-topic different-event negatives |
| Unified Synthetic Temporal / Relation / NER Signals | `data/familyos/unified/output_synthetic` | Structured hard negatives | Mine first | Useful for wrong-person, wrong-time, wrong-relation retrieval confusions |

### Mining strategy for `data/familyos/unified/output_synthetic`

#### Exploration summary

| Signal | Observed count / note |
| --- | --- |
| Total rows | 492,785 |
| Clean rows with only expected core task keys | 491,896 |
| Rows with extra / malformed task keys | 889 |
| `query_memory` + `MEMORY` | 47,754 |
| `log_memory` + `MEMORY` | 18,312 |
| `log_memory` + `DIARY` | 9,635 |
| Rows with `relations` | 234,760 |
| Rows with `temporal` | 334,501 |
| Rows with `ner_family` | 395,212 |

#### Filters to apply before mining

| Filter | Rule | Why |
| --- | --- | --- |
| Task-key cleanliness filter | Keep rows where task keys are in `{emotions, sentiment, ner_family, safety_familyos, intent, ingress, relations, temporal}` plus optional `hub_routing` | Removes malformed synthetic annotations and weird task-key noise |
| Safety filter for first retrieval mix | Prefer `GREEN` and `AMBER`; exclude `RED` and `CRISIS` from initial positive-pair mining | Avoids contaminating generic retrieval with safety escalation behavior |
| Minimum text quality filter | Drop rows with empty text, repeated boilerplate, or obviously corrupted annotations | Prevents degenerate positives / negatives |
| Length filter | Keep texts within normal retrieval length band used by current training config | Keeps mined data aligned with embedding-head sequence budget |

#### Mined datasets to create

| Mined dataset | Source rule | Output type | Why use it |
| --- | --- | --- | --- |
| Memory query-doc pairs | Pair `query_memory` + `MEMORY` utterances with `log_memory` + `MEMORY` / `DIARY` utterances from matching topic buckets | Query-document pairs | Most retrieval-native FamilyOS supervision in the corpus |
| Memory pseudo-triplets | `(query_memory query, matched log_memory memory, wrong memory)` using same-ingress or same-entity negatives | Triplets | Directly trains retrieval ranking instead of only semantic similarity |
| Same-domain different-event negatives | Within same ingress / topic family, pair related texts and sample negatives with different event content | Hard negatives | Teaches event discrimination instead of broad topical similarity |
| Wrong-person negatives | Use NER / relation fields to swap person / kinship while keeping event template similar | Hard negatives | Critical for family memory retrieval correctness |
| Wrong-time negatives | Use temporal spans to construct same-event-but-wrong-date or wrong-frequency negatives | Hard negatives | Critical for reminders, memories, and planning retrieval |
| Relation-confusion negatives | Use relation labels such as `parent_of`, `sibling_of`, `cousin_of` to create near misses | Hard negatives | Helps the head separate family graph confusions |

#### Recommended mining rules

| Mining target | Positive construction | Negative construction |
| --- | --- | --- |
| Query → memory retrieval | `query_memory/MEMORY` as query, `log_memory/MEMORY` or `log_memory/DIARY` as relevant memory when they share topic/entity/time cues | Other memory/log entries from same ingress or entity family but different event |
| Reminder retrieval | `set_reminder/TASK` or `set_reminder/PLANNING` as query-like utterances paired with reminder-style declarative rewrites | Same ingress reminders with different time, person, or task |
| Reflective memory retrieval | `reflect/MEMORY` paired with memory-like statements mentioning same event, person, or place | Same topic nostalgia texts about different events |
| Health / advice retrieval | `seek_advice/HEALTH` paired with declarative health-memory or routine statements when entity and symptom overlap | Similar health queries with different person, condition, or timing |

#### Priority order for mining

| Priority | Mining job | Reason |
| --- | --- | --- |
| P1 | `query_memory` ↔ `log_memory` / `DIARY` | Highest-value retrieval-native supervision |
| P2 | wrong-person and wrong-time hard negatives | Most important FamilyOS retrieval confusions |
| P3 | same-domain different-event negatives | Improves precision within topical clusters |
| P4 | reminder / planning retrieval pairs | Good for task-oriented memory and planning search |
| P5 | health / relationship / finance domain retrieval pairs | Adds broader domain diversity after core memory mining |

#### Mining spec v1

| Spec item | Decision |
| --- | --- |
| Primary bulk mining input | `data/familyos/unified/output_synthetic` |
| QA / spot-check input | `data/familyos/unified/golden_set` |
| Output artifacts | `query_doc_pairs.jsonl`, `pseudo_triplets.jsonl`, `hard_negatives.jsonl`, `mining_manifest.json` |
| Recommended output location | `data/familyos/embeddings/mined_v1/` |
| Primary training use | Retrieval-first head training and bake-off evaluation |
| Initial safety scope | Use `GREEN` and `AMBER` only for positives; keep `RED` and `CRISIS` out of v1 positive mining |

#### Canonical row contract

Every mined record should preserve enough metadata to regenerate or audit the example later.

| Field | Required | Notes |
| --- | --- | --- |
| `source_id` | Yes | Original row id such as `syn_00007` |
| `source_file` | Yes | Shard path for reproducibility |
| `text` | Yes | Raw source text |
| `intent` | Yes | From `tasks.intent` |
| `ingress` | Yes | From `tasks.ingress` |
| `safety_familyos` | Yes | Used for filtering |
| `entities` | Yes | Normalized `PERSON`, `KINSHIP`, `PET`, `FAMILY_EVENT`, `ROUTINE`, `HEIRLOOM` spans |
| `relations` | Yes | Normalized relation predicates if present |
| `temporal` | Yes | Normalized temporal spans grouped by label |
| `hub_routing` | Optional | Keep when present for later routing-aware analysis |
| `mining_tags` | Yes | Tags such as `memory_query`, `wrong_person_negative`, `wrong_time_negative` |

#### Canonical mining pipeline

| Step | Rule | Output |
| --- | --- | --- |
| 1. Ingest | Read all `shard_*.jsonl` from `output_synthetic`; attach shard name and line index | Stable raw row stream |
| 2. Clean | Apply task-key cleanliness, safety, empty-text, and length filters | Filtered candidate rows |
| 3. Normalize | Lowercase for matching, collapse whitespace, preserve original text separately, normalize entity and temporal labels | Comparable feature view |
| 4. Bucket | Group by `(intent, ingress)` and secondary signatures such as entity overlap, relation predicate, temporal type | Candidate pools |
| 5. Score positives | Prefer rows with aligned intent/ingress plus shared entity, relation, event, or temporal cues | Ranked positive candidates |
| 6. Score negatives | Prefer same-domain rows with one critical mismatch: wrong person, wrong date, wrong relation, or different event | Ranked hard negatives |
| 7. Deduplicate | Remove exact duplicates and near-duplicates from the same bucket before export | Cleaner training mix |
| 8. Export | Write pairs, triplets, and hard-negative manifests with source provenance | Auditable mined datasets |

#### Positive-pair construction rules

| Mining family | Query side | Positive side | Positive acceptance rule |
| --- | --- | --- | --- |
| Memory lookup | `query_memory` + `MEMORY` | `log_memory` + `MEMORY` or `DIARY` | Require at least one of: shared entity, shared relation family, shared temporal cue, or strong event/topic overlap |
| Reminder lookup | `set_reminder` + `TASK` or `PLANNING` | Declarative reminder/memory text in same task family | Require shared task object plus compatible time/frequency when present |
| Reflective memory | `reflect` + `MEMORY` | `log_memory` or `share_news` with same event/person/place | Require same event anchor and no direct contradiction in time/person |
| Advice retrieval | `seek_advice` + `HEALTH` / `RELATIONSHIP` / `FINANCE` | Declarative statements about same problem frame | Require entity/domain overlap and compatible issue type |

#### Hard-negative construction rules

| Negative type | How to create it | Keep only if |
| --- | --- | --- |
| Wrong-person | Same event or topic template but entity set differs on the main `PERSON` / `KINSHIP` target | Query and negative remain topically similar but refer to different people |
| Wrong-time | Same activity/event family but temporal span differs on `DATE_REL`, `DATE_ABS`, `TIME`, or `FREQUENCY` | Text is not a duplicate and the time mismatch changes retrieval correctness |
| Wrong-relation | Same people/topic neighborhood but relation predicate changes (`parent_of` vs `sibling_of`, etc.) | Relation change would make the result semantically wrong |
| Same-domain different-event | Same ingress and similar lexical domain, different event anchor | Negative is plausible enough to fool a broad semantic encoder |
| Safety-matched distractor | Same domain and emotion tone, different event | Avoids learning a shortcut from safety/emotion alone |

#### Export schema

| Artifact | Minimal schema |
| --- | --- |
| `query_doc_pairs.jsonl` | `query`, `document`, `query_id`, `document_id`, `pair_type`, `shared_features`, `source_ids` |
| `pseudo_triplets.jsonl` | `anchor`, `positive`, `negative`, `triplet_type`, `hard_negative_type`, `source_ids` |
| `hard_negatives.jsonl` | `query`, `candidate_negative`, `negative_type`, `mismatch_features`, `source_ids` |
| `mining_manifest.json` | filter settings, counts per mining job, acceptance thresholds, shard coverage |

#### Quality gates before training use

| Gate | Threshold / action |
| --- | --- |
| Duplicate control | Remove exact duplicates and near-duplicates above the agreed lexical similarity threshold |
| Per-query diversity | Cap number of positives and negatives contributed by a single source row |
| Bucket balance | Avoid one ingress or one intent dominating the mined output |
| Spot-check audit | Manually review at least 100 pairs/triplets across P1-P3 jobs before training |
| Gold-set sanity | Validate rules against `data/familyos/unified/golden_set` before promoting mined outputs into the main mix |

#### Other folders checked

The following FamilyOS folders were reviewed while defining the spec so we do not over-index on `output_synthetic` blindly.

| Folder | Observed structure | Role in this spec |
| --- | --- | --- |
| `data/familyos/intents` | `gold/`, `silver/`, `archive/`, `README.md` | Auxiliary label source for intent semantics; not a primary retrieval-pair mining source |
| `data/familyos/ingress` | `gold/`, `silver/`, `archive/`, `README.md` | Auxiliary domain labels for bucket design and sanity checks |
| `data/familyos/relations` | `gold/`, `silver/`, `archive/`, `README.md` | Primary auxiliary source for wrong-relation hard-negative templates |
| `data/familyos/temporal` | `gold/`, `silver/`, `archive/`, `README.md` | Primary auxiliary source for wrong-time hard-negative rules |
| `data/familyos/ner_family` | `gold/`, `silver/`, `archive/`, `README.md` | Primary auxiliary source for entity-aware mining and wrong-person negatives |
| `data/familyos/emotions` | `gold/`, `silver/`, `silver_super/`, `README.md` | Secondary auxiliary source; useful for tone matching but not enough for retrieval positives by itself |
| `data/familyos/safety` | `gold/`, `silver/`, `archive/`, `README.md` | Filtering and sampling control, not a positive mining source |
| `data/familyos/unified/golden_set` | `shard_0000.jsonl`, `hash_index.jsonl` | QA slice for rule validation and manual inspection |
| `data/familyos/unified/output` | 14 shards | Alternate unified export; keep out of v1 bulk mining until dedup/resolution rules are finalized |
| `data/familyos/unified/output_healed_merged` | 113 shards | Rich alternate export, but use only after v1 proves stable on `output_synthetic` |

---

## Open-source data we plan to use

| Dataset | Source name | Type | Status | Why use it in mix |
| --- | --- | --- | --- | --- |
| STS Benchmark | `sentence-transformers/stsb` | Pair-score similarity | Ready via config | Calibrates semantic similarity and stabilizes dense space |
| AllNLI | `sentence-transformers/all-nli` | Positive sentence pairs / entailment-style supervision | Ready via config | Standard sentence-embedding training fuel for semantic grouping |
| SICK-R | `mteb/sickr-sts` | Pair-score similarity | Ready via config | Good bridge between generic similarity and relational/narrative text |
| STS12 | `mteb/sts12-sts` | Pair-score similarity | Ready via config | Adds out-of-domain semantic diversity |
| STS13 | `mteb/sts13-sts` | Pair-score similarity | Ready via config | Adds out-of-domain semantic diversity |
| STS14 | `mteb/sts14-sts` | Pair-score similarity | Ready via config | Adds out-of-domain semantic diversity |

---

## Proposed training mix roles

| Mix bucket | Datasets |
| --- | --- |
| Core retrieval training | `familyos/embeddings/silver_synthetic`, `familyos/embeddings/hard_negatives` |
| FamilyOS retrieval mining add-on | `data/familyos/unified/output_synthetic` after mining |
| Semantic regularization | `sentence-transformers/stsb`, `sentence-transformers/all-nli`, `mteb/sickr-sts`, `mteb/sts12-sts`, `mteb/sts13-sts`, `mteb/sts14-sts` |
| Validation / selection | `data/familyos/embeddings/gold` if available + internal retrieval benchmark + MTEB STS |
