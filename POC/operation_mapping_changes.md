# Operation Ontology and Mapping Changes (Step 2)

Scope: manual golden batches (500 tools)

- Source: `POC/manual_golden_batches/golden_batch_001..010_*.json`
- Coverage target: all observed operation verbs and sub-clusters

## Ontology and mapping policy

1. `INTENT_TOKEN_NORMALIZATION`
   - Normalize verb variants to canonical operation families.
   - Keep mappings deterministic and retrieval-oriented.

2. `OPERATION_ONTOLOGY`
   - Maintain canonical families with practical alias sets.
   - Include domain-common verbs from manual corpus profile.

3. `SUBCLUSTER_VERB_TO_CANONICAL`
   - Map first verb in `sub_cluster` name to canonical operation.
   - Use as fallback when explicit sub-cluster override is not present.

4. `CANONICAL_OPERATION_BY_SUBCLUSTER`
   - Keep explicit overrides for ambiguous or high-impact sub-clusters.
   - Prefer exact override for known routing edge cases.

5. Mapping QA
   - Validate sub-cluster mapping coverage and query operation-token coverage.
   - Surface unmapped sub-clusters as required remediation items.

## Step 2 checklist (completed)

- [x] Profiled operation vocabulary from manual golden batches
- [x] Expanded normalization and ontology families in evaluator
- [x] Added canonical sub-cluster resolver and verb fallback mapping
- [x] Added mapping QA computation in evaluation pipeline
- [x] Added report integration for Step 2 QA section
- [x] Generated `operation_ontology_v1.md`
- [x] Generated `operation_mapping_test_report.md`
- [x] Validation rerun completed

## Notes

- Mapping coverage is measured against real manual corpus vocabulary.
- Query token coverage is measured for operation-relevant tokens only.
- No synthetic or script-generated golden data is used.

## Step 2 run snapshot (2026-02-27)

Key QA metrics:

- Sub-cluster mapping coverage: `100/100 (1.000)`
- Query token ontology coverage: `574/606 (0.947)`
- Unmapped sub-clusters: `none`

Key retrieval metrics (`report.txt`):

- Hybrid `TPR@10`: `0.957`
- Hybrid `DMR@10`: `1.000`
- Hybrid `OMR@10`: `0.959`
- Hybrid `P@5`: `0.844`
- Hybrid `MRR`: `0.872`

Interpretation:

- Step 2 acceptance gate is satisfied (`OMR@10 >= 0.90`).
- Ontology/mapping coverage no longer appears to be a limiting factor.
- Next bottleneck is retrieval fusion calibration stability and generalization.
