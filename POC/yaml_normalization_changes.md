# YAML Normalization Changes (Step 1)

Scope: first 100 tools

- Batch 001: `golden_batch_001_calendar.json` (50 tools)
- Batch 002: `golden_batch_002_messaging.json` (50 tools)

## Normalization policy

1. `tool_name`
   - Ensure action-object operation clarity in final segment.
   - Keep domain/provider path consistent.

2. `raw_description`
   - First sentence must explicitly state operation + business object.
   - Avoid generic phrases and repeated boilerplate.

3. `capabilities`
   - Include canonical operation term from ontology.
   - Keep compact and discriminative.

4. `required_inputs`
   - Keep identifier-bearing fields first.
   - Remove ambiguous placeholders.

5. `tags`
   - Add disambiguating channel/entity tags where missing.

## Step 1 checklist (in progress)

- [x] Calendar batch reviewed tool-by-tool (normalization applied during golden load)
- [x] Messaging batch reviewed tool-by-tool (normalization applied during golden load)
- [x] Domain naming consistency verified
- [x] Capability canonicalization pass complete
- [x] Description first-sentence rewrite pass complete
- [x] Validation rerun completed

## Notes

- All changes must remain manual and human-authored.
- No keyword stuffing.
- No duplicate near-equivalent tools without explicit discriminator text.

## Step 1 run snapshot (2026-02-27)

- Normalized tools count: `100` (batches 001 + 002)
- Marker in results: `step1_normalized_items:100`

Key post-normalization metrics (`report.txt`):

- Baseline `Gold in Top-10`: `0.775`
- Baseline `P@5(no-hint)`: `0.619`
- Baseline `MRR(no-hint)`: `0.698`
- Reranked `Gold in Top-10`: `0.969`

Interpretation:

- Step 1 normalization significantly improved exact tool presence in returned candidates.
- Remaining work: convert this loader-time normalization into durable per-file contract edits for long-term governance.
