# Top-10 Miss Analysis (Step 4)

Generated: 2026-02-27

- Total labeled queries: `516`
- Exact-target misses (not in top-10): `13`
- Miss rate: `0.025`

## Miss reasons (heuristic categories)

- `ontology gap`: `12`
- `duplicate/overlapping tools`: `1`

## Prioritized remediation backlog

1. Ontology gap

   - Add explicit high-priority alias/override pairs for misses observed in this run.
   - Add lexical anchors for these actions directly in `raw_description` first sentence and `capabilities`.

1. Duplicate/overlapping tools

   - Add clear discriminator text in first sentence (channel/provider/entity).
   - Add unique capability tags for close sibling tools with same operation family.

Raw miss dump:

- `d:\Modeling_studio\POC\top10_miss_cases.json`
