# IFL Tool Authoring Guide for UltraBERT Retrieval

## Purpose

This guide defines the production authoring schema for capability contracts so `find capability` retrieval is accurate, stable, and explainable using UltraBERT only.

The goal is to make contract text:

1. Semantically precise for embedding search
2. Structurally consistent across IFL teams
3. Compatible with hard filter and soft rank stages in Fabric retrieval

## Retrieval-first principles

1. Keep semantic meaning in `description` and `capabilities`
2. Keep operational constraints explicit (`required_inputs`, `required_context`, `safety`, `availability`)
3. Use deterministic naming so lexical anchors are stable
4. Avoid noisy prose, slogans, and marketing language
5. Put performance/cost telemetry in ranking metadata, not verbose description text

## Canonical contract text schema (IFL v1)

Use this canonical text shape when generating embedding text for a contract:

`name:{name} | domain:{domain_tags} | provider_type:{provider_type} | operation:{operation} | display_name:{display_name} | description:{description} | capabilities:{capabilities} | required_inputs:{required_inputs} | required_context:{required_context} | safety_band_min:{safety_band_min} | availability:{availability} | tags:{tags}`

Where:

- `name` is the canonical tool identifier
- `domain_tags` is space-separated domain tags from contract `domain`
- `operation` is the terminal action token derived from tool name
- `description` is a strict single-purpose operational summary
- `capabilities` are normalized action tokens
- `required_inputs` are exact input parameter names
- `required_context` are exact context keys
- `tags` are compact routing hints (no duplicates)

## Schema-slice ablation protocol (required)

Before freezing a domain schema, run these three embedding slices and compare retrieval metrics:

### v1a

`description + capabilities + operation`

Use when you want strongest semantic focus with minimal structural priors.

### v1b

`v1a + domain`

Use when domain confusion is a measurable failure mode.

### v1c

`v1b + required_inputs`

Use when many tools are semantically similar but differ by required parameters.

### Important rule

Keep `safety_band_min` and `availability` out of embedding text for these slices.
These attributes belong in hard filter and soft ranking unless an ablation proves clear gain.

### Selection criteria

Pick the winner by this order:

1. Highest `P@5 (no-hint)`
2. Highest `MRR (no-hint)`
3. Highest `NDCG@10 (no-hint)`
4. Lowest domain confusion / false positives

Do not select a schema based only on cluster separation metrics.

## Naming convention

### Contract name grammar

Use this grammar:

`tool.{provider_type}.{domain}.{provider_id}.{operation}`

Examples:

- `tool.execute.finance.plaid_platform.retrieve_account_balance`
- `tool.read.health.epic_enterprise.retrieve_lab_results`
- `tool.write.crm.salesforce_cloud.create_campaign`

### Rules

1. Use lowercase snake_case tokens
2. `operation` must start with a verb
3. No generic names like `process_data`, `run_task`, `manage_record`
4. `provider_id` must map to real adapter identity
5. Keep names stable across versions; version in contract `version`, not in operation token

### Allowed operation verb families

- retrieve, list, search, query
- create, submit, generate
- update, patch, sync
- approve, validate, verify
- schedule, trigger, route
- detect, score, classify

## Description generation rules

### Description template

Use a deterministic 2-3 sentence format:

1. Sentence 1: primary job + domain object + provider scope
2. Sentence 2: policy/safety behavior
3. Sentence 3 (optional): failure/degraded behavior

Template:

`Performs {operation_phrase} for {domain_object} through {provider_name} in {domain} workflows.`
`Enforces {policy_controls} with auditable events and deterministic idempotency handling.`
`Supports degraded execution via {fallback_behavior} when upstream limits or timeouts occur.`

### Description quality constraints

1. Must include at least one concrete business object (for example: invoice, claim, account_balance)
2. Must include one policy or safety phrase (for example: policy checks, audit trace)
3. Must not contain subjective adjectives (for example: amazing, powerful, best)
4. Must not include unresolved placeholders
5. Maximum 70 tokens preferred for dense semantic signal

## Capability token rules

Capabilities should be normalized action phrases, 3-8 items:

- Include the operation token itself
- Include 1-2 neighboring actions
- Include 1 policy/control capability
- Include 1 error-handling capability

Example:

`capabilities: retrieve_account_balance list_transactions validate_policy retry_with_backoff emit_audit_event`

## Required inputs and context

### Required inputs

Use exact parameter names required for execution, for example:

`required_inputs: account_id tenant_id`

### Required context

Use exact session/context keys consumed by provider logic, for example:

`required_context: session.user_id session.locale`

This improves retrieval precision for intent-to-executable matching and supports hard filter pass rates.

## Anti-patterns to reject

Reject contracts that have any of these:

1. Generic operation names (`do_work`, `handle_request`)
2. Description copied from provider marketing pages
3. Missing or vague required inputs (`data`, `payload` only)
4. Overloaded multi-operation tools with conflicting semantics
5. Domain tags unrelated to contract behavior

## Validation checklist for IFL submission

Before submitting a new contract, confirm:

1. Name follows grammar and uses real provider identity
2. Operation token is specific and verb-led
3. Description follows deterministic format
4. Capabilities are normalized and non-duplicative
5. Required inputs and context are complete
6. Safety and availability are explicitly set
7. Example query can retrieve this tool in top-5 in offline benchmark

## Example contract snippets

### Finance example

- name: `tool.execute.finance.plaid_platform.retrieve_account_balance`
- description:
  - `Performs account balance retrieval for linked financial accounts through Plaid in finance workflows.`
  - `Enforces policy checks with auditable events and deterministic idempotency handling.`
  - `Supports degraded execution via cached balance fallback when upstream timeouts occur.`
- capabilities:
  - `retrieve_account_balance list_transactions validate_policy retry_with_backoff emit_audit_event`
- required_inputs:
  - `account_id tenant_id`
- required_context:
  - `session.user_id session.locale`

### Health example

- name: `tool.read.health.epic_enterprise.retrieve_lab_results`
- description:
  - `Performs laboratory result retrieval for patient encounters through Epic in health workflows.`
  - `Enforces HIPAA-aware policy checks with complete audit traceability.`
  - `Supports degraded execution via partial-result response when source systems are rate-limited.`
- capabilities:
  - `retrieve_lab_results summarize_diagnostic_history validate_policy emit_audit_event`
- required_inputs:
  - `patient_id encounter_id tenant_id`
- required_context:
  - `session.user_id session.locale`

## Rollout recommendation

1. Start with new contracts only (no bulk rewrite)
2. Add offline benchmark gate for top-5 retrieval
3. Migrate high-traffic legacy contracts in batches
4. Compare retrieval KPIs before and after each batch
5. Keep canonical schema versioned (`ifl_contract_v1`)

## Definition of done

A contract is retrieval-ready when:

1. It passes schema and naming lint
2. It passes required input/context completeness checks
3. It is retrieved in top-5 for at least 80% of canonical intent queries in offline evaluation
4. It does not increase false positive rate in neighboring domains
