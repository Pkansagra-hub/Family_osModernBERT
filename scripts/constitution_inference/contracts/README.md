# Constitution Inference Contracts

This folder defines the **Standard I/O Contract** between P03 (Memory Consolidation) and the Counterfactual Decoder (13th Head).

## Overview

```
P03 (R5 Dream Phase)                    Decoder API
        │                                    │
        │  ┌──────────────────────────┐      │
        │  │    input_schema.json     │      │
        ├──┤  - event_id              ├─────►│
        │  │  - text                  │      │
        │  │  - constitution (3 layers)      │
        │  └──────────────────────────┘      │
        │                                    │
        │                              ┌─────┤
        │  ┌──────────────────────────┐│     │
        │  │   output_schema.json     ││     │
        │◄─┤  - counterfactual        ├┘     │
        │  │  - generation_meta       │      │
        │  │  - trace                 │      │
        │  └──────────────────────────┘      │
        │                                    │
```

## Files

| File | Purpose |
|------|---------|
| `input_schema.json` | JSON Schema for decoder input |
| `output_schema.json` | JSON Schema for decoder output |

## The 3 Constitution Layers

### Layer 1: Family Values (Static)
- Set during onboarding
- Stored in K0 kernel
- Example: `gentle_parenting`, `traditional_strict`

### Layer 2: Individual Preferences (Per-Actor)
- Learned by P06 or set by user
- Overrides Layer 1 for specific actors
- Example: "Dad prefers concise responses"

### Layer 3: Situational Context (Dynamic)
- Computed from current event signals
- Overrides Layers 1 & 2 for this specific event
- Example: High arousal triggers de-escalation mode

## Example Usage

### Python (Sending to Decoder)

```python
import json

# Construct input per contract
decoder_input = {
    "event_id": "evt_abc123",
    "text": "I yelled at my kids this morning",
    "constitution": {
        "family_values": {
            "key": "gentle_parenting",
            "positive_tokens": {"understand": 0.5, "feel": 0.5},
            "negative_tokens": {"punish": -0.5},
            "temperature": 0.7,
            "logits_strength": 0.0
        },
        "individual": {
            "actor_id": "user_dad",
            "response_length": "concise",
            "needs_validation_first": True
        },
        "situational": {
            "affect_arousal": 0.85,
            "affect_valence": -0.6,
            "affect_band": "RED",
            "steering_weight": 1.3,
            "force_deescalation": True
        }
    }
}

# Call decoder
output = decoder.generate(decoder_input)
```

### Expected Output

```json
{
  "event_id": "evt_abc123",
  "counterfactual": "I understand you're feeling overwhelmed. Instead of raising your voice, try taking a deep breath and saying 'I need a moment' before responding.",
  "generation_meta": {
    "constitution_applied": "gentle_parenting",
    "steering_weight_used": 1.3,
    "temperature_final": 0.55,
    "latency_ms": 120
  },
  "trace": {
    "layers_applied": ["family_values", "individual", "situational"],
    "deescalation_triggered": true,
    "empathy_prefix_injected": true
  }
}
```

## Validation

Use JSON Schema validators to ensure contract compliance:

```python
from jsonschema import validate

with open("contracts/input_schema.json") as f:
    schema = json.load(f)

validate(instance=decoder_input, schema=schema)
```
