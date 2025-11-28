# FamilyOS Safety Classification Dataset

This directory contains labeled data for safety policy band classification.

## Label Schema (Policy Bands)

| Band | Description | Response |
|------|-------------|----------|
| GREEN | Safe, routine content | Process normally |
| AMBER | Needs attention, mild concern | Flag for review, continue |
| RED | Serious concern | Escalate to K1, careful response |
| CRISIS | Immediate intervention needed | Emergency protocol |

## Risk Categories

- Self-harm indicators
- Abuse/harassment
- Medical emergencies
- Mental health crisis
- Child safety concerns
- Domestic violence indicators

## File Format

JSONL format:

```json
{
    "text": "I've been feeling really down lately and nothing helps",
    "band": "AMBER",
    "categories": ["mental_health"]
}
```

## Files

- `train.jsonl` - Training data
- `val.jsonl` - Validation data
- `test.jsonl` - Test data (held out)
- `calibration.jsonl` - Calibration data for threshold tuning

## Data Collection

⚠️ **Sensitive Data** - Handle with care

1. Collect examples from various safety scenarios
2. Expert annotation by trained annotators
3. Multi-annotator agreement for edge cases
4. Regular review of guidelines

## Special Considerations

- Must have sufficient CRISIS examples (min 500)
- Include Indian cultural expressions
- Distinguish venting from genuine concern
- Handle ambiguous cases conservatively

## Guidelines

See `docs/annotation/safety_guidelines.md` for detailed instructions.
