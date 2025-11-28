# FamilyOS Ingress Classification Dataset

This directory contains labeled data for domain/topic classification
of incoming text in FamilyOS.

## Label Schema

| Label | Description | Examples |
|-------|-------------|----------|
| DIARY | Personal reflections, journaling | "Today I felt really happy about..." |
| TASK | To-dos, reminders, action items | "Remember to pick up groceries" |
| HEALTH | Medical, wellness, fitness | "Doctor appointment at 3pm" |
| FINANCE | Money, bills, budgets | "Need to pay electricity bill" |
| RELATIONSHIP | Family dynamics, social | "Had a great chat with mum" |
| WORK | Job, career, professional | "Meeting with boss tomorrow" |
| META | System commands, queries | "What tasks do I have today?" |

## File Format

JSONL format:

```json
{
    "text": "I need to remember to call the dentist tomorrow",
    "domain": "HEALTH"
}
```

## Files

- `train.jsonl` - Training data
- `val.jsonl` - Validation data  
- `test.jsonl` - Test data (held out)

## Data Collection

1. Retrospectively label user conversation logs
2. Use LLM bootstrapping for initial labels
3. Human review and correction
4. Balance classes through sampling/augmentation

## Guidelines

See `docs/annotation/ingress_guidelines.md` for labeling instructions.
