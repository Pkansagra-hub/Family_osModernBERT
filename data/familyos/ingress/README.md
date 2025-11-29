# FamilyOS Ingress Classification Dataset

> **Version:** v2 (Enhanced)
> **Schema:** 12 domains (expanded from 7)
> **Reference:** `modeling_studio.data.labels.INGRESS_LABELS`

This directory contains labeled data for domain/topic classification
of incoming text in FamilyOS Unified Encoder.

## Label Schema (v2 Enhanced - 12 Domains)

| ID | Label | Description | Examples |
|----|-------|-------------|----------|
| 0 | DIARY | Personal reflections, journaling | "Today I felt really happy about the weather" |
| 1 | TASK | To-dos, reminders, action items | "Remember to pick up groceries tomorrow" |
| 2 | HEALTH | Medical, wellness, fitness | "Doctor appointment at 3pm for annual checkup" |
| 3 | FINANCE | Money, bills, budgets | "Need to pay electricity bill by Friday" |
| 4 | RELATIONSHIP | Family dynamics, social | "Had a great chat with mum this evening" |
| 5 | WORK | Job, career, professional | "Meeting with boss tomorrow about the project" |
| 6 | META | System commands, queries about FamilyOS | "What tasks do I have today?" |
| 7 | MEMORY | Recalling past events (NEW v2) | "Remember when we went to Goa last summer?" |
| 8 | PLANNING | Future events (NEW v2) | "Next week we should visit grandma" |
| 9 | CELEBRATION | Birthdays, achievements, milestones (NEW v2) | "Emma got straight A's on her report card!" |
| 10 | CONCERN | Worries, anxieties (NEW v2) | "I'm a bit worried about dad's health lately" |
| 11 | GRATITUDE | Appreciation expressions (NEW v2) | "So thankful for my wonderful family" |

### New v2 Domains

| Domain | Description | Why Important |
|--------|-------------|---------------|
| MEMORY | Recalling past events | Memory retrieval, nostalgia tracking |
| PLANNING | Future event discussion | Future event detection, calendar integration |
| CELEBRATION | Milestones and achievements | Milestone detection, positive reinforcement |
| CONCERN | Worries and anxieties | Early amber signal detection, safety integration |
| GRATITUDE | Appreciation expressions | Positive sentiment tracking, well-being indicators |

## File Format

JSONL format with domain label:

```json
{
    "text": "I need to remember to call the dentist tomorrow about Emma's braces",
    "label": 2
}
```

Label ID 2 = HEALTH

## Files

- `train.jsonl` - Training data (~1000+ examples)
- `validation.jsonl` - Validation data (~200 examples)
- `test.jsonl` - Test data (held out)

## Label Mapping

```python
from modeling_studio.data.labels import INGRESS_LABELS

# Encode: label string → ID
domain_id = INGRESS_LABELS.encode("HEALTH")  # 2

# Decode: ID → label string
domain_name = INGRESS_LABELS.decode(2)  # "HEALTH"
```

## Domain Disambiguation

### MEMORY vs DIARY


- **MEMORY**: Explicitly recalling past events ("Remember when...", "That time when...")
- **DIARY**: Current reflections, not explicitly recalling ("Today was a good day")


### CONCERN vs HEALTH

- **CONCERN**: Emotional worry about something ("I'm worried about...")
- **HEALTH**: Factual medical/wellness information ("Doctor appointment...")


### CELEBRATION vs RELATIONSHIP

- **CELEBRATION**: Achievement or milestone focus ("Emma won the spelling bee!")

- **RELATIONSHIP**: Interpersonal dynamics ("Had lunch with mom")

### PLANNING vs TASK

- **PLANNING**: Future event discussion, often family-related ("Let's go to the beach next weekend")
- **TASK**: Specific action items with clear deliverables ("Buy milk")

## Data Collection Guidelines

1. **Balance classes**: Ensure representation across all 12 domains
2. **Realistic language**: Use natural family conversation patterns
3. **Cultural diversity**: Include Indian English expressions
4. **Edge cases**: Include ambiguous examples for boundary learning

## Quality Targets

| Metric | Target |
|--------|--------|
| Accuracy | ≥ 92% |
| Macro F1 | ≥ 88% |
| Per-class Recall | ≥ 80% for each domain |
