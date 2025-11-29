# FamilyOS Temporal Expression Extraction Dataset

> **Version:** v2 (NEW)
> **Schema:** 13 BIO tags (6 entity types)
> **Reference:** `modeling_studio.data.labels.TEMPORAL_LABELS`

This directory contains labeled data for extracting temporal expressions
from text. Temporal extraction is essential for timeline construction,
event scheduling, and memory organization in FamilyOS.

## Label Schema (v2 - 13 BIO Tags)

| ID | Label | Description | Examples |
|----|-------|-------------|----------|
| 0 | O | Outside any temporal entity | Regular words |
| 1 | B-DATE_ABS | Begin absolute date | "January 15", "2024", "March 5th" |
| 2 | I-DATE_ABS | Inside absolute date | Multi-word dates |
| 3 | B-DATE_REL | Begin relative date | "yesterday", "last week", "next month" |
| 4 | I-DATE_REL | Inside relative date | "last week" (week) |
| 5 | B-TIME | Begin time expression | "3pm", "morning", "at noon" |
| 6 | I-TIME | Inside time expression | "3 o'clock" (o'clock) |
| 7 | B-DURATION | Begin duration | "for 2 hours", "all day", "5 minutes" |
| 8 | I-DURATION | Inside duration | "2 hours" (hours) |
| 9 | B-FREQUENCY | Begin frequency/recurring | "every Sunday", "weekly", "daily" |
| 10 | I-FREQUENCY | Inside frequency | "every Sunday" (Sunday) |
| 11 | B-AGE | Begin age/life period | "when she was 5", "in my 20s" |
| 12 | I-AGE | Inside age | "she was 5" (was 5) |

### Temporal Entity Types

| Type | Description | Examples |
|------|-------------|----------|
| DATE_ABS | Specific calendar dates | "January 15, 2024", "March 5th", "2023" |
| DATE_REL | Relative to current time | "yesterday", "last week", "tomorrow", "next month" |
| TIME | Time of day | "3pm", "morning", "at noon", "8:30 AM" |
| DURATION | Length of time | "for 2 hours", "all day", "5 minutes", "a week" |
| FREQUENCY | Recurring patterns | "every Sunday", "weekly", "twice a day", "annually" |
| AGE | Age or life period | "when she was 5", "at age 10", "in my 20s" |

## File Format

JSONL format with tokens and temporal tags (token classification):

```json
{
    "tokens": ["We", "went", "to", "the", "park", "yesterday", "morning"],
    "temporal_tags": [0, 0, 0, 0, 0, 3, 5]
}
```

- `tokens`: List of words/tokens
- `temporal_tags`: List of BIO tag IDs (same length as tokens)

Tag IDs: 0=O, 3=B-DATE_REL ("yesterday"), 5=B-TIME ("morning")

## Files

- `train.jsonl` - Training data (~300+ examples)
- `validation.jsonl` - Validation data (~50 examples)
- `test.jsonl` - Test data (held out)

## Label Mapping

```python
from modeling_studio.data.labels import TEMPORAL_LABELS

# Encode: "B-DATE_REL" → 3
label_id = TEMPORAL_LABELS.encode("B-DATE_REL")

# Decode: 3 → "B-DATE_REL"
label_name = TEMPORAL_LABELS.decode(3)

# All labels
print(TEMPORAL_LABELS.labels)
# ['O', 'B-DATE_ABS', 'I-DATE_ABS', 'B-DATE_REL', 'I-DATE_REL',
#  'B-TIME', 'I-TIME', 'B-DURATION', 'I-DURATION',
#  'B-FREQUENCY', 'I-FREQUENCY', 'B-AGE', 'I-AGE']
```

## Usage Example

```python
from modeling_studio.data.loaders import load_familyos_temporal
from modeling_studio.data.labels import TEMPORAL_LABELS

# Load training data
ds = load_familyos_temporal(split="train")

# Check columns
print(ds.column_names)
# ['tokens', 'temporal_tags']

# View a sample with decoded tags
sample = ds[0]
for token, tag_id in zip(sample['tokens'], sample['temporal_tags']):
    tag = TEMPORAL_LABELS.decode(tag_id)
    if tag != 'O':
        print(f"  {token}: {tag}")
```

## Data Guidelines

### Annotation Rules

1. **BIO Scheme**: Use B- prefix for first token, I- prefix for continuation
2. **Minimal Span**: Tag only the temporal expression itself
3. **Context Words**: Don't tag prepositions like "on", "at", "in" unless part of expression
4. **Ambiguity**: When unclear, prefer the more specific type

### BIO Tagging Examples

```
"We visited on January 15th"
 O    O      O  B-DATE_ABS I-DATE_ABS

"Meet me at 3pm tomorrow"
 O    O  O  B-TIME B-DATE_REL

"Every Sunday we have brunch"
 B-FREQUENCY I-FREQUENCY O O O

"She learned to walk when she was 1"
 O   O       O  O    O    B-AGE I-AGE I-AGE

"The trip lasted for 2 weeks"
 O   O    O      O   B-DURATION I-DURATION
```

### Common Patterns

| Pattern | Type | Example |
|---------|------|---------|
| Day names | DATE_REL or FREQUENCY | "Sunday" (context-dependent) |
| Month + day | DATE_ABS | "March 5th" |
| Time with AM/PM | TIME | "3:30 PM" |
| "ago" expressions | DATE_REL | "2 days ago" |
| "every X" | FREQUENCY | "every morning" |
| "when X was Y" | AGE | "when she was little" |

### Cultural Considerations

Indian date formats are common:
- "5th March" (day before month)
- "Diwali day" → treat as DATE_REL if referring to upcoming/past
- Festival references may need context

## Quality Targets

| Metric | Target |
|--------|--------|
| F1 Score | ≥ 85% |
| Precision | ≥ 83% |
| Recall | ≥ 87% |

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v2.0 | Nov 2025 | Initial v2 release with 13 BIO tags |
