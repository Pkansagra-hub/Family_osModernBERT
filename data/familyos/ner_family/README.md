# FamilyOS Family NER Dataset

> **Version:** v2 (Enhanced)
> **Schema:** 21 BIO tags (expanded from 15)
> **Reference:** `modeling_studio.data.labels.NER_FAMILY_LABELS`

This directory contains annotated data for family-specific named entity recognition,
designed for the FamilyOS Unified Encoder multi-task model.

## Label Schema (v2 Enhanced - 21 BIO Tags)

| ID | Label | Description | Examples |
|----|-------|-------------|----------|
| 0 | O | Outside any entity | Common words |
| 1-2 | B/I-PERSON | Named individuals | "John Smith", "Sarah" |
| 3-4 | B/I-KINSHIP | Family relationship terms | "mom", "dad", "didi", "nana", "bhai" |
| 5-6 | B/I-NICKNAME | Family nicknames | "Panda", "Bunny", "Sweetie", "Baby Bear" |
| 7-8 | B/I-PET | Pet names and references | "Max", "Whiskers", "our dog" |
| 9-10 | B/I-HOME_LOC | Locations within home | "kitchen", "Emma's room", "backyard" |
| 11-12 | B/I-FAMILY_EVENT | Family occasions | "birthday party", "anniversary", "graduation" |
| 13-14 | B/I-ROUTINE | Regular activities | "school run", "dinner time", "bedtime story" |
| 15-16 | B/I-TRADITION | Recurring family rituals (NEW v2) | "Sunday brunch", "movie night", "Diwali celebration" |
| 17-18 | B/I-MILESTONE | Life events to remember (NEW v2) | "first steps", "graduation day", "lost first tooth" |
| 19-20 | B/I-HEIRLOOM | Sentimental objects (NEW v2) | "grandma's necklace", "dad's watch", "family photo album" |

### New v2 Entity Types

| Entity | Description | Why Important |
|--------|-------------|---------------|
| TRADITION | Recurring family rituals | Weekly/annual customs worth tracking |
| MILESTONE | Significant life events | Memory preservation, timeline building |
| HEIRLOOM | Sentimental family objects | Items with emotional significance |

## File Format

JSONL format with BIO tags (integer IDs from `NER_FAMILY_LABELS`):

```json
{
    "tokens": ["Panda", "took", "her", "first", "steps", "in", "the", "kitchen"],
    "ner_tags": [5, 0, 0, 17, 18, 0, 0, 9]
}
```

Decoded as: `["B-NICKNAME", "O", "O", "B-MILESTONE", "I-MILESTONE", "O", "O", "B-HOME_LOC"]`

## Files

- `train.jsonl` - Training data (~500+ examples)
- `validation.jsonl` - Validation data (~100 examples)
- `test.jsonl` - Test data (held out)

## Label Mapping

```python
from modeling_studio.data.labels import NER_FAMILY_LABELS

# Encode: label string → ID
tag_id = NER_FAMILY_LABELS.encode("B-NICKNAME")  # 5

# Decode: ID → label string
tag_name = NER_FAMILY_LABELS.decode(5)  # "B-NICKNAME"
```

## Cultural Coverage

### Indian English Kinship Terms
- "didi" (elder sister), "bhai" (brother), "nana/nani" (maternal grandparents)
- "dada/dadi" (paternal grandparents), "chacha/chachi" (paternal uncle/aunt)
- "mama/masi" (maternal uncle/aunt), "bua" (paternal aunt)

### Western Kinship Terms
- "mom/mum/mother", "dad/daddy/father", "grandma/granny", "grandpa"
- "aunt/auntie", "uncle", "cousin", "sis/sister", "bro/brother"

## Annotation Guidelines

### Entity Boundaries
- Include full entity spans: "Panda's birthday party" → B-NICKNAME + B-FAMILY_EVENT I-FAMILY_EVENT
- Multi-word entities use I- continuation tags

### Ambiguous Cases
- "mom" alone → B-KINSHIP
- "Mom Sarah" → B-KINSHIP B-PERSON (if both are meaningful)
- "Sunday brunch tradition" → B-TRADITION I-TRADITION I-TRADITION

### BIO Validation Rules
- I-tags must immediately follow corresponding B-tags
- No orphan I-tags (must have preceding B-tag of same type)

## Quality Targets

| Metric | Target |
|--------|--------|
| F1 Score | ≥ 88% |
| Entity Coverage | All 10 types represented |
| Inter-annotator Agreement | κ ≥ 0.85 |
