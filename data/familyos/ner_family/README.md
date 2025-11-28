# FamilyOS Family NER Dataset

This directory contains annotated data for family-specific named entity recognition.

## Label Schema

| Label | Description | Examples |
|-------|-------------|----------|
| PERSON | Named individuals | "John", "Sarah" |
| KINSHIP | Family relationship terms | "mum", "dad", "uncle", "sister" |
| NICKNAME | Family nicknames | "Panda", "Bunny", "Sweetie" |
| PET | Pet names | "Max", "Whiskers" |
| HOME_LOCATION | Locations within home | "kitchen", "bedroom", "garden" |
| FAMILY_EVENT | Family occasions | "birthday", "anniversary" |
| ROUTINE | Regular activities | "school run", "dinner time" |

## File Format

JSONL format with BIO tags:

```json
{
    "tokens": ["Panda", "is", "in", "the", "kitchen"],
    "ner_tags": ["B-NICKNAME", "O", "O", "O", "B-HOME_LOCATION"]
}
```

## Files

- `train.jsonl` - Training data
- `val.jsonl` - Validation data
- `test.jsonl` - Test data (held out)

## Data Collection

1. Sample conversation logs (synthetic or real)
2. Annotate with family NER labels
3. Review and quality check
4. Split into train/val/test

## Guidelines

See `docs/annotation/family_ner_guidelines.md` for annotation instructions.
