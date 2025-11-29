# FamilyOS Relation Extraction Dataset

> **Version:** v2 (NEW)
> **Schema:** 15 relation types
> **Reference:** `modeling_studio.data.labels.RELATION_LABELS`

This directory contains labeled data for extracting relationships between
entities (primarily family members) in FamilyOS Unified Encoder.

## Label Schema (v2 - 15 Relations)

| ID | Label | Description | Example |
|----|-------|-------------|---------|
| 0 | no_relation | No relationship between entities | "I saw John and a dog at the park" |
| 1 | parent_of | X is parent of Y | "Mom took Emma to school" → (Mom, parent_of, Emma) |
| 2 | child_of | X is child of Y | "Panda loves her mom" → (Panda, child_of, mom) |
| 3 | spouse_of | X is married to Y | "I went with my husband to dinner" → (I, spouse_of, husband) |
| 4 | sibling_of | X is sibling of Y | "Bhai and I played cricket" → (Bhai, sibling_of, I) |
| 5 | grandparent_of | X is grandparent of Y | "Nani is visiting Emma" → (Nani, grandparent_of, Emma) |
| 6 | grandchild_of | X is grandchild of Y | "Kids visited dada yesterday" → (Kids, grandchild_of, dada) |
| 7 | aunt_uncle_of | X is aunt/uncle of Y | "Chacha brought gifts for the kids" → (Chacha, aunt_uncle_of, kids) |
| 8 | niece_nephew_of | X is niece/nephew of Y | "Emma loves visiting her masi" → (Emma, niece_nephew_of, masi) |
| 9 | cousin_of | X is cousin of Y | "Played with cousin Rohan today" → (I, cousin_of, Rohan) |
| 10 | pet_of | X is pet of Y | "Max is the family dog" → (Max, pet_of, family) |
| 11 | friend_of | X is friend of Y | "Sarah is Emma's best friend" → (Sarah, friend_of, Emma) |
| 12 | colleague_of | X works with Y | "Had lunch with my colleague Raj" → (I, colleague_of, Raj) |
| 13 | lives_at | X lives at Y (location) | "Grandma lives in Mumbai" → (Grandma, lives_at, Mumbai) |
| 14 | owns | X owns Y (heirloom) | "Dad inherited grandpa's watch" → (Dad, owns, watch) |

### Relation Categories

| Category | Relations | Description |
|----------|-----------|-------------|
| Direct Family | parent_of, child_of, spouse_of, sibling_of | Immediate family relations |
| Extended Family | grandparent_of, grandchild_of, aunt_uncle_of, niece_nephew_of, cousin_of | Extended family relations |
| Non-Family | friend_of, colleague_of | Social relationships |
| Other | pet_of, lives_at, owns | Special relationships |

## File Format

JSONL format with entity pair and relation label:

```json
{
    "text": "Mom took Panda to the park after school",
    "entity1": "Mom",
    "entity2": "Panda",
    "relation": 1
}
```

- `text`: The full sentence containing both entities
- `entity1`: The subject entity (first argument)
- `entity2`: The object entity (second argument)
- `relation`: The relation ID from entity1 to entity2

Label ID 1 = parent_of (Mom is parent of Panda)

## Files

- `train.jsonl` - Training data (~500+ examples)
- `validation.jsonl` - Validation data (~100 examples)
- `test.jsonl` - Test data (held out)

## Label Mapping

```python
from modeling_studio.data.labels import RELATION_LABELS

# Encode: "parent_of" → 1
label_id = RELATION_LABELS.encode("parent_of")

# Decode: 1 → "parent_of"
label_name = RELATION_LABELS.decode(1)

# All labels
print(RELATION_LABELS.labels)
# ['no_relation', 'parent_of', 'child_of', 'spouse_of', 'sibling_of', ...]
```

## Usage Example

```python
from modeling_studio.data.loaders import load_familyos_relations
from modeling_studio.data.labels import RELATION_LABELS

# Load training data
ds = load_familyos_relations(split="train")

# Check columns
print(ds.column_names)
# ['text', 'entity1', 'entity2', 'relation']

# View a sample
sample = ds[0]
print(f"Text: {sample['text']}")
print(f"Entity1: {sample['entity1']}")
print(f"Entity2: {sample['entity2']}")
print(f"Relation: {RELATION_LABELS.decode(sample['relation'])}")
```

## Data Guidelines

### Annotation Rules

1. **Directionality**: Relations are directional (parent_of vs child_of)
2. **Single Relation**: Each sample has exactly one relation between the entity pair
3. **Entity Spans**: Entities should match text spans exactly
4. **Inverse Relations**: Include both directions where appropriate:
   - "Mom took Emma" → (Mom, parent_of, Emma) AND (Emma, child_of, Mom)

### Entity Types Expected

- Person names (Emma, Rohan, Sarah)
- Kinship terms (Mom, Dad, Nani, Chacha, Bhai, Didi)
- Nicknames (Panda, Bunny, Sweetie)
- Pet names (Max, Whiskers)
- Locations (Mumbai, kitchen, grandma's house)
- Objects (watch, ring, album)

### Cultural Considerations

Indian kinship terms are common:

- Nani/Dadi (maternal/paternal grandmother)
- Nana/Dada (maternal/paternal grandfather)
- Masi/Bua (maternal/paternal aunt)
- Mama/Chacha (maternal/paternal uncle)
- Bhai/Didi (brother/sister)

## Quality Targets

| Metric | Target |
|--------|--------|
| F1 Score | ≥ 82% |
| Precision | ≥ 80% |
| Recall | ≥ 84% |

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v2.0 | Nov 2025 | Initial v2 release with 15 relation types |
