# FamilyOS Intent Classification Dataset

> **Version:** v2 (NEW)
> **Schema:** 8 intent types
> **Reference:** `modeling_studio.data.labels.INTENT_LABELS`

This directory contains labeled data for classifying user intent when
interacting with FamilyOS. Understanding intent helps route messages
to appropriate handlers and provide better responses.

## Label Schema (v2 - 8 Intents)

| ID | Label | Description | Examples |
|----|-------|-------------|----------|
| 0 | log_memory | Store/record information | "Had dinner with family tonight" |
| 1 | query_memory | Retrieve past information | "What did we do last Sunday?" |
| 2 | set_reminder | Create a reminder/task | "Remind me to call mom tomorrow" |
| 3 | express_feeling | Share emotions/feelings | "Feeling grateful today" |
| 4 | seek_advice | Ask for guidance/help | "What should I do about..." |
| 5 | share_news | Announce/share updates | "Guess what happened today!" |
| 6 | reflect | Contemplation/musing | "Thinking about the past..." |
| 7 | other | Catch-all for misc | General conversation |

### Intent Categories

| Category | Intents | Description |
|----------|---------|-------------|
| Memory Operations | log_memory, query_memory | Storing and retrieving memories |
| Action Requests | set_reminder, seek_advice | User wants system to do something |
| Emotional Expression | express_feeling, reflect | Sharing feelings or contemplating |
| Information Sharing | share_news | Announcing updates or news |
| Miscellaneous | other | Catch-all category |

## File Format

JSONL format with intent label:

```json
{
    "text": "Had a lovely dinner with the family tonight at Olive Garden",
    "label": 0
}
```

Label ID 0 = log_memory (user is recording a memory)

## Files

- `train.jsonl` - Training data (~500+ examples)
- `validation.jsonl` - Validation data (~100 examples)
- `test.jsonl` - Test data (held out)

## Label Mapping

```python
from modeling_studio.data.labels import INTENT_LABELS

# Encode: "log_memory" → 0
label_id = INTENT_LABELS.encode("log_memory")

# Decode: 0 → "log_memory"
label_name = INTENT_LABELS.decode(0)

# All labels
print(INTENT_LABELS.labels)
# ['log_memory', 'query_memory', 'set_reminder', 'express_feeling',
#  'seek_advice', 'share_news', 'reflect', 'other']
```

## Usage Example

```python
from modeling_studio.data.loaders import load_familyos_intents
from modeling_studio.data.labels import INTENT_LABELS

# Load training data
ds = load_familyos_intents(split="train")

# Check columns
print(ds.column_names)
# ['text', 'label']

# View a sample
sample = ds[0]
print(f"Text: {sample['text']}")
print(f"Intent: {INTENT_LABELS.decode(sample['label'])}")
```

## Data Guidelines

### Annotation Rules

1. **Single Intent**: Each message has exactly one primary intent
2. **Context-Free**: Classify based on the message alone, not conversation history
3. **User-Centric**: Focus on what the user wants to accomplish
4. **Ambiguous Cases**: If unclear between intents, prefer the more specific one

### Intent Descriptions

| Intent | Key Phrases | NOT This Intent |
|--------|-------------|-----------------|
| log_memory | "Today I...", "We had...", "Just finished..." | Questions about past |
| query_memory | "What did...", "When was...", "Do you remember..." | Statements about past |
| set_reminder | "Remind me...", "Don't let me forget...", "Schedule..." | Just mentioning plans |
| express_feeling | "I feel...", "So happy/sad...", "Grateful for..." | Sharing news (excited) |
| seek_advice | "What should I...", "Help me decide...", "Any suggestions?" | Rhetorical questions |
| share_news | "Guess what!", "Great news!", "You won't believe..." | Just logging events |
| reflect | "I've been thinking...", "Looking back...", "Life is..." | Expressing current feelings |
| other | Greetings, meta questions, off-topic | Anything fitting above |

### Cultural Considerations

Indian English patterns are common:
- "Did the needful" = completed a task (log_memory)
- "Kindly remind me" = polite reminder request (set_reminder)
- "What all happened" = query about events (query_memory)

## Quality Targets

| Metric | Target |
|--------|--------|
| Accuracy | ≥ 85% |
| Macro F1 | ≥ 82% |
| Per-class F1 | ≥ 75% each |

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v2.0 | Nov 2025 | Initial v2 release with 8 intent types |
