# FamilyOS Emotions Dataset

Multi-label emotion classification dataset tailored for family diary entries, conversations, and personal reflections.

## Overview

| Attribute | Value |
|-----------|-------|
| **Task** | Multi-label Classification |
| **Classes** | 44 emotions |
| **Target Size** | 50,000+ samples |
| **Format** | JSONL |
| **Source** | Synthetic (LLM-generated) |

## Emotion Schema (44 Classes)

### Core Emotions (8)
Basic emotions common across all contexts.

| ID | Emotion | Description | Example |
|----|---------|-------------|---------|
| 0 | `neutral` | No strong emotion | "The meeting is at 3pm" |
| 1 | `joy` | Happiness, delight | "Emma took her first steps!" |
| 2 | `sadness` | Sorrow, unhappiness | "Missing grandpa today" |
| 3 | `anger` | Strong displeasure | "Can't believe they did that" |
| 4 | `fear` | Anxiety, worry | "Scared about the test results" |
| 5 | `surprise` | Unexpected reaction | "Didn't expect them to visit!" |
| 6 | `love` | Deep affection | "My heart is so full" |
| 7 | `disgust` | Strong aversion | "That behavior is unacceptable" |

### Positive Emotions (12)
Emotions associated with positive experiences.

| ID | Emotion | Description | Example |
|----|---------|-------------|---------|
| 8 | `admiration` | Respect, appreciation | "So impressed by her dedication" |
| 9 | `amusement` | Finding something funny | "Kids said the funniest thing" |
| 10 | `approval` | Agreeing, endorsing | "That's exactly the right approach" |
| 11 | `caring` | Showing concern | "Hope mom feels better soon" |
| 12 | `excitement` | Eager anticipation | "Can't wait for the trip!" |
| 13 | `gratitude` | Thankfulness | "So grateful for my family" |
| 14 | `optimism` | Hopeful outlook | "Things will get better" |
| 15 | `pride` | Satisfaction in achievement | "So proud of what she accomplished" |
| 16 | `relief` | Ease after worry | "Thank god the surgery went well" |
| 17 | `contentment` | Peaceful satisfaction | "Just enjoying this quiet moment" |
| 18 | `hope` | Wish for positive outcome | "Hoping for good news tomorrow" |
| 19 | `tenderness` | Gentle affection | "Watching them sleep is precious" |

### Negative Emotions (10)
Emotions associated with negative experiences.

| ID | Emotion | Description | Example |
|----|---------|-------------|---------|
| 20 | `annoyance` | Mild irritation | "Traffic was terrible again" |
| 21 | `disappointment` | Unmet expectations | "Wish they could have made it" |
| 22 | `disapproval` | Disagreement | "That's not how we do things" |
| 23 | `embarrassment` | Self-consciousness | "Can't believe I said that" |
| 24 | `grief` | Deep sorrow from loss | "Still processing the loss" |
| 25 | `nervousness` | Anxious anticipation | "Nervous about the interview" |
| 26 | `remorse` | Regret, guilt | "Wish I had been more patient" |
| 27 | `frustration` | Blocked goals | "Nothing seems to be working" |
| 28 | `overwhelmed` | Too much to handle | "So much going on right now" |
| 29 | `emptiness` | Feeling void | "House feels so quiet now" |

### Family-Specific Emotions (14)
Emotions particularly relevant to family contexts.

| ID | Emotion | Description | Example |
|----|---------|-------------|---------|
| 30 | `nostalgia` | Fond memories of past | "Remember our first family trip?" |
| 31 | `protectiveness` | Urge to keep safe | "Just want to shield them from hurt" |
| 32 | `togetherness` | Family bonding | "Love when we're all together" |
| 33 | `longing` | Missing someone/something | "Wish mom was here to see this" |
| 34 | `warmth` | Comfortable affection | "Sunday dinners are the best" |
| 35 | `playfulness` | Lighthearted fun | "Had a pillow fight with the kids" |
| 36 | `celebration` | Marking achievements | "Time to celebrate this milestone!" |
| 37 | `belonging` | Feeling part of family | "This is where I'm meant to be" |
| 38 | `parental_pride` | Pride in children | "Look how far they've come" |
| 39 | `parental_guilt` | Guilt about parenting | "Should have spent more time with them" |
| 40 | `patience` | Calm endurance | "Deep breaths, they're just kids" |
| 41 | `worry` | Concern for loved ones | "Can't stop thinking about dad's health" |
| 42 | `bittersweet` | Mixed happy-sad | "They're growing up so fast" |
| 43 | `homesickness` | Missing home/family | "Wish I could be there with everyone" |

---

## Data Format

### JSONL Schema

```json
{
  "text": "Emma took her first steps today and I cried happy tears!",
  "emotions": ["joy", "pride", "love", "excitement"],
  "primary_emotion": "joy",
  "intensity": "high",
  "context": "milestone"
}
```

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | ✅ | The diary entry or message |
| `emotions` | list[string] | ✅ | Multi-label emotion tags (1-4 typical) |
| `primary_emotion` | string | ✅ | Dominant emotion |
| `intensity` | string | ❌ | low / medium / high |
| `context` | string | ❌ | Category (see below) |

### Context Categories

- `milestone` - First steps, graduations, achievements
- `daily_life` - Routines, meals, activities
- `health` - Medical, wellness concerns
- `conflict` - Arguments, disagreements
- `celebration` - Birthdays, holidays, events
- `memory` - Recalling past events
- `loss` - Death, separation, endings
- `relationship` - Family dynamics, connections
- `parenting` - Child-rearing moments
- `self_reflection` - Personal thoughts

---

## Directory Structure

```
data/familyos/emotions/
├── README.md           # This file
├── gold/
│   ├── train.jsonl     # Manually curated (target: 500)
│   └── validation.jsonl # Manually curated (target: 100)
└── silver/
    ├── shard_0000.jsonl
    ├── shard_0001.jsonl
    └── ...             # LLM-generated (target: 50,000)
```

---

## Generation Guidelines

### For LLM Data Generator

1. **Multi-label**: Most samples should have 2-4 emotions (real feelings are complex)
2. **Primary emotion**: Always identify the dominant one
3. **Cultural diversity**: Include Indian family contexts (~30%)
   - Use: didi, bhai, nana, nani, mummy, papa, Diwali, etc.
4. **Realistic language**: Natural diary/conversation style
5. **Balanced distribution**: Ensure all 44 emotions are represented
6. **Intensity variation**: Mix low, medium, high intensity

### Quality Criteria

✅ **Good sample:**
```json
{
  "text": "Mummy called today and hearing her voice made me so happy but also made me miss home terribly",
  "emotions": ["joy", "love", "homesickness", "longing"],
  "primary_emotion": "homesickness",
  "intensity": "high",
  "context": "relationship"
}
```

❌ **Bad sample:**
```json
{
  "text": "Had food",
  "emotions": ["neutral"],
  "primary_emotion": "neutral"
}
```
(Too short, no emotional depth)

---

## Co-occurrence Patterns

Common emotion combinations in family contexts:

| Pattern | Emotions | Example Trigger |
|---------|----------|-----------------|
| Proud parent | pride, joy, love | Child's achievement |
| Bittersweet growth | bittersweet, nostalgia, pride | Kids growing up |
| Family gathering | togetherness, warmth, joy | Holiday dinner |
| Missing family | longing, homesickness, sadness | Living far away |
| Parenting stress | overwhelmed, frustration, guilt | Difficult day |
| Loss processing | grief, sadness, emptiness | Death anniversary |
| Health worry | worry, fear, nervousness | Medical tests |
| Celebration | celebration, excitement, joy | Birthday party |

---

## Training Configuration

### Stage C: Emotion Specialization

```yaml
# configs/training/multitask/stage_c_emotions.yaml
base_model: checkpoints/modernbert-multitask-v1  # Stage B output

peft:
  method: lora
  r: 16
  alpha: 32
  target_modules: [q_proj, v_proj]
  dropout: 0.05

training:
  num_epochs: 5
  learning_rate: 5e-5
  batch_size: 32

  # Focus on emotions but prevent forgetting
  task_weights:
    emotions: 5.0
    safety_familyos: 1.0  # Critical - don't forget
    intent: 0.5
    sentiment: 0.3

datasets:
  familyos_emotions_silver:
    task: emotions
    source: local
    data_dir: data/familyos/emotions/silver
    format: jsonl
    max_samples: 50000

  familyos_emotions_gold:
    task: emotions
    source: local
    data_dir: data/familyos/emotions/gold
    format: jsonl
```

---

## Evaluation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| **Macro F1** | ≥ 65% | Average F1 across all 44 emotions |
| **Micro F1** | ≥ 75% | Overall precision/recall |
| **Subset Accuracy** | ≥ 40% | Exact match of all emotions |
| **Hamming Loss** | ≤ 0.05 | Per-label error rate |

### Per-Category Targets

| Category | Macro F1 Target |
|----------|-----------------|
| Core (8) | ≥ 75% |
| Positive (12) | ≥ 65% |
| Negative (10) | ≥ 60% |
| Family-Specific (14) | ≥ 60% |

---

## Comparison: VADER vs Model

After training, compare both approaches:

```bash
python scripts/compare_vader_model_emotions.py \
    --model checkpoints/modernbert-multitask-v2 \
    --output outputs/vader_vs_model_emotions.json
```

### Expected Outcomes

| Approach | Pros | Cons |
|----------|------|------|
| **VADER** | Fast (0.1ms), no GPU | Only valence, no nuance |
| **Model** | 44 emotions, context-aware | Slower (15ms), needs GPU |

**Recommendation**: Use model for nuanced analysis, VADER for quick sentiment.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Dec 2025 | Initial 44-emotion schema |

---

## References

- GoEmotions: [Google Research](https://github.com/google-research/google-research/tree/master/goemotions)
- Plutchik's Wheel of Emotions
- FamilyOS unified_encoder_solution.md
