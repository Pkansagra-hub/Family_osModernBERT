# FamilyOS Safety Classification Dataset

> **Version:** v2 (Enhanced)
> **Schema:** 4 policy bands (GREEN, AMBER, RED, CRISIS)
> **Reference:** `modeling_studio.data.labels.SAFETY_FAMILYOS_LABELS`

This directory contains labeled data for safety policy band classification,
a critical component of the FamilyOS Unified Encoder for protecting family well-being.

## Label Schema (Policy Bands)

| ID | Band | Description | System Response |
|----|------|-------------|-----------------|
| 0 | GREEN | Safe, routine content | Process normally |
| 1 | AMBER | Needs attention, mild concern | Flag for review, monitor trends |
| 2 | RED | Serious concern | Escalate to K1, careful response |
| 3 | CRISIS | Immediate intervention needed | Emergency protocol activation |

## Band Details

### GREEN (ID: 0)
Safe, everyday family content requiring no special handling.

**Examples:**
- "Had a lovely dinner with the family tonight"
- "Kids played in the park after school"
- "Finished the quarterly report at work"

### AMBER (ID: 1)
Mild concern indicators that warrant monitoring but not immediate action.

**Subcategories:**
- `stress`: Work/life stress ("Feeling overwhelmed with deadlines")
- `mild_sadness`: Temporary low mood ("Today was a rough day")
- `frustration`: Family conflicts ("The kids are driving me crazy")
- `health_mention`: Non-urgent health ("My back has been hurting")

**Examples:**
- "Feeling stressed about the upcoming presentation"
- "Had a fight with my spouse about finances"
- "Kids are being really difficult lately"

### RED (ID: 2)
Serious concern requiring escalation and careful response.

**Subcategories:**
- `persistent_sadness`: Ongoing depression indicators
- `isolation`: Social withdrawal patterns
- `hopelessness`: Negative outlook on future
- `substance`: Alcohol/drug mentions

**Examples:**
- "I've been feeling empty for weeks now"
- "Don't feel like seeing anyone anymore"
- "Nothing ever seems to work out for me"
- "Had too much to drink again last night"

### CRISIS (ID: 3)
Immediate intervention required - emergency protocols.

**Subcategories:**
- `self_harm_ideation`: Thoughts of self-harm
- `suicide_ideation`: Thoughts of suicide
- `harm_to_others`: Violence indicators
- `abuse_disclosure`: Reports of abuse

**Examples:**
- "I don't want to be here anymore"
- "Sometimes I think everyone would be better off without me"
- "I want to hurt myself"

**Keyword Overrides (Force CRISIS):**
- "kill myself", "end my life", "suicide", "want to die", "hurt myself"

## File Format

JSONL format with band and optional subcategories:

```json
{
    "text": "I've been feeling really down lately and nothing helps",
    "label": 1,
    "subcategories": ["mild_sadness"]
}
```

```json
{
    "text": "Had a great weekend with the kids at the zoo",
    "label": 0,
    "subcategories": []
}
```

## Files

- `train.jsonl` - Training data (~2000+ examples)
- `validation.jsonl` - Validation data (~400 examples)
- `test.jsonl` - Test data (held out)
- `calibration.jsonl` - Calibration data for threshold tuning

## Label Mapping

```python
from modeling_studio.data.labels import SAFETY_FAMILYOS_LABELS

# Encode: label string → ID
band_id = SAFETY_FAMILYOS_LABELS.encode("AMBER")  # 1

# Decode: ID → label string
band_name = SAFETY_FAMILYOS_LABELS.decode(1)  # "AMBER"
```

## Indian English Considerations

### Normal Venting (NOT Crisis)
These are common Indian English expressions that sound alarming but are normal venting:

- "I'll die of embarrassment" → GREEN/AMBER (expression, not literal)
- "This is killing me" → AMBER (stress expression)
- "I could die" → GREEN/AMBER (hyperbole)
- "My head is bursting" → AMBER (stress/headache)
- "I'm going mad" → AMBER (frustration)

### Cultural Context
- Joint family stress is common but usually AMBER not RED
- "Tension" in Indian English means worry/stress
- Festival stress (Diwali, wedding season) is typically AMBER

## Data Collection Guidelines

⚠️ **Sensitive Data** - Handle with extreme care

1. **Class balance**: Ensure sufficient CRISIS examples (min 500)
2. **Cultural coverage**: Include Indian English expressions
3. **Distinguish venting from genuine concern**: Context matters
4. **Conservative labeling**: When in doubt, label higher risk
5. **Multi-annotator agreement**: Required for RED/CRISIS examples
6. **Expert review**: All CRISIS examples reviewed by mental health expert

## Evaluation Priorities

| Band | Priority | Key Metric | Target |
|------|----------|------------|--------|
| CRISIS | Highest | Recall | ≥ 98% |
| RED | High | Recall | ≥ 95% |
| AMBER | Medium | F1 | ≥ 85% |
| GREEN | Standard | Precision | ≥ 90% |

**Critical:** False negatives on CRISIS are unacceptable. Optimize for high recall.

## Temporal Safety Monitoring

The safety system also monitors trends over time:

| Pattern | Escalation |
|---------|------------|
| 3+ AMBER in 7 days | → RED |
| RED + isolation keywords | → CRISIS |
| Declining sentiment trend | → Increase monitoring |

## Quality Targets

| Metric | Target |
|--------|--------|
| CRISIS Recall | ≥ 98% |
| RED Recall | ≥ 95% |
| Overall Accuracy | ≥ 90% |
| False Negative Rate (CRISIS) | < 2% |
