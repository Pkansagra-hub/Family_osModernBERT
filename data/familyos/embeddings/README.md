# FamilyOS Embedding Clusters Dataset

> **Version:** v2 (Enhanced)
> **Output:** 768-dim vectors
> **Reference:** Embedding capability in Unified Encoder

This directory contains curated text clusters for validating
embedding quality and semantic similarity in the FamilyOS Unified Encoder.

## Purpose

After training, embeddings should cluster texts about similar topics.
This dataset provides known clusters for:

1. **Sanity checking** - Verify embeddings capture semantic similarity
2. **Retrieval testing** - Test memory search and recall accuracy
3. **Cluster validation** - Ensure family-relevant topics are well-separated

## Cluster Categories

### immigration_docs
H-1B visa, USCIS, immigration-related texts common in Indian-American families.

**Examples:**
- "Need to file the I-140 petition by next month"
- "USCIS receipt notice came in the mail today"
- "H-1B extension is pending, waiting anxiously"

### child_activities
School, activities, homework, playdates for children.

**Examples:**
- "Emma has soccer practice at 4pm today"
- "Need to help Panda with her math homework"
- "Planning a playdate with the neighbors' kids"

### health_records
Medical appointments, symptoms, medications.

**Examples:**
- "Doctor's appointment scheduled for Thursday"
- "Need to refill the blood pressure medication"
- "Kids' annual checkup is overdue"

### financial_planning
Budgets, investments, bills, financial decisions.

**Examples:**
- "Reviewed the 401k allocations today"
- "Property tax bill is due next week"
- "Need to start saving for college fund"

### family_memories
Photos, trips, celebrations, nostalgic content.

**Examples:**
- "Looking at old photos from the Goa trip"
- "Remember when Emma took her first steps?"
- "Can't believe it's been 10 years since our wedding"

### daily_routines (NEW v2)
Morning routines, school runs, meal times.

**Examples:**
- "Morning school run was hectic today"
- "Dinner time is always chaotic with the kids"
- "Bedtime story took forever tonight"

### family_traditions (NEW v2)
Weekly/annual family customs and rituals.

**Examples:**
- "Sunday brunch at grandma's house"
- "Getting ready for our annual Diwali celebration"
- "Movie night this Friday with the whole family"

### emotional_support (NEW v2)
Comfort, advice, family support conversations.

**Examples:**
- "Mom always knows what to say when I'm stressed"
- "Dad gave me great career advice yesterday"
- "So grateful for my supportive family"

## File Format

JSONL with cluster labels:

```json
{
    "text": "Need to file the I-140 petition by next month",
    "cluster": "immigration_docs"
}
```

## Files

- `clusters.jsonl` - All clustered examples (~50+ per cluster)
- `pairs.jsonl` - Positive/negative pairs for contrastive evaluation
- `triplets.jsonl` - Anchor/positive/negative triplets

### Pairs Format
```json
{
    "sentence1": "Emma has soccer practice today",
    "sentence2": "Kids have sports activities this afternoon",
    "score": 0.85
}
```

### Triplets Format
```json
{
    "anchor": "Need to file the I-140 petition",
    "positive": "H-1B extension paperwork is due",
    "negative": "Emma has soccer practice today"
}
```

## Evaluation Metrics

### Cluster Coherence
After training, verify:

1. **Intra-cluster similarity**: Texts in same cluster have cosine similarity > 0.7
2. **Inter-cluster separation**: Texts in different clusters have cosine similarity < 0.4
3. **Retrieval recall**: Query from cluster retrieves same-cluster texts in top-5

### Contrastive Evaluation
```python
from sklearn.metrics.pairwise import cosine_similarity

# Positive pair should have high similarity
assert cosine_similarity(embed(s1), embed(s2)) > 0.7

# Negative pair should have lower similarity
assert cosine_similarity(embed(anchor), embed(negative)) < 0.5
```

## Hard Negatives

For robust embedding training, include hard negatives:

| Type | Description | Example |
|------|-------------|---------|
| Same person, different event | Confusing entity overlap | "Emma's birthday" vs "Emma's school play" |
| Same event type, different family | Confusing topic overlap | "Our wedding anniversary" vs "My friend's anniversary" |
| Temporal neighbors | Confusing time proximity | "Last Sunday's brunch" vs "This Sunday's brunch" |

## Quality Targets

| Metric | Target |
|--------|--------|
| Intra-cluster Cosine Sim | > 0.70 |
| Inter-cluster Cosine Sim | < 0.40 |
| Retrieval Recall@5 | > 85% |
| Triplet Accuracy | > 90% |
