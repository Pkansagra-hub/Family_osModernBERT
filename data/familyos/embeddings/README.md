# FamilyOS Embedding Clusters

This directory contains curated text clusters for validating
embedding quality and semantic similarity.

## Purpose

After training, embeddings should cluster texts about similar topics.
This dataset provides known clusters for sanity checking.

## Clusters

### immigration_docs
H-1B visa, USCIS, immigration-related texts.

### child_activities
School, activities, homework, playdates for children.

### health_records
Medical appointments, symptoms, medications.

### financial_planning
Budgets, investments, bills, financial decisions.

### family_memories
Photos, trips, celebrations, nostalgic content.

## File Format

JSONL with cluster labels:

```json
{
    "text": "Need to file the I-140 petition by next month",
    "cluster": "immigration_docs"
}
```

## Files

- `clusters.jsonl` - All clustered examples

## Evaluation

After training, verify that:
1. Texts in same cluster have high cosine similarity
2. Texts in different clusters have lower similarity
3. Retrieval within clusters has high recall
