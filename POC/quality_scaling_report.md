# Retrieval Quality Scaling Report

Date: 2026-02-27

This report measures retrieval quality metrics as corpus size scales to 100k tools.

## Scenario A: random unrelated distractors

- size=500: TPR@10=0.975, DMR@10=1.000, OMR@10=0.977, P@5=0.902, MRR=0.919
- size=5000: TPR@10=0.977, DMR@10=1.000, OMR@10=0.979, P@5=0.904, MRR=0.918
- size=20000: TPR@10=0.977, DMR@10=1.000, OMR@10=0.979, P@5=0.904, MRR=0.918
- size=100000: TPR@10=0.975, DMR@10=1.000, OMR@10=0.977, P@5=0.905, MRR=0.918

## Scenario B: hard near-duplicate distractors

- size=500: TPR@10=0.975, DMR@10=1.000, OMR@10=0.977, P@5=0.902, MRR=0.919
- size=5000: TPR@10=0.961, DMR@10=1.000, OMR@10=0.973, P@5=0.905, MRR=0.918
- size=20000: TPR@10=0.965, DMR@10=1.000, OMR@10=0.973, P@5=0.908, MRR=0.919
- size=100000: TPR@10=0.971, DMR@10=1.000, OMR@10=0.975, P@5=0.910, MRR=0.922

## Interpretation

- Random distractors estimate dilution robustness.
- Hard distractors estimate ambiguity pressure when many close alternatives exist.
- Production behavior at 100k depends more on ambiguity distribution than on count alone.
