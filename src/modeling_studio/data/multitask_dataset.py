"""
Multi-Task Dataset

This module provides dataset classes for loading and combining
multiple datasets for multi-task learning.

Classes:
    - MultiTaskDataset: Combines multiple task datasets
    - TaskDataset: Wrapper adding task information to samples
    - StreamingMultiTaskDataset: For large datasets (streaming mode)

Features:
    - Unified interface across tasks
    - Task-aware batching
    - On-the-fly preprocessing
    - Memory-efficient streaming

Dataset Format:
    Each sample is a dict containing:
    {
        "input_ids": [...],
        "attention_mask": [...],
        "labels": ... (task-specific),
        "task": "task_name"
    }

Usage:
    datasets = {
        "ner": ner_dataset,
        "sentiment": sentiment_dataset,
    }
    multi_dataset = MultiTaskDataset(datasets, task_weights)
    
    for batch in DataLoader(multi_dataset, collate_fn=collator):
        # batch contains samples from potentially different tasks
        pass
"""

# TODO: Implement MultiTaskDataset
#   - __init__(task_datasets: Dict[str, Dataset], weights: Dict[str, float])
#   - __len__(): Total samples across all tasks
#   - __getitem__(idx): Return sample with task label
#   - Interleave samples based on weights

# TODO: Implement TaskDataset
#   - Wraps a single-task dataset
#   - Adds "task" field to each sample
#   - Applies task-specific preprocessing

# TODO: Implement StreamingMultiTaskDataset
#   - For HuggingFace streaming datasets
#   - Interleave iterators from multiple tasks
#   - Handle different dataset lengths gracefully
