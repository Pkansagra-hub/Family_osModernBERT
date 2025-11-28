"""
Task Sampler for Multi-Task Learning

This module handles the sampling strategy for selecting which task
to train on at each step in multi-task learning.

Sampling Strategies:
    - Proportional: Sample based on dataset size * task weight
    - Temperature: Softmax sampling with temperature parameter
    - Uniform: Equal probability for all tasks
    - Sequential: Round-robin through tasks
    - Curriculum: Gradually shift focus (easy -> hard tasks)

The sampler ensures balanced training across tasks while respecting
user-defined task weights and preventing catastrophic forgetting.

Usage:
    sampler = TaskSampler(
        task_sizes={"ner": 10000, "sentiment": 5000},
        task_weights={"ner": 1.0, "sentiment": 2.0},
        strategy="proportional",
        seed=42
    )
    
    for step in range(num_steps):
        task = sampler.sample()
        batch = dataloaders[task].next()
        # train on batch
"""

# TODO: Implement TaskSampler class
#   - __init__(task_sizes, task_weights, strategy, temperature, seed)
#   - sample() -> str: Return task name to train on
#   - get_probabilities() -> Dict[str, float]: Current sampling probs
#   - reset_epoch(): Reset state for new epoch
#   - update_weights(new_weights): Dynamic weight adjustment

# TODO: Implement proportional sampling
#   - P(task) ∝ dataset_size[task] * weight[task]
#   - Normalize to sum to 1

# TODO: Implement temperature sampling
#   - P(task) ∝ exp(log(size[task]) / temperature)
#   - Higher temp = more uniform
#   - Lower temp = favor larger datasets

# TODO: Implement uniform sampling
#   - P(task) = 1 / num_tasks
#   - Ignores dataset sizes

# TODO: Implement sequential sampling
#   - Cycle through tasks in order
#   - Optionally weighted (more steps on some tasks)

# TODO: Implement curriculum sampling
#   - Start with "easier" tasks (e.g., sentiment)
#   - Gradually increase weight of harder tasks (e.g., NER)
#   - Schedule: linear, exponential, step
