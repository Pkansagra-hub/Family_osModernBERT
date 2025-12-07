"""
Multi-Task Trainer

This module provides a custom trainer for multi-task learning that extends
HuggingFace's Trainer with task-specific functionality.

Features:
    - Task sampling: Proportional, temperature-based, or uniform
    - Per-task loss weighting
    - Multi-task evaluation with per-task metrics
    - Dynamic task weighting (uncertainty weighting option)
    - Task-specific learning rates (optional)

Main Classes:
    - MultiTaskTrainer: Extended Trainer for multi-task learning
    - MultiTaskDataLoader: Yields batches with task labels

Training Flow:
    1. Sample task according to strategy
    2. Get batch from task-specific dataloader
    3. Forward pass through shared encoder + task head
    4. Compute task-specific loss with weighting
    5. Backward pass
    6. Optimizer step

Configuration:
    See configs/training/multitask/*.yaml for training configs.

Usage:
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_datasets={"ner": ner_dataset, "sentiment": sent_dataset},
        eval_datasets={"ner": ner_eval, "sentiment": sent_eval},
        task_weights={"ner": 1.0, "sentiment": 1.0},
    )
    trainer.train()
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, IterableDataset
from transformers import Trainer, TrainingArguments

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

# SOTA training features
from modeling_studio.models.losses import FGM, PGD, EmbeddingMixup, RDropLoss
from modeling_studio.trainers.collators import MultiTaskCollator
from modeling_studio.trainers.task_sampler import TaskSampler, create_sampler
from modeling_studio.trainers.task_weighting import UncertaintyWeighting

# =============================================================================
# Multi-Task DataLoader
# =============================================================================


class MultiTaskDataLoader:
    """
    DataLoader wrapper that yields batches from multiple task-specific dataloaders.

    This class manages multiple dataloaders (one per task) and yields batches
    according to a task sampling strategy. Each batch includes a 'task' field
    indicating which task the batch belongs to.

    Args:
        dataloaders: Dictionary mapping task names to DataLoaders
        sampler: TaskSampler instance for selecting which task to sample from
        total_steps: Optional total number of steps (for finite iteration)

    Example:
        >>> loaders = {"ner": ner_loader, "sentiment": sent_loader}
        >>> sampler = ProportionalSampler({"ner": len(ner_ds), "sentiment": len(sent_ds)})
        >>> multi_loader = MultiTaskDataLoader(loaders, sampler)
        >>> for batch in multi_loader:
        ...     task = batch["task"]
        ...     # Process batch for task
    """

    def __init__(
        self,
        dataloaders: dict[str, DataLoader],
        sampler: TaskSampler,
        total_steps: int | None = None,
    ):
        self.dataloaders = dataloaders
        self.sampler = sampler
        self.total_steps = total_steps

        # Required by HuggingFace Trainer for checkpoint resume (skip_first_batches)
        # Create a combined dataset reference from all dataloaders
        self.dataset = ConcatDataset(
            [loader.dataset for loader in dataloaders.values() if hasattr(loader, "dataset")]
        )

        # Required by accelerate's skip_first_batches for checkpoint resume
        # Use the first dataloader's batch_sampler/sampler as reference
        first_loader = next(iter(dataloaders.values()))
        self.batch_sampler = getattr(first_loader, "batch_sampler", None)
        self.sampler_ref = getattr(
            first_loader, "sampler", None
        )  # Store as sampler_ref to avoid conflict

        # Copy DataLoader attributes from first loader for accelerate compatibility
        # These are needed by accelerate's skip_first_batches
        self.num_workers = getattr(first_loader, "num_workers", 0)
        self.collate_fn = getattr(first_loader, "collate_fn", None)
        self.pin_memory = getattr(first_loader, "pin_memory", False)
        self.drop_last = getattr(first_loader, "drop_last", False)
        self.timeout = getattr(first_loader, "timeout", 0)
        self.worker_init_fn = getattr(first_loader, "worker_init_fn", None)
        self.multiprocessing_context = getattr(first_loader, "multiprocessing_context", None)
        self.generator = getattr(first_loader, "generator", None)
        self.persistent_workers = getattr(first_loader, "persistent_workers", False)
        self.pin_memory_device = getattr(first_loader, "pin_memory_device", "")
        self.batch_size = getattr(first_loader, "batch_size", 1)

        # Critical: prefetch_factor must be None when num_workers=0
        # accelerate's skip_first_batches extracts this and creates new DataLoader
        self.prefetch_factor = getattr(first_loader, "prefetch_factor", None)
        if self.num_workers == 0:
            self.prefetch_factor = None

        # Create iterators for each dataloader
        self._iterators: dict[str, Iterator] = {}
        self._reset_iterators()

    def _reset_iterators(self) -> None:
        """Reset all dataloader iterators."""
        self._iterators = {task: iter(loader) for task, loader in self.dataloaders.items()}

    def _get_batch(self, task: str) -> dict[str, Any]:
        """Get a batch from the specified task's dataloader."""
        try:
            batch = next(self._iterators[task])
        except StopIteration:
            # Reset this dataloader and get next batch
            self._iterators[task] = iter(self.dataloaders[task])
            batch = next(self._iterators[task])

        # Add task identifier to batch
        if isinstance(batch, dict):
            batch["task"] = task
        else:
            # If batch is a tuple/list, convert to dict
            batch = {"inputs": batch, "task": task}

        return batch

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over batches from all tasks."""
        self._reset_iterators()
        self.sampler.reset()

        # Calculate total steps - use explicit total or sum of dataloader lengths
        max_steps = self.total_steps
        if max_steps is None:
            max_steps = sum(len(loader) for loader in self.dataloaders.values())

        # Safety check: ensure we have a valid number of steps
        if max_steps <= 0:
            return

        for step in range(max_steps):
            # Sample task
            task = self.sampler.sample()

            # Get batch from task
            batch = self._get_batch(task)

            yield batch

    def __len__(self) -> int:
        """Return total number of batches across all dataloaders."""
        if self.total_steps is not None:
            return self.total_steps
        return sum(len(loader) for loader in self.dataloaders.values())


# =============================================================================
# Multi-Task Dataset Wrapper
# =============================================================================


class MultiTaskIterableDataset(IterableDataset):
    """
    Iterable dataset that wraps multiple task datasets for Trainer compatibility.

    This is needed because HuggingFace Trainer expects a Dataset object,
    but we want to use our custom multi-task sampling logic.

    Args:
        datasets: Dictionary mapping task names to datasets
        sampler: TaskSampler instance for task selection
        total_samples: Total number of samples per epoch
    """

    def __init__(
        self,
        datasets: dict[str, Dataset],
        sampler: TaskSampler,
        total_samples: int | None = None,
    ):
        self.datasets = datasets
        self.sampler = sampler
        self.total_samples = total_samples or sum(len(ds) for ds in datasets.values())

        # Track current indices for each dataset
        self._indices: dict[str, int] = dict.fromkeys(datasets, 0)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over samples from all tasks."""
        self.sampler.reset()
        self._indices = dict.fromkeys(self.datasets, 0)

        for _ in range(self.total_samples):
            # Sample task
            task = self.sampler.sample()

            # Get sample from task dataset
            ds = self.datasets[task]
            idx = self._indices[task] % len(ds)
            sample = ds[idx]
            self._indices[task] += 1

            # Add task identifier
            if isinstance(sample, dict):
                sample["task"] = task
            else:
                sample = {"inputs": sample, "task": task}

            yield sample

    def __len__(self) -> int:
        return self.total_samples


# =============================================================================
# Multi-Task Training Arguments
# =============================================================================


@dataclass
class MultiTaskTrainingArguments(TrainingArguments):
    """Extended training arguments for multi-task learning."""

    # Task sampling
    sampling_strategy: str = field(
        default="proportional",
        metadata={"help": "Task sampling strategy: proportional, temperature, uniform, sequential"},
    )
    sampling_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for temperature-based sampling (higher = more uniform)"},
    )

    # Task weighting
    use_uncertainty_weighting: bool = field(
        default=False,
        metadata={"help": "Use learned uncertainty weighting for task losses"},
    )

    # === SOTA FEATURES ===
    # R-Drop regularization
    use_rdrop: bool = field(
        default=False,
        metadata={"help": "Use R-Drop regularization (two forward passes with dropout)"},
    )
    rdrop_alpha: float = field(
        default=0.5,
        metadata={"help": "R-Drop KL divergence weight"},
    )

    # Adversarial training
    use_adversarial: bool = field(
        default=False,
        metadata={"help": "Use adversarial training (FGM or PGD)"},
    )
    adversarial_type: str = field(
        default="fgm",
        metadata={"help": "Adversarial training type: 'fgm' or 'pgd'"},
    )
    adversarial_epsilon: float = field(
        default=1.0,
        metadata={"help": "Adversarial perturbation magnitude"},
    )
    pgd_steps: int = field(
        default=3,
        metadata={"help": "Number of PGD steps (only for pgd type)"},
    )
    pgd_alpha: float = field(
        default=0.3,
        metadata={"help": "PGD step size (only for pgd type)"},
    )

    # Mixup augmentation
    use_mixup: bool = field(
        default=False,
        metadata={"help": "Use mixup augmentation in embedding space"},
    )
    mixup_alpha: float = field(
        default=0.4,
        metadata={"help": "Mixup beta distribution parameter"},
    )
    mixup_prob: float = field(
        default=0.5,
        metadata={"help": "Probability of applying mixup to a batch"},
    )

    # Progressive Regularization: Enable features gradually by epoch
    # This saves 40%+ training time by disabling expensive features early
    progressive_regularization: bool = field(
        default=False,
        metadata={"help": "Enable progressive regularization (epoch-based feature toggling)"},
    )
    rdrop_start_epoch: int = field(
        default=4,
        metadata={"help": "Epoch to start R-Drop (1-indexed, 0=always on)"},
    )
    mixup_start_epoch: int = field(
        default=4,
        metadata={"help": "Epoch to start Mixup (1-indexed, 0=always on)"},
    )
    adversarial_start_epoch: int = field(
        default=7,
        metadata={"help": "Epoch to start adversarial training (1-indexed, 0=always on)"},
    )


# =============================================================================
# Multi-Task Trainer
# =============================================================================


class MultiTaskTrainer(Trainer):
    """
    Trainer for multi-task learning with multiple datasets and task heads.

    Extends HuggingFace's Trainer to support:
    - Multiple datasets (one per task)
    - Task sampling strategies
    - Per-task loss weighting
    - Task-specific evaluation

    Args:
        model: Multi-task model with capability-based heads
        args: Training arguments
        train_datasets: Dictionary mapping task names to training datasets
        eval_datasets: Dictionary mapping task names to evaluation datasets
        task_weights: Dictionary mapping task names to loss weights
        sampling_strategy: How to sample tasks ("proportional", "temperature", "uniform", "sequential")
        sampling_temperature: Temperature for temperature-based sampling
        tokenizer: Tokenizer for data collation
        data_collator: Custom data collator (uses MultiTaskCollator if None)
        compute_metrics: Function to compute metrics (optional)
        callbacks: List of callbacks
        optimizers: Tuple of (optimizer, lr_scheduler)

    Example:
        >>> model = ModernBertMultiTaskModel.from_pretrained(
        ...     "answerdotai/ModernBERT-base",
        ...     capabilities=["ner_general", "sentiment"],
        ... )
        >>> trainer = MultiTaskTrainer(
        ...     model=model,
        ...     args=training_args,
        ...     train_datasets={"ner_general": ner_ds, "sentiment": sent_ds},
        ...     task_weights={"ner_general": 1.0, "sentiment": 1.0},
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        model: PreTrainedModel | None = None,
        args: TrainingArguments | MultiTaskTrainingArguments | None = None,
        train_datasets: dict[str, Dataset] | None = None,
        eval_datasets: dict[str, Dataset] | None = None,
        task_weights: dict[str, float] | None = None,
        sampling_strategy: str = "proportional",
        sampling_temperature: float = 1.0,
        tokenizer: PreTrainedTokenizerBase | None = None,
        data_collator: Any | None = None,
        compute_metrics: Any | None = None,
        callbacks: list | None = None,
        optimizers: tuple = (None, None),
        preprocess_logits_for_metrics: Any | None = None,
    ):
        # Store multi-task specific attributes
        self.train_datasets = train_datasets or {}
        self.eval_datasets = eval_datasets or {}
        self.task_weights = task_weights or dict.fromkeys(self.train_datasets, 1.0)
        self.sampling_strategy = sampling_strategy
        self.sampling_temperature = sampling_temperature

        # Compute task sizes for sampling
        self.task_sizes = {task: len(ds) for task, ds in self.train_datasets.items()}

        # Create task sampler
        self.task_sampler = self._create_sampler()

        # Create multi-task collator if not provided
        if data_collator is None and tokenizer is not None:
            data_collator = MultiTaskCollator(tokenizer=tokenizer)

        # Create a dummy train_dataset for parent class
        # We'll override get_train_dataloader() to use our multi-task logic
        dummy_dataset = list(self.train_datasets.values())[0] if self.train_datasets else None

        # Also create a dummy eval_dataset for parent class validation
        # We'll override evaluate() to use our multi-task eval_datasets
        dummy_eval_dataset = list(self.eval_datasets.values())[0] if self.eval_datasets else None

        # Initialize parent Trainer
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=dummy_dataset,
            eval_dataset=dummy_eval_dataset,  # Required for eval_strategy != 'no'
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Prefer the new processing_class attribute to avoid tokenizer deprecation warnings
        self.processing_class = (
            tokenizer if tokenizer is not None else getattr(self, "processing_class", None)
        )

        # Track current task for loss computation
        self.current_task: str | None = None

        # === V2 FEATURE: Uncertainty Weighting ===
        # Initialize learned task weights if enabled in args
        self.uncertainty_weighting: UncertaintyWeighting | None = None
        self.task_to_idx: dict[str, int] = {}

        # Check if using MultiTaskTrainingArguments with uncertainty weighting
        use_uncertainty = getattr(args, "use_uncertainty_weighting", False)
        if use_uncertainty and self.train_datasets:
            task_names = sorted(self.train_datasets.keys())
            self.task_to_idx = {task: i for i, task in enumerate(task_names)}
            self.uncertainty_weighting = UncertaintyWeighting(
                num_tasks=len(task_names),
                init_value=0.0,  # σ=1 initially
            )
            # Move to same device as model
            if model is not None:
                self.uncertainty_weighting = self.uncertainty_weighting.to(model.device)

        # === SOTA FEATURES INITIALIZATION ===
        # R-Drop regularization
        self.rdrop_loss: RDropLoss | None = None
        if getattr(args, "use_rdrop", False):
            self.rdrop_loss = RDropLoss(alpha=getattr(args, "rdrop_alpha", 0.5))
            logger.info(f"R-Drop enabled with alpha={args.rdrop_alpha}")

        # Adversarial training (FGM/PGD)
        self.adversarial: FGM | PGD | None = None
        if getattr(args, "use_adversarial", False):
            adv_type = getattr(args, "adversarial_type", "fgm").lower()
            epsilon = getattr(args, "adversarial_epsilon", 1.0)
            if adv_type == "pgd":
                self.adversarial = PGD(
                    model=model,
                    epsilon=epsilon,
                    alpha=getattr(args, "pgd_alpha", 0.3),
                    num_steps=getattr(args, "pgd_steps", 3),
                )
                logger.info(
                    f"PGD adversarial training enabled: eps={epsilon}, steps={args.pgd_steps}"
                )
            else:
                self.adversarial = FGM(model=model, epsilon=epsilon)
                logger.info(f"FGM adversarial training enabled: eps={epsilon}")

        # Mixup augmentation
        self.mixup: EmbeddingMixup | None = None
        if getattr(args, "use_mixup", False):
            self.mixup = EmbeddingMixup(
                alpha=getattr(args, "mixup_alpha", 0.4),
                apply_prob=getattr(args, "mixup_prob", 0.5),
            )
            logger.info(f"Mixup enabled with alpha={args.mixup_alpha}, prob={args.mixup_prob}")

        # === PROGRESSIVE REGULARIZATION ===
        # Store original feature references for epoch-based toggling
        self.progressive_regularization = getattr(args, "progressive_regularization", False)
        if self.progressive_regularization:
            self._rdrop_loss_ref = self.rdrop_loss
            self._adversarial_ref = self.adversarial
            self._mixup_ref = self.mixup
            self._rdrop_start_epoch = getattr(args, "rdrop_start_epoch", 4)
            self._mixup_start_epoch = getattr(args, "mixup_start_epoch", 4)
            self._adversarial_start_epoch = getattr(args, "adversarial_start_epoch", 7)
            logger.info("Progressive Regularization ENABLED:")
            logger.info(f"  R-Drop starts at epoch {self._rdrop_start_epoch}")
            logger.info(f"  Mixup starts at epoch {self._mixup_start_epoch}")
            logger.info(f"  Adversarial starts at epoch {self._adversarial_start_epoch}")
            # Initially disable all features (will be enabled by epoch)
            self.rdrop_loss = None
            self.adversarial = None
            self.mixup = None

    def _create_sampler(self) -> TaskSampler:
        """Create task sampler based on configuration."""
        return create_sampler(
            strategy=self.sampling_strategy,
            task_sizes=self.task_sizes,
            task_weights=self.task_weights,
            temperature=self.sampling_temperature,
        )

    def _create_task_dataloaders(self) -> dict[str, DataLoader]:
        """Create individual dataloaders for each task."""
        from modeling_studio.trainers.collators import get_task_collator

        dataloaders = {}
        for task, dataset in self.train_datasets.items():
            # Get task-specific collator
            collator = get_task_collator(
                task,
                tokenizer=(
                    self.processing_class
                    if self.processing_class is not None
                    else getattr(self, "tokenizer", None)
                ),
            )

            # Build dataloader kwargs, handling prefetch_factor correctly
            # prefetch_factor must be None when num_workers=0
            num_workers = self.args.dataloader_num_workers
            dataloader_kwargs = {
                "batch_size": self.args.per_device_train_batch_size,
                "collate_fn": collator,
                "shuffle": True,
                "num_workers": num_workers,
                "pin_memory": self.args.dataloader_pin_memory if num_workers > 0 else False,
                "drop_last": self.args.dataloader_drop_last,
            }

            # Handle prefetch_factor: must be None when num_workers=0
            if num_workers > 0:
                prefetch_factor = getattr(self.args, "dataloader_prefetch_factor", 2)
                if prefetch_factor is not None:
                    dataloader_kwargs["prefetch_factor"] = prefetch_factor
                if getattr(self.args, "dataloader_persistent_workers", False):
                    dataloader_kwargs["persistent_workers"] = True
            else:
                # Explicitly set to None to prevent accelerate from using default
                dataloader_kwargs["prefetch_factor"] = None

            dataloaders[task] = DataLoader(dataset, **dataloader_kwargs)
        return dataloaders

    def get_train_dataloader(self) -> MultiTaskDataLoader:
        """
        Create the training DataLoader with multi-task sampling.

        Returns a MultiTaskDataLoader that yields batches with task information.
        Each batch contains samples from only one task.

        Returns:
            MultiTaskDataLoader that yields batches with task information
        """
        if self.train_datasets is None or len(self.train_datasets) == 0:
            raise ValueError("Trainer requires train_datasets to be specified")

        # Create per-task dataloaders
        task_dataloaders = self._create_task_dataloaders()

        # Calculate total steps per epoch (total batches)
        total_batches = sum(len(dl) for dl in task_dataloaders.values())

        # Create multi-task dataloader that samples batches from different tasks
        return MultiTaskDataLoader(
            dataloaders=task_dataloaders,
            sampler=self.task_sampler,
            total_steps=total_batches,
        )

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """
        Compute loss for a multi-task batch.

        Routes the batch to the appropriate task head based on the 'task' field
        in the inputs, applies task-specific loss weighting.

        Args:
            model: The multi-task model
            inputs: Batch dictionary with 'task' field
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items in batch (for gradient accumulation)

        Returns:
            Loss tensor, optionally with model outputs
        """
        # Extract task from inputs
        task = inputs.pop("task", None)
        if task is None:
            raise ValueError("Batch must contain 'task' field for multi-task training")

        # Handle batch task (all items should have same task)
        if isinstance(task, list):
            task = task[0]
        elif isinstance(task, torch.Tensor):
            task = task[0].item() if task.dim() > 0 else task.item()

        # Store current task for callbacks/logging (including replay suffix for metrics)
        self.current_task = task

        # Map replay tasks to their base capability
        # e.g., "sentiment_replay" -> "sentiment", "ner_general_replay" -> "ner_general"
        capability = task
        if task.endswith("_replay"):
            capability = task[:-7]  # Strip "_replay" suffix

        # Extract labels
        labels = inputs.pop("labels", None)

        # Remove token_type_ids if present (ModernBERT doesn't use them)
        inputs.pop("token_type_ids", None)

        # Handle embedding task with special input format (anchor/positive/negative)
        if capability == "embedding":
            return self._compute_embedding_loss(model, inputs, labels, return_outputs)

        # === SOTA FEATURE: Mixup in embedding space ===
        # Note: For mixup to work properly, we need raw embeddings
        # This is a simplified version that works at input level
        use_mixup = self.mixup is not None and labels is not None and model.training

        # Forward pass with task-specific head
        outputs = model(
            capability=capability,
            labels=labels,
            return_dict=True,  # Ensure we get structured output
            **inputs,
        )

        # Get loss - handle both MultiTaskOutput and tuple returns (PEFT wrapping)
        if hasattr(outputs, "loss"):
            loss = outputs.loss
        elif isinstance(outputs, tuple):
            # PEFT may return tuple: (loss, logits, ...) or (logits,) if loss is None
            # Check if first element looks like a loss (scalar tensor)
            if len(outputs) > 0 and isinstance(outputs[0], torch.Tensor) and outputs[0].dim() == 0:
                loss = outputs[0]
            elif (
                len(outputs) > 1 and isinstance(outputs[1], torch.Tensor) and outputs[1].dim() == 0
            ):
                loss = outputs[1]
            else:
                raise ValueError(
                    f"Cannot extract loss from tuple outputs: {[type(o) for o in outputs]}"
                )
        elif isinstance(outputs, dict):
            loss = outputs["loss"]
        else:
            raise ValueError(f"Unexpected outputs type: {type(outputs)}")

        # === SOTA FEATURE: R-Drop Regularization ===
        # R-Drop: Regularize dropout by computing KL divergence between two forward passes
        if self.rdrop_loss is not None and model.training:
            # Get logits from first forward pass
            logits1 = outputs.logits if hasattr(outputs, "logits") else None

            if logits1 is not None:
                # Second forward pass with same inputs (different dropout masks)
                outputs2 = model(
                    capability=capability,
                    labels=labels,
                    return_dict=True,
                    **inputs,
                )
                logits2 = outputs2.logits if hasattr(outputs2, "logits") else None

                if logits2 is not None:
                    # Add R-Drop KL divergence loss
                    rdrop_loss_value = self.rdrop_loss(logits1, logits2, loss)
                    loss = rdrop_loss_value

        # === V2 FEATURE: Uncertainty Weighting ===
        # Apply learned uncertainty weighting if enabled
        if self.uncertainty_weighting is not None and task in self.task_to_idx:
            task_idx = self.task_to_idx[task]
            # Uncertainty weighting expects list of losses, but we process one task at a time
            # So we apply the per-task weight formula directly:
            # L_weighted = (1 / (2 * σ²)) * L + log(σ)
            log_var = self.uncertainty_weighting.log_vars[task_idx]
            precision = torch.exp(-log_var)
            weighted_loss = 0.5 * precision * loss + 0.5 * log_var
            # NOTE: Do NOT apply static task_weights when using uncertainty weighting
            # The learned weights replace static weights - combining them causes double-weighting
        else:
            # Standard static task weighting (when uncertainty weighting is disabled)
            task_weight = self.task_weights.get(task, 1.0)
            weighted_loss = loss * task_weight

        if return_outputs:
            return weighted_loss, outputs
        return weighted_loss

    def _compute_embedding_loss(
        self,
        model: PreTrainedModel,
        inputs: dict[str, torch.Tensor],
        labels: torch.Tensor | None,
        return_outputs: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """
        Compute contrastive loss for embedding task.

        Handles multiple input formats:
        - anchor/positive format: Compute contrastive loss
        - Simple input_ids format: Return embeddings (for in-batch negatives)
        """
        import torch.nn.functional as F

        # Check input format
        if "anchor_input_ids" in inputs:
            # Triplet/pair format: compute embeddings for anchor and positive
            anchor_outputs = model(
                capability="embedding",
                input_ids=inputs["anchor_input_ids"],
                attention_mask=inputs["anchor_attention_mask"],
                return_dict=True,
            )
            anchor_embeds = (
                anchor_outputs.logits if hasattr(anchor_outputs, "logits") else anchor_outputs[0]
            )

            positive_outputs = model(
                capability="embedding",
                input_ids=inputs["positive_input_ids"],
                attention_mask=inputs["positive_attention_mask"],
                return_dict=True,
            )
            positive_embeds = (
                positive_outputs.logits
                if hasattr(positive_outputs, "logits")
                else positive_outputs[0]
            )

            # Compute cosine similarity loss (if labels/scores are provided)
            if labels is not None:
                # STS-style regression: labels are similarity scores [0, 1]
                # Note: Normalization now happens during data loading, not here
                cos_sim = F.cosine_similarity(anchor_embeds, positive_embeds)
                loss = F.mse_loss(cos_sim, labels)

                # Surface cosine scores as logits so evaluation metrics receive
                # 1D similarity predictions instead of full embeddings
                if hasattr(anchor_outputs, "logits"):
                    anchor_outputs.logits = cos_sim
                else:
                    anchor_outputs = (loss, cos_sim)
            else:
                # Contrastive loss using in-batch negatives
                # Cosine similarity between all pairs
                cos_sim = F.cosine_similarity(
                    anchor_embeds.unsqueeze(1), positive_embeds.unsqueeze(0), dim=2
                )
                # Labels: diagonal elements are positives (same index)
                batch_labels = torch.arange(cos_sim.size(0), device=cos_sim.device)
                loss = F.cross_entropy(cos_sim * 20, batch_labels)  # temperature scaling

            # Apply task weight
            task_weight = self.task_weights.get("embedding", 1.0)
            weighted_loss = loss * task_weight

            if return_outputs:
                return weighted_loss, anchor_outputs
            return weighted_loss

        elif "input_ids" in inputs:
            # Simple format
            outputs = model(
                capability="embedding",
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                return_dict=True,
            )
            # No loss for simple embedding (used for inference)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            # Preserve computation graph with a zero-valued loss so backward works
            # even when using simple embedding format during training/debug.
            # Using logits.sum() keeps the graph connected to model parameters.
            loss = logits.sum() * 0.0

            if return_outputs:
                return loss, outputs
            return loss
        else:
            raise ValueError(f"Unknown embedding input format. Keys: {list(inputs.keys())}")

    def _get_train_sampler(self) -> None:
        """Override to disable default sampler (we use our own)."""
        return None

    def create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create optimizer with optional head-wise learning rates and uncertainty weighting.

        Can be extended to use different learning rates for:
        - Encoder layers
        - Classification heads
        - Token classification heads
        - Uncertainty weighting parameters (learned task weights)
        """
        # Check if optimizer was already passed to __init__
        if self.optimizer is not None:
            optimizer = self.optimizer
        else:
            # Create the base optimizer
            optimizer = super().create_optimizer()

        # === V2 FEATURE: Add uncertainty weighting parameters to optimizer ===
        if self.uncertainty_weighting is not None:
            # Add the log_vars parameter to the optimizer
            # Use a higher learning rate for task weights (they need to adapt quickly)
            param_groups = list(optimizer.param_groups)

            # Get the base learning rate (as float)
            base_lr = self.args.learning_rate
            if isinstance(base_lr, str):
                base_lr = float(base_lr)

            param_groups.append(
                {
                    "params": list(self.uncertainty_weighting.parameters()),
                    "lr": base_lr * 10,  # 10x base LR for task weights
                    "weight_decay": 0.0,  # No regularization on task weights
                }
            )

            # Recreate optimizer with new param groups
            from torch.optim import AdamW

            optimizer = AdamW(
                param_groups,
                lr=base_lr,
                betas=(0.9, 0.999),
                eps=1e-8,
            )

            # CRITICAL: Update self.optimizer so the Trainer uses the new optimizer
            # with uncertainty weighting parameters included
            self.optimizer = optimizer

            logger.info(
                f"Uncertainty weighting enabled: added {len(list(self.uncertainty_weighting.parameters()))} "
                f"learnable task weight parameters (log_vars) to optimizer with lr={base_lr * 10:.2e}"
            )

        return optimizer

    def _update_progressive_features(self) -> None:
        """
        Toggle regularization features based on current epoch.

        Progressive regularization schedule:
        - Epochs 1-3: No regularization (fast feature learning)
        - Epochs 4-6: R-Drop + Mixup (smooth decision boundaries)
        - Epochs 7+:  R-Drop + Mixup + Adversarial (robustness)

        This saves ~40% training time by disabling expensive features early,
        while still achieving final model robustness.
        """
        # Get current epoch (1-indexed)
        current_epoch = int(self.state.epoch) + 1 if self.state.epoch is not None else 1

        # Check if features should be enabled (only log on change)
        # R-Drop
        if hasattr(self, "_rdrop_loss_ref") and self._rdrop_loss_ref is not None:
            should_enable_rdrop = current_epoch >= self._rdrop_start_epoch
            if should_enable_rdrop and self.rdrop_loss is None:
                self.rdrop_loss = self._rdrop_loss_ref
                logger.info(f"[Epoch {current_epoch}] R-Drop ENABLED")
            elif not should_enable_rdrop and self.rdrop_loss is not None:
                self.rdrop_loss = None

        # Mixup
        if hasattr(self, "_mixup_ref") and self._mixup_ref is not None:
            should_enable_mixup = current_epoch >= self._mixup_start_epoch
            if should_enable_mixup and self.mixup is None:
                self.mixup = self._mixup_ref
                logger.info(f"[Epoch {current_epoch}] Mixup ENABLED")
            elif not should_enable_mixup and self.mixup is not None:
                self.mixup = None

        # Adversarial
        if hasattr(self, "_adversarial_ref") and self._adversarial_ref is not None:
            should_enable_adv = current_epoch >= self._adversarial_start_epoch
            if should_enable_adv and self.adversarial is None:
                self.adversarial = self._adversarial_ref
                logger.info(f"[Epoch {current_epoch}] Adversarial Training ENABLED")
            elif not should_enable_adv and self.adversarial is not None:
                self.adversarial = None

    def training_step(
        self,
        model: PreTrainedModel,
        inputs: dict[str, Any],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        """
        Override training step for SOTA features:
        - FGM/PGD adversarial training
        - EMA update after each step
        - Progressive regularization (epoch-based feature toggling)
        """
        # === PROGRESSIVE REGULARIZATION: Toggle features by epoch ===
        if self.progressive_regularization:
            self._update_progressive_features()

        # === ADVERSARIAL TRAINING ===
        if self.adversarial is not None:
            # Custom adversarial training step
            loss = self._adversarial_training_step(model, inputs, num_items_in_batch)
        else:
            # Standard training step
            loss = super().training_step(model, inputs, num_items_in_batch)

        # === V2 FEATURE: Update EMA after each step ===
        # EMA is attached to trainer in train_stage_a.py
        if hasattr(self, "ema_model") and self.ema_model is not None:
            self.ema_model.update(model)

        return loss

    def _adversarial_training_step(
        self,
        model: PreTrainedModel,
        inputs: dict[str, Any],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        """
        Adversarial training step with FGM or PGD.

        FGM (Fast Gradient Method):
            1. Forward + backward on clean examples
            2. Add perturbation to embeddings based on gradient
            3. Forward + backward on adversarial examples
            4. Restore embeddings

        PGD (Projected Gradient Descent):
            1. Forward + backward on clean examples
            2. Iteratively perturb and project (k steps)
            3. Forward + backward on adversarial examples
            4. Restore embeddings
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Save task field before compute_loss pops it
        task = inputs.get("task")
        labels = inputs.get("labels")

        # Enable gradient computation
        with self.compute_loss_context_manager():
            # First forward pass on clean examples
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        # Scale loss for gradient accumulation
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        # Backward pass on clean examples to get gradients
        self.accelerator.backward(loss)

        # === Apply adversarial perturbation ===
        if isinstance(self.adversarial, PGD):
            # PGD: multi-step attack
            self.adversarial.backup_grad()

            for t in range(self.adversarial.num_steps):
                self.adversarial.attack(is_first=(t == 0))
                model.zero_grad()

                # Restore task and labels for adversarial forward pass
                inputs["task"] = task
                inputs["labels"] = labels

                with self.compute_loss_context_manager():
                    adv_loss = self.compute_loss(
                        model, inputs, num_items_in_batch=num_items_in_batch
                    )

                if self.args.n_gpu > 1:
                    adv_loss = adv_loss.mean()
                if self.args.gradient_accumulation_steps > 1:
                    adv_loss = adv_loss / self.args.gradient_accumulation_steps

                self.accelerator.backward(adv_loss)

            self.adversarial.restore()
            self.adversarial.restore_grad()
        else:
            # FGM: single-step attack
            self.adversarial.attack()

            # Restore task and labels for adversarial forward pass
            inputs["task"] = task
            inputs["labels"] = labels

            with self.compute_loss_context_manager():
                adv_loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

            if self.args.n_gpu > 1:
                adv_loss = adv_loss.mean()
            if self.args.gradient_accumulation_steps > 1:
                adv_loss = adv_loss / self.args.gradient_accumulation_steps

            self.accelerator.backward(adv_loss)
            self.adversarial.restore()

        # Return the clean loss for logging
        return loss.detach()

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Log metrics with task prefix when applicable."""
        # Add current task to logs if available
        if self.current_task is not None and "loss" in logs:
            logs[f"{self.current_task}_loss"] = logs["loss"]

        # === V2 FEATURE: Log uncertainty weights ===
        if self.uncertainty_weighting is not None:
            # Log current learned task weights
            for task, idx in self.task_to_idx.items():
                log_var = self.uncertainty_weighting.log_vars[idx].item()
                # σ² = exp(log_var), weight ~ 1/σ²
                import math

                weight = math.exp(-log_var)
                logs[f"uw_{task}"] = weight

        super().log(logs, start_time)

    def _get_eval_dataloader_for_task(self, task: str, dataset: Dataset) -> DataLoader:
        """Create evaluation dataloader for a specific task."""
        from modeling_studio.trainers.collators import get_task_collator

        collator = get_task_collator(
            task,
            tokenizer=(
                self.processing_class
                if self.processing_class is not None
                else getattr(self, "tokenizer", None)
            ),
        )

        # Build dataloader kwargs, handling prefetch_factor correctly
        # prefetch_factor must be None when num_workers=0
        num_workers = self.args.dataloader_num_workers
        dataloader_kwargs = {
            "batch_size": self.args.per_device_eval_batch_size,
            "collate_fn": collator,
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": self.args.dataloader_pin_memory if num_workers > 0 else False,
        }

        # Handle prefetch_factor: must be None when num_workers=0
        if num_workers > 0:
            prefetch_factor = getattr(self.args, "dataloader_prefetch_factor", 2)
            if prefetch_factor is not None:
                dataloader_kwargs["prefetch_factor"] = prefetch_factor
            if getattr(self.args, "dataloader_persistent_workers", False):
                dataloader_kwargs["persistent_workers"] = True
        else:
            # Explicitly set to None to prevent accelerate from using default
            dataloader_kwargs["prefetch_factor"] = None

        return DataLoader(dataset, **dataloader_kwargs)

    def _compute_metrics_for_task(
        self,
        task: str,
        predictions: torch.Tensor | list,
        labels: torch.Tensor | list,
    ) -> dict[str, float]:
        """Compute metrics for a specific task."""
        import numpy as np

        from modeling_studio.evaluation.metrics import (
            compute_metrics_for_task,
            get_task_problem_type,
        )

        # Get problem type first to determine how to convert labels
        problem_type = get_task_problem_type(task)

        # Convert tensors to numpy
        # Predictions: always use .float() first since they might be bf16
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().float().numpy()

        # Labels: keep as int for classification/NER, only use float for regression/multi-label logits
        if isinstance(labels, torch.Tensor):
            if problem_type in ("token_classification", "single_label_classification"):
                # Integer labels - don't convert to float
                labels = labels.detach().cpu().numpy()
            else:
                # Multi-label or regression - can be float
                labels = labels.detach().cpu().float().numpy()

        # Auto-detect single-label from labels for tasks that may be either
        # (e.g., emotions: multi-label in Stage B, single-label in Stage A)
        actual_problem_type = problem_type
        if problem_type == "multi_label_classification":
            labels_arr = np.asarray(labels)
            if labels_arr.ndim == 1 or (labels_arr.ndim == 2 and labels_arr.shape[1] == 1):
                # Labels are 1D integers = single-label format
                actual_problem_type = "single_label_classification"

        # Convert logits to predictions (argmax for classification)
        if predictions.ndim == 3:
            # Token classification: (batch, seq_len, num_classes) -> (batch, seq_len)
            predictions = np.argmax(predictions, axis=-1)
        elif predictions.ndim == 2 and actual_problem_type == "single_label_classification":
            # Sequence classification: (batch, num_classes) -> (batch,)
            predictions = np.argmax(predictions, axis=-1)
        # For multi-label, keep logits (threshold applied in metrics)

        # Get label list for NER tasks
        label_list = None

        if actual_problem_type == "token_classification":
            label_list = self._get_label_list_for_task(task)

        return compute_metrics_for_task(
            task=task,
            predictions=predictions,
            labels=labels,
            label_list=label_list,
        )

    def _get_label_list_for_task(self, task: str) -> list[str] | None:
        """Get label list for a task (mainly for NER)."""
        try:
            from modeling_studio.data.labels import (
                NER_FAMILY_LABELS,
                NER_GENERAL_LABELS,
                TEMPORAL_LABELS,
            )

            label_schemas = {
                "ner_general": NER_GENERAL_LABELS,
                "ner_family": NER_FAMILY_LABELS,
                "temporal": TEMPORAL_LABELS,
            }

            if task in label_schemas:
                schema = label_schemas[task]
                # Create ordered list from id2label (sorted by ID)
                return [schema.id2label[i] for i in range(schema.num_labels)]

        except ImportError:
            pass

        return None

    def evaluate(
        self,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """
        Run evaluation on all task datasets with per-task metrics.

        Computes task-specific metrics (F1, accuracy, etc.) and aggregates
        them into summary metrics for model selection.

        Args:
            eval_dataset: Optional override for eval datasets
            ignore_keys: Keys to ignore in outputs
            metric_key_prefix: Prefix for metric keys

        Returns:
            Dictionary of metrics with task prefixes, including:
            - eval_{task}_loss: Loss for each task
            - eval_{task}_{metric}: Per-task metrics (f1, accuracy, etc.)
            - eval_avg_score: Average primary metric across tasks
        """
        from modeling_studio.evaluation.metrics import aggregate_metrics

        if eval_dataset is None:
            eval_dataset = self.eval_datasets

        if not eval_dataset:
            return {}

        all_metrics = {}
        per_task_metrics = {}

        # Evaluate each task separately
        for task, dataset in eval_dataset.items():
            task_metrics = self._evaluate_single_task(
                task=task,
                dataset=dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=f"{metric_key_prefix}_{task}",
            )

            # Store raw task metrics for aggregation
            # Extract metrics without prefix for aggregation
            raw_metrics = {}
            prefix = f"{metric_key_prefix}_{task}_"
            for key, value in task_metrics.items():
                if key.startswith(prefix):
                    metric_name = key[len(prefix) :]
                    raw_metrics[metric_name] = value

            per_task_metrics[task] = raw_metrics
            all_metrics.update(task_metrics)

        # Compute aggregate metrics
        if per_task_metrics:
            aggregated = aggregate_metrics(per_task_metrics, self.task_weights)
            for key, value in aggregated.items():
                all_metrics[f"{metric_key_prefix}_{key}"] = value

        # Log metrics (required for HF Trainer to record them)
        self.log(all_metrics)

        self.current_task = None
        return all_metrics

    def _evaluate_single_task(
        self,
        task: str,
        dataset: Dataset,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """
        Evaluate a single task and compute task-specific metrics.

        Args:
            task: Task name
            dataset: Evaluation dataset for this task
            ignore_keys: Keys to ignore in outputs
            metric_key_prefix: Prefix for metric keys

        Returns:
            Dictionary of metrics for this task
        """
        # Set current task for proper head routing
        self.current_task = task

        # Create eval dataloader
        eval_dataloader = self._get_eval_dataloader_for_task(task, dataset)

        # Collect predictions and labels
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0

        self.model.eval()

        for batch in eval_dataloader:
            # Move batch to device
            batch = self._prepare_inputs(batch)

            with torch.no_grad():
                loss, logits, labels = self.prediction_step(
                    self.model,
                    batch,
                    prediction_loss_only=False,
                    ignore_keys=ignore_keys,
                )

            if loss is not None:
                total_loss += loss.item()
                num_batches += 1

            if logits is not None:
                all_predictions.append(logits)
            if labels is not None:
                all_labels.append(labels)

        # Compute average loss
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        metrics = {f"{metric_key_prefix}_loss": avg_loss}

        # Compute task-specific metrics if we have predictions
        if all_predictions and all_labels:
            # Concatenate predictions and labels
            predictions = self._concatenate_tensors(all_predictions)
            labels = self._concatenate_tensors(all_labels)

            # Compute metrics
            task_metrics = self._compute_metrics_for_task(task, predictions, labels)

            # Add with prefix
            for metric_name, value in task_metrics.items():
                metrics[f"{metric_key_prefix}_{metric_name}"] = value

        return metrics

    def _concatenate_tensors(
        self,
        tensors: list[torch.Tensor],
    ) -> torch.Tensor:
        """Concatenate list of tensors, handling different shapes.

        For NER tasks, predictions/labels have shape [batch, seq_len] where seq_len
        varies between batches. We need to pad all tensors to the same seq_len
        before concatenating along dim 0.
        """
        if not tensors:
            return torch.tensor([])

        # Check if all tensors have same number of dimensions
        ndims = [t.dim() for t in tensors]

        if len(set(ndims)) == 1 and ndims[0] == 1:
            # All 1D tensors (classification), simple concatenation
            return torch.cat(tensors, dim=0)
        elif len(set(ndims)) == 1 and ndims[0] == 2:
            # All 2D tensors (NER labels: [batch, seq_len])
            # Check if sequence lengths match
            seq_lens = [t.size(1) for t in tensors]
            if len(set(seq_lens)) == 1:
                # Same sequence lengths, simple concatenation
                return torch.cat(tensors, dim=0)
            else:
                # Different sequence lengths - need to pad to max
                max_seq_len = max(seq_lens)
                padded = []
                for t in tensors:
                    current_len = t.size(1)
                    if current_len < max_seq_len:
                        # Pad with -100 for labels, 0 for predictions
                        pad_value = -100 if t.min() < 0 else 0
                        # Pad on the right side of dimension 1 (sequence dimension)
                        padding = (0, max_seq_len - current_len)
                        t = torch.nn.functional.pad(t, padding, value=pad_value)
                    padded.append(t)
                return torch.cat(padded, dim=0)
        elif len(set(ndims)) == 1 and ndims[0] == 3:
            # 3D tensors (NER logits: [batch, seq_len, num_classes])
            # Need to ensure seq_len (dim 1) is consistent
            seq_lens = [t.size(1) for t in tensors]
            if len(set(seq_lens)) == 1:
                # Same sequence lengths, simple concatenation
                return torch.cat(tensors, dim=0)
            else:
                # Different sequence lengths - need to pad dimension 1
                max_seq_len = max(seq_lens)
                padded = []
                for t in tensors:
                    current_len = t.size(1)
                    if current_len < max_seq_len:
                        # For 3D: [batch, seq, classes], pad seq dimension
                        # F.pad format: (classes_left, classes_right, seq_left, seq_right)
                        pad_value = 0  # For logits, use 0
                        padding = (0, 0, 0, max_seq_len - current_len)
                        t = torch.nn.functional.pad(t, padding, value=pad_value)
                    padded.append(t)
                return torch.cat(padded, dim=0)
        elif len(set(ndims)) == 1 and ndims[0] > 3:
            # Higher dimensional tensors (4D+) - shouldn't typically happen
            return torch.cat(tensors, dim=0)
        else:
            # Mixed dimensions - try to handle gracefully
            # Filter to only tensors with matching dimensions
            mode_ndim = max(set(ndims), key=ndims.count)
            filtered = [t for t in tensors if t.dim() == mode_ndim]
            if filtered:
                return self._concatenate_tensors(filtered)
            else:
                # Fallback: flatten all and concatenate
                return torch.cat([t.flatten() for t in tensors], dim=0)

    def prediction_step(
        self,
        model: PreTrainedModel,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """
        Perform a prediction step for evaluation.

        Routes to the correct task head based on current_task.
        """
        # Use current_task for head routing during evaluation
        task = inputs.pop("task", self.current_task)
        if task is None:
            task = self.current_task

        if isinstance(task, list):
            task = task[0]

        labels = inputs.pop("labels", None)

        # Remove token_type_ids if present (ModernBERT doesn't use them)
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            # Handle embedding task specially (support all embedding formats)
            if task == "embedding":
                return self._embedding_prediction_step(model, inputs, labels, prediction_loss_only)

            outputs = model(
                capability=task,
                labels=labels,
                return_dict=True,
                **inputs,
            )

        # Handle PEFT-wrapped models that return tuple instead of MultiTaskOutput
        if isinstance(outputs, tuple):
            # For PEFT models: tuple is (loss, logits, hidden_states, ...)
            loss = outputs[0] if len(outputs) > 0 else None
            logits = outputs[1] if len(outputs) > 1 else None
        else:
            loss = (
                outputs.loss
                if hasattr(outputs, "loss")
                else outputs.get("loss") if isinstance(outputs, dict) else None
            )
            logits = (
                outputs.logits
                if hasattr(outputs, "logits")
                else outputs.get("logits") if isinstance(outputs, dict) else None
            )

        if prediction_loss_only:
            return (loss, None, None)

        return (loss, logits, labels)

    def _embedding_prediction_step(
        self,
        model: PreTrainedModel,
        inputs: dict[str, Any],
        labels: torch.Tensor | None,
        prediction_loss_only: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Handle embedding task prediction with anchor/positive pairs."""
        import torch.nn.functional as F

        # Detect embedding input format
        anchor_ids = None
        anchor_mask = None
        positive_ids = None
        positive_mask = None

        if "anchor_input_ids" in inputs:
            anchor_ids = inputs["anchor_input_ids"]
            anchor_mask = inputs.get("anchor_attention_mask")
            positive_ids = inputs.get("positive_input_ids")
            positive_mask = inputs.get("positive_attention_mask")
        elif "input_ids_1" in inputs and "input_ids_2" in inputs:
            anchor_ids = inputs["input_ids_1"]
            anchor_mask = inputs.get("attention_mask_1")
            positive_ids = inputs["input_ids_2"]
            positive_mask = inputs.get("attention_mask_2")
        elif "input_ids" in inputs and "positive_input_ids" in inputs:
            anchor_ids = inputs["input_ids"]
            anchor_mask = inputs.get("attention_mask")
            positive_ids = inputs["positive_input_ids"]
            positive_mask = inputs.get("positive_attention_mask")
        elif "input_ids" in inputs:
            anchor_ids = inputs["input_ids"]
            anchor_mask = inputs.get("attention_mask")

        # Compute embeddings for available inputs
        if anchor_ids is None:
            # If format is unknown, return safe defaults to avoid crashing evaluation
            model_device = next(model.parameters()).device
            dummy_loss = torch.tensor(0.0, device=model_device)
            dummy_logits = torch.tensor([], device=model_device)
            return (dummy_loss, dummy_logits, labels)

        anchor_outputs = model(
            capability="embedding",
            input_ids=anchor_ids,
            attention_mask=anchor_mask,
            return_dict=True,
        )
        anchor_embeds = (
            anchor_outputs.logits if hasattr(anchor_outputs, "logits") else anchor_outputs[0]
        )

        if positive_ids is not None:
            positive_outputs = model(
                capability="embedding",
                input_ids=positive_ids,
                attention_mask=positive_mask,
                return_dict=True,
            )
            positive_embeds = (
                positive_outputs.logits
                if hasattr(positive_outputs, "logits")
                else positive_outputs[0]
            )
            cos_sim = F.cosine_similarity(anchor_embeds, positive_embeds)
        else:
            # Without positives, fall back to L2-normalized embeddings compared to self
            cos_sim = torch.ones(anchor_embeds.size(0), device=anchor_embeds.device)

        # Compute loss if labels provided (labels are already normalized to [0,1])
        if labels is not None:
            loss = F.mse_loss(cos_sim, labels)
        else:
            # Contrastive loss when labels are absent but positives exist
            if positive_ids is not None:
                sim_matrix = F.cosine_similarity(
                    anchor_embeds.unsqueeze(1), positive_embeds.unsqueeze(0), dim=2
                )
                batch_labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
                loss = F.cross_entropy(sim_matrix * 20, batch_labels)
            else:
                loss = None

        if prediction_loss_only:
            return (loss, None, None)

        # Return cosine similarity as logits (1D), labels stay as is
        return (loss, cos_sim, labels)


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "MultiTaskTrainer",
    "MultiTaskTrainingArguments",
    "MultiTaskDataLoader",
    "MultiTaskIterableDataset",
]
