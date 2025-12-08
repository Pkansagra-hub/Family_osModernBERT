"""
ModernBERT v3 Phase-Aware Trainer.

This module implements the main training loop that supports phase-based
training with automatic phase transitions, loss tracking, and checkpoint
management.

Training Phases:
    Phase 0.5 (Healing): ~2500 steps
        - Heal cloned layers L23-28
        - Smooth L22->L23 interface
        - Use generic benchmark data

    Phase 1 (Multi-task): ~5000 steps
        - Train on FamilyOS unified data
        - All 12 tasks active
        - LoRA on L23-28

    Phase 2 (Polish): ~1000 steps (optional)
        - Full fine-tune with low LR
        - Focus on safety/emotions

Key Features:
    - Per-layer-group learning rates
    - Warmup + cosine/linear decay scheduler
    - Gradient clipping
    - Mixed precision (fp16/bf16) support
    - Checkpointing at configurable intervals
    - WandB logging integration
    - Evaluation at configurable intervals

Author: FamilyOS Team
Date: December 2025
"""

import dataclasses
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .freezing_v3 import LayerFreezer, TrainingPhase

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """
    Configuration for v3 training.

    This dataclass contains all configuration options for phase-based
    training of ModernBERT v3 models.

    Attributes:
        phase: Training phase ("phase_0.5", "phase_1", "phase_2", "inference")
        max_steps: Maximum number of training steps
        warmup_steps: Number of warmup steps for scheduler
        eval_steps: Evaluate every N steps
        save_steps: Save checkpoint every N steps
        logging_steps: Log metrics every N steps
        learning_rate: Base learning rate
        lr_layers_1_18: Learning rate for frozen layers (usually 0)
        lr_layers_19_22: Learning rate for Semantic band
        lr_layer_23: Learning rate for interface layer L23
        lr_layers_24_28: Learning rate for Family band clones
        weight_decay: Weight decay for AdamW
        max_grad_norm: Maximum gradient norm for clipping
        gradient_accumulation_steps: Number of steps to accumulate gradients
        lr_scheduler_type: Scheduler type ("cosine" or "linear")
        fp16: Use FP16 mixed precision
        bf16: Use BF16 mixed precision
        output_dir: Directory for checkpoints and outputs
        checkpoint_dir: Directory for loading checkpoints (optional)
        use_wandb: Whether to use Weights & Biases logging
        wandb_project: WandB project name
        wandb_run_name: WandB run name (optional)

    Example:
        >>> config = TrainingConfig(
        ...     phase="phase_0.5",
        ...     max_steps=2500,
        ...     warmup_steps=500,
        ...     learning_rate=3e-5,
        ... )
    """

    # Phase settings
    phase: str = "phase_0.5"

    # Training hyperparameters
    max_steps: int = 2500
    warmup_steps: int = 500
    eval_steps: int = 250
    save_steps: int = 500
    logging_steps: int = 50

    # Learning rates (per layer group)
    learning_rate: float = 3e-5
    lr_layers_1_18: float = 0.0  # Frozen in Phase 0.5/1
    lr_layers_19_22: float = 1e-5  # Semantics
    lr_layer_23: float = 5e-5  # Interface
    lr_layers_24_28: float = 3e-5  # Clones

    # Optimization
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # Scheduler
    lr_scheduler_type: str = "cosine"

    # Mixed precision
    fp16: bool = False
    bf16: bool = True

    # Paths
    output_dir: str = "outputs/v3_training"
    checkpoint_dir: str | None = None

    # Logging
    use_wandb: bool = True
    wandb_project: str = "modernbert-v3"
    wandb_run_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return dataclasses.asdict(self)

    def save(self, path: Path | str) -> None:
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainingConfig":
        """Create config from dictionary."""
        return cls(**d)

    @classmethod
    def load(cls, path: Path | str) -> "TrainingConfig":
        """Load config from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclass
class TrainingState:
    """
    Tracks training state.

    This dataclass maintains the current state of training, including
    step counts, metrics history, and losses.

    Attributes:
        global_step: Current training step
        epoch: Current epoch number
        best_metric: Best evaluation metric seen so far
        phase: Current training phase
        losses: List of training losses
        metrics_history: List of evaluation metric dictionaries
    """

    global_step: int = 0
    epoch: int = 0
    best_metric: float = 0.0
    phase: str = "phase_0.5"
    losses: list[float] = field(default_factory=list)
    metrics_history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainingState":
        """Create state from dictionary."""
        return cls(**d)


class ModernBERTv3Trainer:
    """
    Phase-aware trainer for ModernBERT v3.

    This trainer implements phase-based training with layer freezing,
    per-layer learning rates, and comprehensive logging and checkpointing.

    Training Phases:
        Phase 0.5 (Healing): ~2500 steps
            - Heal cloned layers L23-28
            - Smooth L22->L23 interface
            - Use generic benchmark data

        Phase 1 (Multi-task): ~5000 steps
            - Train on FamilyOS unified data
            - All 12 tasks active
            - LoRA on L23-28

        Phase 2 (Polish): ~1000 steps (optional)
            - Full fine-tune with low LR
            - Focus on safety/emotions

    Args:
        model: ModernBERTv3 model to train
        config: TrainingConfig instance
        train_dataloader: DataLoader for training data
        eval_dataloader: Optional DataLoader for evaluation data
        compute_metrics: Optional function for computing custom metrics

    Example:
        >>> model = ModernBERTv3Ultra(config)
        >>> training_config = TrainingConfig(phase="phase_0.5")
        >>> trainer = ModernBERTv3Trainer(
        ...     model=model,
        ...     config=training_config,
        ...     train_dataloader=train_loader,
        ... )
        >>> results = trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader | None = None,
        compute_metrics: Callable | None = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: ModernBERTv3 model to train
            config: TrainingConfig instance
            train_dataloader: DataLoader for training data
            eval_dataloader: Optional DataLoader for evaluation data
            compute_metrics: Optional function for computing custom metrics
        """
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.compute_metrics = compute_metrics

        # State tracking
        self.state = TrainingState(phase=config.phase)

        # Layer freezer
        self.freezer = LayerFreezer(model)

        # Optimizer and scheduler (created in setup)
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.scaler: torch.amp.GradScaler | None = None  # type: ignore[type-arg]

        # Device
        self.device = next(model.parameters()).device

        # WandB run reference
        self._wandb_run = None

    def setup(self) -> None:
        """
        Setup training components.

        This method configures layer freezing, creates the optimizer
        and scheduler, sets up mixed precision, and initializes WandB.
        """
        # Configure freezing for phase
        self.freezer.configure_for_phase(TrainingPhase(self.config.phase))

        # Create optimizer with layer-group LRs
        self.optimizer = self._create_optimizer()

        # Create scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision scaler
        if self.config.fp16:
            self.scaler = torch.cuda.amp.GradScaler()

        # WandB logging
        if self.config.use_wandb:
            self._init_wandb()

        logger.info("Training setup complete")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create optimizer with per-layer-group learning rates.

        Returns:
            AdamW optimizer with parameter groups
        """
        param_groups = self._get_parameter_groups()

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.weight_decay,
        )

        return optimizer

    def _get_parameter_groups(self) -> list[dict[str, Any]]:
        """
        Create parameter groups with layer-specific LRs.

        Groups:
            - layers_1_18: Foundation + Core (frozen or very low LR)
            - layers_19_22: Semantic band
            - layer_23: Interface layer (highest LR)
            - layers_24_28: Family band clones
            - embeddings: Usually frozen
            - task_heads: Same as layers_24_28

        Returns:
            List of parameter group dictionaries
        """
        param_groups: list[dict[str, Any]] = []

        # Get encoder layers safely
        encoder = getattr(self.model, "encoder", self.model)
        layers = getattr(encoder, "layers", None)

        if layers is None:
            # Fallback: just use all model parameters
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            if trainable_params:
                param_groups.append(
                    {
                        "params": trainable_params,
                        "lr": self.config.learning_rate,
                        "name": "all_params",
                    }
                )
            return param_groups

        num_layers = len(layers)  # type: ignore[arg-type]

        # Layers 1-18 (Foundation + Core) - indices 0-17
        if num_layers >= 18:
            layers_1_18_params = []
            for i in range(min(18, num_layers)):
                layers_1_18_params.extend(layers[i].parameters())  # type: ignore[index]

            trainable = [p for p in layers_1_18_params if p.requires_grad]
            if trainable:
                param_groups.append(
                    {
                        "params": trainable,
                        "lr": self.config.lr_layers_1_18,
                        "name": "layers_1_18",
                    }
                )

        # Layers 19-22 (Semantic) - indices 18-21
        if num_layers >= 22:
            layers_19_22_params = []
            for i in range(18, min(22, num_layers)):
                layers_19_22_params.extend(layers[i].parameters())  # type: ignore[index]

            trainable = [p for p in layers_19_22_params if p.requires_grad]
            if trainable:
                param_groups.append(
                    {
                        "params": trainable,
                        "lr": self.config.lr_layers_19_22,
                        "name": "layers_19_22",
                    }
                )

        # Layer 23 (Interface - highest plasticity) - index 22
        if num_layers >= 23:
            layer_23_params = list(layers[22].parameters())  # type: ignore[index]
            trainable = [p for p in layer_23_params if p.requires_grad]
            if trainable:
                param_groups.append(
                    {
                        "params": trainable,
                        "lr": self.config.lr_layer_23,
                        "name": "layer_23",
                    }
                )

        # Layers 24-28 (Family clones) - indices 23-27
        if num_layers >= 28:
            layers_24_28_params = []
            for i in range(23, min(28, num_layers)):
                layers_24_28_params.extend(layers[i].parameters())  # type: ignore[index]

            trainable = [p for p in layers_24_28_params if p.requires_grad]
            if trainable:
                param_groups.append(
                    {
                        "params": trainable,
                        "lr": self.config.lr_layers_24_28,
                        "name": "layers_24_28",
                    }
                )

        # Embeddings
        if hasattr(self.model, "embeddings"):
            emb_params = list(self.model.embeddings.parameters())
            trainable = [p for p in emb_params if p.requires_grad]
            if trainable:
                param_groups.append(
                    {
                        "params": trainable,
                        "lr": self.config.lr_layers_1_18,  # Low LR for embeddings
                        "name": "embeddings",
                    }
                )

        # Task heads
        if hasattr(self.model, "task_heads"):
            head_params = list(self.model.task_heads.parameters())
            trainable = [p for p in head_params if p.requires_grad]
            if trainable:
                param_groups.append(
                    {
                        "params": trainable,
                        "lr": self.config.lr_layers_24_28,
                        "name": "task_heads",
                    }
                )

        # Poolers
        for pooler_name in ["hub_pooler", "combined_pooler", "pair_encoder"]:
            if hasattr(self.model, pooler_name):
                pooler = getattr(self.model, pooler_name)
                pooler_params = list(pooler.parameters())
                trainable = [p for p in pooler_params if p.requires_grad]
                if trainable:
                    param_groups.append(
                        {
                            "params": trainable,
                            "lr": self.config.lr_layers_24_28,
                            "name": pooler_name,
                        }
                    )

        # Final layer norm
        if hasattr(self.model, "final_layer_norm"):
            ln_params = list(self.model.final_layer_norm.parameters())
            trainable = [p for p in ln_params if p.requires_grad]
            if trainable:
                param_groups.append(
                    {
                        "params": trainable,
                        "lr": self.config.lr_layers_24_28,
                        "name": "final_layer_norm",
                    }
                )

        # Log parameter groups
        logger.info("Parameter groups:")
        for group in param_groups:
            n_params = sum(p.numel() for p in group["params"])
            logger.info(f"  {group['name']}: {n_params:,} params, lr={group['lr']}")

        return param_groups

    def _create_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """
        Create learning rate scheduler.

        Returns:
            Learning rate scheduler (cosine or linear with warmup)
        """
        total_steps = self.config.max_steps
        warmup_steps = self.config.warmup_steps

        # Optimizer must be created before scheduler
        assert self.optimizer is not None, "Optimizer must be created before scheduler"

        if self.config.lr_scheduler_type == "cosine":
            from transformers import get_cosine_schedule_with_warmup

            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        elif self.config.lr_scheduler_type == "linear":
            from transformers import get_linear_schedule_with_warmup

            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.lr_scheduler_type}")

        return scheduler

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases logging."""
        try:
            import wandb

            self._wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name or f"v3_{self.config.phase}",
                config=self.config.to_dict(),
            )
        except ImportError:
            logger.warning("wandb not installed, disabling wandb logging")
            self.config.use_wandb = False
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            self.config.use_wandb = False

    def train(self) -> dict[str, Any]:
        """
        Main training loop.

        Returns:
            Dictionary with final step and metrics history
        """
        self.setup()

        logger.info(f"Starting training: {self.config.phase}")
        logger.info(f"  Max steps: {self.config.max_steps}")
        logger.info(f"  Warmup steps: {self.config.warmup_steps}")

        self.model.train()

        progress_bar = tqdm(
            total=self.config.max_steps,
            desc=f"Training ({self.config.phase})",
        )

        accumulated_loss = 0.0
        accumulation_count = 0

        while self.state.global_step < self.config.max_steps:
            for batch in self.train_dataloader:
                # Move batch to device
                batch = self._move_batch_to_device(batch)

                # Forward pass with optional mixed precision
                loss = self._training_step(batch)

                # Backward pass with gradient accumulation
                scaled_loss = loss / self.config.gradient_accumulation_steps
                accumulated_loss += scaled_loss.item()
                accumulation_count += 1

                if self.config.fp16 and self.scaler is not None:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                # Optimizer step after accumulation
                should_step = accumulation_count >= self.config.gradient_accumulation_steps

                if should_step:
                    # Assertions for type safety
                    assert self.optimizer is not None
                    assert self.scheduler is not None

                    # Gradient clipping
                    if self.config.max_grad_norm > 0:
                        if self.config.fp16 and self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm,
                        )

                    # Optimizer step
                    if self.config.fp16 and self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    accumulation_count = 0

                    # Logging
                    if self.state.global_step % self.config.logging_steps == 0:
                        self._log_step(accumulated_loss)
                        self.state.losses.append(accumulated_loss)
                        accumulated_loss = 0.0

                self.state.global_step += 1
                progress_bar.update(1)

                # Evaluation
                if (
                    self.state.global_step % self.config.eval_steps == 0
                    and self.eval_dataloader is not None
                ):
                    metrics = self.evaluate()
                    self._log_eval(metrics)

                # Checkpointing
                if self.state.global_step % self.config.save_steps == 0:
                    self._save_checkpoint()

                if self.state.global_step >= self.config.max_steps:
                    break

        progress_bar.close()

        # Final evaluation and save
        if self.eval_dataloader is not None:
            final_metrics = self.evaluate()
            self._log_eval(final_metrics)

        self._save_checkpoint(final=True)

        # Finish wandb run
        if self.config.use_wandb and self._wandb_run is not None:
            import wandb

            wandb.finish()

        return {
            "final_step": self.state.global_step,
            "metrics": self.state.metrics_history,
        }

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Move batch tensors to device.

        Args:
            batch: Dictionary of batch data

        Returns:
            Batch with tensors moved to device
        """
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }

    def _training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Single training step.

        Args:
            batch: Input batch dictionary

        Returns:
            Loss tensor
        """
        # Mixed precision context
        if self.config.bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return self._compute_loss(batch)
        elif self.config.fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return self._compute_loss(batch)
        else:
            return self._compute_loss(batch)

    def _compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute loss from batch.

        Args:
            batch: Input batch dictionary

        Returns:
            Loss tensor
        """
        outputs = self.model(**batch)

        if hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss
        elif isinstance(outputs, dict) and "loss" in outputs:
            return outputs["loss"]
        else:
            raise ValueError(
                "Model output must contain 'loss'. "
                "Ensure your model returns a loss in its forward pass."
            )

    def evaluate(self) -> dict[str, float]:
        """
        Run evaluation.

        Returns:
            Dictionary of evaluation metrics
        """
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = self._move_batch_to_device(batch)

                try:
                    outputs = self.model(**batch)
                    if hasattr(outputs, "loss") and outputs.loss is not None:
                        total_loss += outputs.loss.item()
                    elif isinstance(outputs, dict) and "loss" in outputs:
                        total_loss += outputs["loss"].item()
                except Exception as e:
                    logger.warning(f"Error during evaluation: {e}")
                    continue

                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        metrics = {"eval_loss": avg_loss}

        if self.compute_metrics is not None:
            try:
                custom_metrics = self.compute_metrics(self.model, self.eval_dataloader)
                metrics.update(custom_metrics)
            except Exception as e:
                logger.warning(f"Error computing custom metrics: {e}")

        self.model.train()
        return metrics

    def _log_step(self, loss: float) -> None:
        """
        Log training step metrics.

        Args:
            loss: Training loss for this step
        """
        lr = self.scheduler.get_last_lr()[0] if self.scheduler else 0.0

        log_dict = {
            "train/loss": loss,
            "train/lr": lr,
            "train/step": self.state.global_step,
        }

        if self.config.use_wandb:
            try:
                import wandb

                wandb.log(log_dict, step=self.state.global_step)
            except Exception:
                pass

        logger.info(f"Step {self.state.global_step}: loss={loss:.4f}, lr={lr:.2e}")

    def _log_eval(self, metrics: dict[str, float]) -> None:
        """
        Log evaluation metrics.

        Args:
            metrics: Dictionary of evaluation metrics
        """
        if self.config.use_wandb:
            try:
                import wandb

                wandb.log(
                    {f"eval/{k}": v for k, v in metrics.items()},
                    step=self.state.global_step,
                )
            except Exception:
                pass

        logger.info(f"Eval @ step {self.state.global_step}: {metrics}")
        self.state.metrics_history.append(metrics)

        # Track best metric
        if "eval_loss" in metrics:
            if self.state.best_metric == 0.0 or metrics["eval_loss"] < self.state.best_metric:
                self.state.best_metric = metrics["eval_loss"]

    def _save_checkpoint(self, final: bool = False) -> None:
        """
        Save training checkpoint.

        Args:
            final: Whether this is the final checkpoint
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_name = "final" if final else f"step_{self.state.global_step}"
        checkpoint_path = output_dir / checkpoint_name
        checkpoint_path.mkdir(exist_ok=True)

        # Save model
        torch.save(
            self.model.state_dict(),
            checkpoint_path / "pytorch_model.bin",
        )

        # Save training state
        torch.save(
            {
                "global_step": self.state.global_step,
                "epoch": self.state.epoch,
                "best_metric": self.state.best_metric,
                "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
                "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            },
            checkpoint_path / "trainer_state.bin",
        )

        # Save config
        self.config.save(checkpoint_path / "training_config.json")

        # Save state as JSON for easy inspection
        with open(checkpoint_path / "training_state.json", "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)

        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Path | str) -> None:
        """
        Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory
        """
        checkpoint_path = Path(checkpoint_path)

        # Load model weights
        model_path = checkpoint_path / "pytorch_model.bin"
        if model_path.exists():
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info(f"Loaded model from {model_path}")

        # Load training state
        trainer_state_path = checkpoint_path / "trainer_state.bin"
        if trainer_state_path.exists():
            state = torch.load(trainer_state_path, map_location=self.device)
            self.state.global_step = state.get("global_step", 0)
            self.state.epoch = state.get("epoch", 0)
            self.state.best_metric = state.get("best_metric", 0.0)

            if self.optimizer and "optimizer_state" in state and state["optimizer_state"]:
                self.optimizer.load_state_dict(state["optimizer_state"])

            if self.scheduler and "scheduler_state" in state and state["scheduler_state"]:
                self.scheduler.load_state_dict(state["scheduler_state"])

            logger.info(f"Loaded trainer state from {trainer_state_path}")

    def get_trainable_params(self) -> int:
        """
        Get total number of trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """
        Get total number of parameters.

        Returns:
            Total number of parameters
        """
        return sum(p.numel() for p in self.model.parameters())

    def print_training_summary(self) -> None:
        """Print a summary of training configuration."""
        print("\n" + "=" * 60)
        print("ModernBERT v3 Training Summary")
        print("=" * 60)
        print(f"Phase: {self.config.phase}")
        print(f"Max steps: {self.config.max_steps}")
        print(f"Warmup steps: {self.config.warmup_steps}")
        print(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print("Learning rates:")
        print(f"  L1-18: {self.config.lr_layers_1_18}")
        print(f"  L19-22: {self.config.lr_layers_19_22}")
        print(f"  L23: {self.config.lr_layer_23}")
        print(f"  L24-28: {self.config.lr_layers_24_28}")
        print(f"Total params: {self.get_total_params():,}")
        print(f"Trainable params: {self.get_trainable_params():,}")
        print(f"Device: {self.device}")
        print("=" * 60)
