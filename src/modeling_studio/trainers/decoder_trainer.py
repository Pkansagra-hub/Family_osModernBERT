"""
Decoder Trainer for Counterfactual Generation.

This module provides a specialized trainer for training the MoE decoder head
on counterfactual generation task (Stage C training).

Features:
    - Encoder/head freezing utilities
    - MoE auxiliary loss handling
    - Decoder-specific compute_loss
    - Resume-friendly checkpoint handling (critical for Colab)
    - Memory-efficient training with gradient checkpointing

Usage:
    trainer = DecoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        decoder_mode=True,
    )
    trainer.train()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments

if TYPE_CHECKING:
    from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


def freeze_encoder_and_heads(
    model: PreTrainedModel,
    freeze_encoder: bool = True,
    freeze_heads: bool = True,
    decoder_head_name: str = "counterfactual",
) -> dict[str, int]:
    """
    Freeze encoder and/or existing heads for Stage C decoder training.

    Args:
        model: The multi-task model with encoder and heads
        freeze_encoder: Whether to freeze encoder parameters
        freeze_heads: Whether to freeze existing head parameters
        decoder_head_name: Name of the decoder head to keep trainable

    Returns:
        dict with param counts: total, trainable, frozen
    """
    frozen_count = 0
    trainable_count = 0

    # Freeze encoder
    if freeze_encoder:
        encoder = getattr(model, "encoder", None)
        if encoder is None:
            # Try getting encoder through different attribute names
            encoder = getattr(model, "modernbert", None)
            if encoder is None:
                encoder = getattr(model, "base_model", None)

        if encoder is not None:
            for param in encoder.parameters():
                param.requires_grad = False
                frozen_count += param.numel()
            logger.info("[FROZEN] Encoder parameters frozen")
        else:
            logger.warning("Could not find encoder to freeze")

    # Freeze existing heads (except decoder head)
    if freeze_heads:
        heads = getattr(model, "heads", {})
        for head_name, head in heads.items():
            if head_name == decoder_head_name:
                # Keep decoder head trainable
                for param in head.parameters():
                    param.requires_grad = True
                    trainable_count += param.numel()
                logger.info(f"[TRAINABLE] Decoder head '{head_name}' remains trainable")
            else:
                for param in head.parameters():
                    param.requires_grad = False
                    frozen_count += param.numel()
                logger.info(f"[FROZEN] Head '{head_name}' frozen")

    # Count remaining trainable params (in case there are other trainable components)
    total_params = sum(p.numel() for p in model.parameters())
    final_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    stats = {
        "total": total_params,
        "trainable": final_trainable,
        "frozen": total_params - final_trainable,
    }

    logger.info("=" * 60)
    logger.info("Parameter Freezing Summary")
    logger.info("=" * 60)
    logger.info(f"  Total params:     {stats['total']:,}")
    logger.info(f"  Trainable params: {stats['trainable']:,}")
    logger.info(f"  Frozen params:    {stats['frozen']:,}")
    logger.info(f"  Trainable ratio:  {100 * stats['trainable'] / stats['total']:.2f}%")
    logger.info("=" * 60)

    return stats


class DecoderTrainer(Trainer):
    """
    Specialized trainer for decoder (counterfactual generation) training.

    Extends HuggingFace's Trainer with:
    - MoE auxiliary loss handling (load balancing + router z-loss)
    - Decoder-specific forward pass
    - Colab-friendly checkpoint saving
    - Memory optimization for large decoder

    Args:
        model: The model with decoder head
        args: Training arguments
        train_dataset: Training dataset (CounterfactualDataset)
        eval_dataset: Evaluation dataset
        data_collator: Data collator (CounterfactualCollator)
        aux_loss_weight: Weight for MoE auxiliary losses (default from config)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
        tokenizer=None,
        aux_loss_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            **kwargs,
        )

        self.aux_loss_weight = aux_loss_weight

        # Track auxiliary losses for logging
        self._aux_loss_accumulator: dict[str, float] = {
            "load_balance": 0.0,
            "z_loss": 0.0,
            "total_aux": 0.0,
        }
        self._aux_loss_count = 0

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """
        Compute loss for decoder training.

        Handles:
        - Forward pass through decoder head
        - Cross-entropy loss for language modeling
        - MoE auxiliary losses (load balancing + z-loss)

        Args:
            model: The model with decoder head
            inputs: Batch with encoder_embeddings, decoder_input_ids, labels
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items (for gradient accumulation)

        Returns:
            Loss tensor, optionally with outputs
        """
        # Get the decoder head
        # The model structure depends on how it's set up
        if hasattr(model, "heads") and "counterfactual" in model.heads:
            decoder_head = model.heads["counterfactual"]
        elif hasattr(model, "decoder"):
            decoder_head = model.decoder
        else:
            # Assume model itself is the decoder
            decoder_head = model

        # Extract inputs - check both possible key names from dataset/collator
        encoder_embeddings = inputs.get("encoder_hidden_states")
        if encoder_embeddings is None:
            encoder_embeddings = inputs.get("encoder_embeddings")
        encoder_attention_mask = inputs.get("encoder_attention_mask")
        decoder_input_ids = inputs.get("decoder_input_ids")
        if decoder_input_ids is None:
            decoder_input_ids = inputs.get("input_ids")
        labels = inputs.get("labels")

        # Forward pass through decoder
        outputs = decoder_head(
            encoder_hidden_states=encoder_embeddings,
            encoder_attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )

        # Extract loss components
        if isinstance(outputs, dict):
            loss = outputs.get("loss", outputs.get("lm_loss"))
            aux_loss = outputs.get("aux_loss", torch.tensor(0.0, device=loss.device))
            logits = outputs.get("logits")
        elif hasattr(outputs, "loss"):
            loss = outputs.loss
            aux_loss = getattr(outputs, "aux_loss", torch.tensor(0.0, device=loss.device))
            logits = getattr(outputs, "logits", None)
        else:
            raise ValueError(f"Unexpected outputs type: {type(outputs)}")

        # Combine losses
        total_loss = loss + self.aux_loss_weight * aux_loss

        # Accumulate aux losses for logging
        if self.model.training:
            self._aux_loss_accumulator["total_aux"] += aux_loss.item()
            self._aux_loss_count += 1

            # Extract component losses if available
            if isinstance(outputs, dict):
                aux_losses = outputs.get("aux_losses", {})
                if "load_balance" in aux_losses:
                    self._aux_loss_accumulator["load_balance"] += aux_losses["load_balance"]
                if "z_loss" in aux_losses:
                    self._aux_loss_accumulator["z_loss"] += aux_losses["z_loss"]

        if return_outputs:
            return total_loss, outputs
        return total_loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Override to add aux loss logging."""
        # Add averaged aux losses to logs
        if self._aux_loss_count > 0:
            avg_factor = self._aux_loss_count
            logs["aux_loss/total"] = self._aux_loss_accumulator["total_aux"] / avg_factor
            logs["aux_loss/load_balance"] = self._aux_loss_accumulator["load_balance"] / avg_factor
            logs["aux_loss/z_loss"] = self._aux_loss_accumulator["z_loss"] / avg_factor

            # Reset accumulators
            self._aux_loss_accumulator = {"load_balance": 0.0, "z_loss": 0.0, "total_aux": 0.0}
            self._aux_loss_count = 0

        super().log(logs, start_time)

    def _save_checkpoint(self, model, trial, metrics=None):
        """Override to ensure complete checkpoint for resume."""
        # Call parent's _save_checkpoint
        super()._save_checkpoint(model, trial)

        # Log checkpoint info for Colab users
        checkpoint_folder = f"checkpoint-{self.state.global_step}"
        output_dir = Path(self.args.output_dir) / checkpoint_folder
        logger.info(f"[CHECKPOINT] Saved to {output_dir}")
        logger.info(
            f"[CHECKPOINT] Resume with: --resume_from_checkpoint {output_dir}"
        )

    def get_train_dataloader(self) -> DataLoader:
        """Create train dataloader with decoder-specific settings."""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset

        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "drop_last": self.args.dataloader_drop_last,
        }

        # Only add prefetch_factor if num_workers > 0
        if self.args.dataloader_num_workers > 0:
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
            dataloader_params["persistent_workers"] = self.args.dataloader_persistent_workers

        return DataLoader(train_dataset, shuffle=True, **dataloader_params)

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        """Create eval dataloader."""
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "drop_last": False,
        }

        # Only add prefetch_factor if num_workers > 0
        if self.args.dataloader_num_workers > 0:
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return DataLoader(eval_dataset, shuffle=False, **dataloader_params)
