"""
ModernBERT v3.3 Ultra - Main Model Class

This module implements the complete ModernBERTv3Ultra model that combines
embeddings, encoder, poolers, and provides a unified interface for all
downstream tasks. This is the primary entry point for v3.

Key Components:
    - ModernBERTEmbeddingsV3: Word and position embeddings with hub tokens
    - ModernBERTEncoderV3: 28-layer transformer encoder with multi-scale attention
    - HubTokenPooler: Extracts hub token representations
    - PairEncoderV3: Sentence-pair classification with [REL] hub

Architecture:
    - 28 transformer layers (vs 22 in v2)
    - 4 hub tokens: [EMO], [MEM], [REL], [TASK]
    - Multi-scale sliding window attention (64→128→256→512)
    - Global attention for hub tokens (positions 0-4)
    - LoRA adapters on Family Band (L23-28)

Token Layout:
    [CLS] [EMO] [MEM] [REL] [TASK] <text> [SEP] [PAD]...
    pos 0   1     2     3     4     5+

Capabilities (12 total):
    Hub-routed (9): emotions, sentiment, safety_*, embedding, nli, relation, intent, ingress
    Token-level (3): ner_general, ner_family, temporal

Author: FamilyOS Team
Date: December 2025
Version: 3.3
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from .config_v3 import ModernBERTv3Config
from .embeddings_v3 import ModernBERTEmbeddingsV3
from .encoder_v3 import ModernBERTEncoderV3
from .hub_tokens import (
    HUB_TOKEN_REGISTRY,
    TOKEN_LEVEL_CAPABILITIES,
    get_hub_for_capability,
    get_hub_positions,
)
from .pair_encoder_v3 import PairEncoderV3
from .poolers_v3 import CombinedPooler, HubTokenPooler


@dataclass
class ModernBERTv3Output:
    """
    Output container for ModernBERT v3 forward pass.

    Attributes:
        last_hidden_state: Final layer output [batch, seq, hidden]
        pooled_outputs: Dict of hub token representations
        hidden_states: All layer outputs (if output_hidden_states=True)
        attentions: All attention weights (if output_attentions=True)

    Example:
        >>> model = ModernBERTv3Ultra(config)
        >>> output = model(input_ids)
        >>> print(output.last_hidden_state.shape)  # [batch, seq, 768]
        >>> print(output.pooled_outputs.keys())  # dict_keys(["[CLS]", "[EMO]", ...])
    """

    last_hidden_state: torch.Tensor
    pooled_outputs: dict
    hidden_states: list | None = None
    attentions: list | None = None


class ModernBERTv3Ultra(nn.Module):
    """
    ModernBERT v3.3 Ultra - Unified FamilyOS Encoder.

    Architecture:
        - 28 transformer layers (vs 22 in v2)
        - 4 hub tokens: [EMO], [MEM], [REL], [TASK]
        - Multi-scale sliding window attention (64→128→256→512)
        - Global attention for hub tokens (positions 0-4)
        - LoRA adapters on Family Band (L23-28)

    Token Layout:
        [CLS] [EMO] [MEM] [REL] [TASK] <text> [SEP] [PAD]...
        pos 0   1     2     3     4     5+

    Capabilities (12 total):
        Hub-routed (9): emotions, sentiment, safety_*, embedding, nli, relation, intent, ingress
        Token-level (3): ner_general, ner_family, temporal

    Args:
        config: ModernBERTv3Config instance with model parameters

    Example:
        >>> config = ModernBERTv3Config()
        >>> model = ModernBERTv3Ultra(config)
        >>> input_ids = torch.randint(0, 50268, (2, 128))
        >>> output = model(input_ids)
        >>> print(output.last_hidden_state.shape)  # [2, 128, 768]
    """

    def __init__(self, config: ModernBERTv3Config):
        super().__init__()
        self.config = config

        # Embeddings
        self.embeddings = ModernBERTEmbeddingsV3(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            max_position_embeddings=config.max_position_embeddings,
            hidden_dropout_prob=config.hidden_dropout_prob,
            pad_token_id=0,  # Default PAD token
            use_rotary_embeddings=True,  # RoPE mode
        )

        # Encoder (28 layers)
        self.encoder = ModernBERTEncoderV3(
            num_layers=config.num_layers,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            use_flash_attention=False,  # Use SDPA for correctness
            gradient_checkpointing=False,  # Disabled by default
            lora_layers=config.lora_target_layers,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
        )

        # Poolers
        self.hub_pooler = HubTokenPooler(
            hidden_size=config.hidden_size,
            add_projection=False,
        )
        self.combined_pooler = CombinedPooler(hidden_size=config.hidden_size)

        # Pair encoder for NLI/relation tasks
        self.pair_encoder = PairEncoderV3(
            hidden_size=config.hidden_size,
            num_labels=3,  # Will be reconfigured per task
            pooling_strategy="rel_hub",
        )

        # Final LayerNorm (optional, some models use this)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

        # Hub positions cache
        self.hub_positions = get_hub_positions()
        self.num_hub_tokens = len(HUB_TOKEN_REGISTRY)

        # Initialize weights
        self.apply(self._init_weights)

        print("\n✓ ModernBERTv3Ultra initialized:")
        print(f"  - Layers: {config.num_layers}")
        print(f"  - Hidden: {config.hidden_size}")
        print(f"  - Heads: {config.num_attention_heads}")
        print(f"  - Hub tokens: {list(HUB_TOKEN_REGISTRY.keys())}")
        print(f"  - LoRA layers: {config.lora_target_layers}")

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize weights for linear, embedding, and LayerNorm layers.

        Args:
            module: Module to initialize
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
    ) -> ModernBERTv3Output | tuple:
        """
        Forward pass for ModernBERT v3.

        Args:
            input_ids: [batch, seq_len] token IDs
            attention_mask: [batch, seq_len] padding mask (1=valid, 0=pad)
            token_type_ids: [batch, seq_len] type IDs (unused)
            position_ids: [batch, seq_len] position IDs (optional)
            output_hidden_states: Return all layer hidden states
            output_attentions: Return all attention weights
            return_dict: Return ModernBERTv3Output or tuple

        Returns:
            ModernBERTv3Output or tuple of tensors

        Example:
            >>> model = ModernBERTv3Ultra(config)
            >>> input_ids = torch.randint(0, 50268, (2, 128))
            >>> output = model(input_ids)
            >>> print(output.last_hidden_state.shape)  # [2, 128, 768]
        """
        # Embeddings
        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )

        # Encoder
        encoder_output, all_hidden_states, all_attentions = self.encoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        # Final LayerNorm
        last_hidden_state = self.final_layer_norm(encoder_output)

        # Pool hub token representations
        pooled_outputs = self.hub_pooler(last_hidden_state, attention_mask)

        if return_dict:
            return ModernBERTv3Output(
                last_hidden_state=last_hidden_state,
                pooled_outputs=pooled_outputs,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
            )
        else:
            return (last_hidden_state, pooled_outputs, all_hidden_states, all_attentions)

    def get_representation_for_capability(
        self,
        last_hidden_state: torch.Tensor,
        pooled_outputs: dict[str, torch.Tensor],
        capability: str,
    ) -> torch.Tensor:
        """
        Get the appropriate representation for a capability.

        Hub-routed capabilities get the hub token representation.
        Token-level capabilities get the full sequence.

        Args:
            last_hidden_state: [batch, seq, hidden]
            pooled_outputs: Dict of hub representations
            capability: Capability name

        Returns:
            Representation tensor

        Example:
            >>> output = model(input_ids)
            >>> emo_repr = model.get_representation_for_capability(
            ...     output.last_hidden_state, output.pooled_outputs, "emotions"
            ... )
            >>> print(emo_repr.shape)  # [batch, 768]
        """
        if capability in TOKEN_LEVEL_CAPABILITIES:
            # Token-level tasks (NER, temporal) need full sequence
            return last_hidden_state
        else:
            # Hub-routed tasks
            hub_token = get_hub_for_capability(capability)
            return pooled_outputs[hub_token]

    def get_embedding_representation(
        self,
        last_hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get embedding for retrieval/similarity tasks.

        Uses the [MEM] hub token at position 2.

        Args:
            last_hidden_state: [batch, seq, hidden]

        Returns:
            Embedding [batch, hidden]

        Example:
            >>> output = model(input_ids)
            >>> embedding = model.get_embedding_representation(output.last_hidden_state)
            >>> print(embedding.shape)  # [batch, 768]
        """
        mem_position = self.hub_positions["[MEM]"]
        return last_hidden_state[:, mem_position, :]

    def freeze_for_phase(self, phase: str) -> None:
        """
        Configure model freezing for a training phase.

        Args:
            phase: "phase0.5" (healing) or "phase1" (full training)

        Example:
            >>> model.freeze_for_phase("phase1")
            ✓ Model configured for phase1:
              ❄️ Frozen: Embeddings, L1-18
              🔥 Trainable: L19-28
        """
        if phase in ["phase0.5", "phase1"]:
            # Freeze embeddings (except hub tokens - handled separately)
            for param in self.embeddings.parameters():
                param.requires_grad_(False)

            # Freeze Foundation + Context bands (L1-18)
            self.encoder.freeze_layers(list(range(1, 19)))

            # Unfreeze Semantic + Family bands (L19-28)
            self.encoder.unfreeze_layers(list(range(19, 29)))

            print(f"✓ Model configured for {phase}:")
            print("  ❄️ Frozen: Embeddings, L1-18")
            print("  🔥 Trainable: L19-28")

    def merge_lora_weights(self) -> None:
        """
        Merge LoRA weights into base weights for inference.

        Call this before exporting the model for deployment.

        Example:
            >>> model.merge_lora_weights()
            ✓ LoRA weights merged into base model
        """
        for layer in self.encoder.layers:
            if hasattr(layer, "merge_lora_weights") and callable(layer.merge_lora_weights):
                layer.merge_lora_weights()
        print("✓ LoRA weights merged into base model")

    def get_input_embeddings(self) -> nn.Embedding:
        """
        Get word embeddings.

        Returns:
            Word embeddings module
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        """
        Set word embeddings.

        Args:
            new_embeddings: New embeddings module
        """
        self.embeddings.word_embeddings = new_embeddings

    def resize_token_embeddings(self, new_vocab_size: int) -> None:
        """
        Resize embeddings for new vocabulary (e.g., adding hub tokens).

        Args:
            new_vocab_size: New vocabulary size

        Example:
            >>> model.resize_token_embeddings(50432)
            ✓ Resized embeddings: 50268 → 50432
        """
        self.embeddings.resize_token_embeddings(new_vocab_size)
        self.config.vocab_size = new_vocab_size

    @property
    def num_parameters(self) -> int:
        """
        Total number of parameters.

        Returns:
            Total parameter count
        """
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """
        Number of trainable parameters.

        Returns:
            Trainable parameter count
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_model_summary(self) -> None:
        """
        Print summary of model architecture.

        Example:
            >>> model.print_model_summary()
            ======================================================================
            📊 ModernBERT v3.3 Ultra - Model Summary
            ======================================================================
              Total parameters:     165,432,192
              Trainable parameters: 165,432,192
              Layers: 28
              ...
        """
        print("\n" + "=" * 70)
        print("📊 ModernBERT v3.3 Ultra - Model Summary")
        print("=" * 70)
        print(f"  Total parameters:     {self.num_parameters:,}")
        print(f"  Trainable parameters: {self.num_trainable_parameters:,}")
        print(f"  Layers: {self.config.num_layers}")
        print(f"  Hidden size: {self.config.hidden_size}")
        print(f"  Attention heads: {self.config.num_attention_heads}")
        print(f"  Hub tokens: {list(self.hub_positions.keys())}")
        print("=" * 70)
        self.encoder.print_layer_summary()


def create_modernbert_v3_ultra(
    from_v2_checkpoint: str | None = None,
    **config_overrides,
) -> ModernBERTv3Ultra:
    """
    Factory function to create ModernBERT v3 Ultra.

    Args:
        from_v2_checkpoint: Path to v2 checkpoint for initialization
        **config_overrides: Override default config values

    Returns:
        Initialized ModernBERTv3Ultra model

    Example:
        >>> model = create_modernbert_v3_ultra(num_layers=24, hidden_size=512)
        >>> print(model.config.num_layers)  # 24
    """
    # Create config with defaults
    config = ModernBERTv3Config(**config_overrides)

    # Create model
    model = ModernBERTv3Ultra(config)

    # Initialize from v2 if provided
    if from_v2_checkpoint:
        print(f"⚠️  v2 checkpoint loading not yet implemented: {from_v2_checkpoint}")
        # from .initialization_v3 import initialize_from_v2
        # initialize_from_v2(model, from_v2_checkpoint)

    return model


# ======================================================================
# Multi-Task Model with Hub Routing (Issue 3.1.5)
# ======================================================================


class ModernBERTv3ForMultiTask(ModernBERTv3Ultra):
    """
    ModernBERT v3 with multi-task heads and hub routing.

    Extends the base model with:
        - Task-specific classification/regression heads
        - Hub token routing to appropriate heads
        - Multi-task loss computation
        - Gradient masking for hub specialization

    Architecture:
        - Shares encoder across all tasks (efficient)
        - Routes hub tokens to task-specific heads
        - Supports simultaneous training on multiple tasks

    Example:
        >>> config = ModernBERTv3Config()
        >>> model = ModernBERTv3ForMultiTask(config)
        >>> model.register_task_head("emotions", ClassificationHead(768, 7))
        >>> output = model.forward_for_task(input_ids, task="emotions", labels=labels)
    """

    def __init__(self, config: ModernBERTv3Config, task_heads: dict | None = None):
        super().__init__(config)

        # Hub router for capability routing
        from .routing_v3 import HubRouter

        self.hub_router = HubRouter()

        # Task heads registry
        self.task_heads = nn.ModuleDict()

        # Loss weights per task (can be adjusted during training)
        self.task_loss_weights: dict[str, float] = {}

        # Active capabilities for current batch
        self._active_capabilities: list[str] = []

        # Register task heads if provided
        if task_heads:
            for task_name, head in task_heads.items():
                self.register_task_head(task_name, head)

        print("\n✓ ModernBERTv3ForMultiTask initialized:")
        print(f"  - Base layers: {config.num_layers}")
        print("  - Hub router: enabled")
        print(f"  - Task heads: {len(self.task_heads)} registered")

    def register_task_head(
        self,
        task_name: str,
        head: nn.Module,
        loss_weight: float = 1.0,
    ) -> None:
        """
        Register a task head.

        Args:
            task_name: Capability name (e.g., "emotions", "ner_general")
            head: Classification/regression head module
            loss_weight: Weight for this task's loss (default: 1.0)

        Example:
            >>> model.register_task_head("emotions", ClassificationHead(768, 7))
              ✓ Registered head: emotions → [EMO] (hub)
        """
        from .routing_v3 import create_hub_routing_info

        self.task_heads[task_name] = head
        self.task_loss_weights[task_name] = loss_weight

        routing_info = create_hub_routing_info(task_name)
        hub_display = routing_info["hub_token"] or "N/A (token-level)"
        print(f"  ✓ Registered head: {task_name} → {hub_display} " f"({routing_info['pool_type']})")

    def forward_for_task(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        task: str | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for a single task.

        Args:
            input_ids: [batch, seq_len] token IDs
            attention_mask: [batch, seq_len] padding mask
            task: Task/capability name
            labels: Ground truth labels (optional, for loss computation)

        Returns:
            Dict with 'logits', optionally 'loss', 'hidden_states'

        Example:
            >>> output = model.forward_for_task(
            ...     input_ids, task="emotions", labels=labels
            ... )
            >>> print(output["logits"].shape)  # [batch, num_labels]
            >>> print(output["loss"])  # scalar
        """
        if task is None:
            raise ValueError("task parameter is required")

        if task not in self.task_heads:
            raise ValueError(f"Unknown task: {task}. Registered: {list(self.task_heads.keys())}")

        # Get encoder output
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Route to appropriate representation
        representation, pool_type = self.hub_router.get_representation_for_capability(
            hidden_states=outputs.last_hidden_state,
            pooled_outputs=outputs.pooled_outputs,
            capability=task,
        )

        # Get task head and compute logits
        head = self.task_heads[task]
        logits = head(representation)

        result = {"logits": logits, "pool_type": pool_type}

        # Compute loss if labels provided
        if labels is not None:
            loss = self._compute_task_loss(task, logits, labels, attention_mask)
            result["loss"] = loss

        return result

    def forward_multitask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        task_labels: dict[str, torch.Tensor] | None = None,
        active_tasks: list[str] | None = None,
        return_all_logits: bool = True,
    ) -> dict:
        """
        Multi-task forward pass with hub routing.

        Processes multiple tasks in a single forward pass, routing
        hub tokens to appropriate task heads.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            task_labels: Dict mapping task names to label tensors
            active_tasks: List of tasks to compute (default: all registered)
            return_all_logits: Return logits for all tasks

        Returns:
            Dict with:
                - 'total_loss': Weighted sum of all task losses
                - 'task_losses': Dict of individual task losses
                - 'task_logits': Dict of task logits (if return_all_logits)
                - 'hub_representations': Dict of hub token vectors
                - 'last_hidden_state': Full sequence representations

        Example:
            >>> output = model.forward_multitask(
            ...     input_ids,
            ...     task_labels={"emotions": emo_labels, "sentiment": sent_labels},
            ...     active_tasks=["emotions", "sentiment"],
            ... )
            >>> print(output["total_loss"])  # scalar
            >>> print(output["task_logits"]["emotions"].shape)  # [batch, 7]
        """
        if active_tasks is None:
            active_tasks = list(self.task_heads.keys())

        self._active_capabilities = active_tasks

        # Get encoder output (single forward pass)
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        task_losses = {}
        task_logits = {}
        total_loss = torch.tensor(0.0, device=input_ids.device)

        # Process each active task
        for task in active_tasks:
            if task not in self.task_heads:
                continue

            # Get appropriate representation via hub routing
            representation, pool_type = self.hub_router.get_representation_for_capability(
                hidden_states=outputs.last_hidden_state,
                pooled_outputs=outputs.pooled_outputs,
                capability=task,
            )

            # Compute logits
            head = self.task_heads[task]
            logits = head(representation)

            if return_all_logits:
                task_logits[task] = logits

            # Compute loss if labels provided for this task
            if task_labels and task in task_labels:
                labels = task_labels[task]
                loss = self._compute_task_loss(task, logits, labels, attention_mask)
                task_losses[task] = loss

                # Add weighted loss to total
                weight = self.task_loss_weights.get(task, 1.0)
                total_loss = total_loss + weight * loss

        return {
            "total_loss": total_loss if task_losses else None,
            "task_losses": task_losses,
            "task_logits": task_logits,
            "hub_representations": outputs.pooled_outputs,
            "last_hidden_state": outputs.last_hidden_state,
        }

    def _compute_task_loss(
        self,
        task: str,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute loss for a task.

        Handles different loss types:
            - Classification: CrossEntropyLoss
            - Token-level: CrossEntropyLoss with mask
            - Regression: MSELoss

        Args:
            task: Task name
            logits: Model predictions
            labels: Ground truth labels
            attention_mask: Mask for token-level tasks

        Returns:
            Loss tensor (scalar)

        Note:
            This is a simple internal implementation for basic loss computation.
            TODO (Epic 5.3.4): Refactor to use HubWeightedMultiTaskLoss from
            losses_v3.py for more sophisticated loss weighting and hub-specific
            gradient control.
        """
        if task in TOKEN_LEVEL_CAPABILITIES:
            # Token-level classification (NER, temporal)
            # logits: [batch, seq, num_labels]
            # labels: [batch, seq]
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
        elif task in ["stsb", "similarity"]:
            # Regression
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.squeeze(-1), labels.float())
        else:
            # Sequence classification
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return loss

    def get_hub_gradient_mask(
        self,
        device: torch.device,
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        """
        Get gradient masks for hub tokens based on active capabilities.

        Used to ensure gradients only flow through hubs that are
        being used for active tasks.

        Args:
            device: Device to create masks on
            batch_size: Batch size

        Returns:
            Dict mapping hub tokens to masks [batch]

        Example:
            >>> model._active_capabilities = ["emotions", "sentiment"]
            >>> masks = model.get_hub_gradient_mask(torch.device("cpu"), 4)
            >>> masks["[EMO]"].shape
            torch.Size([4])
        """
        return self.hub_router.get_hub_gradient_mask(
            active_capabilities=self._active_capabilities,
            batch_size=batch_size,
            device=device,
        )

    def set_task_loss_weight(self, task: str, weight: float) -> None:
        """
        Set loss weight for a task.

        Args:
            task: Task name
            weight: Loss weight (higher = more important)

        Example:
            >>> model.set_task_loss_weight("emotions", 2.0)
            ✓ Loss weight for 'emotions' set to 2.0
        """
        if task not in self.task_heads:
            raise ValueError(f"Unknown task: {task}")
        self.task_loss_weights[task] = weight
        print(f"✓ Loss weight for '{task}' set to {weight}")

    def print_routing_table(self) -> None:
        """
        Print hub routing configuration.

        Example:
            >>> model.print_routing_table()
            📊 Hub Routing Table:
            ------------------------------------------------------------
            Task                 Pool Type    Hub Token
            ------------------------------------------------------------
            emotions             hub          [EMO]
            ner_general          token        N/A (token-level)
            ...
        """
        from .routing_v3 import create_hub_routing_info

        print("\n📊 Hub Routing Table:")
        print("-" * 60)
        print(f"{'Task':<20} {'Pool Type':<12} {'Hub Token':<12}")
        print("-" * 60)
        for task in self.task_heads.keys():
            info = create_hub_routing_info(task)
            hub = info["hub_token"] or "N/A (token-level)"
            print(f"{task:<20} {info['pool_type']:<12} {hub:<12}")
        print("-" * 60)


# ======================================================================
# Task Head Classes
# ======================================================================


class ClassificationHead(nn.Module):
    """
    Simple classification head for hub-routed tasks.

    Used for sequence-level classification tasks like:
    - Emotion detection (7 classes)
    - Sentiment analysis (3 classes)
    - Safety classification (2-5 classes)

    Example:
        >>> head = ClassificationHead(768, 7)
        >>> pooled = torch.randn(4, 768)
        >>> logits = head(pooled)
        >>> print(logits.shape)  # [4, 7]
    """

    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            pooled_output: [batch, hidden] hub token representation

        Returns:
            Logits [batch, num_labels]
        """
        return self.classifier(self.dropout(pooled_output))


class TokenClassificationHead(nn.Module):
    """
    Token-level classification head for NER/temporal.

    Used for token-level tasks like:
    - Named Entity Recognition (9 classes)
    - Temporal expression detection (5 classes)

    Example:
        >>> head = TokenClassificationHead(768, 9)
        >>> sequence = torch.randn(4, 128, 768)
        >>> logits = head(sequence)
        >>> print(logits.shape)  # [4, 128, 9]
    """

    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            sequence_output: [batch, seq, hidden] full sequence

        Returns:
            Logits [batch, seq, num_labels]
        """
        return self.classifier(self.dropout(sequence_output))


class RegressionHead(nn.Module):
    """
    Regression head for similarity tasks.

    Used for regression tasks like:
    - Semantic similarity (STS-B)
    - Relevance scoring

    Example:
        >>> head = RegressionHead(768)
        >>> pooled = torch.randn(4, 768)
        >>> scores = head(pooled)
        >>> print(scores.shape)  # [4, 1]
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            pooled_output: [batch, hidden] hub token representation

        Returns:
            Scores [batch, 1]
        """
        return self.regressor(self.dropout(pooled_output))


def create_v3_multitask_model(
    config: ModernBERTv3Config,
    task_configs: dict[str, dict],
) -> ModernBERTv3ForMultiTask:
    """
    Factory function to create v3 with task heads.

    Args:
        config: Model config
        task_configs: Dict mapping task names to head configs
            Example:
            {
                "emotions": {"type": "classification", "num_labels": 7},
                "ner_general": {"type": "token_classification", "num_labels": 9},
                "embedding": {"type": "none"},  # Uses raw hub output
            }

    Returns:
        Configured multi-task model

    Example:
        >>> config = ModernBERTv3Config()
        >>> task_configs = {
        ...     "emotions": {"type": "classification", "num_labels": 7},
        ...     "sentiment": {"type": "classification", "num_labels": 3},
        ...     "ner_general": {"type": "token_classification", "num_labels": 9},
        ... }
        >>> model = create_v3_multitask_model(config, task_configs)
    """
    model = ModernBERTv3ForMultiTask(config)

    for task_name, head_config in task_configs.items():
        head_type = head_config.get("type", "classification")

        if head_type == "classification":
            head = ClassificationHead(
                hidden_size=config.hidden_size,
                num_labels=head_config["num_labels"],
                dropout=head_config.get("dropout", 0.1),
            )
        elif head_type == "token_classification":
            head = TokenClassificationHead(
                hidden_size=config.hidden_size,
                num_labels=head_config["num_labels"],
                dropout=head_config.get("dropout", 0.1),
            )
        elif head_type == "regression":
            head = RegressionHead(
                hidden_size=config.hidden_size,
                dropout=head_config.get("dropout", 0.1),
            )
        elif head_type == "none":
            # No head - uses raw hub output (e.g., for embeddings)
            continue
        else:
            raise ValueError(f"Unknown head type: {head_type}")

        model.register_task_head(
            task_name,
            head,
            loss_weight=head_config.get("loss_weight", 1.0),
        )

    return model
