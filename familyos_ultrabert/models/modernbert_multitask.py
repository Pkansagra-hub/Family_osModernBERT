"""
ModernBERT Multi-Task Model Architecture - Enhanced v2

This module contains the unified multi-task encoder model built on ModernBERT.
The architecture consists of:
- Shared ModernBERT backbone encoder
- Task-specific classification/regression heads
- Embedding projection head with optional Matryoshka support

Enhanced v2: 9 → 12 capabilities with family-specific additions:
- TEMPORAL: Temporal expression extraction
- RELATION: Family relationship extraction
- INTENT: User intent classification

Epic 5.0 Enhancements:
- Shared poolers (CLSMeanPooler, AttentionPooler)
- Task-specific adapters (BottleneckAdapter, TaskGroupAdapter)
- Cross-attention pair encoder for NLI/Relation tasks

Key Features:
- Single forward pass for multiple tasks
- Dynamic head selection based on requested capabilities
- Gradient scaling per task for balanced multi-task learning
- Support for both sequence and token classification
- Optional adapter layers for parameter-efficient fine-tuning

Classes:
    - ModernBertMultiTaskModel: Main unified model
    - MultiTaskOutput: Output container for multi-task model

Usage:
    from familyos_ultrabert.models.modernbert_multitask import ModernBertMultiTaskModel
    from familyos_ultrabert.data.labels import Capability

    model = ModernBertMultiTaskModel.from_pretrained(
        "answerdotai/ModernBERT-base",
        capabilities=[Capability.NER_GENERAL, Capability.SENTIMENT, Capability.INTENT],
    )
    outputs = model(input_ids, attention_mask, capability=Capability.NER_GENERAL)

    # With Epic 5.0 enhancements:
    model = ModernBertMultiTaskModel.from_pretrained(
        "answerdotai/ModernBERT-base",
        capabilities=[Capability.NLI, Capability.RELATION],
        shared_pooler="cls_mean",      # Use CLSMeanPooler
        use_adapters=True,              # Enable task-group adapters
        use_pair_encoder=True,          # Enable cross-attention for NLI/Relation
    )
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

if TYPE_CHECKING:
    from transformers import PretrainedConfig

from familyos_ultrabert.data.labels import CAPABILITY_TO_LABELS, Capability, get_num_labels

logger = logging.getLogger(__name__)

# Import heads
from familyos_ultrabert.models.heads import (  # noqa: E402
    EmbeddingHead,
    EnhancedSafetyHead,
    GlobalPointerNERHead,
    HierarchicalEmotionHead,
    IntentHead,
    IntentHeadV2,  # V2 Label-Description Embedding
    IngressHeadV2,  # V2 Label-Description Embedding
    NLIHead,
    RelationHead,
    SafetyHead,  # type: ignore  # noqa: F401
    SequenceClassificationHead,
    TemporalHead,
    TokenClassificationHead,
)

# Import Epic 5.0 components (optional - for enhanced mode)
try:
    from familyos_ultrabert.models.adapters import TaskGroupAdapter  # type: ignore
    from familyos_ultrabert.models.pair_encoder import CrossAttentionPairEncoder  # type: ignore
    from familyos_ultrabert.models.poolers import get_pooler

    EPIC_5_AVAILABLE = True
except ImportError:
    EPIC_5_AVAILABLE = False
    logger.debug("Epic 5.0 components not available (adapters, pair_encoder, poolers)")


# =============================================================================
# Task Group Configuration
# =============================================================================


# Define which capabilities belong to which task group (for adapters)
TASK_GROUPS = {
    "token_tasks": [
        Capability.NER_GENERAL,
        Capability.NER_FAMILY,
        Capability.TEMPORAL,
    ],
    "sequence_tasks": [
        Capability.SENTIMENT,
        Capability.EMOTIONS,
        Capability.SAFETY_GENERIC,
        Capability.SAFETY_FAMILYOS,
        Capability.INGRESS,
        Capability.INTENT,
        Capability.INTENT_V2,  # V2 Label-Description Embedding
        Capability.INGRESS_V2,  # V2 Label-Description Embedding
    ],
    "pair_tasks": [
        Capability.NLI,
        Capability.RELATION,
    ],
    "embedding_tasks": [
        Capability.EMBEDDING,
    ],
}


def get_task_group(capability: Capability) -> str:
    """Get the task group for a capability."""
    for group_name, capabilities in TASK_GROUPS.items():
        if capability in capabilities:
            return group_name
    return "sequence_tasks"  # Default fallback


# =============================================================================
# Head Type Mapping (Enhanced v2: 9 → 12 capabilities)
# =============================================================================


CAPABILITY_TO_HEAD_TYPE: dict[Capability, type[nn.Module]] = {
    # GlobalPointer NER heads (v4 SOTA span-based NER)
    Capability.NER_GENERAL: GlobalPointerNERHead,
    Capability.NER_FAMILY: GlobalPointerNERHead,
    Capability.TEMPORAL: GlobalPointerNERHead,  # Temporal expressions as spans
    # Sequence classification heads
    Capability.SENTIMENT: SequenceClassificationHead,
    Capability.EMOTIONS: HierarchicalEmotionHead,  # FIXED: Use enhanced head with 44 emotions
    Capability.SAFETY_GENERIC: SequenceClassificationHead,  # Stage A: Multi-label with ASL
    Capability.SAFETY_FAMILYOS: SafetyHead,  # Stage B: Band-based classification (4 bands, 13 subcats)
    Capability.INGRESS: SequenceClassificationHead,
    Capability.INTENT: IntentHead,  # Legacy single-label
    # V2 Label-Description Embedding heads (SOTA multi-label)
    Capability.INTENT_V2: IntentHeadV2,
    Capability.INGRESS_V2: IngressHeadV2,
    # Special heads
    Capability.NLI: NLIHead,
    Capability.RELATION: RelationHead,  # NEW
    Capability.EMBEDDING: EmbeddingHead,
}


def get_problem_type(capability: Capability) -> str:
    """Get the problem type for a capability (for loss computation)."""
    labels = CAPABILITY_TO_LABELS.get(capability)
    if labels is None:
        return "embedding"
    return labels.problem_type


# =============================================================================
# Multi-Task Model Output
# =============================================================================


class MultiTaskOutput:
    """
    Output container for multi-task model.

    Attributes:
        loss: Total loss (if labels provided)
        logits: Task-specific logits or embeddings
        hidden_states: Encoder hidden states (if output_hidden_states=True)
        attentions: Attention weights (if output_attentions=True)
        capability: The capability that produced this output
    """

    def __init__(
        self,
        loss: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        hidden_states: tuple[torch.Tensor, ...] | None = None,
        attentions: tuple[torch.Tensor, ...] | None = None,
        capability: Capability | str | None = None,
    ):
        self.loss = loss
        self.logits = logits
        self.hidden_states = hidden_states
        self.attentions = attentions
        self.capability = capability

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "loss": self.loss,
            "logits": self.logits,
            "hidden_states": self.hidden_states,
            "attentions": self.attentions,
            "capability": str(self.capability) if self.capability else None,
        }


# =============================================================================
# ModernBERT Multi-Task Model
# =============================================================================


class ModernBertMultiTaskModel(PreTrainedModel):
    """
    Multi-task model with ModernBERT encoder and task-specific heads.

    This model uses a shared ModernBERT encoder with multiple task heads.
    Each forward pass routes to a specific head based on the `capability` parameter.

    Args:
        config: ModernBERT config
        capabilities: List of capabilities to enable (creates corresponding heads)
        freeze_encoder: Whether to freeze encoder weights (for head-only training)
        head_dropout: Dropout probability for classification heads
        shared_pooler: Pooler type for sequence heads ("cls", "mean", "cls_mean", "attention", None)
        use_adapters: Whether to use task-group adapters (Epic 5.0)
        adapter_bottleneck_size: Bottleneck size for adapters (default: 64)
        use_pair_encoder: Whether to use cross-attention pair encoder for NLI/Relation
        pair_encoder_num_layers: Number of cross-attention layers (default: 1)

    Example:
        >>> model = ModernBertMultiTaskModel.from_pretrained(
        ...     "answerdotai/ModernBERT-base",
        ...     capabilities=[Capability.NER_GENERAL, Capability.SENTIMENT],
        ... )
        >>> outputs = model(input_ids, attention_mask, capability="ner_general")

        # With Epic 5.0 enhancements:
        >>> model = ModernBertMultiTaskModel.from_pretrained(
        ...     "answerdotai/ModernBERT-base",
        ...     capabilities=[Capability.NLI, Capability.SENTIMENT],
        ...     shared_pooler="cls_mean",
        ...     use_adapters=True,
        ...     use_pair_encoder=True,
        ... )
    """

    # List of modules that should not be split across devices
    _no_split_modules: list[str] = ["ModernBertEncoderLayer"]

    # Support Flash Attention 2.0 and SDPA
    _supports_flash_attn_2: bool = True
    _supports_sdpa: bool = True

    def __init__(
        self,
        config: PretrainedConfig,
        capabilities: list[Capability | str] | None = None,
        freeze_encoder: bool = False,
        head_dropout: float = 0.1,
        # Epic 5.0 parameters
        shared_pooler: Literal["cls", "mean", "cls_mean", "attention"] | None = None,
        use_adapters: bool = False,
        adapter_bottleneck_size: int = 64,
        use_pair_encoder: bool = False,
        pair_encoder_num_layers: int = 1,
        # GlobalPointer config (from checkpoint metadata)
        _globalpointer_config: dict | None = None,
    ):
        super().__init__(config)

        # Store configuration
        self.capabilities = self._normalize_capabilities(capabilities)
        self.freeze_encoder = freeze_encoder
        self.head_dropout = head_dropout
        self._globalpointer_config = _globalpointer_config or {}

        # Epic 5.0 configuration
        self._shared_pooler_type = shared_pooler
        self._use_adapters = use_adapters
        self._adapter_bottleneck_size = adapter_bottleneck_size
        self._use_pair_encoder = use_pair_encoder
        self._pair_encoder_num_layers = pair_encoder_num_layers

        # Validate Epic 5.0 components are available if requested
        if (use_adapters or use_pair_encoder or shared_pooler) and not EPIC_5_AVAILABLE:
            raise ImportError(
                "Epic 5.0 components requested but not available. "
                "Ensure adapters.py, pair_encoder.py, and poolers.py are installed."
            )

        # Initialize encoder (lazy load - set by from_pretrained or _init_encoder)
        self.encoder: nn.Module | None = None

        # Initialize shared pooler (Epic 5.0)
        self.shared_pooler: nn.Module | None = None
        if shared_pooler is not None and EPIC_5_AVAILABLE:
            hidden_size = getattr(self.config, "hidden_size", 768)
            self.shared_pooler = get_pooler(shared_pooler, hidden_size=hidden_size)
            logger.info(f"Initialized shared pooler: {shared_pooler}")

        # Initialize task-group adapters (Epic 5.0)
        self.task_adapters: nn.Module | None = None
        if use_adapters and EPIC_5_AVAILABLE:
            hidden_size = getattr(self.config, "hidden_size", 768)
            # Determine which task groups are needed based on capabilities
            needed_groups = set()
            for cap in self.capabilities:
                needed_groups.add(get_task_group(cap))
            self.task_adapters = TaskGroupAdapter(
                hidden_size=hidden_size,
                task_groups=list(needed_groups),
                bottleneck_size=adapter_bottleneck_size,
                activation="gelu",
                dropout=head_dropout,
            )
            logger.info(f"Initialized task-group adapters for: {needed_groups}")

        # Initialize cross-attention pair encoder (Epic 5.0)
        self.pair_encoder: nn.Module | None = None
        if use_pair_encoder and EPIC_5_AVAILABLE:
            hidden_size = getattr(self.config, "hidden_size", 768)
            self.pair_encoder = CrossAttentionPairEncoder(
                hidden_size=hidden_size,
                num_heads=8,
                num_layers=pair_encoder_num_layers,
                pooling_strategy="attention",
            )
            logger.info(
                f"Initialized cross-attention pair encoder with {pair_encoder_num_layers} layers"
            )

        # Initialize task heads
        self.heads = nn.ModuleDict()
        self._init_heads()

        # Post-init (weight initialization, etc.)
        self.post_init()

    def _normalize_capabilities(
        self,
        capabilities: list[Capability | str] | None,
    ) -> list[Capability]:
        """Convert capability strings to Capability enum."""
        if capabilities is None:
            # Default: all capabilities
            return list(Capability)

        normalized = []
        for cap in capabilities:
            if isinstance(cap, str):
                cap = Capability(cap)
            normalized.append(cap)
        return normalized

    def _init_heads(self) -> None:
        """Initialize task-specific heads based on capabilities."""
        hidden_size = getattr(self.config, "hidden_size", 768)

        for capability in self.capabilities:
            head_cls = CAPABILITY_TO_HEAD_TYPE.get(capability)
            if head_cls is None:
                raise ValueError(f"Unknown capability: {capability}")

            num_labels = get_num_labels(capability)
            problem_type = get_problem_type(capability)

            # Create head with appropriate parameters
            if capability == Capability.EMBEDDING:
                head = head_cls(
                    hidden_size=hidden_size,
                    normalize=True,
                )
            elif capability in (Capability.NER_GENERAL, Capability.NER_FAMILY, Capability.TEMPORAL):
                # GlobalPointer heads for span-based NER (v4)
                # Use config from checkpoint metadata if available
                cap_key = capability.value
                gp_info = self._globalpointer_config.get(cap_key, {})
                gp_num_labels = gp_info.get("num_labels", num_labels)
                gp_head_size = gp_info.get("head_size", 64)
                head = head_cls(
                    hidden_size=hidden_size,
                    num_labels=gp_num_labels,
                    head_size=gp_head_size,
                    dropout=self.head_dropout,
                    use_rope=True,
                    loss_type="globalpointer",
                )
            elif capability == Capability.SAFETY_FAMILYOS:
                # SafetyHead: 4 bands with 13 subcategories (indices 0-12)
                # Used for Stage B FamilyOS domain adaptation
                head = head_cls(
                    hidden_size=hidden_size,
                    num_bands=4,  # GREEN, AMBER, RED, CRISIS
                    num_subcategories=13,  # 13 subcategories: none(0) + 4 AMBER + 4 RED + 4 CRISIS
                    dropout=self.head_dropout,
                    use_hierarchical=True,
                )
            elif capability == Capability.EMOTIONS:
                # HierarchicalEmotionHead: 44 FamilyOS emotions
                # CRITICAL: Use plain BCE loss - ASL/Focal causes collapse!
                # Expert advice: "Never use ASL, Focal, or class-balanced anything"
                head = head_cls(
                    hidden_size=hidden_size,
                    num_emotions=num_labels,  # 44 emotions from labels.py
                    num_secondary=3,
                    dropout=self.head_dropout,
                    pooling="cls",
                    use_intensity=True,
                    use_valence_arousal=False,
                    use_familyos=True,  # Enable FamilyOS 44-emotion schema
                    # DISABLED: All complex losses that caused collapse
                    use_asl=False,  # ← DISABLED - use plain BCE instead
                    asl_gamma_neg=0.0,  # Not used when use_asl=False
                    asl_gamma_pos=0.0,  # Not used when use_asl=False
                    asl_clip=0.0,  # Not used when use_asl=False
                    use_hierarchical_loss=False,  # ← DISABLED for stability
                    use_label_correlation=False,  # ← DISABLED for stability
                    use_emotion_attention=False,  # Disabled (more compute)
                    use_dynamic_thresholds=False,  # ← DISABLED for stability
                    use_mixup=False,  # ← DISABLED for stability
                    label_smoothing=0.0,  # ← DISABLED for stability
                )
            elif capability == Capability.SAFETY_GENERIC:
                # SOTA Multi-label safety head with ASL for Stage A
                # Uses Asymmetric Loss for handling class imbalance in toxicity detection
                #
                # Updated for CURATED Civil Comments (balanced dataset):
                #   - toxic: 61.2% -> weight 1.6
                #   - severe_toxic: 12.5% -> weight 8.0
                #   - obscene: 17.0% -> weight 5.9
                #   - threat: 16.3% -> weight 6.1
                #   - insult: 43.6% -> weight 2.3
                #   - identity_hate: 21.3% -> weight 4.7
                #   - sexually_explicit: 12.5% -> weight 8.0
                #   - profanity: 12.5% -> weight 8.0
                #
                # pos_weight = 100 / percentage (inverse frequency)
                import torch

                safety_pos_weight = torch.tensor(
                    [
                        1.6,  # toxic (idx 0) - 61.2% positive
                        8.0,  # severe_toxic (idx 1) - 12.5% positive
                        5.9,  # obscene (idx 2) - 17.0% positive
                        6.1,  # threat (idx 3) - 16.3% positive
                        2.3,  # insult (idx 4) - 43.6% positive
                        4.7,  # identity_hate (idx 5) - 21.3% positive
                        8.0,  # sexually_explicit (idx 6) - 12.5% positive
                        8.0,  # profanity (idx 7) - 12.5% positive
                    ]
                )
                head = head_cls(
                    hidden_size=hidden_size,
                    num_labels=num_labels,  # 8 toxicity types
                    dropout=self.head_dropout,
                    problem_type="multi_label_classification",
                    use_asl=True,  # SOTA: Asymmetric Loss for multi-label
                    asl_gamma_neg=4.0,  # Standard ASL (data is balanced now)
                    asl_gamma_pos=1.0,  # Standard ASL
                    asl_clip=0.05,  # Probability clipping
                    pos_weight=safety_pos_weight,  # Mild reweighting for balanced data
                )
            elif capability in (Capability.INTENT_V2, Capability.INGRESS_V2):
                # V2 Label-Description Embedding heads (Milestone 2)
                # Different constructor signature - no problem_type arg
                head = head_cls(
                    hidden_size=hidden_size,
                    num_labels=num_labels,
                    dropout=self.head_dropout,
                    multi_label=True,  # K1 requirement: multi-label classification
                )
            else:
                head = head_cls(
                    hidden_size=hidden_size,
                    num_labels=num_labels,
                    dropout=self.head_dropout,
                    problem_type=problem_type,
                )

            self.heads[capability.value] = head

    def _init_encoder(self) -> None:
        """Initialize encoder from ModernBERT."""
        from transformers import AutoModel

        if self.encoder is None:
            self.encoder = AutoModel.from_config(self.config)

            if self.freeze_encoder:
                for param in self.encoder.parameters():  # type: ignore
                    param.requires_grad = False

    def get_encoder(self) -> nn.Module:
        """Get the encoder module."""
        if self.encoder is None:
            self._init_encoder()
        return self.encoder  # pyright: ignore[reportReturnType]

    def get_head(self, capability: Capability | str) -> nn.Module:
        """Get a specific task head."""
        if isinstance(capability, Capability):
            capability = capability.value
        return self.heads[capability]

    def freeze_encoder_weights(self) -> None:
        """Freeze all encoder weights."""
        if self.encoder is not None:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.freeze_encoder = True

    def unfreeze_encoder_weights(self) -> None:
        """Unfreeze all encoder weights."""
        if self.encoder is not None:
            for param in self.encoder.parameters():
                param.requires_grad = True
        self.freeze_encoder = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        capability: Capability | str | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
        **kwargs,
    ) -> MultiTaskOutput | tuple:
        """
        Forward pass through encoder and selected task head.

        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            token_type_ids: Token type IDs for sentence pairs (batch_size, seq_len)
            labels: Task labels (shape depends on task)
            capability: Which capability/head to use for this batch
            output_hidden_states: Whether to return all hidden states
            output_attentions: Whether to return attention weights
            return_dict: Whether to return a structured output

        Returns:
            MultiTaskOutput or tuple with loss and logits
        """
        # Ensure encoder is initialized
        if self.encoder is None:
            self._init_encoder()

        # Normalize capability
        if capability is None:
            raise ValueError("Must specify a capability for forward pass")
        if isinstance(capability, str):
            capability = Capability(capability)

        # Check capability is enabled
        if capability.value not in self.heads:
            raise ValueError(
                f"Capability '{capability}' not enabled. " f"Available: {list(self.heads.keys())}"
            )

        # Encode input
        # Note: ModernBERT doesn't support token_type_ids, so we don't pass it
        encoder_outputs: BaseModelOutput = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )  # type: ignore

        # Get sequence output (batch_size, seq_len, hidden_size)
        sequence_output = encoder_outputs.last_hidden_state

        # Epic 5.0: Apply task-group adapters if enabled
        if self.task_adapters is not None:
            task_group = get_task_group(capability)
            sequence_output = self.task_adapters(sequence_output, task_group=task_group)

        # Get the appropriate head and compute output
        head = self.heads[capability.value]

        # Forward through head
        if capability == Capability.EMBEDDING:
            # Embedding head: pass attention mask for pooling
            logits = head(sequence_output, attention_mask=attention_mask)
            loss = None
        elif capability in (Capability.NLI, Capability.RELATION) and self.pair_encoder is not None:
            # Epic 5.0: Use cross-attention pair encoder for NLI/Relation tasks
            # For pair tasks, we need to split by sep_token or use token_type_ids
            # The pair encoder expects (text_a_embeds, text_b_embeds, masks)
            # For now, we use the head directly but provide the pair encoder output
            # NOTE: Full pair encoding requires input splitting - handled by head
            head_output = head(
                sequence_output,
                attention_mask=attention_mask,
                labels=labels,
                pair_encoder=self.pair_encoder,  # Pass encoder for head to use
            )
            logits = head_output.get("logits")
            loss = head_output.get("loss")
        else:
            # Classification heads
            head_output = head(
                sequence_output,
                attention_mask=attention_mask,
                labels=labels,
            )
            logits = head_output.get("logits")
            loss = head_output.get("loss")

        if not return_dict:
            output = (logits,)
            if loss is not None:
                output = (loss,) + output
            return output

        return MultiTaskOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions if output_attentions else None,
            capability=capability,
        )

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
    ) -> ModernBertMultiTaskModel:
        """
        Load model from a training checkpoint (saved with Trainer).

        This handles the 'encoder.' prefix in saved state dicts and loads
        capabilities from capabilities.json.

        Args:
            checkpoint_path: Path to checkpoint directory
            device: Device to load model on

        Returns:
            Loaded ModernBertMultiTaskModel
        """
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel

        checkpoint_path = (
            Path(checkpoint_path) if not isinstance(checkpoint_path, Path) else checkpoint_path
        )  # pyright: ignore[reportAssignmentType]

        # Load state dict first to detect actual head types
        safetensors_path = checkpoint_path / "model.safetensors"  # type: ignore
        pytorch_path = checkpoint_path / "pytorch_model.bin"  # type: ignore

        if safetensors_path.exists():
            state_dict = load_file(str(safetensors_path))
        elif pytorch_path.exists():
            state_dict = torch.load(str(pytorch_path), map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(
                f"No model weights found at {checkpoint_path}. "
                f"Expected 'model.safetensors' or 'pytorch_model.bin'"
            )

        # Detect actual head names from state dict (authoritative source)
        checkpoint_heads = set()
        for key in state_dict.keys():
            if key.startswith("heads."):
                head_name = key.split(".")[1]
                checkpoint_heads.add(head_name)
        logger.info(f"Detected heads in checkpoint: {sorted(checkpoint_heads)}")

        # Load capabilities and Epic 5.0 config
        caps_file = checkpoint_path / "capabilities.json"  # type: ignore
        capabilities = None
        epic_5_config = {}
        if caps_file.exists():
            with open(caps_file) as f:
                caps_data = json.load(f)
                # Handle both old format (list) and new format (dict with 'capabilities' key)
                if isinstance(caps_data, list):
                    capabilities = [Capability(c) for c in caps_data]
                elif isinstance(caps_data, dict):
                    if "capabilities" in caps_data:
                        capabilities = [Capability(c) for c in caps_data["capabilities"]]
                    if "epic_5_0" in caps_data:
                        epic_5_config = caps_data["epic_5_0"]

        # Reconcile capabilities with actual checkpoint heads
        # V2 heads replace V1: if checkpoint has intent_v2, use that instead of intent
        # Special case: EmbeddingHead has no parameters, always create it
        PARAMETER_FREE_HEADS = {"embedding"}  # Heads that don't need checkpoint weights

        if capabilities:
            reconciled = []
            for cap in capabilities:
                cap_name = cap.value if isinstance(cap, Capability) else cap
                # Check for V2 replacement
                if cap_name == "intent" and "intent_v2" in checkpoint_heads:
                    reconciled.append(Capability.INTENT_V2)
                    logger.info("  Detected intent_v2 head, using INTENT_V2 capability")
                elif cap_name == "ingress" and "ingress_v2" in checkpoint_heads:
                    reconciled.append(Capability.INGRESS_V2)
                    logger.info("  Detected ingress_v2 head, using INGRESS_V2 capability")
                elif cap_name in PARAMETER_FREE_HEADS:
                    # Always create parameter-free heads (e.g., EmbeddingHead)
                    reconciled.append(cap)
                    logger.info(f"  Creating parameter-free head: {cap_name}")
                elif cap_name not in checkpoint_heads:
                    # Skip capabilities not in checkpoint
                    logger.warning(f"  Capability '{cap_name}' not found in checkpoint, skipping")
                else:
                    reconciled.append(cap)
            # Add any heads in checkpoint but not in capabilities.json
            for head_name in checkpoint_heads:
                if head_name not in [c.value if isinstance(c, Capability) else c for c in reconciled]:
                    try:
                        reconciled.append(Capability(head_name))
                        logger.info(f"  Added missing capability from checkpoint: {head_name}")
                    except ValueError:
                        logger.warning(f"  Unknown head '{head_name}' in checkpoint, skipping")
            capabilities = reconciled

        # Load GlobalPointer metadata if available (for correct num_labels)
        gp_metadata_file = checkpoint_path / "globalpointer_metadata.json"  # type: ignore
        globalpointer_config = {}
        if gp_metadata_file.exists():
            with open(gp_metadata_file) as f:
                gp_data = json.load(f)
                if "head_info" in gp_data:
                    globalpointer_config = gp_data["head_info"]
                    logger.info(
                        f"Loaded GlobalPointer config for: {list(globalpointer_config.keys())}"
                    )

        # Load config - use local_files_only to avoid HuggingFace Hub validation issues
        config = AutoConfig.from_pretrained(str(checkpoint_path), local_files_only=True)

        # Create model instance with Epic 5.0 parameters
        model = cls(
            config=config,
            capabilities=capabilities,  # type: ignore
            freeze_encoder=False,
            head_dropout=0.1,
            shared_pooler=epic_5_config.get("shared_pooler"),
            use_adapters=epic_5_config.get("use_adapters", False),
            adapter_bottleneck_size=epic_5_config.get("adapter_bottleneck_size", 64),
            use_pair_encoder=epic_5_config.get("use_pair_encoder", False),
            pair_encoder_num_layers=epic_5_config.get("pair_encoder_num_layers", 1),
            _globalpointer_config=globalpointer_config,  # Pass to __init__ for correct head init
        )

        # State dict already loaded above - separate by component
        encoder_state = {}
        head_state = {}
        adapter_state = {}
        pair_encoder_state = {}
        pooler_state = {}

        for key, value in state_dict.items():
            if key.startswith("encoder."):
                encoder_state[key[8:]] = value  # Remove 'encoder.' prefix
            elif key.startswith("heads."):
                head_state[key] = value
            elif key.startswith("task_adapters."):
                adapter_state[key] = value
            elif key.startswith("pair_encoder."):
                pair_encoder_state[key] = value
            elif key.startswith("shared_pooler."):
                pooler_state[key] = value

        # Initialize encoder - from checkpoint if available, else from pretrained
        if encoder_state:
            # Full checkpoint with encoder weights
            model.encoder = AutoModel.from_config(config)
            missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
            if missing:
                logger.warning(f"Encoder missing keys: {len(missing)}")
            if unexpected:
                logger.warning(f"Encoder unexpected keys: {len(unexpected)}")
        else:
            # Heads-only checkpoint - load encoder from pretrained base model
            # Note: _name_or_path may be set to the checkpoint path itself, so we
            # explicitly use the known base model for ModernBERT
            base_model_name = "answerdotai/ModernBERT-base"
            logger.info(f"No encoder weights in checkpoint, loading from: {base_model_name}")
            try:
                model.encoder = AutoModel.from_pretrained(base_model_name)
            except Exception as e:
                # Fallback to local config
                logger.warning(f"Could not load pretrained encoder ({e}), using random init")
                model.encoder = AutoModel.from_config(config)

        # Load heads and Epic 5.0 components
        components_state = {**head_state, **adapter_state, **pair_encoder_state, **pooler_state}
        if components_state:
            # Filter out keys with size mismatches (e.g., Stage A 7-label -> Stage B 44-label)
            model_state = model.state_dict()
            filtered_state = {}
            skipped_keys = []
            for key, value in components_state.items():
                if key in model_state:
                    if model_state[key].shape == value.shape:
                        filtered_state[key] = value
                    else:
                        skipped_keys.append(
                            f"{key}: checkpoint {value.shape} vs model {model_state[key].shape}"
                        )
                else:
                    # Key not in model, let load_state_dict handle it
                    filtered_state[key] = value

            if skipped_keys:
                logger.warning(
                    f"Skipping {len(skipped_keys)} keys with size mismatch (will be reinitialized):"
                )
                for sk in skipped_keys:
                    logger.warning(f"  - {sk}")

            if filtered_state:
                missing_h, unexpected_h = model.load_state_dict(filtered_state, strict=False)
            loaded_count = len(filtered_state)
            logger.info(
                f"Loaded {loaded_count} component parameters (heads, adapters, pair_encoder, pooler)"
            )

        model.to(device)  # type: ignore
        model.eval()

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        capabilities: list[Capability | str] | None = None,
        freeze_encoder: bool = False,
        head_dropout: float = 0.1,
        # Epic 5.0 parameters
        shared_pooler: Literal["cls_mean", "attention"] | None = None,
        use_adapters: bool = False,
        adapter_bottleneck_size: int = 64,
        use_pair_encoder: bool = False,
        pair_encoder_num_layers: int = 1,
        **kwargs,
    ) -> ModernBertMultiTaskModel:
        """
        Load a pretrained ModernBERT and add task heads.

        Args:
            pretrained_model_name_or_path: HuggingFace model name or path
            capabilities: List of capabilities to enable
            freeze_encoder: Whether to freeze encoder weights
            head_dropout: Dropout for classification heads
            shared_pooler: Epic 5.0 - Type of shared pooler (None, "cls_mean", "attention")
            use_adapters: Epic 5.0 - Whether to use task-group adapters
            adapter_bottleneck_size: Epic 5.0 - Bottleneck dimension for adapters
            use_pair_encoder: Epic 5.0 - Whether to use cross-attention pair encoder
            pair_encoder_num_layers: Epic 5.0 - Number of cross-attention layers
            **kwargs: Additional arguments for from_pretrained

        Returns:
            Initialized ModernBertMultiTaskModel

        Example:
            >>> model = ModernBertMultiTaskModel.from_pretrained(
            ...     "answerdotai/ModernBERT-base",
            ...     capabilities=[Capability.NER_GENERAL, Capability.SENTIMENT],
            ... )

            # With Epic 5.0 enhancements:
            >>> model = ModernBertMultiTaskModel.from_pretrained(
            ...     "answerdotai/ModernBERT-base",
            ...     capabilities=[Capability.NLI, Capability.RELATION],
            ...     shared_pooler="cls_mean",
            ...     use_adapters=True,
            ...     use_pair_encoder=True,
            ... )
        """
        from transformers import AutoConfig, AutoModel

        # Load config
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Create model instance with Epic 5.0 parameters
        model = cls(
            config=config,
            capabilities=capabilities,
            freeze_encoder=freeze_encoder,
            head_dropout=head_dropout,
            shared_pooler=shared_pooler,
            use_adapters=use_adapters,
            adapter_bottleneck_size=adapter_bottleneck_size,
            use_pair_encoder=use_pair_encoder,
            pair_encoder_num_layers=pair_encoder_num_layers,
        )

        # Load pretrained encoder weights
        model.encoder = AutoModel.from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            **kwargs,
        )

        # Freeze if requested
        if freeze_encoder:
            model.freeze_encoder_weights()

        return model

    def save_pretrained(
        self,
        save_directory: str,
        **kwargs,
    ) -> None:
        """
        Save model to directory.

        Saves:
            - config.json: Model configuration
            - model.safetensors: All weights (encoder + heads + adapters)
            - capabilities.json: List of enabled capabilities
            - training_config.json: Training config including Epic 5.0 settings
        """
        os.makedirs(save_directory, exist_ok=True)

        # Handle shared tensors (e.g., pair_encoder shared between model and heads)
        # The pair_encoder is shared by reference, so we need to use safe_serialization=False
        # or explicitly handle the shared weights
        if "safe_serialization" not in kwargs:
            kwargs["safe_serialization"] = False

        # Save using parent class method
        super().save_pretrained(save_directory, **kwargs)

        # Save capabilities and Epic 5.0 config
        capabilities_path = os.path.join(save_directory, "capabilities.json")

        config_data = {
            "capabilities": [c.value for c in self.capabilities],
            "epic_5_0": {
                "shared_pooler": self._shared_pooler_type,
                "use_adapters": self._use_adapters,
                "adapter_bottleneck_size": self._adapter_bottleneck_size,
                "use_pair_encoder": self._use_pair_encoder,
                "pair_encoder_num_layers": self._pair_encoder_num_layers,
            },
        }
        with open(capabilities_path, "w") as f:
            json.dump(config_data, f, indent=2)

    def get_input_embeddings(self) -> nn.Module:
        """Get input embeddings layer."""
        if self.encoder is None:
            self._init_encoder()
        return self.encoder.get_input_embeddings()  # type: ignore

    def set_input_embeddings(self, value: nn.Module) -> None:
        """Set input embeddings layer."""
        if self.encoder is None:
            self._init_encoder()
        self.encoder.set_input_embeddings(value)  # type: ignore

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        """Enable gradient checkpointing for memory efficiency."""
        if self.encoder is None:
            self._init_encoder()
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs)  # type: ignore

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing."""
        if self.encoder is None:
            self._init_encoder()
        self.encoder.gradient_checkpointing_disable()  # type: ignore


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "ModernBertMultiTaskModel",
    "MultiTaskOutput",
    "CAPABILITY_TO_HEAD_TYPE",
    "get_problem_type",
]
#   - Instantiate appropriate head classes
#   - Handle frozen vs trainable heads
