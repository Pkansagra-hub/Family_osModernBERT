"""
Hub-Aware Task Heads for ModernBERT v3.3 Ultra.

This module implements task heads that automatically receive the correct representation
based on their capability's hub routing. Each head knows which hub token provides
its input representation.

Hub Routing:
    - [EMO] → emotions, sentiment, safety_*
    - [MEM] → embedding (no head, raw output)
    - [REL] → nli, relation
    - [TASK] → intent, ingress
    - Token-level → ner_general, ner_family, temporal (use full sequence)

Head Classes:
    - HubAwareClassificationHead: Standard classification for hub-routed tasks
    - HubAwareTokenClassificationHead: Token-level classification (NER, temporal)
    - HubAwareHierarchicalHead: Hierarchical emotions (Ekman → GoEmotions)
    - HubAwareSafetyHead: Safety classification with temperature calibration
    - HubAwareNLIHead: NLI with [REL] hub and two-layer classifier

Author: FamilyOS Team
Date: December 2025
Version: 3.3
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .hub_tokens import (
    TOKEN_LEVEL_CAPABILITIES,
    get_hub_for_capability,
    get_hub_positions,
)


@dataclass
class HeadConfig:
    """
    Configuration for a task head.

    Attributes:
        name: Task/capability name (e.g., "emotions", "ner_general")
        num_labels: Number of output labels
        head_type: Type of head ("classification", "token", "regression", "hierarchical")
        hub_token: Which hub token routes to this head (e.g., "[EMO]")
        hidden_size: Model hidden dimension (default: 768)
        dropout: Dropout probability (default: 0.1)
        loss_weight: Weight for this task's loss in multi-task training (default: 1.0)
        hierarchy: Optional hierarchy specification for hierarchical heads
    """

    name: str
    num_labels: int
    head_type: str  # "classification", "token", "regression", "hierarchical"
    hub_token: str  # Which hub routes to this head
    hidden_size: int = 768
    dropout: float = 0.1
    loss_weight: float = 1.0
    hierarchy: dict | None = None  # For hierarchical heads


# ======================================================================
# Hub-Aware Classification Heads
# ======================================================================


class HubAwareClassificationHead(nn.Module):
    """
    Classification head that receives input from a specific hub token.

    This head automatically extracts the correct hub token representation
    and applies a simple linear classifier.

    Used for: emotions, sentiment, safety_*, intent, ingress, relation

    Example:
        >>> head = HubAwareClassificationHead(768, 7, hub_token="[EMO]")
        >>> pooled_outputs = {"[EMO]": torch.randn(4, 768)}
        >>> logits = head(None, pooled_outputs)
        >>> print(logits.shape)  # [4, 7]
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_labels: int = 2,
        dropout: float = 0.1,
        hub_token: str = "[CLS]",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.hub_token = hub_token
        self.hub_position = get_hub_positions()[hub_token]

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        hidden_states: torch.Tensor | None,
        pooled_outputs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: [batch, seq, hidden] - full sequence (optional)
            pooled_outputs: Dict of hub representations (preferred)

        Returns:
            Logits [batch, num_labels]
        """
        if pooled_outputs is not None and self.hub_token in pooled_outputs:
            # Use pre-pooled hub representation
            pooled = pooled_outputs[self.hub_token]
        elif hidden_states is not None:
            # Extract from sequence
            pooled = hidden_states[:, self.hub_position, :]
        else:
            raise ValueError("Either hidden_states or pooled_outputs must be provided")

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

    def extra_repr(self) -> str:
        return f"hub={self.hub_token}, labels={self.num_labels}"


class HubAwareTokenClassificationHead(nn.Module):
    """
    Token-level classification head for sequence labeling.

    This head receives the full sequence output and applies classification
    at each token position. Used for NER and temporal expression detection.

    Used for: ner_general, ner_family, temporal

    Note: Token-level tasks do NOT use hub pooling - they need
    the full sequence output.

    Example:
        >>> head = HubAwareTokenClassificationHead(768, 9)
        >>> sequence = torch.randn(4, 128, 768)
        >>> logits = head(sequence)
        >>> print(logits.shape)  # [4, 128, 9]
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_labels: int = 9,  # NER tags
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: [batch, seq, hidden]
            attention_mask: [batch, seq] (optional, for masking predictions)

        Returns:
            Logits [batch, seq, num_labels]
        """
        # Apply dropout and classifier to full sequence
        # Hub token positions (0-4) will be masked during loss computation
        sequence_output = self.dropout(hidden_states)
        logits = self.classifier(sequence_output)

        return logits

    def get_predictions(
        self,
        logits: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get predicted labels, masking special tokens.

        Args:
            logits: [batch, seq, num_labels]
            attention_mask: [batch, seq]

        Returns:
            Predictions [batch, seq] with -100 for special positions
        """
        predictions = logits.argmax(dim=-1)

        # Mask positions 0-4 (CLS + hub tokens)
        predictions[:, :5] = -100

        # Mask padding
        predictions = predictions.masked_fill(attention_mask == 0, -100)

        return predictions

    def extra_repr(self) -> str:
        return f"labels={self.num_labels}"


class HubAwareHierarchicalHead(nn.Module):
    """
    Hierarchical classification head for emotions.

    This head implements a two-level hierarchy:
    1. Primary: Ekman emotions (7 classes)
    2. Secondary: GoEmotions (28 classes), conditioned on primary

    Uses [EMO] hub token for representation.

    Example:
        >>> head = HubAwareHierarchicalHead(768)
        >>> pooled_outputs = {"[EMO]": torch.randn(4, 768)}
        >>> primary_logits, secondary_logits = head(None, pooled_outputs)
        >>> print(primary_logits.shape)  # [4, 7]
        >>> print(secondary_logits.shape)  # [4, 28]
    """

    def __init__(
        self,
        hidden_size: int = 768,
        primary_labels: int = 7,  # Ekman emotions
        secondary_labels: int = 28,  # GoEmotions
        dropout: float = 0.1,
        hub_token: str = "[EMO]",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.primary_labels = primary_labels
        self.secondary_labels = secondary_labels
        self.hub_token = hub_token
        self.hub_position = get_hub_positions()[hub_token]

        self.dropout = nn.Dropout(dropout)

        # Primary classifier (Ekman)
        self.primary_classifier = nn.Linear(hidden_size, primary_labels)

        # Secondary classifier (GoEmotions) - conditioned on primary
        self.secondary_classifier = nn.Sequential(
            nn.Linear(hidden_size + primary_labels, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, secondary_labels),
        )

        # Emotion hierarchy mapping (Ekman -> GoEmotions indices)
        # This would be populated from labels.py emotion hierarchy
        # For now, return identity (no masking)
        self.hierarchy_mask = self._build_hierarchy_mask(primary_labels, secondary_labels)

    def _build_hierarchy_mask(
        self,
        primary: int,
        secondary: int,
    ) -> torch.Tensor:
        """
        Build mask enforcing hierarchy constraints.

        This would ideally map Ekman emotions to their corresponding GoEmotions.
        For now, returns an all-ones mask (no constraints).

        Args:
            primary: Number of primary labels
            secondary: Number of secondary labels

        Returns:
            Mask tensor [primary, secondary]
        """
        # TODO: Implement actual hierarchy from emotion taxonomy
        return torch.ones(primary, secondary)

    def forward(
        self,
        hidden_states: torch.Tensor | None,
        pooled_outputs: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with hierarchical predictions.

        Args:
            hidden_states: [batch, seq, hidden] (optional)
            pooled_outputs: Dict of hub representations (preferred)

        Returns:
            Tuple of (primary_logits, secondary_logits)
                - primary_logits: [batch, primary_labels]
                - secondary_logits: [batch, secondary_labels]
        """
        if pooled_outputs is not None and self.hub_token in pooled_outputs:
            pooled = pooled_outputs[self.hub_token]
        elif hidden_states is not None:
            pooled = hidden_states[:, self.hub_position, :]
        else:
            raise ValueError("Either hidden_states or pooled_outputs must be provided")

        pooled = self.dropout(pooled)

        # Primary prediction (Ekman)
        primary_logits = self.primary_classifier(pooled)
        primary_probs = torch.softmax(primary_logits, dim=-1)

        # Secondary prediction conditioned on primary
        secondary_input = torch.cat([pooled, primary_probs], dim=-1)
        secondary_logits = self.secondary_classifier(secondary_input)

        return primary_logits, secondary_logits

    def extra_repr(self) -> str:
        return (
            f"hub={self.hub_token}, "
            f"primary={self.primary_labels}, "
            f"secondary={self.secondary_labels}"
        )


class HubAwareSafetyHead(nn.Module):
    """
    Safety classification head with calibrated outputs.

    Uses [EMO] hub token (safety correlates with emotional content).

    Features:
        - Binary classification (safe/unsafe)
        - Temperature-based confidence calibration
        - Threshold-based prediction with confidence filtering

    Example:
        >>> head = HubAwareSafetyHead(768, confidence_threshold=0.7)
        >>> pooled_outputs = {"[EMO]": torch.randn(4, 768)}
        >>> logits, confidence = head(None, pooled_outputs, return_confidence=True)
        >>> print(logits.shape)  # [4, 2]
        >>> print(confidence.shape)  # [4]
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_labels: int = 2,  # Safe / Unsafe
        dropout: float = 0.1,
        hub_token: str = "[EMO]",
        confidence_threshold: float = 0.5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.hub_token = hub_token
        self.hub_position = get_hub_positions()[hub_token]
        self.confidence_threshold = confidence_threshold

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Temperature for calibration (learned parameter)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(
        self,
        hidden_states: torch.Tensor | None,
        pooled_outputs: dict[str, torch.Tensor] | None = None,
        return_confidence: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional confidence.

        Args:
            hidden_states: [batch, seq, hidden] (optional)
            pooled_outputs: Dict of hub representations (preferred)
            return_confidence: Whether to return confidence scores

        Returns:
            Logits [batch, 2] or (logits, confidence) if return_confidence=True
        """
        if pooled_outputs is not None and self.hub_token in pooled_outputs:
            pooled = pooled_outputs[self.hub_token]
        elif hidden_states is not None:
            pooled = hidden_states[:, self.hub_position, :]
        else:
            raise ValueError("Either hidden_states or pooled_outputs must be provided")

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled) / self.temperature

        if return_confidence:
            probs = torch.softmax(logits, dim=-1)
            confidence = probs.max(dim=-1).values
            return logits, confidence

        return logits

    def predict_with_threshold(
        self,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with confidence threshold.

        Args:
            logits: [batch, num_labels]

        Returns:
            Tuple of (predictions, is_confident)
                - predictions: [batch] - predicted class indices
                - is_confident: [batch] - boolean mask indicating if prediction meets threshold
        """
        probs = torch.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values
        predictions = logits.argmax(dim=-1)
        is_confident = confidence >= self.confidence_threshold

        return predictions, is_confident

    def extra_repr(self) -> str:
        return (
            f"hub={self.hub_token}, "
            f"labels={self.num_labels}, "
            f"threshold={self.confidence_threshold}"
        )


class HubAwareNLIHead(nn.Module):
    """
    NLI head using [REL] hub token.

    This head uses the [REL] hub token which captures relationship
    information between premise and hypothesis in NLI tasks.

    Labels: entailment (0), neutral (1), contradiction (2)

    Example:
        >>> head = HubAwareNLIHead(768)
        >>> pooled_outputs = {"[REL]": torch.randn(4, 768)}
        >>> logits = head(None, pooled_outputs)
        >>> print(logits.shape)  # [4, 3]
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_labels: int = 3,
        dropout: float = 0.1,
        hub_token: str = "[REL]",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.hub_token = hub_token
        self.hub_position = get_hub_positions()[hub_token]

        self.dropout = nn.Dropout(dropout)

        # Two-layer classifier for NLI
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels),
        )

    def forward(
        self,
        hidden_states: torch.Tensor | None,
        pooled_outputs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Forward pass using [REL] hub.

        Args:
            hidden_states: [batch, seq, hidden] (optional)
            pooled_outputs: Dict of hub representations (preferred)

        Returns:
            Logits [batch, num_labels]
        """
        if pooled_outputs is not None and self.hub_token in pooled_outputs:
            pooled = pooled_outputs[self.hub_token]
        elif hidden_states is not None:
            pooled = hidden_states[:, self.hub_position, :]
        else:
            raise ValueError("Either hidden_states or pooled_outputs must be provided")

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

    def extra_repr(self) -> str:
        return f"hub={self.hub_token}, labels={self.num_labels}"


# ======================================================================
# Head Registry & Factory Functions
# ======================================================================

# Head registry mapping capabilities to their head classes
HEAD_REGISTRY: dict[str, type | None] = {
    # EMO hub heads
    "emotions": HubAwareHierarchicalHead,
    "sentiment": HubAwareClassificationHead,
    "safety_generic": HubAwareSafetyHead,
    "safety_familyos": HubAwareSafetyHead,
    # MEM hub heads
    "embedding": None,  # No head - uses raw [MEM] output
    # REL hub heads
    "nli": HubAwareNLIHead,
    "relation": HubAwareClassificationHead,
    # TASK hub heads
    "intent": HubAwareClassificationHead,
    "ingress": HubAwareClassificationHead,
    # Token-level heads (no hub pooling)
    "ner_general": HubAwareTokenClassificationHead,
    "ner_family": HubAwareTokenClassificationHead,
    "temporal": HubAwareTokenClassificationHead,
}


def create_head_for_capability(
    capability: str,
    hidden_size: int = 768,
    num_labels: int | None = None,
    **kwargs,
) -> nn.Module | None:
    """
    Factory function to create appropriate head for a capability.

    Args:
        capability: Task/capability name (e.g., "emotions", "ner_general", "nli")
        hidden_size: Model hidden size (default: 768)
        num_labels: Number of output labels (task-dependent, auto-inferred if None)
        **kwargs: Additional head-specific arguments (dropout, confidence_threshold, etc.)

    Returns:
        Configured head module or None for embedding task

    Raises:
        ValueError: If capability is unknown

    Example:
        >>> head = create_head_for_capability("emotions", 768)
        >>> isinstance(head, HubAwareHierarchicalHead)
        True
        >>> head = create_head_for_capability("ner_general", 768, num_labels=9)
        >>> isinstance(head, HubAwareTokenClassificationHead)
        True
    """
    if capability not in HEAD_REGISTRY:
        raise ValueError(
            f"Unknown capability: {capability}. " f"Supported: {list(HEAD_REGISTRY.keys())}"
        )

    head_class = HEAD_REGISTRY[capability]

    if head_class is None:
        return None  # Embedding task - no head needed

    # Get hub token for this capability
    hub_token = get_hub_for_capability(capability)

    # Default label counts per capability
    default_labels = {
        "emotions": 7,  # Ekman primary
        "sentiment": 3,  # pos/neg/neu
        "safety_generic": 2,
        "safety_familyos": 2,
        "nli": 3,
        "relation": 10,  # Family relations
        "intent": 15,  # Intent types
        "ingress": 8,  # Ingress categories
        "ner_general": 9,  # BIO tags
        "ner_family": 9,
        "temporal": 5,
    }

    if num_labels is None:
        num_labels = default_labels.get(capability, 2)

    # Handle special cases
    if capability == "emotions":
        return HubAwareHierarchicalHead(
            hidden_size=hidden_size,
            primary_labels=7,
            secondary_labels=28,
            hub_token=hub_token,
            **kwargs,
        )

    if capability in TOKEN_LEVEL_CAPABILITIES:
        return HubAwareTokenClassificationHead(
            hidden_size=hidden_size,
            num_labels=num_labels,
            **kwargs,
        )

    # Standard classification head (sentiment, safety, nli, relation, intent, ingress)
    return head_class(
        hidden_size=hidden_size,
        num_labels=num_labels,
        hub_token=hub_token,
        **kwargs,
    )


def create_all_heads(
    hidden_size: int = 768,
    capabilities: list[str] | None = None,
) -> nn.ModuleDict:
    """
    Create heads for all (or specified) capabilities.

    Args:
        hidden_size: Model hidden size (default: 768)
        capabilities: List of capabilities to create heads for (default: all)

    Returns:
        ModuleDict of capability -> head

    Example:
        >>> heads = create_all_heads(768, ["emotions", "sentiment", "nli"])
        ✓ Created 3 task heads
        >>> list(heads.keys())
        ['emotions', 'sentiment', 'nli']
    """
    if capabilities is None:
        capabilities = list(HEAD_REGISTRY.keys())

    heads = nn.ModuleDict()

    for cap in capabilities:
        head = create_head_for_capability(cap, hidden_size)
        if head is not None:
            heads[cap] = head

    print(f"✓ Created {len(heads)} task heads")
    return heads
