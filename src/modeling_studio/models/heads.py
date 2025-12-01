"""
Task-Specific Classification Heads - Enhanced v2

This module contains the individual head implementations for the multi-task model.
Each head is designed to be modular and can be attached to any encoder backbone.

Heads Implemented:
    - BaseHead: Abstract base class with common functionality
    - SequenceClassificationHead: Text classification (sentiment, emotions, etc.)
    - TokenClassificationHead: Token-level classification (NER, Temporal)
    - EmbeddingHead: Dense vector representations
    - NLIHead: Natural language inference with pair encoding
    - SafetyHead: Safety classification with calibration
    - EnhancedSafetyHead: Advanced safety with keyword override & 12 subcategories (NEW)
    - RelationHead: Family relationship extraction (NEW)
    - IntentHead: User intent classification (NEW)

Design Principles:
    - Each head owns its loss computation
    - Heads can be frozen/unfrozen independently
    - Support for task-specific dropout rates
    - Calibration hooks for threshold tuning

Epic 5.0 Enhancements:
    - External pooler support for shared pooling (SequenceClassificationHead)
    - CrossAttentionPairEncoder integration (NLIHead, RelationHead)
    - Backward compatible - all new features are optional
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

# Type checking imports for Epic 5.0 components
if TYPE_CHECKING:
    pass

# =============================================================================
# Base Head
# =============================================================================


class BaseHead(ABC, nn.Module):
    """
    Abstract base class for all task heads.

    Provides common functionality for classification heads including:
    - Dropout regularization
    - Freeze/unfreeze methods
    - Loss computation interface
    - Class-weighted and focal loss for multi-label tasks

    Args:
        hidden_size: Size of encoder hidden states
        num_labels: Number of output labels/classes
        dropout: Dropout probability
        problem_type: Type of classification problem
        class_weights: Optional tensor of per-class weights for multi-label BCE
        use_focal_loss: Whether to use focal loss for multi-label (reduces easy negative dominance)
        focal_gamma: Focal loss gamma parameter (default 2.0)
        label_smoothing: Label smoothing factor (0.0 = no smoothing, 0.1 = typical). Default: 0.0
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int = 2,
        dropout: float = 0.1,
        problem_type: str = "single_label_classification",
        class_weights: torch.Tensor | None = None,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.dropout_prob = dropout
        self.problem_type = problem_type
        self.dropout = nn.Dropout(dropout)

        # Multi-label loss configuration
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the head.

        Args:
            hidden_states: Encoder output (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask (batch_size, seq_len)
            labels: Target labels (shape depends on task)

        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        pass

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss based on problem type.

        For multi_label_classification:
        - Supports class-weighted BCE (set self.class_weights)
        - Supports focal loss (set self.use_focal_loss=True)
        """
        if self.problem_type == "single_label_classification":
            loss = F.cross_entropy(
                logits.view(-1, self.num_labels), 
                labels.view(-1),
                label_smoothing=self.label_smoothing
            )
            # DEBUG: Check for abnormal loss
            if loss.item() > 10:
                print(f"[DEBUG] HIGH LOSS in {self.__class__.__name__}:")
                print(f"  problem_type: {self.problem_type}")
                print(f"  num_labels: {self.num_labels}")
                print(f"  logits shape: {logits.shape}, dtype: {logits.dtype}")
                print(f"  labels shape: {labels.shape}, dtype: {labels.dtype}")
                print(f"  labels min/max: {labels.min().item()}/{labels.max().item()}")
                print(f"  logits min/max: {logits.min().item():.4f}/{logits.max().item():.4f}")
                print(f"  loss: {loss.item():.4f}")
            return loss
        elif self.problem_type == "multi_label_classification":
            # Apply focal loss or weighted BCE for better handling of class imbalance
            if self.use_focal_loss:
                loss = self._focal_bce_loss(logits, labels.float())
            elif self.class_weights is not None:
                loss = F.binary_cross_entropy_with_logits(
                    logits, labels.float(), weight=self.class_weights
                )
            else:
                loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            # DEBUG: Check for abnormal loss
            if loss.item() > 10:
                print(f"[DEBUG] HIGH LOSS in {self.__class__.__name__}:")
                print(f"  problem_type: {self.problem_type}")
                print(f"  num_labels: {self.num_labels}")
                print(f"  logits shape: {logits.shape}, dtype: {logits.dtype}")
                print(f"  labels shape: {labels.shape}, dtype: {labels.dtype}")
                print(f"  labels min/max: {labels.min().item()}/{labels.max().item()}")
                print(f"  logits min/max: {logits.min().item():.4f}/{logits.max().item():.4f}")
                print(f"  loss: {loss.item():.4f}")
            return loss
        elif self.problem_type == "regression":
            return F.mse_loss(logits.squeeze(-1), labels)
        else:
            raise ValueError(f"Unknown problem type: {self.problem_type}")

    def _focal_bce_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Focal loss for multi-label classification.

        Focal loss down-weights easy examples (clear negatives) and focuses
        on hard examples, which helps with class imbalance in multi-label tasks.

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

        Args:
            logits: Raw model outputs (batch_size, num_labels)
            labels: Binary labels (batch_size, num_labels)

        Returns:
            Scalar focal loss
        """
        probs = torch.sigmoid(logits)
        # p_t is the probability of the correct class
        p_t = probs * labels + (1 - probs) * (1 - labels)
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t).pow(self.focal_gamma)

        # BCE loss per element
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")

        # Apply focal weight and class weights
        focal_loss = focal_weight * bce
        if self.class_weights is not None:
            focal_loss = focal_loss * self.class_weights

        return focal_loss.mean()

    def freeze(self) -> None:
        """Freeze all parameters in this head."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all parameters in this head."""
        for param in self.parameters():
            param.requires_grad = True


# =============================================================================
# Sequence Classification Head
# =============================================================================


class SequenceClassificationHead(BaseHead):
    """
    Head for sequence-level classification tasks.

    Supports:
        - Single-label classification (sentiment, safety bands)
        - Multi-label classification (emotions, toxicity)
        - Regression

    Architecture:
        hidden_states -> pooling -> dropout -> dense -> output

    Epic 5.0 Enhancement:
        - Accepts external_pooler for shared pooling across heads
        - Falls back to internal pooling if no external pooler provided

    Args:
        hidden_size: Size of encoder hidden states
        num_labels: Number of output classes
        dropout: Dropout probability
        problem_type: 'single_label_classification', 'multi_label_classification', or 'regression'
        pooling: Pooling strategy ('cls', 'mean', 'max') - used if no external_pooler
        class_weights: Optional per-class weights for multi-label BCE loss
        use_focal_loss: Whether to use focal loss for multi-label tasks
        focal_gamma: Focal loss gamma parameter (default 2.0)
        external_pooler: Epic 5.0 - External pooler module (CLSMeanPooler, AttentionPooler)
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int = 2,
        dropout: float = 0.1,
        problem_type: str = "single_label_classification",
        pooling: str = "cls",
        class_weights: torch.Tensor | None = None,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        external_pooler: nn.Module | None = None,
    ):
        super().__init__(
            hidden_size,
            num_labels,
            dropout,
            problem_type,
            class_weights=class_weights,
            use_focal_loss=use_focal_loss,
            focal_gamma=focal_gamma,
        )
        self.pooling = pooling

        # Epic 5.0: External pooler support
        self.external_pooler = external_pooler
        self._use_external_pooler = external_pooler is not None

        # Classification layers
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with small random values."""
        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Pool sequence representations to a single vector.

        Epic 5.0: Uses external pooler if provided, otherwise falls back to
        internal pooling strategy.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)

        Returns:
            Pooled representation (batch_size, hidden_size)
        """
        # Epic 5.0: Use external pooler if available
        if self._use_external_pooler and self.external_pooler is not None:
            return self.external_pooler(hidden_states, attention_mask)

        # Fallback to internal pooling
        if self.pooling == "cls":
            # Use [CLS] token (first token)
            return hidden_states[:, 0, :]

        elif self.pooling == "mean":
            # Mean pooling over non-padded tokens
            if attention_mask is None:
                return hidden_states.mean(dim=1)

            # Expand mask for broadcasting
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_hidden = (hidden_states * mask).sum(dim=1)
            sum_mask = mask.sum(dim=1).clamp(min=1e-9)
            return sum_hidden / sum_mask

        elif self.pooling == "max":
            # Max pooling
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                hidden_states = hidden_states.masked_fill(mask == 0, -1e9)
            return hidden_states.max(dim=1)[0]

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for sequence classification."""
        # Pool to single vector
        pooled = self.pool(hidden_states, attention_mask)

        # Classification
        x = self.dropout(pooled)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.classifier(x)

        output = {"logits": logits}

        # Compute loss if labels provided
        if labels is not None:
            loss = self.compute_loss(logits, labels)
            output["loss"] = loss

        return output


# =============================================================================
# Token Classification Head
# =============================================================================


class TokenClassificationHead(BaseHead):
    """
    Head for token-level classification tasks (NER, POS tagging).

    Architecture:
        hidden_states -> dropout -> classifier

    Args:
        hidden_size: Size of encoder hidden states
        num_labels: Number of output labels (e.g., 9 for BIO-NER)
        dropout: Dropout probability
        problem_type: Always 'token_classification'
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int = 9,
        dropout: float = 0.1,
        problem_type: str = "token_classification",
    ):
        super().__init__(hidden_size, num_labels, dropout, problem_type)

        # Token classifier
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for token classification.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
            labels: (batch_size, seq_len) with -100 for ignored tokens

        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        # Apply dropout and classify each token
        x = self.dropout(hidden_states)
        logits = self.classifier(x)  # (batch_size, seq_len, num_labels)

        output = {"logits": logits}

        # Compute loss if labels provided
        if labels is not None:
            # Flatten for loss computation
            # Use -100 as ignore index (standard in HuggingFace)
            loss = F.cross_entropy(
                logits.view(-1, self.num_labels),
                labels.view(-1),
                ignore_index=-100,
            )
            # DEBUG: Check for abnormal loss
            if loss.item() > 10:
                print("[DEBUG] HIGH LOSS in TokenClassificationHead:")
                print(f"  num_labels: {self.num_labels}")
                print(f"  logits shape: {logits.shape}, dtype: {logits.dtype}")
                print(f"  labels shape: {labels.shape}, dtype: {labels.dtype}")
                valid_labels = labels[labels != -100]
                print(
                    f"  valid labels min/max: {valid_labels.min().item()}/{valid_labels.max().item()}"
                )
                print(f"  valid labels count: {valid_labels.numel()}")
                print(f"  logits min/max: {logits.min().item():.4f}/{logits.max().item():.4f}")
                print(f"  loss: {loss.item():.4f}")
            output["loss"] = loss

        return output


# =============================================================================
# Embedding Head
# =============================================================================


class EmbeddingHead(nn.Module):
    """
    Head for generating sentence embeddings.

    Produces dense vector representations suitable for:
        - Semantic similarity
        - Retrieval/search
        - Clustering

    Architecture:
        hidden_states -> pooling -> [projection] -> [normalize]

    Args:
        hidden_size: Size of encoder hidden states
        output_dim: Output embedding dimension (None = same as hidden_size)
        pooling: Pooling strategy ('cls', 'mean', 'max')
        normalize: Whether to L2-normalize output embeddings
    """

    def __init__(
        self,
        hidden_size: int,
        output_dim: int | None = None,
        pooling: str = "mean",
        normalize: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_dim = output_dim or hidden_size
        self.pooling = pooling
        self.normalize = normalize

        # Optional projection layer
        if output_dim is not None and output_dim != hidden_size:
            self.projection = nn.Linear(hidden_size, output_dim)
        else:
            self.projection = None

    def pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool sequence to single vector."""
        if self.pooling == "cls":
            return hidden_states[:, 0, :]

        elif self.pooling == "mean":
            if attention_mask is None:
                return hidden_states.mean(dim=1)

            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden = (hidden_states * mask).sum(dim=1)
            sum_mask = mask.sum(dim=1).clamp(min=1e-9)
            return sum_hidden / sum_mask

        elif self.pooling == "max":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                hidden_states = hidden_states.masked_fill(mask == 0, -1e9)
            return hidden_states.max(dim=1)[0]

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Generate sentence embeddings.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)

        Returns:
            Embeddings (batch_size, output_dim)
        """
        # Pool to single vector
        embeddings = self.pool(hidden_states, attention_mask)

        # Optional projection
        if self.projection is not None:
            embeddings = self.projection(embeddings)

        # Optional L2 normalization
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def freeze(self) -> None:
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True


# =============================================================================
# NLI Head
# =============================================================================


class NLIHead(SequenceClassificationHead):
    """
    Head for Natural Language Inference.

    NLI involves classifying the relationship between premise-hypothesis pairs:
        - Entailment: hypothesis follows from premise
        - Neutral: hypothesis neither follows nor contradicts
        - Contradiction: hypothesis contradicts premise

    Epic 5.0 Enhancement:
        - Accepts external pair_encoder for cross-attention between premise/hypothesis
        - Falls back to standard sequence classification if no pair encoder provided

    Args:
        hidden_size: Size of encoder hidden states
        dropout: Dropout probability
        pooling: Pooling strategy
        external_pooler: Epic 5.0 - External pooler for shared pooling
        pair_encoder: Epic 5.0 - CrossAttentionPairEncoder for cross-attention
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int = 3,  # Always 3 for NLI
        dropout: float = 0.1,
        problem_type: str = "single_label_classification",
        pooling: str = "cls",
        external_pooler: nn.Module | None = None,
        pair_encoder: nn.Module | None = None,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_labels=num_labels,
            dropout=dropout,
            problem_type=problem_type,
            pooling=pooling,
            external_pooler=external_pooler,
        )

        # Epic 5.0: Pair encoder support
        self.pair_encoder = pair_encoder
        self._use_pair_encoder = pair_encoder is not None

        # If using pair encoder, may need different input size for classifier
        # Pair encoder outputs hidden_size, so no change needed

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        pair_encoder: nn.Module | None = None,
        # Epic 5.0: Optional pair inputs for cross-attention
        text_a_hidden: torch.Tensor | None = None,
        text_b_hidden: torch.Tensor | None = None,
        text_a_mask: torch.Tensor | None = None,
        text_b_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for NLI classification.

        Epic 5.0: Supports cross-attention pair encoding when pair_encoder is provided.

        Args:
            hidden_states: Encoder output (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask (batch_size, seq_len)
            labels: Target labels (batch_size,)
            pair_encoder: Optional pair encoder passed from model forward
            text_a_hidden: Premise hidden states (for pair encoding)
            text_b_hidden: Hypothesis hidden states (for pair encoding)
            text_a_mask: Premise attention mask
            text_b_mask: Hypothesis attention mask

        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        # Determine which pair encoder to use (passed or instance)
        active_pair_encoder = pair_encoder or self.pair_encoder

        # Epic 5.0: Use cross-attention pair encoding if available
        if (
            active_pair_encoder is not None
            and text_a_hidden is not None
            and text_b_hidden is not None
        ):
            # Use cross-attention pair encoder
            pair_repr = active_pair_encoder(
                text_a_hidden,
                text_b_hidden,
                text_a_mask,
                text_b_mask,
            )

            # Classification on pair representation
            x = self.dropout(pair_repr)
            x = self.dense(x)
            x = torch.tanh(x)
            x = self.dropout(x)
            logits = self.classifier(x)

            output = {"logits": logits}

            if labels is not None:
                loss = self.compute_loss(logits, labels)
                output["loss"] = loss

            return output

        # Fallback to standard sequence classification
        return super().forward(hidden_states, attention_mask, labels)


# =============================================================================
# Safety Head (Optional specialized head with calibration)
# =============================================================================


class SafetyHead(SequenceClassificationHead):
    """
    Specialized head for safety classification with calibration and subcategory support.

    Features:
        - Temperature scaling for confidence calibration
        - Threshold configuration for policy bands
        - Optional focal loss for class imbalance
        - Subcategory classification (12 subcategories) - Issue 3.6.8
        - Hierarchical loss (band → subcategory)

    Safety Bands (4):
        - GREEN (0): Safe, routine content
        - AMBER (1): Needs attention, mild concern
        - RED (2): Serious concern, escalate to K1
        - CRISIS (3): Immediate intervention needed

    Safety Subcategories (12):
        - GREEN: none (0)
        - AMBER: stress (1), mild_sadness (2), frustration (3), health_mention (4)
        - RED: persistent_sadness (5), isolation (6), hopelessness (7), substance (8)
        - CRISIS: self_harm_ideation (9), suicide_ideation (10), harm_to_others (11), abuse_disclosure (12)

    Args:
        hidden_size: Size of encoder hidden states
        num_bands: Number of safety bands (default 4: GREEN, AMBER, RED, CRISIS)
        num_subcategories: Number of subcategories (default 13: 0-12)
        dropout: Dropout probability
        temperature: Initial temperature for calibration (1.0 = no scaling)
        use_focal_loss: Whether to use focal loss
        focal_gamma: Focal loss gamma parameter
        use_hierarchical: Whether to use hierarchical classification (band → subcategory)
        band_loss_weight: Weight for band loss in combined loss (default 0.6)
    """

    # Band definitions
    BAND_NAMES = ["GREEN", "AMBER", "RED", "CRISIS"]
    BAND_TO_ID = {"GREEN": 0, "AMBER": 1, "RED": 2, "CRISIS": 3}
    ID_TO_BAND = {0: "GREEN", 1: "AMBER", 2: "RED", 3: "CRISIS"}

    # Subcategory definitions
    SUBCATEGORY_NAMES = [
        "none",  # 0 - GREEN
        "stress",  # 1 - AMBER
        "mild_sadness",  # 2 - AMBER
        "frustration",  # 3 - AMBER
        "health_mention",  # 4 - AMBER
        "persistent_sadness",  # 5 - RED
        "isolation",  # 6 - RED
        "hopelessness",  # 7 - RED
        "substance",  # 8 - RED
        "self_harm_ideation",  # 9 - CRISIS
        "suicide_ideation",  # 10 - CRISIS
        "harm_to_others",  # 11 - CRISIS
        "abuse_disclosure",  # 12 - CRISIS
    ]
    SUBCATEGORY_TO_ID = {name: i for i, name in enumerate(SUBCATEGORY_NAMES)}
    ID_TO_SUBCATEGORY = dict(enumerate(SUBCATEGORY_NAMES))

    # Subcategory to band mapping
    SUBCATEGORY_TO_BAND_ID = {
        0: 0,  # none -> GREEN
        1: 1,  # stress -> AMBER
        2: 1,  # mild_sadness -> AMBER
        3: 1,  # frustration -> AMBER
        4: 1,  # health_mention -> AMBER
        5: 2,  # persistent_sadness -> RED
        6: 2,  # isolation -> RED
        7: 2,  # hopelessness -> RED
        8: 2,  # substance -> RED
        9: 3,  # self_harm_ideation -> CRISIS
        10: 3,  # suicide_ideation -> CRISIS
        11: 3,  # harm_to_others -> CRISIS
        12: 3,  # abuse_disclosure -> CRISIS
    }

    # Band to valid subcategory IDs
    BAND_TO_SUBCATEGORY_IDS = {
        0: [0],  # GREEN -> none
        1: [1, 2, 3, 4],  # AMBER -> stress, mild_sadness, frustration, health_mention
        2: [5, 6, 7, 8],  # RED -> persistent_sadness, isolation, hopelessness, substance
        3: [
            9,
            10,
            11,
            12,
        ],  # CRISIS -> self_harm_ideation, suicide_ideation, harm_to_others, abuse_disclosure
    }

    def __init__(
        self,
        hidden_size: int,
        num_bands: int = 4,
        num_subcategories: int = 13,
        dropout: float = 0.1,
        problem_type: str = "single_label_classification",
        pooling: str = "cls",
        temperature: float = 1.0,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        use_hierarchical: bool = True,
        band_loss_weight: float = 0.6,
    ):
        # Initialize base with num_bands as num_labels for backwards compatibility
        super().__init__(
            hidden_size=hidden_size,
            num_labels=num_bands,
            dropout=dropout,
            problem_type=problem_type,
            pooling=pooling,
        )
        self.num_bands = num_bands
        self.num_subcategories = num_subcategories
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.use_hierarchical = use_hierarchical
        self.band_loss_weight = band_loss_weight

        # Subcategory classifier
        self.subcategory_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_subcategories),
        )

        # Band to subcategory mapping as buffer for efficient masking
        self._register_band_subcategory_buffers()

    def _register_band_subcategory_buffers(self) -> None:
        """Register buffers for hierarchical masking."""
        # Create mask matrix: [num_bands, num_subcategories]
        mask = torch.zeros(self.num_bands, self.num_subcategories, dtype=torch.bool)
        for band_id, subcat_ids in self.BAND_TO_SUBCATEGORY_IDS.items():
            for subcat_id in subcat_ids:
                mask[band_id, subcat_id] = True
        self.register_buffer("band_subcat_mask", mask)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        subcategory_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass with temperature scaling and subcategory prediction.

        Args:
            hidden_states: Encoder output [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Band labels [batch_size] (0-3)
            subcategory_labels: Subcategory labels [batch_size] (0-12)

        Returns:
            Dictionary with:
                - logits: Band logits [batch_size, num_bands]
                - band: Predicted band name(s)
                - band_confidence: Band prediction confidence
                - subcategory_logits: Subcategory logits [batch_size, num_subcategories]
                - subcategory: Predicted subcategory name(s)
                - subcategory_confidence: Subcategory prediction confidence
                - loss: Combined loss (if labels provided)
        """
        batch_size = hidden_states.size(0)

        # Get base class output for band prediction
        base_output = super().forward(hidden_states, attention_mask, labels=None)

        # Apply temperature scaling to band logits
        band_logits = base_output["logits"] / self.temperature

        # Pool for subcategory classification
        if self.pooling == "cls":
            pooled = hidden_states[:, 0]
        elif self.pooling == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = hidden_states.mean(dim=1)
        else:
            pooled = hidden_states[:, 0]

        # Subcategory logits
        subcategory_logits = self.subcategory_classifier(self.dropout(pooled))
        subcategory_logits = subcategory_logits / self.temperature

        # Apply hierarchical masking if enabled
        if self.use_hierarchical:
            predicted_bands = band_logits.argmax(dim=-1)  # [batch_size]
            subcategory_logits = self._apply_hierarchical_mask(subcategory_logits, predicted_bands)

        # Compute probabilities and predictions
        band_probs = F.softmax(band_logits, dim=-1)
        subcategory_probs = F.softmax(subcategory_logits, dim=-1)

        predicted_band_ids = band_probs.argmax(dim=-1)
        predicted_subcat_ids = subcategory_probs.argmax(dim=-1)

        band_confidence = band_probs.max(dim=-1).values
        subcategory_confidence = subcategory_probs.max(dim=-1).values

        # Convert to names (use int() for type safety)
        band_names = [self.ID_TO_BAND[int(idx.item())] for idx in predicted_band_ids]
        subcategory_names = [
            self.ID_TO_SUBCATEGORY[int(idx.item())] for idx in predicted_subcat_ids
        ]

        output = {
            "logits": band_logits,
            "band_logits": band_logits,
            "band": band_names[0] if batch_size == 1 else band_names,
            "band_confidence": band_confidence[0].item() if batch_size == 1 else band_confidence,
            "subcategory_logits": subcategory_logits,
            "subcategory": subcategory_names[0] if batch_size == 1 else subcategory_names,
            "subcategory_confidence": (
                subcategory_confidence[0].item() if batch_size == 1 else subcategory_confidence
            ),
            "band_probs": band_probs,
            "subcategory_probs": subcategory_probs,
        }

        # Compute loss if labels provided
        if labels is not None:
            if self.use_focal_loss:
                band_loss = self._focal_loss(band_logits, labels)
            else:
                band_loss = F.cross_entropy(band_logits, labels)

            output["band_loss"] = band_loss

            if subcategory_labels is not None:
                subcat_loss = F.cross_entropy(subcategory_logits, subcategory_labels)
                output["subcategory_loss"] = subcat_loss
                # Hierarchical loss: band is primary, subcategory is secondary
                output["loss"] = (
                    self.band_loss_weight * band_loss + (1 - self.band_loss_weight) * subcat_loss
                )
            else:
                output["loss"] = band_loss

        return output

    def _apply_hierarchical_mask(
        self,
        subcategory_logits: torch.Tensor,
        predicted_bands: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply hierarchical masking to subcategory logits.

        Masks out subcategories that are not valid for the predicted band.

        Args:
            subcategory_logits: [batch_size, num_subcategories]
            predicted_bands: [batch_size] predicted band indices

        Returns:
            Masked subcategory logits with invalid subcategories set to -inf
        """
        # Get mask for each sample based on predicted band
        mask = self.band_subcat_mask[predicted_bands]  # [batch_size, num_subcategories]

        # Apply mask: set invalid subcategories to -inf
        masked_logits = subcategory_logits.masked_fill(~mask, float("-inf"))

        return masked_logits

    def _focal_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss for handling class imbalance.

        Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
        """
        ce_loss = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.focal_gamma) * ce_loss
        return focal_loss.mean()

    def set_temperature(self, temperature: float) -> None:
        """Set temperature for calibration."""
        self.temperature.data = torch.tensor(temperature)

    def calibrate(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        """
        Learn temperature parameter on validation set.

        Args:
            val_logits: Logits from validation set
            val_labels: Labels from validation set
            lr: Learning rate for optimization
            max_iter: Maximum iterations

        Returns:
            Optimal temperature value
        """
        # Only optimize temperature
        self.temperature.requires_grad = True
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(val_logits / self.temperature, val_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature.requires_grad = False

        return self.temperature.item()


# =============================================================================
# Enhanced Safety Head (v2 - Issue 3.5.8)
# =============================================================================


class EnhancedSafetyHead(nn.Module):
    """
    Enhanced safety classification head with keyword override detection,
    hierarchical classification, and 12 safety subcategories.

    This head provides robust safety classification with multiple layers:
        1. Keyword Override: CRISIS keywords always trigger highest severity
        2. Hierarchical Classification: Band → Subcategory cascade
        3. Learnable Temperature: Confidence calibration
        4. 12 Subcategories: Fine-grained safety categorization

    Safety Bands (4 levels):
        - GREEN (0): Safe content, no concerns
        - AMBER (1): Caution needed, potential concerns
        - RED (2): Harmful content detected
        - CRISIS (3): Immediate intervention required (self-harm, violence)

    Subcategories (12 types):
        GREEN:
            - general_safe: Normal safe content
            - positive_interaction: Encouraging/supportive content

        AMBER:
            - mild_profanity: Light swearing, frustration
            - sensitive_topic: Discussion of sensitive subjects
            - boundary_test: Testing limits/boundaries
            - emotional_distress: Expressing negative emotions

        RED:
            - harassment: Bullying, targeting individuals
            - explicit_content: Adult/explicit material
            - misinformation: False/harmful information
            - hate_speech: Discriminatory content

        CRISIS:
            - self_harm: Self-harm ideation/threats
            - violence_threat: Threats of violence to others

    Keyword Override:
        Certain keywords/phrases always trigger CRISIS regardless of model prediction:
        - "I want to kill myself"
        - "kill myself"
        - "end my life"
        - "suicide"
        - "I want to die"
        etc.

    Args:
        hidden_size: Size of encoder hidden states (default 768)
        num_bands: Number of safety bands (default 4)
        num_subcategories: Number of subcategories (default 12)
        dropout: Dropout probability
        initial_temperature: Initial temperature for calibration
        use_hierarchical: Whether to use hierarchical classification
        keyword_override: Whether to enable keyword override detection

    Example:
        >>> head = EnhancedSafetyHead(hidden_size=768)
        >>> hidden_states = torch.randn(1, 32, 768)
        >>> attention_mask = torch.ones(1, 32)
        >>> output = head(hidden_states, attention_mask, text="I want to kill myself")
        >>> assert output["band"] == "CRISIS"  # Keyword override
        >>> assert "subcategory" in output
    """

    # Safety band definitions
    BAND_NAMES = ["GREEN", "AMBER", "RED", "CRISIS"]
    BAND_TO_ID = {"GREEN": 0, "AMBER": 1, "RED": 2, "CRISIS": 3}
    ID_TO_BAND = {0: "GREEN", 1: "AMBER", 2: "RED", 3: "CRISIS"}

    # Subcategory definitions with parent band mapping
    SUBCATEGORIES = [
        # GREEN (0-1)
        "general_safe",  # 0
        "positive_interaction",  # 1
        # AMBER (2-5)
        "mild_profanity",  # 2
        "sensitive_topic",  # 3
        "boundary_test",  # 4
        "emotional_distress",  # 5
        # RED (6-9)
        "harassment",  # 6
        "explicit_content",  # 7
        "misinformation",  # 8
        "hate_speech",  # 9
        # CRISIS (10-11)
        "self_harm",  # 10
        "violence_threat",  # 11
    ]

    SUBCATEGORY_TO_ID = {name: i for i, name in enumerate(SUBCATEGORIES)}
    ID_TO_SUBCATEGORY = {i: name for i, name in enumerate(SUBCATEGORIES)}

    # Mapping from subcategory to parent band
    SUBCATEGORY_TO_BAND = {
        "general_safe": "GREEN",
        "positive_interaction": "GREEN",
        "mild_profanity": "AMBER",
        "sensitive_topic": "AMBER",
        "boundary_test": "AMBER",
        "emotional_distress": "AMBER",
        "harassment": "RED",
        "explicit_content": "RED",
        "misinformation": "RED",
        "hate_speech": "RED",
        "self_harm": "CRISIS",
        "violence_threat": "CRISIS",
    }

    # Band to valid subcategory indices
    BAND_TO_SUBCATEGORY_IDS = {
        "GREEN": [0, 1],
        "AMBER": [2, 3, 4, 5],
        "RED": [6, 7, 8, 9],
        "CRISIS": [10, 11],
    }

    # CRISIS override keywords/phrases (case-insensitive)
    CRISIS_KEYWORDS = [
        # Self-harm related
        "i want to kill myself",
        "kill myself",
        "end my life",
        "end it all",
        "suicide",
        "i want to die",
        "want to die",
        "better off dead",
        "don't want to live",
        "don't want to be alive",
        "take my own life",
        "harm myself",
        "hurt myself",
        "cut myself",
        "self harm",
        "self-harm",
        # Violence to others
        "kill you",
        "kill them",
        "kill him",
        "kill her",
        "murder",
        "shoot up",
        "bomb threat",
        "i will hurt",
        "going to hurt",
    ]

    def __init__(
        self,
        hidden_size: int = 768,
        num_bands: int = 4,
        num_subcategories: int = 12,
        dropout: float = 0.1,
        initial_temperature: float = 1.0,
        use_hierarchical: bool = True,
        keyword_override: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_bands = num_bands
        self.num_subcategories = num_subcategories
        self.use_hierarchical = use_hierarchical
        self.keyword_override = keyword_override

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Pooling (CLS token)
        # Band classifier
        self.band_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_bands),
        )

        # Subcategory classifier
        self.subcategory_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_subcategories),
        )

        # Hierarchical attention (band-conditioned subcategory)
        if use_hierarchical:
            self.band_to_subcat_attention = nn.ModuleList(
                [
                    nn.Linear(hidden_size // 2, len(self.BAND_TO_SUBCATEGORY_IDS[band]))
                    for band in self.BAND_NAMES
                ]
            )

        # Learnable temperature for calibration
        self.log_temperature = nn.Parameter(torch.tensor(initial_temperature).log())

        # Confidence threshold for each band
        self.register_buffer(
            "band_thresholds",
            torch.tensor([0.5, 0.3, 0.2, 0.1]),  # Lower threshold for higher severity
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        subcategory_labels: torch.Tensor | None = None,
        text: str | list[str] | None = None,
    ) -> dict[str, torch.Tensor | str | list[str]]:
        """
        Forward pass with optional keyword override.

        Args:
            hidden_states: Encoder output [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Band labels [batch_size] (0-3)
            subcategory_labels: Subcategory labels [batch_size] (0-11)
            text: Optional input text for keyword override detection

        Returns:
            Dictionary with:
                - band_logits: Raw logits for bands [batch_size, 4]
                - subcategory_logits: Raw logits for subcategories [batch_size, 12]
                - band_probs: Probabilities for bands [batch_size, 4]
                - subcategory_probs: Probabilities for subcategories [batch_size, 12]
                - band: Predicted band name(s)
                - subcategory: Predicted subcategory name(s)
                - keyword_override: Boolean mask for keyword overrides
                - loss: Combined loss (if labels provided)
        """
        batch_size = hidden_states.size(0)
        device = hidden_states.device

        # Pool to [CLS] token representation
        pooled = hidden_states[:, 0]  # [batch_size, hidden_size]
        pooled = self.dropout(pooled)

        # Get temperature
        temperature = self.log_temperature.exp()

        # Band classification
        band_logits = self.band_classifier(pooled) / temperature  # [batch_size, 4]
        band_probs = F.softmax(band_logits, dim=-1)

        # Subcategory classification
        subcategory_logits = self.subcategory_classifier(pooled) / temperature  # [batch_size, 12]
        subcategory_probs = F.softmax(subcategory_logits, dim=-1)

        # Hierarchical masking: mask invalid subcategories based on predicted band
        if self.use_hierarchical:
            predicted_bands = band_logits.argmax(dim=-1)  # [batch_size]
            subcategory_mask = self._create_subcategory_mask(predicted_bands, device)
            # Apply mask (set invalid subcategories to -inf before softmax)
            masked_subcat_logits = subcategory_logits.masked_fill(~subcategory_mask, float("-inf"))
            subcategory_probs = F.softmax(masked_subcat_logits, dim=-1)

        # Get predictions
        predicted_band_ids = band_probs.argmax(dim=-1)  # [batch_size]
        predicted_subcat_ids = subcategory_probs.argmax(dim=-1)  # [batch_size]

        # Keyword override detection
        keyword_override_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if self.keyword_override and text is not None:
            keyword_override_mask = self._detect_crisis_keywords(text, device)

            # Override predictions for flagged samples
            if keyword_override_mask.any():
                predicted_band_ids = predicted_band_ids.clone()
                predicted_subcat_ids = predicted_subcat_ids.clone()

                predicted_band_ids[keyword_override_mask] = self.BAND_TO_ID["CRISIS"]
                # Default to self_harm subcategory for keyword override
                predicted_subcat_ids[keyword_override_mask] = self.SUBCATEGORY_TO_ID["self_harm"]

        # Convert to names
        band_names = [self.ID_TO_BAND[idx.item()] for idx in predicted_band_ids]
        subcategory_names = [self.ID_TO_SUBCATEGORY[idx.item()] for idx in predicted_subcat_ids]

        output = {
            "band_logits": band_logits,
            "subcategory_logits": subcategory_logits,
            "band_probs": band_probs,
            "subcategory_probs": subcategory_probs,
            "band_ids": predicted_band_ids,
            "subcategory_ids": predicted_subcat_ids,
            "band": band_names[0] if batch_size == 1 else band_names,
            "subcategory": subcategory_names[0] if batch_size == 1 else subcategory_names,
            "keyword_override": keyword_override_mask,
            "temperature": temperature.item(),
        }

        # Compute loss if labels provided
        if labels is not None:
            band_loss = F.cross_entropy(band_logits, labels)
            output["band_loss"] = band_loss

            if subcategory_labels is not None:
                subcat_loss = F.cross_entropy(subcategory_logits, subcategory_labels)
                output["subcategory_loss"] = subcat_loss
                # Combined loss with weighting (band is more important)
                output["loss"] = 0.6 * band_loss + 0.4 * subcat_loss
            else:
                output["loss"] = band_loss

        return output

    def _create_subcategory_mask(
        self,
        band_ids: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create mask for valid subcategories given predicted bands.

        Args:
            band_ids: Predicted band indices [batch_size]
            device: Target device

        Returns:
            Boolean mask [batch_size, num_subcategories]
        """
        batch_size = band_ids.size(0)
        mask = torch.zeros(batch_size, self.num_subcategories, dtype=torch.bool, device=device)

        for i, band_id in enumerate(band_ids):
            band_name = self.ID_TO_BAND[band_id.item()]
            valid_indices = self.BAND_TO_SUBCATEGORY_IDS[band_name]
            mask[i, valid_indices] = True

        return mask

    def _detect_crisis_keywords(
        self,
        text: str | list[str],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Detect CRISIS keywords in input text.

        Args:
            text: Input text or list of texts
            device: Target device

        Returns:
            Boolean mask indicating which samples contain crisis keywords
        """
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        batch_size = len(texts)
        mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for i, t in enumerate(texts):
            t_lower = t.lower()
            for keyword in self.CRISIS_KEYWORDS:
                if keyword in t_lower:
                    mask[i] = True
                    break

        return mask

    def set_temperature(self, temperature: float) -> None:
        """Set temperature for calibration."""
        self.log_temperature.data = torch.tensor(temperature).log()

    def calibrate(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        """
        Learn temperature parameter on validation set.

        Args:
            val_logits: Band logits from validation set
            val_labels: Band labels from validation set
            lr: Learning rate for optimization
            max_iter: Maximum iterations

        Returns:
            Optimal temperature value
        """
        self.log_temperature.requires_grad = True
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            temp = self.log_temperature.exp()
            loss = F.cross_entropy(val_logits / temp, val_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.log_temperature.requires_grad = False

        return self.log_temperature.exp().item()

    def get_severity_score(
        self,
        band_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute a single severity score from band probabilities.

        Score is weighted average where higher bands have higher weights.
        Range: [0, 1] where 1 is most severe.

        Args:
            band_probs: Band probabilities [batch_size, 4]

        Returns:
            Severity scores [batch_size]
        """
        weights = torch.tensor([0.0, 0.33, 0.66, 1.0], device=band_probs.device)
        return (band_probs * weights).sum(dim=-1)

    def freeze(self) -> None:
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def add_crisis_keyword(self, keyword: str) -> None:
        """Add a new crisis keyword for override detection."""
        keyword_lower = keyword.lower()
        if keyword_lower not in self.CRISIS_KEYWORDS:
            self.CRISIS_KEYWORDS.append(keyword_lower)

    def remove_crisis_keyword(self, keyword: str) -> bool:
        """Remove a crisis keyword. Returns True if removed."""
        keyword_lower = keyword.lower()
        if keyword_lower in self.CRISIS_KEYWORDS:
            self.CRISIS_KEYWORDS.remove(keyword_lower)
            return True
        return False


# =============================================================================
# Relation Extraction Head (NEW - v2)
# =============================================================================


class RelationHead(BaseHead):
    """
    Head for family relationship extraction between entity pairs.

    Given two entity spans, classifies the relationship between them.
    Supports 15 relation types including family relations (parent_of, child_of,
    spouse_of, etc.) and non-family relations (friend_of, colleague_of, etc.).

    Architecture:
        entity1_repr + entity2_repr -> concat -> dense -> dropout -> classifier

    Epic 5.0 Enhancement:
        - Accepts external pair_encoder for cross-attention between entity contexts
        - Falls back to standard entity concatenation if no pair encoder provided

    Args:
        hidden_size: Size of encoder hidden states
        num_labels: Number of relation types (15 for FamilyOS)
        dropout: Dropout probability
        pair_encoder: Epic 5.0 - CrossAttentionPairEncoder for entity context attention
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int = 15,
        dropout: float = 0.1,
        problem_type: str = "single_label_classification",
        pair_encoder: nn.Module | None = None,
    ):
        super().__init__(hidden_size, num_labels, dropout, problem_type)

        # Epic 5.0: Pair encoder support
        self.pair_encoder = pair_encoder
        self._use_pair_encoder = pair_encoder is not None

        # Entity pair representation
        # Input: concat of two entity representations (2 * hidden_size)
        # Or pair encoder output (hidden_size)
        self.entity_pair_dense = nn.Linear(hidden_size * 2, hidden_size)

        # Epic 5.0: Alternative dense layer for pair encoder output
        if self._use_pair_encoder:
            self.pair_encoded_dense = nn.Linear(hidden_size, hidden_size)
        else:
            self.pair_encoded_dense = None

        # Relation classifier
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.xavier_uniform_(self.entity_pair_dense.weight)
        nn.init.zeros_(self.entity_pair_dense.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        if self.pair_encoded_dense is not None:
            nn.init.xavier_uniform_(self.pair_encoded_dense.weight)
            nn.init.zeros_(self.pair_encoded_dense.bias)

    def get_entity_repr(
        self,
        hidden_states: torch.Tensor,
        entity_start: torch.Tensor,
        entity_end: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get representation for an entity span.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            entity_start: Start indices (batch_size,)
            entity_end: End indices (batch_size,)

        Returns:
            Entity representation (batch_size, hidden_size)
        """
        batch_size = hidden_states.size(0)
        device = hidden_states.device

        # Get start token representation
        batch_indices = torch.arange(batch_size, device=device)
        start_repr = hidden_states[batch_indices, entity_start]

        # Option: Could also use mean of span or [start; end] concat
        # For simplicity, using start token representation
        return start_repr

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        entity1_start: torch.Tensor | None = None,
        entity1_end: torch.Tensor | None = None,
        entity2_start: torch.Tensor | None = None,
        entity2_end: torch.Tensor | None = None,
        pair_encoder: nn.Module | None = None,
        # Epic 5.0: Optional entity context for cross-attention
        entity1_context: torch.Tensor | None = None,
        entity2_context: torch.Tensor | None = None,
        entity1_mask: torch.Tensor | None = None,
        entity2_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for relation extraction.

        Epic 5.0: Supports cross-attention pair encoding when pair_encoder is provided.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
            labels: Relation labels (batch_size,)
            entity1_start/end: Entity 1 span indices
            entity2_start/end: Entity 2 span indices
            pair_encoder: Optional pair encoder passed from model forward
            entity1_context: Entity 1 context hidden states (for pair encoding)
            entity2_context: Entity 2 context hidden states (for pair encoding)
            entity1_mask: Entity 1 context attention mask
            entity2_mask: Entity 2 context attention mask

        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        # Determine which pair encoder to use (passed or instance)
        active_pair_encoder = pair_encoder or self.pair_encoder

        # Epic 5.0: Use cross-attention pair encoding if available with entity contexts
        if (
            active_pair_encoder is not None
            and entity1_context is not None
            and entity2_context is not None
        ):
            # Use cross-attention pair encoder for entity contexts
            pair_repr = active_pair_encoder(
                entity1_context,
                entity2_context,
                entity1_mask,
                entity2_mask,
            )

            # Classification on pair representation
            x = self.dropout(pair_repr)
            if self.pair_encoded_dense is not None:
                x = self.pair_encoded_dense(x)
            else:
                # Fallback: project to expected size
                x = self.entity_pair_dense(torch.cat([x, x], dim=-1))
            x = torch.relu(x)
            x = self.dropout(x)
            logits = self.classifier(x)

            output = {"logits": logits}

            if labels is not None:
                loss = self.compute_loss(logits, labels)
                output["loss"] = loss

            return output

        # Standard entity-based relation extraction
        if entity1_start is None or entity2_start is None:
            # If no entity spans provided, use CLS for both (fallback)
            entity1_repr = hidden_states[:, 0, :]
            entity2_repr = hidden_states[:, 0, :]
        else:
            # Get entity representations
            entity1_repr = self.get_entity_repr(
                hidden_states,
                entity1_start,
                entity1_end if entity1_end is not None else entity1_start,
            )
            entity2_repr = self.get_entity_repr(
                hidden_states,
                entity2_start,
                entity2_end if entity2_end is not None else entity2_start,
            )

        # Concatenate entity representations
        pair_repr = torch.cat([entity1_repr, entity2_repr], dim=-1)

        # Classification
        x = self.dropout(pair_repr)
        x = self.entity_pair_dense(x)
        x = torch.relu(x)
        x = self.dropout(x)
        logits = self.classifier(x)

        output = {"logits": logits}

        if labels is not None:
            loss = self.compute_loss(logits, labels)
            output["loss"] = loss

        return output


# =============================================================================
# Intent Classification Head (NEW - v2)
# =============================================================================


class IntentHead(SequenceClassificationHead):
    """
    Head for user intent classification in FamilyOS interactions.

    Classifies user messages into 8 intent categories:
        - log_memory: Store a memory
        - query_memory: Retrieve past information
        - set_reminder: Create a reminder
        - express_feeling: Share emotions
        - seek_advice: Ask for guidance
        - share_news: Report something new
        - reflect: Contemplate/reminisce
        - other: Catch-all

    Uses sequence classification with confidence thresholds for routing.

    Args:
        hidden_size: Size of encoder hidden states
        num_labels: Number of intent types (8 for FamilyOS)
        dropout: Dropout probability
        confidence_threshold: Minimum confidence for intent (below = 'other')
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int = 8,
        dropout: float = 0.1,
        problem_type: str = "single_label_classification",
        pooling: str = "cls",
        confidence_threshold: float = 0.5,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_labels=num_labels,
            dropout=dropout,
            problem_type=problem_type,
            pooling=pooling,
        )
        self.confidence_threshold = confidence_threshold

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with confidence scoring."""
        output = super().forward(hidden_states, attention_mask, labels)

        # Add confidence scores
        probs = F.softmax(output["logits"], dim=-1)
        confidence, predicted = probs.max(dim=-1)

        output["confidence"] = confidence
        output["predicted_intent"] = predicted

        # Flag low-confidence predictions
        output["low_confidence_mask"] = confidence < self.confidence_threshold

        return output

    def set_confidence_threshold(self, threshold: float) -> None:
        """Update confidence threshold."""
        self.confidence_threshold = threshold


# =============================================================================
# Temporal Expression Head (NEW - v2)
# =============================================================================


class TemporalHead(TokenClassificationHead):
    """
    Head for temporal expression extraction.

    Extracts and classifies temporal expressions for timeline construction:
        - DATE_ABS: Absolute dates (January 15, 2024)
        - DATE_REL: Relative dates (yesterday, last week)
        - TIME: Times (3pm, morning)
        - DURATION: Durations (for 2 hours, all day)
        - FREQUENCY: Frequencies (every Sunday, weekly)
        - AGE: Ages/periods (when she was 5, in my 20s)

    Uses BIO tagging scheme with 13 labels.

    Args:
        hidden_size: Size of encoder hidden states
        num_labels: Number of temporal labels (13 BIO tags)
        dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int = 13,
        dropout: float = 0.1,
        problem_type: str = "token_classification",
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_labels=num_labels,
            dropout=dropout,
            problem_type=problem_type,
        )

        # Optional: Add CRF layer for better sequence modeling
        # self.crf = CRF(num_labels)

    def extract_temporal_spans(
        self,
        logits: torch.Tensor,
        attention_mask: torch.Tensor,
        id2label: dict[int, str] | None = None,
    ) -> list[list[dict]]:
        """
        Extract temporal spans from predictions.

        Args:
            logits: (batch_size, seq_len, num_labels)
            attention_mask: (batch_size, seq_len)
            id2label: Mapping from label IDs to names

        Returns:
            List of temporal spans per batch item
        """
        predictions = logits.argmax(dim=-1)
        batch_spans = []

        for batch_idx in range(predictions.size(0)):
            spans = []
            current_span = None

            for token_idx in range(predictions.size(1)):
                if attention_mask[batch_idx, token_idx] == 0:
                    continue

                label_id = predictions[batch_idx, token_idx].item()
                label = id2label[label_id] if id2label else str(label_id)

                if label.startswith("B-"):
                    if current_span is not None:
                        spans.append(current_span)
                    current_span = {
                        "type": label[2:],
                        "start": token_idx,
                        "end": token_idx,
                    }
                elif label.startswith("I-") and current_span is not None:
                    if label[2:] == current_span["type"]:
                        current_span["end"] = token_idx
                    else:
                        spans.append(current_span)
                        current_span = None
                else:
                    if current_span is not None:
                        spans.append(current_span)
                        current_span = None

            if current_span is not None:
                spans.append(current_span)

            batch_spans.append(spans)

        return batch_spans


# =============================================================================
# Hierarchical Emotion Head (v2 - Issue 3.6.6)
# =============================================================================


class HierarchicalEmotionHead(nn.Module):
    """
    Hierarchical emotion classification head with primary/secondary emotions
    and intensity scoring.

    This head provides multi-level emotion analysis:
        1. Primary Emotion: Single strongest emotion
        2. Secondary Emotions: Top-k additional emotions (default k=3)
        3. Emotion Intensity: 0-1 score for each emotion
        4. Valence/Arousal: Optional dimensional emotion representation

    Emotion Categories (32 fine-grained by default):
        - Joy: happiness, amusement, excitement, pride, satisfaction
        - Sadness: grief, disappointment, loneliness, regret
        - Anger: frustration, irritation, rage, resentment
        - Fear: anxiety, worry, terror, nervousness
        - Surprise: amazement, astonishment, confusion
        - Disgust: contempt, revulsion, disapproval
        - Trust: admiration, acceptance, gratitude
        - Anticipation: interest, optimism, vigilance

    Architecture:
        hidden_states -> pooling -> shared_layer -> emotion_classifier
                                                 -> intensity_regressor
                                                 -> valence_arousal (optional)

    Args:
        hidden_size: Size of encoder hidden states (default 768)
        num_emotions: Number of emotion categories (default 44)
        num_secondary: Max secondary emotions to return (default 3)
        dropout: Dropout probability (default 0.1)
        pooling: Pooling strategy ('cls', 'mean', 'max')
        use_intensity: Enable intensity scoring (default True)
        use_valence_arousal: Enable valence/arousal output (default False)
        intensity_threshold: Min intensity to include emotion (default 0.1)

    Example:
        >>> head = HierarchicalEmotionHead(hidden_size=768, num_emotions=44, use_familyos=True)
        >>> output = head(hidden_states, attention_mask)
        >>> print(output["primary_emotion"])  # "joy"
        >>> print(output["secondary_emotions"])  # ["love", "pride"]
        >>> print(output["emotion_scores"]["joy"])  # 0.85
    """

    # FamilyOS emotion labels (44 emotions - matches data/familyos/emotions schema)
    FAMILYOS_EMOTION_LABELS = [
        # Core Emotions (8) - IDs 0-7
        "neutral",
        "joy",
        "sadness",
        "anger",
        "fear",
        "surprise",
        "love",
        "disgust",
        # Positive Emotions (12) - IDs 8-19
        "admiration",
        "amusement",
        "approval",
        "caring",
        "excitement",
        "gratitude",
        "optimism",
        "pride",
        "relief",
        "contentment",
        "hope",
        "tenderness",
        # Negative Emotions (10) - IDs 20-29
        "annoyance",
        "disappointment",
        "disapproval",
        "embarrassment",
        "grief",
        "nervousness",
        "remorse",
        "frustration",
        "overwhelmed",
        "emptiness",
        # Family-Specific Emotions (14) - IDs 30-43
        "nostalgia",
        "protectiveness",
        "togetherness",
        "longing",
        "warmth",
        "playfulness",
        "celebration",
        "belonging",
        "parental_pride",
        "parental_guilt",
        "patience",
        "worry",
        "bittersweet",
        "homesickness",
    ]

    # Legacy 32-emotion labels (for backward compatibility)
    DEFAULT_EMOTION_LABELS = [
        # Joy family (0-4)
        "joy",
        "happiness",
        "amusement",
        "excitement",
        "pride",
        # Sadness family (5-9)
        "sadness",
        "grief",
        "disappointment",
        "loneliness",
        "regret",
        # Anger family (10-14)
        "anger",
        "frustration",
        "irritation",
        "rage",
        "resentment",
        # Fear family (15-19)
        "fear",
        "anxiety",
        "worry",
        "terror",
        "nervousness",
        # Surprise family (20-22)
        "surprise",
        "amazement",
        "confusion",
        # Disgust family (23-25)
        "disgust",
        "contempt",
        "disapproval",
        # Trust family (26-28)
        "trust",
        "admiration",
        "gratitude",
        # Anticipation family (29-31)
        "anticipation",
        "interest",
        "optimism",
    ]

    # FamilyOS emotion family groupings (for 44-emotion schema)
    FAMILYOS_EMOTION_FAMILIES = {
        # Core emotions as their own families
        "joy": ["joy", "amusement", "excitement", "contentment", "playfulness", "celebration"],
        "sadness": ["sadness", "grief", "disappointment", "longing", "emptiness", "homesickness"],
        "anger": ["anger", "annoyance", "frustration", "disapproval"],
        "fear": ["fear", "nervousness", "worry", "overwhelmed"],
        "surprise": ["surprise"],
        "love": ["love", "caring", "tenderness", "warmth", "togetherness", "belonging"],
        "disgust": ["disgust"],
        # Positive emotion families
        "pride": ["pride", "admiration", "approval", "parental_pride"],
        "gratitude": ["gratitude", "relief", "hope", "optimism"],
        # Negative emotion families
        "guilt": ["remorse", "embarrassment", "parental_guilt"],
        # Family-specific families
        "nostalgia": ["nostalgia", "bittersweet"],
        "protection": ["protectiveness", "patience"],
        "neutral": ["neutral"],
    }

    # Legacy emotion family groupings (for 32-emotion schema)
    EMOTION_FAMILIES = {
        "joy": ["joy", "happiness", "amusement", "excitement", "pride"],
        "sadness": ["sadness", "grief", "disappointment", "loneliness", "regret"],
        "anger": ["anger", "frustration", "irritation", "rage", "resentment"],
        "fear": ["fear", "anxiety", "worry", "terror", "nervousness"],
        "surprise": ["surprise", "amazement", "confusion"],
        "disgust": ["disgust", "contempt", "disapproval"],
        "trust": ["trust", "admiration", "gratitude"],
        "anticipation": ["anticipation", "interest", "optimism"],
    }

    def __init__(
        self,
        hidden_size: int = 768,
        num_emotions: int = 44,  # Default to FamilyOS 44 emotions
        num_secondary: int = 3,
        dropout: float = 0.1,
        pooling: str = "cls",
        use_intensity: bool = True,
        use_valence_arousal: bool = False,
        intensity_threshold: float = 0.1,
        emotion_labels: list[str] | None = None,
        use_familyos: bool = True,  # Use FamilyOS 44-emotion schema by default
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_emotions = num_emotions
        self.num_secondary = num_secondary
        self.pooling = pooling
        self.use_intensity = use_intensity
        self.use_valence_arousal = use_valence_arousal
        self.intensity_threshold = intensity_threshold
        self.use_familyos = use_familyos

        # Emotion labels - priority: explicit > familyos > legacy
        if emotion_labels is not None:
            self.emotion_labels = emotion_labels
        elif use_familyos:
            self.emotion_labels = self.FAMILYOS_EMOTION_LABELS[:num_emotions]
        else:
            self.emotion_labels = self.DEFAULT_EMOTION_LABELS[:num_emotions]

        assert (
            len(self.emotion_labels) == num_emotions
        ), f"Expected {num_emotions} emotion labels, got {len(self.emotion_labels)}"

        # Select appropriate emotion families
        self._emotion_families = (
            self.FAMILYOS_EMOTION_FAMILIES if use_familyos else self.EMOTION_FAMILIES
        )

        # Build label to index mapping
        self.label2id = {label: i for i, label in enumerate(self.emotion_labels)}
        self.id2label = {i: label for i, label in enumerate(self.emotion_labels)}

        # Layers
        self.dropout = nn.Dropout(dropout)

        # Shared representation layer
        self.shared_dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()

        # Emotion classification head (multi-label, each emotion independently)
        self.emotion_classifier = nn.Linear(hidden_size, num_emotions)

        # Intensity regression head (0-1 score per emotion)
        if use_intensity:
            self.intensity_regressor = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, num_emotions),
                nn.Sigmoid(),  # Output 0-1 range
            )

        # Valence-Arousal head (optional dimensional representation)
        if use_valence_arousal:
            self.valence_arousal = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 2),  # [valence, arousal]
                nn.Tanh(),  # Output -1 to 1 range
            )

        # Learnable temperature for calibration
        self.temperature = nn.Parameter(torch.ones(1))

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.xavier_uniform_(self.shared_dense.weight)
        nn.init.zeros_(self.shared_dense.bias)
        nn.init.xavier_uniform_(self.emotion_classifier.weight)
        nn.init.zeros_(self.emotion_classifier.bias)

    def pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool sequence representations."""
        if self.pooling == "cls":
            return hidden_states[:, 0, :]
        elif self.pooling == "mean":
            if attention_mask is None:
                return hidden_states.mean(dim=1)
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_hidden = (hidden_states * mask).sum(dim=1)
            sum_mask = mask.sum(dim=1).clamp(min=1e-9)
            return sum_hidden / sum_mask
        elif self.pooling == "max":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                hidden_states = hidden_states.masked_fill(mask == 0, -1e9)
            return hidden_states.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | str | list | dict]:
        """
        Forward pass through the hierarchical emotion head.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
            labels: Optional multi-hot labels (batch_size, num_emotions)

        Returns:
            Dictionary containing:
                - logits: Raw emotion logits (batch_size, num_emotions)
                - probabilities: Emotion probabilities (batch_size, num_emotions)
                - primary_emotion: Primary emotion label (str or list[str])
                - primary_emotion_id: Primary emotion index (int or list[int])
                - secondary_emotions: Top-k secondary emotions (list)
                - emotion_scores: Dict mapping emotion label to score
                - intensity: Intensity scores if enabled (batch_size, num_emotions)
                - valence_arousal: [valence, arousal] if enabled (batch_size, 2)
                - loss: Loss if labels provided
        """
        # Pool sequence representation
        pooled = self.pool(hidden_states, attention_mask)

        # Shared representation
        shared = self.dropout(pooled)
        shared = self.shared_dense(shared)
        shared = self.activation(shared)
        shared = self.dropout(shared)

        # Emotion classification (multi-label)
        logits = self.emotion_classifier(shared)
        scaled_logits = logits / self.temperature.clamp(min=0.01)
        probabilities = torch.sigmoid(scaled_logits)

        # Build output dictionary
        output: dict[str, torch.Tensor | str | list | dict] = {
            "logits": logits,
            "probabilities": probabilities,
        }

        # Get batch size for processing
        batch_size = hidden_states.size(0)

        # Process predictions for each sample
        primary_emotions = []
        primary_emotion_ids = []
        secondary_emotions_list = []
        emotion_scores_list = []

        for batch_idx in range(batch_size):
            probs = probabilities[batch_idx]

            # Get sorted indices by probability
            sorted_indices = torch.argsort(probs, descending=True)
            sorted_probs = probs[sorted_indices]

            # Primary emotion (highest probability)
            primary_idx = sorted_indices[0].item()
            primary_label = self.id2label[primary_idx]
            primary_emotions.append(primary_label)
            primary_emotion_ids.append(primary_idx)

            # Secondary emotions (next top-k above threshold)
            secondary = []
            for i in range(1, min(len(sorted_indices), self.num_secondary + 1)):
                idx = sorted_indices[i].item()
                prob = sorted_probs[i].item()
                if prob >= self.intensity_threshold:
                    secondary.append(self.id2label[idx])
            secondary_emotions_list.append(secondary)

            # Emotion scores dict
            scores = {self.id2label[i]: probs[i].item() for i in range(self.num_emotions)}
            emotion_scores_list.append(scores)

        # Handle single sample vs batch
        if batch_size == 1:
            output["primary_emotion"] = primary_emotions[0]
            output["primary_emotion_id"] = primary_emotion_ids[0]
            output["secondary_emotions"] = secondary_emotions_list[0]
            output["emotion_scores"] = emotion_scores_list[0]
        else:
            output["primary_emotion"] = primary_emotions
            output["primary_emotion_id"] = primary_emotion_ids
            output["secondary_emotions"] = secondary_emotions_list
            output["emotion_scores"] = emotion_scores_list

        # Intensity scores
        if self.use_intensity:
            intensity = self.intensity_regressor(shared)
            output["intensity"] = intensity

            # Add intensity to emotion scores
            if batch_size == 1:
                output["emotion_intensity"] = {
                    self.id2label[i]: intensity[0, i].item() for i in range(self.num_emotions)
                }
            else:
                output["emotion_intensity"] = [
                    {self.id2label[i]: intensity[b, i].item() for i in range(self.num_emotions)}
                    for b in range(batch_size)
                ]

        # Valence-Arousal
        if self.use_valence_arousal:
            va = self.valence_arousal(shared)
            output["valence_arousal"] = va
            if batch_size == 1:
                output["valence"] = va[0, 0].item()
                output["arousal"] = va[0, 1].item()
            else:
                output["valence"] = [va[b, 0].item() for b in range(batch_size)]
                output["arousal"] = [va[b, 1].item() for b in range(batch_size)]

        # Compute loss if labels provided
        if labels is not None:
            # Multi-label BCE loss for emotion classification
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())

            # Add intensity loss if intensity labels provided
            if self.use_intensity and labels.dim() > 1 and labels.size(-1) > self.num_emotions:
                # Assume labels has shape (batch, num_emotions * 2) with intensity
                intensity_labels = labels[:, self.num_emotions :]
                intensity_loss = F.mse_loss(intensity, intensity_labels)
                loss = loss + 0.5 * intensity_loss

            output["loss"] = loss

        return output

    def get_primary_family(self, emotion: str) -> str:
        """Get the emotion family for a given emotion."""
        for family, members in self._emotion_families.items():
            if emotion in members:
                return family
        return emotion  # Return emotion itself if not in a family

    def get_emotion_distribution(
        self,
        probabilities: torch.Tensor,
    ) -> dict[str, float]:
        """
        Get emotion family distribution.

        Args:
            probabilities: (batch_size, num_emotions) or (num_emotions,)

        Returns:
            Dictionary mapping emotion family to aggregated probability
        """
        if probabilities.dim() == 1:
            probabilities = probabilities.unsqueeze(0)

        distribution = {}
        for family, members in self._emotion_families.items():
            family_probs = []
            for member in members:
                if member in self.label2id:
                    idx = self.label2id[member]
                    family_probs.append(probabilities[:, idx].mean().item())
            if family_probs:
                distribution[family] = sum(family_probs) / len(family_probs)

        return distribution

    def freeze(self) -> None:
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def calibrate_temperature(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        """
        Calibrate temperature using validation data.

        Uses LBFGS optimizer to find optimal temperature that
        minimizes BCE loss on validation set.

        Args:
            val_logits: Validation logits
            val_labels: Validation labels
            lr: Learning rate for optimization
            max_iter: Maximum iterations

        Returns:
            Optimal temperature value
        """
        self.temperature.requires_grad = True
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled_logits = val_logits / self.temperature.clamp(min=0.01)
            loss = F.binary_cross_entropy_with_logits(scaled_logits, val_labels.float())
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature.requires_grad = False

        return self.temperature.item()

    @classmethod
    def for_familyos(
        cls,
        hidden_size: int = 768,
        dropout: float = 0.1,
        **kwargs,
    ) -> HierarchicalEmotionHead:
        """
        Factory method to create a HierarchicalEmotionHead for FamilyOS 44-emotion schema.

        This is the recommended way to create the head for FamilyOS emotion classification.

        Args:
            hidden_size: Size of encoder hidden states
            dropout: Dropout probability
            **kwargs: Additional arguments passed to __init__

        Returns:
            HierarchicalEmotionHead configured for FamilyOS 44 emotions

        Example:
            >>> head = HierarchicalEmotionHead.for_familyos(hidden_size=768)
            >>> assert head.num_emotions == 44
            >>> assert "parental_pride" in head.emotion_labels
        """
        return cls(
            hidden_size=hidden_size,
            num_emotions=44,
            dropout=dropout,
            use_familyos=True,
            emotion_labels=cls.FAMILYOS_EMOTION_LABELS,
            **kwargs,
        )

    @classmethod
    def for_goemotions(
        cls,
        hidden_size: int = 768,
        dropout: float = 0.1,
        **kwargs,
    ) -> HierarchicalEmotionHead:
        """
        Factory method to create a HierarchicalEmotionHead for legacy 32-emotion schema.

        Args:
            hidden_size: Size of encoder hidden states
            dropout: Dropout probability
            **kwargs: Additional arguments passed to __init__

        Returns:
            HierarchicalEmotionHead configured for 32 emotions
        """
        return cls(
            hidden_size=hidden_size,
            num_emotions=32,
            dropout=dropout,
            use_familyos=False,
            emotion_labels=cls.DEFAULT_EMOTION_LABELS,
            **kwargs,
        )


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "BaseHead",
    "SequenceClassificationHead",
    "TokenClassificationHead",
    "EmbeddingHead",
    "NLIHead",
    "SafetyHead",
    "EnhancedSafetyHead",  # NEW - v2
    "HierarchicalEmotionHead",  # NEW - v2 (Issue 3.6.6)
    "RelationHead",  # NEW
    "IntentHead",  # NEW
    "TemporalHead",  # NEW
]
