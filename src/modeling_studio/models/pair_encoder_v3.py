"""
ModernBERT v3.3 Ultra - Pair Encoder with [REL] Hub Token

This module implements pair encoders for sentence-pair tasks (NLI, relation
extraction, semantic similarity) that leverage the [REL] hub token for
relationship representation.

Key Innovation:
    The [REL] hub token (position 3) is specifically designed to capture
    relationships between text_a and text_b through cross-attention across
    the full sequence, enabling more effective pair classification.

Token Layout for Pairs:
    [CLS] [EMO] [MEM] [REL] [TASK] <text_a> [SEP] <text_b> [SEP] [PAD]...
    pos 0   1     2     3     4        5+

Use Cases:
    - Natural Language Inference (NLI)
    - Relation extraction
    - Semantic similarity
    - Paraphrase detection
    - Duplicate detection

Author: FamilyOS Team
Date: December 2025
Version: 3.3
"""

import torch
import torch.nn as nn

from .hub_tokens import get_hub_positions


class PairEncoderV3(nn.Module):
    """
    Pair Encoder for sentence-pair tasks in v3.

    Token Layout for Pairs:
        [CLS] [EMO] [MEM] [REL] [TASK] <text_a> [SEP] <text_b> [SEP] [PAD]...

    Key Innovation: The [REL] hub token (position 3) captures the
    relationship between text_a and text_b through cross-attention
    across the full sequence.

    Use Cases:
        - NLI (Natural Language Inference)
        - Relation extraction
        - Semantic similarity
        - Paraphrase detection

    Args:
        hidden_size: Dimension of hidden states (default: 768)
        num_labels: Number of output labels (default: 3 for NLI)
        classifier_dropout: Dropout probability for classifier (default: 0.1)
        use_rel_hub: Whether to use [REL] hub token (default: True)
        pooling_strategy: Strategy for pooling representations
            - "rel_hub": Use [REL] hub token (default, recommended)
            - "cls": Traditional CLS token
            - "mean": Mean pooling over text tokens
            - "concat": Concatenate CLS + REL + mean_diff

    Example:
        >>> pair_encoder = PairEncoderV3(num_labels=3)  # NLI
        >>> encoder_output = torch.randn(2, 128, 768)  # [batch, seq, hidden]
        >>> logits = pair_encoder(encoder_output)
        >>> print(logits.shape)  # [2, 3]
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_labels: int = 3,  # NLI: entailment, neutral, contradiction
        classifier_dropout: float = 0.1,
        use_rel_hub: bool = True,
        pooling_strategy: str = "rel_hub",
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.use_rel_hub = use_rel_hub
        self.pooling_strategy = pooling_strategy

        # Get hub positions
        self.hub_positions = get_hub_positions()
        self.rel_position = self.hub_positions["[REL]"]  # Position 3
        self.cls_position = self.hub_positions["[CLS]"]  # Position 0

        # Determine classifier input size based on strategy
        if pooling_strategy == "concat":
            classifier_input_size = hidden_size * 3  # CLS + REL + mean_diff
        elif pooling_strategy == "rel_hub":
            classifier_input_size = hidden_size
        else:
            classifier_input_size = hidden_size

        # Classifier head
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(classifier_input_size, num_labels)

        # Optional: Cross-attention refinement layer
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=classifier_dropout,
            batch_first=True,
        )
        self.use_cross_attention = False  # Can be enabled for enhanced fusion

    def forward(
        self,
        encoder_output: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        text_a_mask: torch.Tensor | None = None,
        text_b_mask: torch.Tensor | None = None,
        return_pooled: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for pair classification.

        Args:
            encoder_output: [batch, seq_len, hidden_size] from encoder
            attention_mask: [batch, seq_len] padding mask
            text_a_mask: [batch, seq_len] mask for first sentence (optional)
            text_b_mask: [batch, seq_len] mask for second sentence (optional)
            return_pooled: Also return the pooled representation

        Returns:
            Classification logits [batch, num_labels]
            Optionally: (logits, pooled_representation)
        """
        # Extract pooled representation based on strategy
        if self.pooling_strategy == "rel_hub":
            # Use [REL] hub token - designed for relationship representation
            pooled = encoder_output[:, self.rel_position, :]  # [batch, hidden]

        elif self.pooling_strategy == "cls":
            # Traditional CLS pooling
            pooled = encoder_output[:, self.cls_position, :]

        elif self.pooling_strategy == "mean":
            # Mean pooling over non-special tokens
            if attention_mask is not None:
                # Mask out positions 0-4 (CLS + hubs)
                text_mask = attention_mask.clone()
                text_mask[:, :5] = 0  # Exclude special tokens
                mask_expanded = text_mask.unsqueeze(-1).float()
                sum_hidden = (encoder_output * mask_expanded).sum(dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                pooled = sum_hidden / sum_mask
            else:
                pooled = encoder_output[:, 5:, :].mean(dim=1)

        elif self.pooling_strategy == "concat":
            # Concatenate CLS + REL + mean difference
            cls_repr = encoder_output[:, self.cls_position, :]
            rel_repr = encoder_output[:, self.rel_position, :]

            # Compute mean representations for text_a and text_b if masks provided
            if text_a_mask is not None and text_b_mask is not None:
                a_mask = text_a_mask.unsqueeze(-1).float()
                b_mask = text_b_mask.unsqueeze(-1).float()
                mean_a = (encoder_output * a_mask).sum(1) / a_mask.sum(1).clamp(min=1e-9)
                mean_b = (encoder_output * b_mask).sum(1) / b_mask.sum(1).clamp(min=1e-9)
                mean_diff = torch.abs(mean_a - mean_b)
            else:
                mean_diff = rel_repr  # Fallback

            pooled = torch.cat([cls_repr, rel_repr, mean_diff], dim=-1)

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        # Optional cross-attention refinement
        if self.use_cross_attention and text_a_mask is not None and text_b_mask is not None:
            # Get text_a and text_b representations
            # This is a placeholder for more sophisticated fusion
            pass

        # Classification
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        if return_pooled:
            return logits, pooled
        return logits

    def get_rel_hub_representation(
        self,
        encoder_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract the [REL] hub token representation.

        This is the primary representation for relationship tasks.

        Args:
            encoder_output: [batch, seq_len, hidden_size]

        Returns:
            [REL] representation [batch, hidden_size]
        """
        return encoder_output[:, self.rel_position, :]

    def set_pooling_strategy(self, strategy: str) -> None:
        """
        Change pooling strategy at runtime.

        Strategies:
            - "rel_hub": Use [REL] hub token (default, recommended)
            - "cls": Traditional CLS token
            - "mean": Mean pooling over text tokens
            - "concat": Concatenate CLS + REL + mean_diff

        Args:
            strategy: New pooling strategy

        Raises:
            ValueError: If strategy is not valid
        """
        valid_strategies = ["rel_hub", "cls", "mean", "concat"]
        if strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}")
        self.pooling_strategy = strategy
        print(f"✓ Pair encoder pooling strategy set to: {strategy}")

    def extra_repr(self) -> str:
        """Return extra representation string for debugging."""
        return (
            f"hidden_size={self.hidden_size}, num_labels={self.num_labels}, "
            f"pooling={self.pooling_strategy}"
        )


class SiamesePairEncoderV3(nn.Module):
    """
    Siamese-style pair encoder for semantic similarity.

    Uses the [MEM] hub token for embedding representation
    and [REL] hub for explicit relationship modeling.

    Good for:
        - Semantic textual similarity (STS)
        - Duplicate detection
        - Embedding-based retrieval ranking

    Args:
        hidden_size: Dimension of hidden states (default: 768)
        similarity_function: Function to compute similarity
            - "cosine": Cosine similarity (default)
            - "euclidean": Negative Euclidean distance
            - "learned": Learned similarity with neural network

    Example:
        >>> siamese = SiamesePairEncoderV3(similarity_function="cosine")
        >>> output_a = torch.randn(2, 128, 768)  # [batch, seq, hidden]
        >>> output_b = torch.randn(2, 128, 768)
        >>> similarity = siamese(output_a, output_b)
        >>> print(similarity.shape)  # [2]
    """

    def __init__(
        self,
        hidden_size: int = 768,
        similarity_function: str = "cosine",  # "cosine", "euclidean", "learned"
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.similarity_function = similarity_function
        self.hub_positions = get_hub_positions()

        # For learned similarity
        if similarity_function == "learned":
            self.similarity_layer = nn.Sequential(
                nn.Linear(hidden_size * 4, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, 1),
            )

    def forward(
        self,
        encoder_output_a: torch.Tensor,
        encoder_output_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute similarity between two encoded sequences.

        Args:
            encoder_output_a: [batch, seq_len, hidden] for text A
            encoder_output_b: [batch, seq_len, hidden] for text B

        Returns:
            Similarity scores [batch] or [batch, 1]

        Note:
            Uses [MEM] hub token (position 2) for embedding representation,
            which is specifically designed for semantic similarity tasks.
        """
        # Use [MEM] hub for embedding representation
        mem_position = self.hub_positions["[MEM]"]
        embed_a = encoder_output_a[:, mem_position, :]  # [batch, hidden]
        embed_b = encoder_output_b[:, mem_position, :]  # [batch, hidden]

        if self.similarity_function == "cosine":
            # Cosine similarity
            sim = nn.functional.cosine_similarity(embed_a, embed_b, dim=-1)

        elif self.similarity_function == "euclidean":
            # Negative euclidean distance (higher = more similar)
            dist = torch.norm(embed_a - embed_b, p=2, dim=-1)
            sim = -dist

        elif self.similarity_function == "learned":
            # Learned similarity with element-wise operations
            concat = torch.cat(
                [
                    embed_a,
                    embed_b,
                    embed_a * embed_b,  # Element-wise product
                    torch.abs(embed_a - embed_b),  # Absolute difference
                ],
                dim=-1,
            )
            sim = self.similarity_layer(concat).squeeze(-1)

        else:
            raise ValueError(f"Unknown similarity function: {self.similarity_function}")

        return sim

    def extra_repr(self) -> str:
        """Return extra representation string for debugging."""
        return f"hidden_size={self.hidden_size}, " f"similarity={self.similarity_function}"
