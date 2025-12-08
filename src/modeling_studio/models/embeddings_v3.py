"""
ModernBERT v3.3 Ultra Embeddings Module.

This module handles word embeddings, position embeddings, and hub token slots
for the v3 architecture. The embedding layer is compatible with v2 weight transfer
while adding space for 4 hub tokens.

Token Layout:
    [CLS] [EMO] [MEM] [REL] [TASK] <text tokens> [SEP] [PAD]...
    pos 0   1     2     3     4        5+

Key Features:
    - Extended vocabulary: v2 vocab (50264) + 4 hub tokens = 50268
    - Optional position embeddings (RoPE mode skips addition here)
    - LayerNorm and Dropout for regularization
    - Hub token extraction utilities
    - Resize support for adding hub tokens to v2 checkpoints
"""

import torch
import torch.nn as nn

from .hub_tokens import HUB_TOKEN_REGISTRY, get_hub_positions


class ModernBERTEmbeddingsV3(nn.Module):
    """
    Embeddings module for ModernBERT v3.3 Ultra.

    Token layout:
        [CLS] [EMO] [MEM] [REL] [TASK] <text tokens> [SEP] [PAD]...
        pos 0   1     2     3     4        5+

    Components:
        1. Word embeddings (v2 vocab + 4 hub tokens)
        2. Position embeddings (RoPE-style or learned)
        3. LayerNorm
        4. Dropout

    The hub token embeddings (positions 1-4 in vocab) are initialized
    via semantic centroid initialization from v2 embeddings.

    Args:
        vocab_size: Total vocabulary size including hub tokens (default: 50372)
        hidden_size: Hidden dimension (default: 768)
        max_position_embeddings: Maximum sequence length (default: 8192)
        hidden_dropout_prob: Dropout probability (default: 0.1)
        pad_token_id: Padding token ID (default: 0)
        use_rotary_embeddings: Whether to use RoPE (position added in attention)

    Example:
        >>> embeddings = ModernBERTEmbeddingsV3(vocab_size=50372)
        >>> input_ids = torch.randint(0, 50372, (2, 128))
        >>> embeds = embeddings(input_ids)
        >>> embeds.shape
        torch.Size([2, 128, 768])
    """

    def __init__(
        self,
        vocab_size: int = 50372,  # v2 vocab (50368) + 4 hub tokens
        hidden_size: int = 768,
        max_position_embeddings: int = 8192,
        hidden_dropout_prob: float = 0.1,
        pad_token_id: int = 0,
        use_rotary_embeddings: bool = True,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id
        self.use_rotary_embeddings = use_rotary_embeddings

        # Word embeddings
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)

        # Position embeddings (if not using RoPE)
        if not use_rotary_embeddings:
            self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        else:
            self.position_embeddings = None
            # RoPE is applied in attention layers, not here

        # Token type embeddings (optional, not used in ModernBERT)
        self.token_type_embeddings = None

        # LayerNorm and Dropout
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        # Hub token position indices
        self.hub_positions = get_hub_positions()
        self.num_hub_tokens = len(HUB_TOKEN_REGISTRY)
        self.text_start_position = 5  # After [CLS] + 4 hubs

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for embeddings.

        Args:
            input_ids: [batch, seq_len] token IDs
            position_ids: [batch, seq_len] position IDs (optional)
            token_type_ids: [batch, seq_len] type IDs (unused)
            inputs_embeds: [batch, seq_len, hidden] pre-computed embeddings

        Returns:
            Embeddings [batch, seq_len, hidden_size]

        Example:
            >>> embeddings = ModernBERTEmbeddingsV3()
            >>> input_ids = torch.randint(0, 50268, (2, 128))
            >>> output = embeddings(input_ids)
            >>> output.shape
            torch.Size([2, 128, 768])
        """
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        batch_size, seq_len = inputs_embeds.shape[:2]

        # Add position embeddings if not using RoPE
        if self.position_embeddings is not None:
            if position_ids is None:
                position_ids = torch.arange(seq_len, dtype=torch.long, device=inputs_embeds.device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

            position_embeds = self.position_embeddings(position_ids)
            embeddings = inputs_embeds + position_embeds
        else:
            embeddings = inputs_embeds

        # LayerNorm and Dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

    def get_hub_token_embeddings(self) -> dict[str, torch.Tensor]:
        """
        Extract hub token embeddings for inspection.

        Returns:
            Dict mapping hub token names to their embedding vectors

        Example:
            >>> embeddings = ModernBERTEmbeddingsV3()
            >>> hub_embeds = embeddings.get_hub_token_embeddings()
            >>> list(hub_embeds.keys())
            ['[EMO]', '[MEM]', '[REL]', '[TASK]']
            >>> hub_embeds['[EMO]'].shape
            torch.Size([768])
        """
        hub_embeds = {}
        for token_name, position in self.hub_positions.items():
            if token_name == "[CLS]":
                continue  # Skip CLS
            # Hub tokens are at sequence positions 1-4
            # In vocabulary: v2 vocab ends at 50263 (0-indexed), so hub tokens are at:
            # [EMO]=50264, [MEM]=50265, [REL]=50266, [TASK]=50267
            vocab_index = 50263 + position  # 50263 is last v2 token (0-indexed)
            if vocab_index < self.vocab_size:
                hub_embeds[token_name] = self.word_embeddings.weight[vocab_index].detach()
        return hub_embeds

    def resize_token_embeddings(self, new_vocab_size: int) -> None:
        """
        Resize embedding matrix to accommodate new vocabulary size.

        Used when adding hub tokens to v2 vocabulary.
        Vocab size is padded to next multiple of 256 for GPU efficiency.

        Args:
            new_vocab_size: Minimum new vocabulary size (will be rounded up to 256 multiple)

        Example:
            >>> embeddings = ModernBERTEmbeddingsV3(vocab_size=50368)
            >>> embeddings.resize_token_embeddings(50372)
            [OK] Resized embeddings: 50368 -> 50432 (padded to 256 multiple)
        """
        old_vocab_size = self.word_embeddings.num_embeddings

        # Round up to next multiple of 256 for GPU efficiency
        padded_vocab_size = ((new_vocab_size + 255) // 256) * 256

        if padded_vocab_size == old_vocab_size:
            return

        # Create new embedding matrix
        new_embeddings = nn.Embedding(
            padded_vocab_size,
            self.hidden_size,
            padding_idx=self.pad_token_id,
        )

        # Copy old embeddings
        num_to_copy = min(old_vocab_size, padded_vocab_size)
        new_embeddings.weight.data[:num_to_copy] = self.word_embeddings.weight.data[:num_to_copy]

        # Initialize new embeddings (hub tokens + padding) with small random values
        if padded_vocab_size > old_vocab_size:
            nn.init.normal_(
                new_embeddings.weight.data[old_vocab_size:],
                mean=0.0,
                std=0.02,
            )

        self.word_embeddings = new_embeddings
        self.vocab_size = padded_vocab_size
        if padded_vocab_size != new_vocab_size:
            print(
                f"[OK] Resized embeddings: {old_vocab_size} -> {padded_vocab_size} (padded to 256 multiple)"
            )
        else:
            print(f"[OK] Resized embeddings: {old_vocab_size} -> {padded_vocab_size}")

    def get_num_params(self) -> dict[str, int]:
        """
        Get parameter counts for embeddings.

        Returns:
            Dictionary with parameter counts by component

        Example:
            >>> embeddings = ModernBERTEmbeddingsV3()
            >>> params = embeddings.get_num_params()
            >>> params['word_embeddings']
            38605824  # 50268 * 768
        """
        params = {
            "word_embeddings": self.word_embeddings.weight.numel(),
            "position_embeddings": (
                0 if self.position_embeddings is None else self.position_embeddings.weight.numel()
            ),
            "layer_norm": sum(p.numel() for p in self.LayerNorm.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }
        return params

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f"vocab_size={self.vocab_size}, hidden_size={self.hidden_size}, "
            f"max_position={self.max_position_embeddings}, "
            f"rotary={'yes' if self.use_rotary_embeddings else 'no'}"
        )
