"""
Hub Token Injection Tokenizer (ModernBERT v3.3 Ultra).

This module provides a tokenizer wrapper that automatically injects
hub tokens ([EMO], [MEM], [REL], [TASK]) after [CLS] for all inputs.

Example:
    Input:  "Mom is happy today"
    Output: "[CLS] [EMO] [MEM] [REL] [TASK] Mom is happy today [SEP]"
"""

from __future__ import annotations

import torch
from transformers import AutoTokenizer

from .hub_tokens import HUB_TOKEN_REGISTRY


class HubTokenizer:
    """
    Wrapper tokenizer that injects hub tokens after [CLS].

    This tokenizer extends ModernBERT-base tokenizer to include 4 specialized
    hub tokens that route to different capability heads in the v3 architecture.

    Attributes:
        base_tokenizer: Underlying ModernBERT tokenizer
        hub_token_ids: Mapping of hub token strings to token IDs
        hub_sequence: List of hub token IDs in order [EMO, MEM, REL, TASK]
        num_hub_tokens: Number of hub tokens (always 4)
    """

    def __init__(self, base_tokenizer_name: str = "answerdotai/ModernBERT-base"):
        """
        Initialize the hub tokenizer.

        Args:
            base_tokenizer_name: HuggingFace model name for base tokenizer
        """
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name)

        # Add hub tokens to vocabulary
        hub_tokens = list(HUB_TOKEN_REGISTRY.keys())
        num_added = self.base_tokenizer.add_special_tokens(
            {"additional_special_tokens": hub_tokens}
        )
        print(f"Added {num_added} hub tokens to vocabulary")

        # Cache hub token IDs
        self.hub_token_ids = {
            token: self.base_tokenizer.convert_tokens_to_ids(token) for token in hub_tokens
        }
        self.cls_token_id = self.base_tokenizer.cls_token_id
        self.sep_token_id = self.base_tokenizer.sep_token_id
        self.pad_token_id = self.base_tokenizer.pad_token_id

        # Hub token sequence: [EMO], [MEM], [REL], [TASK]
        self.hub_sequence = [
            self.hub_token_ids["[EMO]"],
            self.hub_token_ids["[MEM]"],
            self.hub_token_ids["[REL]"],
            self.hub_token_ids["[TASK]"],
        ]
        self.num_hub_tokens = len(self.hub_sequence)

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size (including hub tokens)."""
        return len(self.base_tokenizer)

    def __call__(
        self,
        text: str | list[str],
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: str = "pt",
        **kwargs,
    ) -> dict:
        """
        Tokenize text with hub token injection.

        Args:
            text: Input text or list of texts
            max_length: Maximum sequence length (including hub tokens)
            padding: Padding strategy ('max_length', 'longest', or False)
            truncation: Whether to truncate sequences
            return_tensors: Format for returned tensors ('pt' or 'np')
            **kwargs: Additional arguments (ignored)

        Returns:
            BatchEncoding with input_ids, attention_mask, and hub_token_mask

        Example:
            >>> tokenizer = HubTokenizer()
            >>> output = tokenizer("Mom is happy")
            >>> output.keys()
            dict_keys(['input_ids', 'attention_mask', 'hub_token_mask'])
        """
        # Adjust max_length to account for hub tokens
        # Original: [CLS] text [SEP] -> New: [CLS] [EMO] [MEM] [REL] [TASK] text [SEP]
        adjusted_max_length = max_length - self.num_hub_tokens

        # Tokenize without special tokens (we'll add them manually)
        if isinstance(text, str):
            text = [text]

        batch_input_ids = []
        batch_attention_mask = []
        batch_hub_token_mask = []

        for t in text:
            # Tokenize text only (no CLS/SEP)
            encoded = self.base_tokenizer.encode(
                t,
                add_special_tokens=False,
                max_length=adjusted_max_length - 2,  # Reserve for CLS and SEP
                truncation=truncation,
            )

            # Build sequence: [CLS] [EMO] [MEM] [REL] [TASK] <text> [SEP]
            input_ids = [self.cls_token_id] + self.hub_sequence + encoded + [self.sep_token_id]

            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = [1] * len(input_ids)

            # Create hub token mask (1 for hub positions, 0 otherwise)
            # Position 0 = CLS (not a hub), Positions 1-4 = hub tokens
            hub_token_mask = [0] + [1] * self.num_hub_tokens + [0] * (len(encoded) + 1)

            # Pad to max_length
            padding_length = max_length - len(input_ids)
            if padding_length > 0:
                input_ids += [self.pad_token_id] * padding_length
                attention_mask += [0] * padding_length
                hub_token_mask += [0] * padding_length

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_hub_token_mask.append(hub_token_mask)

        # Convert to tensors
        result = {
            "input_ids": torch.tensor(batch_input_ids),
            "attention_mask": torch.tensor(batch_attention_mask),
            "hub_token_mask": torch.tensor(batch_hub_token_mask),
        }

        return result

    def get_hub_token_positions(self) -> dict[str, int]:
        """
        Get the positions of hub tokens in the sequence.

        Returns:
            Dict mapping hub token name to position index

        Example:
            >>> tokenizer.get_hub_token_positions()
            {'[CLS]': 0, '[EMO]': 1, '[MEM]': 2, '[REL]': 3, '[TASK]': 4}
        """
        return {
            "[CLS]": 0,
            "[EMO]": 1,
            "[MEM]": 2,
            "[REL]": 3,
            "[TASK]": 4,
        }

    def get_text_start_position(self) -> int:
        """
        Get the position where actual text tokens start.

        Returns:
            Position index (always 5 for v3)

        Example:
            >>> tokenizer.get_text_start_position()
            5  # After [CLS] + 4 hub tokens
        """
        return 5  # After [CLS] + 4 hub tokens

    def decode(self, token_ids: list[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: List or tensor of token IDs
            skip_special_tokens: Whether to remove special tokens from output

        Returns:
            Decoded text string
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.base_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(
        self, token_ids: list[list[int]] | torch.Tensor, skip_special_tokens: bool = True
    ) -> list[str]:
        """
        Decode batch of token IDs back to text.

        Args:
            token_ids: Batch of token ID sequences
            skip_special_tokens: Whether to remove special tokens

        Returns:
            List of decoded text strings
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.base_tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)

    def save_pretrained(self, path: str):
        """
        Save tokenizer to disk.

        Args:
            path: Directory path to save tokenizer files
        """
        self.base_tokenizer.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path: str) -> HubTokenizer:
        """
        Load tokenizer from disk.

        Args:
            path: Directory path containing tokenizer files

        Returns:
            Loaded HubTokenizer instance
        """
        instance = cls.__new__(cls)
        instance.base_tokenizer = AutoTokenizer.from_pretrained(path)

        # Re-cache hub token IDs
        hub_tokens = list(HUB_TOKEN_REGISTRY.keys())
        instance.hub_token_ids = {
            token: instance.base_tokenizer.convert_tokens_to_ids(token) for token in hub_tokens
        }
        instance.cls_token_id = instance.base_tokenizer.cls_token_id
        instance.sep_token_id = instance.base_tokenizer.sep_token_id
        instance.pad_token_id = instance.base_tokenizer.pad_token_id

        instance.hub_sequence = [
            instance.hub_token_ids["[EMO]"],
            instance.hub_token_ids["[MEM]"],
            instance.hub_token_ids["[REL]"],
            instance.hub_token_ids["[TASK]"],
        ]
        instance.num_hub_tokens = len(instance.hub_sequence)

        return instance

    def __len__(self) -> int:
        """Get vocabulary size."""
        return len(self.base_tokenizer)

    def __repr__(self) -> str:
        """String representation of tokenizer."""
        return (
            f"HubTokenizer(base={self.base_tokenizer.__class__.__name__}, "
            f"vocab_size={self.vocab_size}, hub_tokens={self.num_hub_tokens})"
        )
