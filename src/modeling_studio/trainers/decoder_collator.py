"""
Decoder Collator for Counterfactual Generation Training.

This module provides the collator for batching counterfactual samples
with proper padding of encoder embeddings and decoder sequences.

Classes:
    - CounterfactualCollator: Main collator for Stage C decoder training

Key Features:
    - Pads encoder embeddings to batch maximum length
    - Pads decoder input_ids and labels
    - Creates attention masks for both encoder and decoder
    - Handles both pooled and full-sequence encoder embeddings
    - Labels use -100 for padding positions (ignored in loss)

Usage:
    from modeling_studio.trainers.decoder_collator import CounterfactualCollator

    collator = CounterfactualCollator(
        tokenizer=tokenizer,
        max_output_length=256,
        pad_to_multiple_of=8,
    )

    batch = collator(samples)
    # batch contains:
    #   - encoder_hidden_states: (B, enc_seq_len, hidden_dim)
    #   - encoder_attention_mask: (B, enc_seq_len)
    #   - decoder_input_ids: (B, dec_seq_len)
    #   - decoder_attention_mask: (B, dec_seq_len)
    #   - labels: (B, dec_seq_len) with -100 for padding
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

# Label value to ignore in loss computation
IGNORE_INDEX = -100


@dataclass
class CounterfactualCollator:
    """
    Data collator for counterfactual decoder training.

    Handles batching of variable-length encoder embeddings and decoder sequences
    with proper padding and attention mask creation.

    Args:
        tokenizer: Tokenizer for getting pad_token_id
        max_output_length: Maximum decoder sequence length (optional truncation)
        pad_to_multiple_of: Pad sequences to multiple of this value for efficiency
        encoder_hidden_size: Hidden size of encoder embeddings (for validation)

    Attributes:
        pad_token_id: Token ID used for padding decoder sequences
    """

    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
    max_output_length: int | None = None
    pad_to_multiple_of: int | None = 8
    encoder_hidden_size: int = 768

    @property
    def pad_token_id(self) -> int:
        """Get padding token ID from tokenizer."""
        if self.tokenizer.pad_token_id is not None:
            return self.tokenizer.pad_token_id
        if self.tokenizer.eos_token_id is not None:
            return self.tokenizer.eos_token_id
        return 0

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Collate a list of samples into a batch.

        Args:
            features: List of dicts from CounterfactualDataset, each containing:
                - encoder_embeddings: (enc_seq_len, hidden_dim) or (hidden_dim,)
                - encoder_attention_mask: (enc_seq_len,) or (1,)
                - decoder_input_ids: (dec_seq_len,)
                - labels: (dec_seq_len,)
                - sample_id: int

        Returns:
            Batched dict with:
                - encoder_hidden_states: (B, max_enc_len, hidden_dim)
                - encoder_attention_mask: (B, max_enc_len)
                - decoder_input_ids: (B, max_dec_len)
                - decoder_attention_mask: (B, max_dec_len)
                - labels: (B, max_dec_len)
        """
        batch_size = len(features)

        # Separate encoder and decoder components
        encoder_embeddings = [f["encoder_embeddings"] for f in features]
        encoder_masks = [f["encoder_attention_mask"] for f in features]
        decoder_input_ids = [f["decoder_input_ids"] for f in features]
        labels = [f["labels"] for f in features]

        # Pad encoder embeddings
        encoder_batch = self._pad_encoder_embeddings(encoder_embeddings, encoder_masks)

        # Pad decoder sequences
        decoder_batch = self._pad_decoder_sequences(decoder_input_ids, labels)

        return {
            "encoder_hidden_states": encoder_batch["hidden_states"],
            "encoder_attention_mask": encoder_batch["attention_mask"],
            "decoder_input_ids": decoder_batch["input_ids"],
            "decoder_attention_mask": decoder_batch["attention_mask"],
            "labels": decoder_batch["labels"],
        }

    def _pad_encoder_embeddings(
        self,
        embeddings: list[torch.Tensor],
        attention_masks: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Pad encoder embeddings to batch maximum length.

        Handles both:
            - Pooled embeddings: (hidden_dim,) -> (1, hidden_dim)
            - Sequence embeddings: (seq_len, hidden_dim)

        Args:
            embeddings: List of embedding tensors
            attention_masks: List of attention mask tensors

        Returns:
            Dict with padded hidden_states and attention_mask
        """
        # Ensure embeddings are 2D (seq_len, hidden_dim)
        processed_embeddings = []
        for emb in embeddings:
            if emb.dim() == 1:
                # Pooled embedding: (hidden_dim,) -> (1, hidden_dim)
                emb = emb.unsqueeze(0)
            processed_embeddings.append(emb)

        # Find max sequence length
        max_len = max(emb.shape[0] for emb in processed_embeddings)
        hidden_dim = processed_embeddings[0].shape[-1]

        # Apply pad_to_multiple_of
        if self.pad_to_multiple_of is not None and max_len % self.pad_to_multiple_of != 0:
            max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        batch_size = len(processed_embeddings)

        # Create padded tensors
        padded_embeddings = torch.zeros(batch_size, max_len, hidden_dim)
        padded_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

        for i, (emb, mask) in enumerate(zip(processed_embeddings, attention_masks)):
            seq_len = emb.shape[0]
            padded_embeddings[i, :seq_len, :] = emb
            # Expand mask if needed (pooled mode has mask of length 1)
            if mask.shape[0] < seq_len:
                mask = torch.ones(seq_len, dtype=torch.long)
            padded_mask[i, :seq_len] = mask[:seq_len]

        return {
            "hidden_states": padded_embeddings,
            "attention_mask": padded_mask,
        }

    def _pad_decoder_sequences(
        self,
        input_ids: list[torch.Tensor],
        labels: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Pad decoder input_ids and labels.

        Labels are padded with IGNORE_INDEX (-100) so padding doesn't
        contribute to the loss.

        Args:
            input_ids: List of decoder input_id tensors
            labels: List of label tensors

        Returns:
            Dict with padded input_ids, attention_mask, and labels
        """
        # Find max length
        max_len = max(ids.shape[0] for ids in input_ids)

        # Apply max_output_length truncation
        if self.max_output_length is not None:
            max_len = min(max_len, self.max_output_length)

        # Apply pad_to_multiple_of
        if self.pad_to_multiple_of is not None and max_len % self.pad_to_multiple_of != 0:
            max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        batch_size = len(input_ids)

        # Create padded tensors
        padded_input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        padded_attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        padded_labels = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long)

        for i, (ids, lbl) in enumerate(zip(input_ids, labels)):
            seq_len = min(ids.shape[0], max_len)

            # Truncate if needed
            ids = ids[:seq_len]
            lbl = lbl[:seq_len]

            padded_input_ids[i, :seq_len] = ids
            padded_attention_mask[i, :seq_len] = 1
            padded_labels[i, :seq_len] = lbl

        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "labels": padded_labels,
        }


@dataclass
class SequenceCounterfactualCollator(CounterfactualCollator):
    """
    Specialized collator for full-sequence encoder embeddings.

    Same as CounterfactualCollator but with explicit handling for
    variable-length encoder sequences used in cross-attention.

    Use this when training with full_sequence=True in CounterfactualDataset.
    """

    def _pad_encoder_embeddings(
        self,
        embeddings: list[torch.Tensor],
        attention_masks: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Pad full-sequence encoder embeddings.

        Ensures proper handling of variable-length sequences from
        the encoder for cross-attention in the decoder.
        """
        # All embeddings should already be 2D for full-sequence mode
        max_len = max(emb.shape[0] for emb in embeddings)
        hidden_dim = embeddings[0].shape[-1]

        # Apply pad_to_multiple_of
        if self.pad_to_multiple_of is not None and max_len % self.pad_to_multiple_of != 0:
            max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        batch_size = len(embeddings)

        # Create padded tensors
        padded_embeddings = torch.zeros(batch_size, max_len, hidden_dim)
        padded_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

        for i, (emb, mask) in enumerate(zip(embeddings, attention_masks)):
            seq_len = emb.shape[0]
            padded_embeddings[i, :seq_len, :] = emb
            padded_mask[i, :seq_len] = mask[:seq_len]

        return {
            "hidden_states": padded_embeddings,
            "attention_mask": padded_mask,
        }
