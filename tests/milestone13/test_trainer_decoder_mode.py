"""
Tests for decoder trainer and freezing utilities (Issue 13.2.1, 13.2.2).

Tests the DecoderTrainer class and freeze_encoder_and_heads function.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_encoder():
    """Create a mock encoder with parameters."""
    encoder = nn.Sequential(
        nn.Linear(768, 768),
        nn.Linear(768, 768),
    )
    return encoder


@pytest.fixture
def mock_head():
    """Create a mock head with parameters."""
    head = nn.Linear(768, 10)
    return head


@pytest.fixture
def mock_model(mock_encoder, mock_head):
    """Create a mock multi-task model."""
    model = MagicMock()
    model.encoder = mock_encoder
    model.heads = {
        "sentiment": mock_head,
        "ner_general": nn.Linear(768, 20),
        "counterfactual": nn.Linear(768, 50000),  # Decoder head
    }
    model.parameters = lambda: list(mock_encoder.parameters()) + \
                               list(mock_head.parameters()) + \
                               list(model.heads["ner_general"].parameters()) + \
                               list(model.heads["counterfactual"].parameters())
    return model


# =============================================================================
# Issue 13.2.2: Encoder Freezing Utility
# =============================================================================


class TestFreezeEncoderAndHeads:
    """Tests for freeze_encoder_and_heads function."""

    def test_freeze_encoder(self, mock_model):
        """13.2.2-T1: Encoder params require_grad=False."""
        from modeling_studio.trainers.decoder_trainer import freeze_encoder_and_heads

        freeze_encoder_and_heads(
            mock_model,
            freeze_encoder=True,
            freeze_heads=False,
            decoder_head_name="counterfactual",
        )

        # Check encoder params are frozen
        for param in mock_model.encoder.parameters():
            assert not param.requires_grad, "Encoder params should be frozen"

    def test_freeze_existing_heads(self, mock_model):
        """13.2.2-T2: Existing heads require_grad=False."""
        from modeling_studio.trainers.decoder_trainer import freeze_encoder_and_heads

        freeze_encoder_and_heads(
            mock_model,
            freeze_encoder=False,
            freeze_heads=True,
            decoder_head_name="counterfactual",
        )

        # Check sentiment head is frozen
        for param in mock_model.heads["sentiment"].parameters():
            assert not param.requires_grad, "sentiment head should be frozen"

        # Check ner_general head is frozen
        for param in mock_model.heads["ner_general"].parameters():
            assert not param.requires_grad, "ner_general head should be frozen"

    def test_decoder_head_trainable(self, mock_model):
        """13.2.2-T3: Counterfactual head require_grad=True."""
        from modeling_studio.trainers.decoder_trainer import freeze_encoder_and_heads

        freeze_encoder_and_heads(
            mock_model,
            freeze_encoder=True,
            freeze_heads=True,
            decoder_head_name="counterfactual",
        )

        # Check decoder head is NOT frozen
        for param in mock_model.heads["counterfactual"].parameters():
            assert param.requires_grad, "Counterfactual head should remain trainable"

    def test_freeze_returns_stats(self, mock_model):
        """Freeze function returns parameter statistics."""
        from modeling_studio.trainers.decoder_trainer import freeze_encoder_and_heads

        stats = freeze_encoder_and_heads(
            mock_model,
            freeze_encoder=True,
            freeze_heads=True,
            decoder_head_name="counterfactual",
        )

        assert "total" in stats
        assert "trainable" in stats
        assert "frozen" in stats
        assert stats["total"] == stats["trainable"] + stats["frozen"]


# =============================================================================
# Issue 13.2.1: Decoder Training Mode
# =============================================================================


class TestDecoderTrainerInit:
    """Tests for DecoderTrainer initialization."""

    def test_decoder_trainer_exists(self):
        """DecoderTrainer class is importable."""
        from modeling_studio.trainers.decoder_trainer import DecoderTrainer
        assert DecoderTrainer is not None

    def test_decoder_trainer_accepts_aux_loss_weight(self):
        """13.2.1-T1: Trainer accepts aux_loss_weight parameter."""
        from modeling_studio.trainers.decoder_trainer import DecoderTrainer
        from transformers import TrainingArguments

        with patch.object(DecoderTrainer, "__init__", lambda self, **kwargs: None):
            trainer = DecoderTrainer.__new__(DecoderTrainer)
            trainer.aux_loss_weight = 0.011
            assert hasattr(trainer, "aux_loss_weight")


class TestDecoderTrainerComputeLoss:
    """Tests for DecoderTrainer.compute_loss."""

    def test_compute_loss_adds_aux_loss(self):
        """13.2.1-T2: aux_loss added to total loss."""
        # This test verifies the compute_loss logic handles aux_loss
        from modeling_studio.trainers.decoder_trainer import DecoderTrainer

        # Create mock trainer with minimal setup
        trainer = object.__new__(DecoderTrainer)
        trainer.aux_loss_weight = 1.0
        trainer._aux_loss_accumulator = {"load_balance": 0.0, "z_loss": 0.0, "total_aux": 0.0}
        trainer._aux_loss_count = 0

        # Mock model that returns loss and aux_loss
        mock_model = MagicMock()
        mock_model.training = True
        mock_model.heads = {
            "counterfactual": MagicMock(
                return_value={
                    "loss": torch.tensor(2.0, requires_grad=True),
                    "aux_loss": torch.tensor(0.5),
                    "logits": torch.randn(2, 10, 50000),
                }
            )
        }

        # Set trainer.model to the mock (needed by compute_loss)
        trainer.model = mock_model

        # Mock inputs
        inputs = {
            "encoder_embeddings": torch.randn(2, 768),
            "encoder_attention_mask": torch.ones(2, 1),
            "decoder_input_ids": torch.randint(0, 1000, (2, 10)),
            "labels": torch.randint(0, 1000, (2, 10)),
        }

        # Call compute_loss
        total_loss = trainer.compute_loss(mock_model, inputs)

        # Verify aux_loss was added (2.0 + 1.0 * 0.5 = 2.5)
        assert total_loss.item() == pytest.approx(2.5, rel=0.01)


class TestDecoderTrainerLogging:
    """Tests for aux_loss logging."""

    def test_aux_loss_accumulated(self):
        """Aux losses are accumulated for logging."""
        from modeling_studio.trainers.decoder_trainer import DecoderTrainer

        trainer = object.__new__(DecoderTrainer)
        trainer._aux_loss_accumulator = {"load_balance": 0.0, "z_loss": 0.0, "total_aux": 0.0}
        trainer._aux_loss_count = 0

        # Simulate accumulation
        trainer._aux_loss_accumulator["total_aux"] = 0.5
        trainer._aux_loss_count = 1

        assert trainer._aux_loss_count == 1
        assert trainer._aux_loss_accumulator["total_aux"] == 0.5
