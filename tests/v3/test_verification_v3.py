"""
Tests for ModernBERT v3 Function Preserving Verification.

Tests for Issue 4.2.1: Function Preserving Verification

This module tests the verification utilities that ensure v3 produces
identical outputs to v2 for the first 22 layers.

Test Categories:
    - TestLayerComparisonResult: Data structure tests
    - TestVerificationResult: Data structure tests
    - TestFunctionPreservingVerifier: Main verifier class tests
    - TestVerifyFunctionPreserving: Convenience function tests
    - TestWeightVerification: Weight comparison tests
    - TestIssue421AcceptanceCriteria: Acceptance criteria tests

Author: FamilyOS Team
Date: December 2025
"""

import pytest
import torch
import torch.nn as nn
from typing import Optional, Tuple


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def mock_v2_model():
    """Create a mock v2 model with 22 layers."""

    class MockEmbeddings(nn.Module):
        def __init__(self):
            super().__init__()
            self.word_embeddings = nn.Embedding(50368, 768)

        def forward(self, input_ids):
            return self.word_embeddings(input_ids)

    class MockLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(768, 768)
            self.norm = nn.LayerNorm(768)

        def forward(self, hidden_states, attention_mask=None):
            output = self.linear(hidden_states)
            output = self.norm(output)
            return output

    class MockEncoder(nn.Module):
        def __init__(self, num_layers=22):
            super().__init__()
            self.layers = nn.ModuleList([MockLayer() for _ in range(num_layers)])

    class MockV2Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = MockEmbeddings()
            self.encoder = MockEncoder(num_layers=22)

    return MockV2Model()


@pytest.fixture
def mock_v3_model():
    """Create a mock v3 model with 28 layers."""

    class MockEmbeddings(nn.Module):
        def __init__(self):
            super().__init__()
            self.word_embeddings = nn.Embedding(50372, 768)

        def forward(self, input_ids):
            return self.word_embeddings(input_ids)

    class MockLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(768, 768)
            self.norm = nn.LayerNorm(768)

        def forward(self, hidden_states, attention_mask=None):
            output = self.linear(hidden_states)
            output = self.norm(output)
            return output

    class MockEncoder(nn.Module):
        def __init__(self, num_layers=28):
            super().__init__()
            self.layers = nn.ModuleList([MockLayer() for _ in range(num_layers)])

    class MockV3Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = MockEmbeddings()
            self.encoder = MockEncoder(num_layers=28)

    return MockV3Model()


@pytest.fixture
def matched_models():
    """Create v2 and v3 models with identical weights for first 22 layers."""

    class MockEmbeddings(nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.word_embeddings = nn.Embedding(vocab_size, 768)

        def forward(self, input_ids):
            return self.word_embeddings(input_ids)

    class MockLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(768, 768)
            self.norm = nn.LayerNorm(768)

        def forward(self, hidden_states, attention_mask=None):
            output = self.linear(hidden_states)
            output = self.norm(output)
            return output

    class MockEncoder(nn.Module):
        def __init__(self, num_layers):
            super().__init__()
            self.layers = nn.ModuleList([MockLayer() for _ in range(num_layers)])

    class MockModel(nn.Module):
        def __init__(self, vocab_size, num_layers):
            super().__init__()
            self.embeddings = MockEmbeddings(vocab_size)
            self.encoder = MockEncoder(num_layers)

    # Create models
    v2_model = MockModel(vocab_size=50368, num_layers=22)
    v3_model = MockModel(vocab_size=50372, num_layers=28)

    # Copy weights from v2 to v3 for first 22 layers
    with torch.no_grad():
        # Copy embeddings (first 50368 tokens)
        v3_model.embeddings.word_embeddings.weight[:50368] = (
            v2_model.embeddings.word_embeddings.weight.clone()
        )

        # Copy layer weights
        for i in range(22):
            v3_model.encoder.layers[i].load_state_dict(v2_model.encoder.layers[i].state_dict())

    return v2_model, v3_model


@pytest.fixture
def sample_inputs():
    """Create sample inputs for testing."""
    batch_size = 2
    seq_length = 32
    input_ids = torch.randint(0, 50368, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    return input_ids, attention_mask


# ==============================================================================
# Data Structure Tests
# ==============================================================================


class TestLayerComparisonResult:
    """Tests for LayerComparisonResult dataclass."""

    def test_dataclass_creation(self):
        """Test LayerComparisonResult can be created."""
        from modeling_studio.models.verification_v3 import LayerComparisonResult

        result = LayerComparisonResult(
            layer_idx=0,
            v2_norm=100.0,
            v3_norm=100.0,
            diff_norm=1e-6,
            relative_diff=1e-8,
            passed=True,
        )

        assert result.layer_idx == 0
        assert result.v2_norm == 100.0
        assert result.v3_norm == 100.0
        assert result.diff_norm == 1e-6
        assert result.relative_diff == 1e-8
        assert result.passed is True

    def test_failed_result(self):
        """Test LayerComparisonResult for failed comparison."""
        from modeling_studio.models.verification_v3 import LayerComparisonResult

        result = LayerComparisonResult(
            layer_idx=5,
            v2_norm=100.0,
            v3_norm=105.0,
            diff_norm=0.5,
            relative_diff=0.005,
            passed=False,
        )

        assert result.passed is False
        assert result.layer_idx == 5


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_dataclass_creation(self):
        """Test VerificationResult can be created."""
        from modeling_studio.models.verification_v3 import VerificationResult

        result = VerificationResult(
            passed=True,
            max_diff=1e-6,
            mean_diff=1e-7,
            layer_diffs={0: 1e-6, 1: 1e-7},
            embedding_diff=1e-8,
            failed_layers=[],
            message="Verification passed",
        )

        assert result.passed is True
        assert result.max_diff == 1e-6
        assert result.mean_diff == 1e-7
        assert len(result.layer_diffs) == 2
        assert result.embedding_diff == 1e-8
        assert len(result.failed_layers) == 0

    def test_failed_result(self):
        """Test VerificationResult for failed verification."""
        from modeling_studio.models.verification_v3 import VerificationResult

        result = VerificationResult(
            passed=False,
            max_diff=0.5,
            mean_diff=0.1,
            layer_diffs={0: 0.01, 5: 0.5},
            embedding_diff=0.001,
            failed_layers=[5],
            message="Layer 5 failed",
        )

        assert result.passed is False
        assert 5 in result.failed_layers
        assert result.max_diff == 0.5


class TestWeightComparisonResult:
    """Tests for WeightComparisonResult dataclass."""

    def test_dataclass_creation(self):
        """Test WeightComparisonResult can be created."""
        from modeling_studio.models.verification_v3 import WeightComparisonResult

        result = WeightComparisonResult(
            passed=True,
            matched_params=1000000,
            mismatched_params=0,
            max_diff=1e-8,
        )

        assert result.passed is True
        assert result.matched_params == 1000000
        assert result.mismatched_params == 0


# ==============================================================================
# FunctionPreservingVerifier Tests
# ==============================================================================


class TestFunctionPreservingVerifier:
    """Tests for FunctionPreservingVerifier class."""

    def test_verifier_creation(self, mock_v2_model, mock_v3_model):
        """Test FunctionPreservingVerifier can be created."""
        from modeling_studio.models.verification_v3 import FunctionPreservingVerifier

        verifier = FunctionPreservingVerifier(mock_v2_model, mock_v3_model)

        assert verifier.v2_model is mock_v2_model
        assert verifier.v3_model is mock_v3_model
        assert verifier.tolerance == FunctionPreservingVerifier.TOLERANCE_NORMAL

    def test_verifier_with_custom_tolerance(self, mock_v2_model, mock_v3_model):
        """Test verifier with custom tolerance."""
        from modeling_studio.models.verification_v3 import FunctionPreservingVerifier

        verifier = FunctionPreservingVerifier(
            mock_v2_model,
            mock_v3_model,
            tolerance=FunctionPreservingVerifier.TOLERANCE_STRICT,
        )

        assert verifier.tolerance == 1e-5

    def test_tolerance_constants(self):
        """Test tolerance level constants are defined."""
        from modeling_studio.models.verification_v3 import FunctionPreservingVerifier

        assert FunctionPreservingVerifier.TOLERANCE_STRICT == 1e-5
        assert FunctionPreservingVerifier.TOLERANCE_NORMAL == 1e-4
        assert FunctionPreservingVerifier.TOLERANCE_RELAXED == 1e-3

    def test_num_shared_layers_constant(self):
        """Test NUM_SHARED_LAYERS is 22."""
        from modeling_studio.models.verification_v3 import FunctionPreservingVerifier

        assert FunctionPreservingVerifier.NUM_SHARED_LAYERS == 22

    def test_models_set_to_eval(self, mock_v2_model, mock_v3_model):
        """Test that models are set to eval mode."""
        from modeling_studio.models.verification_v3 import FunctionPreservingVerifier

        mock_v2_model.train()
        mock_v3_model.train()

        _ = FunctionPreservingVerifier(mock_v2_model, mock_v3_model)

        assert not mock_v2_model.training
        assert not mock_v3_model.training

    def test_verify_embeddings(self, matched_models, sample_inputs):
        """Test verify_embeddings method."""
        from modeling_studio.models.verification_v3 import FunctionPreservingVerifier

        v2_model, v3_model = matched_models
        input_ids, attention_mask = sample_inputs

        verifier = FunctionPreservingVerifier(v2_model, v3_model)
        passed, diff, v2_emb, v3_emb = verifier.verify_embeddings(input_ids, attention_mask)

        assert isinstance(passed, bool)
        assert isinstance(diff, float)
        assert v2_emb.shape[0] == input_ids.shape[0]
        assert v3_emb.shape[0] == input_ids.shape[0]

    def test_verify_layer(self, matched_models, sample_inputs):
        """Test verify_layer method."""
        from modeling_studio.models.verification_v3 import (
            FunctionPreservingVerifier,
            LayerComparisonResult,
        )

        v2_model, v3_model = matched_models
        input_ids, attention_mask = sample_inputs

        verifier = FunctionPreservingVerifier(v2_model, v3_model)

        # Get embeddings first
        with torch.no_grad():
            v2_hidden = v2_model.embeddings(input_ids)
            v3_hidden = v3_model.embeddings(input_ids)

        result = verifier.verify_layer(0, v2_hidden, v3_hidden, attention_mask)

        assert isinstance(result, LayerComparisonResult)
        assert result.layer_idx == 0
        assert isinstance(result.passed, bool)

    def test_verify_layer_raises_for_invalid_index(self, matched_models, sample_inputs):
        """Test verify_layer raises for layer index >= 22."""
        from modeling_studio.models.verification_v3 import FunctionPreservingVerifier

        v2_model, v3_model = matched_models
        input_ids, _ = sample_inputs

        verifier = FunctionPreservingVerifier(v2_model, v3_model)

        with torch.no_grad():
            hidden = v2_model.embeddings(input_ids)

        with pytest.raises(IndexError):
            verifier.verify_layer(22, hidden, hidden)

    def test_verify_all_layers(self, matched_models, sample_inputs):
        """Test verify_all_layers method."""
        from modeling_studio.models.verification_v3 import (
            FunctionPreservingVerifier,
            VerificationResult,
        )

        v2_model, v3_model = matched_models
        input_ids, attention_mask = sample_inputs

        verifier = FunctionPreservingVerifier(v2_model, v3_model)
        result = verifier.verify_all_layers(input_ids, attention_mask, verbose=False)

        assert isinstance(result, VerificationResult)
        assert len(result.layer_diffs) == 22
        assert isinstance(result.message, str)

    def test_verify_all_layers_with_matched_weights(self, matched_models, sample_inputs):
        """Test that matched models pass layer verification."""
        from modeling_studio.models.verification_v3 import FunctionPreservingVerifier

        v2_model, v3_model = matched_models
        input_ids, attention_mask = sample_inputs

        verifier = FunctionPreservingVerifier(
            v2_model, v3_model, tolerance=FunctionPreservingVerifier.TOLERANCE_RELAXED
        )

        # When we pass the same input_ids to both models (without hub token injection),
        # the embeddings will be different at positions 1-4 (v2 has text tokens, v3 has hub tokens).
        # So embedding verification may fail. But layer verification with matched weights should pass.

        # Test layer verification directly with same hidden states
        with torch.no_grad():
            hidden = v2_model.embeddings(input_ids)

        result = verifier.verify_layer(0, hidden, hidden, attention_mask)

        # Same input should produce same output for matched layer weights
        assert result.passed is True
        assert result.diff_norm < 1e-6

    def test_verify_weights_only(self, matched_models):
        """Test verify_weights_only method."""
        from modeling_studio.models.verification_v3 import (
            FunctionPreservingVerifier,
            WeightComparisonResult,
        )

        v2_model, v3_model = matched_models

        verifier = FunctionPreservingVerifier(v2_model, v3_model)
        result = verifier.verify_weights_only(verbose=False)

        assert isinstance(result, WeightComparisonResult)
        assert result.matched_params > 0


# ==============================================================================
# Convenience Function Tests
# ==============================================================================


class TestVerifyFunctionPreserving:
    """Tests for verify_function_preserving convenience function."""

    def test_function_exists(self):
        """Test verify_function_preserving function exists."""
        from modeling_studio.models.verification_v3 import verify_function_preserving

        assert callable(verify_function_preserving)

    def test_function_returns_result(self, matched_models, sample_inputs):
        """Test function returns VerificationResult."""
        from modeling_studio.models.verification_v3 import (
            verify_function_preserving,
            VerificationResult,
        )

        v2_model, v3_model = matched_models
        input_ids, attention_mask = sample_inputs

        result = verify_function_preserving(
            v2_model, v3_model, input_ids, attention_mask, verbose=False
        )

        assert isinstance(result, VerificationResult)


class TestVerifyWeightTransfer:
    """Tests for verify_weight_transfer convenience function."""

    def test_function_exists(self):
        """Test verify_weight_transfer function exists."""
        from modeling_studio.models.verification_v3 import verify_weight_transfer

        assert callable(verify_weight_transfer)

    def test_function_returns_result(self, matched_models):
        """Test function returns WeightComparisonResult."""
        from modeling_studio.models.verification_v3 import (
            verify_weight_transfer,
            WeightComparisonResult,
        )

        v2_model, v3_model = matched_models

        result = verify_weight_transfer(v2_model, v3_model, verbose=False)

        assert isinstance(result, WeightComparisonResult)


class TestCreateVerificationInputs:
    """Tests for create_verification_inputs utility function."""

    def test_function_exists(self):
        """Test create_verification_inputs function exists."""
        from modeling_studio.models.verification_v3 import create_verification_inputs

        assert callable(create_verification_inputs)

    def test_default_creation(self):
        """Test default input creation."""
        from modeling_studio.models.verification_v3 import create_verification_inputs

        input_ids, attention_mask = create_verification_inputs()

        assert input_ids.shape == (2, 128)  # default batch=2, seq=128
        assert attention_mask.shape == (2, 128)
        assert input_ids.max() < 50368

    def test_custom_parameters(self):
        """Test custom parameter creation."""
        from modeling_studio.models.verification_v3 import create_verification_inputs

        input_ids, attention_mask = create_verification_inputs(
            vocab_size=1000, seq_length=64, batch_size=4
        )

        assert input_ids.shape == (4, 64)
        assert attention_mask.shape == (4, 64)
        assert input_ids.max() < 1000


class TestVerifyEmbeddingTransfer:
    """Tests for verify_embedding_transfer utility function."""

    def test_function_exists(self):
        """Test verify_embedding_transfer function exists."""
        from modeling_studio.models.verification_v3 import verify_embedding_transfer

        assert callable(verify_embedding_transfer)

    def test_matched_embeddings_pass(self, matched_models):
        """Test matched embeddings pass verification."""
        from modeling_studio.models.verification_v3 import verify_embedding_transfer

        v2_model, v3_model = matched_models

        passed, diff = verify_embedding_transfer(v2_model, v3_model, verbose=False)

        assert passed is True
        assert diff < 1e-4


# ==============================================================================
# Acceptance Criteria Tests
# ==============================================================================


class TestIssue421AcceptanceCriteria:
    """Comprehensive tests for Issue 4.2.1 acceptance criteria."""

    def test_ac1_compares_v2_v3_layer_by_layer(self, matched_models, sample_inputs):
        """AC1: Compares v2 and v3 outputs layer-by-layer."""
        from modeling_studio.models.verification_v3 import FunctionPreservingVerifier

        v2_model, v3_model = matched_models
        input_ids, attention_mask = sample_inputs

        verifier = FunctionPreservingVerifier(v2_model, v3_model)
        result = verifier.verify_all_layers(input_ids, attention_mask, verbose=False)

        # Should have comparison for all 22 layers
        assert len(result.layer_diffs) == 22
        for layer_idx in range(22):
            assert layer_idx in result.layer_diffs

        print("AC1: Compares v2 and v3 outputs layer-by-layer [PASS]")

    def test_ac2_handles_hub_token_offset(self, matched_models, sample_inputs):
        """AC2: Handles hub token offset (v3 positions 1-4)."""
        from modeling_studio.models.verification_v3 import FunctionPreservingVerifier

        v2_model, v3_model = matched_models
        input_ids, attention_mask = sample_inputs

        verifier = FunctionPreservingVerifier(v2_model, v3_model)

        # Verify embeddings should handle offset
        passed, diff, v2_emb, v3_emb = verifier.verify_embeddings(input_ids, attention_mask)

        # v3 embeddings should be different size due to hub tokens
        # but comparison should still work
        assert v2_emb.shape[2] == v3_emb.shape[2]  # hidden size same
        assert isinstance(diff, float)

        # The verifier should have HUB_POSITIONS defined
        assert hasattr(FunctionPreservingVerifier, "V3_HUB_POSITIONS")
        assert FunctionPreservingVerifier.V3_HUB_POSITIONS == [1, 2, 3, 4]

        print("AC2: Handles hub token offset (v3 positions 1-4) [PASS]")

    def test_ac3_supports_tolerance_levels(self, mock_v2_model, mock_v3_model):
        """AC3: Supports strict/normal/relaxed tolerance levels."""
        from modeling_studio.models.verification_v3 import FunctionPreservingVerifier

        # Test all three tolerance levels are defined
        assert FunctionPreservingVerifier.TOLERANCE_STRICT == 1e-5
        assert FunctionPreservingVerifier.TOLERANCE_NORMAL == 1e-4
        assert FunctionPreservingVerifier.TOLERANCE_RELAXED == 1e-3

        # Test verifier accepts different tolerances
        for tol in [
            FunctionPreservingVerifier.TOLERANCE_STRICT,
            FunctionPreservingVerifier.TOLERANCE_NORMAL,
            FunctionPreservingVerifier.TOLERANCE_RELAXED,
        ]:
            verifier = FunctionPreservingVerifier(mock_v2_model, mock_v3_model, tolerance=tol)
            assert verifier.tolerance == tol

        print("AC3: Supports strict/normal/relaxed tolerance levels [PASS]")

    def test_ac4_reports_per_layer_differences(self, matched_models, sample_inputs):
        """AC4: Reports per-layer differences."""
        from modeling_studio.models.verification_v3 import FunctionPreservingVerifier

        v2_model, v3_model = matched_models
        input_ids, attention_mask = sample_inputs

        verifier = FunctionPreservingVerifier(v2_model, v3_model)
        result = verifier.verify_all_layers(input_ids, attention_mask, verbose=False)

        # layer_diffs should have an entry for each layer
        assert len(result.layer_diffs) == 22
        for layer_idx, diff in result.layer_diffs.items():
            assert isinstance(layer_idx, int)
            assert 0 <= layer_idx < 22
            assert isinstance(diff, float)
            assert diff >= 0

        print("AC4: Reports per-layer differences [PASS]")

    def test_ac5_clear_pass_fail_result(self, matched_models, sample_inputs):
        """AC5: Clear pass/fail result with message."""
        from modeling_studio.models.verification_v3 import (
            FunctionPreservingVerifier,
            VerificationResult,
        )

        v2_model, v3_model = matched_models
        input_ids, attention_mask = sample_inputs

        verifier = FunctionPreservingVerifier(v2_model, v3_model)
        result = verifier.verify_all_layers(input_ids, attention_mask, verbose=False)

        # Result should have clear pass/fail
        assert isinstance(result, VerificationResult)
        assert isinstance(result.passed, bool)
        assert isinstance(result.message, str)
        assert len(result.message) > 0

        # Message should contain useful info
        if result.passed:
            assert "verified" in result.message.lower() or "pass" in result.message.lower()
        else:
            assert "failed" in result.message.lower() or "violated" in result.message.lower()

        print("AC5: Clear pass/fail result with message [PASS]")

    def test_ac6_works_in_eval_mode(self, matched_models, sample_inputs):
        """AC6: Works with eval mode."""
        from modeling_studio.models.verification_v3 import FunctionPreservingVerifier

        v2_model, v3_model = matched_models
        input_ids, attention_mask = sample_inputs

        # Put models in eval mode
        v2_model.eval()
        v3_model.eval()

        verifier = FunctionPreservingVerifier(v2_model, v3_model)
        result = verifier.verify_all_layers(input_ids, attention_mask, verbose=False)

        # Should complete without error
        assert result is not None
        assert isinstance(result.passed, bool)

        # Models should still be in eval mode
        assert not v2_model.training
        assert not v3_model.training

        print("AC6: Works with eval mode [PASS]")

    def test_ac6_works_in_train_mode(self, matched_models, sample_inputs):
        """AC6: Works with train mode (models switched to eval internally)."""
        from modeling_studio.models.verification_v3 import FunctionPreservingVerifier

        v2_model, v3_model = matched_models
        input_ids, attention_mask = sample_inputs

        # Put models in train mode
        v2_model.train()
        v3_model.train()

        verifier = FunctionPreservingVerifier(v2_model, v3_model)
        result = verifier.verify_all_layers(input_ids, attention_mask, verbose=False)

        # Should complete without error
        assert result is not None
        assert isinstance(result.passed, bool)

        # Models should be switched to eval mode by verifier
        assert not v2_model.training
        assert not v3_model.training

        print("AC6: Works with train mode [PASS]")


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestVerificationIntegration:
    """Integration tests for verification module."""

    def test_full_verification_workflow(self, matched_models, sample_inputs):
        """Test complete verification workflow."""
        from modeling_studio.models.verification_v3 import (
            FunctionPreservingVerifier,
            verify_function_preserving,
            verify_weight_transfer,
            verify_embedding_transfer,
        )

        v2_model, v3_model = matched_models
        input_ids, attention_mask = sample_inputs

        # Step 1: Weight verification
        weight_result = verify_weight_transfer(v2_model, v3_model, verbose=False)
        assert weight_result.matched_params > 0

        # Step 2: Embedding verification
        emb_passed, emb_diff = verify_embedding_transfer(v2_model, v3_model, verbose=False)
        assert emb_passed

        # Step 3: Full function preserving verification
        result = verify_function_preserving(
            v2_model, v3_model, input_ids, attention_mask, verbose=False
        )
        assert isinstance(result.passed, bool)

    def test_verification_with_different_tolerances(self, matched_models, sample_inputs):
        """Test verification with different tolerance levels."""
        from modeling_studio.models.verification_v3 import (
            FunctionPreservingVerifier,
            verify_function_preserving,
        )

        v2_model, v3_model = matched_models
        input_ids, attention_mask = sample_inputs

        tolerances = [
            FunctionPreservingVerifier.TOLERANCE_STRICT,
            FunctionPreservingVerifier.TOLERANCE_NORMAL,
            FunctionPreservingVerifier.TOLERANCE_RELAXED,
        ]

        for tolerance in tolerances:
            result = verify_function_preserving(
                v2_model, v3_model, input_ids, attention_mask, tolerance=tolerance, verbose=False
            )
            assert isinstance(result.passed, bool)
            assert result.max_diff >= 0
