"""
ModernBERT v3 Function Preserving Verification.

This module implements verification tests to ensure v3 produces identical
outputs to v2 for the first 22 layers when processing the same input.
This validates the "function preserving growth" property.

Function Preserving Property:
    For layers L1-L22, given identical input embeddings, the layer outputs
    should be identical within numerical precision tolerance.

Tolerance Levels:
    - Strict (1e-5): Bit-exact on same hardware
    - Normal (1e-4): Accounts for minor precision differences
    - Relaxed (1e-3): Allows floating point drift

Key Classes:
    - VerificationResult: Complete verification results with per-layer diffs
    - LayerComparisonResult: Single layer comparison result
    - FunctionPreservingVerifier: Main verifier class

Functions:
    - verify_function_preserving: Convenience function for verification
    - verify_weight_transfer: Quick weight-only verification
    - create_verification_inputs: Generate test inputs for verification

Author: FamilyOS Team
Date: December 2025
Version: 3.3
Epic: 4.2 Verification
Issue: 4.2.1 - Function Preserving Verification
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ==============================================================================
# Data Structures
# ==============================================================================


@dataclass
class LayerComparisonResult:
    """
    Comparison result for a single layer.

    Attributes:
        layer_idx: Layer index (0-based)
        v2_norm: L2 norm of v2 layer output
        v3_norm: L2 norm of v3 layer output
        diff_norm: Maximum absolute difference between outputs
        relative_diff: Relative difference (diff_norm / v2_norm)
        passed: Whether difference is within tolerance

    Example:
        >>> result = LayerComparisonResult(
        ...     layer_idx=0, v2_norm=100.0, v3_norm=100.0,
        ...     diff_norm=1e-6, relative_diff=1e-8, passed=True
        ... )
        >>> print(f"Layer {result.layer_idx}: {'PASS' if result.passed else 'FAIL'}")
        Layer 0: PASS
    """

    layer_idx: int
    v2_norm: float
    v3_norm: float
    diff_norm: float
    relative_diff: float
    passed: bool


@dataclass
class VerificationResult:
    """
    Complete results from function preserving verification.

    Attributes:
        passed: Whether all verifications passed
        max_diff: Maximum difference across all layers
        mean_diff: Mean difference across all layers
        layer_diffs: Dict mapping layer index to max difference
        embedding_diff: Difference in embedding outputs
        failed_layers: List of layer indices that failed verification
        message: Human-readable result summary

    Example:
        >>> result = verify_function_preserving(v2_model, v3_model, inputs, mask)
        >>> if result.passed:
        ...     print("Function preserving property verified!")
        >>> else:
        ...     print(f"Failed layers: {result.failed_layers}")
    """

    passed: bool
    max_diff: float
    mean_diff: float
    layer_diffs: Dict[int, float] = field(default_factory=dict)
    embedding_diff: float = 0.0
    failed_layers: List[int] = field(default_factory=list)
    message: str = ""


@dataclass
class WeightComparisonResult:
    """
    Result of weight-level comparison between v2 and v3.

    Attributes:
        passed: Whether all weight comparisons passed
        matched_params: Number of parameters that matched
        mismatched_params: Number of parameters that differed
        missing_in_v2: Keys present in v3 but not v2
        missing_in_v3: Keys present in v2 but not v3
        max_diff: Maximum weight difference
        layer_results: Per-layer weight comparison results
    """

    passed: bool
    matched_params: int = 0
    mismatched_params: int = 0
    missing_in_v2: List[str] = field(default_factory=list)
    missing_in_v3: List[str] = field(default_factory=list)
    max_diff: float = 0.0
    layer_results: Dict[int, Dict[str, Any]] = field(default_factory=dict)


# ==============================================================================
# Function Preserving Verifier
# ==============================================================================


class FunctionPreservingVerifier:
    """
    Verifies that v3 model preserves v2 function for layers 1-22.

    Function Preserving Property:
        For layers L1-L22, given identical input embeddings,
        the layer outputs should be identical (within numerical precision).

    The verification process:
        1. Forward input through both models' embeddings
        2. Compare embedding outputs (accounting for hub token offset)
        3. Forward through each layer sequentially
        4. Compare layer outputs at each step
        5. Report per-layer and aggregate statistics

    Tolerance Levels:
        - Strict (1e-5): Bit-exact on same hardware
        - Normal (1e-4): Accounts for precision differences
        - Relaxed (1e-3): Allows minor floating point drift

    Attributes:
        v2_model: Original v2 model (22 layers)
        v3_model: Initialized v3 model (28 layers)
        tolerance: Maximum allowed difference
        num_shared_layers: Number of layers shared between v2 and v3 (22)

    Example:
        >>> verifier = FunctionPreservingVerifier(v2_model, v3_model)
        >>> result = verifier.verify_all_layers(input_ids, attention_mask)
        >>> print(f"Verification {'PASSED' if result.passed else 'FAILED'}")
    """

    # Tolerance level constants
    TOLERANCE_STRICT = 1e-5
    TOLERANCE_NORMAL = 1e-4
    TOLERANCE_RELAXED = 1e-3

    # Number of shared layers between v2 and v3
    NUM_SHARED_LAYERS = 22

    # Hub token positions in v3 (not present in v2)
    V3_HUB_POSITIONS = [1, 2, 3, 4]  # [EMO], [MEM], [REL], [TASK]

    def __init__(
        self,
        v2_model: nn.Module,
        v3_model: nn.Module,
        tolerance: float = TOLERANCE_NORMAL,
    ):
        """
        Initialize the function preserving verifier.

        Args:
            v2_model: Original v2 model with 22 layers
            v3_model: Initialized v3 model with 28 layers
            tolerance: Maximum allowed difference between outputs.
                Default is TOLERANCE_NORMAL (1e-4).

        Raises:
            ValueError: If models don't have expected structure
        """
        self.v2_model = v2_model
        self.v3_model = v3_model
        self.tolerance = tolerance

        # Put both models in eval mode for consistent behavior
        self.v2_model.eval()
        self.v3_model.eval()

    def verify_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[bool, float, torch.Tensor, torch.Tensor]:
        """
        Verify embedding layer produces identical output.

        Compares embedding outputs between v2 and v3, accounting for
        the fact that v3 has hub tokens at positions 1-4.

        Note: Only compares shared positions (CLS at 0, text tokens).
        Hub token positions in v3 have no v2 equivalent.

        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Optional attention mask

        Returns:
            Tuple of (passed, max_diff, v2_embeddings, v3_embeddings)

        Example:
            >>> passed, diff, v2_emb, v3_emb = verifier.verify_embeddings(
            ...     input_ids, attention_mask
            ... )
            >>> print(f"Embedding diff: {diff:.2e}")
        """
        with torch.no_grad():
            # Get embeddings from both models
            v2_emb = self._get_embeddings(self.v2_model, input_ids)
            v3_emb = self._get_embeddings(self.v3_model, input_ids)

            # For comparison, we need to align positions:
            # v2: [CLS, tok1, tok2, ...]
            # v3: [CLS, EMO, MEM, REL, TASK, tok1, tok2, ...]
            #
            # We compare:
            # - CLS token: v2[0] vs v3[0]
            # - Text tokens: v2[1:] vs v3[5:]

            # Compare CLS token
            v2_cls = v2_emb[:, 0:1, :]
            v3_cls = v3_emb[:, 0:1, :]
            cls_diff = (v2_cls - v3_cls).abs().max().item()

            # Compare text tokens (accounting for hub token offset)
            v2_text = v2_emb[:, 1:, :]
            v3_text = v3_emb[:, 5:, :]

            # Align lengths (v3 may have different length due to hub tokens)
            min_len = min(v2_text.shape[1], v3_text.shape[1])
            if min_len > 0:
                text_diff = (v2_text[:, :min_len, :] - v3_text[:, :min_len, :]).abs().max().item()
            else:
                text_diff = 0.0

            max_diff = max(cls_diff, text_diff)
            passed = max_diff < self.tolerance

            return passed, max_diff, v2_emb, v3_emb

    def verify_layer(
        self,
        layer_idx: int,
        v2_hidden_states: torch.Tensor,
        v3_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> LayerComparisonResult:
        """
        Verify a single layer produces identical output.

        Forwards identical hidden states through both v2 and v3 layer
        and compares the outputs.

        Args:
            layer_idx: Layer index (0-21 for shared layers)
            v2_hidden_states: Input hidden states for v2 layer
            v3_hidden_states: Input hidden states for v3 layer
            attention_mask: Optional attention mask

        Returns:
            LayerComparisonResult with comparison details

        Raises:
            IndexError: If layer_idx >= NUM_SHARED_LAYERS
        """
        if layer_idx >= self.NUM_SHARED_LAYERS:
            raise IndexError(
                f"Layer index {layer_idx} exceeds shared layers "
                f"(0-{self.NUM_SHARED_LAYERS - 1})"
            )

        with torch.no_grad():
            # Get layers from both models
            v2_layer = self._get_layer(self.v2_model, layer_idx)
            v3_layer = self._get_layer(self.v3_model, layer_idx)

            # Forward through both layers
            v2_output = self._forward_layer(v2_layer, v2_hidden_states, attention_mask)
            v3_output = self._forward_layer(v3_layer, v3_hidden_states, attention_mask)

            # Compute statistics
            v2_norm = v2_output.norm().item()
            v3_norm = v3_output.norm().item()

            # Compute difference
            diff = (v2_output - v3_output).abs()
            diff_norm = diff.max().item()
            relative_diff = diff_norm / (v2_norm + 1e-8)

            passed = diff_norm < self.tolerance

            return LayerComparisonResult(
                layer_idx=layer_idx,
                v2_norm=v2_norm,
                v3_norm=v3_norm,
                diff_norm=diff_norm,
                relative_diff=relative_diff,
                passed=passed,
            )

    def verify_all_layers(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        verbose: bool = True,
    ) -> VerificationResult:
        """
        Verify all 22 shared layers produce identical outputs.

        This is the main verification method that:
        1. Verifies embedding outputs match
        2. Propagates hidden states through each layer
        3. Compares outputs at each layer
        4. Aggregates results into VerificationResult

        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Optional attention mask
            verbose: Whether to print progress (default True)

        Returns:
            VerificationResult with complete verification details

        Example:
            >>> result = verifier.verify_all_layers(input_ids, attention_mask)
            >>> if result.passed:
            ...     print("All layers verified!")
            >>> else:
            ...     print(f"Failed: {result.failed_layers}")
        """
        if verbose:
            print("\n" + "=" * 70)
            print("Function Preserving Verification")
            print("=" * 70)

        failed_layers: List[int] = []
        layer_diffs: Dict[int, float] = {}

        with torch.no_grad():
            # Step 1: Verify embeddings
            emb_passed, emb_diff, v2_hidden, v3_hidden = self.verify_embeddings(
                input_ids, attention_mask
            )

            if verbose:
                status = "PASS" if emb_passed else "FAIL"
                print(f"Embeddings: [{status}] diff={emb_diff:.2e}")

            # Step 2: Verify each shared layer
            if verbose:
                print("\nLayer-by-layer verification:")
                print("-" * 50)

            for layer_idx in range(self.NUM_SHARED_LAYERS):
                result = self.verify_layer(layer_idx, v2_hidden, v3_hidden, attention_mask)
                layer_diffs[layer_idx] = result.diff_norm

                if verbose:
                    status = "PASS" if result.passed else "FAIL"
                    print(
                        f"  L{layer_idx:2d}: [{status}] "
                        f"diff={result.diff_norm:.2e} "
                        f"rel={result.relative_diff:.2e}"
                    )

                if not result.passed:
                    failed_layers.append(layer_idx)

                # Propagate hidden states through both models
                v2_layer = self._get_layer(self.v2_model, layer_idx)
                v3_layer = self._get_layer(self.v3_model, layer_idx)

                v2_hidden = self._forward_layer(v2_layer, v2_hidden, attention_mask)
                v3_hidden = self._forward_layer(v3_layer, v3_hidden, attention_mask)

        # Compute summary statistics
        max_diff = max(layer_diffs.values()) if layer_diffs else 0.0
        mean_diff = sum(layer_diffs.values()) / len(layer_diffs) if layer_diffs else 0.0
        passed = len(failed_layers) == 0 and emb_passed

        # Create result
        result = VerificationResult(
            passed=passed,
            max_diff=max_diff,
            mean_diff=mean_diff,
            layer_diffs=layer_diffs,
            embedding_diff=emb_diff,
            failed_layers=failed_layers,
            message=self._create_message(passed, failed_layers, max_diff),
        )

        # Print summary
        if verbose:
            print("-" * 50)
            if passed:
                print(f"\nPASSED: All layers within tolerance ({self.tolerance:.0e})")
            else:
                print(f"\nFAILED: {len(failed_layers)} layers exceeded tolerance")
                print(f"   Failed layers: {failed_layers}")

            print(f"   Max diff: {max_diff:.2e}")
            print(f"   Mean diff: {mean_diff:.2e}")
            print("=" * 70)

        return result

    def verify_weights_only(
        self,
        verbose: bool = True,
    ) -> WeightComparisonResult:
        """
        Verify weights match between v2 and v3 for shared layers.

        This performs a direct weight comparison without forward passes,
        useful for verifying that weight transfer was successful.

        Args:
            verbose: Whether to print progress

        Returns:
            WeightComparisonResult with comparison details
        """
        if verbose:
            print("\n" + "=" * 70)
            print("Weight Transfer Verification")
            print("=" * 70)

        matched = 0
        mismatched = 0
        max_diff = 0.0
        layer_results: Dict[int, Dict[str, Any]] = {}

        with torch.no_grad():
            for layer_idx in range(self.NUM_SHARED_LAYERS):
                v2_layer = self._get_layer(self.v2_model, layer_idx)
                v3_layer = self._get_layer(self.v3_model, layer_idx)

                layer_matched = 0
                layer_mismatched = 0
                layer_max_diff = 0.0

                v2_state = dict(v2_layer.named_parameters())
                v3_state = dict(v3_layer.named_parameters())

                for name, v2_param in v2_state.items():
                    if name in v3_state:
                        v3_param = v3_state[name]
                        if v2_param.shape == v3_param.shape:
                            diff = (v2_param - v3_param).abs().max().item()
                            layer_max_diff = max(layer_max_diff, diff)
                            if diff < self.tolerance:
                                layer_matched += v2_param.numel()
                            else:
                                layer_mismatched += v2_param.numel()
                        else:
                            layer_mismatched += v2_param.numel()

                matched += layer_matched
                mismatched += layer_mismatched
                max_diff = max(max_diff, layer_max_diff)

                layer_results[layer_idx] = {
                    "matched": layer_matched,
                    "mismatched": layer_mismatched,
                    "max_diff": layer_max_diff,
                }

                if verbose:
                    status = "PASS" if layer_mismatched == 0 else "FAIL"
                    print(f"  L{layer_idx:2d}: [{status}] max_diff={layer_max_diff:.2e}")

        passed = mismatched == 0 and max_diff < self.tolerance

        if verbose:
            print("-" * 50)
            status = "PASSED" if passed else "FAILED"
            print(f"\n{status}: matched={matched:,}, mismatched={mismatched:,}")
            print(f"   Max weight diff: {max_diff:.2e}")
            print("=" * 70)

        return WeightComparisonResult(
            passed=passed,
            matched_params=matched,
            mismatched_params=mismatched,
            max_diff=max_diff,
            layer_results=layer_results,
        )

    def _get_embeddings(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Get embedding output from model."""
        if hasattr(model, "embeddings"):
            return model.embeddings(input_ids)
        elif hasattr(model, "embed_tokens"):
            return model.embed_tokens(input_ids)
        elif hasattr(model, "word_embeddings"):
            return model.word_embeddings(input_ids)
        else:
            raise AttributeError(
                "Cannot find embeddings in model. "
                "Expected attributes: embeddings, embed_tokens, or word_embeddings"
            )

    def _get_layer(self, model: nn.Module, layer_idx: int) -> nn.Module:
        """Get encoder layer by index."""
        if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
            return model.encoder.layers[layer_idx]
        elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            return model.encoder.layer[layer_idx]
        elif hasattr(model, "layers"):
            return model.layers[layer_idx]
        elif hasattr(model, "layer"):
            return model.layer[layer_idx]
        else:
            raise AttributeError(
                f"Cannot find layer {layer_idx} in model. "
                "Expected attributes: encoder.layers, encoder.layer, layers, or layer"
            )

    def _forward_layer(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward through a layer, handling different output formats."""
        # Try different forward signatures
        try:
            output = layer(hidden_states, attention_mask=attention_mask)
        except TypeError:
            try:
                output = layer(hidden_states, attention_mask)
            except TypeError:
                output = layer(hidden_states)

        # Handle tuple outputs (hidden_states, attention_weights, ...)
        if isinstance(output, tuple):
            return output[0]
        return output

    def _create_message(
        self,
        passed: bool,
        failed_layers: List[int],
        max_diff: float,
    ) -> str:
        """Create human-readable result message."""
        if passed:
            return (
                f"Function preserving property verified "
                f"(max_diff={max_diff:.2e}, tolerance={self.tolerance:.0e})"
            )
        else:
            return (
                f"Function preserving property VIOLATED: "
                f"{len(failed_layers)} layers failed "
                f"(max_diff={max_diff:.2e}, tolerance={self.tolerance:.0e})"
            )


# ==============================================================================
# Convenience Functions
# ==============================================================================


def verify_function_preserving(
    v2_model: nn.Module,
    v3_model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    tolerance: float = FunctionPreservingVerifier.TOLERANCE_NORMAL,
    verbose: bool = True,
) -> VerificationResult:
    """
    Verify v3 preserves v2 function for shared layers.

    Convenience function that creates a verifier and runs full verification.

    Args:
        v2_model: Original v2 model (22 layers)
        v3_model: Initialized v3 model (28 layers)
        input_ids: Test input token IDs
        attention_mask: Optional test attention mask
        tolerance: Maximum allowed difference (default 1e-4)
        verbose: Whether to print progress

    Returns:
        VerificationResult with complete verification details

    Example:
        >>> result = verify_function_preserving(v2_model, v3_model, inputs, mask)
        >>> assert result.passed, f"Verification failed: {result.message}"
    """
    verifier = FunctionPreservingVerifier(v2_model, v3_model, tolerance)
    return verifier.verify_all_layers(input_ids, attention_mask, verbose=verbose)


def verify_weight_transfer(
    v2_model: nn.Module,
    v3_model: nn.Module,
    tolerance: float = FunctionPreservingVerifier.TOLERANCE_NORMAL,
    verbose: bool = True,
) -> WeightComparisonResult:
    """
    Verify weights transferred correctly from v2 to v3.

    Quick verification that only checks weights, not forward passes.

    Args:
        v2_model: Original v2 model
        v3_model: Initialized v3 model
        tolerance: Maximum allowed difference
        verbose: Whether to print progress

    Returns:
        WeightComparisonResult with comparison details
    """
    verifier = FunctionPreservingVerifier(v2_model, v3_model, tolerance)
    return verifier.verify_weights_only(verbose=verbose)


def create_verification_inputs(
    vocab_size: int = 50368,
    seq_length: int = 128,
    batch_size: int = 2,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create random inputs for verification testing.

    Args:
        vocab_size: Vocabulary size (default 50368 for ModernBERT)
        seq_length: Sequence length
        batch_size: Batch size
        device: Target device

    Returns:
        Tuple of (input_ids, attention_mask)

    Example:
        >>> input_ids, attention_mask = create_verification_inputs()
        >>> result = verify_function_preserving(v2, v3, input_ids, attention_mask)
    """
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    attention_mask = torch.ones(batch_size, seq_length, device=device)
    return input_ids, attention_mask


def verify_embedding_transfer(
    v2_model: nn.Module,
    v3_model: nn.Module,
    tolerance: float = FunctionPreservingVerifier.TOLERANCE_NORMAL,
    verbose: bool = True,
) -> Tuple[bool, float]:
    """
    Verify word embeddings transferred correctly.

    Compares the first 50,368 embeddings (v2 vocab) between v2 and v3.
    Hub token embeddings (positions 50368-50371) are excluded.

    Args:
        v2_model: Original v2 model
        v3_model: Initialized v3 model
        tolerance: Maximum allowed difference
        verbose: Whether to print progress

    Returns:
        Tuple of (passed, max_diff)
    """
    with torch.no_grad():
        # Get embedding weights
        v2_emb = _get_embedding_weight(v2_model)
        v3_emb = _get_embedding_weight(v3_model)

        # Compare only v2 vocab portion (first 50368 tokens)
        v2_vocab_size = v2_emb.shape[0]
        v3_vocab = v3_emb[:v2_vocab_size, :]

        diff = (v2_emb - v3_vocab).abs().max().item()
        passed = diff < tolerance

        if verbose:
            status = "PASSED" if passed else "FAILED"
            print(f"Embedding transfer: [{status}] max_diff={diff:.2e}")

        return passed, diff


def _get_embedding_weight(model: nn.Module) -> torch.Tensor:
    """Get word embedding weight tensor from model."""
    if hasattr(model, "embeddings"):
        if hasattr(model.embeddings, "word_embeddings"):
            return model.embeddings.word_embeddings.weight
        elif hasattr(model.embeddings, "weight"):
            return model.embeddings.weight
    elif hasattr(model, "embed_tokens"):
        return model.embed_tokens.weight
    elif hasattr(model, "word_embeddings"):
        return model.word_embeddings.weight

    raise AttributeError("Cannot find embedding weights in model")
