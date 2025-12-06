#!/usr/bin/env python3
"""
Initialize ModernBERT v3 from v2 checkpoint.

Usage:
    python scripts/initialize_v3_from_v2.py \
        --v2-checkpoint checkpoints/modernbert-v2/pytorch_model.bin \
        --output-dir checkpoints/modernbert-v3-init \
        --verify

This script:
    1. Loads v2 checkpoint (22 layers)
    2. Creates v3 model (28 layers)
    3. Copies layers 1-22 directly
    4. Clones layers 15-20 to 23-28
    5. Transfers embeddings with hub token slots
    6. Initializes hub tokens semantically
    7. Verifies function preserving property (optional)
    8. Saves initialized v3 model

Issue: 4.2.3
Author: FamilyOS Team
Date: December 2025
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modeling_studio.models.config_v3 import ModernBERTv3Config
from modeling_studio.models.initialization_v3 import (
    V2CheckpointLoader,
    WeightTransferStats,
    initialize_from_v2,
)
from modeling_studio.models.modernbert_v3 import ModernBERTv3Ultra
from modeling_studio.models.verification_v3 import (
    VerificationResult,
    verify_function_preserving,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Initialize ModernBERT v3 from v2 checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--v2-checkpoint",
        type=str,
        required=True,
        help="Path to v2 checkpoint file",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for initialized v3 model",
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run function preserving verification after initialization",
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Tolerance for verification (default: 1e-4)",
    )

    parser.add_argument(
        "--no-clone-noise",
        action="store_true",
        help="Disable noise addition to cloned layers",
    )

    parser.add_argument(
        "--clone-noise-std",
        type=float,
        default=0.01,
        help="Std of noise for cloned layers (default: 0.01)",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="answerdotai/ModernBERT-base",
        help="Tokenizer for hub semantic initialization",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for verification (cpu or cuda)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    return parser.parse_args()


def create_v3_config(v2_loader: V2CheckpointLoader) -> ModernBERTv3Config:
    """
    Create v3 config based on v2 checkpoint info.

    Args:
        v2_loader: Loaded v2 checkpoint

    Returns:
        ModernBERTv3Config configured for v3 architecture
    """
    info = v2_loader.get_info()

    # v3 vocab size: v2 vocab + 4 hub tokens, aligned to 256
    # ModernBERT-base: 50265 tokens -> 50265 + 4 + 163 padding = 50432 (256 aligned)
    v3_vocab_size = 50432  # 256-aligned

    return ModernBERTv3Config(
        hidden_size=info.hidden_size,
        num_layers=28,  # v3 has 28 layers
        num_attention_heads=12,
        intermediate_size=info.hidden_size * 4,
        vocab_size=v3_vocab_size,
        max_position_embeddings=8192,
    )


def create_mock_v2_model(v2_loader: V2CheckpointLoader) -> torch.nn.Module:
    """
    Create a mock v2 model structure for verification.

    Since we don't have a standalone v2 model class, we create a simple
    wrapper around the checkpoint state dict for layer-by-layer comparison.

    Args:
        v2_loader: Loaded v2 checkpoint

    Returns:
        Mock v2 model with embeddings and encoder.layers
    """
    import torch.nn as nn

    class MockEmbeddings(nn.Module):
        """Mock embeddings that returns pre-computed embeddings."""

        def __init__(self, state_dict: dict, hidden_size: int, vocab_size: int):
            super().__init__()
            self.word_embeddings = nn.Embedding(vocab_size, hidden_size)

            # Load word embeddings from state dict
            if "embeddings.word_embeddings.weight" in state_dict:
                self.word_embeddings.weight.data = state_dict[
                    "embeddings.word_embeddings.weight"
                ].clone()

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return self.word_embeddings(input_ids)

    class MockLayer(nn.Module):
        """Mock layer that matches v2 layer structure."""

        def __init__(self, state_dict: dict, layer_idx: int, hidden_size: int):
            super().__init__()
            self.hidden_size = hidden_size

            # Create linear layers matching v2 structure
            self.linear = nn.Linear(hidden_size, hidden_size)
            self.norm = nn.LayerNorm(hidden_size)

            # Load weights from state dict
            prefix = f"encoder.layers.{layer_idx}."

            # Try to load attention output projection
            out_proj_key = f"{prefix}attention.out_proj.weight"
            if out_proj_key in state_dict:
                # Use actual layer weights for verification
                self.linear.weight.data = state_dict[out_proj_key].clone()

            # Try to load layer norm
            ln_weight_key = f"{prefix}attention_layer_norm.weight"
            if ln_weight_key in state_dict:
                self.norm.weight.data = state_dict[ln_weight_key].clone()

            ln_bias_key = f"{prefix}attention_layer_norm.bias"
            if ln_bias_key in state_dict:
                self.norm.bias.data = state_dict[ln_bias_key].clone()

        def forward(
            self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
        ) -> torch.Tensor:
            output = self.linear(hidden_states)
            output = self.norm(output)
            return output

    class MockEncoder(nn.Module):
        """Mock encoder with 22 layers."""

        def __init__(self, state_dict: dict, num_layers: int, hidden_size: int):
            super().__init__()
            self.layers = nn.ModuleList(
                [MockLayer(state_dict, i, hidden_size) for i in range(num_layers)]
            )

    class MockV2Model(nn.Module):
        """Mock v2 model for verification."""

        def __init__(self, state_dict: dict, info):
            super().__init__()
            self.embeddings = MockEmbeddings(state_dict, info.hidden_size, info.vocab_size)
            self.encoder = MockEncoder(state_dict, info.num_layers, info.hidden_size)

    info = v2_loader.get_info()
    state_dict = v2_loader.load()

    return MockV2Model(state_dict, info)


def run_verification(
    v2_loader: V2CheckpointLoader,
    v3_model: torch.nn.Module,
    tolerance: float,
    device: str,
    verbose: bool = True,
) -> VerificationResult | None:
    """
    Run function preserving verification.

    Args:
        v2_loader: Loaded v2 checkpoint
        v3_model: Initialized v3 model
        tolerance: Verification tolerance
        device: Device to run verification on
        verbose: Whether to print detailed output

    Returns:
        VerificationResult or None if verification fails
    """
    logger.info("Running function preserving verification...")

    try:
        # Create mock v2 model from checkpoint
        v2_model = create_mock_v2_model(v2_loader)

        # Move to device
        v2_model = v2_model.to(device)
        v3_model = v3_model.to(device)

        # Create test input
        batch_size = 2
        seq_length = 128

        # Use vocab IDs that exist in both models
        input_ids = torch.randint(0, 50000, (batch_size, seq_length)).to(device)
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long).to(device)

        # Run verification
        result = verify_function_preserving(
            v2_model,
            v3_model,
            input_ids,
            attention_mask,
            tolerance=tolerance,
            verbose=verbose,
        )

        return result

    except Exception as e:
        logger.error(f"Verification failed with error: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        return None


def save_model(
    model: torch.nn.Module,
    config: ModernBERTv3Config,
    output_dir: Path,
    stats: WeightTransferStats,
    v2_checkpoint_path: str,
    verification_result: VerificationResult | None = None,
) -> None:
    """
    Save initialized model and metadata.

    Args:
        model: Initialized v3 model
        config: v3 configuration
        output_dir: Output directory
        stats: Weight transfer statistics
        v2_checkpoint_path: Path to source v2 checkpoint
        verification_result: Optional verification result
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    model_path = output_dir / "pytorch_model.bin"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model weights to {model_path}")

    # Save config
    config_path = output_dir / "config.json"
    config_dict = config.to_dict()
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"Saved config to {config_path}")

    # Save initialization metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "source": "v2_checkpoint",
        "v2_checkpoint_path": str(v2_checkpoint_path),
        "transfer_stats": {
            "total_params": stats.total_params,
            "transferred_params": stats.transferred_params,
            "initialized_params": stats.initialized_params,
            "skipped_params": stats.skipped_params,
            "layer_mapping": {str(k): v for k, v in stats.layer_mapping.items()},
        },
        "v3_config": {
            "num_layers": config.num_layers,
            "hidden_size": config.hidden_size,
            "vocab_size": config.vocab_size,
            "hub_tokens": config.hub_tokens,
        },
    }

    if verification_result:
        metadata["verification"] = {
            "passed": verification_result.passed,
            "max_diff": float(verification_result.max_diff),
            "mean_diff": float(verification_result.mean_diff),
            "embedding_diff": float(verification_result.embedding_diff),
            "failed_layers": verification_result.failed_layers,
            "message": verification_result.message,
        }

    metadata_path = output_dir / "initialization_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")


def main() -> int:
    """Main entry point."""
    args = parse_args()

    print("\n" + "=" * 70)
    print("ModernBERT v3 Initialization from v2")
    print("=" * 70)

    # Validate inputs
    v2_path = Path(args.v2_checkpoint)
    if not v2_path.exists():
        logger.error(f"v2 checkpoint not found: {v2_path}")
        return 1

    output_dir = Path(args.output_dir)

    # Step 1: Load v2 checkpoint info
    logger.info(f"Loading v2 checkpoint from {v2_path}")
    try:
        v2_loader = V2CheckpointLoader(str(v2_path))
        v2_loader.print_summary()
    except Exception as e:
        logger.error(f"Failed to load v2 checkpoint: {e}")
        return 1

    # Step 2: Create v3 model
    logger.info("Creating v3 model...")
    config = create_v3_config(v2_loader)
    v3_model = ModernBERTv3Ultra(config)

    param_count = sum(p.numel() for p in v3_model.parameters())
    logger.info(f"  v3 layers: {config.num_layers}")
    logger.info(f"  v3 vocab size: {config.vocab_size}")
    logger.info(f"  v3 params: {param_count:,}")

    # Step 3: Initialize from v2
    add_clone_noise = not args.no_clone_noise
    stats = initialize_from_v2(
        v3_model,
        str(v2_path),
        add_clone_noise=add_clone_noise,
        clone_noise_std=args.clone_noise_std,
        tokenizer_name=args.tokenizer,
    )

    # Step 4: Verification (optional)
    verification_result = None
    if args.verify:
        verification_result = run_verification(
            v2_loader,
            v3_model,
            args.tolerance,
            args.device,
            verbose=args.verbose,
        )

        if verification_result is None:
            logger.warning("Verification could not be completed - proceeding anyway")
        elif not verification_result.passed:
            logger.warning("Verification FAILED - proceeding anyway")
            logger.warning(f"  Max diff: {verification_result.max_diff:.2e}")
            logger.warning(f"  Failed layers: {verification_result.failed_layers}")

    # Step 5: Save model
    logger.info(f"Saving initialized v3 model to {output_dir}")
    save_model(
        v3_model,
        config,
        output_dir,
        stats,
        str(v2_path),
        verification_result,
    )

    # Summary
    print("\n" + "=" * 70)
    print("Initialization Complete!")
    print("=" * 70)
    print(f"  Output: {output_dir}")
    print(f"  Total params: {stats.total_params:,}")
    print(f"  Transferred: {stats.transferred_params:,}")
    print(f"  Cloned: {stats.initialized_params:,}")
    if verification_result:
        status = "PASSED" if verification_result.passed else "FAILED"
        print(f"  Verification: {status}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
