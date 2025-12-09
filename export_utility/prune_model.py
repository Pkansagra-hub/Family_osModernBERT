"""
Model Pruning Utilities for ModernBERT Multitask.

Provides structured and unstructured pruning to reduce model size
while maintaining capability quality.

Pruning Methods:
- Magnitude-based pruning (unstructured)
- Movement pruning (training-aware)
- Structured pruning (attention heads, layers)
- Block pruning (groups of weights)

Usage:
    python prune_model.py --checkpoint-path /path/to/checkpoint \
        --output-path /path/to/output \
        --method magnitude \
        --sparsity 0.3 \
        --validate
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class PruningConfig:
    """Configuration for model pruning."""

    method: str = "magnitude"  # magnitude, movement, structured, block
    sparsity: float = 0.3  # Target sparsity (0.0 to 1.0)
    structured_target: str = "attention"  # attention, ffn, layer
    block_size: int = 64  # For block pruning
    prune_embeddings: bool = False  # Whether to prune embedding layers
    prune_heads: bool = True  # Whether to prune task heads
    gradual_steps: int = 1  # For gradual pruning
    importance_scores_path: Optional[str] = None  # For movement pruning
    validate_after: bool = True  # Validate model after pruning

    # Layer-specific configs
    min_layer_idx: int = 0  # Start pruning from this layer
    max_layer_idx: int = -1  # End pruning at this layer (-1 = all)
    skip_layers: List[int] = field(default_factory=list)  # Layers to skip


class PruningAnalyzer:
    """Analyze model for pruning opportunities."""

    def __init__(self, model: nn.Module):
        self.model = model

    def get_weight_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get weight statistics for each layer."""
        stats = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and len(param.shape) >= 2:
                weights = param.data.abs()
                stats[name] = {
                    "mean": weights.mean().item(),
                    "std": weights.std().item(),
                    "min": weights.min().item(),
                    "max": weights.max().item(),
                    "median": weights.median().item(),
                    "num_params": param.numel(),
                    "shape": list(param.shape),
                    "near_zero_10pct": (weights < weights.mean() * 0.1).float().mean().item(),
                    "near_zero_1pct": (weights < weights.mean() * 0.01).float().mean().item(),
                }
        return stats

    def get_attention_head_importance(self) -> Dict[str, List[float]]:
        """Estimate attention head importance based on weight norms."""
        head_importance = {}

        for name, param in self.model.named_parameters():
            if "attention" in name.lower() and "weight" in name:
                if len(param.shape) == 2:
                    # Assume multi-head attention weight
                    weights = param.data
                    # Try to detect number of heads
                    if weights.shape[0] % 12 == 0:  # Common head count
                        num_heads = 12
                        head_dim = weights.shape[0] // num_heads
                        heads = weights.view(num_heads, head_dim, -1)
                        importance = [heads[i].abs().mean().item() for i in range(num_heads)]
                        head_importance[name] = importance

        return head_importance

    def get_layer_importance(self) -> Dict[str, float]:
        """Estimate layer importance based on gradient flow potential."""
        layer_importance = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Use Frobenius norm as importance proxy
                layer_importance[name] = param.data.norm().item()

        return layer_importance

    def recommend_pruning_config(
        self,
        target_reduction: float = 0.3
    ) -> PruningConfig:
        """Recommend pruning configuration based on analysis."""
        stats = self.get_weight_statistics()

        # Find layers with high near-zero weight ratios
        prunable_layers = []
        for name, layer_stats in stats.items():
            if layer_stats["near_zero_10pct"] > 0.2:
                prunable_layers.append(name)

        # Determine best method
        avg_near_zero = sum(
            s["near_zero_10pct"] for s in stats.values()
        ) / len(stats)

        if avg_near_zero > 0.3:
            method = "magnitude"  # Many near-zero weights
        else:
            method = "block"  # Need structured approach

        config = PruningConfig(
            method=method,
            sparsity=target_reduction,
            prune_embeddings=False,
            validate_after=True,
        )

        logger.info(f"Recommended pruning: {method} with {target_reduction:.0%} sparsity")
        logger.info(f"Found {len(prunable_layers)} highly prunable layers")

        return config


class ModelPruner:
    """Main pruning engine."""

    def __init__(
        self,
        model: nn.Module,
        config: PruningConfig,
        device: str = "cuda"
    ):
        self.model = model
        self.config = config
        self.device = device
        self.pruned_params: Dict[str, float] = {}
        self.original_size: int = 0
        self.pruned_size: int = 0

    def _get_prunable_modules(self) -> List[Tuple[str, nn.Module]]:
        """Get list of modules that can be pruned."""
        prunable = []

        for name, module in self.model.named_modules():
            # Skip embedding layers if configured
            if not self.config.prune_embeddings and "embedding" in name.lower():
                continue

            # Skip task heads if configured
            if not self.config.prune_heads and "head" in name.lower():
                continue

            # Check layer index constraints
            if "layer" in name.lower():
                try:
                    # Extract layer number
                    parts = name.split(".")
                    for i, part in enumerate(parts):
                        if part == "layer" and i + 1 < len(parts):
                            layer_idx = int(parts[i + 1])
                            if layer_idx < self.config.min_layer_idx:
                                continue
                            if self.config.max_layer_idx >= 0 and layer_idx > self.config.max_layer_idx:
                                continue
                            if layer_idx in self.config.skip_layers:
                                continue
                except (ValueError, IndexError):
                    pass

            # Only prune Linear layers
            if isinstance(module, nn.Linear):
                prunable.append((name, module))

        return prunable

    def magnitude_pruning(self) -> Dict[str, Any]:
        """Apply magnitude-based unstructured pruning."""
        logger.info(f"Applying magnitude pruning with {self.config.sparsity:.0%} sparsity")

        prunable = self._get_prunable_modules()
        results = {"pruned_layers": [], "total_params_pruned": 0}

        for name, module in tqdm(prunable, desc="Pruning layers"):
            original_nonzero = module.weight.data.nonzero().size(0)

            # Apply L1 unstructured pruning
            prune.l1_unstructured(module, name="weight", amount=self.config.sparsity)

            # Make pruning permanent
            prune.remove(module, "weight")

            new_nonzero = module.weight.data.nonzero().size(0)
            params_pruned = original_nonzero - new_nonzero

            self.pruned_params[name] = {
                "original": original_nonzero,
                "remaining": new_nonzero,
                "pruned": params_pruned,
                "sparsity": 1 - (new_nonzero / max(original_nonzero, 1))
            }

            results["pruned_layers"].append(name)
            results["total_params_pruned"] += params_pruned

        return results

    def structured_pruning(self) -> Dict[str, Any]:
        """Apply structured pruning (entire neurons/heads)."""
        logger.info(f"Applying structured pruning targeting {self.config.structured_target}")

        prunable = self._get_prunable_modules()
        results = {"pruned_layers": [], "total_params_pruned": 0}

        for name, module in tqdm(prunable, desc="Structured pruning"):
            # Filter based on target
            if self.config.structured_target == "attention" and "attention" not in name.lower():
                continue
            if self.config.structured_target == "ffn" and "intermediate" not in name.lower():
                continue

            original_size = module.weight.numel()

            # Prune entire output neurons based on L2 norm
            prune.ln_structured(
                module,
                name="weight",
                amount=self.config.sparsity,
                n=2,  # L2 norm
                dim=0  # Prune output neurons
            )

            # Make permanent
            prune.remove(module, "weight")

            # Count remaining non-zero rows
            row_norms = module.weight.data.norm(dim=1)
            active_neurons = (row_norms > 0).sum().item()
            total_neurons = module.weight.size(0)

            self.pruned_params[name] = {
                "original_neurons": total_neurons,
                "active_neurons": active_neurons,
                "pruned_neurons": total_neurons - active_neurons,
                "sparsity": 1 - (active_neurons / total_neurons)
            }

            results["pruned_layers"].append(name)
            results["total_params_pruned"] += original_size - module.weight.nonzero().size(0)

        return results

    def block_pruning(self) -> Dict[str, Any]:
        """Apply block-wise pruning for better hardware efficiency."""
        logger.info(f"Applying block pruning with block size {self.config.block_size}")

        prunable = self._get_prunable_modules()
        results = {"pruned_layers": [], "total_blocks_pruned": 0}
        block_size = self.config.block_size

        for name, module in tqdm(prunable, desc="Block pruning"):
            weight = module.weight.data
            H, W = weight.shape

            # Pad if necessary
            pad_h = (block_size - H % block_size) % block_size
            pad_w = (block_size - W % block_size) % block_size

            if pad_h > 0 or pad_w > 0:
                weight = torch.nn.functional.pad(weight, (0, pad_w, 0, pad_h))

            # Reshape into blocks
            new_H, new_W = weight.shape
            blocks = weight.view(
                new_H // block_size, block_size,
                new_W // block_size, block_size
            ).permute(0, 2, 1, 3)

            # Calculate block importance (L2 norm)
            block_importance = blocks.norm(dim=(2, 3))

            # Determine threshold for pruning
            flat_importance = block_importance.flatten()
            k = int(flat_importance.numel() * self.config.sparsity)
            if k > 0:
                threshold = torch.kthvalue(flat_importance, k).values.item()

                # Create block mask
                block_mask = block_importance > threshold

                # Apply mask
                mask = block_mask.unsqueeze(2).unsqueeze(3).expand_as(blocks)
                blocks = blocks * mask.float()

                # Reshape back
                weight = blocks.permute(0, 2, 1, 3).reshape(new_H, new_W)

                # Remove padding
                weight = weight[:H, :W]
                module.weight.data = weight

                blocks_pruned = (~block_mask).sum().item()
                results["total_blocks_pruned"] += blocks_pruned

                self.pruned_params[name] = {
                    "total_blocks": block_mask.numel(),
                    "pruned_blocks": blocks_pruned,
                    "block_sparsity": blocks_pruned / block_mask.numel()
                }

            results["pruned_layers"].append(name)

        return results

    def gradual_pruning(self, target_sparsity: float, num_steps: int) -> List[Dict[str, Any]]:
        """Apply gradual magnitude pruning over multiple steps."""
        logger.info(f"Applying gradual pruning: {target_sparsity:.0%} over {num_steps} steps")

        step_results = []
        current_sparsity = 0.0
        sparsity_increment = target_sparsity / num_steps

        for step in range(num_steps):
            current_sparsity += sparsity_increment
            logger.info(f"Step {step + 1}/{num_steps}: target sparsity {current_sparsity:.0%}")

            # Update config and prune
            original_sparsity = self.config.sparsity
            self.config.sparsity = sparsity_increment  # Incremental

            result = self.magnitude_pruning()
            result["step"] = step + 1
            result["cumulative_sparsity"] = current_sparsity
            step_results.append(result)

            self.config.sparsity = original_sparsity

        return step_results

    def prune(self) -> Dict[str, Any]:
        """Apply pruning based on configuration."""
        # Calculate original size
        self.original_size = sum(
            p.numel() for p in self.model.parameters()
        )

        # Apply pruning method
        if self.config.method == "magnitude":
            if self.config.gradual_steps > 1:
                results = self.gradual_pruning(
                    self.config.sparsity,
                    self.config.gradual_steps
                )
            else:
                results = self.magnitude_pruning()
        elif self.config.method == "structured":
            results = self.structured_pruning()
        elif self.config.method == "block":
            results = self.block_pruning()
        else:
            raise ValueError(f"Unknown pruning method: {self.config.method}")

        # Calculate final statistics
        total_nonzero = sum(
            (p != 0).sum().item() for p in self.model.parameters()
        )
        self.pruned_size = total_nonzero

        actual_sparsity = 1 - (self.pruned_size / self.original_size)

        summary = {
            "method": self.config.method,
            "target_sparsity": self.config.sparsity,
            "actual_sparsity": actual_sparsity,
            "original_params": self.original_size,
            "remaining_params": self.pruned_size,
            "pruned_params": self.original_size - self.pruned_size,
            "compression_ratio": self.original_size / max(self.pruned_size, 1),
            "layer_details": self.pruned_params,
        }

        if isinstance(results, list):
            summary["gradual_steps"] = results
        else:
            summary.update(results)

        return summary

    def save_pruned_model(
        self,
        output_path: str,
        source_checkpoint: str = None,
        save_sparse: bool = False,
        quantize: bool = False
    ) -> Dict[str, Any]:
        """Save the pruned model in checkpoint-compatible format."""
        import shutil
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        save_info = {"format": "pytorch", "files": []}

        # Copy config files from source checkpoint if provided
        if source_checkpoint:
            source_path = Path(source_checkpoint)
            files_to_copy = [
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "capabilities.json",
            ]
            for fname in files_to_copy:
                src_file = source_path / fname
                if src_file.exists():
                    shutil.copy2(src_file, output_path / fname)
                    save_info["files"].append(fname)
                    logger.info(f"Copied {fname} from source checkpoint")

        # Save model weights using safetensors for compatibility
        try:
            from safetensors.torch import save_file

            # Convert state dict to regular tensors (not sparse)
            state_dict = {}
            for name, param in self.model.state_dict().items():
                if param.is_sparse:
                    state_dict[name] = param.to_dense()
                else:
                    state_dict[name] = param

            save_file(state_dict, output_path / "model.safetensors")
            save_info["files"].append("model.safetensors")
            save_info["format"] = "safetensors"
            logger.info("Saved model as model.safetensors")
        except ImportError:
            # Fallback to pytorch format
            torch.save(self.model.state_dict(), output_path / "model_pruned.pt")
            save_info["files"].append("model_pruned.pt")
            logger.info("Saved model as model_pruned.pt (safetensors not available)")

        # Save pruning metadata
        metadata = {
            "config": {
                "method": self.config.method,
                "sparsity": self.config.sparsity,
                "structured_target": self.config.structured_target,
                "block_size": self.config.block_size,
            },
            "statistics": {
                "original_params": self.original_size,
                "remaining_params": self.pruned_size,
                "actual_sparsity": 1 - (self.pruned_size / self.original_size),
            },
            "layer_details": self.pruned_params,
        }

        with open(output_path / "pruning_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        save_info["files"].append("pruning_metadata.json")

        logger.info(f"Saved pruned model to {output_path}")
        return save_info


class PruningValidator:
    """Validate pruned model quality."""

    def __init__(
        self,
        original_model: nn.Module,
        pruned_model: nn.Module,
        device: str = "cuda"
    ):
        self.original_model = original_model.to(device)
        self.pruned_model = pruned_model.to(device)
        self.device = device

    def compare_outputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        capability: str
    ) -> Dict[str, float]:
        """Compare outputs between original and pruned models."""
        self.original_model.eval()
        self.pruned_model.eval()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            # Get original output
            orig_output = self.original_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                capability=capability
            )

            # Get pruned output
            pruned_output = self.pruned_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                capability=capability
            )

        # Handle different output types
        if isinstance(orig_output, dict):
            orig_logits = orig_output.get("logits", orig_output.get("embeddings"))
            pruned_logits = pruned_output.get("logits", pruned_output.get("embeddings"))
        elif hasattr(orig_output, "logits"):
            orig_logits = orig_output.logits
            pruned_logits = pruned_output.logits
        else:
            orig_logits = orig_output
            pruned_logits = pruned_output

        if orig_logits is None or pruned_logits is None:
            return {"error": "Could not extract logits"}

        # Calculate similarity metrics
        cos_sim = torch.nn.functional.cosine_similarity(
            orig_logits.flatten().unsqueeze(0),
            pruned_logits.flatten().unsqueeze(0)
        ).item()

        mse = torch.nn.functional.mse_loss(orig_logits, pruned_logits).item()

        # Check prediction agreement
        if orig_logits.dim() >= 2:
            orig_pred = orig_logits.argmax(dim=-1)
            pruned_pred = pruned_logits.argmax(dim=-1)
            agreement = (orig_pred == pruned_pred).float().mean().item()
        else:
            agreement = cos_sim

        return {
            "cosine_similarity": cos_sim,
            "mse": mse,
            "prediction_agreement": agreement,
        }

    def run_validation(
        self,
        test_samples: List[Dict[str, Any]],
        tokenizer: Any
    ) -> Dict[str, Any]:
        """Run full validation on test samples."""
        results = {"per_capability": {}, "overall": {}}
        all_similarities = []
        all_agreements = []

        for sample in tqdm(test_samples, desc="Validating"):
            text = sample.get("text", sample.get("input"))
            capability = sample.get("capability", "sentiment")

            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            comparison = self.compare_outputs(
                inputs["input_ids"],
                inputs["attention_mask"],
                capability
            )

            if "error" not in comparison:
                if capability not in results["per_capability"]:
                    results["per_capability"][capability] = []
                results["per_capability"][capability].append(comparison)
                all_similarities.append(comparison["cosine_similarity"])
                all_agreements.append(comparison["prediction_agreement"])

        # Calculate aggregated metrics
        for cap, comparisons in results["per_capability"].items():
            results["per_capability"][cap] = {
                "avg_cosine_similarity": sum(c["cosine_similarity"] for c in comparisons) / len(comparisons),
                "avg_mse": sum(c["mse"] for c in comparisons) / len(comparisons),
                "avg_prediction_agreement": sum(c["prediction_agreement"] for c in comparisons) / len(comparisons),
                "num_samples": len(comparisons),
            }

        if all_similarities:
            results["overall"] = {
                "avg_cosine_similarity": sum(all_similarities) / len(all_similarities),
                "avg_prediction_agreement": sum(all_agreements) / len(all_agreements),
                "quality_preserved": sum(all_agreements) / len(all_agreements) > 0.95,
            }

        return results


def load_model_for_pruning(checkpoint_path: str, device: str = "cuda"):
    """Load model from checkpoint for pruning."""
    checkpoint_path = Path(checkpoint_path)

    # Try to import our ModernBertMultiTaskModel
    try:
        from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

        # Use load_checkpoint which properly loads encoder + heads
        model = ModernBertMultiTaskModel.load_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
        )
        logger.info(f"Loaded ModernBertMultiTaskModel with capabilities: {[c.value for c in model.capabilities]}")

    except ImportError as e:
        logger.warning(f"Could not import ModernBertMultiTaskModel: {e}")
        # Fallback to transformers AutoModel
        from transformers import AutoModel
        model = AutoModel.from_pretrained(str(checkpoint_path))
        model = model.to(device)
        logger.info("Loaded model using transformers AutoModel")

    return model


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Model Pruning for ModernBERT Multitask")

    # Required arguments
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save pruned model"
    )

    # Pruning configuration
    parser.add_argument(
        "--method",
        type=str,
        default="magnitude",
        choices=["magnitude", "structured", "block"],
        help="Pruning method"
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.3,
        help="Target sparsity (0.0 to 1.0)"
    )
    parser.add_argument(
        "--structured-target",
        type=str,
        default="attention",
        choices=["attention", "ffn", "layer"],
        help="Target for structured pruning"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=64,
        help="Block size for block pruning"
    )
    parser.add_argument(
        "--gradual-steps",
        type=int,
        default=1,
        help="Number of gradual pruning steps"
    )

    # Options
    parser.add_argument(
        "--prune-embeddings",
        action="store_true",
        help="Also prune embedding layers"
    )
    parser.add_argument(
        "--prune-heads",
        action="store_true",
        default=True,
        help="Prune task-specific heads"
    )
    parser.add_argument(
        "--no-prune-heads",
        action="store_false",
        dest="prune_heads",
        help="Don't prune task-specific heads"
    )
    parser.add_argument(
        "--save-sparse",
        action="store_true",
        help="Save in sparse tensor format"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate pruned model quality"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze model without pruning"
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )

    args = parser.parse_args()

    # Load model
    logger.info(f"Loading model from {args.checkpoint_path}")
    model = load_model_for_pruning(args.checkpoint_path, args.device)

    # Create config
    config = PruningConfig(
        method=args.method,
        sparsity=args.sparsity,
        structured_target=args.structured_target,
        block_size=args.block_size,
        prune_embeddings=args.prune_embeddings,
        prune_heads=args.prune_heads,
        gradual_steps=args.gradual_steps,
        validate_after=args.validate,
    )

    # Analyze model
    analyzer = PruningAnalyzer(model)

    if args.analyze_only:
        logger.info("Analyzing model...")
        stats = analyzer.get_weight_statistics()
        head_importance = analyzer.get_attention_head_importance()

        # Print analysis
        print("\n=== Weight Statistics ===")
        for name, layer_stats in list(stats.items())[:10]:
            print(f"\n{name}:")
            print(f"  Mean: {layer_stats['mean']:.6f}")
            print(f"  Near-zero (10%): {layer_stats['near_zero_10pct']:.2%}")
            print(f"  Near-zero (1%): {layer_stats['near_zero_1pct']:.2%}")

        print("\n=== Recommended Configuration ===")
        recommended = analyzer.recommend_pruning_config(args.sparsity)
        print(f"Method: {recommended.method}")
        print(f"Sparsity: {recommended.sparsity:.0%}")

        return

    # Apply pruning
    logger.info("Starting pruning...")
    pruner = ModelPruner(model, config, args.device)
    results = pruner.prune()

    # Print results
    print("\n" + "=" * 60)
    print("PRUNING RESULTS")
    print("=" * 60)
    print(f"Method: {results['method']}")
    print(f"Target Sparsity: {results['target_sparsity']:.0%}")
    print(f"Actual Sparsity: {results['actual_sparsity']:.2%}")
    print(f"Original Parameters: {results['original_params']:,}")
    print(f"Remaining Parameters: {results['remaining_params']:,}")
    print(f"Pruned Parameters: {results['pruned_params']:,}")
    print(f"Compression Ratio: {results['compression_ratio']:.2f}x")
    print("=" * 60)

    # Save model
    logger.info("Saving pruned model...")
    save_info = pruner.save_pruned_model(
        args.output_path,
        source_checkpoint=args.checkpoint_path,
    )

    print(f"\nSaved to: {args.output_path}")
    print(f"Files: {save_info['files']}")

    # Validation
    if args.validate:
        logger.info("Validation requested but no test data provided")
        logger.info("Use PruningValidator class with test samples for validation")


if __name__ == "__main__":
    main()
