# src/modeling_studio/trainers/gradient_utils_v3.py
"""
Gradient Clipping and Monitoring Utilities for v3 Training.

This module provides gradient clipping and monitoring utilities specifically
designed for Phase 0.5 training where the L22->L23 interface is particularly
sensitive to gradient explosions.

Key components:
    - GradientClipConfig: Configuration for gradient clipping
    - GradientStats: Statistics about gradients
    - GradientClipper: Main gradient clipping class
    - InterfaceGradientMonitor: Specialized L22->L23 monitor
    - clip_gradients(): Convenience function
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class GradientClipConfig:
    """
    Configuration for gradient clipping.

    Attributes:
        max_grad_norm: Maximum gradient norm for global clipping
        per_layer_clip: Whether to apply per-layer clipping
        interface_clip: Clip threshold for L23 (interface layer)
        family_clip: Clip threshold for L24-28 (Family band)
        feeder_clip: Clip threshold for L19-22 (Feeder band)
        log_grad_norms: Whether to log gradient norms
        log_every_n_steps: How often to log gradient stats
        explosion_threshold: Threshold for gradient explosion warning
        nan_check: Whether to check for NaN gradients
    """

    # Global gradient clipping
    max_grad_norm: float = 1.0

    # Per-layer gradient clipping (optional, more fine-grained)
    per_layer_clip: bool = False
    interface_clip: float = 0.5  # L23: tighter clip at interface
    family_clip: float = 1.0  # L24-28
    feeder_clip: float = 1.0  # L19-22

    # Gradient monitoring
    log_grad_norms: bool = True
    log_every_n_steps: int = 100

    # Gradient explosion detection
    explosion_threshold: float = 10.0  # Warn if grad norm > threshold
    nan_check: bool = True  # Check for NaN gradients


@dataclass
class GradientStats:
    """
    Statistics about gradients.

    Attributes:
        total_norm: Total gradient norm
        layer_norms: Per-layer gradient norms
        max_grad: Maximum gradient value
        min_grad: Minimum gradient value
        has_nan: Whether NaN gradients were detected
        has_inf: Whether Inf gradients were detected
        clipped: Whether gradients were clipped
    """

    total_norm: float = 0.0
    layer_norms: dict[str, float] = field(default_factory=dict)
    max_grad: float = 0.0
    min_grad: float = 0.0
    has_nan: bool = False
    has_inf: bool = False
    clipped: bool = False


# ============================================================================
# Gradient Clipper
# ============================================================================


class GradientClipper:
    """
    Gradient clipping and monitoring for v3 training.

    Provides:
    1. Global gradient clipping (standard)
    2. Per-layer gradient clipping (for interface sensitivity)
    3. Gradient norm monitoring
    4. NaN/Inf detection
    5. Gradient explosion warnings

    The v3 architecture has special gradient sensitivity at the L22->L23
    interface boundary. This clipper supports tighter clipping at the
    interface layer to prevent gradient explosions during Phase 0.5.

    Example:
        >>> config = GradientClipConfig(max_grad_norm=1.0, per_layer_clip=True)
        >>> clipper = GradientClipper(model, config)
        >>> # In training loop
        >>> loss.backward()
        >>> stats = clipper.clip_gradients()
        >>> optimizer.step()
    """

    def __init__(
        self,
        model: nn.Module,
        config: GradientClipConfig,
    ):
        """
        Initialize GradientClipper.

        Args:
            model: Model to clip gradients for
            config: Gradient clipping configuration
        """
        self.model = model
        self.config = config
        self.encoder = model.encoder if hasattr(model, "encoder") else model

        # Tracking
        self.step = 0
        self.gradient_history: list[GradientStats] = []
        self.explosion_count = 0

    def clip_gradients(self) -> GradientStats:
        """
        Clip gradients and return statistics.

        This method:
        1. Checks for NaN/Inf gradients (if enabled)
        2. Computes per-layer gradient norms (if logging enabled)
        3. Applies global or per-layer clipping
        4. Detects gradient explosions
        5. Logs statistics periodically

        Returns:
            GradientStats with clipping info
        """
        stats = GradientStats()

        # Check for NaN/Inf first
        if self.config.nan_check:
            stats.has_nan, stats.has_inf = self._check_gradient_health()
            if stats.has_nan or stats.has_inf:
                logger.warning(f"Step {self.step}: NaN={stats.has_nan}, Inf={stats.has_inf}")
                self._zero_bad_gradients()

        # Calculate gradient norms per layer
        if self.config.log_grad_norms:
            stats.layer_norms = self._compute_layer_norms()

        # Apply clipping
        if self.config.per_layer_clip:
            stats = self._per_layer_clip(stats)
        else:
            stats = self._global_clip(stats)

        # Check for gradient explosion
        if stats.total_norm > self.config.explosion_threshold:
            self.explosion_count += 1
            logger.warning(
                f"Step {self.step}: Gradient explosion detected! "
                f"Norm={stats.total_norm:.2f} > {self.config.explosion_threshold}"
            )

        # Log periodically
        if self.config.log_grad_norms and self.step % self.config.log_every_n_steps == 0:
            self._log_gradient_stats(stats)

        self.step += 1
        self.gradient_history.append(stats)

        return stats

    def _check_gradient_health(self) -> tuple[bool, bool]:
        """
        Check for NaN or Inf gradients.

        Returns:
            Tuple of (has_nan, has_inf)
        """
        has_nan = False
        has_inf = False

        for param in self.model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    has_nan = True
                if torch.isinf(param.grad).any():
                    has_inf = True

            if has_nan and has_inf:
                break

        return has_nan, has_inf

    def _zero_bad_gradients(self) -> int:
        """
        Zero out NaN and Inf gradients.

        Returns:
            Number of gradient values zeroed
        """
        zeroed = 0
        for param in self.model.parameters():
            if param.grad is not None:
                bad_mask = torch.isnan(param.grad) | torch.isinf(param.grad)
                if bad_mask.any():
                    param.grad[bad_mask] = 0.0
                    zeroed += bad_mask.sum().item()
        return int(zeroed)

    def _compute_layer_norms(self) -> dict[str, float]:
        """
        Compute gradient norm per layer.

        Returns:
            Dict mapping layer name to gradient norm
        """
        layer_norms = {}

        # Check if encoder has layers attribute
        if not hasattr(self.encoder, "layers"):
            return layer_norms

        for layer_idx in range(len(self.encoder.layers)):
            layer = self.encoder.layers[layer_idx]
            layer_grad_norm = 0.0

            for param in layer.parameters():
                if param.grad is not None:
                    layer_grad_norm += param.grad.data.norm(2).item() ** 2

            layer_norms[f"layer_{layer_idx + 1}"] = math.sqrt(layer_grad_norm)

        return layer_norms

    def _global_clip(self, stats: GradientStats) -> GradientStats:
        """
        Apply global gradient clipping.

        Args:
            stats: GradientStats to update

        Returns:
            Updated GradientStats
        """
        # Get all parameters with gradients
        params = [p for p in self.model.parameters() if p.grad is not None]

        if not params:
            return stats

        # Compute total norm before clipping
        total_norm_sq = 0.0
        for p in params:
            total_norm_sq += p.grad.data.norm(2).item() ** 2
        total_norm_before = math.sqrt(total_norm_sq)

        # Apply clipping
        torch.nn.utils.clip_grad_norm_(
            params,
            max_norm=self.config.max_grad_norm,
        )

        stats.total_norm = total_norm_before
        stats.clipped = total_norm_before > self.config.max_grad_norm

        return stats

    def _per_layer_clip(self, stats: GradientStats) -> GradientStats:
        """
        Apply per-layer gradient clipping.

        Uses different clip thresholds for different layer bands:
        - L23 (interface): tighter clip (0.5)
        - L24-28 (family): standard clip (1.0)
        - L19-22 (feeder): standard clip (1.0)

        Args:
            stats: GradientStats to update

        Returns:
            Updated GradientStats
        """
        total_norm_sq = 0.0

        # Check if encoder has layers attribute
        if not hasattr(self.encoder, "layers"):
            return self._global_clip(stats)

        for layer_idx in range(len(self.encoder.layers)):
            layer = self.encoder.layers[layer_idx]

            # Determine clip value based on layer position
            if layer_idx == 22:  # Interface layer (L23)
                max_norm = self.config.interface_clip
            elif layer_idx >= 23:  # Family band (L24-28)
                max_norm = self.config.family_clip
            elif layer_idx >= 18:  # Feeder band (L19-22)
                max_norm = self.config.feeder_clip
            else:  # Foundation/Core (should be frozen)
                max_norm = self.config.max_grad_norm

            # Clip this layer
            layer_params = [p for p in layer.parameters() if p.grad is not None]
            if layer_params:
                # Compute norm before clipping
                layer_norm_sq = 0.0
                for p in layer_params:
                    layer_norm_sq += p.grad.data.norm(2).item() ** 2
                layer_norm = math.sqrt(layer_norm_sq)

                # Apply clipping
                torch.nn.utils.clip_grad_norm_(
                    layer_params,
                    max_norm=max_norm,
                )
                total_norm_sq += layer_norm**2

        stats.total_norm = math.sqrt(total_norm_sq)
        stats.clipped = True  # Per-layer always applies clipping

        return stats

    def _log_gradient_stats(self, stats: GradientStats) -> None:
        """Log gradient statistics."""
        logger.info(f"Step {self.step} gradient stats:")
        logger.info(f"  Total norm: {stats.total_norm:.4f}")
        logger.info(f"  Clipped: {stats.clipped}")

        if stats.layer_norms:
            # Show key layers
            for key in ["layer_22", "layer_23", "layer_24", "layer_28"]:
                if key in stats.layer_norms:
                    logger.info(f"  {key}: {stats.layer_norms[key]:.4f}")

    def get_gradient_summary(self) -> dict[str, Any]:
        """
        Get summary of gradient history.

        Returns:
            Dict with gradient statistics summary
        """
        if not self.gradient_history:
            return {}

        norms = [s.total_norm for s in self.gradient_history]
        clipped = sum(1 for s in self.gradient_history if s.clipped)

        return {
            "mean_norm": sum(norms) / len(norms),
            "max_norm": max(norms),
            "min_norm": min(norms),
            "clip_count": clipped,
            "clip_ratio": clipped / len(self.gradient_history),
            "explosion_count": self.explosion_count,
        }

    def clear_history(self) -> None:
        """Clear gradient history to prevent memory leaks."""
        self.gradient_history.clear()

    def reset(self) -> None:
        """Reset clipper state."""
        self.step = 0
        self.gradient_history.clear()
        self.explosion_count = 0


# ============================================================================
# Interface Gradient Monitor
# ============================================================================


class InterfaceGradientMonitor:
    """
    Specialized monitor for L22->L23 interface gradients.

    The interface between v2 (L22) and v3 (L23) is the most sensitive
    region during healing. This monitor tracks gradient flow across
    this boundary.

    A healthy interface should have:
    - Comparable gradient norms on both sides (ratio between 0.1 and 10)
    - No sudden spikes or drops in gradient magnitude
    - Consistent flow patterns across training

    Example:
        >>> monitor = InterfaceGradientMonitor(model)
        >>> # In training loop after backward
        >>> stats = monitor.record()
        >>> if not stats["interface_healthy"]:
        ...     logger.warning("Interface gradient imbalance detected")
    """

    def __init__(self, model: nn.Module):
        """
        Initialize InterfaceGradientMonitor.

        Args:
            model: Model to monitor
        """
        self.model = model
        self.encoder = model.encoder if hasattr(model, "encoder") else model
        self.history: list[dict[str, float]] = []

    def record(self) -> dict[str, float]:
        """
        Record gradient statistics at interface.

        Records:
        - L22 gradient norm (last v2 layer)
        - L23 gradient norm (first v3 layer)
        - L23/L22 ratio (measures gradient flow balance)
        - Interface health status

        Returns:
            Dict with interface gradient statistics
        """
        stats: dict[str, float] = {}

        # Check if encoder has enough layers
        if not hasattr(self.encoder, "layers") or len(self.encoder.layers) < 23:
            stats["l22_grad_norm"] = 0.0
            stats["l23_grad_norm"] = 0.0
            stats["l23_l22_ratio"] = 0.0
            stats["interface_healthy"] = 1.0  # Assume healthy if can't check
            self.history.append(stats)
            return stats

        # L22 (last v2 layer) gradients
        l22 = self.encoder.layers[21]
        l22_norm = self._layer_grad_norm(l22)
        stats["l22_grad_norm"] = l22_norm

        # L23 (first v3 layer) gradients
        l23 = self.encoder.layers[22]
        l23_norm = self._layer_grad_norm(l23)
        stats["l23_grad_norm"] = l23_norm

        # Ratio (measures gradient flow)
        if l22_norm > 0:
            stats["l23_l22_ratio"] = l23_norm / l22_norm
        else:
            stats["l23_l22_ratio"] = 0.0

        # Interface is healthy if ratio is between 0.1 and 10
        stats["interface_healthy"] = 1.0 if 0.1 < stats["l23_l22_ratio"] < 10.0 else 0.0

        self.history.append(stats)
        return stats

    def _layer_grad_norm(self, layer: nn.Module) -> float:
        """
        Compute gradient L2 norm for a layer.

        Args:
            layer: Layer to compute gradient norm for

        Returns:
            L2 norm of all gradients in layer
        """
        norm_sq = 0.0
        for param in layer.parameters():
            if param.grad is not None:
                norm_sq += param.grad.data.norm(2).item() ** 2
        return math.sqrt(norm_sq)

    def get_interface_health(self) -> dict[str, Any]:
        """
        Get interface health summary.

        Returns:
            Dict with health metrics and status message
        """
        if not self.history:
            return {"healthy": True, "message": "No data yet"}

        healthy_count = sum(1 for h in self.history if h.get("interface_healthy", 0) > 0.5)
        health_ratio = healthy_count / len(self.history)

        ratios = [h["l23_l22_ratio"] for h in self.history if "l23_l22_ratio" in h]
        mean_ratio = sum(ratios) / len(ratios) if ratios else 0.0

        return {
            "healthy": health_ratio > 0.9,
            "health_ratio": health_ratio,
            "mean_l23_l22_ratio": mean_ratio,
            "message": ("OK" if health_ratio > 0.9 else "WARNING: Interface gradient imbalance"),
        }

    def clear_history(self) -> None:
        """Clear history to prevent memory leaks."""
        self.history.clear()


# ============================================================================
# Convenience Functions
# ============================================================================


def clip_gradients(
    model: nn.Module,
    max_norm: float = 1.0,
    per_layer: bool = False,
) -> float:
    """
    Clip gradients for a model.

    This is a convenience function that creates a temporary GradientClipper
    and applies clipping. For repeated use, create a GradientClipper instance.

    Args:
        model: Model to clip
        max_norm: Maximum gradient norm
        per_layer: Whether to clip per-layer

    Returns:
        Total gradient norm before clipping
    """
    config = GradientClipConfig(
        max_grad_norm=max_norm,
        per_layer_clip=per_layer,
        log_grad_norms=False,
    )

    clipper = GradientClipper(model, config)
    stats = clipper.clip_gradients()

    return stats.total_norm


def create_gradient_clipper(
    model: nn.Module,
    max_grad_norm: float = 1.0,
    per_layer_clip: bool = False,
    interface_clip: float = 0.5,
    log_every_n_steps: int = 100,
) -> GradientClipper:
    """
    Create a GradientClipper with common settings.

    Args:
        model: Model to clip gradients for
        max_grad_norm: Maximum gradient norm for global clipping
        per_layer_clip: Whether to use per-layer clipping
        interface_clip: Clip threshold for L23 interface layer
        log_every_n_steps: How often to log statistics

    Returns:
        Configured GradientClipper
    """
    config = GradientClipConfig(
        max_grad_norm=max_grad_norm,
        per_layer_clip=per_layer_clip,
        interface_clip=interface_clip,
        log_every_n_steps=log_every_n_steps,
    )
    return GradientClipper(model, config)
