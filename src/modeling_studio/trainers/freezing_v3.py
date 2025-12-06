"""
Layer Freezing Utilities for v3 Phase-Based Training.

This module implements layer freezing utilities that freeze/unfreeze layers
by band for phase-based training. This is critical for preserving v2
capabilities while training new layers.

Layer Bands:
    Foundation (L1-6):   window=64,  frozen in Phase 0.5/1
    Core (L7-18):        window=128, frozen in Phase 0.5/1
    Feeder (L19-22):     window=256, trainable in Phase 0.5/1
    Family (L23-28):     window=512, trainable in Phase 0.5/1

Training Phases:
    Phase 0.5 (Healing): L19-28 trainable, heal cloned layers
    Phase 1 (Multi-task): L19-28 trainable, learn FamilyOS tasks
    Phase 2 (Full fine-tune): All trainable with low LR on L1-18
    Inference: All frozen

Author: FamilyOS Team
Date: December 2025
"""

import logging
from enum import Enum

import torch.nn as nn

logger = logging.getLogger(__name__)


class LayerBand(Enum):
    """Layer bands in v3 architecture."""

    FOUNDATION = "foundation"  # L1-6: window=64
    CORE = "core"  # L7-18: window=128
    FEEDER = "feeder"  # L19-22: window=256
    FAMILY = "family"  # L23-28: window=512


# Layer indices for each band (0-indexed)
LAYER_BANDS: dict[LayerBand, list[int]] = {
    LayerBand.FOUNDATION: list(range(0, 6)),  # L1-6
    LayerBand.CORE: list(range(6, 18)),  # L7-18
    LayerBand.FEEDER: list(range(18, 22)),  # L19-22
    LayerBand.FAMILY: list(range(22, 28)),  # L23-28
}


class TrainingPhase(Enum):
    """Training phases for v3."""

    PHASE_0_5 = "phase_0.5"  # Healing: L19-28 trainable
    PHASE_1 = "phase_1"  # Multi-task: L19-28 trainable
    PHASE_2 = "phase_2"  # Full fine-tune: all trainable
    INFERENCE = "inference"  # All frozen


# Which bands are trainable in each phase
PHASE_TRAINABLE_BANDS: dict[TrainingPhase, list[LayerBand]] = {
    TrainingPhase.PHASE_0_5: [LayerBand.FEEDER, LayerBand.FAMILY],
    TrainingPhase.PHASE_1: [LayerBand.FEEDER, LayerBand.FAMILY],
    TrainingPhase.PHASE_2: [
        LayerBand.FOUNDATION,
        LayerBand.CORE,
        LayerBand.FEEDER,
        LayerBand.FAMILY,
    ],
    TrainingPhase.INFERENCE: [],
}


class LayerFreezer:
    """
    Manages layer freezing for phase-based training.

    Freeze Strategy:
        Phase 0.5 (Healing):
            - Frozen: L1-18 (Foundation + Core)
            - Trainable: L19-28 (Feeder + Family)
            - Purpose: Heal cloned layers without forgetting

        Phase 1 (Multi-task):
            - Frozen: L1-18
            - Trainable: L19-28 + task heads
            - Purpose: Learn FamilyOS tasks

        Phase 2 (Full fine-tune):
            - All trainable with low LR on L1-18
            - Purpose: Optional final polish

    Args:
        model: ModernBERTv3Ultra or similar with encoder.layers

    Example:
        >>> from modeling_studio.models.modernbert_v3 import ModernBERTv3Ultra
        >>> model = ModernBERTv3Ultra(config)
        >>> freezer = LayerFreezer(model)
        >>> stats = freezer.configure_for_phase(TrainingPhase.PHASE_1)
        >>> print(f"Trainable: {stats['trainable_params']:,}")
    """

    def __init__(self, model: nn.Module):
        """
        Initialize the layer freezer.

        Args:
            model: ModernBERTv3Ultra or similar with encoder.layers
        """
        self.model = model
        self.encoder = model.encoder if hasattr(model, "encoder") else model
        self._frozen_layers: set[int] = set()
        self._frozen_components: set[str] = set()

        # Validate model has layers
        if not hasattr(self.encoder, "layers"):
            raise ValueError("Model encoder must have 'layers' attribute")

        self.num_layers = len(self.encoder.layers)  # type: ignore[arg-type]

    def get_layer(self, layer_idx: int) -> nn.Module:
        """
        Get layer by index.

        Args:
            layer_idx: Layer index (0-indexed)

        Returns:
            The layer module at the given index

        Raises:
            IndexError: If layer_idx is out of range
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(f"Layer index {layer_idx} out of range [0, {self.num_layers})")
        return self.encoder.layers[layer_idx]  # type: ignore[index,return-value]

    def freeze_layer(self, layer_idx: int) -> int:
        """
        Freeze a single layer.

        Args:
            layer_idx: Layer index (0-indexed)

        Returns:
            Number of parameters frozen
        """
        layer = self.get_layer(layer_idx)
        frozen_params = 0
        for param in layer.parameters():
            if param.requires_grad:
                param.requires_grad = False
                frozen_params += param.numel()
        self._frozen_layers.add(layer_idx)
        return frozen_params

    def unfreeze_layer(self, layer_idx: int) -> int:
        """
        Unfreeze a single layer.

        Args:
            layer_idx: Layer index (0-indexed)

        Returns:
            Number of parameters unfrozen
        """
        layer = self.get_layer(layer_idx)
        unfrozen_params = 0
        for param in layer.parameters():
            if not param.requires_grad:
                param.requires_grad = True
                unfrozen_params += param.numel()
        self._frozen_layers.discard(layer_idx)
        return unfrozen_params

    def freeze_band(self, band: LayerBand) -> int:
        """
        Freeze all layers in a band.

        Args:
            band: The layer band to freeze

        Returns:
            Number of parameters frozen
        """
        frozen_params = 0
        layer_indices = LAYER_BANDS[band]

        for layer_idx in layer_indices:
            if layer_idx < self.num_layers:
                layer = self.get_layer(layer_idx)
                for param in layer.parameters():
                    if param.requires_grad:
                        param.requires_grad = False
                        frozen_params += param.numel()
                self._frozen_layers.add(layer_idx)

        logger.info(
            "Froze %s band (L%d-L%d): %s params",
            band.value,
            min(layer_indices) + 1,
            max(layer_indices) + 1,
            f"{frozen_params:,}",
        )
        return frozen_params

    def unfreeze_band(self, band: LayerBand) -> int:
        """
        Unfreeze all layers in a band.

        Args:
            band: The layer band to unfreeze

        Returns:
            Number of parameters unfrozen
        """
        unfrozen_params = 0
        layer_indices = LAYER_BANDS[band]

        for layer_idx in layer_indices:
            if layer_idx < self.num_layers:
                layer = self.get_layer(layer_idx)
                for param in layer.parameters():
                    if not param.requires_grad:
                        param.requires_grad = True
                        unfrozen_params += param.numel()
                self._frozen_layers.discard(layer_idx)

        logger.info("Unfroze %s band: %s params", band.value, f"{unfrozen_params:,}")
        return unfrozen_params

    def freeze_embeddings(self) -> int:
        """
        Freeze embedding layer.

        Returns:
            Number of parameters frozen
        """
        frozen = 0
        embeddings = getattr(self.model, "embeddings", None)

        if embeddings is not None:
            for param in embeddings.parameters():
                if param.requires_grad:
                    param.requires_grad = False
                    frozen += param.numel()
            self._frozen_components.add("embeddings")

        logger.info("Froze embeddings: %s params", f"{frozen:,}")
        return frozen

    def unfreeze_embeddings(self) -> int:
        """
        Unfreeze embedding layer.

        Returns:
            Number of parameters unfrozen
        """
        unfrozen = 0
        embeddings = getattr(self.model, "embeddings", None)

        if embeddings is not None:
            for param in embeddings.parameters():
                if not param.requires_grad:
                    param.requires_grad = True
                    unfrozen += param.numel()
            self._frozen_components.discard("embeddings")

        logger.info("Unfroze embeddings: %s params", f"{unfrozen:,}")
        return unfrozen

    def freeze_hub_tokens(self, freeze: bool = True) -> None:
        """
        Freeze or unfreeze hub token embeddings only.

        Hub tokens are at positions 50368-50371 in word_embeddings.

        Note: Cannot selectively freeze parts of a parameter tensor.
        This is handled via gradient masking instead (Issue 5.1.5).

        Args:
            freeze: Whether to freeze (True) or unfreeze (False)
        """
        # Note: Can't selectively freeze parts of a parameter
        # This is handled via gradient masking in the optimizer
        logger.info("Hub token freezing handled via gradient masking")

    def is_layer_frozen(self, layer_idx: int) -> bool:
        """
        Check if a layer is frozen.

        Args:
            layer_idx: Layer index (0-indexed)

        Returns:
            True if the layer is frozen
        """
        return layer_idx in self._frozen_layers

    def is_band_frozen(self, band: LayerBand) -> bool:
        """
        Check if all layers in a band are frozen.

        Args:
            band: The layer band to check

        Returns:
            True if all layers in the band are frozen
        """
        layer_indices = LAYER_BANDS[band]
        return all(idx in self._frozen_layers for idx in layer_indices)

    def configure_for_phase(self, phase: TrainingPhase) -> dict[str, int]:
        """
        Configure model freezing for a training phase.

        Args:
            phase: Training phase

        Returns:
            Stats about frozen/trainable params
        """
        print(f"\nConfiguring model for {phase.value}...")

        trainable_bands = PHASE_TRAINABLE_BANDS[phase]

        # Freeze all bands first
        for band in LayerBand:
            self.freeze_band(band)

        # Unfreeze trainable bands
        for band in trainable_bands:
            self.unfreeze_band(band)

        # Always freeze embeddings in Phase 0.5/1
        if phase in [TrainingPhase.PHASE_0_5, TrainingPhase.PHASE_1]:
            self.freeze_embeddings()
        else:
            self.unfreeze_embeddings()

        # Compute stats
        stats = self.get_freeze_stats()
        self._print_freeze_summary(phase, stats)

        return stats

    def get_freeze_stats(self) -> dict[str, int]:
        """
        Get freezing statistics.

        Returns:
            Dictionary with total_params, trainable_params, frozen_params,
            frozen_layers, and trainable_layers counts
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "frozen_layers": len(self._frozen_layers),
            "trainable_layers": self.num_layers - len(self._frozen_layers),
        }

    def get_frozen_layers(self) -> list[int]:
        """
        Get list of frozen layer indices.

        Returns:
            Sorted list of frozen layer indices (0-indexed)
        """
        return sorted(self._frozen_layers)

    def get_trainable_layers(self) -> list[int]:
        """
        Get list of trainable layer indices.

        Returns:
            Sorted list of trainable layer indices (0-indexed)
        """
        return sorted(set(range(self.num_layers)) - self._frozen_layers)

    def _print_freeze_summary(self, phase: TrainingPhase, stats: dict[str, int]) -> None:
        """
        Print freeze configuration summary.

        Args:
            phase: Current training phase
            stats: Freeze statistics
        """
        print("\n" + "-" * 50)
        print(f"Phase: {phase.value}")
        print(f"  Frozen layers: {self.get_frozen_layers()}")
        print(f"  Trainable layers: {self.get_trainable_layers()}")
        print(f"  Total params: {stats['total_params']:,}")
        trainable_pct = 100 * stats["trainable_params"] / stats["total_params"]
        frozen_pct = 100 * stats["frozen_params"] / stats["total_params"]
        print(f"  Trainable: {stats['trainable_params']:,} ({trainable_pct:.1f}%)")
        print(f"  Frozen: {stats['frozen_params']:,} ({frozen_pct:.1f}%)")
        print("-" * 50)


def configure_model_for_phase(
    model: nn.Module,
    phase: str,
) -> dict[str, int]:
    """
    Configure model freezing for a training phase.

    This is a convenience function that creates a LayerFreezer and
    configures the model for the specified phase.

    Args:
        model: ModernBERTv3 model
        phase: Phase name ("phase_0.5", "phase_1", "phase_2", "inference")

    Returns:
        Freeze statistics

    Example:
        >>> stats = configure_model_for_phase(model, "phase_1")
        >>> print(f"Trainable: {stats['trainable_params']:,}")
    """
    phase_enum = TrainingPhase(phase)
    freezer = LayerFreezer(model)
    return freezer.configure_for_phase(phase_enum)


def get_band_for_layer(layer_idx: int) -> LayerBand | None:
    """
    Get the band that a layer belongs to.

    Args:
        layer_idx: Layer index (0-indexed)

    Returns:
        The LayerBand the layer belongs to, or None if out of range
    """
    for band, indices in LAYER_BANDS.items():
        if layer_idx in indices:
            return band
    return None


def get_layers_for_band(band: LayerBand) -> list[int]:
    """
    Get layer indices for a band.

    Args:
        band: The layer band

    Returns:
        List of layer indices (0-indexed)
    """
    return LAYER_BANDS[band].copy()


def get_trainable_bands_for_phase(phase: str | TrainingPhase) -> list[LayerBand]:
    """
    Get which bands are trainable for a given phase.

    Args:
        phase: Phase name or TrainingPhase enum

    Returns:
        List of trainable bands
    """
    if isinstance(phase, str):
        phase = TrainingPhase(phase)
    return PHASE_TRAINABLE_BANDS[phase].copy()
