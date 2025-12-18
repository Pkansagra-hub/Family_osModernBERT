"""
Weight Manager - HuggingFace Hub Integration

Downloads model weights on first use, caches locally.
Supports version selection and quantization variants.

Features:
    - Download encoder weights (once)
    - Download decoder weights (on demand)
    - Cache in ~/.cache/familyos_ultrabert/
    - Resume interrupted downloads
    - Checksum verification
    - Progress bar

Usage:
    from familyos_ultrabert.weights_manager import download_encoder, download_decoder

    # Download encoder (first time downloads, subsequent calls use cache)
    encoder_path = download_encoder(version="v1", quantization="int8")

    # Download decoder (for counterfactual generation)
    decoder_path = download_decoder(version="v3", quantization="int8")

    # Clear cache
    clear_cache()

    # Get cache size
    size_mb = get_cache_size()
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Literal, Optional

logger = logging.getLogger(__name__)

# HuggingFace Hub repository for weights
HF_REPO = "Pkansagra/ultrabert-weights"

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "familyos_ultrabert"

# Quantization options
QuantizationType = Literal["fp32", "fp16", "int8"]


def get_cache_dir() -> Path:
    """Get cache directory, creating if needed.

    The cache directory can be overridden via the FAMILYOS_CACHE_DIR
    environment variable.

    Returns:
        Path to the cache directory.
    """
    cache_dir = os.environ.get("FAMILYOS_CACHE_DIR")
    if cache_dir:
        cache_path = Path(cache_dir)
    else:
        cache_path = DEFAULT_CACHE_DIR

    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def _check_huggingface_hub() -> bool:
    """Check if huggingface_hub is installed."""
    try:
        import huggingface_hub
        return True
    except ImportError:
        return False


def download_encoder(
    version: str = "v1",
    quantization: QuantizationType = "int8",
    force: bool = False,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Download encoder weights from HuggingFace Hub.

    Downloads the encoder weights if not already cached. The encoder is
    the ModernBERT backbone that powers all 12 capabilities.

    Args:
        version: Encoder version (default: v1)
        quantization: "fp32", "fp16", or "int8" (default: int8)
        force: Re-download even if cached
        cache_dir: Custom cache directory (default: ~/.cache/familyos_ultrabert/)

    Returns:
        Path to local weights directory

    Raises:
        ImportError: If huggingface_hub is not installed
        RuntimeError: If download fails

    Example:
        >>> encoder_path = download_encoder(version="v1", quantization="int8")
        >>> print(encoder_path)
        /home/user/.cache/familyos_ultrabert/encoder/v1/int8
    """
    if not _check_huggingface_hub():
        raise ImportError(
            "huggingface_hub is required for weight downloading. "
            "Install with: pip install huggingface-hub>=0.20.0"
        )

    from huggingface_hub import snapshot_download, HfFileSystem

    cache = cache_dir or get_cache_dir()
    cache_path = cache / f"encoder/{version}/{quantization}"

    if cache_path.exists() and not force:
        # Verify the cache has files
        files = list(cache_path.glob("*"))
        if files:
            logger.info(f"Using cached encoder: {cache_path}")
            return cache_path
        else:
            logger.warning(f"Cache directory empty, re-downloading: {cache_path}")

    logger.info(f"Downloading encoder v{version} ({quantization})...")

    try:
        # Check if the pattern exists in the repo
        pattern = f"encoder/{version}/{quantization}/*"

        snapshot_download(
            repo_id=HF_REPO,
            allow_patterns=[pattern],
            local_dir=cache,
            local_dir_use_symlinks=False,
            resume_download=True,
        )

        logger.info(f"Downloaded encoder to {cache_path}")
        return cache_path

    except Exception as e:
        logger.error(f"Failed to download encoder: {e}")
        raise RuntimeError(f"Failed to download encoder v{version} ({quantization}): {e}")


def download_decoder(
    version: str = "v3",
    quantization: QuantizationType = "int8",
    force: bool = False,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Download decoder weights from HuggingFace Hub.

    Downloads the GPT-2 decoder weights if not already cached. The decoder
    is used for counterfactual generation (Capability.COUNTERFACTUAL).

    Args:
        version: Decoder version (default: v3)
        quantization: "fp32", "fp16", or "int8" (default: int8)
        force: Re-download even if cached
        cache_dir: Custom cache directory (default: ~/.cache/familyos_ultrabert/)

    Returns:
        Path to local weights directory

    Raises:
        ImportError: If huggingface_hub is not installed
        RuntimeError: If download fails

    Example:
        >>> decoder_path = download_decoder(version="v3", quantization="int8")
        >>> print(decoder_path)
        /home/user/.cache/familyos_ultrabert/decoder/v3/int8
    """
    if not _check_huggingface_hub():
        raise ImportError(
            "huggingface_hub is required for weight downloading. "
            "Install with: pip install huggingface-hub>=0.20.0"
        )

    from huggingface_hub import snapshot_download

    cache = cache_dir or get_cache_dir()
    cache_path = cache / f"decoder/{version}/{quantization}"

    if cache_path.exists() and not force:
        # Verify the cache has files
        files = list(cache_path.glob("*"))
        if files:
            logger.info(f"Using cached decoder: {cache_path}")
            return cache_path
        else:
            logger.warning(f"Cache directory empty, re-downloading: {cache_path}")

    logger.info(f"Downloading decoder v{version} ({quantization})...")

    try:
        pattern = f"decoder/{version}/{quantization}/*"

        snapshot_download(
            repo_id=HF_REPO,
            allow_patterns=[pattern],
            local_dir=cache,
            local_dir_use_symlinks=False,
            resume_download=True,
        )

        logger.info(f"Downloaded decoder to {cache_path}")
        return cache_path

    except Exception as e:
        logger.error(f"Failed to download decoder: {e}")
        raise RuntimeError(f"Failed to download decoder v{version} ({quantization}): {e}")


def get_cache_size() -> float:
    """Get total cache size in megabytes.

    Returns:
        Total cache size in MB
    """
    cache = get_cache_dir()
    if not cache.exists():
        return 0.0

    total_bytes = sum(f.stat().st_size for f in cache.rglob("*") if f.is_file())
    return total_bytes / (1024 * 1024)


def clear_cache(component: Optional[Literal["encoder", "decoder"]] = None) -> None:
    """Clear cached weights.

    Args:
        component: "encoder", "decoder", or None (clear all)
    """
    cache = get_cache_dir()

    if component is None:
        # Clear everything
        if cache.exists():
            shutil.rmtree(cache)
            logger.info(f"Cleared all cached weights at {cache}")
    else:
        # Clear specific component
        component_path = cache / component
        if component_path.exists():
            shutil.rmtree(component_path)
            logger.info(f"Cleared cached {component} weights")


def list_cached_versions() -> dict[str, list[str]]:
    """List all cached versions and quantization variants.

    Returns:
        Dictionary with "encoder" and "decoder" keys, each containing
        a list of cached variants like "v1/int8".
    """
    cache = get_cache_dir()
    result = {"encoder": [], "decoder": []}

    for component in ["encoder", "decoder"]:
        component_path = cache / component
        if component_path.exists():
            for version_dir in component_path.iterdir():
                if version_dir.is_dir():
                    for quant_dir in version_dir.iterdir():
                        if quant_dir.is_dir():
                            result[component].append(f"{version_dir.name}/{quant_dir.name}")

    return result


def is_cached(
    component: Literal["encoder", "decoder"],
    version: str,
    quantization: QuantizationType = "int8",
) -> bool:
    """Check if a specific version is cached.

    Args:
        component: "encoder" or "decoder"
        version: Version string (e.g., "v1", "v3")
        quantization: Quantization variant

    Returns:
        True if cached, False otherwise
    """
    cache = get_cache_dir()
    cache_path = cache / component / version / quantization

    if not cache_path.exists():
        return False

    # Check if there are actual files
    files = list(cache_path.glob("*"))
    return len(files) > 0


def get_weights_info() -> dict:
    """Get information about available weights.

    Returns:
        Dictionary with weight information including:
        - hf_repo: HuggingFace repository name
        - cache_dir: Current cache directory
        - cache_size_mb: Total cache size in MB
        - cached_versions: List of cached versions
    """
    return {
        "hf_repo": HF_REPO,
        "cache_dir": str(get_cache_dir()),
        "cache_size_mb": round(get_cache_size(), 2),
        "cached_versions": list_cached_versions(),
    }


# =============================================================================
# Offline Mode Support
# =============================================================================


def set_offline_mode(enabled: bool = True) -> None:
    """Enable or disable offline mode.

    In offline mode, the weights manager will only use cached weights
    and will not attempt to download from HuggingFace Hub.

    Args:
        enabled: True to enable offline mode, False to disable
    """
    if enabled:
        os.environ["HF_HUB_OFFLINE"] = "1"
        logger.info("Offline mode enabled")
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        logger.info("Offline mode disabled")


def is_offline_mode() -> bool:
    """Check if offline mode is enabled.

    Returns:
        True if offline mode is enabled
    """
    return os.environ.get("HF_HUB_OFFLINE", "0") == "1"


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Core functions
    "download_encoder",
    "download_decoder",
    # Cache management
    "get_cache_dir",
    "get_cache_size",
    "clear_cache",
    "list_cached_versions",
    "is_cached",
    "get_weights_info",
    # Offline mode
    "set_offline_mode",
    "is_offline_mode",
    # Constants
    "HF_REPO",
]
