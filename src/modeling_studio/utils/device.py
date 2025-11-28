"""
Device and environment utilities.
"""

import os
import torch
from typing import Literal


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_map(strategy: str = "auto") -> str | dict | None:
    """
    Get device map for model loading.
    
    Args:
        strategy: Device mapping strategy
            - "auto": Automatic device mapping
            - "cpu": Load on CPU
            - "cuda": Load on single GPU
            - "balanced": Balance across GPUs
            
    Returns:
        Device map for model loading
    """
    if strategy == "auto":
        return "auto"
    elif strategy == "cpu":
        return {"": "cpu"}
    elif strategy == "cuda":
        return {"": 0}
    elif strategy == "balanced":
        return "balanced"
    return None


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "auto": "auto",
    }
    return dtype_map.get(dtype_str, torch.float32)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Enable deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def print_gpu_memory() -> None:
    """Print GPU memory usage."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            allocated = torch.cuda.memory_allocated(i) / 1e9
            cached = torch.cuda.memory_reserved(i) / 1e9
            print(f"GPU {i}: {allocated:.2f}GB allocated, {cached:.2f}GB cached, {total:.2f}GB total")


def get_num_gpus() -> int:
    """Get the number of available GPUs."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def setup_environment() -> None:
    """Set up environment variables for training."""
    # Disable tokenizers parallelism warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Enable TF32 for faster training on Ampere+ GPUs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
