"""
Issue 1.2.2: Tests for `modeling_studio/utils/device.py`

Tests:
- test_get_device_returns_device - Verify returns `torch.device` instance
- test_get_device_cpu_fallback - On CPU-only system, returns `cpu` device
- test_get_device_map_auto - `get_device_map("auto")` returns `"auto"`
- test_get_device_map_cpu - `get_device_map("cpu")` returns `{"": "cpu"}`
- test_get_device_map_cuda - `get_device_map("cuda")` returns `{"": 0}`
- test_get_device_map_balanced - `get_device_map("balanced")` returns `"balanced"`
- test_get_torch_dtype_float32 - `get_torch_dtype("float32")` returns `torch.float32`
- test_get_torch_dtype_float16 - `get_torch_dtype("float16")` returns `torch.float16`
- test_get_torch_dtype_bfloat16 - `get_torch_dtype("bfloat16")` returns `torch.bfloat16`
- test_get_torch_dtype_alias - `get_torch_dtype("fp16")` returns `torch.float16`
- test_set_seed_reproducibility - Set seed, generate random, reset seed, verify same output
- test_get_num_gpus - Returns integer >= 0
- test_print_gpu_memory - Runs without error (smoke test)
- test_setup_environment - Verifies `TOKENIZERS_PARALLELISM` env var set
"""

import os

import torch


class TestGetDevice:
    """Test get_device function."""

    def test_get_device_returns_device(self):
        """Verify returns `torch.device` instance."""
        from modeling_studio.utils.device import get_device

        device = get_device()

        assert isinstance(device, torch.device)

    def test_get_device_valid_type(self):
        """Verify device type is valid (cuda, mps, or cpu)."""
        from modeling_studio.utils.device import get_device

        device = get_device()

        assert device.type in ["cuda", "mps", "cpu"]

    def test_get_device_cpu_fallback(self, monkeypatch):
        """On CPU-only system, returns `cpu` device."""
        from modeling_studio.utils.device import get_device

        # Mock CUDA not available
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        # Mock MPS not available (if applicable)
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

        device = get_device()

        assert device.type == "cpu"


class TestGetDeviceMap:
    """Test get_device_map function."""

    def test_get_device_map_auto(self):
        """get_device_map('auto') returns 'auto'."""
        from modeling_studio.utils.device import get_device_map

        result = get_device_map("auto")

        assert result == "auto"

    def test_get_device_map_cpu(self):
        """get_device_map('cpu') returns {'': 'cpu'}."""
        from modeling_studio.utils.device import get_device_map

        result = get_device_map("cpu")

        assert result == {"": "cpu"}

    def test_get_device_map_cuda(self):
        """get_device_map('cuda') returns {'': 0}."""
        from modeling_studio.utils.device import get_device_map

        result = get_device_map("cuda")

        assert result == {"": 0}

    def test_get_device_map_balanced(self):
        """get_device_map('balanced') returns 'balanced'."""
        from modeling_studio.utils.device import get_device_map

        result = get_device_map("balanced")

        assert result == "balanced"

    def test_get_device_map_unknown(self):
        """get_device_map with unknown strategy returns None."""
        from modeling_studio.utils.device import get_device_map

        result = get_device_map("unknown_strategy")

        assert result is None

    def test_get_device_map_default(self):
        """get_device_map with no args uses 'auto' default."""
        from modeling_studio.utils.device import get_device_map

        result = get_device_map()

        assert result == "auto"


class TestGetTorchDtype:
    """Test get_torch_dtype function."""

    def test_get_torch_dtype_float32(self):
        """get_torch_dtype('float32') returns torch.float32."""
        from modeling_studio.utils.device import get_torch_dtype

        result = get_torch_dtype("float32")

        assert result == torch.float32

    def test_get_torch_dtype_float16(self):
        """get_torch_dtype('float16') returns torch.float16."""
        from modeling_studio.utils.device import get_torch_dtype

        result = get_torch_dtype("float16")

        assert result == torch.float16

    def test_get_torch_dtype_bfloat16(self):
        """get_torch_dtype('bfloat16') returns torch.bfloat16."""
        from modeling_studio.utils.device import get_torch_dtype

        result = get_torch_dtype("bfloat16")

        assert result == torch.bfloat16

    def test_get_torch_dtype_alias_fp32(self):
        """get_torch_dtype('fp32') returns torch.float32."""
        from modeling_studio.utils.device import get_torch_dtype

        result = get_torch_dtype("fp32")

        assert result == torch.float32

    def test_get_torch_dtype_alias_fp16(self):
        """get_torch_dtype('fp16') returns torch.float16."""
        from modeling_studio.utils.device import get_torch_dtype

        result = get_torch_dtype("fp16")

        assert result == torch.float16

    def test_get_torch_dtype_alias_bf16(self):
        """get_torch_dtype('bf16') returns torch.bfloat16."""
        from modeling_studio.utils.device import get_torch_dtype

        result = get_torch_dtype("bf16")

        assert result == torch.bfloat16

    def test_get_torch_dtype_auto(self):
        """get_torch_dtype('auto') returns 'auto' string."""
        from modeling_studio.utils.device import get_torch_dtype

        result = get_torch_dtype("auto")

        assert result == "auto"

    def test_get_torch_dtype_unknown_fallback(self):
        """get_torch_dtype with unknown string falls back to float32."""
        from modeling_studio.utils.device import get_torch_dtype

        result = get_torch_dtype("unknown_dtype")

        assert result == torch.float32


class TestSetSeed:
    """Test set_seed function for reproducibility."""

    def test_set_seed_reproducibility(self):
        """Set seed, generate random, reset seed, verify same output."""
        import random

        import numpy as np

        from modeling_studio.utils.device import set_seed

        seed = 42

        # First generation
        set_seed(seed)
        random_val1 = random.random()
        np_val1 = np.random.random()
        torch_val1 = torch.rand(1).item()

        # Second generation with same seed
        set_seed(seed)
        random_val2 = random.random()
        np_val2 = np.random.random()
        torch_val2 = torch.rand(1).item()

        assert random_val1 == random_val2
        assert np_val1 == np_val2
        assert torch_val1 == torch_val2

    def test_set_seed_different_seeds_different_output(self):
        """Different seeds produce different outputs."""
        from modeling_studio.utils.device import set_seed

        set_seed(42)
        val1 = torch.rand(1).item()

        set_seed(123)
        val2 = torch.rand(1).item()

        # Extremely unlikely to be equal with different seeds
        assert val1 != val2

    def test_set_seed_accepts_integer(self):
        """set_seed accepts integer argument without error."""
        from modeling_studio.utils.device import set_seed

        # Should not raise
        set_seed(0)
        set_seed(42)
        set_seed(2**31 - 1)  # Large seed


class TestGetNumGpus:
    """Test get_num_gpus function."""

    def test_get_num_gpus_returns_integer(self):
        """Returns integer >= 0."""
        from modeling_studio.utils.device import get_num_gpus

        result = get_num_gpus()

        assert isinstance(result, int)
        assert result >= 0

    def test_get_num_gpus_matches_cuda(self):
        """Result matches torch.cuda.device_count when CUDA available."""
        from modeling_studio.utils.device import get_num_gpus

        result = get_num_gpus()

        if torch.cuda.is_available():
            assert result == torch.cuda.device_count()
        else:
            assert result == 0


class TestPrintGpuMemory:
    """Test print_gpu_memory function."""

    def test_print_gpu_memory_runs(self, capsys):
        """Runs without error (smoke test)."""
        from modeling_studio.utils.device import print_gpu_memory

        # Should not raise any exception
        print_gpu_memory()

        # On systems with GPU, should print something
        # On CPU-only, may print nothing - both are valid


class TestSetupEnvironment:
    """Test setup_environment function."""

    def test_setup_environment_sets_tokenizers_parallelism(self):
        """Verifies `TOKENIZERS_PARALLELISM` env var set."""
        from modeling_studio.utils.device import setup_environment

        # Clear the env var first if it exists
        if "TOKENIZERS_PARALLELISM" in os.environ:
            del os.environ["TOKENIZERS_PARALLELISM"]

        setup_environment()

        assert "TOKENIZERS_PARALLELISM" in os.environ
        assert os.environ["TOKENIZERS_PARALLELISM"] == "false"

    def test_setup_environment_runs_without_error(self):
        """setup_environment runs without raising exceptions."""
        from modeling_studio.utils.device import setup_environment

        # Should not raise
        setup_environment()
