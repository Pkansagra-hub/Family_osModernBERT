"""
Issue 1.2.1: Tests for `modeling_studio/utils/__init__.py`

Tests:
- test_all_exports - Verify all functions in `__all__` are importable
- test_setup_logging_export - Verify `setup_logging` is exported
- test_get_device_export - Verify `get_device` is exported
- test_set_seed_export - Verify `set_seed` is exported
"""


class TestUtilsInit:
    """Test utils module exports and initialization."""

    def test_all_exports(self):
        """Verify all functions in `__all__` are importable."""
        from modeling_studio import utils

        assert hasattr(utils, "__all__")

        for name in utils.__all__:
            assert hasattr(utils, name), f"{name} not found in utils module"
            obj = getattr(utils, name)
            assert obj is not None, f"{name} is None"

    def test_setup_logging_export(self):
        """Verify `setup_logging` is exported."""
        from modeling_studio.utils import setup_logging

        assert setup_logging is not None
        assert callable(setup_logging)

    def test_get_device_export(self):
        """Verify `get_device` is exported."""
        from modeling_studio.utils import get_device

        assert get_device is not None
        assert callable(get_device)

    def test_set_seed_export(self):
        """Verify `set_seed` is exported."""
        from modeling_studio.utils import set_seed

        assert set_seed is not None
        assert callable(set_seed)

    def test_get_logger_export(self):
        """Verify `get_logger` is exported."""
        from modeling_studio.utils import get_logger

        assert get_logger is not None
        assert callable(get_logger)

    def test_get_device_map_export(self):
        """Verify `get_device_map` is exported."""
        from modeling_studio.utils import get_device_map

        assert get_device_map is not None
        assert callable(get_device_map)

    def test_get_torch_dtype_export(self):
        """Verify `get_torch_dtype` is exported."""
        from modeling_studio.utils import get_torch_dtype

        assert get_torch_dtype is not None
        assert callable(get_torch_dtype)

    def test_print_gpu_memory_export(self):
        """Verify `print_gpu_memory` is exported."""
        from modeling_studio.utils import print_gpu_memory

        assert print_gpu_memory is not None
        assert callable(print_gpu_memory)

    def test_get_num_gpus_export(self):
        """Verify `get_num_gpus` is exported."""
        from modeling_studio.utils import get_num_gpus

        assert get_num_gpus is not None
        assert callable(get_num_gpus)

    def test_setup_environment_export(self):
        """Verify `setup_environment` is exported."""
        from modeling_studio.utils import setup_environment

        assert setup_environment is not None
        assert callable(setup_environment)

    def test_all_exports_count(self):
        """Verify expected number of exports in __all__."""
        from modeling_studio.utils import __all__

        expected_exports = [
            "setup_logging",
            "get_logger",
            "get_device",
            "get_device_map",
            "get_torch_dtype",
            "set_seed",
            "print_gpu_memory",
            "get_num_gpus",
            "setup_environment",
        ]

        for export in expected_exports:
            assert export in __all__, f"{export} missing from __all__"
