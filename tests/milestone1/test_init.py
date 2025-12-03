"""
Issue 1.1.1: Tests for `modeling_studio/__init__.py`

Tests:
- test_version_exists - Verify `__version__` is defined
- test_author_exists - Verify `__author__` is defined
- test_config_import - Verify `Config` class can be imported from package root
"""

import pytest


class TestPackageInit:
    """Test package initialization and exports."""

    def test_version_exists(self):
        """Verify `__version__` is defined."""
        import modeling_studio

        assert hasattr(modeling_studio, "__version__")
        assert isinstance(modeling_studio.__version__, str)
        assert len(modeling_studio.__version__) > 0
        # Version should follow semver pattern
        parts = modeling_studio.__version__.split(".")
        assert len(parts) >= 2  # At least major.minor

    def test_author_exists(self):
        """Verify `__author__` is defined."""
        import modeling_studio

        assert hasattr(modeling_studio, "__author__")
        assert isinstance(modeling_studio.__author__, str)
        assert len(modeling_studio.__author__) > 0

    def test_config_import(self):
        """Verify `Config` class can be imported from package root."""
        from modeling_studio import Config

        assert Config is not None
        # Verify it's actually the Config class
        config = Config()
        assert hasattr(config, "model")
        assert hasattr(config, "training")
        assert hasattr(config, "data")

    def test_version_format(self):
        """Verify version follows semantic versioning."""
        import modeling_studio

        version = modeling_studio.__version__
        parts = version.split(".")

        # Should have at least major.minor.patch
        assert len(parts) >= 2

        # Each part should be a valid number (for release versions)
        for i, part in enumerate(parts[:3]):  # Only check first 3 parts
            # Handle pre-release suffixes like "0.1.0-alpha"
            clean_part = part.split("-")[0].split("+")[0]
            assert clean_part.isdigit(), f"Version part {i} '{part}' is not a valid number"

    def test_package_docstring(self):
        """Verify package has a docstring."""
        import modeling_studio

        assert modeling_studio.__doc__ is not None
        assert len(modeling_studio.__doc__) > 0
