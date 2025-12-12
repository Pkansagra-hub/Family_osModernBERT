"""Milestone 7: benchmark CLI behavior.

These tests are structured to avoid loading model weights.
"""

from __future__ import annotations

import os
import sys


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class TestBenchmarksCLI:
    """Verify CLI argument validation and error messages."""

    def test_cli_rejects_unknown_suite(self, capsys):
        """Unknown suite names should return exit code 2 with helpful output."""
        from familyos_ultrabert.benchmarks import cli

        code = cli(["--suite", "does_not_exist", "--format", "text"])
        captured = capsys.readouterr()

        assert code == 2
        assert "Unknown suite" in captured.out
        assert "Available suites" in captured.out
