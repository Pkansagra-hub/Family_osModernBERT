"""
Tests for Capability enum (Issue 14.1.1).

Tests that COUNTERFACTUAL capability is properly registered.
"""

from __future__ import annotations

import pytest


# =============================================================================
# Issue 14.1.1: Add COUNTERFACTUAL Capability
# =============================================================================


class TestCounterfactualCapabilityExists:
    """Tests for Capability.COUNTERFACTUAL existence (14.1.1-T1)."""

    def test_counterfactual_capability_exists(self):
        """14.1.1-T1: Capability.COUNTERFACTUAL exists."""
        from modeling_studio.data.labels import Capability

        assert hasattr(Capability, "COUNTERFACTUAL")
        assert Capability.COUNTERFACTUAL is not None

    def test_counterfactual_capability_value(self):
        """14.1.1-T1b: COUNTERFACTUAL has value 'counterfactual'."""
        from modeling_studio.data.labels import Capability

        assert Capability.COUNTERFACTUAL.value == "counterfactual"

    def test_counterfactual_capability_str(self):
        """Capability.COUNTERFACTUAL converts to string correctly."""
        from modeling_studio.data.labels import Capability

        assert str(Capability.COUNTERFACTUAL) == "counterfactual"


class TestCapabilityCount:
    """Tests for Capability enum member count (14.1.1-T2)."""

    def test_capability_count(self):
        """14.1.1-T2: Capability enum has 13 members."""
        from modeling_studio.data.labels import Capability

        # Count all enum members
        member_count = len(list(Capability))
        assert member_count == 13, f"Expected 13 capabilities, got {member_count}"

    def test_all_capabilities_present(self):
        """All expected capabilities are present."""
        from modeling_studio.data.labels import Capability

        expected = [
            "NER_GENERAL",
            "SENTIMENT",
            "EMOTIONS",
            "SAFETY_GENERIC",
            "NLI",
            "EMBEDDING",
            "TEMPORAL",
            "NER_FAMILY",
            "INGRESS",
            "SAFETY_FAMILYOS",
            "RELATION",
            "INTENT",
            "COUNTERFACTUAL",  # NEW
        ]

        for cap_name in expected:
            assert hasattr(Capability, cap_name), f"Missing capability: {cap_name}"


class TestCapabilityToLabelsMapping:
    """Tests for COUNTERFACTUAL in CAPABILITY_TO_LABELS."""

    def test_counterfactual_in_mapping(self):
        """COUNTERFACTUAL is in CAPABILITY_TO_LABELS mapping."""
        from modeling_studio.data.labels import CAPABILITY_TO_LABELS, Capability

        assert Capability.COUNTERFACTUAL in CAPABILITY_TO_LABELS

    def test_counterfactual_has_no_labels(self):
        """COUNTERFACTUAL has None labels (decoder generates text)."""
        from modeling_studio.data.labels import CAPABILITY_TO_LABELS, Capability

        # Decoder generates text, no label schema needed
        assert CAPABILITY_TO_LABELS[Capability.COUNTERFACTUAL] is None

    def test_get_labels_for_capability(self):
        """get_labels_for_capability works with COUNTERFACTUAL."""
        from modeling_studio.data.labels import get_labels_for_capability

        labels = get_labels_for_capability("counterfactual")
        assert labels is None  # No labels for decoder

    def test_get_num_labels(self):
        """get_num_labels returns 0 for COUNTERFACTUAL."""
        from modeling_studio.data.labels import get_num_labels

        num_labels = get_num_labels("counterfactual")
        assert num_labels == 0  # No labels for decoder
