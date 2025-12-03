"""
Milestone 2: Data Pipeline Tests
Issue 2.2.3: data/cultural_mappings.py

Tests for:
- INDIAN_ENGLISH_MAPPINGS: Expression normalization
- INDIAN_VENTING_PATTERNS: Hyperbolic expressions for safety FP prevention
- KINSHIP_VARIANTS: Indian family term variations
- FAMILY_STRUCTURE_TYPES: Family structure classifications
- IndianEnglishNormalizer: Preprocessing class
"""

# =============================================================================
# Indian English Mappings Tests
# =============================================================================


class TestIndianEnglishMappingsDefined:
    """Test INDIAN_ENGLISH_MAPPINGS has entries."""

    def test_mappings_is_dict(self):
        """INDIAN_ENGLISH_MAPPINGS should be a dictionary."""
        from modeling_studio.data.cultural_mappings import INDIAN_ENGLISH_MAPPINGS

        assert isinstance(INDIAN_ENGLISH_MAPPINGS, dict)

    def test_mappings_not_empty(self):
        """INDIAN_ENGLISH_MAPPINGS should have entries."""
        from modeling_studio.data.cultural_mappings import INDIAN_ENGLISH_MAPPINGS

        assert len(INDIAN_ENGLISH_MAPPINGS) > 0


class TestDoingNeedfulMapping:
    """Test 'doing the needful' → 'doing what's needed'."""

    def test_doing_the_needful_mapped(self):
        """'doing the needful' should be mapped."""
        from modeling_studio.data.cultural_mappings import INDIAN_ENGLISH_MAPPINGS

        assert "doing the needful" in INDIAN_ENGLISH_MAPPINGS
        assert INDIAN_ENGLISH_MAPPINGS["doing the needful"] == "doing what's needed"

    def test_do_the_needful_mapped(self):
        """'do the needful' should be mapped."""
        from modeling_studio.data.cultural_mappings import INDIAN_ENGLISH_MAPPINGS

        assert "do the needful" in INDIAN_ENGLISH_MAPPINGS


class TestRevertBackMapping:
    """Test 'revert back' → 'respond'."""

    def test_revert_back_mapped(self):
        """'revert back' should be mapped to 'respond'."""
        from modeling_studio.data.cultural_mappings import INDIAN_ENGLISH_MAPPINGS

        assert "revert back" in INDIAN_ENGLISH_MAPPINGS
        assert INDIAN_ENGLISH_MAPPINGS["revert back"] == "respond"

    def test_revert_to_me_mapped(self):
        """'revert to me' should be mapped."""
        from modeling_studio.data.cultural_mappings import INDIAN_ENGLISH_MAPPINGS

        assert "revert to me" in INDIAN_ENGLISH_MAPPINGS


class TestPassedOutMapping:
    """Test 'passed out from college' → 'graduated from college'."""

    def test_passed_out_from_college_mapped(self):
        """'passed out from college' should be mapped."""
        from modeling_studio.data.cultural_mappings import INDIAN_ENGLISH_MAPPINGS

        assert "passed out from college" in INDIAN_ENGLISH_MAPPINGS
        assert INDIAN_ENGLISH_MAPPINGS["passed out from college"] == "graduated from college"

    def test_passed_out_from_school_mapped(self):
        """'passed out from school' should be mapped."""
        from modeling_studio.data.cultural_mappings import INDIAN_ENGLISH_MAPPINGS

        assert "passed out from school" in INDIAN_ENGLISH_MAPPINGS


class TestTodayMorningMapping:
    """Test 'today morning' → 'this morning'."""

    def test_today_morning_mapped(self):
        """'today morning' should be mapped to 'this morning'."""
        from modeling_studio.data.cultural_mappings import INDIAN_ENGLISH_MAPPINGS

        assert "today morning" in INDIAN_ENGLISH_MAPPINGS
        assert INDIAN_ENGLISH_MAPPINGS["today morning"] == "this morning"

    def test_today_evening_mapped(self):
        """'today evening' should be mapped."""
        from modeling_studio.data.cultural_mappings import INDIAN_ENGLISH_MAPPINGS

        assert "today evening" in INDIAN_ENGLISH_MAPPINGS


class TestLakhCroreMapping:
    """Test Indian number words mapped."""

    def test_lakh_mapped(self):
        """'lakh' should be mapped to 'hundred thousand'."""
        from modeling_studio.data.cultural_mappings import INDIAN_ENGLISH_MAPPINGS

        assert "lakh" in INDIAN_ENGLISH_MAPPINGS
        assert INDIAN_ENGLISH_MAPPINGS["lakh"] == "hundred thousand"

    def test_crore_mapped(self):
        """'crore' should be mapped to 'ten million'."""
        from modeling_studio.data.cultural_mappings import INDIAN_ENGLISH_MAPPINGS

        assert "crore" in INDIAN_ENGLISH_MAPPINGS
        assert INDIAN_ENGLISH_MAPPINGS["crore"] == "ten million"


# =============================================================================
# Indian Venting Patterns Tests
# =============================================================================


class TestIndianVentingPatternsDefined:
    """Test INDIAN_VENTING_PATTERNS is frozenset."""

    def test_venting_patterns_is_frozenset(self):
        """INDIAN_VENTING_PATTERNS should be a frozenset."""
        from modeling_studio.data.cultural_mappings import INDIAN_VENTING_PATTERNS

        assert isinstance(INDIAN_VENTING_PATTERNS, frozenset)

    def test_venting_patterns_not_empty(self):
        """INDIAN_VENTING_PATTERNS should have entries."""
        from modeling_studio.data.cultural_mappings import INDIAN_VENTING_PATTERNS

        assert len(INDIAN_VENTING_PATTERNS) > 0


class TestVentingDieOfEmbarrassment:
    """Test 'I'll die of embarrassment' recognized."""

    def test_die_of_embarrassment_in_patterns(self):
        """'I'll die of embarrassment' should be in patterns."""
        from modeling_studio.data.cultural_mappings import INDIAN_VENTING_PATTERNS

        assert "I'll die of embarrassment" in INDIAN_VENTING_PATTERNS

    def test_dying_of_embarrassment_in_patterns(self):
        """'dying of embarrassment' should be in patterns."""
        from modeling_studio.data.cultural_mappings import INDIAN_VENTING_PATTERNS

        assert "dying of embarrassment" in INDIAN_VENTING_PATTERNS


class TestVentingKillingMe:
    """Test 'this is killing me' recognized."""

    def test_killing_me_in_patterns(self):
        """'this is killing me' should be in patterns."""
        from modeling_studio.data.cultural_mappings import INDIAN_VENTING_PATTERNS

        assert "this is killing me" in INDIAN_VENTING_PATTERNS

    def test_youre_killing_me_in_patterns(self):
        """'you're killing me' should be in patterns."""
        from modeling_studio.data.cultural_mappings import INDIAN_VENTING_PATTERNS

        assert "you're killing me" in INDIAN_VENTING_PATTERNS


class TestVentingHeadBursting:
    """Test 'my head will burst' recognized."""

    def test_head_will_burst_in_patterns(self):
        """'my head will burst' should be in patterns."""
        from modeling_studio.data.cultural_mappings import INDIAN_VENTING_PATTERNS

        assert "my head will burst" in INDIAN_VENTING_PATTERNS

    def test_head_is_bursting_in_patterns(self):
        """'head is bursting' should be in patterns."""
        from modeling_studio.data.cultural_mappings import INDIAN_VENTING_PATTERNS

        assert "head is bursting" in INDIAN_VENTING_PATTERNS


class TestVentingPatternsNotCrisis:
    """Test venting patterns should NOT trigger CRISIS."""

    def test_is_venting_function_exists(self):
        """is_venting function should exist."""
        from modeling_studio.data.cultural_mappings import is_venting

        assert callable(is_venting)

    def test_venting_expressions_detected(self):
        """Venting expressions should be detected."""
        from modeling_studio.data.cultural_mappings import is_venting

        assert is_venting("I'll die of embarrassment") is True
        assert is_venting("this is killing me") is True

    def test_non_venting_not_detected(self):
        """Non-venting text should not be detected."""
        from modeling_studio.data.cultural_mappings import is_venting

        assert is_venting("I had a good day today") is False

    def test_normalizer_venting_detection(self):
        """IndianEnglishNormalizer should detect venting."""
        from modeling_studio.data.cultural_mappings import IndianEnglishNormalizer

        normalizer = IndianEnglishNormalizer()
        assert normalizer.is_venting_expression("my head will burst") is True


# =============================================================================
# Indian Kinship All Regions Tests
# =============================================================================


class TestIndianKinshipAllRegions:
    """Test North, South, Bengali, Marathi, Gujarati covered."""

    def test_kinship_variants_defined(self):
        """KINSHIP_VARIANTS should be defined."""
        from modeling_studio.data.cultural_mappings import KINSHIP_VARIANTS

        assert isinstance(KINSHIP_VARIANTS, dict)
        assert len(KINSHIP_VARIANTS) > 0

    def test_north_indian_terms(self):
        """Should include North Indian terms."""
        from modeling_studio.data.cultural_mappings import KINSHIP_VARIANTS

        # Check mom variants include North Indian terms
        mom_variants = KINSHIP_VARIANTS.get("mom", frozenset())
        assert "mummy" in mom_variants or "amma" in mom_variants

    def test_south_indian_terms(self):
        """Should include South Indian terms."""
        from modeling_studio.data.cultural_mappings import KINSHIP_VARIANTS

        # Check dad variants include South Indian terms
        dad_variants = KINSHIP_VARIANTS.get("dad", frozenset())
        assert "appa" in dad_variants

    def test_bengali_terms(self):
        """Should include Bengali terms."""
        from modeling_studio.data.cultural_mappings import KINSHIP_VARIANTS

        # Check dad variants include Bengali terms
        dad_variants = KINSHIP_VARIANTS.get("dad", frozenset())
        assert "baba" in dad_variants

    def test_marathi_terms(self):
        """Should include Marathi terms."""
        from modeling_studio.data.cultural_mappings import KINSHIP_VARIANTS

        # Check mom variants include Marathi terms
        mom_variants = KINSHIP_VARIANTS.get("mom", frozenset())
        assert "aai" in mom_variants


class TestKinshipMaternalPaternal:
    """Test maternal/paternal distinctions preserved."""

    def test_grandma_maternal_exists(self):
        """Maternal grandmother variant should exist."""
        from modeling_studio.data.cultural_mappings import KINSHIP_VARIANTS

        assert "grandma_maternal" in KINSHIP_VARIANTS

    def test_grandma_paternal_exists(self):
        """Paternal grandmother variant should exist."""
        from modeling_studio.data.cultural_mappings import KINSHIP_VARIANTS

        assert "grandma_paternal" in KINSHIP_VARIANTS

    def test_maternal_has_nani(self):
        """Maternal grandmother should include 'nani'."""
        from modeling_studio.data.cultural_mappings import KINSHIP_VARIANTS

        maternal = KINSHIP_VARIANTS.get("grandma_maternal", frozenset())
        assert "nani" in maternal or "naani" in maternal

    def test_paternal_has_dadi(self):
        """Paternal grandmother should include 'dadi'."""
        from modeling_studio.data.cultural_mappings import KINSHIP_VARIANTS

        paternal = KINSHIP_VARIANTS.get("grandma_paternal", frozenset())
        assert "dadi" in paternal or "daadi" in paternal

    def test_uncle_maternal_exists(self):
        """Maternal uncle variant should exist."""
        from modeling_studio.data.cultural_mappings import KINSHIP_VARIANTS

        assert "uncle_maternal" in KINSHIP_VARIANTS

    def test_uncle_paternal_exists(self):
        """Paternal uncle variant should exist."""
        from modeling_studio.data.cultural_mappings import KINSHIP_VARIANTS

        assert "uncle_paternal" in KINSHIP_VARIANTS


# =============================================================================
# IndianEnglishNormalizer Tests
# =============================================================================


class TestNormalizerClassExists:
    """Test IndianEnglishNormalizer class defined."""

    def test_class_exists(self):
        """IndianEnglishNormalizer should exist."""
        from modeling_studio.data.cultural_mappings import IndianEnglishNormalizer

        normalizer = IndianEnglishNormalizer()
        assert normalizer is not None

    def test_class_has_normalize_method(self):
        """IndianEnglishNormalizer should have normalize method."""
        from modeling_studio.data.cultural_mappings import IndianEnglishNormalizer

        normalizer = IndianEnglishNormalizer()
        assert hasattr(normalizer, "normalize")
        assert callable(normalizer.normalize)


class TestNormalizerApply:
    """Test Normalizer transforms text correctly."""

    def test_normalize_needful(self):
        """Normalizer should transform 'needful' expression."""
        from modeling_studio.data.cultural_mappings import IndianEnglishNormalizer

        normalizer = IndianEnglishNormalizer()
        result = normalizer.normalize("I'll be doing the needful")
        assert "what's needed" in result

    def test_normalize_revert_back(self):
        """Normalizer should transform 'revert back' expression."""
        from modeling_studio.data.cultural_mappings import IndianEnglishNormalizer

        normalizer = IndianEnglishNormalizer()
        result = normalizer.normalize("Please revert back to me")
        assert "respond" in result

    def test_normalize_preserves_non_indian(self):
        """Normalizer should preserve non-Indian English."""
        from modeling_studio.data.cultural_mappings import IndianEnglishNormalizer

        normalizer = IndianEnglishNormalizer()
        original = "I had a great day today"
        result = normalizer.normalize(original)
        assert result == original

    def test_normalize_empty_string(self):
        """Normalizer should handle empty string."""
        from modeling_studio.data.cultural_mappings import IndianEnglishNormalizer

        normalizer = IndianEnglishNormalizer()
        result = normalizer.normalize("")
        assert result == ""

    def test_normalize_convenience_function(self):
        """normalize_indian_english convenience function should work."""
        from modeling_studio.data.cultural_mappings import normalize_indian_english

        result = normalize_indian_english("doing the needful")
        assert "what's needed" in result


class TestNormalizerKinshipDetection:
    """Test normalizer kinship term detection."""

    def test_is_kinship_term_method(self):
        """Normalizer should have is_kinship_term method."""
        from modeling_studio.data.cultural_mappings import IndianEnglishNormalizer

        normalizer = IndianEnglishNormalizer()
        assert hasattr(normalizer, "is_kinship_term")

    def test_detects_kinship_terms(self):
        """Normalizer should detect kinship terms."""
        from modeling_studio.data.cultural_mappings import IndianEnglishNormalizer

        normalizer = IndianEnglishNormalizer()
        assert normalizer.is_kinship_term("amma") is True
        assert normalizer.is_kinship_term("appa") is True
        assert normalizer.is_kinship_term("random") is False

    def test_get_kinship_category(self):
        """Normalizer should get kinship category."""
        from modeling_studio.data.cultural_mappings import IndianEnglishNormalizer

        normalizer = IndianEnglishNormalizer()
        category = normalizer.get_kinship_category("amma")
        assert category == "mom"


# =============================================================================
# Family Structure Types Tests
# =============================================================================


class TestFamilyStructureTypes:
    """Test family structure classifications defined."""

    def test_family_structure_types_defined(self):
        """FAMILY_STRUCTURE_TYPES should be defined."""
        from modeling_studio.data.cultural_mappings import FAMILY_STRUCTURE_TYPES

        assert isinstance(FAMILY_STRUCTURE_TYPES, dict)
        assert len(FAMILY_STRUCTURE_TYPES) > 0

    def test_nuclear_family_defined(self):
        """Nuclear family structure should be defined."""
        from modeling_studio.data.cultural_mappings import FAMILY_STRUCTURE_TYPES

        assert "nuclear" in FAMILY_STRUCTURE_TYPES

    def test_joint_family_defined(self):
        """Joint family structure should be defined."""
        from modeling_studio.data.cultural_mappings import FAMILY_STRUCTURE_TYPES

        assert "joint_family" in FAMILY_STRUCTURE_TYPES

    def test_extended_family_defined(self):
        """Extended family structure should be defined."""
        from modeling_studio.data.cultural_mappings import FAMILY_STRUCTURE_TYPES

        assert "extended" in FAMILY_STRUCTURE_TYPES

    def test_family_structure_type_class(self):
        """FamilyStructureType class should exist."""
        from modeling_studio.data.cultural_mappings import FamilyStructureType

        structure = FamilyStructureType(
            name="test",
            description="Test structure",
            typical_members=["a", "b"],
            cultural_notes="Test notes",
        )
        assert structure.name == "test"
        assert structure.description == "Test structure"

    def test_structure_has_attributes(self):
        """Family structures should have required attributes."""
        from modeling_studio.data.cultural_mappings import FAMILY_STRUCTURE_TYPES

        nuclear = FAMILY_STRUCTURE_TYPES["nuclear"]
        assert hasattr(nuclear, "name")
        assert hasattr(nuclear, "description")
        assert hasattr(nuclear, "typical_members")
        assert hasattr(nuclear, "cultural_notes")


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Test that all public APIs are exported."""

    def test_all_exports_defined(self):
        """__all__ should be defined with public APIs."""
        from modeling_studio.data import cultural_mappings

        assert hasattr(cultural_mappings, "__all__")
        assert "INDIAN_ENGLISH_MAPPINGS" in cultural_mappings.__all__
        assert "INDIAN_VENTING_PATTERNS" in cultural_mappings.__all__
        assert "KINSHIP_VARIANTS" in cultural_mappings.__all__
        assert "IndianEnglishNormalizer" in cultural_mappings.__all__

    def test_utility_functions_exported(self):
        """Utility functions should be exported."""
        from modeling_studio.data import cultural_mappings

        assert "normalize_indian_english" in cultural_mappings.__all__
        assert "is_venting" in cultural_mappings.__all__
        assert "get_kinship_variants" in cultural_mappings.__all__

    def test_get_kinship_variants_function(self):
        """get_kinship_variants function should work."""
        from modeling_studio.data.cultural_mappings import get_kinship_variants

        variants = get_kinship_variants("mom")
        assert isinstance(variants, frozenset)
        assert len(variants) > 0
