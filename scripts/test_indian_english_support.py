#!/usr/bin/env python3
"""Test Indian English Support - Issue 3.6.7 Acceptance Criteria."""

from modeling_studio.data.cultural_mappings import (
    INDIAN_ENGLISH_MAPPINGS,
    INDIAN_VENTING_PATTERNS,
    KINSHIP_VARIANTS,
    IndianEnglishNormalizer,
)


def test_indian_english_support():
    print("=" * 60)
    print("Issue 3.6.7: Indian English Support Test")
    print("=" * 60)
    print()

    normalizer = IndianEnglishNormalizer()

    # Expression mapping
    print("--- Expression Mapping ---")
    result = normalizer.normalize("doing the needful")
    print(f'  "doing the needful" -> "{result}"')
    assert result == "doing what's needed", f"Expected 'doing what's needed', got '{result}'"
    print("  ✓ PASS")

    result = normalizer.normalize("I passed out from college")
    print(f'  "I passed out from college" -> "{result}"')
    assert (
        result == "I graduated from college"
    ), f"Expected 'I graduated from college', got '{result}'"
    print("  ✓ PASS")
    print()

    # Kinship variants
    print("--- Kinship Variants ---")
    assert "amma" in KINSHIP_VARIANTS["mom"], "amma not in mom variants"
    print('  ✓ "amma" in KINSHIP_VARIANTS["mom"]: True')

    assert "bhai" in KINSHIP_VARIANTS["brother"], "bhai not in brother variants"
    print('  ✓ "bhai" in KINSHIP_VARIANTS["brother"]: True')

    print(f"  Total kinship categories: {len(KINSHIP_VARIANTS)}")
    print()

    # Venting patterns (should NOT trigger CRISIS)
    print("--- Venting Patterns (Safety FP Prevention) ---")
    assert "I'll die of embarrassment" in INDIAN_VENTING_PATTERNS
    print('  ✓ "I\'ll die of embarrassment" in INDIAN_VENTING_PATTERNS')

    # Test venting detection
    assert normalizer.is_venting_expression("I'll die of embarrassment")
    print('  ✓ is_venting_expression("I\'ll die of embarrassment") = True')

    assert normalizer.is_venting_expression("kill me now")
    print('  ✓ is_venting_expression("kill me now") = True')

    print(f"  Total venting patterns: {len(INDIAN_VENTING_PATTERNS)}")
    print()

    # Additional features
    print("--- Additional Features ---")
    print(f"  INDIAN_ENGLISH_MAPPINGS: {len(INDIAN_ENGLISH_MAPPINGS)} expressions")
    categories = list(KINSHIP_VARIANTS.keys())[:5]
    print(f"  Kinship categories: {categories}...")

    # Test kinship extraction
    terms = normalizer.extract_kinship_terms("Amma and Papa went to the market")
    print('  extract_kinship_terms("Amma and Papa went to the market"):')
    for term, cat, start, end in terms:
        print(f"    {term} -> {cat} [{start}:{end}]")
    print()

    # Test kinship category lookup
    print("--- Kinship Category Lookup ---")
    cat = normalizer.get_kinship_category("didi")
    print(f'  get_kinship_category("didi") = {cat}')

    cat = normalizer.get_kinship_category("bhaiya")
    print(f'  get_kinship_category("bhaiya") = {cat}')
    print()

    print("=" * 60)
    print("✅ Indian English support complete")
    print("=" * 60)


if __name__ == "__main__":
    test_indian_english_support()
