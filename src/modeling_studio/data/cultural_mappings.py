"""
Indian English Cultural Mappings and Normalization

This module provides comprehensive support for Indian English expressions,
kinship terminology, and cultural patterns to improve model accuracy for
FamilyOS users in India.

Key Components:
    - INDIAN_ENGLISH_MAPPINGS: Expression normalization dictionary
    - INDIAN_VENTING_PATTERNS: Hyperbolic expressions for safety FP prevention
    - KINSHIP_VARIANTS: Indian family term variations (regional + religious)
    - FAMILY_STRUCTURE_TYPES: Family structure classifications
    - IndianEnglishNormalizer: Preprocessing class

Regional Coverage:
    - North Indian: Hindi-influenced terms (mummy, papa, bhai, didi)
    - South Indian: Tamil/Telugu/Malayalam/Kannada terms (amma, appa, anna, akka)
    - Bengali: Bengali-influenced terms (baba, ma, dada, didi)
    - Marathi: Marathi terms (aai, baba, dada, tai)
    - Gujarati: Gujarati terms (mummy, papa, bhai, ben)

Issue: 3.6.7 - Implement Indian English Support
Epic: 3.6 - Production Readiness
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from re import Pattern

logger = logging.getLogger(__name__)


# =============================================================================
# Indian English Expression Mappings
# =============================================================================

# Common Indian English expressions mapped to standard equivalents
# Format: {"indian_expression": "standard_expression"}
INDIAN_ENGLISH_MAPPINGS: dict[str, str] = {
    # Professional/Formal expressions
    "doing the needful": "doing what's needed",
    "do the needful": "do what's needed",
    "kindly do the needful": "please do what's needed",
    "revert back": "respond",
    "revert to me": "respond to me",
    "please revert": "please respond",
    "kindly revert": "please respond",
    "prepone": "move earlier",
    "preponed": "moved earlier",
    "updation": "update",
    "updations": "updates",
    "upgradation": "upgrade",
    "mugging up": "memorizing",
    "mugged up": "memorized",
    "by hearting": "memorizing",
    "by heart": "memorize",
    # Educational expressions
    "passed out from college": "graduated from college",
    "passed out from school": "graduated from school",
    "passed out": "graduated",
    "pass out ceremony": "graduation ceremony",
    "backlog": "failed subject",
    "backlogs": "failed subjects",
    "have a doubt": "have a question",
    "having doubt": "having a question",
    "doubts": "questions",
    "clear my doubt": "answer my question",
    "out of station": "out of town",
    "native place": "hometown",
    "nativity": "hometown",
    # Time expressions
    "today morning": "this morning",
    "today evening": "this evening",
    "today night": "tonight",
    "yesterday night": "last night",
    "tomorrow morning": "tomorrow morning",
    "day after": "the day after tomorrow",
    "day before": "the day before yesterday",
    "one week back": "a week ago",
    "two days back": "two days ago",
    "back then": "at that time",
    # Quantity/Degree expressions
    "lakh": "hundred thousand",
    "lakhs": "hundred thousands",
    "crore": "ten million",
    "crores": "ten millions",
    "paining": "hurting",
    "pained": "hurt",
    # Food/Lifestyle
    "taking food": "eating",
    "took food": "ate",
    "non-veg": "non-vegetarian",
    "veg": "vegetarian",
    "pure veg": "strictly vegetarian",
    "tiffin": "lunch box",
    # Relationship expressions
    "love marriage": "marriage for love",
    "arranged marriage": "arranged marriage",
    "inter-caste": "inter-caste",
    "same gotra": "same clan",
    # Miscellaneous
    "shifting": "moving",
    "shifted house": "moved house",
    "cousin brother": "male cousin",
    "cousin sister": "female cousin",
    "co-brother": "brother-in-law",
    "co-sister": "sister-in-law",
    "family friend": "friend of the family",
    "foreign": "abroad",
    "foreign return": "returned from abroad",
    # Affirmations/Responses
    "no issues": "no problem",
    "no problem only": "no problem at all",
    "itself": "only",
    "only na": "right",
    "no na": "isn't it",
    "what all": "what things",
    "how come": "why",
    "why like this": "why is this happening",
}


# =============================================================================
# Indian Venting Patterns (Safety FP Prevention)
# =============================================================================

# These hyperbolic expressions are common in Indian English and should NOT
# trigger CRISIS safety flags. They are expressions of frustration, not
# actual harm intent.
INDIAN_VENTING_PATTERNS: frozenset[str] = frozenset(
    {
        # Death/Dying hyperbole (very common in India)
        "I'll die of embarrassment",
        "die of embarrassment",
        "dying of embarrassment",
        "I could die",
        "just kill me now",
        "kill me",
        "dying here",
        "I'm dying",
        "death would be better",
        "I'll die laughing",
        "dying laughing",
        "I'm dead",
        "I died",
        "this is killing me",
        "you're killing me",
        "I'll die of shame",
        "I want to die of shame",
        "so embarrassed I could die",
        "I'll die if",
        "I would die",
        # Frustration expressions
        "I'll go mad",
        "going mad",
        "I'm going crazy",
        "driving me crazy",
        "making me crazy",
        "I'll lose my mind",
        "losing my mind",
        "I'm losing it",
        "this is madness",
        "it's too much",
        "can't take it anymore",
        "had it up to here",
        "at my wit's end",
        "stressed to the max",
        "super stressed",
        # Dramatic expressions
        "I'll burst",
        "about to burst",
        "my head will burst",
        "head is bursting",
        "heart will burst",
        "I'll explode",
        "about to explode",
        "I'm finished",
        "it's all over",
        "my life is over",
        "this is the end",
        "I'm done for",
        "I'm ruined",
        "I'm destroyed",
        "completely destroyed",
        # Academic/Work stress
        "I'll fail",
        "going to fail",
        "I'm a failure",
        "total failure",
        "board exams will kill me",
        "exams are killing me",
        "this job is killing me",
        "work is killing me",
        "boss is killing me",
        "deadlines are killing me",
        "stress is killing me",
        "pressure is killing me",
        # Family/Social pressure
        "parents will kill me",
        "mom will kill me",
        "dad will kill me",
        "mummy will kill me",
        "papa will kill me",
        "amma will kill me",
        "they'll kill me",
        "family will kill me",
        "in-laws will kill me",
        "society will kill me",
        # Shame/Honor expressions
        "what will people say",
        "log kya kahenge",
        "shame on the family",
        "bring shame",
        "family honor",
        "family name",
        "reputation ruined",
        "how will I show my face",
        "can't show my face",
        "too ashamed",
        "so much shame",
        # Physical discomfort hyperbole
        "dying of heat",
        "dying of cold",
        "dying of hunger",
        "dying of thirst",
        "starving to death",
        "freezing to death",
        "boiling to death",
        "sweating to death",
        "tired to death",
        "bored to death",
        "waiting to death",
        "working to death",
        # Emotional hyperbole
        "heart is breaking",
        "heart broke",
        "heartbroken",
        "crying rivers",
        "cried my eyes out",
        "tears won't stop",
        "so hurt",
        "deeply hurt",
        "wounded",
        "can't bear it",
        "unbearable",
        # Requests for dramatic help
        "someone save me",
        "God save me",
        "bhagwan bachao",
        "help me God",
        "kill me now",
        "shoot me",
        "end my misery",
        "put me out of my misery",
    }
)


# =============================================================================
# Kinship Variants
# =============================================================================

# Comprehensive mapping of standard kinship terms to Indian variants
# Organized by relationship, including regional and religious variations
KINSHIP_VARIANTS: dict[str, frozenset[str]] = {
    # Mother
    "mom": frozenset(
        {
            "ma",
            "maa",
            "mummy",
            "mumma",
            "mommy",
            "mother",
            # Hindi/North Indian
            "amma",
            "ammi",
            "ammiji",
            # South Indian
            "avva",
            "aayi",
            "thayi",
            # Bengali (ma, maa already listed)
            # Marathi
            "aai",
            "aaii",
            # Gujarati
            "ba",
            # Punjabi
            "bebe",
            "mataji",
            # Muslim
            "ammijaan",
            "walida",
            # Formal
            "maaji",
            "matashree",
        }
    ),
    # Father
    "dad": frozenset(
        {
            "pa",
            "papa",
            "daddy",
            "father",
            "pops",
            # Hindi/North Indian
            "bapu",
            "bapuji",
            "pitaji",
            "pita",
            # South Indian
            "appa",
            "nanna",
            "ayya",
            "anna",
            "thaatha",
            # Bengali
            "baba",
            "babu",
            # Marathi
            "dada",
            # Gujarati
            "pappa",
            # Punjabi
            "paaji",
            "papaji",
            # Muslim
            "abbu",
            "abbujaan",
            "abba",
            "walid",
            # Formal
            "pitashree",
        }
    ),
    # Brother (elder)
    "elder_brother": frozenset(
        {
            "bro",
            "brother",
            "big brother",
            "elder brother",
            # Hindi/North Indian
            "bhai",
            "bhaiya",
            "bhaiyya",
            "bhaisaab",
            # South Indian
            "anna",
            "annayya",
            "thambi",
            "akka",
            # Bengali
            "dada",
            "dadabhai",
            # Marathi
            "bhau",
            # Gujarati
            "mota bhai",
            # Muslim
            "bhaijaan",
        }
    ),
    # Brother (younger)
    "brother": frozenset(
        {
            "bro",
            "brother",
            "little brother",
            "younger brother",
            # Hindi
            "bhai",
            "chhota bhai",
            "chhotu",
            # South Indian
            "thambi",
            "thamma",
            "chinna",
            # Bengali
            "chhoto",
            # Others
            "bhaiya",
        }
    ),
    # Sister (elder)
    "elder_sister": frozenset(
        {
            "sis",
            "sister",
            "big sister",
            "elder sister",
            # Hindi/North Indian
            "didi",
            "di",
            "didiya",
            "jiji",
            # South Indian
            "akka",
            "akkayya",
            "chechi",
            "akkan",
            # Bengali
            "didimoni",
            # Marathi
            "tai",
            "taai",
            # Gujarati
            "ben",
            "moti ben",
            # Muslim
            "aapa",
            "aapi",
            "baji",
        }
    ),
    # Sister (younger)
    "sister": frozenset(
        {
            "sis",
            "sister",
            "little sister",
            "younger sister",
            # Hindi
            "chhoti",
            "chhoti bahen",
            "behenji",
            # South Indian
            "thangachi",
            "thangai",
            "chelli",
            # Bengali
            "bon",
            # Others
            "behen",
            "bahen",
        }
    ),
    # Grandmother (maternal)
    "grandma_maternal": frozenset(
        {
            "grandma",
            "grandmother",
            "granny",
            "nana",
            # Hindi
            "naani",
            "naniji",
            "nani",
            # South Indian
            "ammamma",
            "patti",
            "avva",
            "ajji",
            # Bengali
            "didima",
            "didu",
            # Marathi
            "aaji",
            # Gujarati
            "ba",
        }
    ),
    # Grandmother (paternal)
    "grandma_paternal": frozenset(
        {
            "grandma",
            "grandmother",
            "granny",
            # Hindi
            "daadi",
            "dadiji",
            "dadi",
            # South Indian
            "paati",
            "nannamma",
            "ajji",
            "ammamma",
            # Bengali
            "thakurma",
            "thamma",
            # Marathi
            "aaji",
            # Gujarati
            "ba",
        }
    ),
    # Grandfather (maternal)
    "grandpa_maternal": frozenset(
        {
            "grandpa",
            "grandfather",
            "gramps",
            # Hindi
            "naana",
            "nanaji",
            # South Indian
            "thatha",
            "thaathaa",
            "ajja",
            "muthappa",
            # Bengali
            "didun",
            "natun dadu",
            # Marathi
            "aajoba",
            # Gujarati
            "nana",
        }
    ),
    # Grandfather (paternal)
    "grandpa_paternal": frozenset(
        {
            "grandpa",
            "grandfather",
            "gramps",
            # Hindi
            "daada",
            "dadaji",
            "dada",
            # South Indian
            "thatha",
            "thaathaa",
            "ajja",
            "muthappa",
            # Bengali
            "dadu",
            "thakurda",
            # Marathi
            "aajoba",
        }
    ),
    # Aunt (mother's sister)
    "aunt_maternal": frozenset(
        {
            "aunt",
            "auntie",
            "aunty",
            # Hindi
            "mausi",
            "masi",
            "maasiji",
            # South Indian
            "chitti",
            "periamma",
            "athai",
            # Bengali
            "mashima",
            # Marathi
            "mavshi",
        }
    ),
    # Aunt (father's sister)
    "aunt_paternal": frozenset(
        {
            "aunt",
            "auntie",
            "aunty",
            # Hindi
            "bua",
            "fui",
            "buaji",
            # South Indian
            "athai",
            "chitthi",
            # Bengali
            "pishi",
            "pishima",
            # Marathi
            "atya",
            # Gujarati
            "foi",
            "fai",
        }
    ),
    # Uncle (mother's brother)
    "uncle_maternal": frozenset(
        {
            "uncle",
            # Hindi
            "mama",
            "mamaji",
            "mamu",
            # South Indian
            "maamaa",
            "chittappa",
            # Bengali
            "mamoni",
        }
    ),
    # Uncle (father's brother)
    "uncle_paternal": frozenset(
        {
            "uncle",
            # Hindi
            "chacha",
            "chachaji",
            "tau",
            "tauji",
            # South Indian
            "periappa",
            "chittappa",
            "anna",
            # Bengali
            "kaku",
            "jethu",
            # Marathi/Gujarati
            "kaka",
        }
    ),
    # Spouse (wife)
    "wife": frozenset(
        {
            "wife",
            "wifey",
            "spouse",
            "partner",
            # Hindi
            "patni",
            "biwi",
            "gharwali",
            "shrimati",
            # Bengali
            "bou",
            "stri",
            # South Indian
            "pondatti",
            "bharya",
            # Formal
            "dharampatni",
            "ardhangini",
        }
    ),
    # Spouse (husband)
    "husband": frozenset(
        {
            "husband",
            "hubby",
            "spouse",
            "partner",
            # Hindi
            "pati",
            "shauhar",
            "patidev",
            # Bengali
            "swami",
            # South Indian
            "kanavan",
            "bhartru",
            # Formal
            "bhartar",
        }
    ),
    # In-laws
    "mother_in_law": frozenset(
        {
            "mother-in-law",
            "mil",
            # Hindi
            "saas",
            "saasuma",
            "saasji",
            # South Indian
            "maami",
            "atthai",
            # Bengali
            "sasuri",
            "mamoni",
        }
    ),
    "father_in_law": frozenset(
        {
            "father-in-law",
            "fil",
            # Hindi
            "sasur",
            "sasurji",
            # South Indian
            "maama",
            "mamiyaar",
        }
    ),
}


# =============================================================================
# Family Structure Types
# =============================================================================


@dataclass
class FamilyStructureType:
    """Definition of a family structure type."""

    name: str
    description: str
    typical_members: list[str]
    cultural_notes: str


FAMILY_STRUCTURE_TYPES: dict[str, FamilyStructureType] = {
    "nuclear": FamilyStructureType(
        name="nuclear",
        description="Parents and their children living together",
        typical_members=["mother", "father", "children"],
        cultural_notes="Becoming more common in urban India",
    ),
    "joint_family": FamilyStructureType(
        name="joint_family",
        description="Extended family living together (traditional Indian)",
        typical_members=[
            "grandparents",
            "parents",
            "children",
            "uncles",
            "aunts",
            "cousins",
        ],
        cultural_notes="Traditional Indian family structure, common in rural areas",
    ),
    "extended": FamilyStructureType(
        name="extended",
        description="Nuclear family with frequent involvement of extended relatives",
        typical_members=["parents", "children", "grandparents_nearby", "relatives"],
        cultural_notes="Common in semi-urban India, grandparents may live nearby",
    ),
    "blended": FamilyStructureType(
        name="blended",
        description="Family with step-parents or step-siblings",
        typical_members=["step_parent", "biological_parent", "step_siblings", "siblings"],
        cultural_notes="Less common but increasing in urban India",
    ),
    "single_parent": FamilyStructureType(
        name="single_parent",
        description="One parent raising children",
        typical_members=["parent", "children"],
        cultural_notes="Increasing in India, still carries social stigma in some areas",
    ),
    "multigenerational": FamilyStructureType(
        name="multigenerational",
        description="Three or more generations living together",
        typical_members=["great_grandparents", "grandparents", "parents", "children"],
        cultural_notes="Common in traditional families, especially with elderly care",
    ),
    "child_free": FamilyStructureType(
        name="child_free",
        description="Married couple without children",
        typical_members=["spouse1", "spouse2"],
        cultural_notes="Uncommon but increasing in urban India",
    ),
}


# =============================================================================
# Indian English Normalizer
# =============================================================================


class IndianEnglishNormalizer:
    """
    Normalizer for Indian English expressions.

    This class preprocesses text to normalize Indian English expressions
    to their standard equivalents while preserving meaning.

    Features:
        - Expression mapping (needful → what's needed)
        - Kinship term recognition
        - Venting pattern detection (for safety FP prevention)
        - Case-insensitive matching with original case preservation

    Example:
        >>> normalizer = IndianEnglishNormalizer()
        >>> normalizer.normalize("I'll be doing the needful")
        "I'll be doing what's needed"
        >>> normalizer.normalize("Mummy said to come home")
        "Mummy said to come home"  # Kinship terms preserved
    """

    def __init__(
        self,
        preserve_kinship: bool = True,
        normalize_expressions: bool = True,
        detect_venting: bool = True,
    ):
        """
        Initialize the normalizer.

        Args:
            preserve_kinship: Keep kinship terms as-is (don't normalize)
            normalize_expressions: Apply expression mappings
            detect_venting: Enable venting pattern detection
        """
        self.preserve_kinship = preserve_kinship
        self.normalize_expressions = normalize_expressions
        self.detect_venting = detect_venting

        # Build compiled patterns for efficient matching
        self._expression_patterns: list[tuple[Pattern[str], str]] = []
        if normalize_expressions:
            self._build_expression_patterns()

        # Build kinship term set for recognition
        self._kinship_terms: set[str] = set()
        if preserve_kinship:
            self._build_kinship_set()

        # Build venting patterns
        self._venting_patterns: list[Pattern[str]] = []
        if detect_venting:
            self._build_venting_patterns()

    def _build_expression_patterns(self) -> None:
        """Build compiled regex patterns for expressions."""
        # Sort by length (longest first) to match longer phrases first
        sorted_expressions = sorted(
            INDIAN_ENGLISH_MAPPINGS.items(), key=lambda x: len(x[0]), reverse=True
        )

        for indian_expr, standard_expr in sorted_expressions:
            # Create case-insensitive pattern with word boundaries
            pattern = re.compile(r"\b" + re.escape(indian_expr) + r"\b", re.IGNORECASE)
            self._expression_patterns.append((pattern, standard_expr))

    def _build_kinship_set(self) -> None:
        """Build set of all kinship terms."""
        for variants in KINSHIP_VARIANTS.values():
            self._kinship_terms.update(term.lower() for term in variants)

    def _build_venting_patterns(self) -> None:
        """Build compiled patterns for venting expressions."""
        for pattern in INDIAN_VENTING_PATTERNS:
            compiled = re.compile(r"\b" + re.escape(pattern) + r"\b", re.IGNORECASE)
            self._venting_patterns.append(compiled)

    def normalize(self, text: str) -> str:
        """
        Normalize Indian English expressions in text.

        Args:
            text: Input text with potential Indian English expressions

        Returns:
            Text with expressions normalized to standard English
        """
        if not text:
            return text

        result = text

        # Apply expression mappings
        for pattern, replacement in self._expression_patterns:
            result = pattern.sub(replacement, result)

        return result

    def is_kinship_term(self, word: str) -> bool:
        """
        Check if a word is a recognized kinship term.

        Args:
            word: Word to check

        Returns:
            True if word is a kinship term
        """
        return word.lower() in self._kinship_terms

    def get_kinship_category(self, term: str) -> str | None:
        """
        Get the standard kinship category for a term.

        Args:
            term: Kinship term to look up

        Returns:
            Standard category (e.g., "mom", "dad") or None if not found
        """
        term_lower = term.lower()
        for category, variants in KINSHIP_VARIANTS.items():
            if term_lower in {v.lower() for v in variants}:
                return category
        return None

    def is_venting_expression(self, text: str) -> bool:
        """
        Check if text contains venting/hyperbolic expressions.

        These expressions should NOT trigger safety escalations as they
        are common cultural expressions of frustration.

        Args:
            text: Text to check

        Returns:
            True if text contains venting expressions
        """
        for pattern in self._venting_patterns:
            if pattern.search(text):
                return True
        return False

    def get_venting_matches(self, text: str) -> list[str]:
        """
        Get all venting expressions found in text.

        Args:
            text: Text to check

        Returns:
            List of matched venting expressions
        """
        matches = []
        for pattern in self._venting_patterns:
            found = pattern.findall(text)
            matches.extend(found)
        return matches

    def extract_kinship_terms(self, text: str) -> list[tuple[str, str, int, int]]:
        """
        Extract kinship terms from text with their positions.

        Args:
            text: Text to analyze

        Returns:
            List of (term, category, start, end) tuples
        """
        results = []
        words = re.finditer(r"\b\w+\b", text)

        for match in words:
            word = match.group()
            category = self.get_kinship_category(word)
            if category:
                results.append((word, category, match.start(), match.end()))

        return results


# =============================================================================
# Utility Functions
# =============================================================================


def normalize_indian_english(text: str) -> str:
    """
    Convenience function to normalize Indian English expressions.

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    normalizer = IndianEnglishNormalizer()
    return normalizer.normalize(text)


def is_venting(text: str) -> bool:
    """
    Check if text is a venting expression (for safety).

    Args:
        text: Input text

    Returns:
        True if text contains hyperbolic venting
    """
    normalizer = IndianEnglishNormalizer()
    return normalizer.is_venting_expression(text)


def get_kinship_variants(standard_term: str) -> frozenset[str]:
    """
    Get all Indian variants for a standard kinship term.

    Args:
        standard_term: Standard term (e.g., "mom", "dad")

    Returns:
        Set of variant terms
    """
    return KINSHIP_VARIANTS.get(standard_term, frozenset())


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Mappings
    "INDIAN_ENGLISH_MAPPINGS",
    "INDIAN_VENTING_PATTERNS",
    "KINSHIP_VARIANTS",
    "FAMILY_STRUCTURE_TYPES",
    # Classes
    "FamilyStructureType",
    "IndianEnglishNormalizer",
    # Functions
    "normalize_indian_english",
    "is_venting",
    "get_kinship_variants",
]
