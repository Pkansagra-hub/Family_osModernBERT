"""
Data Augmentation for FamilyOS Multi-Task Learning

This module provides data augmentation utilities specifically designed for
family-related text data, with support for:
    - Kinship term variations across cultures (mom→mum/mummy/amma/aai)
    - Nickname pattern augmentation
    - Back-translation for paraphrase generation
    - Synonym replacement
    - Character-level augmentation for robustness

Key Classes:
    - FamilyAugmenter: Main augmentation class with kinship/nickname variations
    - BackTranslator: Paraphrase generation via back-translation

Augmentation Strategies:
    - Kinship replacement: Replace family terms with cultural variants
    - Nickname generation: Create plausible family nickname variations
    - Back-translation: Generate paraphrases via translation round-trip
    - Random masking: Prepare data for MLM-style training

Usage:
    from modeling_studio.data.augmentation import FamilyAugmenter, back_translate

    augmenter = FamilyAugmenter()
    augmented = augmenter.augment_kinship("Mom made dinner")
    # ["Mum made dinner", "Mummy made dinner", "Amma made dinner", ...]

    paraphrases = back_translate("Had a great day with family")
    # ["Spent a wonderful day with relatives", ...]
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Kinship Term Variations (Multi-Cultural)
# =============================================================================

# English variants of "mother"
MOTHER_VARIANTS = [
    "mom",
    "mum",
    "mommy",
    "mummy",
    "mama",
    "ma",
    "mother",
    "momma",
    "mam",
]

# Indian variants of "mother"
MOTHER_VARIANTS_INDIAN = [
    "amma",
    "aai",
    "maa",
    "maaji",
    "mataji",
    "ammi",
    "ummi",  # Urdu/Arabic influence
]

# All mother variants
ALL_MOTHER_VARIANTS = MOTHER_VARIANTS + MOTHER_VARIANTS_INDIAN

# English variants of "father"
FATHER_VARIANTS = [
    "dad",
    "daddy",
    "papa",
    "pa",
    "pops",
    "father",
    "dada",
    "pop",
]

# Indian variants of "father"
FATHER_VARIANTS_INDIAN = [
    "appa",
    "baba",
    "babuji",
    "pitaji",
    "abba",
    "abbu",  # Urdu influence
]

# All father variants
ALL_FATHER_VARIANTS = FATHER_VARIANTS + FATHER_VARIANTS_INDIAN

# Grandmother variants
GRANDMOTHER_VARIANTS = [
    "grandma",
    "grandmother",
    "granny",
    "nana",
    "nan",
    "gram",
    "grammy",
    "grams",
    "memaw",
    "meemaw",
    # Indian
    "dadi",
    "nani",
    "ammamma",
    "ajji",
    "aaji",
    # Spanish
    "abuela",
    "abuelita",
    # Filipino
    "lola",
]

# Grandfather variants
GRANDFATHER_VARIANTS = [
    "grandpa",
    "grandfather",
    "gramps",
    "granddad",
    "grandad",
    "papaw",
    "pawpaw",
    "pops",
    "granpop",
    # Indian
    "dada",
    "nana",
    "thatha",
    "ajja",
    "aaji",
    # Spanish
    "abuelo",
    "abuelito",
    # Filipino
    "lolo",
]

# Brother variants
BROTHER_VARIANTS = [
    "brother",
    "bro",
    "bruh",
    # Indian
    "bhai",
    "bhaiya",
    "anna",
    "dada",
    # Spanish
    "hermano",
    # Filipino
    "kuya",
]

# Sister variants
SISTER_VARIANTS = [
    "sister",
    "sis",
    # Indian
    "didi",
    "di",
    "akka",
    "chechi",
    # Spanish
    "hermana",
    # Filipino
    "ate",
]

# Uncle variants
UNCLE_VARIANTS = [
    "uncle",
    # Indian - distinguishes maternal/paternal
    "chacha",
    "mama",
    "tau",
    "fufa",
    "mausa",
    # Spanish
    "tio",
    # Filipino
    "tito",
]

# Aunt variants
AUNT_VARIANTS = [
    "aunt",
    "auntie",
    "aunty",
    # Indian
    "chachi",
    "mami",
    "tai",
    "bua",
    "mausi",
    # Spanish
    "tia",
    # Filipino
    "tita",
]

# Son variants
SON_VARIANTS = [
    "son",
    "boy",
    # Indian
    "beta",
    "bachcha",
    "puttar",
]

# Daughter variants
DAUGHTER_VARIANTS = [
    "daughter",
    "girl",
    # Indian
    "beti",
    "bachchi",
]

# Complete mapping from standard term to all variants
KINSHIP_VARIANTS: dict[str, list[str]] = {
    "mom": ALL_MOTHER_VARIANTS,
    "mother": ALL_MOTHER_VARIANTS,
    "dad": ALL_FATHER_VARIANTS,
    "father": ALL_FATHER_VARIANTS,
    "grandma": GRANDMOTHER_VARIANTS,
    "grandmother": GRANDMOTHER_VARIANTS,
    "grandpa": GRANDFATHER_VARIANTS,
    "grandfather": GRANDFATHER_VARIANTS,
    "brother": BROTHER_VARIANTS,
    "sister": SISTER_VARIANTS,
    "uncle": UNCLE_VARIANTS,
    "aunt": AUNT_VARIANTS,
    "son": SON_VARIANTS,
    "daughter": DAUGHTER_VARIANTS,
}

# Reverse mapping: variant -> standard term
VARIANT_TO_STANDARD: dict[str, str] = {}
for standard, variants in KINSHIP_VARIANTS.items():
    for variant in variants:
        VARIANT_TO_STANDARD[variant.lower()] = standard


# =============================================================================
# Nickname Patterns
# =============================================================================

# Common nickname patterns for family members
NICKNAME_PATTERNS = [
    # Animal nicknames
    "panda",
    "bunny",
    "bear",
    "teddy",
    "tiger",
    "cub",
    "kitty",
    "puppy",
    "monkey",
    "bug",
    "butterfly",
    # Sweet nicknames
    "sweetie",
    "honey",
    "sugar",
    "cupcake",
    "pumpkin",
    "cookie",
    "muffin",
    "buttercup",
    "angel",
    # Size-based
    "baby",
    "little one",
    "tiny",
    "mini",
    "junior",
    # Affectionate
    "sunshine",
    "star",
    "buddy",
    "champ",
    "kiddo",
    "darling",
    "love",
    "sweetheart",
    "precious",
    # Indian
    "sona",
    "shona",
    "gudiya",
    "guddu",
    "chiku",
    "pappu",
    "babu",
    "raja",
    "rani",
    "chotu",
]

# Nickname templates (for generating variations)
NICKNAME_TEMPLATES = [
    "{name}y",  # Add -y suffix (Bob -> Bobby)
    "{name}ie",  # Add -ie suffix (Rob -> Robbie)
    "little {name}",
    "big {name}",
    "{name} bear",
    "{name} bug",
]


# =============================================================================
# Back-Translation Languages
# =============================================================================

BACKTRANSLATION_LANGUAGES = {
    "hindi": "hi",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "russian": "ru",
    "chinese": "zh",
    "japanese": "ja",
    "korean": "ko",
    "portuguese": "pt",
    "italian": "it",
}


# =============================================================================
# FamilyAugmenter Class
# =============================================================================


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""

    # Kinship augmentation
    kinship_replacement_prob: float = 0.3
    max_kinship_variants: int = 5
    include_indian_variants: bool = True
    include_spanish_variants: bool = True
    include_filipino_variants: bool = True

    # Nickname augmentation
    nickname_replacement_prob: float = 0.2
    max_nickname_variants: int = 3

    # Back-translation
    backtranslation_languages: list[str] = field(
        default_factory=lambda: ["spanish", "french", "german"]
    )
    max_backtranslation_variants: int = 3

    # Random seed
    seed: int | None = None


class FamilyAugmenter:
    """
    Data augmenter for family-related text with kinship and nickname variations.

    Provides multiple augmentation strategies:
        - Kinship term replacement with cultural variants
        - Nickname pattern augmentation
        - Combined augmentation

    Args:
        config: AugmentationConfig or individual settings
        **kwargs: Override config settings

    Example:
        >>> augmenter = FamilyAugmenter()
        >>> augmented = augmenter.augment_kinship("Mom made dinner")
        >>> print(augmented)
        ['Mum made dinner', 'Mummy made dinner', 'Amma made dinner', ...]
    """

    def __init__(
        self,
        config: AugmentationConfig | None = None,
        **kwargs: Any,
    ):
        if config is not None:
            self.config = config
        else:
            self.config = AugmentationConfig(**kwargs)

        if self.config.seed is not None:
            random.seed(self.config.seed)

        # Build kinship patterns for regex matching
        self._kinship_pattern = self._build_kinship_pattern()

    def _build_kinship_pattern(self) -> re.Pattern:
        """Build regex pattern to find kinship terms in text."""
        # Get all variants
        all_variants = set()
        for variants in KINSHIP_VARIANTS.values():
            all_variants.update(v.lower() for v in variants)

        # Sort by length (longest first) to match longer terms first
        sorted_variants = sorted(all_variants, key=len, reverse=True)

        # Build pattern with word boundaries
        pattern = r"\b(" + "|".join(re.escape(v) for v in sorted_variants) + r")\b"
        return re.compile(pattern, re.IGNORECASE)

    def augment_kinship(
        self,
        text: str,
        max_variants: int | None = None,
    ) -> list[str]:
        """
        Augment text by replacing kinship terms with cultural variants.

        Args:
            text: Input text containing kinship terms
            max_variants: Maximum number of variants to generate

        Returns:
            List of augmented texts with kinship term variations
        """
        if max_variants is None:
            max_variants = self.config.max_kinship_variants

        # Find all kinship terms in text
        matches = list(self._kinship_pattern.finditer(text))

        if not matches:
            return []

        augmented = []

        for match in matches:
            original_term = match.group(0)
            term_lower = original_term.lower()

            # Get standard form
            standard = VARIANT_TO_STANDARD.get(term_lower)
            if standard is None:
                continue

            # Get all variants
            variants = KINSHIP_VARIANTS.get(standard, [])

            # Filter variants based on config
            filtered_variants = self._filter_variants(variants, original_term)

            # Generate augmented texts
            for variant in filtered_variants[:max_variants]:
                # Preserve original casing pattern
                if original_term.istitle():
                    variant = variant.title()
                elif original_term.isupper():
                    variant = variant.upper()

                # Replace in text
                aug_text = text[: match.start()] + variant + text[match.end() :]
                if aug_text != text and aug_text not in augmented:
                    augmented.append(aug_text)

        return augmented[:max_variants]

    def _filter_variants(
        self,
        variants: list[str],
        original: str,
    ) -> list[str]:
        """Filter variants based on config settings."""
        filtered = []

        for variant in variants:
            variant_lower = variant.lower()

            # Skip the original
            if variant_lower == original.lower():
                continue

            # Check if it's an Indian variant
            is_indian = variant_lower in (
                set(MOTHER_VARIANTS_INDIAN)
                | set(FATHER_VARIANTS_INDIAN)
                | {"dadi", "nani", "dada", "nana", "bhai", "bhaiya", "didi", "di"}
                | {"chacha", "mama", "tau", "fufa", "mausa"}
                | {"chachi", "mami", "tai", "bua", "mausi"}
                | {"beta", "beti"}
            )

            # Check if it's a Spanish variant
            is_spanish = variant_lower in {
                "abuela",
                "abuelita",
                "abuelo",
                "abuelito",
                "tio",
                "tia",
                "hermano",
                "hermana",
            }

            # Check if it's a Filipino variant
            is_filipino = variant_lower in {"lola", "lolo", "kuya", "ate", "tito", "tita"}

            # Apply filters
            if is_indian and not self.config.include_indian_variants:
                continue
            if is_spanish and not self.config.include_spanish_variants:
                continue
            if is_filipino and not self.config.include_filipino_variants:
                continue

            filtered.append(variant)

        # Shuffle for variety
        random.shuffle(filtered)

        return filtered

    def augment_nicknames(
        self,
        text: str,
        nicknames: list[str] | None = None,
        max_variants: int | None = None,
    ) -> list[str]:
        """
        Augment text by replacing/generating nickname variations.

        Args:
            text: Input text
            nicknames: List of nicknames to find/replace in text
            max_variants: Maximum variants to generate

        Returns:
            List of augmented texts with nickname variations
        """
        if max_variants is None:
            max_variants = self.config.max_nickname_variants

        if nicknames is None:
            # Try to find existing nicknames in text
            nicknames = self._find_nicknames(text)

        if not nicknames:
            return []

        augmented = []

        for nickname in nicknames:
            # Find nickname in text (case-insensitive)
            pattern = re.compile(rf"\b{re.escape(nickname)}\b", re.IGNORECASE)
            if not pattern.search(text):
                continue

            # Generate variations
            variations = self._generate_nickname_variations(nickname)

            for variation in variations[:max_variants]:
                # Preserve original casing
                match = pattern.search(text)
                if match:
                    original = match.group(0)
                    if original.istitle():
                        variation = variation.title()
                    elif original.isupper():
                        variation = variation.upper()

                aug_text = pattern.sub(variation, text, count=1)
                if aug_text != text and aug_text not in augmented:
                    augmented.append(aug_text)

        return augmented[:max_variants]

    def _find_nicknames(self, text: str) -> list[str]:
        """Find potential nicknames in text."""
        found = []
        text_lower = text.lower()

        for nickname in NICKNAME_PATTERNS:
            if nickname.lower() in text_lower:
                found.append(nickname)

        return found

    def _generate_nickname_variations(self, nickname: str) -> list[str]:
        """Generate variations of a nickname."""
        variations = []

        # Similar nicknames from patterns
        for pattern_nickname in NICKNAME_PATTERNS:
            if pattern_nickname.lower() != nickname.lower():
                variations.append(pattern_nickname)

        # Template-based variations
        base_name = nickname.rstrip("yie")
        if len(base_name) >= 2:
            for template in NICKNAME_TEMPLATES:
                try:
                    variation = template.format(name=base_name)
                    if variation.lower() != nickname.lower():
                        variations.append(variation)
                except Exception:
                    pass

        random.shuffle(variations)
        return variations

    def augment(
        self,
        text: str,
        strategies: list[str] | None = None,
        max_total: int = 10,
    ) -> list[str]:
        """
        Apply multiple augmentation strategies to text.

        Args:
            text: Input text
            strategies: List of strategies to apply
                ('kinship', 'nickname', 'all')
            max_total: Maximum total variants across all strategies

        Returns:
            List of augmented texts
        """
        if strategies is None:
            strategies = ["kinship", "nickname"]

        all_augmented: list[str] = []

        if "kinship" in strategies or "all" in strategies:
            kinship_augmented = self.augment_kinship(text)
            all_augmented.extend(kinship_augmented)

        if "nickname" in strategies or "all" in strategies:
            nickname_augmented = self.augment_nicknames(text)
            all_augmented.extend(nickname_augmented)

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for aug in all_augmented:
            if aug not in seen:
                seen.add(aug)
                unique.append(aug)

        return unique[:max_total]

    def augment_batch(
        self,
        texts: list[str],
        strategies: list[str] | None = None,
        max_per_text: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Augment a batch of texts.

        Args:
            texts: List of input texts
            strategies: Augmentation strategies
            max_per_text: Maximum augmentations per text

        Returns:
            List of dicts with 'original' and 'augmented' keys
        """
        results = []

        for text in texts:
            augmented = self.augment(text, strategies, max_per_text)
            results.append(
                {
                    "original": text,
                    "augmented": augmented,
                }
            )

        return results


# =============================================================================
# Back-Translation
# =============================================================================


class BackTranslator:
    """
    Back-translation augmentation via translation round-trip.

    Translates text to intermediate language(s) and back to generate
    paraphrases that preserve meaning but vary in expression.

    Note: Requires a translation model/API. Default implementation
    provides a placeholder that should be replaced with actual translation.

    Args:
        languages: List of intermediate languages
        model_name: Translation model to use (placeholder for API)

    Example:
        >>> translator = BackTranslator(languages=["spanish", "french"])
        >>> paraphrases = translator.translate("Had a great day with family")
    """

    def __init__(
        self,
        languages: list[str] | None = None,
        model_name: str = "placeholder",
    ):
        self.languages = languages or ["spanish", "french", "german"]
        self.model_name = model_name

        # Placeholder for actual translation model
        self._translator = None

        # Try to load translation model if available
        self._try_load_translator()

    def _try_load_translator(self) -> None:
        """Try to load translation model."""
        try:
            # Check if transformers is available
            import importlib.util

            if importlib.util.find_spec("transformers") is not None:
                # Use MarianMT for translation (if available)
                # This is a placeholder - in production, use proper translation
                logger.info("Translation pipeline would be loaded here")
                self._translator = None  # Placeholder
            else:
                raise ImportError("transformers not found")

        except ImportError:
            logger.warning(
                "transformers not available for back-translation. "
                "Using fallback paraphrase generation."
            )
            self._translator = None

    def translate(
        self,
        text: str,
        languages: list[str] | None = None,
    ) -> list[str]:
        """
        Generate paraphrases via back-translation.

        Args:
            text: Input text
            languages: Override intermediate languages

        Returns:
            List of paraphrased texts
        """
        if languages is None:
            languages = self.languages

        if self._translator is not None:
            # Use actual translation
            return self._translate_with_model(text, languages)
        else:
            # Use fallback paraphrase generation
            return self._fallback_paraphrase(text)

    def _translate_with_model(
        self,
        text: str,
        languages: list[str],
    ) -> list[str]:
        """Translate using actual translation model."""
        # Placeholder for actual implementation
        # In production, this would:
        # 1. Translate text to each intermediate language
        # 2. Translate back to English
        # 3. Return unique paraphrases

        paraphrases = []

        for _lang in languages:
            # Placeholder: actual translation would happen here
            # translated = translate_to(text, lang)
            # back_translated = translate_to(translated, "en")
            # paraphrases.append(back_translated)
            pass

        return paraphrases

    def _fallback_paraphrase(self, text: str) -> list[str]:
        """
        Generate simple paraphrases without translation model.

        Uses rule-based substitutions for common patterns.
        """
        paraphrases = []

        # Simple substitution rules (expand as needed)
        substitutions = [
            # Time expressions
            (r"\bhad a great\b", ["had a wonderful", "had an amazing", "had a lovely"]),
            (r"\bhad a good\b", ["had a nice", "had a pleasant", "had an enjoyable"]),
            (
                r"\bwith the family\b",
                ["with my family", "with our family", "with the whole family"],
            ),
            (r"\bwith family\b", ["with the family", "with relatives", "with loved ones"]),
            # Family activities
            (r"\bmade dinner\b", ["cooked dinner", "prepared dinner", "fixed dinner"]),
            (r"\bhad dinner\b", ["ate dinner", "enjoyed dinner", "shared dinner"]),
            # Emotional expressions
            (r"\blove spending time\b", ["enjoy spending time", "cherish spending time"]),
            (r"\breally enjoyed\b", ["truly enjoyed", "greatly enjoyed", "thoroughly enjoyed"]),
        ]

        for pattern, replacements in substitutions:
            if re.search(pattern, text, re.IGNORECASE):
                for replacement in replacements:
                    paraphrase = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                    if paraphrase != text and paraphrase not in paraphrases:
                        paraphrases.append(paraphrase)

        return paraphrases


def back_translate(
    text: str,
    languages: list[str] | None = None,
) -> list[str]:
    """
    Convenience function for back-translation augmentation.

    Args:
        text: Input text to paraphrase
        languages: Intermediate languages for translation

    Returns:
        List of paraphrased texts

    Example:
        >>> paraphrases = back_translate("Had a great day with family")
        >>> print(paraphrases)
        ['Had a wonderful day with family', 'Had a lovely day with the family']
    """
    translator = BackTranslator(languages=languages)
    return translator.translate(text)


# =============================================================================
# Additional Augmentation Utilities
# =============================================================================


def random_mask(
    text: str,
    mask_token: str = "[MASK]",
    mask_prob: float = 0.15,
) -> str:
    """
    Randomly mask words for MLM-style training.

    Args:
        text: Input text
        mask_token: Token to use for masking
        mask_prob: Probability of masking each word

    Returns:
        Text with randomly masked words
    """
    words = text.split()
    masked_words = []

    for word in words:
        if random.random() < mask_prob:
            masked_words.append(mask_token)
        else:
            masked_words.append(word)

    return " ".join(masked_words)


def random_swap(text: str, swap_prob: float = 0.1) -> str:
    """
    Randomly swap adjacent words.

    Args:
        text: Input text
        swap_prob: Probability of swapping each adjacent pair

    Returns:
        Text with randomly swapped words
    """
    words = text.split()

    for i in range(len(words) - 1):
        if random.random() < swap_prob:
            words[i], words[i + 1] = words[i + 1], words[i]

    return " ".join(words)


def random_delete(text: str, delete_prob: float = 0.1) -> str:
    """
    Randomly delete words.

    Args:
        text: Input text
        delete_prob: Probability of deleting each word

    Returns:
        Text with randomly deleted words
    """
    words = text.split()
    kept_words = [w for w in words if random.random() >= delete_prob]

    # Keep at least one word
    if not kept_words:
        kept_words = [random.choice(words)]

    return " ".join(kept_words)


def synonym_replacement(
    text: str,
    synonyms: dict[str, list[str]] | None = None,
    replacement_prob: float = 0.2,
) -> str:
    """
    Replace words with synonyms.

    Args:
        text: Input text
        synonyms: Dictionary mapping words to synonym lists
        replacement_prob: Probability of replacing each word

    Returns:
        Text with synonym replacements
    """
    if synonyms is None:
        # Default family-related synonyms
        synonyms = {
            "happy": ["joyful", "delighted", "pleased", "glad"],
            "sad": ["unhappy", "sorrowful", "melancholy", "down"],
            "love": ["adore", "cherish", "treasure", "care for"],
            "like": ["enjoy", "appreciate", "fancy", "favor"],
            "good": ["great", "wonderful", "excellent", "fine"],
            "bad": ["poor", "terrible", "awful", "unpleasant"],
            "big": ["large", "huge", "enormous", "massive"],
            "small": ["little", "tiny", "miniature", "petite"],
            "nice": ["pleasant", "lovely", "delightful", "agreeable"],
            "talk": ["speak", "chat", "converse", "discuss"],
        }

    words = text.split()
    result_words = []

    for word in words:
        word_lower = word.lower()
        if word_lower in synonyms and random.random() < replacement_prob:
            synonym = random.choice(synonyms[word_lower])
            # Preserve casing
            if word.istitle():
                synonym = synonym.title()
            elif word.isupper():
                synonym = synonym.upper()
            result_words.append(synonym)
        else:
            result_words.append(word)

    return " ".join(result_words)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Config
    "AugmentationConfig",
    # Main classes
    "FamilyAugmenter",
    "BackTranslator",
    # Convenience functions
    "back_translate",
    # Augmentation utilities
    "random_mask",
    "random_swap",
    "random_delete",
    "synonym_replacement",
    # Data
    "KINSHIP_VARIANTS",
    "VARIANT_TO_STANDARD",
    "NICKNAME_PATTERNS",
    "ALL_MOTHER_VARIANTS",
    "ALL_FATHER_VARIANTS",
    "GRANDMOTHER_VARIANTS",
    "GRANDFATHER_VARIANTS",
    "BROTHER_VARIANTS",
    "SISTER_VARIANTS",
    "UNCLE_VARIANTS",
    "AUNT_VARIANTS",
]
