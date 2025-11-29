"""
Text Preprocessing Utilities

This module provides text preprocessing functions for cleaning and
preparing text data before tokenization.

Preprocessing Steps:
    - Text cleaning (whitespace, special chars)
    - Unicode normalization
    - Lowercasing (optional, task-dependent)
    - Truncation to max length
    - Handling of special formats (URLs, emails, mentions)

Task-Specific Preprocessing:
    - NER: Preserve casing, handle sentence boundaries
    - Sentiment: Optionally normalize punctuation/emojis
    - Safety: Preserve offensive content for detection
    - Embedding: Clean but preserve semantic content

FamilyOS-Specific:
    - Handle Indian English expressions
    - Preserve family nicknames and terms
    - Process conversation/diary formats
    - Normalize kinship terms across cultures

Usage:
    from modeling_studio.data.preprocessing import TextPreprocessor

    preprocessor = TextPreprocessor(
        lowercase=False,
        normalize_unicode=True,
        clean_whitespace=True,
    )
    clean = preprocessor("  Hello   World!!! ")
    # -> "Hello World!!!"
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)


# =============================================================================
# Kinship Term Mappings (Multi-Cultural)
# =============================================================================

# Indian kinship terms mapped to standard forms
INDIAN_KINSHIP_TERMS = {
    # Maternal side
    "nana": "grandfather",
    "nani": "grandmother",
    "mama": "uncle",
    "mami": "aunt",
    "mausa": "uncle",
    "mausi": "aunt",
    # Paternal side
    "dada": "grandfather",
    "dadi": "grandmother",
    "chacha": "uncle",
    "chachi": "aunt",
    "tau": "uncle",
    "tai": "aunt",
    "bua": "aunt",
    "fufa": "uncle",
    # Siblings
    "bhai": "brother",
    "bhaiya": "brother",
    "didi": "sister",
    "di": "sister",
    # Others
    "beta": "son",
    "beti": "daughter",
    "bahu": "daughter-in-law",
    "sasur": "father-in-law",
    "saas": "mother-in-law",
    "jiju": "brother-in-law",
    "sala": "brother-in-law",
    "devar": "brother-in-law",
    "nanad": "sister-in-law",
    "bhabhi": "sister-in-law",
}

# Filipino kinship terms
FILIPINO_KINSHIP_TERMS = {
    "lolo": "grandfather",
    "lola": "grandmother",
    "tatay": "father",
    "nanay": "mother",
    "kuya": "older_brother",
    "ate": "older_sister",
    "tito": "uncle",
    "tita": "aunt",
    "pinsan": "cousin",
}

# Spanish/Latino kinship terms (common in US)
SPANISH_KINSHIP_TERMS = {
    "abuela": "grandmother",
    "abuelo": "grandfather",
    "tio": "uncle",
    "tia": "aunt",
    "primo": "cousin",
    "prima": "cousin",
    "mami": "mom",
    "papi": "dad",
    "hermano": "brother",
    "hermana": "sister",
}

# Combined kinship terms
ALL_KINSHIP_TERMS = {
    **INDIAN_KINSHIP_TERMS,
    **FILIPINO_KINSHIP_TERMS,
    **SPANISH_KINSHIP_TERMS,
}


# =============================================================================
# TextPreprocessor Configuration
# =============================================================================


@dataclass
class PreprocessConfig:
    """Configuration for text preprocessing pipeline."""

    # Basic cleaning
    lowercase: bool = False
    normalize_unicode: bool = True
    clean_whitespace: bool = True
    strip_accents: bool = False

    # Content handling
    remove_urls: bool = False
    remove_emails: bool = False
    remove_mentions: bool = False
    remove_hashtags: bool = False

    # Emoji handling: 'keep', 'remove', 'replace'
    emoji_handling: Literal["keep", "remove", "replace"] = "keep"
    emoji_placeholder: str = "[EMOJI]"

    # Punctuation
    normalize_punctuation: bool = False
    collapse_punctuation: bool = True  # "!!!" -> "!"

    # Length
    max_length: int | None = None
    truncation_strategy: Literal["head", "tail", "middle"] = "tail"

    # FamilyOS-specific
    preserve_kinship_terms: bool = True
    normalize_kinship_to_english: bool = False

    # Task-specific presets
    task: str | None = None


# =============================================================================
# Core Text Cleaning Functions
# =============================================================================


def clean_text(
    text: str,
    lowercase: bool = False,
    normalize_unicode: bool = True,
    clean_whitespace: bool = True,
) -> str:
    """
    Basic text cleaning: normalize unicode, remove control chars, clean whitespace.

    Args:
        text: Input text
        lowercase: Whether to lowercase
        normalize_unicode: Whether to normalize unicode (NFKC)
        clean_whitespace: Whether to collapse/strip whitespace

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Unicode normalization (NFKC: compatibility decomposition + composition)
    if normalize_unicode:
        text = unicodedata.normalize("NFKC", text)

    # Remove control characters (except common whitespace)
    text = "".join(
        char for char in text if not unicodedata.category(char).startswith("C") or char in "\n\t "
    )

    # Clean whitespace
    if clean_whitespace:
        # Replace multiple spaces/tabs with single space
        text = re.sub(r"[ \t]+", " ", text)
        # Collapse multiple newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Strip leading/trailing
        text = text.strip()

    # Lowercase
    if lowercase:
        text = text.lower()

    return text


def normalize_whitespace(text: str) -> str:
    """Normalize all whitespace to single spaces, strip ends."""
    return " ".join(text.split())


def normalize_unicode(text: str, form: str = "NFKC") -> str:
    """Normalize unicode using specified form (NFC, NFKC, NFD, NFKD)."""
    return unicodedata.normalize(form, text)


def remove_control_chars(text: str) -> str:
    """Remove unicode control characters."""
    return "".join(
        char for char in text if not unicodedata.category(char).startswith("C") or char in "\n\t "
    )


def strip_accents(text: str) -> str:
    """Remove diacritics/accents from text."""
    # NFD decomposition separates base char from combining chars
    normalized = unicodedata.normalize("NFD", text)
    return "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"  # Mn = Mark, Nonspacing
    )


# =============================================================================
# Content Removal Functions
# =============================================================================

# URL pattern
URL_PATTERN = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)

# Email pattern
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

# @mentions pattern
MENTION_PATTERN = re.compile(r"@[\w_]+")

# #hashtag pattern
HASHTAG_PATTERN = re.compile(r"#[\w_]+")

# Emoji pattern (simplified - covers most common)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # Emoticons
    "\U0001f300-\U0001f5ff"  # Symbols & pictographs
    "\U0001f680-\U0001f6ff"  # Transport & map
    "\U0001f700-\U0001f77f"  # Alchemical
    "\U0001f780-\U0001f7ff"  # Geometric extended
    "\U0001f800-\U0001f8ff"  # Supplemental arrows
    "\U0001f900-\U0001f9ff"  # Supplemental symbols
    "\U0001fa00-\U0001fa6f"  # Chess
    "\U0001fa70-\U0001faff"  # Symbols extended
    "\U00002702-\U000027b0"  # Dingbats
    "\U0001f1e0-\U0001f1ff"  # Flags
    "]+",
    flags=re.UNICODE,
)


def remove_urls(text: str, replacement: str = "") -> str:
    """Remove URLs from text."""
    return URL_PATTERN.sub(replacement, text)


def remove_emails(text: str, replacement: str = "") -> str:
    """Remove email addresses from text."""
    return EMAIL_PATTERN.sub(replacement, text)


def remove_mentions(text: str, replacement: str = "") -> str:
    """Remove @mentions from text."""
    return MENTION_PATTERN.sub(replacement, text)


def remove_hashtags(text: str, replacement: str = "") -> str:
    """Remove #hashtags from text."""
    return HASHTAG_PATTERN.sub(replacement, text)


def handle_emojis(
    text: str,
    strategy: Literal["keep", "remove", "replace"] = "keep",
    placeholder: str = "[EMOJI]",
) -> str:
    """Handle emojis based on strategy."""
    if strategy == "keep":
        return text
    elif strategy == "remove":
        return EMOJI_PATTERN.sub("", text)
    else:  # replace
        return EMOJI_PATTERN.sub(placeholder, text)


# =============================================================================
# Punctuation Handling
# =============================================================================


def normalize_punctuation(text: str) -> str:
    """Normalize various punctuation marks to standard ASCII."""
    # Quotes
    text = re.sub(r"[''‚]", "'", text)
    text = re.sub(r'[""„]', '"', text)

    # Dashes
    text = re.sub(r"[–—]", "-", text)

    # Ellipsis
    text = re.sub(r"…", "...", text)

    # Spaces (various unicode spaces to regular)
    text = re.sub(r"[\u00A0\u2000-\u200B\u202F\u205F\u3000]", " ", text)

    return text


def collapse_punctuation(text: str) -> str:
    """Collapse repeated punctuation marks."""
    # Multiple exclamation/question marks
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"\.{4,}", "...", text)

    return text


# =============================================================================
# Truncation Functions
# =============================================================================


def truncate_text(
    text: str,
    max_length: int,
    strategy: Literal["head", "tail", "middle"] = "tail",
) -> str:
    """
    Truncate text to max_length characters.

    Args:
        text: Input text
        max_length: Maximum character length
        strategy:
            - 'head': Keep beginning, truncate end
            - 'tail': Truncate beginning, keep end
            - 'middle': Keep beginning and end, remove middle

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    if strategy == "head":
        return text[:max_length]
    elif strategy == "tail":
        return text[-max_length:]
    else:  # middle
        half = max_length // 2
        return text[:half] + "..." + text[-(max_length - half - 3) :]


# =============================================================================
# NER-Specific Preprocessing
# =============================================================================


def preprocess_for_ner(
    text: str,
    preserve_casing: bool = True,
    handle_sentence_boundaries: bool = True,
) -> str:
    """
    Preprocess text for NER task.

    - Preserves casing (important for proper nouns)
    - Preserves sentence boundaries
    - Minimal cleaning to preserve entity spans

    Args:
        text: Input text
        preserve_casing: Keep original casing
        handle_sentence_boundaries: Ensure proper sentence boundaries

    Returns:
        Preprocessed text for NER
    """
    # Minimal cleaning - preserve spans
    text = normalize_unicode(text)
    text = remove_control_chars(text)

    # Don't collapse whitespace aggressively - just normalize
    text = re.sub(r"[ \t]+", " ", text)

    # Ensure sentence boundaries are clean
    if handle_sentence_boundaries:
        # Add space after sentence-ending punctuation if missing
        text = re.sub(r"([.!?])([A-Z])", r"\1 \2", text)

    text = text.strip()

    if not preserve_casing:
        text = text.lower()

    return text


def align_ner_labels(
    original_text: str,
    cleaned_text: str,
    original_labels: list[str],
) -> list[str]:
    """
    Align NER labels after text cleaning.

    Maps labels from original text positions to cleaned text positions.

    Args:
        original_text: Original text
        cleaned_text: Cleaned text
        original_labels: BIO labels for original text tokens

    Returns:
        Labels aligned to cleaned text tokens
    """
    # Simple implementation: assumes tokenization by whitespace
    # For production, use character-level alignment

    original_tokens = original_text.split()
    cleaned_tokens = cleaned_text.split()

    if len(original_tokens) != len(original_labels):
        logger.warning(
            f"Token/label mismatch: {len(original_tokens)} tokens, "
            f"{len(original_labels)} labels"
        )
        return original_labels

    # Create mapping from original to cleaned
    aligned_labels = []
    orig_idx = 0

    for clean_token in cleaned_tokens:
        # Find matching token in original
        while orig_idx < len(original_tokens):
            orig_token = original_tokens[orig_idx]
            # Check if cleaned token matches (after similar cleaning)
            if clean_token.lower() == orig_token.lower() or clean_token in orig_token:
                aligned_labels.append(original_labels[orig_idx])
                orig_idx += 1
                break
            orig_idx += 1
        else:
            # No match found, use O
            aligned_labels.append("O")

    return aligned_labels


# =============================================================================
# Safety-Specific Preprocessing
# =============================================================================


def preprocess_for_safety(text: str) -> str:
    """
    Preprocess text for safety classification.

    Minimal cleaning - we need to detect offensive/harmful content,
    so we shouldn't remove or sanitize it.

    Args:
        text: Input text

    Returns:
        Minimally preprocessed text
    """
    # Very minimal cleaning
    text = normalize_unicode(text)
    text = normalize_whitespace(text)

    # DON'T remove:
    # - Offensive words (we need to detect them)
    # - Emojis (can indicate tone/intent)
    # - Repeated punctuation (can indicate emphasis/distress)
    # - URLs (may be harmful links)

    return text


# =============================================================================
# Embedding-Specific Preprocessing
# =============================================================================


def preprocess_for_embedding(
    text: str,
    remove_special: bool = True,
) -> str:
    """
    Preprocess text for embedding generation.

    Clean but preserve semantic content.

    Args:
        text: Input text
        remove_special: Whether to remove URLs, emails, mentions

    Returns:
        Preprocessed text for embedding
    """
    text = clean_text(
        text,
        lowercase=False,  # Preserve casing for embeddings
        normalize_unicode=True,
        clean_whitespace=True,
    )

    if remove_special:
        text = remove_urls(text, replacement=" ")
        text = remove_emails(text, replacement=" ")
        text = remove_mentions(text, replacement=" ")

    # Normalize punctuation
    text = normalize_punctuation(text)

    # Final whitespace cleanup
    text = normalize_whitespace(text)

    return text


# =============================================================================
# FamilyOS-Specific Preprocessing
# =============================================================================


def normalize_kinship_term(
    term: str,
    to_english: bool = True,
    kinship_map: dict[str, str] | None = None,
) -> str:
    """
    Normalize a kinship term to standard form.

    Args:
        term: Kinship term (e.g., "nana", "didi", "lolo")
        to_english: Whether to convert to English equivalent
        kinship_map: Custom mapping (defaults to ALL_KINSHIP_TERMS)

    Returns:
        Normalized term
    """
    if kinship_map is None:
        kinship_map = ALL_KINSHIP_TERMS

    term_lower = term.lower()

    if term_lower in kinship_map:
        if to_english:
            return kinship_map[term_lower]
        else:
            return term_lower  # Standardize case only

    return term  # Unknown term, return as-is


def preserve_family_terms(
    text: str,
    protected_terms: set[str] | None = None,
) -> tuple[str, dict[str, str]]:
    """
    Protect family terms from modification during preprocessing.

    Replaces family terms with placeholders, returns mapping to restore.

    Args:
        text: Input text
        protected_terms: Set of terms to protect (defaults to kinship terms)

    Returns:
        Tuple of (text with placeholders, mapping to restore)
    """
    if protected_terms is None:
        protected_terms = set(ALL_KINSHIP_TERMS.keys())

    mapping: dict[str, str] = {}
    placeholder_idx = 0

    # Find and replace protected terms
    for term in protected_terms:
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        matches = pattern.findall(text)

        for match in matches:
            placeholder = f"__FAMILY_TERM_{placeholder_idx}__"
            mapping[placeholder] = match
            text = pattern.sub(placeholder, text, count=1)
            placeholder_idx += 1

    return text, mapping


def restore_family_terms(text: str, mapping: dict[str, str]) -> str:
    """Restore protected family terms from placeholders."""
    for placeholder, original in mapping.items():
        text = text.replace(placeholder, original)
    return text


def handle_conversation_format(text: str) -> list[dict[str, str]]:
    """
    Parse conversation format text into structured turns.

    Handles formats like:
        - "Speaker: message"
        - "[Speaker] message"
        - "Speaker says: message"

    Args:
        text: Conversation text

    Returns:
        List of {"speaker": str, "message": str} dicts
    """
    turns = []

    # Pattern: "Speaker: message" or "[Speaker] message"
    patterns = [
        re.compile(r"^([A-Za-z_]+):\s*(.+)$", re.MULTILINE),
        re.compile(r"^\[([A-Za-z_]+)\]\s*(.+)$", re.MULTILINE),
        re.compile(r"^([A-Za-z_]+)\s+says?:\s*(.+)$", re.MULTILINE),
    ]

    for pattern in patterns:
        matches = pattern.findall(text)
        if matches:
            turns = [
                {"speaker": speaker.strip(), "message": message.strip()}
                for speaker, message in matches
            ]
            break

    # If no pattern matched, treat whole text as single turn
    if not turns:
        turns = [{"speaker": "unknown", "message": text.strip()}]

    return turns


def handle_diary_format(text: str) -> dict[str, Any]:
    """
    Parse diary entry format.

    Handles formats like:
        - Date header followed by entry
        - Mood indicators
        - Tags/categories

    Args:
        text: Diary entry text

    Returns:
        Structured diary entry dict
    """
    entry: dict[str, Any] = {
        "date": None,
        "mood": None,
        "tags": [],
        "content": text,
    }

    # Extract date (common formats)
    date_patterns = [
        re.compile(r"^(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"),
        re.compile(
            r"^((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})",
            re.IGNORECASE,
        ),
        re.compile(r"^((?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*,?\s+\w+\s+\d{1,2})", re.IGNORECASE),
    ]

    for pattern in date_patterns:
        match = pattern.search(text)
        if match:
            entry["date"] = match.group(1)
            text = text[match.end() :].strip()
            break

    # Extract mood indicators
    mood_pattern = re.compile(r"\b(mood|feeling):\s*(\w+)", re.IGNORECASE)
    mood_match = mood_pattern.search(text)
    if mood_match:
        entry["mood"] = mood_match.group(2).lower()

    # Extract hashtags as tags
    entry["tags"] = re.findall(r"#(\w+)", text)

    # Clean remaining content
    entry["content"] = text.strip()

    return entry


# =============================================================================
# Main TextPreprocessor Class
# =============================================================================


class TextPreprocessor:
    """
    Configurable text preprocessing pipeline.

    Combines multiple preprocessing steps with flexible configuration.
    Can be configured for different tasks or custom pipelines.

    Args:
        config: PreprocessConfig or individual settings
        **kwargs: Override config settings

    Example:
        >>> preprocessor = TextPreprocessor(
        ...     lowercase=False,
        ...     normalize_unicode=True,
        ...     clean_whitespace=True,
        ... )
        >>> clean = preprocessor("  Hello   World!!! ")
        >>> assert clean == "Hello World!!!"
    """

    def __init__(
        self,
        config: PreprocessConfig | None = None,
        **kwargs: Any,
    ):
        if config is not None:
            self.config = config
        else:
            self.config = PreprocessConfig(**kwargs)

        # Apply task-specific presets
        if self.config.task:
            self._apply_task_preset(self.config.task)

        # Build pipeline
        self._pipeline = self._build_pipeline()

    def _apply_task_preset(self, task: str) -> None:
        """Apply task-specific configuration presets."""
        presets = {
            "ner": {
                "lowercase": False,
                "normalize_unicode": True,
                "clean_whitespace": False,  # Preserve spans
                "remove_urls": False,
                "collapse_punctuation": False,
            },
            "safety": {
                "lowercase": False,
                "normalize_unicode": True,
                "clean_whitespace": True,
                "remove_urls": False,  # Keep for harmful link detection
                "emoji_handling": "keep",  # Emojis indicate tone
                "collapse_punctuation": False,  # Emphasis matters
            },
            "sentiment": {
                "lowercase": False,
                "normalize_unicode": True,
                "clean_whitespace": True,
                "emoji_handling": "keep",
                "collapse_punctuation": True,
            },
            "embedding": {
                "lowercase": False,
                "normalize_unicode": True,
                "clean_whitespace": True,
                "remove_urls": True,
                "remove_mentions": True,
                "emoji_handling": "remove",
                "collapse_punctuation": True,
            },
        }

        if task.lower() in presets:
            for key, value in presets[task.lower()].items():
                setattr(self.config, key, value)

    def _build_pipeline(self) -> list[Callable[[str], str]]:
        """Build preprocessing pipeline based on config."""
        pipeline: list[Callable[[str], str]] = []

        # Unicode normalization first
        if self.config.normalize_unicode:
            pipeline.append(lambda t: normalize_unicode(t, "NFKC"))

        # Strip accents (if enabled)
        if self.config.strip_accents:
            pipeline.append(strip_accents)

        # Content removal
        if self.config.remove_urls:
            pipeline.append(lambda t: remove_urls(t, " "))
        if self.config.remove_emails:
            pipeline.append(lambda t: remove_emails(t, " "))
        if self.config.remove_mentions:
            pipeline.append(lambda t: remove_mentions(t, " "))
        if self.config.remove_hashtags:
            pipeline.append(lambda t: remove_hashtags(t, " "))

        # Emoji handling
        if self.config.emoji_handling != "keep":
            pipeline.append(
                lambda t: handle_emojis(
                    t,
                    strategy=self.config.emoji_handling,
                    placeholder=self.config.emoji_placeholder,
                )
            )

        # Punctuation handling
        if self.config.normalize_punctuation:
            pipeline.append(normalize_punctuation)
        if self.config.collapse_punctuation:
            pipeline.append(collapse_punctuation)

        # Whitespace (should come late)
        if self.config.clean_whitespace:
            pipeline.append(normalize_whitespace)

        # Lowercasing (should be last before truncation)
        if self.config.lowercase:
            pipeline.append(str.lower)

        # Truncation (always last)
        if self.config.max_length:
            pipeline.append(
                lambda t: truncate_text(
                    t,
                    max_length=self.config.max_length,  # type: ignore
                    strategy=self.config.truncation_strategy,
                )
            )

        return pipeline

    def __call__(self, text: str) -> str:
        """
        Apply preprocessing pipeline to text.

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        if not text:
            return ""

        result = text

        # Protect kinship terms if configured
        family_mapping: dict[str, str] | None = None
        if self.config.preserve_kinship_terms and not self.config.normalize_kinship_to_english:
            result, family_mapping = preserve_family_terms(result)

        # Apply pipeline
        for transform in self._pipeline:
            result = transform(result)

        # Restore protected terms
        if family_mapping:
            result = restore_family_terms(result, family_mapping)

        return result

    def process_batch(self, texts: list[str]) -> list[str]:
        """Process a batch of texts."""
        return [self(text) for text in texts]


# =============================================================================
# Factory Functions
# =============================================================================


def get_preprocessor(
    task: str | None = None,
    **kwargs: Any,
) -> TextPreprocessor:
    """
    Factory function to create task-specific preprocessor.

    Args:
        task: Task name ('ner', 'safety', 'sentiment', 'embedding', None)
        **kwargs: Override settings

    Returns:
        Configured TextPreprocessor
    """
    return TextPreprocessor(task=task, **kwargs)


def get_ner_preprocessor(**kwargs: Any) -> TextPreprocessor:
    """Get preprocessor configured for NER task."""
    return get_preprocessor(task="ner", **kwargs)


def get_safety_preprocessor(**kwargs: Any) -> TextPreprocessor:
    """Get preprocessor configured for safety task."""
    return get_preprocessor(task="safety", **kwargs)


def get_sentiment_preprocessor(**kwargs: Any) -> TextPreprocessor:
    """Get preprocessor configured for sentiment task."""
    return get_preprocessor(task="sentiment", **kwargs)


def get_embedding_preprocessor(**kwargs: Any) -> TextPreprocessor:
    """Get preprocessor configured for embedding task."""
    return get_preprocessor(task="embedding", **kwargs)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Config
    "PreprocessConfig",
    # Main class
    "TextPreprocessor",
    # Core functions
    "clean_text",
    "normalize_whitespace",
    "normalize_unicode",
    "remove_control_chars",
    "strip_accents",
    # Content removal
    "remove_urls",
    "remove_emails",
    "remove_mentions",
    "remove_hashtags",
    "handle_emojis",
    # Punctuation
    "normalize_punctuation",
    "collapse_punctuation",
    # Truncation
    "truncate_text",
    # Task-specific
    "preprocess_for_ner",
    "preprocess_for_safety",
    "preprocess_for_embedding",
    "align_ner_labels",
    # FamilyOS-specific
    "normalize_kinship_term",
    "preserve_family_terms",
    "restore_family_terms",
    "handle_conversation_format",
    "handle_diary_format",
    # Kinship data
    "INDIAN_KINSHIP_TERMS",
    "FILIPINO_KINSHIP_TERMS",
    "SPANISH_KINSHIP_TERMS",
    "ALL_KINSHIP_TERMS",
    # Factory
    "get_preprocessor",
    "get_ner_preprocessor",
    "get_safety_preprocessor",
    "get_sentiment_preprocessor",
    "get_embedding_preprocessor",
]
