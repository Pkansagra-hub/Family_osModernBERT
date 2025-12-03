"""
Milestone 2: Data Pipeline Tests
Issue 2.2.1: data/preprocessing.py

Tests for:
- clean_text: Basic cleaning (whitespace, control chars)
- Unicode normalization (NFKC)
- Lowercasing
- PreprocessConfig defaults
- Content removal: URLs, emails, mentions, hashtags
- Emoji handling: keep, remove, replace
- Punctuation handling
- Truncation strategies: head, tail, middle
- Kinship terms: mapping and preservation
- Task-specific preprocessing
"""

import pytest


# =============================================================================
# clean_text Tests
# =============================================================================


class TestCleanTextBasic:
    """Test basic cleaning (whitespace, control chars)."""

    def test_clean_text_normalizes_whitespace(self):
        """clean_text should normalize multiple spaces to single."""
        from modeling_studio.data.preprocessing import clean_text

        result = clean_text("Hello   World")
        assert result == "Hello World"

    def test_clean_text_strips_ends(self):
        """clean_text should strip leading/trailing whitespace."""
        from modeling_studio.data.preprocessing import clean_text

        result = clean_text("  Hello World  ")
        assert result == "Hello World"

    def test_clean_text_normalizes_tabs(self):
        """clean_text should normalize tabs to spaces."""
        from modeling_studio.data.preprocessing import clean_text

        result = clean_text("Hello\t\tWorld")
        assert result == "Hello World"

    def test_clean_text_empty_string(self):
        """clean_text should handle empty string."""
        from modeling_studio.data.preprocessing import clean_text

        result = clean_text("")
        assert result == ""

    def test_clean_text_removes_control_chars(self):
        """clean_text should remove control characters."""
        from modeling_studio.data.preprocessing import clean_text

        # Control character (e.g., null byte)
        result = clean_text("Hello\x00World")
        assert "\x00" not in result


class TestCleanTextUnicodeNormalization:
    """Test NFKC normalization applied."""

    def test_clean_text_nfkc_normalization(self):
        """clean_text should apply NFKC normalization."""
        from modeling_studio.data.preprocessing import clean_text

        # Full-width characters to ASCII
        result = clean_text("Ｈｅｌｌｏ", normalize_unicode=True)
        assert result == "Hello"

    def test_clean_text_normalize_ligatures(self):
        """clean_text should normalize ligatures."""
        from modeling_studio.data.preprocessing import clean_text

        # ﬁ ligature to fi
        result = clean_text("ﬁnd", normalize_unicode=True)
        assert result == "find"

    def test_clean_text_skip_normalization(self):
        """clean_text should skip normalization when flag is False."""
        from modeling_studio.data.preprocessing import clean_text

        result = clean_text("Ｈｅｌｌｏ", normalize_unicode=False)
        assert "Ｈ" in result  # Full-width H still present


class TestCleanTextLowercase:
    """Test lowercase when flag set."""

    def test_clean_text_lowercase_enabled(self):
        """clean_text should lowercase when flag is True."""
        from modeling_studio.data.preprocessing import clean_text

        result = clean_text("Hello WORLD", lowercase=True)
        assert result == "hello world"

    def test_clean_text_lowercase_disabled(self):
        """clean_text should preserve case when flag is False."""
        from modeling_studio.data.preprocessing import clean_text

        result = clean_text("Hello WORLD", lowercase=False)
        assert result == "Hello WORLD"


# =============================================================================
# PreprocessConfig Tests
# =============================================================================


class TestPreprocessConfigDefaults:
    """Test verify default config values."""

    def test_preprocess_config_exists(self):
        """PreprocessConfig should exist."""
        from modeling_studio.data.preprocessing import PreprocessConfig

        config = PreprocessConfig()
        assert config is not None

    def test_preprocess_config_lowercase_default(self):
        """lowercase should default to False."""
        from modeling_studio.data.preprocessing import PreprocessConfig

        config = PreprocessConfig()
        assert config.lowercase is False

    def test_preprocess_config_normalize_unicode_default(self):
        """normalize_unicode should default to True."""
        from modeling_studio.data.preprocessing import PreprocessConfig

        config = PreprocessConfig()
        assert config.normalize_unicode is True

    def test_preprocess_config_clean_whitespace_default(self):
        """clean_whitespace should default to True."""
        from modeling_studio.data.preprocessing import PreprocessConfig

        config = PreprocessConfig()
        assert config.clean_whitespace is True

    def test_preprocess_config_remove_urls_default(self):
        """remove_urls should default to False."""
        from modeling_studio.data.preprocessing import PreprocessConfig

        config = PreprocessConfig()
        assert config.remove_urls is False

    def test_preprocess_config_emoji_handling_default(self):
        """emoji_handling should default to 'keep'."""
        from modeling_studio.data.preprocessing import PreprocessConfig

        config = PreprocessConfig()
        assert config.emoji_handling == "keep"

    def test_preprocess_config_preserve_kinship_default(self):
        """preserve_kinship_terms should default to True."""
        from modeling_studio.data.preprocessing import PreprocessConfig

        config = PreprocessConfig()
        assert config.preserve_kinship_terms is True


# =============================================================================
# Content Removal Tests
# =============================================================================


class TestRemoveUrls:
    """Test URLs removed when flag set."""

    def test_remove_urls_http(self):
        """remove_urls should remove http URLs."""
        from modeling_studio.data.preprocessing import remove_urls

        result = remove_urls("Check http://example.com for info")
        assert "http://example.com" not in result

    def test_remove_urls_https(self):
        """remove_urls should remove https URLs."""
        from modeling_studio.data.preprocessing import remove_urls

        result = remove_urls("Visit https://example.com today")
        assert "https://example.com" not in result

    def test_remove_urls_with_replacement(self):
        """remove_urls should use replacement string."""
        from modeling_studio.data.preprocessing import remove_urls

        result = remove_urls("See https://example.com now", replacement="[URL]")
        assert "[URL]" in result


class TestRemoveEmails:
    """Test emails removed when flag set."""

    def test_remove_emails_basic(self):
        """remove_emails should remove email addresses."""
        from modeling_studio.data.preprocessing import remove_emails

        result = remove_emails("Contact test@example.com for help")
        assert "test@example.com" not in result

    def test_remove_emails_with_subdomain(self):
        """remove_emails should handle subdomain emails."""
        from modeling_studio.data.preprocessing import remove_emails

        result = remove_emails("Email user@mail.example.org")
        assert "@mail.example.org" not in result


class TestRemoveMentions:
    """Test @mentions removed when flag set."""

    def test_remove_mentions_basic(self):
        """remove_mentions should remove @mentions."""
        from modeling_studio.data.preprocessing import remove_mentions

        result = remove_mentions("Hello @user how are you")
        assert "@user" not in result

    def test_remove_mentions_multiple(self):
        """remove_mentions should remove multiple mentions."""
        from modeling_studio.data.preprocessing import remove_mentions

        result = remove_mentions("@alice and @bob are friends")
        assert "@alice" not in result
        assert "@bob" not in result


class TestRemoveHashtags:
    """Test #hashtags removed when flag set."""

    def test_remove_hashtags_basic(self):
        """remove_hashtags should remove hashtags."""
        from modeling_studio.data.preprocessing import remove_hashtags

        result = remove_hashtags("This is #trending now")
        assert "#trending" not in result

    def test_remove_hashtags_multiple(self):
        """remove_hashtags should remove multiple hashtags."""
        from modeling_studio.data.preprocessing import remove_hashtags

        result = remove_hashtags("#hello #world today")
        assert "#hello" not in result
        assert "#world" not in result


# =============================================================================
# Emoji Handling Tests
# =============================================================================


class TestEmojiHandlingKeep:
    """Test emojis preserved."""

    def test_emoji_handling_keep_preserves(self):
        """handle_emojis with 'keep' should preserve emojis."""
        from modeling_studio.data.preprocessing import handle_emojis

        result = handle_emojis("Hello 😀 World", strategy="keep")
        assert "😀" in result


class TestEmojiHandlingRemove:
    """Test emojis removed."""

    def test_emoji_handling_remove_removes(self):
        """handle_emojis with 'remove' should remove emojis."""
        from modeling_studio.data.preprocessing import handle_emojis

        result = handle_emojis("Hello 😀 World", strategy="remove")
        assert "😀" not in result

    def test_emoji_handling_remove_preserves_text(self):
        """handle_emojis with 'remove' should preserve text."""
        from modeling_studio.data.preprocessing import handle_emojis

        result = handle_emojis("Hello 😀 World", strategy="remove")
        assert "Hello" in result
        assert "World" in result


class TestEmojiHandlingReplace:
    """Test emojis replaced with placeholder."""

    def test_emoji_handling_replace_uses_placeholder(self):
        """handle_emojis with 'replace' should use placeholder."""
        from modeling_studio.data.preprocessing import handle_emojis

        result = handle_emojis("Hello 😀 World", strategy="replace", placeholder="[EMO]")
        assert "[EMO]" in result
        assert "😀" not in result


# =============================================================================
# Punctuation Tests
# =============================================================================


class TestCollapsePunctuation:
    """Test '!!!' becomes '!'."""

    def test_collapse_punctuation_exclamation(self):
        """collapse_punctuation should collapse multiple exclamation marks."""
        from modeling_studio.data.preprocessing import collapse_punctuation

        result = collapse_punctuation("Wow!!!")
        assert result == "Wow!"

    def test_collapse_punctuation_question(self):
        """collapse_punctuation should collapse multiple question marks."""
        from modeling_studio.data.preprocessing import collapse_punctuation

        result = collapse_punctuation("Really???")
        assert result == "Really?"

    def test_collapse_punctuation_ellipsis(self):
        """collapse_punctuation should collapse long ellipsis to ...."""
        from modeling_studio.data.preprocessing import collapse_punctuation

        result = collapse_punctuation("Wait.......")
        assert result == "Wait..."


# =============================================================================
# Truncation Tests
# =============================================================================


class TestTruncationHead:
    """Test keeps beginning of text."""

    def test_truncate_head_keeps_beginning(self):
        """truncate_text with 'head' should keep beginning."""
        from modeling_studio.data.preprocessing import truncate_text

        result = truncate_text("Hello World", max_length=5, strategy="head")
        assert result == "Hello"

    def test_truncate_head_no_truncation_needed(self):
        """truncate_text should not truncate if text is short enough."""
        from modeling_studio.data.preprocessing import truncate_text

        result = truncate_text("Hi", max_length=10, strategy="head")
        assert result == "Hi"


class TestTruncationTail:
    """Test keeps end of text."""

    def test_truncate_tail_keeps_end(self):
        """truncate_text with 'tail' should keep end."""
        from modeling_studio.data.preprocessing import truncate_text

        result = truncate_text("Hello World", max_length=5, strategy="tail")
        assert result == "World"


class TestTruncationMiddle:
    """Test keeps beginning and end."""

    def test_truncate_middle_keeps_both_ends(self):
        """truncate_text with 'middle' should keep beginning and end."""
        from modeling_studio.data.preprocessing import truncate_text

        text = "The quick brown fox jumps over the lazy dog"
        result = truncate_text(text, max_length=20, strategy="middle")

        # Should have beginning, ellipsis, and end
        assert result.startswith("The")
        assert "..." in result
        assert result.endswith("dog")


# =============================================================================
# Kinship Terms Tests
# =============================================================================


class TestKinshipTermsMapping:
    """Test Indian kinship terms recognized."""

    def test_indian_kinship_terms_defined(self):
        """INDIAN_KINSHIP_TERMS should be defined."""
        from modeling_studio.data.preprocessing import INDIAN_KINSHIP_TERMS

        assert isinstance(INDIAN_KINSHIP_TERMS, dict)
        assert len(INDIAN_KINSHIP_TERMS) > 0

    def test_indian_kinship_terms_has_nani(self):
        """INDIAN_KINSHIP_TERMS should have 'nani'."""
        from modeling_studio.data.preprocessing import INDIAN_KINSHIP_TERMS

        assert "nani" in INDIAN_KINSHIP_TERMS

    def test_indian_kinship_terms_has_dada(self):
        """INDIAN_KINSHIP_TERMS should have 'dada'."""
        from modeling_studio.data.preprocessing import INDIAN_KINSHIP_TERMS

        assert "dada" in INDIAN_KINSHIP_TERMS


class TestKinshipNormalizeToEnglish:
    """Test normalizes 'nani' to 'grandmother'."""

    def test_normalize_kinship_term_nani(self):
        """normalize_kinship_term should convert 'nani' to 'grandmother'."""
        from modeling_studio.data.preprocessing import normalize_kinship_term

        result = normalize_kinship_term("nani", to_english=True)
        assert result == "grandmother"

    def test_normalize_kinship_term_dadi(self):
        """normalize_kinship_term should convert 'dadi' to 'grandmother'."""
        from modeling_studio.data.preprocessing import normalize_kinship_term

        result = normalize_kinship_term("dadi", to_english=True)
        assert result == "grandmother"

    def test_normalize_kinship_term_unknown(self):
        """normalize_kinship_term should return unknown terms unchanged."""
        from modeling_studio.data.preprocessing import normalize_kinship_term

        result = normalize_kinship_term("unknown_term", to_english=True)
        assert result == "unknown_term"

    def test_normalize_kinship_term_case_insensitive(self):
        """normalize_kinship_term should be case insensitive."""
        from modeling_studio.data.preprocessing import normalize_kinship_term

        result = normalize_kinship_term("Nani", to_english=True)
        assert result == "grandmother"


class TestPreserveKinshipTerms:
    """Test kinship terms not altered when flag set."""

    def test_preserve_family_terms_returns_tuple(self):
        """preserve_family_terms should return (text, mapping)."""
        from modeling_studio.data.preprocessing import preserve_family_terms

        result = preserve_family_terms("My nani is kind")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_preserve_family_terms_creates_placeholders(self):
        """preserve_family_terms should replace terms with placeholders."""
        from modeling_studio.data.preprocessing import preserve_family_terms

        text, mapping = preserve_family_terms("My nani is kind")
        assert "nani" not in text.lower() or "__FAMILY_TERM_" in text
        assert len(mapping) > 0

    def test_restore_family_terms_restores(self):
        """restore_family_terms should restore original terms."""
        from modeling_studio.data.preprocessing import (
            preserve_family_terms,
            restore_family_terms,
        )

        original = "My nani is kind"
        text, mapping = preserve_family_terms(original)
        restored = restore_family_terms(text, mapping)
        assert "nani" in restored


# =============================================================================
# Task-Specific Preprocessing Tests
# =============================================================================


class TestTaskSpecificPreprocessing:
    """Test different preprocessing for NER vs sentiment."""

    def test_ner_preprocessor_preserves_case(self):
        """NER preprocessor should preserve casing."""
        from modeling_studio.data.preprocessing import get_ner_preprocessor

        preprocessor = get_ner_preprocessor()
        result = preprocessor("John Smith works at Google")
        assert "John" in result  # Capital preserved

    def test_sentiment_preprocessor_collapses_punctuation(self):
        """Sentiment preprocessor should collapse punctuation."""
        from modeling_studio.data.preprocessing import get_sentiment_preprocessor

        preprocessor = get_sentiment_preprocessor()
        result = preprocessor("Wow!!! Amazing!!!")
        # Should collapse multiple ! to single !
        assert "!!!" not in result

    def test_safety_preprocessor_keeps_urls(self):
        """Safety preprocessor should keep URLs."""
        from modeling_studio.data.preprocessing import get_safety_preprocessor

        preprocessor = get_safety_preprocessor()
        result = preprocessor("Visit http://harmful.site now")
        # Safety needs to detect harmful links
        assert "http" in result

    def test_embedding_preprocessor_removes_urls(self):
        """Embedding preprocessor should remove URLs."""
        from modeling_studio.data.preprocessing import get_embedding_preprocessor

        preprocessor = get_embedding_preprocessor()
        result = preprocessor("Check http://example.com for info")
        assert "http://example.com" not in result


# =============================================================================
# TextPreprocessor Class Tests
# =============================================================================


class TestTextPreprocessorClass:
    """Test TextPreprocessor class."""

    def test_text_preprocessor_init(self):
        """TextPreprocessor should initialize."""
        from modeling_studio.data.preprocessing import TextPreprocessor

        preprocessor = TextPreprocessor()
        assert preprocessor is not None

    def test_text_preprocessor_callable(self):
        """TextPreprocessor should be callable."""
        from modeling_studio.data.preprocessing import TextPreprocessor

        preprocessor = TextPreprocessor()
        result = preprocessor("Hello World")
        assert isinstance(result, str)

    def test_text_preprocessor_with_config(self):
        """TextPreprocessor should accept config."""
        from modeling_studio.data.preprocessing import PreprocessConfig, TextPreprocessor

        config = PreprocessConfig(lowercase=True)
        preprocessor = TextPreprocessor(config=config)
        result = preprocessor("Hello WORLD")
        assert result == "hello world"

    def test_text_preprocessor_with_kwargs(self):
        """TextPreprocessor should accept kwargs."""
        from modeling_studio.data.preprocessing import TextPreprocessor

        preprocessor = TextPreprocessor(lowercase=True, remove_urls=True)
        result = preprocessor("HELLO http://example.com")
        assert "hello" in result
        assert "http://" not in result

    def test_text_preprocessor_process_batch(self):
        """TextPreprocessor.process_batch should process list of texts."""
        from modeling_studio.data.preprocessing import TextPreprocessor

        preprocessor = TextPreprocessor(lowercase=True)
        results = preprocessor.process_batch(["Hello", "WORLD"])
        assert results == ["hello", "world"]


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Test that all public APIs are exported."""

    def test_all_exports_defined(self):
        """__all__ should be defined with public APIs."""
        from modeling_studio.data import preprocessing

        assert hasattr(preprocessing, "__all__")
        assert "clean_text" in preprocessing.__all__
        assert "PreprocessConfig" in preprocessing.__all__
        assert "TextPreprocessor" in preprocessing.__all__

    def test_kinship_terms_exported(self):
        """Kinship term mappings should be exported."""
        from modeling_studio.data import preprocessing

        assert "INDIAN_KINSHIP_TERMS" in preprocessing.__all__
        assert "ALL_KINSHIP_TERMS" in preprocessing.__all__

    def test_task_preprocessors_exported(self):
        """Task-specific preprocessor factories should be exported."""
        from modeling_studio.data import preprocessing

        assert "get_preprocessor" in preprocessing.__all__
        assert "get_ner_preprocessor" in preprocessing.__all__
        assert "get_safety_preprocessor" in preprocessing.__all__
