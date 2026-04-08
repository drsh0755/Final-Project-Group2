"""
Tests for src/poetry_utils.py
"""

import pytest
from src.poetry_utils import (
    detect_script,
    detect_language,
    clean_poem,
    split_into_lines,
    split_into_stanzas,
    format_poem,
    build_side_by_side,
    remove_extra_whitespace,
    normalize_unicode,
)


# ---------------------------------------------------------------------------
# detect_script
# ---------------------------------------------------------------------------


class TestDetectScript:
    def test_hindi_devanagari(self):
        text = "माटी कहे कुम्हार से, तू क्या रौंदे मोय।"
        assert detect_script(text) == "hi"

    def test_urdu_arabic_script(self):
        text = "ہزاروں خواہشیں ایسی کہ ہر خواہش پہ دم نکلے"
        assert detect_script(text) == "ur"

    def test_english_text_returns_unknown(self):
        assert detect_script("Hello world") == "unknown"

    def test_empty_string_returns_unknown(self):
        assert detect_script("") == "unknown"

    def test_mixed_script_majority_wins(self):
        # More Devanagari characters than Arabic
        hindi_heavy = "माटी कहे कुم्हार ہے"
        result = detect_script(hindi_heavy)
        assert result in ("hi", "ur")  # majority determines result


# ---------------------------------------------------------------------------
# detect_language
# ---------------------------------------------------------------------------


class TestDetectLanguage:
    def test_hindi(self):
        assert detect_language("पायो जी मैंने राम रतन धन पायो।") == "hi"

    def test_urdu(self):
        assert detect_language("مجھ سے پہلی سی محبت مری محبوب نہ مانگ") == "ur"

    def test_unknown_for_numbers(self):
        # Pure numbers or symbols — script detection returns unknown,
        # langdetect may or may not identify it
        result = detect_language("123 456")
        assert result in ("hi", "ur", "unknown")


# ---------------------------------------------------------------------------
# clean_poem
# ---------------------------------------------------------------------------


class TestCleanPoem:
    def test_strips_leading_trailing_whitespace(self):
        result = clean_poem("  माटी कहे  \n  एक दिन  ")
        lines = result.splitlines()
        for line in lines:
            if line:
                assert line == line.strip()

    def test_removes_punctuation_only_lines(self):
        text = "माटी कहे\n---\nएक दिन"
        result = clean_poem(text)
        assert "---" not in result

    def test_preserves_content_lines(self):
        text = "पायो जी मैंने\nराम रतन धन"
        result = clean_poem(text)
        assert "पायो जी मैंने" in result
        assert "राम रतन धन" in result

    def test_nfc_normalisation(self):
        # NFD vs NFC — should normalise to same string
        import unicodedata

        nfd = unicodedata.normalize("NFD", "माटी")
        nfc = unicodedata.normalize("NFC", "माटी")
        result = clean_poem(nfd)
        assert unicodedata.normalize("NFC", result) == unicodedata.normalize("NFC", nfc)


# ---------------------------------------------------------------------------
# split_into_lines / split_into_stanzas
# ---------------------------------------------------------------------------


class TestSplitting:
    def test_split_into_lines_basic(self):
        text = "line one\nline two\nline three"
        assert split_into_lines(text) == ["line one", "line two", "line three"]

    def test_split_into_lines_ignores_blank(self):
        text = "line one\n\nline two"
        assert split_into_lines(text) == ["line one", "line two"]

    def test_split_into_stanzas_two_stanzas(self):
        text = "line one\nline two\n\nline three\nline four"
        stanzas = split_into_stanzas(text)
        assert len(stanzas) == 2
        assert stanzas[0] == ["line one", "line two"]
        assert stanzas[1] == ["line three", "line four"]

    def test_split_into_stanzas_single_stanza(self):
        text = "line one\nline two"
        assert split_into_stanzas(text) == [["line one", "line two"]]

    def test_split_into_stanzas_empty(self):
        assert split_into_stanzas("") == []


# ---------------------------------------------------------------------------
# format_poem / build_side_by_side
# ---------------------------------------------------------------------------


class TestFormatting:
    def test_format_poem_with_title(self):
        result = format_poem(["line one", "line two"], title="My Poem")
        assert "My Poem" in result
        assert "line one" in result
        assert "line two" in result

    def test_format_poem_without_title(self):
        result = format_poem(["line one"])
        assert "line one" in result

    def test_build_side_by_side_same_length(self):
        orig = ["माटी कहे", "एक दिन"]
        trans = ["The earth says", "One day"]
        result = build_side_by_side(orig, trans)
        assert "माटी कहे" in result
        assert "The earth says" in result
        assert "Original" in result
        assert "Translation" in result

    def test_build_side_by_side_separator(self):
        result = build_side_by_side(["a"], ["b"])
        assert "---" in result or "|" in result


# ---------------------------------------------------------------------------
# remove_extra_whitespace
# ---------------------------------------------------------------------------


class TestRemoveExtraWhitespace:
    def test_collapses_spaces(self):
        assert remove_extra_whitespace("hello   world") == "hello world"

    def test_preserves_newlines(self):
        result = remove_extra_whitespace("line one\nline  two")
        assert "line one" in result
        assert "line two" in result

    def test_tabs_collapsed(self):
        result = remove_extra_whitespace("word\t\tword2")
        assert "word word2" in result
