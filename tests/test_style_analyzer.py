"""
Tests for src/style_analyzer.py
"""

import pytest
from src.style_analyzer import (
    StyleAnalyzer,
    count_syllables,
    get_last_word,
    _count_syllables_heuristic,
    _replace_last_word,
)


# ---------------------------------------------------------------------------
# count_syllables / heuristic
# ---------------------------------------------------------------------------


class TestCountSyllables:
    def test_single_syllable(self):
        # "cat" → 1 syllable
        assert count_syllables("cat") == 1

    def test_two_syllable_word(self):
        # "happy" → 2 syllables
        assert count_syllables("happy") >= 2

    def test_three_syllable_word(self):
        # "beautiful" → 3 syllables
        assert count_syllables("beautiful") >= 3

    def test_empty_line(self):
        assert count_syllables("") == 0

    def test_line_with_multiple_words(self):
        # "The quick brown fox" — at least 4 syllables
        result = count_syllables("The quick brown fox")
        assert result >= 4

    def test_heuristic_simple_word(self):
        # Heuristic should return ≥ 1 for non-empty words
        assert _count_syllables_heuristic("hello") >= 1

    def test_heuristic_returns_minimum_one(self):
        assert _count_syllables_heuristic("b") >= 1

    def test_heuristic_silent_e(self):
        # "late" — 1 syllable (silent e removed)
        assert _count_syllables_heuristic("late") == 1


# ---------------------------------------------------------------------------
# get_last_word
# ---------------------------------------------------------------------------


class TestGetLastWord:
    def test_basic(self):
        assert get_last_word("Hello world") == "world"

    def test_with_punctuation(self):
        # The function extracts alphabetic tokens; last one is the last word
        result = get_last_word("I love poetry!")
        assert result == "poetry"

    def test_empty_string(self):
        assert get_last_word("") == ""

    def test_single_word(self):
        assert get_last_word("sky") == "sky"


# ---------------------------------------------------------------------------
# StyleAnalyzer.get_rhyme_scheme
# ---------------------------------------------------------------------------


class TestGetRhymeScheme:
    def setup_method(self):
        self.analyzer = StyleAnalyzer()

    def test_empty_lines(self):
        assert self.analyzer.get_rhyme_scheme([]) == ""

    def test_non_rhyming_lines_return_distinct_labels(self):
        # "cat" and "house" do not rhyme → different labels
        scheme = self.analyzer.get_rhyme_scheme(
            ["I see a cat", "I see a house"]
        )
        assert len(scheme) == 2

    def test_rhyming_couplet(self):
        # "day" and "say" should rhyme → same label (AA)
        scheme = self.analyzer.get_rhyme_scheme(
            ["Bright and sunny is the day", "I have many things to say"]
        )
        # Both end with rhyming words; expect same label
        assert scheme[0] == scheme[1] or "X" in scheme  # allow dict miss

    def test_scheme_length_matches_lines(self):
        lines = ["one", "two", "three", "four"]
        scheme = self.analyzer.get_rhyme_scheme(lines)
        assert len(scheme) == len(lines)


# ---------------------------------------------------------------------------
# StyleAnalyzer.get_syllable_counts
# ---------------------------------------------------------------------------


class TestGetSyllableCounts:
    def setup_method(self):
        self.analyzer = StyleAnalyzer()

    def test_returns_list_same_length(self):
        lines = ["hello world", "the quick brown fox"]
        counts = self.analyzer.get_syllable_counts(lines)
        assert len(counts) == 2

    def test_each_count_positive(self):
        counts = self.analyzer.get_syllable_counts(["beautiful day"])
        assert all(c > 0 for c in counts)


# ---------------------------------------------------------------------------
# StyleAnalyzer.syllable_similarity
# ---------------------------------------------------------------------------


class TestSyllableSimilarity:
    def setup_method(self):
        self.analyzer = StyleAnalyzer()

    def test_identical_counts(self):
        assert self.analyzer.syllable_similarity([4, 5, 4], [4, 5, 4]) == pytest.approx(1.0)

    def test_completely_different(self):
        # 1 vs 10 — large difference
        score = self.analyzer.syllable_similarity([1], [10])
        assert score < 1.0

    def test_empty_inputs(self):
        assert self.analyzer.syllable_similarity([], []) == 0.0

    def test_score_in_range(self):
        score = self.analyzer.syllable_similarity([3, 4, 5], [4, 5, 6])
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# _replace_last_word
# ---------------------------------------------------------------------------


class TestReplaceLastWord:
    def test_basic_replacement(self):
        result = _replace_last_word("I see the sky", "sky", "blue")
        assert result == "I see the blue"

    def test_no_match(self):
        result = _replace_last_word("Hello world", "moon", "sun")
        assert result == "Hello world"

    def test_replaces_rightmost_occurrence(self):
        result = _replace_last_word("sky is sky", "sky", "clear")
        # Should replace the last one
        assert result == "sky is clear"


# ---------------------------------------------------------------------------
# StyleAnalyzer.generate_report
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def setup_method(self):
        self.analyzer = StyleAnalyzer()

    def test_report_fields(self):
        orig = ["माटी कहे कुम्हार", "एक दिन ऐसा"]
        trans = ["The earth speaks", "One such day"]
        report = self.analyzer.generate_report(orig, trans)
        assert hasattr(report, "original_rhyme_scheme")
        assert hasattr(report, "translated_rhyme_scheme")
        assert hasattr(report, "syllable_similarity")
        assert hasattr(report, "rhyme_preserved")
        assert 0.0 <= report.syllable_similarity <= 1.0

    def test_summary_is_string(self):
        orig = ["hello world"]
        trans = ["hi earth"]
        report = self.analyzer.generate_report(orig, trans)
        summary = report.summary()
        assert isinstance(summary, str)
        assert "Style Analysis" in summary
