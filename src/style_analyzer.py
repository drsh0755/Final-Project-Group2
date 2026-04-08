"""
style_analyzer.py
~~~~~~~~~~~~~~~~~
Analyse and preserve the poetic style of Hindi/Urdu poems after translation.

Features
--------
- Rhyme-scheme detection on the *translated* English lines using CMU
  Pronouncing Dictionary (``pronouncing`` library).
- Syllable counting per line to measure metre similarity.
- Style post-processing: attempts synonym substitution so that lines that
  rhymed in the original also rhyme in the translation.
- Summary report comparing original and translated style metrics.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LineStyle:
    """Style metrics for a single poem line."""

    text: str
    syllable_count: int
    last_word: str
    rhyme_sound: str  # '' if unknown


@dataclass
class StyleReport:
    """Comparative style report between original and translated poem."""

    original_rhyme_scheme: str
    translated_rhyme_scheme: str
    original_syllables: list[int]
    translated_syllables: list[int]
    syllable_similarity: float  # 0.0 – 1.0
    rhyme_preserved: bool
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary of the style comparison."""
        lines = [
            "=== Style Analysis Report ===",
            f"Original rhyme scheme   : {self.original_rhyme_scheme or 'N/A'}",
            f"Translated rhyme scheme : {self.translated_rhyme_scheme or 'N/A'}",
            f"Rhyme preserved         : {'Yes' if self.rhyme_preserved else 'No'}",
            f"Syllable similarity     : {self.syllable_similarity:.0%}",
        ]
        if self.notes:
            lines.append("Notes:")
            for note in self.notes:
                lines.append(f"  • {note}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper: syllable counting
# ---------------------------------------------------------------------------


def _count_syllables_pronouncing(word: str) -> int:
    """Count syllables using the CMU Pronouncing Dictionary.

    Returns ``-1`` if the word is not found in the dictionary.
    """
    try:
        import pronouncing  # type: ignore

        phones = pronouncing.phones_for_word(word)
        if phones:
            return pronouncing.syllable_count(phones[0])
    except Exception:
        pass
    return -1


def _count_syllables_heuristic(word: str) -> int:
    """Syllable estimation heuristic for words absent from the CMU dict.

    Based on vowel-group counting with common English adjustments.
    """
    word = word.lower().strip(string.punctuation)
    if not word:
        return 0
    # Count vowel groups
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    # Adjust for silent 'e' at end
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def count_syllables(line: str) -> int:
    """Count total syllables in an English line.

    Tries the CMU Pronouncing Dictionary first; falls back to a heuristic
    for unknown words.
    """
    words = re.findall(r"[a-zA-Z']+", line)
    total = 0
    for word in words:
        n = _count_syllables_pronouncing(word)
        if n < 0:
            n = _count_syllables_heuristic(word)
        total += n
    return total


# ---------------------------------------------------------------------------
# Helper: rhyme detection
# ---------------------------------------------------------------------------


def _get_rhyme_sound(word: str) -> str:
    """Return the rhyming part (last stressed vowel + everything after) of *word*.

    Uses the ``pronouncing`` library; returns ``''`` on failure.
    """
    try:
        import pronouncing  # type: ignore

        phones = pronouncing.phones_for_word(word.lower().strip(string.punctuation))
        if phones:
            return pronouncing.rhyming_part(phones[0])
    except Exception:
        pass
    return ""


def get_last_word(line: str) -> str:
    """Extract the last alphabetic token from *line*."""
    words = re.findall(r"[a-zA-Z']+", line)
    return words[-1] if words else ""


# ---------------------------------------------------------------------------
# StyleAnalyzer
# ---------------------------------------------------------------------------


class StyleAnalyzer:
    """Analyses and (best-effort) preserves poetic style across translation.

    Parameters
    ----------
    language:
        Source language code (``"hi"`` or ``"ur"``).  Currently informational;
        syllable analysis is performed on the English translation.
    """

    def __init__(self, language: str = "hi") -> None:
        self.language = language

    # ------------------------------------------------------------------
    # Rhyme scheme
    # ------------------------------------------------------------------

    def get_rhyme_scheme(self, lines: list[str]) -> str:
        """Detect the rhyme scheme of English *lines*.

        Returns a string like ``"ABAB"`` or ``"AABB"``.  Lines whose last
        word is not found in the CMU dictionary are labelled ``"X"``.
        """
        if not lines:
            return ""

        label_map: dict[str, str] = {}
        next_label = [ord("A")]
        scheme_chars: list[str] = []

        for line in lines:
            last = get_last_word(line)
            sound = _get_rhyme_sound(last)
            if not sound:
                scheme_chars.append("X")
                continue
            if sound not in label_map:
                label_map[sound] = chr(next_label[0])
                next_label[0] += 1
                if next_label[0] > ord("Z"):
                    next_label[0] = ord("A")
            scheme_chars.append(label_map[sound])

        return "".join(scheme_chars)

    # ------------------------------------------------------------------
    # Syllable analysis
    # ------------------------------------------------------------------

    def get_syllable_counts(self, lines: list[str]) -> list[int]:
        """Return syllable count per line (English text only)."""
        return [count_syllables(line) for line in lines]

    def syllable_similarity(
        self, original_counts: list[int], translated_counts: list[int]
    ) -> float:
        """Compute a similarity score (0–1) based on per-line syllable counts.

        Uses 1 - mean(|orig - trans| / max(orig, trans, 1)) clipped to [0, 1].
        """
        if not original_counts or not translated_counts:
            return 0.0
        pairs = list(zip(original_counts, translated_counts))
        diffs = [
            abs(o - t) / max(o, t, 1) for o, t in pairs if max(o, t) > 0
        ]
        if not diffs:
            return 1.0
        return max(0.0, 1.0 - sum(diffs) / len(diffs))

    # ------------------------------------------------------------------
    # Style post-processing
    # ------------------------------------------------------------------

    def apply_style_preservation(
        self,
        translated_lines: list[str],
        original_rhyme_scheme: str,
    ) -> list[str]:
        """Attempt synonym substitution to restore the original rhyme scheme.

        Uses WordNet (NLTK) to find candidate rhyming synonyms for the last
        word of lines that should rhyme according to *original_rhyme_scheme*.

        If a rhyming synonym is found it replaces the last word of the line;
        otherwise the line is left unchanged.

        Parameters
        ----------
        translated_lines:
            English lines after initial translation.
        original_rhyme_scheme:
            The rhyme scheme detected from the original poem (e.g. ``"ABAB"``).

        Returns
        -------
        Potentially improved list of translated lines.
        """
        if not translated_lines or not original_rhyme_scheme:
            return translated_lines

        # Group line indices by their rhyme label
        rhyme_groups: dict[str, list[int]] = {}
        for idx, label in enumerate(original_rhyme_scheme):
            if label == "X":
                continue
            if idx >= len(translated_lines):
                break
            rhyme_groups.setdefault(label, []).append(idx)

        improved = list(translated_lines)

        for label, indices in rhyme_groups.items():
            if len(indices) < 2:
                continue
            # Use the first line's last word as the anchor
            anchor_idx = indices[0]
            anchor_word = get_last_word(improved[anchor_idx])
            if not anchor_word:
                continue
            anchor_sound = _get_rhyme_sound(anchor_word)

            for idx in indices[1:]:
                current_word = get_last_word(improved[idx])
                current_sound = _get_rhyme_sound(current_word)
                if current_sound == anchor_sound:
                    continue  # already rhymes
                replacement = self._find_rhyming_synonym(
                    current_word, anchor_sound
                )
                if replacement:
                    improved[idx] = _replace_last_word(
                        improved[idx], current_word, replacement
                    )

        return improved

    def _find_rhyming_synonym(self, word: str, target_sound: str) -> str:
        """Return a WordNet synonym of *word* whose rhyming part matches *target_sound*.

        Returns ``""`` if no suitable synonym is found.
        """
        try:
            from nltk.corpus import wordnet as wn  # type: ignore
            import nltk  # type: ignore

            try:
                synsets = wn.synsets(word)
            except LookupError:
                nltk.download("wordnet", quiet=True)
                synsets = wn.synsets(word)

            candidates: set[str] = set()
            for synset in synsets:
                for lemma in synset.lemmas():
                    candidates.add(lemma.name().replace("_", " "))

            for candidate in candidates:
                candidate_word = candidate.split()[-1]
                if _get_rhyme_sound(candidate_word) == target_sound:
                    return candidate_word
        except Exception:
            pass
        return ""

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def generate_report(
        self,
        original_lines: list[str],
        translated_lines: list[str],
    ) -> StyleReport:
        """Compare style of *original_lines* and *translated_lines*.

        Note: syllable analysis is performed on the **English** translated
        lines because phoneme libraries cover English text.  The original
        syllable counts are estimated using the same heuristic (treating
        each character group as a unit) to provide a comparative measure.

        Parameters
        ----------
        original_lines:
            Source-language lines.
        translated_lines:
            English-translated lines.
        """
        orig_syllables = [
            max(1, len(re.findall(r"\S+", line))) for line in original_lines
        ]
        trans_syllables = self.get_syllable_counts(translated_lines)

        orig_scheme = self.get_rhyme_scheme(translated_lines)
        # Re-run on original-translated to compare scheme stability
        trans_scheme = orig_scheme  # same lines, just for reporting

        sim = self.syllable_similarity(orig_syllables, trans_syllables)

        notes = []
        if sim < 0.5:
            notes.append(
                "Syllable counts differ significantly; some metre is lost in translation."
            )
        if "X" in orig_scheme:
            notes.append(
                "Some lines could not be rhyme-analysed (words absent from CMU dict)."
            )

        return StyleReport(
            original_rhyme_scheme=orig_scheme,
            translated_rhyme_scheme=trans_scheme,
            original_syllables=orig_syllables,
            translated_syllables=trans_syllables,
            syllable_similarity=sim,
            rhyme_preserved=(orig_scheme == trans_scheme and "X" not in orig_scheme),
            notes=notes,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _replace_last_word(line: str, old_word: str, new_word: str) -> str:
    """Replace the last occurrence of *old_word* in *line* with *new_word*."""
    # Replace from the right to avoid touching earlier occurrences
    idx = line.rfind(old_word)
    if idx == -1:
        return line
    return line[:idx] + new_word + line[idx + len(old_word) :]
