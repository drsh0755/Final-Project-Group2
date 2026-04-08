"""
poetry_utils.py
~~~~~~~~~~~~~~~
Utility helpers for the Poetry Translation System.

Responsibilities
----------------
- Language detection (Hindi vs Urdu).
- Text cleaning and normalisation.
- Poem-line splitting and formatting.
- Unicode range constants for Hindi / Urdu scripts.
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Unicode ranges
# ---------------------------------------------------------------------------

#: Devanagari block — used for Hindi (and Sanskrit, Marathi, etc.)
DEVANAGARI_RANGE = ("\u0900", "\u097F")

#: Arabic Presentation Forms + Arabic Supplement — used for Urdu
ARABIC_RANGE = ("\u0600", "\u06FF")

# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


def detect_script(text: str) -> str:
    """Detect whether *text* is primarily Devanagari (Hindi) or Arabic-script (Urdu).

    Parameters
    ----------
    text:
        Raw poem or song text.

    Returns
    -------
    ``"hi"``   if the text appears to be Hindi (Devanagari script).
    ``"ur"``   if the text appears to be Urdu (Nastaliq / Arabic script).
    ``"unknown"`` if neither script dominates.
    """
    devanagari_count = sum(
        1 for ch in text if DEVANAGARI_RANGE[0] <= ch <= DEVANAGARI_RANGE[1]
    )
    arabic_count = sum(
        1 for ch in text if ARABIC_RANGE[0] <= ch <= ARABIC_RANGE[1]
    )

    if devanagari_count == 0 and arabic_count == 0:
        return "unknown"
    return "hi" if devanagari_count >= arabic_count else "ur"


def detect_language(text: str) -> str:
    """High-level language detection that falls back to ``langdetect`` when
    both scripts are absent (e.g. romanised / transliterated input).

    Returns ``"hi"``, ``"ur"``, or ``"unknown"``.
    """
    script = detect_script(text)
    if script != "unknown":
        return script

    try:
        from langdetect import detect, LangDetectException  # type: ignore

        code = detect(text)
        if code in ("hi", "ur"):
            return code
    except Exception:
        pass

    return "unknown"


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------


def remove_extra_whitespace(text: str) -> str:
    """Collapse runs of whitespace (spaces/tabs) within each line."""
    lines = text.splitlines()
    cleaned = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
    return "\n".join(cleaned)


def normalize_unicode(text: str) -> str:
    """Apply Unicode NFC normalisation so composed characters are canonical."""
    return unicodedata.normalize("NFC", text)


def clean_poem(text: str) -> str:
    """Clean a raw poem string.

    Steps
    -----
    1. NFC-normalise Unicode.
    2. Strip trailing/leading whitespace per line.
    3. Remove lines that are entirely punctuation or whitespace.
    4. Collapse internal whitespace.
    """
    text = normalize_unicode(text)
    text = remove_extra_whitespace(text)
    lines = text.splitlines()
    cleaned_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        # Keep empty lines (stanza breaks) but drop punctuation-only lines
        if stripped and re.fullmatch(r"[\W\s]+", stripped, flags=re.UNICODE):
            continue
        cleaned_lines.append(stripped)
    return "\n".join(cleaned_lines)


# ---------------------------------------------------------------------------
# Poem splitting helpers
# ---------------------------------------------------------------------------


def split_into_lines(text: str) -> list[str]:
    """Return a list of non-empty lines from a poem string."""
    return [line for line in text.splitlines() if line.strip()]


def split_into_stanzas(text: str) -> list[list[str]]:
    """Split a poem into stanzas (groups of lines separated by blank lines).

    Returns a list of stanzas, where each stanza is a list of line strings.
    """
    stanzas: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip():
            current.append(line.strip())
        else:
            if current:
                stanzas.append(current)
                current = []
    if current:
        stanzas.append(current)
    return stanzas


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_poem(lines: list[str], title: str = "") -> str:
    """Format a list of translated lines into a display-ready poem string.

    Parameters
    ----------
    lines:
        Translated (English) lines of the poem.
    title:
        Optional title to prepend.

    Returns
    -------
    A formatted multi-line string suitable for printing.
    """
    output_parts: list[str] = []
    if title:
        output_parts.append(title)
        output_parts.append("=" * len(title))
    output_parts.extend(lines)
    return "\n".join(output_parts)


def build_side_by_side(
    original_lines: list[str],
    translated_lines: list[str],
    col_width: int = 45,
) -> str:
    """Build a side-by-side view of original and translated lines.

    Parameters
    ----------
    original_lines:
        Lines from the source poem (Hindi/Urdu).
    translated_lines:
        Corresponding translated lines (English).
    col_width:
        Width of each column in characters.

    Returns
    -------
    A formatted string with two columns.
    """
    header = f"{'Original':<{col_width}} | {'Translation':<{col_width}}"
    separator = "-" * (col_width * 2 + 3)
    rows = [header, separator]
    for orig, trans in zip(original_lines, translated_lines):
        rows.append(f"{orig:<{col_width}} | {trans:<{col_width}}")
    return "\n".join(rows)
