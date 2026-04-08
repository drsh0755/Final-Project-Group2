"""
app.py
~~~~~~
Command-line interface for the Hindi/Urdu → English Poetry Translation System.

Usage
-----
Translate a poem from a file::

    python app.py --file data/sample_poems_hindi.txt --lang hi

Translate inline text::

    python app.py --text "तू किसी रेल सी गुज़रती है" --lang hi

Full pipeline (translation + topic modelling + style report)::

    python app.py --file data/sample_poems_urdu.txt --lang ur --topics --style

Run with ``--help`` for all options.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    poem_text: str,
    lang: str,
    *,
    run_topics: bool = True,
    run_style: bool = True,
    verbose: bool = False,
) -> dict:
    """Execute the full translation pipeline.

    Parameters
    ----------
    poem_text:
        Raw Hindi or Urdu poem text.
    lang:
        ``"hi"`` for Hindi, ``"ur"`` for Urdu.
    run_topics:
        Whether to perform LDA topic modelling.
    run_style:
        Whether to generate a style report.
    verbose:
        Print additional progress information.

    Returns
    -------
    Dictionary with keys:
    - ``"original_lines"`` — list of source lines.
    - ``"translated_lines"`` — list of translated English lines.
    - ``"topics"`` — list of :class:`~src.topic_modeling.Topic` (or ``[]``).
    - ``"style_report"`` — :class:`~src.style_analyzer.StyleReport` or ``None``.
    """
    from src.poetry_utils import (
        clean_poem,
        detect_language,
        split_into_lines,
        build_side_by_side,
    )
    from src.translator import PoetryTranslator
    from src.style_analyzer import StyleAnalyzer
    from src.topic_modeling import TopicModeler

    # --- 0. Clean input --------------------------------------------------------
    poem_text = clean_poem(poem_text)

    # --- 1. Language detection -------------------------------------------------
    detected = detect_language(poem_text)
    if detected != "unknown" and detected != lang:
        logger.warning(
            "Detected language '%s' differs from requested '%s'. Using '%s'.",
            detected,
            lang,
            lang,
        )

    if verbose:
        print(f"[info] Language : {lang}")

    # --- 2. Topic modelling (on source text) -----------------------------------
    topics = []
    if run_topics:
        if verbose:
            print("[info] Extracting topics …")
        try:
            modeler = TopicModeler(num_topics=3)
            topics = modeler.get_topics(poem_text, language=lang)
            print(TopicModeler.describe_topics(topics))
            print()
        except Exception as exc:
            logger.warning("Topic modelling skipped: %s", exc)

    # --- 3. Translation --------------------------------------------------------
    if verbose:
        print("[info] Translating poem …")
    translator = PoetryTranslator(source_lang=lang)
    translated_lines = translator.translate_poem(poem_text)

    original_lines = split_into_lines(poem_text)
    non_empty_translated = [l for l in translated_lines if l.strip()]

    # --- 4. Style analysis & post-processing -----------------------------------
    style_report = None
    if run_style and original_lines and non_empty_translated:
        if verbose:
            print("[info] Analysing style …")
        analyzer = StyleAnalyzer(language=lang)
        rhyme_scheme = analyzer.get_rhyme_scheme(non_empty_translated)
        improved = analyzer.apply_style_preservation(
            non_empty_translated, rhyme_scheme
        )
        # Reconstruct translated_lines preserving blank-line gaps
        improved_iter = iter(improved)
        translated_lines = [
            next(improved_iter) if line.strip() else ""
            for line in translated_lines
        ]
        non_empty_translated = [l for l in translated_lines if l.strip()]
        style_report = analyzer.generate_report(original_lines, non_empty_translated)

    # --- 5. Output -------------------------------------------------------------
    print("=== Translation ===")
    print(build_side_by_side(original_lines, non_empty_translated))
    print()

    if style_report:
        print(style_report.summary())
        print()

    return {
        "original_lines": original_lines,
        "translated_lines": non_empty_translated,
        "topics": topics,
        "style_report": style_report,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hindi/Urdu → English Poetry Translation System\n"
            "Translates poems and songs while preserving poetic rhythm and style."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--file",
        type=Path,
        metavar="PATH",
        help="Path to a text file containing the poem.",
    )
    source.add_argument(
        "--text",
        type=str,
        metavar="TEXT",
        help="Inline poem text (quote the full string).",
    )
    parser.add_argument(
        "--lang",
        choices=["hi", "ur"],
        default="hi",
        help="Source language: 'hi' for Hindi, 'ur' for Urdu (default: hi).",
    )
    parser.add_argument(
        "--topics",
        action="store_true",
        help="Run LDA topic modelling on the source poem.",
    )
    parser.add_argument(
        "--style",
        action="store_true",
        help="Analyse and report poetic style metrics.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress information.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Read poem text
    if args.file:
        try:
            poem_text = args.file.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"Error reading file: {exc}", file=sys.stderr)
            return 1
    else:
        poem_text = args.text

    if not poem_text.strip():
        print("Error: poem text is empty.", file=sys.stderr)
        return 1

    try:
        run_pipeline(
            poem_text,
            lang=args.lang,
            run_topics=args.topics,
            run_style=args.style,
            verbose=args.verbose,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Pipeline error: {exc}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
