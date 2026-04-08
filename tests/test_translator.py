"""
Tests for src/translator.py

Heavy model downloads and the torch/transformers libraries are mocked so
these tests run without network access or GPU resources.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from src.translator import PoetryTranslator, MODEL_IDS


# ---------------------------------------------------------------------------
# Torch stub — injected into sys.modules before importing translator code
# ---------------------------------------------------------------------------


def _make_torch_stub() -> ModuleType:
    """Return a minimal mock of the torch module sufficient for _translate_batch."""
    torch_stub = ModuleType("torch")

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    torch_stub.no_grad = _NoGrad  # type: ignore[attr-defined]
    return torch_stub


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_mock_translator(
    source_lang: str = "hi",
) -> tuple[PoetryTranslator, MagicMock, MagicMock]:
    """Return a PoetryTranslator with mocked model, tokenizer, and _translate_batch.

    ``_translate_batch`` is replaced with a stub that returns
    ``"translated line"`` for each input sentence, avoiding the need
    for torch or actual model weights.
    """
    translator = PoetryTranslator(source_lang=source_lang)

    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    mock_tokenizer.return_value = MagicMock()
    mock_model.generate.return_value = [[1, 2, 3]]
    mock_tokenizer.decode.return_value = "translated line"

    translator._tokenizer = mock_tokenizer
    translator._model = mock_model

    # Stub _translate_batch so tests don't need torch installed
    translator._translate_batch = lambda sentences: ["translated line"] * len(sentences)  # type: ignore[method-assign]

    return translator, mock_tokenizer, mock_model


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestPoetryTranslatorInit:
    def test_valid_hindi(self):
        t = PoetryTranslator(source_lang="hi")
        assert t.source_lang == "hi"

    def test_valid_urdu(self):
        t = PoetryTranslator(source_lang="ur")
        assert t.source_lang == "ur"

    def test_invalid_lang_raises(self):
        with pytest.raises(ValueError, match="Unsupported source language"):
            PoetryTranslator(source_lang="fr")

    def test_model_not_loaded_on_init(self):
        t = PoetryTranslator(source_lang="hi")
        assert t._model is None
        assert t._tokenizer is None


# ---------------------------------------------------------------------------
# MODEL_IDS mapping
# ---------------------------------------------------------------------------


class TestModelIds:
    def test_hindi_model_id(self):
        assert "Helsinki-NLP" in MODEL_IDS["hi"]
        assert "hi-en" in MODEL_IDS["hi"]

    def test_urdu_model_id(self):
        assert "Helsinki-NLP" in MODEL_IDS["ur"]
        assert "ur-en" in MODEL_IDS["ur"]


# ---------------------------------------------------------------------------
# translate()
# ---------------------------------------------------------------------------


class TestTranslate:
    def test_translate_returns_string(self):
        translator, mock_tok, mock_model = make_mock_translator("hi")
        result = translator.translate("माटी कहे")
        assert isinstance(result, str)
        assert result == "translated line"

    def test_translate_calls_translate_batch(self):
        called_with = []
        translator = PoetryTranslator(source_lang="hi")
        translator._translate_batch = lambda sentences: (  # type: ignore[method-assign]
            called_with.extend(sentences) or ["out"] * len(sentences)
        )
        translator.translate("कुछ पंक्तियाँ")
        assert "कुछ पंक्तियाँ" in called_with


# ---------------------------------------------------------------------------
# translate_poem()
# ---------------------------------------------------------------------------


class TestTranslatePoem:
    def test_returns_list(self):
        translator, _, _ = make_mock_translator("hi")
        result = translator.translate_poem("line one\nline two")
        assert isinstance(result, list)

    def test_preserves_blank_line_gaps(self):
        translator, _, _ = make_mock_translator("hi")
        poem = "line one\nline two\n\nline three"
        result = translator.translate_poem(poem)
        # Blank line should be preserved as empty string
        assert "" in result

    def test_empty_poem_returns_empty_list(self):
        translator, _, _ = make_mock_translator("hi")
        assert translator.translate_poem("") == []
        assert translator.translate_poem("   ") == []

    def test_output_length_matches_input_lines(self):
        translator, _, _ = make_mock_translator("hi")
        poem = "line one\nline two\nline three"
        result = translator.translate_poem(poem)
        assert len(result) == 3

    def test_batching_multiple_lines(self):
        """Verify batching does not drop lines."""
        translator, _, _ = make_mock_translator("hi")
        # 10 lines — exceeds default batch_size=8
        poem = "\n".join([f"line {i}" for i in range(10)])
        result = translator.translate_poem(poem, batch_size=3)
        non_empty = [r for r in result if r]
        assert len(non_empty) == 10
