"""
translator.py
~~~~~~~~~~~~~
Hindi/Urdu → English poetry translator.

Uses Helsinki-NLP's MarianMT models (``opus-mt-hi-en`` and ``opus-mt-ur-en``)
from Hugging Face Transformers to perform neural machine translation.

Key design choices
------------------
- Translate *line-by-line* to preserve the line structure of the poem.
- Support batched translation for efficiency.
- Models are loaded lazily (only when first needed) to keep import time fast.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import MarianMTModel, MarianTokenizer  # pragma: no cover

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------

#: Hugging Face model IDs keyed by ISO 639-1 language code.
MODEL_IDS: dict[str, str] = {
    "hi": "Helsinki-NLP/opus-mt-hi-en",
    "ur": "Helsinki-NLP/opus-mt-ur-en",
}

#: Maximum token length accepted by MarianMT models.
MAX_LENGTH = 512


# ---------------------------------------------------------------------------
# Translator class
# ---------------------------------------------------------------------------


class PoetryTranslator:
    """Translates Hindi or Urdu poetry into English using MarianMT.

    Parameters
    ----------
    source_lang:
        ``"hi"`` for Hindi (Devanagari) or ``"ur"`` for Urdu (Nastaliq).

    Examples
    --------
    >>> translator = PoetryTranslator(source_lang="hi")
    >>> translated = translator.translate_poem("तू किसी रेल सी गुज़रती है\\nमैं किसी पुल सा थरथराता हूँ")
    """

    def __init__(self, source_lang: str = "hi") -> None:
        if source_lang not in MODEL_IDS:
            raise ValueError(
                f"Unsupported source language '{source_lang}'. "
                f"Choose from: {list(MODEL_IDS.keys())}"
            )
        self.source_lang = source_lang
        self._model: MarianMTModel | None = None
        self._tokenizer: MarianTokenizer | None = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Download (first time) and cache the MarianMT model and tokenizer."""
        from transformers import MarianMTModel, MarianTokenizer  # type: ignore

        model_id = MODEL_IDS[self.source_lang]
        logger.info("Loading translation model '%s' …", model_id)
        self._tokenizer = MarianTokenizer.from_pretrained(model_id)
        self._model = MarianMTModel.from_pretrained(model_id)
        logger.info("Model loaded successfully.")

    @property
    def model(self) -> "MarianMTModel":
        if self._model is None:
            self._load_model()
        return self._model  # type: ignore[return-value]

    @property
    def tokenizer(self) -> "MarianTokenizer":
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Core translation helpers
    # ------------------------------------------------------------------

    def _translate_batch(self, sentences: list[str]) -> list[str]:
        """Translate a list of sentences in one forward pass.

        Parameters
        ----------
        sentences:
            Non-empty strings in the source language.

        Returns
        -------
        List of translated English strings in the same order.
        """
        import torch  # type: ignore

        inputs = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        with torch.no_grad():
            translated_ids = self.model.generate(**inputs)
        return [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in translated_ids
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate(self, text: str) -> str:
        """Translate a single sentence or short paragraph.

        Parameters
        ----------
        text:
            Source-language text.

        Returns
        -------
        Translated English string.
        """
        results = self._translate_batch([text.strip()])
        return results[0]

    def translate_poem(
        self,
        poem_text: str,
        batch_size: int = 8,
    ) -> list[str]:
        """Translate a poem line-by-line to preserve its structure.

        Empty lines (stanza separators) are preserved as empty strings in
        the returned list so the caller can reconstruct stanza breaks.

        Parameters
        ----------
        poem_text:
            Full poem text with newline-separated lines.
        batch_size:
            Number of lines to translate per model forward pass.

        Returns
        -------
        List of translated lines (empty strings for blank lines).
        """
        raw_lines = poem_text.splitlines()

        # Separate blank/non-blank lines while remembering positions
        non_blank: list[tuple[int, str]] = []
        for idx, line in enumerate(raw_lines):
            if line.strip():
                non_blank.append((idx, line.strip()))

        if not non_blank:
            return []

        # Translate in batches
        translated_map: dict[int, str] = {}
        for start in range(0, len(non_blank), batch_size):
            chunk = non_blank[start : start + batch_size]
            idxs = [i for i, _ in chunk]
            texts = [t for _, t in chunk]
            results = self._translate_batch(texts)
            for idx, result in zip(idxs, results):
                translated_map[idx] = result

        # Reconstruct with blank-line gaps
        output: list[str] = []
        for idx, line in enumerate(raw_lines):
            if line.strip():
                output.append(translated_map[idx])
            else:
                output.append("")

        return output
