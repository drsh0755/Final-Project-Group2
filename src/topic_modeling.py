"""
topic_modeling.py
~~~~~~~~~~~~~~~~~
LDA-based topic modelling to identify thematic content in poems.

The module operates in two phases:

1. **Preprocessing** — tokenise the input text and remove stopwords.
   For source-language (Hindi/Urdu) text the tokens are simply whitespace-
   split unicode words; for English text standard NLTK stopwords are applied.

2. **Topic extraction** — build a gensim Dictionary + Bag-of-Words corpus,
   then train an LDA model to surface the dominant topics.

Usage example::

    modeler = TopicModeler(num_topics=3)
    topics = modeler.get_topics("रात के तारे चमकते हैं ...")
    for t in topics:
        print(t)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Topic:
    """Represents a single LDA topic."""

    topic_id: int
    keywords: list[str]
    weight: float  # dominant document weight for this topic

    def __str__(self) -> str:
        kw = ", ".join(self.keywords)
        return f"Topic {self.topic_id} ({self.weight:.1%}): {kw}"


# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------

#: Minimal Hindi stopwords (Devanagari) used to reduce noise.
HINDI_STOPWORDS: frozenset[str] = frozenset(
    [
        "है", "हैं", "था", "थे", "थी", "हो", "हूँ", "और", "या", "पर",
        "में", "को", "से", "के", "का", "की", "एक", "यह", "वह", "भी",
        "तो", "कि", "नहीं", "पर", "लिए", "आ", "जा", "कर", "मैं", "तुम",
        "हम", "वे", "ने", "द्वारा", "साथ", "अपने", "ही", "जो", "इस",
    ]
)

#: Minimal Urdu stopwords (Arabic script).
URDU_STOPWORDS: frozenset[str] = frozenset(
    [
        "ہے", "ہیں", "تھا", "تھے", "تھی", "ہو", "اور", "یا", "پر",
        "میں", "کو", "سے", "کے", "کا", "کی", "ایک", "یہ", "وہ", "بھی",
        "تو", "کہ", "نہیں", "لیے", "آ", "جا", "کر", "میں", "تم",
        "ہم", "وہ", "نے", "ساتھ", "اپنے", "ہی", "جو", "اس",
    ]
)


def _get_english_stopwords() -> frozenset[str]:
    """Load NLTK English stopwords, downloading them if necessary."""
    try:
        import nltk  # type: ignore
        from nltk.corpus import stopwords  # type: ignore

        try:
            return frozenset(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            return frozenset(stopwords.words("english"))
    except Exception:
        # Minimal fallback if NLTK is unavailable
        return frozenset(
            [
                "the", "a", "an", "and", "or", "but", "in", "on", "at",
                "to", "for", "of", "with", "is", "are", "was", "were",
                "i", "you", "he", "she", "we", "they", "it", "this", "that",
            ]
        )


# ---------------------------------------------------------------------------
# TopicModeler
# ---------------------------------------------------------------------------


class TopicModeler:
    """Extract thematic topics from a poem using Latent Dirichlet Allocation.

    Parameters
    ----------
    num_topics:
        Number of topics to extract.
    top_n_words:
        Number of keywords to surface per topic.
    passes:
        Number of LDA training passes over the corpus.
    """

    def __init__(
        self,
        num_topics: int = 3,
        top_n_words: int = 5,
        passes: int = 10,
    ) -> None:
        self.num_topics = num_topics
        self.top_n_words = top_n_words
        self.passes = passes

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(self, text: str, language: str = "hi") -> list[list[str]]:
        """Tokenise *text* and remove stopwords.

        Each line (or sentence) becomes a document in the returned corpus.
        Short documents (< 2 tokens) are filtered out.

        Parameters
        ----------
        text:
            Raw poem text.
        language:
            ``"hi"`` for Hindi, ``"ur"`` for Urdu, or ``"en"`` for English.

        Returns
        -------
        List of token lists (one per document/line).
        """
        if language == "en":
            stopwords = _get_english_stopwords()
        elif language == "ur":
            stopwords = URDU_STOPWORDS
        else:
            stopwords = HINDI_STOPWORDS

        documents: list[list[str]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            # Unicode word tokenisation (handles both scripts)
            tokens = re.findall(r"\w+", line, flags=re.UNICODE)
            tokens = [t for t in tokens if t not in stopwords and len(t) > 1]
            if len(tokens) >= 2:
                documents.append(tokens)

        return documents

    # ------------------------------------------------------------------
    # Topic extraction
    # ------------------------------------------------------------------

    def get_topics(self, text: str, language: str = "hi") -> list[Topic]:
        """Extract topics from *text*.

        Parameters
        ----------
        text:
            Poem or song text to analyse.
        language:
            Source language of *text* (``"hi"``, ``"ur"``, or ``"en"``).

        Returns
        -------
        List of :class:`Topic` objects sorted by descending weight.
        Raises :class:`ValueError` if the text is too short to model.
        """
        from gensim import corpora, models  # type: ignore

        documents = self.preprocess(text, language=language)
        if not documents:
            raise ValueError(
                "The poem is too short or contains no usable tokens for topic modelling."
            )

        dictionary = corpora.Dictionary(documents)
        # Filter extremes to avoid single-occurrence noise
        dictionary.filter_extremes(no_below=1, no_above=1.0)

        corpus = [dictionary.doc2bow(doc) for doc in documents]

        effective_topics = min(self.num_topics, len(dictionary))
        if effective_topics < 1:
            raise ValueError("Vocabulary too small for topic modelling.")

        lda_model = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=effective_topics,
            passes=self.passes,
            random_state=42,
        )

        # Compute dominant topic weights by averaging over all documents
        topic_weights: dict[int, float] = {i: 0.0 for i in range(effective_topics)}
        for bow in corpus:
            doc_topics = lda_model.get_document_topics(bow, minimum_probability=0.0)
            for tid, prob in doc_topics:
                topic_weights[tid] += prob
        n_docs = max(len(corpus), 1)
        topic_weights = {k: v / n_docs for k, v in topic_weights.items()}

        topics: list[Topic] = []
        for tid in range(effective_topics):
            raw_terms = lda_model.show_topic(tid, topn=self.top_n_words)
            keywords = [word for word, _ in raw_terms]
            topics.append(
                Topic(
                    topic_id=tid,
                    keywords=keywords,
                    weight=topic_weights[tid],
                )
            )

        return sorted(topics, key=lambda t: t.weight, reverse=True)

    # ------------------------------------------------------------------
    # Theme label (human-readable)
    # ------------------------------------------------------------------

    @staticmethod
    def describe_topics(topics: list[Topic]) -> str:
        """Return a formatted multi-line description of discovered topics."""
        if not topics:
            return "No topics found."
        lines = ["=== Discovered Themes ==="]
        for topic in topics:
            lines.append(str(topic))
        return "\n".join(lines)
