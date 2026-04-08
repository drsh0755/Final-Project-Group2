"""
Tests for src/topic_modeling.py
"""

import pytest
from src.topic_modeling import TopicModeler, Topic, HINDI_STOPWORDS, URDU_STOPWORDS


# ---------------------------------------------------------------------------
# TopicModeler.preprocess
# ---------------------------------------------------------------------------


class TestPreprocess:
    def setup_method(self):
        self.modeler = TopicModeler(num_topics=2)

    def test_returns_list_of_lists(self):
        result = self.modeler.preprocess("माटी कहे कुम्हार से", language="hi")
        assert isinstance(result, list)
        assert all(isinstance(doc, list) for doc in result)

    def test_removes_hindi_stopwords(self):
        stopword = "और"  # common Hindi stopword
        text = f"{stopword} माटी कुम्हार"
        result = self.modeler.preprocess(text, language="hi")
        for doc in result:
            assert stopword not in doc

    def test_removes_urdu_stopwords(self):
        stopword = "ہے"  # common Urdu stopword
        text = f"{stopword} خواہشیں غزل محبت"
        result = self.modeler.preprocess(text, language="ur")
        for doc in result:
            assert stopword not in doc

    def test_removes_english_stopwords(self):
        text = "the earth speaks with wisdom every day"
        result = self.modeler.preprocess(text, language="en")
        for doc in result:
            assert "the" not in doc

    def test_blank_lines_excluded(self):
        text = "माटी कहे\n\nकुम्हार से"
        result = self.modeler.preprocess(text, language="hi")
        # Blank lines should not produce empty documents
        assert all(len(doc) >= 2 for doc in result)

    def test_short_lines_filtered(self):
        # Single token lines should be dropped (< 2 tokens)
        text = "है\nमाटी कहे कुम्हार"
        result = self.modeler.preprocess(text, language="hi")
        for doc in result:
            assert len(doc) >= 2

    def test_empty_text_returns_empty(self):
        assert self.modeler.preprocess("", language="hi") == []


# ---------------------------------------------------------------------------
# TopicModeler.get_topics
# ---------------------------------------------------------------------------


class TestGetTopics:
    def setup_method(self):
        self.modeler = TopicModeler(num_topics=2, passes=5)

    def test_returns_list_of_topics(self):
        text = (
            "माटी कहे कुम्हार जाति पूछो साधु ज्ञान\n"
            "बड़ा हुआ खजूर पेड़ छाया फल दूर पंथी\n"
            "रात तारे चमकते आकाश प्रेम ज्योति मन\n"
            "जलाया दिल राख जस्तजो प्रभु भक्ति भाव\n"
            "ज्ञान माया जगत सत्य पथ मोह धर्म काल\n"
        )
        topics = self.modeler.get_topics(text, language="hi")
        assert isinstance(topics, list)
        assert len(topics) > 0

    def test_topic_has_keywords(self):
        text = (
            "माटी कहे कुम्हार जाति पूछो साधु ज्ञान\n"
            "बड़ा हुआ खजूर पेड़ छाया फल दूर पंथी\n"
            "रात तारे चमकते आकाश प्रेम ज्योति मन\n"
            "जलाया दिल राख जस्तजो प्रभु भक्ति भाव\n"
        )
        topics = self.modeler.get_topics(text, language="hi")
        for topic in topics:
            assert isinstance(topic, Topic)
            assert len(topic.keywords) > 0

    def test_topics_sorted_by_weight_descending(self):
        text = (
            "माटी कहे कुम्हार जाति ज्ञान\n"
            "बड़ा हुआ खजूर पेड़ छाया\n"
            "रात तारे चमकते आकाश प्रेम\n"
            "जलाया दिल राख प्रभु भक्ति\n"
        )
        topics = self.modeler.get_topics(text, language="hi")
        weights = [t.weight for t in topics]
        assert weights == sorted(weights, reverse=True)

    def test_empty_text_raises_value_error(self):
        with pytest.raises(ValueError):
            self.modeler.get_topics("", language="hi")

    def test_too_short_raises_value_error(self):
        # Single stopword line — no usable tokens
        with pytest.raises(ValueError):
            self.modeler.get_topics("और\nहै", language="hi")

    def test_english_text_topics(self):
        text = (
            "the earth and clay speak wisdom every morning\n"
            "bright stars shine in the sky above us always\n"
            "love and devotion guide the heart through darkness\n"
            "wisdom knowledge truth light guide path journey soul\n"
        )
        topics = self.modeler.get_topics(text, language="en")
        assert len(topics) > 0


# ---------------------------------------------------------------------------
# TopicModeler.describe_topics
# ---------------------------------------------------------------------------


class TestDescribeTopics:
    def test_empty_returns_no_topics(self):
        result = TopicModeler.describe_topics([])
        assert "No topics found" in result

    def test_non_empty_returns_formatted(self):
        topic = Topic(topic_id=0, keywords=["love", "night", "star"], weight=0.4)
        result = TopicModeler.describe_topics([topic])
        assert "Topic 0" in result
        assert "love" in result

    def test_contains_heading(self):
        topic = Topic(topic_id=0, keywords=["a", "b"], weight=0.5)
        result = TopicModeler.describe_topics([topic])
        assert "Themes" in result


# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------


class TestStopwords:
    def test_hindi_stopwords_non_empty(self):
        assert len(HINDI_STOPWORDS) > 0

    def test_urdu_stopwords_non_empty(self):
        assert len(URDU_STOPWORDS) > 0

    def test_hindi_common_word_in_stopwords(self):
        assert "और" in HINDI_STOPWORDS

    def test_urdu_common_word_in_stopwords(self):
        assert "ہے" in URDU_STOPWORDS


# ---------------------------------------------------------------------------
# Topic __str__
# ---------------------------------------------------------------------------


class TestTopicStr:
    def test_str_contains_keywords(self):
        t = Topic(topic_id=1, keywords=["moon", "night", "sky"], weight=0.35)
        s = str(t)
        assert "moon" in s
        assert "night" in s
        assert "sky" in s
        assert "Topic 1" in s
