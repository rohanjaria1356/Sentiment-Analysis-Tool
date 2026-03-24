"""Tests for nlp.topic_model module."""

import pytest
from nlp.preprocessor import prepare_corpus
from nlp.topic_model import extract_topics, get_topic_label


# Fixture: a small corpus
SAMPLE_DOCS = [
    "The camera quality on this phone is amazing and the battery lasts all day",
    "Great phone with excellent camera and long battery life",
    "The restaurant food was delicious and the service was outstanding",
    "Amazing dining experience with wonderful food and friendly staff",
    "The movie plot was confusing and the acting was terrible",
    "Boring movie with bad acting and a weak storyline",
    "The software update improved performance and fixed many bugs",
    "After updating the software everything runs much faster and smoother",
]


@pytest.fixture
def cleaned_corpus():
    _, cleaned = prepare_corpus(SAMPLE_DOCS)
    return cleaned


class TestExtractTopics:
    """Tests for extract_topics."""

    def test_lda_returns_correct_keys(self, cleaned_corpus):
        result = extract_topics(cleaned_corpus, n_topics=3, method="lda")
        assert "topics" in result
        assert "doc_topic_matrix" in result
        assert "doc_topic_assignments" in result
        assert "feature_names" in result

    def test_nmf_returns_correct_keys(self, cleaned_corpus):
        result = extract_topics(cleaned_corpus, n_topics=3, method="nmf")
        assert "topics" in result
        assert "doc_topic_assignments" in result

    def test_correct_number_of_topics(self, cleaned_corpus):
        for n in [2, 3, 4]:
            result = extract_topics(cleaned_corpus, n_topics=n, method="lda")
            assert len(result["topics"]) == n

    def test_topic_has_keywords(self, cleaned_corpus):
        result = extract_topics(cleaned_corpus, n_topics=3, method="lda")
        for topic in result["topics"]:
            assert "keywords" in topic
            assert len(topic["keywords"]) > 0

    def test_topic_has_label(self, cleaned_corpus):
        result = extract_topics(cleaned_corpus, n_topics=3, method="lda")
        for topic in result["topics"]:
            assert "label" in topic
            assert len(topic["label"]) > 0

    def test_assignments_length_matches_docs(self, cleaned_corpus):
        result = extract_topics(cleaned_corpus, n_topics=3, method="lda")
        assert len(result["doc_topic_assignments"]) == len(cleaned_corpus)

    def test_invalid_method_raises(self, cleaned_corpus):
        with pytest.raises(ValueError):
            extract_topics(cleaned_corpus, n_topics=3, method="invalid")


class TestGetTopicLabel:
    """Tests for get_topic_label."""

    def test_generates_label(self):
        label = get_topic_label(["camera", "phone", "battery", "screen"])
        assert "Camera" in label
        assert "Phone" in label

    def test_max_words_limits_output(self):
        label = get_topic_label(["a", "b", "c", "d", "e"], max_words=2)
        parts = label.split(" / ")
        assert len(parts) == 2
