"""Tests for nlp.sentiment module."""

import pytest
from nlp.sentiment import (
    analyze_sentiment_vader,
    analyze_sentiment_textblob,
    analyze_document,
    aggregate_sentiment,
)


class TestVader:
    """Tests for VADER sentiment analysis."""

    def test_positive_text(self):
        result = analyze_sentiment_vader("This product is absolutely amazing and wonderful!")
        assert result["label"] == "Positive"
        assert result["compound"] > 0

    def test_negative_text(self):
        result = analyze_sentiment_vader("This is terrible, awful, and completely broken.")
        assert result["label"] == "Negative"
        assert result["compound"] < 0

    def test_neutral_text(self):
        result = analyze_sentiment_vader("The meeting is at 3pm in room 204.")
        assert result["label"] == "Neutral"

    def test_returns_all_keys(self):
        result = analyze_sentiment_vader("Test text.")
        assert all(k in result for k in ["compound", "pos", "neg", "neu", "label"])


class TestTextBlob:
    """Tests for TextBlob sentiment analysis."""

    def test_positive_text(self):
        result = analyze_sentiment_textblob("This is a great and wonderful product!")
        assert result["polarity"] > 0

    def test_negative_text(self):
        result = analyze_sentiment_textblob("This is an ugly and terrible experience.")
        assert result["polarity"] < 0

    def test_returns_all_keys(self):
        result = analyze_sentiment_textblob("Test text.")
        assert all(k in result for k in ["polarity", "subjectivity", "label"])


class TestAnalyzeDocument:
    """Tests for the combined analysis function."""

    def test_returns_both_scores(self):
        result = analyze_document("Great product, I love it!")
        assert "vader_compound" in result
        assert "textblob_polarity" in result
        assert "label" in result

    def test_consensus_label_exists(self):
        result = analyze_document("This is wonderful!")
        assert result["label"] in ("Positive", "Negative", "Neutral")


class TestAggregateSentiment:
    """Tests for aggregate_sentiment."""

    def test_empty_input(self):
        result = aggregate_sentiment([])
        assert result["count"] == 0

    def test_counts_correct(self):
        docs = [
            analyze_document("Amazing!"),
            analyze_document("Terrible!"),
            analyze_document("The meeting is at noon."),
        ]
        result = aggregate_sentiment(docs)
        assert result["count"] == 3
        assert result["positive"] + result["negative"] + result["neutral"] == 3

    def test_percentages_sum_to_100(self):
        docs = [analyze_document(f"Text number {i}") for i in range(10)]
        result = aggregate_sentiment(docs)
        total_pct = result["positive_pct"] + result["negative_pct"] + result["neutral_pct"]
        assert abs(total_pct - 100.0) < 0.5  # floating-point tolerance
