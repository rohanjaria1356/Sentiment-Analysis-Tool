"""
sentiment.py — Sentiment analysis using VADER and TextBlob.

Provides per-document and aggregated sentiment scoring.
"""

from __future__ import annotations

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

# Ensure VADER lexicon is available
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

# Module-level analyser (initialised once)
_vader = SentimentIntensityAnalyzer()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_sentiment_vader(text: str) -> dict:
    """
    Analyse sentiment of a single text using VADER.

    Returns dict with keys: compound, pos, neg, neu, label.
    """
    scores = _vader.polarity_scores(text)
    label = _compound_to_label(scores["compound"])
    return {
        "compound": round(scores["compound"], 4),
        "pos": round(scores["pos"], 4),
        "neg": round(scores["neg"], 4),
        "neu": round(scores["neu"], 4),
        "label": label,
    }


def analyze_sentiment_textblob(text: str) -> dict:
    """
    Analyse sentiment of a single text using TextBlob.

    Returns dict with keys: polarity, subjectivity, label.
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    label = _polarity_to_label(polarity)
    return {
        "polarity": round(polarity, 4),
        "subjectivity": round(subjectivity, 4),
        "label": label,
    }


def analyze_document(text: str) -> dict:
    """Run both VADER and TextBlob on a single document and merge results."""
    vader = analyze_sentiment_vader(text)
    tb = analyze_sentiment_textblob(text)
    # Use VADER compound as the primary score; TextBlob as secondary
    return {
        "vader_compound": vader["compound"],
        "vader_label": vader["label"],
        "vader_pos": vader["pos"],
        "vader_neg": vader["neg"],
        "vader_neu": vader["neu"],
        "textblob_polarity": tb["polarity"],
        "textblob_subjectivity": tb["subjectivity"],
        "textblob_label": tb["label"],
        # Consensus label — agree on positive/negative, else neutral
        "label": vader["label"] if vader["label"] == tb["label"] else _consensus(vader, tb),
    }


def aggregate_sentiment(doc_sentiments: list[dict]) -> dict:
    """
    Aggregate per-document sentiment dicts into summary statistics.

    Parameters
    ----------
    doc_sentiments : list[dict]
        Output of ``analyze_document`` for each document.

    Returns
    -------
    dict with counts, averages, and distribution.
    """
    n = len(doc_sentiments)
    if n == 0:
        return {"count": 0}

    pos_count = sum(1 for d in doc_sentiments if d["label"] == "Positive")
    neg_count = sum(1 for d in doc_sentiments if d["label"] == "Negative")
    neu_count = sum(1 for d in doc_sentiments if d["label"] == "Neutral")

    avg_compound = sum(d["vader_compound"] for d in doc_sentiments) / n
    avg_polarity = sum(d["textblob_polarity"] for d in doc_sentiments) / n
    avg_subjectivity = sum(d["textblob_subjectivity"] for d in doc_sentiments) / n

    return {
        "count": n,
        "positive": pos_count,
        "negative": neg_count,
        "neutral": neu_count,
        "positive_pct": round(pos_count / n * 100, 1),
        "negative_pct": round(neg_count / n * 100, 1),
        "neutral_pct": round(neu_count / n * 100, 1),
        "avg_vader_compound": round(avg_compound, 4),
        "avg_textblob_polarity": round(avg_polarity, 4),
        "avg_textblob_subjectivity": round(avg_subjectivity, 4),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compound_to_label(compound: float) -> str:
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


def _polarity_to_label(polarity: float) -> str:
    if polarity > 0.1:
        return "Positive"
    if polarity < -0.1:
        return "Negative"
    return "Neutral"


def _consensus(vader: dict, tb: dict) -> str:
    """When VADER and TextBlob disagree, use the stronger signal."""
    vader_strength = abs(vader["compound"])
    tb_strength = abs(tb["polarity"])
    if vader_strength >= tb_strength:
        return vader["label"]
    return tb["label"]
