"""
app.py — Flask entry point for the Topic-Sentiment Analysis Tool.

Routes
------
GET  /          → Upload / paste form
POST /analyze   → Run NLP pipeline → render results dashboard
"""

import os
import io
import json
import csv
import traceback

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash

from nlp.preprocessor import prepare_corpus, split_into_sentences
from nlp.topic_model import extract_topics
from nlp.sentiment import analyze_document, aggregate_sentiment

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "topic-sentiment-dev-key-2024")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

ALLOWED_EXTENSIONS = {"txt", "csv"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Render the upload / paste page."""
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Accept text or file upload → run the full NLP pipeline → return results.
    """
    try:
        documents = _extract_documents(request)

        if not documents or all(len(d.strip()) == 0 for d in documents):
            flash("Please provide some text or upload a file with content.", "error")
            return redirect(url_for("index"))

        # Filter out empty docs
        documents = [d.strip() for d in documents if d.strip()]

        # --- User-selected options ----------------------------------------
        n_topics = int(request.form.get("n_topics", 5))
        method = request.form.get("method", "lda").lower()

        # Clamp topic count
        n_topics = max(2, min(n_topics, 15))

        # If fewer docs than topics, reduce topic count
        if len(documents) < n_topics:
            n_topics = max(2, len(documents))

        # --- NLP Pipeline -------------------------------------------------
        # 1. Preprocess
        token_lists, cleaned_docs = prepare_corpus(documents)

        # Check if corpus is too empty after cleaning
        non_empty = [d for d in cleaned_docs if d.strip()]
        if len(non_empty) < 2:
            flash("After cleaning, not enough text remains for topic analysis. Please provide more content.", "error")
            return redirect(url_for("index"))

        # 2. Topic extraction
        topic_results = extract_topics(
            cleaned_docs, n_topics=n_topics, method=method
        )

        # 3. Sentiment analysis (on original text for accuracy)
        doc_sentiments = [analyze_document(doc) for doc in documents]

        # 4. Aggregate sentiment
        overall_sentiment = aggregate_sentiment(doc_sentiments)

        # 5. Per-topic sentiment
        topic_sentiments = _compute_topic_sentiments(
            topic_results, doc_sentiments
        )

        # --- Build template context --------------------------------------
        # Prepare doc-level table data
        doc_table = []
        for i, doc in enumerate(documents):
            doc_table.append({
                "id": i + 1,
                "text": doc[:300] + ("…" if len(doc) > 300 else ""),
                "full_text": doc,
                "topic_id": topic_results["doc_topic_assignments"][i],
                "topic_label": topic_results["topics"][topic_results["doc_topic_assignments"][i]]["label"],
                "sentiment": doc_sentiments[i]["label"],
                "vader_compound": doc_sentiments[i]["vader_compound"],
                "textblob_polarity": doc_sentiments[i]["textblob_polarity"],
            })

        # Serialise data for Plotly charts (JSON-safe)
        chart_data = _build_chart_data(
            topic_results, topic_sentiments, overall_sentiment
        )

        return render_template(
            "results.html",
            documents=doc_table,
            topics=topic_results["topics"],
            topic_sentiments=topic_sentiments,
            overall_sentiment=overall_sentiment,
            chart_data_json=json.dumps(chart_data),
            n_docs=len(documents),
            n_topics=n_topics,
            method=method.upper(),
        )

    except Exception as exc:
        traceback.print_exc()
        flash(f"Analysis failed: {exc}", "error")
        return redirect(url_for("index"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_documents(req) -> list[str]:
    """
    Extract a list of document strings from the request.

    Priority: uploaded file → pasted text.
    For pasted text, split on double newlines or treat each line as a doc.
    """
    # Check for file upload
    file = req.files.get("file")
    if file and file.filename and allowed_file(file.filename):
        ext = file.filename.rsplit(".", 1)[1].lower()
        raw = file.read().decode("utf-8", errors="replace")

        if ext == "csv":
            return _parse_csv(raw)
        else:  # txt
            return _parse_txt(raw)

    # Fall back to pasted text
    text = req.form.get("text", "").strip()
    if text:
        return _parse_txt(text)

    return []


def _parse_txt(raw: str) -> list[str]:
    """
    Split plain text into documents.

    Strategy: if there are paragraph breaks (double newline), split on those.
    Otherwise split on single newlines. If only one block, split into sentences.
    """
    # Try paragraph split first
    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
    if len(paragraphs) >= 3:
        return paragraphs

    # Try line split
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    if len(lines) >= 3:
        return lines

    # Fall back to sentence split
    sentences = split_into_sentences(raw)
    if len(sentences) >= 2:
        return sentences

    # Last resort: return as single doc
    return [raw]


def _parse_csv(raw: str) -> list[str]:
    """Parse CSV — look for a 'text' or 'review' column, else use first column."""
    reader = csv.DictReader(io.StringIO(raw))
    fieldnames = reader.fieldnames or []

    # Find the text column
    text_col = None
    for candidate in ["text", "review", "content", "comment", "message", "body", "Text", "Review", "Content"]:
        if candidate in fieldnames:
            text_col = candidate
            break

    if text_col is None and fieldnames:
        text_col = fieldnames[0]

    if text_col is None:
        return []

    docs = []
    for row in reader:
        val = row.get(text_col, "").strip()
        if val:
            docs.append(val)
    return docs


def _compute_topic_sentiments(topic_results: dict, doc_sentiments: list[dict]) -> list[dict]:
    """Aggregate sentiment scores per topic."""
    n_topics = len(topic_results["topics"])
    assignments = topic_results["doc_topic_assignments"]

    topic_sents = []
    for t in range(n_topics):
        # Indices of docs assigned to this topic
        indices = [i for i, a in enumerate(assignments) if a == t]
        if not indices:
            topic_sents.append({
                "topic_id": t,
                "count": 0,
                "avg_compound": 0,
                "positive": 0,
                "negative": 0,
                "neutral": 0,
            })
            continue

        compounds = [doc_sentiments[i]["vader_compound"] for i in indices]
        labels = [doc_sentiments[i]["label"] for i in indices]
        topic_sents.append({
            "topic_id": t,
            "count": len(indices),
            "avg_compound": round(float(np.mean(compounds)), 4),
            "positive": sum(1 for l in labels if l == "Positive"),
            "negative": sum(1 for l in labels if l == "Negative"),
            "neutral": sum(1 for l in labels if l == "Neutral"),
        })

    return topic_sents


def _build_chart_data(topic_results, topic_sentiments, overall_sentiment):
    """Build JSON-serialisable data for Plotly charts."""

    # 1. Sentiment distribution (donut chart)
    sentiment_dist = {
        "labels": ["Positive", "Neutral", "Negative"],
        "values": [
            overall_sentiment.get("positive", 0),
            overall_sentiment.get("neutral", 0),
            overall_sentiment.get("negative", 0),
        ],
        "colors": ["#00e676", "#ffd740", "#ff5252"],
    }

    # 2. Topic-sentiment heatmap
    topic_labels = [t["label"] for t in topic_results["topics"]]
    heatmap_z = []
    for ts in topic_sentiments:
        total = ts["count"] if ts["count"] > 0 else 1
        heatmap_z.append([
            round(ts["positive"] / total * 100, 1),
            round(ts["neutral"] / total * 100, 1),
            round(ts["negative"] / total * 100, 1),
        ])

    heatmap = {
        "x": ["Positive", "Neutral", "Negative"],
        "y": topic_labels,
        "z": heatmap_z,
    }

    # 3. Topic document counts (bar chart)
    topic_bars = {
        "labels": topic_labels,
        "counts": [ts["count"] for ts in topic_sentiments],
        "avg_compounds": [ts["avg_compound"] for ts in topic_sentiments],
    }

    # 4. Per-topic avg compound (bar chart)
    compound_bars = {
        "labels": topic_labels,
        "values": [ts["avg_compound"] for ts in topic_sentiments],
    }

    return {
        "sentiment_dist": sentiment_dist,
        "heatmap": heatmap,
        "topic_bars": topic_bars,
        "compound_bars": compound_bars,
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
