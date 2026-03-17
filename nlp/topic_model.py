"""
topic_model.py — Topic extraction using LDA and NMF.

Uses scikit-learn's TF-IDF vectorizer with Latent Dirichlet Allocation (LDA)
or Non-negative Matrix Factorization (NMF) to discover latent topics in a
collection of documents.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_topics(
    cleaned_docs: list[str],
    n_topics: int = 5,
    method: str = "lda",
    n_top_words: int = 10,
    max_features: int = 5000,
) -> dict:
    """
    Extract topics from pre-processed documents.

    Parameters
    ----------
    cleaned_docs : list[str]
        Cleaned, lemmatized documents (joined strings).
    n_topics : int
        Number of topics to extract.
    method : str
        ``'lda'`` or ``'nmf'``.
    n_top_words : int
        Number of keywords per topic.
    max_features : int
        Maximum vocabulary size for the vectorizer.

    Returns
    -------
    dict with keys:
        - topics : list[dict]  — each with ``topic_id``, ``keywords``, ``label``
        - doc_topic_matrix : np.ndarray  — shape (n_docs, n_topics)
        - doc_topic_assignments : list[int]  — dominant topic per document
        - feature_names : list[str]
    """
    if method == "lda":
        vectorizer = CountVectorizer(max_features=max_features)
        doc_term_matrix = vectorizer.fit_transform(cleaned_docs)
        model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=20,
            learning_method="online",
        )
    elif method == "nmf":
        vectorizer = TfidfVectorizer(max_features=max_features)
        doc_term_matrix = vectorizer.fit_transform(cleaned_docs)
        model = NMF(
            n_components=n_topics,
            random_state=42,
            max_iter=300,
        )
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'lda' or 'nmf'.")

    doc_topic_matrix = model.fit_transform(doc_term_matrix)

    feature_names = vectorizer.get_feature_names_out()
    topics = _build_topic_list(model, feature_names, n_top_words)
    doc_topic_assignments = np.argmax(doc_topic_matrix, axis=1).tolist()

    return {
        "topics": topics,
        "doc_topic_matrix": doc_topic_matrix,
        "doc_topic_assignments": doc_topic_assignments,
        "feature_names": feature_names.tolist(),
    }


def get_topic_label(keywords: list[str], max_words: int = 3) -> str:
    """Generate a short human-readable label from the top keywords."""
    return " / ".join(keywords[:max_words]).title()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_topic_list(model, feature_names, n_top_words: int) -> list[dict]:
    """Build a list of topic dicts from a fitted model."""
    topics = []
    for idx, component in enumerate(model.components_):
        top_indices = component.argsort()[::-1][:n_top_words]
        keywords = [feature_names[i] for i in top_indices]
        topics.append(
            {
                "topic_id": idx,
                "keywords": keywords,
                "label": get_topic_label(keywords),
            }
        )
    return topics
