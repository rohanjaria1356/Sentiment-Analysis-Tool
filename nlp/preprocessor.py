"""
preprocessor.py — Text cleaning, tokenization, and lemmatization.

Handles all text preprocessing before topic modelling and sentiment analysis.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer


# ---------------------------------------------------------------------------
# Ensure required NLTK resources are available
# ---------------------------------------------------------------------------
def _download_nltk_data():
    """Download NLTK data files if they are not already present."""
    resources = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "omw-1.4",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
    ]
    for res in resources:
        try:
            nltk.data.find(f"tokenizers/{res}" if "punkt" in res else res)
        except LookupError:
            nltk.download(res, quiet=True)


_download_nltk_data()

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# Common contractions → expansion map
CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'ve": " have",
    "'m": " am",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Lower-case, expand contractions, strip URLs / special chars."""
    text = text.lower()

    # Expand contractions
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove digits
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_and_lemmatize(text: str) -> list[str]:
    """Tokenize → remove stopwords → lemmatize."""
    tokens = word_tokenize(text)
    tokens = [
        LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in STOP_WORDS and len(tok) > 2
    ]
    return tokens


def split_into_sentences(text: str) -> list[str]:
    """Split raw text into individual sentences."""
    return sent_tokenize(text)


def prepare_corpus(documents: list[str]) -> tuple[list[list[str]], list[str]]:
    """
    Full preprocessing pipeline for a list of documents.

    Returns
    -------
    token_lists : list[list[str]]
        Per-document token lists (for inspection / word-clouds).
    cleaned_docs : list[str]
        Joined strings ready for TF-IDF vectorization.
    """
    token_lists = []
    cleaned_docs = []
    for doc in documents:
        cleaned = clean_text(doc)
        tokens = tokenize_and_lemmatize(cleaned)
        token_lists.append(tokens)
        cleaned_docs.append(" ".join(tokens))
    return token_lists, cleaned_docs
