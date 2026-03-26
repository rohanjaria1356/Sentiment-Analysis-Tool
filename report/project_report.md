# Topic-Sentiment Analysis Tool — Project Report

---

**Course:** CSA2001 — Artificial Intelligence & Machine Learning  
**Project Type:** Bring Your Own Project (BYOP)  
**Author:** Rohan  
**Date:** March 2026

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Problem Statement](#3-problem-statement)
4. [Literature Survey](#4-literature-survey)
5. [Methodology](#5-methodology)
6. [System Architecture](#6-system-architecture)
7. [Implementation Details](#7-implementation-details)
8. [Results & Analysis](#8-results--analysis)
9. [Challenges & Learnings](#9-challenges--learnings)
10. [Future Scope](#10-future-scope)
11. [Conclusion](#11-conclusion)
12. [References](#12-references)

---

## 1. Abstract

This project presents **TopicSenti**, a command-line Topic-Sentiment Analysis Tool that combines unsupervised topic modelling with lexicon-based sentiment analysis to extract actionable insights from unstructured text. The tool accepts user-supplied text (pasted interactively or provided as CSV/TXT files via command-line arguments) and automatically discovers latent topics using Latent Dirichlet Allocation (LDA) or Non-negative Matrix Factorization (NMF), then measures the emotional tone of each topic using VADER and TextBlob sentiment analysers. Results are presented through a richly formatted terminal interface featuring colour-coded sentiment panels, topic keyword displays, document detail tables, and ASCII sentiment bar charts. A bonus web dashboard (Flask + Plotly.js) is also included. The system is built with Python, NLTK, scikit-learn, and Rich, requiring no GPU and running entirely on commodity hardware.

**Keywords:** Natural Language Processing, Topic Modelling, Sentiment Analysis, LDA, NMF, VADER, TextBlob, CLI, Command-Line Interface

---

## 2. Introduction

The volume of text data generated daily — from product reviews and social media posts to survey responses and news articles — has grown exponentially. Organisations and individuals need tools to efficiently extract meaning from this data without reading every document manually.

Two fundamental questions arise when analysing a corpus of text:
1. **What are people talking about?** — addressed by Topic Modelling
2. **How do they feel about it?** — addressed by Sentiment Analysis

While these techniques have been studied independently, integrating them into a single, user-friendly tool creates significant practical value. A product manager, for example, can quickly identify that customers discussing "battery life" are mostly positive, while those discussing "customer service" are overwhelmingly negative — enabling targeted improvements.

This project builds such a tool, applying concepts learned in the CSA2001 course including text preprocessing, feature extraction, unsupervised learning, and classification.

---

## 3. Problem Statement

**Problem:** Manually analysing large collections of text to understand thematic content and associated sentiments is time-consuming, subjective, and impractical at scale.

**Proposed Solution:** A command-line tool that:
- Accepts text input interactively (paste) or from files (CSV/TXT)
- Automatically extracts topics using LDA or NMF
- Analyses sentiment per document using VADER and TextBlob
- Aggregates sentiment at the topic level
- Presents results through richly formatted terminal output (tables, panels, charts)
- Also provides a bonus web dashboard for visual exploration

**Real-World Relevance:** This tool can be applied to:
- **Product teams** analysing customer reviews to identify pain points
- **Researchers** exploring themes in interview transcripts or survey data
- **Content creators** understanding audience sentiment across different topics
- **Businesses** monitoring brand perception across feedback channels

---

## 4. Literature Survey

### 4.1 Topic Modelling

Topic modelling is an unsupervised machine learning technique for discovering abstract "topics" in a collection of documents.

**Latent Dirichlet Allocation (LDA)** [Blei et al., 2003] is a generative probabilistic model that assumes each document is a mixture of topics, and each topic is a distribution over words. LDA uses Dirichlet priors and iterative inference to discover these latent structures. It remains one of the most widely used topic models due to its interpretability and effectiveness.

**Non-negative Matrix Factorization (NMF)** [Lee & Seung, 1999] is a linear algebra approach that factorises the document-term matrix into two non-negative matrices — one representing document-topic associations and the other topic-term associations. NMF often produces more coherent topics than LDA on shorter texts and is deterministically faster to compute.

### 4.2 Sentiment Analysis

Sentiment analysis determines the emotional tone of text, typically classifying it as positive, negative, or neutral.

**VADER (Valence Aware Dictionary and sEntiment Reasoner)** [Hutto & Gilbert, 2014] is a rule-based sentiment analysis tool specifically designed for social media text. It uses a hand-crafted lexicon and grammatical rules (handling negation, degree modifiers, punctuation emphasis) to produce a compound polarity score between −1 and +1.

**TextBlob** [Loria, 2018] provides pattern-based sentiment analysis using a built-in lexicon. It returns both polarity (−1 to +1) and subjectivity (0 to 1) scores, offering an additional dimension of analysis.

### 4.3 Integrated Approaches

Several studies have combined topic modelling with sentiment analysis. Lin & He (2009) proposed the Joint Sentiment-Topic (JST) model, a probabilistic model that jointly captures sentiment and topic. Our approach follows a simpler sequential integration — first discovering topics, then overlaying sentiment — which provides comparable insight with significantly lower computational cost and easier interpretability.

### 4.4 Text Preprocessing

Effective NLP pipelines require thorough text preprocessing [Manning et al., 2008]:
- **Tokenisation:** splitting text into words or subwords
- **Stop word removal:** filtering out common words (the, is, at) that carry little meaning
- **Lemmatisation:** reducing words to their base form (running → run, better → good)
- **Cleaning:** removing URLs, HTML tags, special characters, and digits

The NLTK library provides robust implementations of all these steps.

---

## 5. Methodology

### 5.1 Overall Pipeline

```
Raw Text → Preprocessing → Topic Modelling → Sentiment Analysis → Visualisation
```

### 5.2 Text Preprocessing

1. Convert to lowercase
2. Expand common contractions (can't → cannot, won't → will not)
3. Remove URLs, HTML tags, digits, and punctuation
4. Tokenise using NLTK's word_tokenize
5. Remove English stop words
6. Lemmatise using NLTK's WordNetLemmatizer
7. Filter tokens shorter than 3 characters

### 5.3 Topic Extraction

Two algorithms are offered:

**LDA Path:**
- Build a Count (bag-of-words) matrix from the cleaned corpus
- Fit scikit-learn's LatentDirichletAllocation with user-specified number of topics
- Extract top keywords per topic and assign each document to its dominant topic

**NMF Path:**
- Build a TF-IDF matrix from the cleaned corpus
- Fit scikit-learn's NMF decomposition
- Same extraction and assignment logic

### 5.4 Sentiment Analysis

Each document (original, uncleaned text) is scored by both analysers:

1. **VADER** — produces compound, positive, negative, and neutral scores
2. **TextBlob** — produces polarity and subjectivity scores

A **consensus label** is computed:
- If both VADER and TextBlob agree on the label → use that label
- If they disagree → use the label from the analyser with the stronger signal (higher absolute score)

### 5.5 Aggregation

- Per-topic sentiment is computed by grouping documents by their assigned topic and averaging sentiment scores
- Overall sentiment distribution (positive/neutral/negative counts and percentages) is computed across all documents

---

## 6. System Architecture

```
┌──────────────────────────────────────────────────────┐
│                Terminal / Browser (User)              │
│  ┌────────────────────┐  ┌─────────────────────────┐ │
│  │   CLI (cli.py)     │  │  Web UI (app.py) BONUS  │ │
│  │   Primary CUI      │  │  Flask + Plotly.js      │ │
│  └────────┬───────────┘  └────────┬────────────────┘ │
└───────────┼───────────────────────┼──────────────────┘
            │                       │
            └───────────┬───────────┘
                        ▼
            ┌───────────────────────┐
            │     NLP Engine        │
            │  (Shared Core Logic)  │
            └───────────┬───────────┘
                        │
           ┌────────────┼────────────┐
           ▼            ▼            ▼
      preprocessor   topic_model   sentiment
      (NLTK)         (sklearn)     (VADER+TextBlob)
```

The architecture separates the **NLP engine** (shared core) from the **interface layer** (CLI or Web). The CLI (`cli.py`) is the primary CUI-based interface, while the Flask web UI (`app.py`) serves as a bonus visual dashboard.

### 6.1 Technology Justification

| Choice | Rationale |
|--------|-----------|
| Rich (CLI framework) | Beautiful terminal formatting, colour-coded output, tables |
| VADER + TextBlob (not BERT) | No GPU required; fast; interpretable scores |
| LDA + NMF (not BERTopic) | Standard algorithms covered in course; no heavy dependencies |
| Flask (bonus web UI) | Lightweight; quick to set up for visual exploration |
| Plotly.js (bonus charts) | Interactive browser charts with minimal code |

---

## 7. Implementation Details

### 7.1 Preprocessing Module (`nlp/preprocessor.py`)

Key functions:
- `clean_text(text)` — normalisation pipeline (lowercase, contractions, URL removal, etc.)
- `tokenize_and_lemmatize(text)` — NLTK tokenisation + stopword removal + lemmatisation
- `prepare_corpus(documents)` — batch processing returning both token lists and joined strings

NLTK data files are auto-downloaded on first import, ensuring zero manual setup.

### 7.2 Topic Modelling Module (`nlp/topic_model.py`)

Key functions:
- `extract_topics(cleaned_docs, n_topics, method)` — fits LDA or NMF and returns topics, document-topic matrix, and per-document assignments
- `get_topic_label(keywords)` — generates a human-readable label from top keywords

The module uses a maximum vocabulary of 5,000 features to balance expressiveness with performance.

### 7.3 Sentiment Module (`nlp/sentiment.py`)

Key functions:
- `analyze_sentiment_vader(text)` — VADER compound + breakdown scores
- `analyze_sentiment_textblob(text)` — TextBlob polarity + subjectivity
- `analyze_document(text)` — combined analysis with consensus logic
- `aggregate_sentiment(doc_sentiments)` — summary statistics across documents

### 7.4 CLI Interface (`cli.py`)

The primary CUI-based interface built with the Rich library provides:
- **Interactive mode** — paste text directly into the terminal
- **File mode** — analyse CSV or TXT files via `-f` flag  
- **Inline mode** — pass text via `-t` flag
- **Configurable options** — `-n` for topic count, `-m` for algorithm (lda/nmf)
- **Rich formatted output** — colour-coded sentiment panels, topic keyword displays, document detail tables, ASCII sentiment bar charts

### 7.5 Flask Web UI (`app.py`) — Bonus

A bonus web dashboard that orchestrates the same NLP pipeline via HTTP:
- `GET /` — serves the upload/paste form
- `POST /analyze` — renders results with interactive Plotly.js charts

---

## 8. Results & Analysis

### 8.1 Testing with Sample Data

The tool was tested with a dataset of 50 sample reviews spanning multiple domains (electronics, restaurants, hotels, movies, software, courses). With 5 topics and LDA:

- The algorithm successfully grouped related reviews (e.g., food/restaurant reviews clustered together, tech product reviews formed separate topics)
- Sentiment distribution closely matched manual annotation (positive reviews correctly classified as positive, negative as negative)
- The consensus mechanism between VADER and TextBlob reduced misclassification compared to using either alone

### 8.2 Algorithm Comparison

| Aspect | LDA | NMF |
|--------|-----|-----|
| Topic coherence | Good for longer documents | Better for shorter texts |
| Speed | Slightly slower (iterative) | Faster (matrix factorisation) |
| Interpretability | Probabilistic (soft assignments) | Deterministic (cleaner topics) |

### 8.3 Unit Tests

All 25+ unit tests pass, covering:
- Text preprocessing (cleaning, tokenisation, lemmatisation)
- Topic extraction (LDA, NMF, parameter validation)
- Sentiment analysis (positive/negative/neutral classification, aggregation)

---

## 9. Challenges & Learnings

### 9.1 Challenges Faced

1. **Short text topic modelling** — Very short documents (single sentences) produce sparse vectors that make topic modelling unreliable. Mitigation: sentence aggregation into paragraphs when possible; minimum document threshold.

2. **Sentiment of neutral text** — Factual or informational text (e.g., "The meeting is at 3pm") is correctly classified as neutral by VADER but can get slight polarity from TextBlob. Mitigation: consensus mechanism with strength-based tiebreaking.

3. **CSV format variability** — Users upload CSVs with different column names. Mitigation: auto-detection logic that searches for common text column names.

4. **NLTK data management** — Ensuring NLTK resources are available without manual `nltk.download()` calls. Mitigation: auto-download on module import with `quiet=True`.

### 9.2 Key Learnings

- **End-to-end NLP pipelines** require careful attention to text preprocessing; "garbage in, garbage out" is very real
- **Unsupervised learning** (LDA/NMF) requires thoughtful parameter tuning — the number of topics significantly affects result quality
- **Multiple sentiment sources** provide more robust classifications than any single analyser
- **Web application design** involves balancing backend computation time with frontend responsiveness

---

## 10. Future Scope

1. **Transformer-based models** — Integrate BERT or RoBERTa for more accurate sentiment analysis, especially on nuanced or sarcastic text
2. **BERTopic** — Use transformer embeddings + HDBSCAN for more coherent topic extraction
3. **Real-time data sources** — Add API integration for Twitter/Reddit to analyse live social media sentiment
4. **Multi-language support** — Extend preprocessing and sentiment analysis beyond English
5. **Historical analysis** — Store results in a database to track sentiment trends over time
6. **Export functionality** — Allow users to download results as CSV or PDF reports

---

## 11. Conclusion

This project successfully demonstrates the integration of topic modelling and sentiment analysis into a practical, user-friendly web application. By combining LDA/NMF for topic extraction with VADER and TextBlob for sentiment scoring, TopicSenti enables users to quickly understand both **what** is being discussed and **how** people feel about it.

The tool is lightweight (no GPU required), well-documented, and easily extensible. It meaningfully applies multiple AI/ML concepts from the CSA2001 course — text preprocessing, feature extraction, unsupervised learning, and classification — to solve a real-world information overload problem.

---

## 12. References

1. Blei, D.M., Ng, A.Y., & Jordan, M.I. (2003). Latent Dirichlet Allocation. *Journal of Machine Learning Research*, 3, 993-1022.

2. Lee, D.D., & Seung, H.S. (1999). Learning the parts of objects by non-negative matrix factorization. *Nature*, 401(6755), 788-791.

3. Hutto, C.J., & Gilbert, E. (2014). VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text. *Proceedings of the AAAI Conference on Weblogs and Social Media*.

4. Loria, S. (2018). TextBlob: Simplified Text Processing. [Online]. Available: https://textblob.readthedocs.io/

5. Lin, C., & He, Y. (2009). Joint Sentiment/Topic Model for Sentiment Analysis. *Proceedings of the 18th ACM Conference on Information and Knowledge Management (CIKM)*.

6. Manning, C.D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.

7. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media.

8. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

---
