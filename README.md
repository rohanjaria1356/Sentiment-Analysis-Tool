#  TopicSenti — Topic-Sentiment Analysis Tool

> **An NLP-powered command-line tool** that extracts hidden topics from text and analyses the sentiment associated with each topic. Built as a BYOP capstone project for CSA2001 (AI & Machine Learning).

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.8+-154F5B?logo=python&logoColor=white)

---

##  Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage (CLI)](#usage-cli)
- [Usage (Web UI — Bonus)](#usage-web-ui--bonus)
- [How It Works](#how-it-works)
- [Sample Data](#sample-data)
- [Running Tests](#running-tests)

---

## Overview

People generate enormous volumes of text — product reviews, survey responses, news articles, social media posts. Manually reading through all of it to understand **what people are talking about** and **how they feel about it** is impractical.

**TopicSenti** solves this by combining two powerful NLP techniques:

1. **Topic Modelling** (LDA / NMF) — discovers hidden thematic clusters
2. **Sentiment Analysis** (VADER + TextBlob) — scores the emotional tone of each document

The **primary interface is a command-line tool (CLI)** that provides rich, coloured terminal output with formatted tables, sentiment bars, and topic summaries. A **bonus web dashboard** (Flask + Plotly.js) is also included.

---

## Features

-  **CLI-first design** — beautiful terminal output with Rich formatting
-  **Interactive mode** — paste text directly, or provide via file
-  **File support** — analyse CSV or TXT files from the command line
-  **Topic extraction** using LDA or NMF algorithms
-  **Dual sentiment analysis** with VADER and TextBlob (consensus scoring)
-  **Coloured output** — sentiment-coded tables, bars, and summary panels
-  **Configurable** number of topics (2–15) and algorithm selection
-  **Bonus web UI** — Flask dashboard with interactive Plotly.js charts
-  **No GPU required** — runs on any machine with Python 3.10+

## Tech Stack

| Layer              | Technology                | Purpose                              |
| ------------------ | ------------------------- | ------------------------------------ |
| Language           | Python 3.10+              | Core language                        |
| CLI Framework      | Rich                      | Coloured terminal output & tables    |
| Text Preprocessing | NLTK                      | Tokenization, stopwords, lemmatisation |
| Topic Modelling    | scikit-learn (LDA & NMF)  | Latent topic extraction              |
| Sentiment Analysis | VADER (NLTK) + TextBlob   | Lexicon-based sentiment scoring      |
| Web UI (bonus)     | Flask + Plotly.js          | Interactive browser dashboard        |
| Testing            | pytest                    | Unit tests                           |

## Project Structure

```
aiml_project/
├── cli.py                     # CLI entry point (primary interface)
├── app.py                     # Flask web UI (bonus interface)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── .gitignore
│
├── nlp/                       # NLP engine (core logic)
│   ├── __init__.py
│   ├── preprocessor.py        # Text cleaning & tokenisation
│   ├── topic_model.py         # LDA/NMF topic extraction
│   └── sentiment.py           # VADER + TextBlob sentiment
│
├── static/                    # Web UI frontend assets
│   ├── css/style.css
│   └── js/main.js
│
├── templates/                 # Web UI Jinja2 templates
│   ├── base.html
│   ├── index.html
│   └── results.html
│
├── sample_data/               # Example datasets
│   └── sample_reviews.csv     # 50 sample reviews for demo
│
├── tests/                     # Unit tests
│   ├── test_preprocessor.py
│   ├── test_topic_model.py
│   └── test_sentiment.py
│
└── report/                    # Project report
    ├── project_report.md
    └── project_report.pdf
```


## Setup & Installation

### Prerequisites

- **Python 3.10 or higher** — [Download Python](https://www.python.org/downloads/)
- **pip** (comes with Python)

### Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/topic-sentiment-analysis.git
   cd topic-sentiment-analysis
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

> **Note:** On first run, NLTK will automatically download required data files (~50 MB). This is a one-time download.

---

## Usage (CLI)

The CLI is the **primary interface**. It supports three modes:

### Interactive Mode — Paste Text

```bash
python cli.py
```

You'll be prompted to paste text. Press **Enter** on an empty line when done.

### File Mode — Analyse a CSV or TXT File

```bash
# Analyse a CSV file (auto-detects text column)
python cli.py -f sample_data/sample_reviews.csv

# Analyse a TXT file
python cli.py -f mydata.txt

# Specify number of topics and algorithm
python cli.py -f sample_data/sample_reviews.csv -n 5 -m lda
```

### Inline Text Mode

```bash
python cli.py -t "Great product! I love it. Terrible service though. The food was okay."
```

### All CLI Options

```
usage: TopicSenti CLI [-h] [-f FILE] [-t TEXT] [-n TOPICS] [-m {lda,nmf}]

Options:
  -f, --file FILE        Path to a .csv or .txt file to analyse
  -t, --text TEXT        Text to analyse (enclose in quotes)
  -n, --topics TOPICS    Number of topics to extract (default: 5)
  -m, --method {lda,nmf} Topic modelling algorithm (default: lda)
```

### CLI Output

The CLI displays:
- **Sentiment Overview** — counts & percentages (positive/neutral/negative), average scores
- **Discovered Topics** — keywords, document counts, sentiment bars per topic
- **Document Details** — formatted table with topic assignment & sentiment per document
- **Sentiment Bar Chart** — ASCII bar chart of topic sentiment scores

## Usage (Web UI — Bonus)

A bonus Flask web dashboard is also available:

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser. The web UI provides:
- Upload/paste form
- Interactive Plotly.js charts (donut, heatmap, bar charts)
- Dark-themed dashboard with glassmorphism design

## How It Works

```
Input Text → Preprocessing → Topic Modelling → Sentiment Analysis → Results
```

1. **Preprocessing** — lowercase, remove URLs/HTML/digits/punctuation, expand contractions, tokenise, remove stopwords, lemmatise
2. **Topic Modelling** — TF-IDF or Count vectorisation → LDA or NMF to discover `n` latent topics
3. **Sentiment Analysis** — VADER (rule-based, compound score) + TextBlob (pattern-based, polarity/subjectivity) with consensus logic
4. **Output** — Results displayed in CLI (Rich tables & panels) or Web UI (Plotly.js charts)

## Sample Data

A sample CSV with 50 reviews is included at `sample_data/sample_reviews.csv`. Quick test:

```bash
python cli.py -f sample_data/sample_reviews.csv -n 5 -m lda
```

## Running Tests

```bash
# From the project root
pytest tests/ -v
```

## Author

**Rohan Jaria**

---
