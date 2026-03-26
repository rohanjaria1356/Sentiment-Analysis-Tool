"""
generate_report_pdf.py — Convert the project report Markdown to a styled PDF.

Uses fpdf2 to produce a professional-looking PDF with proper formatting,
headings, tables, and code blocks.
"""

import re
from fpdf import FPDF


class ReportPDF(FPDF):
    """Custom PDF with header/footer and styled sections."""

    DARK_BLUE = (30, 58, 138)
    ACCENT = (99, 102, 241)
    TEXT_COLOR = (30, 30, 30)
    MUTED = (100, 100, 100)
    TABLE_HEADER_BG = (230, 235, 245)
    TABLE_ROW_ALT = (245, 247, 250)
    CODE_BG = (240, 242, 245)

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=25)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*self.MUTED)
        self.cell(0, 8, "TopicSenti - Project Report", align="L")
        self.cell(0, 8, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*self.ACCENT)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*self.MUTED)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    # ----- Title page -----
    def title_page(self):
        self.add_page()
        self.ln(50)
        # Title
        self.set_font("Helvetica", "B", 28)
        self.set_text_color(*self.DARK_BLUE)
        self.multi_cell(0, 14, "Topic-Sentiment\nAnalysis Tool", align="C")
        self.ln(6)

        # Subtitle
        self.set_font("Helvetica", "", 14)
        self.set_text_color(*self.ACCENT)
        self.cell(0, 10, "Project Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(6)

        # Divider
        self.set_draw_color(*self.ACCENT)
        self.set_line_width(0.8)
        self.line(60, self.get_y(), 150, self.get_y())
        self.ln(12)

        # Meta info
        self.set_font("Helvetica", "", 11)
        self.set_text_color(*self.TEXT_COLOR)
        meta_lines = [
            "Course: CSA2001 - Artificial Intelligence & Machine Learning",
            "Project Type: Bring Your Own Project (BYOP)",
            "Author: Rohan",
            "Date: March 2026",
        ]
        for line in meta_lines:
            self.cell(0, 8, line, align="C", new_x="LMARGIN", new_y="NEXT")

    # ----- Section heading -----
    def section_heading(self, number, title):
        self.ln(6)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*self.DARK_BLUE)
        self.cell(0, 10, f"{number}. {title}", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*self.ACCENT)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 80, self.get_y())
        self.ln(4)
        self.set_text_color(*self.TEXT_COLOR)

    # ----- Sub heading -----
    def sub_heading(self, title):
        self.ln(3)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*self.ACCENT)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*self.TEXT_COLOR)
        self.ln(1)

    # ----- Body text -----
    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*self.TEXT_COLOR)
        # Handle bold markers
        text = self._clean_md(text)
        self.multi_cell(0, 6, text)
        self.ln(2)

    # ----- Bullet list -----
    def bullet_list(self, items):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*self.TEXT_COLOR)
        for item in items:
            item = self._clean_md(item)
            self.cell(8)
            self.cell(4, 6, "-")  # bullet
            self.multi_cell(0, 6, f" {item}")
            self.ln(1)
        self.ln(2)

    # ----- Numbered list -----
    def numbered_list(self, items):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*self.TEXT_COLOR)
        for i, item in enumerate(items, 1):
            item = self._clean_md(item)
            self.cell(8)
            self.cell(8, 6, f"{i}.")
            self.multi_cell(0, 6, f" {item}")
            self.ln(1)
        self.ln(2)

    # ----- Table -----
    def styled_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            n = len(headers)
            col_widths = [190 / n] * n

        # Header
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(*self.TABLE_HEADER_BG)
        self.set_text_color(*self.DARK_BLUE)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 8, h, border=1, fill=True, align="C")
        self.ln()

        # Rows
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*self.TEXT_COLOR)
        for r_idx, row in enumerate(rows):
            if r_idx % 2 == 1:
                self.set_fill_color(*self.TABLE_ROW_ALT)
                fill = True
            else:
                fill = False
            for i, cell in enumerate(row):
                cell = self._clean_md(cell)
                self.cell(col_widths[i], 7, cell, border=1, fill=fill)
            self.ln()
        self.ln(4)

    # ----- Code block -----
    def code_block(self, code):
        self.set_font("Courier", "", 8)
        self.set_fill_color(*self.CODE_BG)
        self.set_text_color(50, 50, 50)
        for line in code.strip().split("\n"):
            self.cell(0, 5, "  " + line, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(4)
        self.set_text_color(*self.TEXT_COLOR)

    @staticmethod
    def _clean_md(text):
        """Remove markdown formatting like ** and ` for PDF rendering."""
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'`(.*?)`', r'\1', text)
        text = text.replace("\u2014", "-").replace("\u2192", "->")
        text = text.replace("\u2013", "-").replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("\u201c", "\"").replace("\u201d", "\"")
        # Replace unicode chars that Helvetica can't handle
        text = text.encode('latin-1', errors='replace').decode('latin-1')
        return text


def build_report():
    pdf = ReportPDF()
    pdf.alias_nb_pages()

    # ===== TITLE PAGE =====
    pdf.title_page()

    # ===== TABLE OF CONTENTS =====
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*pdf.DARK_BLUE)
    pdf.cell(0, 12, "Table of Contents", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    toc_items = [
        ("1", "Abstract"),
        ("2", "Introduction"),
        ("3", "Problem Statement"),
        ("4", "Literature Survey"),
        ("5", "Methodology"),
        ("6", "System Architecture"),
        ("7", "Implementation Details"),
        ("8", "Results & Analysis"),
        ("9", "Challenges & Learnings"),
        ("10", "Future Scope"),
        ("11", "Conclusion"),
        ("12", "References"),
    ]

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*pdf.TEXT_COLOR)
    for num, title in toc_items:
        pdf.cell(12, 8, num + ".")
        pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")

    # ===== 1. ABSTRACT =====
    pdf.add_page()
    pdf.section_heading("1", "Abstract")
    pdf.body_text(
        "This project presents TopicSenti, a command-line Topic-Sentiment Analysis Tool that combines "
        "unsupervised topic modelling with lexicon-based sentiment analysis to extract actionable insights "
        "from unstructured text. The tool accepts user-supplied text (pasted interactively or provided as "
        "CSV/TXT files via command-line arguments) and automatically discovers latent topics using Latent "
        "Dirichlet Allocation (LDA) or Non-negative Matrix Factorization (NMF), then measures the emotional "
        "tone of each topic using VADER and TextBlob sentiment analysers. Results are presented through a "
        "richly formatted terminal interface featuring colour-coded sentiment panels, topic keyword displays, "
        "document detail tables, and ASCII sentiment bar charts. A bonus web dashboard is also included. "
        "The system is built with Python, NLTK, scikit-learn, and Rich, requiring no GPU."
    )
    pdf.body_text("Keywords: Natural Language Processing, Topic Modelling, Sentiment Analysis, LDA, NMF, VADER, TextBlob, CLI, Command-Line Interface")

    # ===== 2. INTRODUCTION =====
    pdf.section_heading("2", "Introduction")
    pdf.body_text(
        "The volume of text data generated daily - from product reviews and social media posts to survey "
        "responses and news articles - has grown exponentially. Organisations and individuals need tools to "
        "efficiently extract meaning from this data without reading every document manually."
    )
    pdf.body_text("Two fundamental questions arise when analysing a corpus of text:")
    pdf.numbered_list([
        "What are people talking about? - addressed by Topic Modelling",
        "How do they feel about it? - addressed by Sentiment Analysis",
    ])
    pdf.body_text(
        "While these techniques have been studied independently, integrating them into a single, user-friendly "
        "tool creates significant practical value. A product manager, for example, can quickly identify that "
        "customers discussing 'battery life' are mostly positive, while those discussing 'customer service' are "
        "overwhelmingly negative - enabling targeted improvements."
    )
    pdf.body_text(
        "This project builds such a tool, applying concepts learned in the CSA2001 course including text "
        "preprocessing, feature extraction, unsupervised learning, and classification."
    )

    # ===== 3. PROBLEM STATEMENT =====
    pdf.section_heading("3", "Problem Statement")
    pdf.body_text(
        "Problem: Manually analysing large collections of text to understand thematic content and associated "
        "sentiments is time-consuming, subjective, and impractical at scale."
    )
    pdf.sub_heading("Proposed Solution")
    pdf.body_text("A command-line tool that:")
    pdf.bullet_list([
        "Accepts text input interactively (paste) or from files (CSV/TXT)",
        "Automatically extracts topics using LDA or NMF",
        "Analyses sentiment per document using VADER and TextBlob",
        "Aggregates sentiment at the topic level",
        "Presents results through richly formatted terminal output (tables, panels, charts)",
        "Also provides a bonus web dashboard for visual exploration",
    ])
    pdf.sub_heading("Real-World Applications")
    pdf.bullet_list([
        "Product teams analysing customer reviews to identify pain points",
        "Researchers exploring themes in interview transcripts or survey data",
        "Content creators understanding audience sentiment across different topics",
        "Businesses monitoring brand perception across feedback channels",
    ])

    # ===== 4. LITERATURE SURVEY =====
    pdf.section_heading("4", "Literature Survey")

    pdf.sub_heading("4.1 Topic Modelling")
    pdf.body_text(
        "Topic modelling is an unsupervised machine learning technique for discovering abstract 'topics' in "
        "a collection of documents."
    )
    pdf.body_text(
        "Latent Dirichlet Allocation (LDA) [Blei et al., 2003] is a generative probabilistic model that assumes "
        "each document is a mixture of topics, and each topic is a distribution over words. LDA uses Dirichlet "
        "priors and iterative inference to discover these latent structures."
    )
    pdf.body_text(
        "Non-negative Matrix Factorization (NMF) [Lee & Seung, 1999] is a linear algebra approach that factorises "
        "the document-term matrix into two non-negative matrices. NMF often produces more coherent topics than LDA "
        "on shorter texts and is deterministically faster to compute."
    )

    pdf.sub_heading("4.2 Sentiment Analysis")
    pdf.body_text(
        "VADER (Valence Aware Dictionary and sEntiment Reasoner) [Hutto & Gilbert, 2014] is a rule-based sentiment "
        "analysis tool specifically designed for social media text. It uses a hand-crafted lexicon and grammatical "
        "rules to produce a compound polarity score between -1 and +1."
    )
    pdf.body_text(
        "TextBlob [Loria, 2018] provides pattern-based sentiment analysis using a built-in lexicon. It returns both "
        "polarity (-1 to +1) and subjectivity (0 to 1) scores."
    )

    pdf.sub_heading("4.3 Integrated Approaches")
    pdf.body_text(
        "Several studies have combined topic modelling with sentiment analysis. Lin & He (2009) proposed the Joint "
        "Sentiment-Topic (JST) model. Our approach follows a simpler sequential integration - first discovering "
        "topics, then overlaying sentiment - which provides comparable insight with significantly lower computational "
        "cost and easier interpretability."
    )

    # ===== 5. METHODOLOGY =====
    pdf.add_page()
    pdf.section_heading("5", "Methodology")

    pdf.sub_heading("5.1 Overall Pipeline")
    pdf.code_block("Raw Text -> Preprocessing -> Topic Modelling -> Sentiment Analysis -> Visualisation")

    pdf.sub_heading("5.2 Text Preprocessing")
    pdf.numbered_list([
        "Convert to lowercase",
        "Expand common contractions (can't -> cannot, won't -> will not)",
        "Remove URLs, HTML tags, digits, and punctuation",
        "Tokenise using NLTK's word_tokenize",
        "Remove English stop words",
        "Lemmatise using NLTK's WordNetLemmatizer",
        "Filter tokens shorter than 3 characters",
    ])

    pdf.sub_heading("5.3 Topic Extraction")
    pdf.body_text("Two algorithms are offered:")
    pdf.body_text(
        "LDA Path: Build a Count (bag-of-words) matrix from the cleaned corpus, fit scikit-learn's "
        "LatentDirichletAllocation with user-specified number of topics, extract top keywords per topic "
        "and assign each document to its dominant topic."
    )
    pdf.body_text(
        "NMF Path: Build a TF-IDF matrix from the cleaned corpus, fit scikit-learn's NMF decomposition, "
        "same extraction and assignment logic."
    )

    pdf.sub_heading("5.4 Sentiment Analysis")
    pdf.body_text("Each document (original, uncleaned text) is scored by both analysers:")
    pdf.numbered_list([
        "VADER - produces compound, positive, negative, and neutral scores",
        "TextBlob - produces polarity and subjectivity scores",
    ])
    pdf.body_text(
        "A consensus label is computed: if both agree, use that label; if they disagree, use the label "
        "from the analyser with the stronger signal (higher absolute score)."
    )

    # ===== 6. SYSTEM ARCHITECTURE =====
    pdf.section_heading("6", "System Architecture")
    pdf.code_block(
        "Terminal / Browser (User)\n"
        "  |\n"
        "  +-- CLI (cli.py) - Primary CUI\n"
        "  +-- Web UI (app.py) - Bonus\n"
        "  |\n"
        "  v\n"
        "NLP Engine (Shared Core Logic)\n"
        "  |\n"
        "  +-- preprocessor (NLTK)\n"
        "  +-- topic_model (sklearn LDA/NMF)\n"
        "  +-- sentiment (VADER + TextBlob)"
    )
    pdf.body_text(
        "The architecture separates the NLP engine (shared core) from the interface layer "
        "(CLI or Web). The CLI (cli.py) is the primary CUI-based interface, while the Flask "
        "web UI (app.py) serves as a bonus visual dashboard."
    )

    pdf.sub_heading("Technology Justification")
    pdf.styled_table(
        ["Choice", "Rationale"],
        [
            ["Rich (CLI framework)", "Beautiful terminal output, tables, colours"],
            ["VADER + TextBlob", "No GPU required; fast; interpretable"],
            ["LDA + NMF", "Standard algorithms covered in course"],
            ["Flask (bonus web UI)", "Lightweight; quick visual exploration"],
            ["Plotly.js (bonus)", "Interactive browser charts"],
        ],
        [60, 130],
    )

    # ===== 7. IMPLEMENTATION DETAILS =====
    pdf.add_page()
    pdf.section_heading("7", "Implementation Details")

    pdf.sub_heading("7.1 Preprocessing Module (nlp/preprocessor.py)")
    pdf.body_text("Key functions:")
    pdf.bullet_list([
        "clean_text(text) - normalisation pipeline (lowercase, contractions, URL removal, etc.)",
        "tokenize_and_lemmatize(text) - NLTK tokenisation + stopword removal + lemmatisation",
        "prepare_corpus(documents) - batch processing returning both token lists and joined strings",
    ])
    pdf.body_text("NLTK data files are auto-downloaded on first import, ensuring zero manual setup.")

    pdf.sub_heading("7.2 Topic Modelling Module (nlp/topic_model.py)")
    pdf.body_text("Key functions:")
    pdf.bullet_list([
        "extract_topics(cleaned_docs, n_topics, method) - fits LDA or NMF and returns topics, document-topic matrix, and per-document assignments",
        "get_topic_label(keywords) - generates a human-readable label from top keywords",
    ])
    pdf.body_text("The module uses a maximum vocabulary of 5,000 features to balance expressiveness with performance.")

    pdf.sub_heading("7.3 Sentiment Module (nlp/sentiment.py)")
    pdf.body_text("Key functions:")
    pdf.bullet_list([
        "analyze_sentiment_vader(text) - VADER compound + breakdown scores",
        "analyze_sentiment_textblob(text) - TextBlob polarity + subjectivity",
        "analyze_document(text) - combined analysis with consensus logic",
        "aggregate_sentiment(doc_sentiments) - summary statistics across documents",
    ])

    pdf.sub_heading("7.4 CLI Interface (cli.py)")
    pdf.body_text("The primary CUI-based interface built with the Rich library provides:")
    pdf.bullet_list([
        "Interactive mode - paste text directly into the terminal",
        "File mode - analyse CSV or TXT files via -f flag",
        "Inline mode - pass text via -t flag",
        "Configurable options - -n for topic count, -m for algorithm (lda/nmf)",
        "Rich formatted output - colour-coded sentiment panels, topic keyword displays, document detail tables, ASCII sentiment bar charts",
    ])

    pdf.sub_heading("7.5 Flask Web UI (app.py) - Bonus")
    pdf.body_text("A bonus web dashboard that orchestrates the same NLP pipeline via HTTP:")
    pdf.bullet_list([
        "GET / - serves the upload/paste form",
        "POST /analyze - renders results with interactive Plotly.js charts",
    ])

    # ===== 8. RESULTS & ANALYSIS =====
    pdf.add_page()
    pdf.section_heading("8", "Results & Analysis")

    pdf.sub_heading("8.1 Testing with Sample Data")
    pdf.body_text(
        "The tool was tested with a dataset of 50 sample reviews spanning multiple domains (electronics, "
        "restaurants, hotels, movies, software, courses). With 5 topics and LDA:"
    )
    pdf.bullet_list([
        "The algorithm successfully grouped related reviews (e.g., food/restaurant reviews clustered together, tech product reviews formed separate topics)",
        "Sentiment distribution closely matched manual annotation (positive reviews correctly classified as positive, negative as negative)",
        "The consensus mechanism between VADER and TextBlob reduced misclassification compared to using either alone",
    ])

    pdf.sub_heading("8.2 Algorithm Comparison")
    pdf.styled_table(
        ["Aspect", "LDA", "NMF"],
        [
            ["Topic coherence", "Good for longer docs", "Better for shorter texts"],
            ["Speed", "Slightly slower", "Faster (matrix factorisation)"],
            ["Interpretability", "Probabilistic (soft)", "Deterministic (cleaner)"],
        ],
        [50, 70, 70],
    )

    pdf.sub_heading("8.3 Unit Tests")
    pdf.body_text(
        "All 25+ unit tests pass, covering text preprocessing (cleaning, tokenisation, lemmatisation), "
        "topic extraction (LDA, NMF, parameter validation), and sentiment analysis (positive/negative/neutral "
        "classification, aggregation)."
    )

    # ===== 9. CHALLENGES & LEARNINGS =====
    pdf.section_heading("9", "Challenges & Learnings")

    pdf.sub_heading("9.1 Challenges Faced")
    pdf.numbered_list([
        "Short text topic modelling - Very short documents produce sparse vectors. Mitigation: sentence aggregation and minimum document thresholds.",
        "Sentiment of neutral text - Factual text can get slight polarity from TextBlob. Mitigation: consensus mechanism with strength-based tiebreaking.",
        "CSV format variability - Users upload CSVs with different column names. Mitigation: auto-detection logic.",
        "NLTK data management - Ensuring resources are available without manual downloads. Mitigation: auto-download on module import.",
    ])

    pdf.sub_heading("9.2 Key Learnings")
    pdf.bullet_list([
        "End-to-end NLP pipelines require careful attention to text preprocessing",
        "Unsupervised learning (LDA/NMF) requires thoughtful parameter tuning",
        "Multiple sentiment sources provide more robust classifications",
        "Web application design involves balancing backend computation with frontend responsiveness",
    ])

    # ===== 10. FUTURE SCOPE =====
    pdf.section_heading("10", "Future Scope")
    pdf.numbered_list([
        "Transformer-based models - Integrate BERT or RoBERTa for more accurate sentiment analysis",
        "BERTopic - Use transformer embeddings + HDBSCAN for more coherent topic extraction",
        "Real-time data sources - Add API integration for Twitter/Reddit",
        "Multi-language support - Extend preprocessing beyond English",
        "Historical analysis - Store results in a database to track sentiment trends over time",
        "Export functionality - Allow users to download results as CSV or PDF reports",
    ])

    # ===== 11. CONCLUSION =====
    pdf.section_heading("11", "Conclusion")
    pdf.body_text(
        "This project successfully demonstrates the integration of topic modelling and sentiment analysis into "
        "a practical, user-friendly web application. By combining LDA/NMF for topic extraction with VADER and "
        "TextBlob for sentiment scoring, TopicSenti enables users to quickly understand both what is being discussed "
        "and how people feel about it."
    )
    pdf.body_text(
        "The tool is lightweight (no GPU required), well-documented, and easily extensible. It meaningfully applies "
        "multiple AI/ML concepts from the CSA2001 course - text preprocessing, feature extraction, unsupervised "
        "learning, and classification - to solve a real-world information overload problem."
    )

    # ===== 12. REFERENCES =====
    pdf.add_page()
    pdf.section_heading("12", "References")
    refs = [
        "Blei, D.M., Ng, A.Y., & Jordan, M.I. (2003). Latent Dirichlet Allocation. Journal of Machine Learning Research, 3, 993-1022.",
        "Lee, D.D., & Seung, H.S. (1999). Learning the parts of objects by non-negative matrix factorization. Nature, 401(6755), 788-791.",
        "Hutto, C.J., & Gilbert, E. (2014). VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text. Proceedings of AAAI ICWSM.",
        "Loria, S. (2018). TextBlob: Simplified Text Processing. https://textblob.readthedocs.io/",
        "Lin, C., & He, Y. (2009). Joint Sentiment/Topic Model for Sentiment Analysis. Proceedings of ACM CIKM.",
        "Manning, C.D., Raghavan, P., & Schutze, H. (2008). Introduction to Information Retrieval. Cambridge University Press.",
        "Bird, S., Klein, E., & Loper, E. (2009). Natural Language Processing with Python. O'Reilly Media.",
        "Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825-2830.",
    ]
    pdf.numbered_list(refs)

    # ===== SAVE =====
    output_path = "report/project_report.pdf"
    pdf.output(output_path)
    print(f"PDF report generated: {output_path}")


if __name__ == "__main__":
    build_report()
