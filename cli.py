"""
cli.py — Command-line interface for the Topic-Sentiment Analysis Tool.

Usage:
    python cli.py                          # Interactive mode (paste text)
    python cli.py -f data.csv              # Analyse a CSV file
    python cli.py -f data.txt              # Analyse a TXT file
    python cli.py -f data.csv -n 5 -m lda  # 5 topics using LDA
    python cli.py --help                   # Show all options
"""

import argparse
import csv
import io
import os
import sys
import textwrap

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich import box

from nlp.preprocessor import prepare_corpus, split_into_sentences
from nlp.topic_model import extract_topics
from nlp.sentiment import analyze_document, aggregate_sentiment

# ---------------------------------------------------------------------------
# Rich console
# ---------------------------------------------------------------------------
console = Console()

# Colour palette (matching the web UI)
POSITIVE_STYLE = "bold green"
NEGATIVE_STYLE = "bold red"
NEUTRAL_STYLE = "bold yellow"
ACCENT_STYLE = "bold #6366f1"
MUTED_STYLE = "dim"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Banner
    print_banner()

    # Get documents
    if args.file:
        documents = load_file(args.file)
    elif args.text:
        documents = parse_text_input(args.text)
    else:
        documents = interactive_input()

    if not documents or all(len(d.strip()) == 0 for d in documents):
        console.print("\n[red]Error:[/red] No text provided. Exiting.\n")
        sys.exit(1)

    documents = [d.strip() for d in documents if d.strip()]
    console.print(f"\n[{ACCENT_STYLE}]Loaded {len(documents)} document(s)[/]\n")

    # Clamp parameters
    n_topics = max(2, min(args.topics, 15))
    if len(documents) < n_topics:
        n_topics = max(2, len(documents))
        console.print(f"[yellow]Note:[/yellow] Reduced topics to {n_topics} (not enough documents)\n")

    method = args.method.lower()
    if method not in ("lda", "nmf"):
        console.print(f"[red]Error:[/red] Unknown method '{method}'. Use 'lda' or 'nmf'.\n")
        sys.exit(1)

    # Run NLP Pipeline with progress spinner
    with Progress(
        SpinnerColumn(style="bold #6366f1"),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        # Step 1: Preprocessing
        task = progress.add_task("Preprocessing text...", total=None)
        token_lists, cleaned_docs = prepare_corpus(documents)
        progress.update(task, description="[green]✓[/green] Preprocessing complete")

        non_empty = [d for d in cleaned_docs if d.strip()]
        if len(non_empty) < 2:
            console.print("\n[red]Error:[/red] After cleaning, not enough text remains. Provide more content.\n")
            sys.exit(1)

        # Step 2: Topic extraction
        progress.update(task, description=f"Extracting {n_topics} topics using {method.upper()}...")
        topic_results = extract_topics(cleaned_docs, n_topics=n_topics, method=method)
        progress.update(task, description=f"[green]✓[/green] Extracted {n_topics} topics")

        # Step 3: Sentiment analysis
        progress.update(task, description="Running sentiment analysis (VADER + TextBlob)...")
        doc_sentiments = [analyze_document(doc) for doc in documents]
        progress.update(task, description="[green]✓[/green] Sentiment analysis complete")

        # Step 4: Aggregation
        progress.update(task, description="Aggregating results...")
        overall_sentiment = aggregate_sentiment(doc_sentiments)
        topic_sentiments = compute_topic_sentiments(topic_results, doc_sentiments)
        progress.update(task, description="[green]✓[/green] All processing complete")

    console.print()

    # Display results
    print_overall_sentiment(overall_sentiment)
    print_topics(topic_results, topic_sentiments)
    print_document_table(documents, topic_results, doc_sentiments)
    print_sentiment_bar_chart(topic_results, topic_sentiments)
    print_footer()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        prog="TopicSenti CLI",
        description="Topic-Sentiment Analysis Tool — Discover topics and analyse sentiment from text.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python cli.py                              Interactive mode
              python cli.py -f reviews.csv               Analyse CSV file
              python cli.py -f article.txt -n 3 -m nmf   3 topics, NMF method
              python cli.py -t "Great product! I love it. Terrible service though."
        """),
    )
    parser.add_argument("-f", "--file", type=str, help="Path to a .csv or .txt file to analyse")
    parser.add_argument("-t", "--text", type=str, help="Text to analyse (enclose in quotes)")
    parser.add_argument("-n", "--topics", type=int, default=5, help="Number of topics to extract (default: 5)")
    parser.add_argument("-m", "--method", type=str, default="lda", choices=["lda", "nmf"],
                        help="Topic modelling algorithm: lda or nmf (default: lda)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------
def interactive_input():
    """Prompt user to paste text interactively."""
    console.print(
        Panel(
            "[bold]Paste your text below.[/bold]\n"
            "You can paste multiple lines (reviews, paragraphs, etc.).\n"
            "When done, press [bold cyan]Enter[/bold cyan] on an empty line.",
            title="[bold #6366f1]Text Input[/bold #6366f1]",
            border_style="#6366f1",
        )
    )

    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "" and lines:
            break
        lines.append(line)

    raw = "\n".join(lines)
    return parse_text_input(raw)


def parse_text_input(raw: str) -> list[str]:
    """Split raw text into documents."""
    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
    if len(paragraphs) >= 3:
        return paragraphs

    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    if len(lines) >= 3:
        return lines

    sentences = split_into_sentences(raw)
    if len(sentences) >= 2:
        return sentences

    return [raw]


def load_file(path: str) -> list[str]:
    """Load documents from a CSV or TXT file."""
    if not os.path.isfile(path):
        console.print(f"[red]Error:[/red] File not found: {path}")
        sys.exit(1)

    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()

    if ext == ".csv":
        return parse_csv(raw)
    else:
        return parse_text_input(raw)


def parse_csv(raw: str) -> list[str]:
    """Parse CSV and extract text column."""
    reader = csv.DictReader(io.StringIO(raw))
    fieldnames = reader.fieldnames or []

    text_col = None
    for candidate in ["text", "review", "content", "comment", "message", "body", "Text", "Review", "Content"]:
        if candidate in fieldnames:
            text_col = candidate
            break

    if text_col is None and fieldnames:
        text_col = fieldnames[0]

    if text_col is None:
        console.print("[red]Error:[/red] Could not find a text column in the CSV.")
        sys.exit(1)

    docs = []
    for row in reader:
        val = row.get(text_col, "").strip()
        if val:
            docs.append(val)

    console.print(f"[{MUTED_STYLE}]Using column: '{text_col}' ({len(docs)} rows)[/]")
    return docs


# ---------------------------------------------------------------------------
# Topic sentiment computation
# ---------------------------------------------------------------------------
def compute_topic_sentiments(topic_results: dict, doc_sentiments: list[dict]) -> list[dict]:
    """Aggregate sentiment per topic."""
    n_topics = len(topic_results["topics"])
    assignments = topic_results["doc_topic_assignments"]

    topic_sents = []
    for t in range(n_topics):
        indices = [i for i, a in enumerate(assignments) if a == t]
        if not indices:
            topic_sents.append({"topic_id": t, "count": 0, "avg_compound": 0,
                                "positive": 0, "negative": 0, "neutral": 0})
            continue

        compounds = [doc_sentiments[i]["vader_compound"] for i in indices]
        labels = [doc_sentiments[i]["label"] for i in indices]
        topic_sents.append({
            "topic_id": t,
            "count": len(indices),
            "avg_compound": round(sum(compounds) / len(compounds), 4),
            "positive": sum(1 for l in labels if l == "Positive"),
            "negative": sum(1 for l in labels if l == "Negative"),
            "neutral": sum(1 for l in labels if l == "Neutral"),
        })
    return topic_sents


# ---------------------------------------------------------------------------
# Display functions
# ---------------------------------------------------------------------------
def print_banner():
    banner = Text()
    banner.append("╔══════════════════════════════════════════════════════╗\n", style="#6366f1")
    banner.append("║           ", style="#6366f1")
    banner.append("TopicSenti", style="bold white")
    banner.append("                                ║\n", style="#6366f1")
    banner.append("║     ", style="#6366f1")
    banner.append("Topic-Sentiment Analysis Tool", style="bold #a855f7")
    banner.append("                ║\n", style="#6366f1")
    banner.append("║                                                      ║\n", style="#6366f1")
    banner.append("║  ", style="#6366f1")
    banner.append("Discover topics & sentiment from any text", style="dim white")
    banner.append("         ║\n", style="#6366f1")
    banner.append("╚══════════════════════════════════════════════════════╝", style="#6366f1")
    console.print(banner)


def print_overall_sentiment(overall: dict):
    """Print overall sentiment summary."""
    console.print(Panel(
        f"[bold]Overall Sentiment Summary[/bold]\n\n"
        f"  Documents analysed:  [bold]{overall['count']}[/bold]\n\n"
        f"  [{POSITIVE_STYLE}]Positive:  {overall['positive']:>3}  ({overall['positive_pct']}%)[/]\n"
        f"  [{NEUTRAL_STYLE}]Neutral:   {overall['neutral']:>3}  ({overall['neutral_pct']}%)[/]\n"
        f"  [{NEGATIVE_STYLE}]Negative:  {overall['negative']:>3}  ({overall['negative_pct']}%)[/]\n\n"
        f"  Avg VADER compound:      [bold]{overall['avg_vader_compound']:>8}[/bold]\n"
        f"  Avg TextBlob polarity:   [bold]{overall['avg_textblob_polarity']:>8}[/bold]\n"
        f"  Avg TextBlob subjectivity: [bold]{overall['avg_textblob_subjectivity']:>6}[/bold]",
        title="[bold #6366f1]Sentiment Overview[/bold #6366f1]",
        border_style="#6366f1",
        padding=(1, 2),
    ))
    console.print()


def print_topics(topic_results: dict, topic_sentiments: list[dict]):
    """Print discovered topics with keywords and sentiment."""
    console.print(Panel(
        "[bold]Discovered Topics[/bold]",
        border_style="#6366f1",
        padding=(0, 2),
    ))

    for topic in topic_results["topics"]:
        tid = topic["topic_id"]
        ts = topic_sentiments[tid]

        # Sentiment color for the avg score
        score = ts["avg_compound"]
        if score >= 0.05:
            score_style = POSITIVE_STYLE
        elif score <= -0.05:
            score_style = NEGATIVE_STYLE
        else:
            score_style = NEUTRAL_STYLE

        # Build sentiment bar
        total = ts["count"] if ts["count"] > 0 else 1
        pos_pct = int(ts["positive"] / total * 20)
        neu_pct = int(ts["neutral"] / total * 20)
        neg_pct = 20 - pos_pct - neu_pct
        sent_bar = f"[green]{'█' * pos_pct}[/green][yellow]{'█' * neu_pct}[/yellow][red]{'█' * neg_pct}[/red]"

        keywords = ", ".join(topic["keywords"][:8])

        console.print(
            f"\n  [{ACCENT_STYLE}]Topic {tid + 1}[/] — [bold]{topic['label']}[/bold]\n"
            f"    Keywords: [{MUTED_STYLE}]{keywords}[/]\n"
            f"    Documents: {ts['count']}  |  "
            f"Avg score: [{score_style}]{score:+.4f}[/]\n"
            f"    Sentiment: {sent_bar}  "
            f"[green]+{ts['positive']}[/green] [yellow]~{ts['neutral']}[/yellow] [red]-{ts['negative']}[/red]"
        )

    console.print()


def print_document_table(documents: list[str], topic_results: dict, doc_sentiments: list[dict]):
    """Print per-document detail table."""
    table = Table(
        title="Document Details",
        box=box.ROUNDED,
        border_style="#6366f1",
        header_style="bold #6366f1",
        title_style="bold #6366f1",
        show_lines=True,
        padding=(0, 1),
    )

    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Text", max_width=50, overflow="fold")
    table.add_column("Topic", width=20)
    table.add_column("Sentiment", width=10, justify="center")
    table.add_column("VADER", width=8, justify="right")
    table.add_column("TextBlob", width=8, justify="right")

    for i, doc in enumerate(documents):
        tid = topic_results["doc_topic_assignments"][i]
        topic_label = topic_results["topics"][tid]["label"]
        sent = doc_sentiments[i]

        # Colour sentiment
        label = sent["label"]
        if label == "Positive":
            label_str = f"[green]{label}[/green]"
        elif label == "Negative":
            label_str = f"[red]{label}[/red]"
        else:
            label_str = f"[yellow]{label}[/yellow]"

        # Truncate text
        text_display = doc[:80] + ("..." if len(doc) > 80 else "")

        table.add_row(
            str(i + 1),
            text_display,
            topic_label,
            label_str,
            f"{sent['vader_compound']:+.3f}",
            f"{sent['textblob_polarity']:+.3f}",
        )

    console.print(table)
    console.print()


def print_sentiment_bar_chart(topic_results: dict, topic_sentiments: list[dict]):
    """Print a simple ASCII bar chart of avg sentiment per topic."""
    console.print(Panel(
        "[bold]Topic Sentiment Scores[/bold] (VADER compound)",
        border_style="#6366f1",
        padding=(0, 2),
    ))

    max_bar = 30  # max bar width in chars

    for topic in topic_results["topics"]:
        tid = topic["topic_id"]
        ts = topic_sentiments[tid]
        score = ts["avg_compound"]

        # Normalise to bar width: score is -1 to +1, map to 0..max_bar
        bar_len = int(abs(score) * max_bar)
        bar_len = max(1, bar_len)

        if score >= 0.05:
            bar = f"[green]{'█' * bar_len}[/green]"
            label_style = "green"
        elif score <= -0.05:
            bar = f"[red]{'█' * bar_len}[/red]"
            label_style = "red"
        else:
            bar = f"[yellow]{'█' * bar_len}[/yellow]"
            label_style = "yellow"

        name = f"Topic {tid + 1}"
        console.print(f"  {name:<10} [{label_style}]{score:+.4f}[/] {bar}")

    console.print()


def print_footer():
    console.print(
        Panel(
            "[dim]Built with NLTK, scikit-learn, VADER & TextBlob[/dim]\n"
            "[dim]TopicSenti — Topic-Sentiment Analysis Tool (BYOP Capstone)[/dim]",
            border_style="dim",
            padding=(0, 2),
        )
    )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
