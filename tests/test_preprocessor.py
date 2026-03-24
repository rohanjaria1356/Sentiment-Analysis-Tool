"""Tests for nlp.preprocessor module."""

import pytest
from nlp.preprocessor import clean_text, tokenize_and_lemmatize, split_into_sentences, prepare_corpus


class TestCleanText:
    """Tests for the clean_text function."""

    def test_lowercases(self):
        assert clean_text("HELLO WORLD") == "hello world"

    def test_removes_urls(self):
        result = clean_text("Visit https://example.com for more info")
        assert "https" not in result
        assert "example.com" not in result

    def test_removes_html(self):
        result = clean_text("This is <b>bold</b> text")
        assert "<b>" not in result
        assert "</b>" not in result

    def test_removes_digits(self):
        result = clean_text("Price is 100 dollars in 2024")
        assert "100" not in result
        assert "2024" not in result

    def test_expands_contractions(self):
        result = clean_text("I can't believe it won't work")
        assert "cannot" in result
        assert "will not" in result

    def test_removes_punctuation(self):
        result = clean_text("Hello, world! How are you?")
        assert "," not in result
        assert "!" not in result
        assert "?" not in result

    def test_collapses_whitespace(self):
        result = clean_text("too   many    spaces")
        assert "  " not in result


class TestTokenizeAndLemmatize:
    """Tests for tokenize_and_lemmatize."""

    def test_removes_stopwords(self):
        tokens = tokenize_and_lemmatize("this is a simple test")
        assert "this" not in tokens
        assert "is" not in tokens

    def test_lemmatizes(self):
        tokens = tokenize_and_lemmatize("the cats were running quickly")
        assert "cat" in tokens

    def test_removes_short_tokens(self):
        tokens = tokenize_and_lemmatize("i am at the go")
        # All tokens <= 2 chars should be removed
        for tok in tokens:
            assert len(tok) > 2


class TestSplitIntoSentences:
    """Tests for split_into_sentences."""

    def test_splits_sentences(self):
        text = "Hello there. How are you? I am fine."
        sentences = split_into_sentences(text)
        assert len(sentences) == 3

    def test_single_sentence(self):
        sentences = split_into_sentences("Just one sentence")
        assert len(sentences) == 1


class TestPrepareCorpus:
    """Tests for prepare_corpus."""

    def test_returns_two_lists(self):
        docs = ["This is document one.", "This is document two."]
        token_lists, cleaned_docs = prepare_corpus(docs)
        assert len(token_lists) == 2
        assert len(cleaned_docs) == 2

    def test_cleaned_docs_are_strings(self):
        docs = ["Hello world test document."]
        _, cleaned_docs = prepare_corpus(docs)
        assert isinstance(cleaned_docs[0], str)

    def test_token_lists_are_lists(self):
        docs = ["Hello world test document."]
        token_lists, _ = prepare_corpus(docs)
        assert isinstance(token_lists[0], list)
