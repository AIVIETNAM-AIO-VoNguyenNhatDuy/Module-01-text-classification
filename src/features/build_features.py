from __future__ import annotations

from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer


def stopwords_for_ratio(ratio: float) -> list[str] | None:
    """Return a deterministic subset of sklearn English stopwords."""
    if ratio <= 0:
        return None
    if ratio > 1:
        raise ValueError("ratio must be between 0.0 and 1.0")

    words = sorted(ENGLISH_STOP_WORDS)
    count = round(len(words) * ratio)
    return words[:count]


def build_vectorizer(kind: str = "tfidf", stop_words: list[str] | None = None):
    if kind == "tfidf":
        return TfidfVectorizer(stop_words=stop_words, min_df=2)
    if kind == "count":
        return CountVectorizer(stop_words=stop_words, min_df=2)
    raise ValueError(f"Unsupported vectorizer kind: {kind}")
