from __future__ import annotations

from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer


def stopwords_for_ratio(ratio: float) -> list[str] | None:
    """Return a deterministic subset of sklearn English stopwords.

    Args:
        ratio: Fraction of stopwords to include, in [0.0, 1.0].
            0.0 returns None (no stopword removal); 1.0 returns the full list.

    Returns:
        A sorted list of stopwords of length `round(total * ratio)`,
        or None if ratio is 0.0.

    Raises:
        ValueError: If ratio is greater than 1.0.
    """
    if ratio <= 0:
        return None
    if ratio > 1:
        raise ValueError("ratio must be between 0.0 and 1.0")

    words = sorted(ENGLISH_STOP_WORDS)
    count = round(len(words) * ratio)
    return words[:count]


def build_vectorizer(
    kind: str = "tfidf",
    stop_words: list[str] | None = None,
) -> TfidfVectorizer | CountVectorizer:
    """Instantiate a text vectorizer by name.

    Args:
        kind: Vectorizer type — either "tfidf" or "count".
        stop_words: List of stopwords to filter out, or None to keep all words.

    Returns:
        An unfitted TfidfVectorizer or CountVectorizer with min_df=2.

    Raises:
        ValueError: If kind is not "tfidf" or "count".
    """
    if kind == "tfidf":
        return TfidfVectorizer(stop_words=stop_words, min_df=2)
    if kind == "count":
        return CountVectorizer(stop_words=stop_words, min_df=2)
    raise ValueError(f"Unsupported vectorizer kind: {kind}")
