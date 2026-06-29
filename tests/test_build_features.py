from src.features.build_features import build_vectorizer, stopwords_for_ratio


def test_stopword_ratio_zero_keeps_stopwords() -> None:
    assert stopwords_for_ratio(0.0) is None


def test_stopword_ratio_returns_deterministic_subset() -> None:
    first = stopwords_for_ratio(0.2)
    second = stopwords_for_ratio(0.2)

    assert first == second
    assert first is not None
    assert len(first) > 0


def test_build_tfidf_vectorizer(sample_texts: list[str]) -> None:
    vectorizer = build_vectorizer("tfidf")
    matrix = vectorizer.fit_transform(sample_texts)

    assert matrix.shape[0] == len(sample_texts)
