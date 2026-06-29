from src.data.preprocess import clean_text


def test_clean_text_lowercases_and_removes_punctuation() -> None:
    assert clean_text("Hello, WORLD!!! 123") == "hello world"
