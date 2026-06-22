from __future__ import annotations

import pandas as pd
from IPython.display import display
from matplotlib.axes import Axes
from sklearn.feature_extraction.text import CountVectorizer


def show_long_short_docs(df: pd.DataFrame, text_col: str) -> None:
    """Print the shortest and longest documents in a DataFrame.

    Args:
        df: DataFrame containing the text column and a `target_name` column.
        text_col: Name of the column holding the document text.

    Returns:
        None
    """
    word_counts = df[text_col].dropna().astype(str).str.split().str.len()

    short_docs = df.loc[word_counts.nsmallest(3).index]
    long_docs = df.loc[word_counts.nlargest(1).index]

    print("--- Short document examples ---")
    display(short_docs[[text_col, "target_name"]])

    print("\n--- Long document examples ---")
    display(long_docs[[text_col, "target_name"]])


def plot_top_ngrams(
    corpus: list[str],
    title: str,
    ax: Axes,
    n: int = 10,
    ngram_range: tuple[int, int] = (1, 1),
) -> None:
    """Plot a horizontal bar chart of the top n most frequent n-grams.

    Args:
        corpus: List of document strings to analyse.
        title: Chart title.
        ax: Matplotlib Axes to draw on.
        n: Number of top n-grams to display.
        ngram_range: The (min, max) n-gram sizes to extract.

    Returns:
        None
    """
    vec = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:n]
    df = pd.DataFrame(words_freq, columns=["text", "count"])
    ax.barh(df["text"][::-1], df["count"][::-1])
    ax.set_title(title)
