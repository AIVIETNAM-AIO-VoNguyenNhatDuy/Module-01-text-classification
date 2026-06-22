from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.datasets import fetch_20newsgroups


def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw text into a normalised `clean_text` column.

    Strips email addresses, signature blocks, punctuation, and extra
    whitespace, then drops rows that are empty or shorter than 5 tokens.

    Args:
        df: DataFrame with a `text` column containing raw newsgroup posts.

    Returns:
        A new DataFrame with a `clean_text` column added and short/empty
        rows removed, with the index reset.
    """
    df["clean_text"] = df["text"].astype(str)
    df["clean_text"] = df["clean_text"].apply(lambda x: re.sub(r"\S+@\S+", " ", x))
    df["clean_text"] = df["clean_text"].apply(lambda x: re.sub(r"(?m)^--.*$", " ", x))
    df["clean_text"] = df["clean_text"].str.lower()
    df["clean_text"] = df["clean_text"].apply(lambda x: re.sub(r"[^a-z\s]", " ", x))
    df["clean_text"] = df["clean_text"].apply(lambda x: re.sub(r"\s+", " ", x).strip())
    df["clean_text"] = df["clean_text"].replace("", np.nan)
    df = df.dropna(subset=["clean_text"])
    df = df[df["clean_text"].str.split().str.len() >= 5]
    df = df.reset_index(drop=True)

    return df


def load_20newsgroups_raw(
    categories: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch train/test splits from sklearn and return them as DataFrames.

    Headers, footers, and quoted replies are stripped before returning.

    Args:
        categories: Subset of 20 Newsgroups category names to load.
            Loads all 20 categories if None.

    Returns:
        A tuple (train_df, test_df), each with columns `text`, `target`,
        and `target_name`.
    """
    train_data = fetch_20newsgroups(
        subset="train",
        categories=categories,
        remove=("headers", "footers", "quotes"),
    )
    test_data = fetch_20newsgroups(
        subset="test",
        categories=categories,
        remove=("headers", "footers", "quotes"),
    )

    train_df = pd.DataFrame(
        {
            "text": train_data.data,
            "target": train_data.target,
            "target_name": [train_data.target_names[i] for i in train_data.target],
        }
    )
    test_df = pd.DataFrame(
        {
            "text": test_data.data,
            "target": test_data.target,
            "target_name": [test_data.target_names[i] for i in test_data.target],
        }
    )

    return train_df, test_df


def save_csv(df: pd.DataFrame, file_path: Path | str) -> None:
    """Save a DataFrame to a CSV file without the row index.

    Args:
        df: DataFrame to save.
        file_path: Destination file path.

    Returns:
        None
    """
    df.to_csv(file_path, index=False)


def load_20newsgroups_processed(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the preprocessed 20 Newsgroups CSVs from disk.

    Args:
        data_dir: Path to the data directory containing the `processed/` subdirectory.

    Returns:
        A tuple (train_df, test_df) with columns `text`, `target`,
        `target_name`, and `clean_text`.
    """
    train_df = pd.read_csv(data_dir / "processed" / "20newsgroups_train.csv")
    test_df = pd.read_csv(data_dir / "processed" / "20newsgroups_test.csv")

    return train_df, test_df
