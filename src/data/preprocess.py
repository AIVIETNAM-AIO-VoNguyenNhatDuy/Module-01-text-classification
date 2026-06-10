from __future__ import annotations

import re
import string

from sklearn.datasets import fetch_20newsgroups


def clean_text(text: str, remove_numbers: bool = True) -> str:
    """Apply minimal deterministic text cleanup for notebook and pipeline reuse."""
    text = text.lower()
    if remove_numbers:
        text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans({char: " " for char in string.punctuation}))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_20newsgroups_data(categories: list[str] | None = None):
    """Load train/test splits with headers, footers, and quotes removed."""
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
    return train_data, test_data
