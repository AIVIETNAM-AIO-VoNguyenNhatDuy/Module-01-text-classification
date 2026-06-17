from __future__ import annotations

import re
import string
import pandas as pd
import numpy as np

from sklearn.datasets import fetch_20newsgroups

def clean_text(df):    
    df['clean_text'] = df['text'].astype(str)
    
    df['clean_text'] = df['clean_text'].apply(lambda x: re.sub(r'\S+@\S+', ' ', x))
    
    df['clean_text'] = df['clean_text'].apply(lambda x: re.sub(r'(?m)^--.*$', ' ', x))
    
    df['clean_text'] = df['clean_text'].str.lower()
    
    df['clean_text'] = df['clean_text'].apply(lambda x: re.sub(r'[^a-z\s]', ' ', x))
    
    df['clean_text'] = df['clean_text'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    
    df['clean_text'] = df['clean_text'].replace('', np.nan)
    df = df.dropna(subset=['clean_text'])

    df = df[df['clean_text'].str.split().str.len() >= 5]
    df = df.reset_index(drop=True)

    return df


def load_20newsgroups_raw(categories: list[str] | None = None):
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

    train_df = pd.DataFrame({
        'text': train_data.data,
        'target': train_data.target,
        'target_name': [train_data.target_names[i] for i in train_data.target]
    })
    test_df = pd.DataFrame({
        'text': test_data.data,
        'target': test_data.target,
        'target_name': [test_data.target_names[i] for i in test_data.target]
    })
    
    return train_df, test_df
