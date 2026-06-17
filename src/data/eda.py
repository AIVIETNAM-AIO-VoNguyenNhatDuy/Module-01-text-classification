from __future__ import annotations

from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import pandas as pd

def show_long_short_docs(df, text_col):
    word_counts = df[text_col].dropna().astype(str).str.split().str.len()
    
    short_docs = df.loc[word_counts.nsmallest(3).index]
    long_docs = df.loc[word_counts.nlargest(1).index]
    
    print("--- Short document examples ---")
    display(short_docs[[text_col, 'target_name']])
    
    print("\n--- Long document examples ---")
    display(long_docs[[text_col, 'target_name']])

def plot_top_ngrams(corpus, title, ax, n=10, ngram_range=(1, 1)):
    vec = CountVectorizer(ngram_range=ngram_range, stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:n]
    df = pd.DataFrame(words_freq, columns=['text', 'count'])
    ax.barh(df['text'][::-1], df['count'][::-1])
    ax.set_title(title)