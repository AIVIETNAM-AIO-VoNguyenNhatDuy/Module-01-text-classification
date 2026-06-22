from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


def build_model(name: str) -> BaseEstimator:
    """Instantiate a classifier by name.

    Args:
        name: One of "naive_bayes", "logistic_regression", or "random_forest".

    Returns:
        An unfitted sklearn classifier instance.

    Raises:
        ValueError: If name is not a supported model.
    """
    if name == "naive_bayes":
        return MultinomialNB()
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    raise ValueError(f"Unsupported model: {name}")
