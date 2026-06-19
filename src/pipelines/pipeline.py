from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from configs.config import MODEL_NAMES
from data.preprocess import load_20newsgroups_data
from evaluation.metrics import evaluate_predictions
from features.build_features import build_vectorizer, stopwords_for_ratio
from models.model import build_model


def train_pipeline(
    *,
    categories: list[str],
    stopword_ratios: list[float],
    model_path: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Train all model/ratio combinations, save the best pipeline, and return metrics.

    Iterates over every (stopword_ratio, model_name) pair, fits a TF-IDF +
    classifier pipeline, evaluates on the test set, and persists the pipeline
    with the highest weighted F1.

    Args:
        categories: 20 Newsgroups category names to load for train/test.
        stopword_ratios: Fractions of stopwords to remove, e.g. [0.0, 0.5, 1.0].
        model_path: Destination path to save the best fitted pipeline.
        output_dir: Directory where per-run reports and confusion matrices are written.

    Returns:
        A DataFrame with one row per (ratio, model) run containing run_id,
        accuracy, macro_f1, weighted_f1, vectorizer, model, and stopword_ratio.
    """
    train_data, test_data = load_20newsgroups_data(categories=categories)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    best_pipeline = None
    best_weighted_f1 = -1.0

    for ratio in stopword_ratios:
        for model_name in MODEL_NAMES:
            run_id = f"tfidf_{model_name}_stopwords_{ratio:.1f}"
            pipeline = Pipeline(
                steps=[
                    ("vectorizer", build_vectorizer("tfidf", stopwords_for_ratio(ratio))),
                    ("model", build_model(model_name)),
                ]
            )
            pipeline.fit(train_data.data, train_data.target)
            predictions = pipeline.predict(test_data.data)

            result = evaluate_predictions(
                y_true=test_data.target,
                y_pred=predictions,
                target_names=test_data.target_names,
                run_id=run_id,
                output_dir=output_dir,
            )
            result.update(
                {
                    "vectorizer": "tfidf",
                    "model": model_name,
                    "stopword_ratio": ratio,
                }
            )
            rows.append(result)

            if result["weighted_f1"] > best_weighted_f1:
                best_weighted_f1 = float(result["weighted_f1"])
                best_pipeline = pipeline

    if best_pipeline is not None:
        joblib.dump(best_pipeline, model_path)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    print(metrics.to_string(index=False))
    print(f"\nSaved best model to {model_path}")
    return metrics
