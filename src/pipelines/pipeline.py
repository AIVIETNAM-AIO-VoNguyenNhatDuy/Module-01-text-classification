from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data.preprocess import load_20newsgroups_processed
from src.evaluation.metrics import evaluate_predictions
from src.features.build_features import build_vectorizer, stopwords_for_ratio
from src.models.model import build_model


def train_pipeline(
    *,
    categories: list[str],
    stopword_ratios: list[float],
    model_names: list[str],
    data_dir: Path,
    model_path: Path,
    output_dir: Path,
    images_dir: Path,
) -> pd.DataFrame:
    """Train all model/ratio combinations, save the best pipeline, and return metrics.

    Iterates over every (stopword_ratio, model_name) pair, fits a TF-IDF +
    classifier pipeline, evaluates on the test set, and persists the pipeline
    with the highest weighted F1.

    Args:
        categories: 20 Newsgroups category names to load for train/test.
        stopword_ratios: Fractions of stopwords to remove, e.g. [0.0, 0.5, 1.0].
        model_names: Names of classifiers to benchmark.
        data_dir: Directory containing the processed dataset.
        model_path: Destination path to save the best fitted pipeline.
        output_dir: Directory where per-run reports and confusion matrices are written.
        images_dir: Directory where confusion matrix images are saved.

    Returns:
        A DataFrame with one row per (ratio, model) run containing run_id,
        accuracy, macro_f1, weighted_f1, vectorizer, model, stopword_ratio,
        cv_best_score, and best_params.
    """
    train_df, test_df = load_20newsgroups_processed(data_dir)
    train_df = train_df[train_df["target_name"].isin(categories)].reset_index(drop=True)
    test_df = test_df[test_df["target_name"].isin(categories)].reset_index(drop=True)
    target_names = train_df.sort_values("target")["target_name"].drop_duplicates().tolist()

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    best_pipeline = None
    best_weighted_f1 = -1.0

    for ratio in stopword_ratios:
        for model_name in model_names:
            run_id = f"tfidf_{model_name}_stopwords_{ratio:.1f}"
            pipeline = Pipeline(
                steps=[
                    ("vectorizer", build_vectorizer("tfidf", stopwords_for_ratio(ratio))),
                    ("model", build_model(model_name, n_jobs=1)),
                ]
            )
            pipeline.fit(train_df["clean_text"], train_df["target"])
            model_step = pipeline.named_steps["model"]
            predictions = pipeline.predict(test_df["clean_text"])

            result = evaluate_predictions(
                y_true=test_df["target"],
                y_pred=predictions,
                target_names=target_names,
                run_id=run_id,
                output_dir=output_dir,
                images_dir=images_dir,
            )
            result.update(
                {
                    "vectorizer": "tfidf",
                    "model": model_name,
                    "stopword_ratio": ratio,
                    "cv_best_score": float(model_step.best_score_),
                    "best_params": json.dumps(model_step.best_params_, sort_keys=True),
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
