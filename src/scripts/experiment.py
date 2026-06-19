from pathlib import Path

import mlflow
import hydra
from omegaconf import DictConfig
from sklearn.pipeline import Pipeline

from data.preprocess import load_20newsgroups_processed
from evaluation.metrics import evaluate_predictions
from features.build_features import build_vectorizer, stopwords_for_ratio
from models.model import build_model


def build_pipeline(model_name: str, ratio: float) -> Pipeline:
    """Build a TF-IDF vectorizer + classifier as a single sklearn Pipeline.

    Args:
        model_name: Name of the classifier (e.g. "naive_bayes", "logistic_regression").
        ratio: Fraction of stopwords to remove, in [0.0, 1.0].

    Returns:
        An unfitted sklearn Pipeline with a vectorizer and model step.
    """
    return Pipeline(
        [
            ("vectorizer", build_vectorizer("tfidf", stopwords_for_ratio(ratio))),
            ("model", build_model(model_name)),
        ]
    )


def log_to_mlflow(
    run_name: str,
    model_name: str,
    ratio: float,
    categories: list[str],
    result: dict,
) -> None:
    """Log params, metrics, and artifacts for a single run to MLflow.

    Args:
        run_name: Display name for the MLflow run.
        model_name: Name of the classifier used.
        ratio: Stopword removal ratio used.
        categories: List of 20 Newsgroups categories used.
        result: Evaluation result dict from evaluate_predictions.

    Returns:
        None
    """
    mlflow.set_experiment("text-classification")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "vectorizer": "tfidf",
                "model": model_name,
                "stopword_ratio": ratio,
                "categories": categories,
            }
        )
        mlflow.log_metrics(
            {
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
                "weighted_f1": result["weighted_f1"],
            }
        )
        mlflow.log_artifact(result["report_path"])
        mlflow.log_artifact(result["confusion_matrix_path"])
        mlflow.log_artifact(result["confusion_matrix_image"])


def run_experiment(cfg: DictConfig) -> None:
    """Run a single training and evaluation experiment from a Hydra config.

    Args:
        cfg: Hydra config containing model, stopword_ratio, categories, and paths.

    Returns:
        None
    """
    model_name = cfg.model
    ratio = cfg.stopword_ratio
    categories = list(cfg.categories)
    output_dir = Path(cfg.paths.results_dir)
    model_path = Path(cfg.paths.model_path)
    data_dir = Path(cfg.paths.data_dir)
    images_dir = Path(cfg.paths.images_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    train_df, test_df = load_20newsgroups_processed(data_dir)
    if categories:
        train_df = train_df[train_df["target_name"].isin(categories)].reset_index(drop=True)
        test_df = test_df[test_df["target_name"].isin(categories)].reset_index(drop=True)

    target_names = sorted(train_df["target_name"].unique().tolist())
    label_to_idx = {name: i for i, name in enumerate(target_names)}

    # Train and evaluate
    pipeline = build_pipeline(model_name, ratio)
    pipeline.fit(train_df["clean_text"], train_df["target_name"].map(label_to_idx))
    predictions = pipeline.predict(test_df["clean_text"])

    run_name = f"tfidf_{model_name}_stopwords_{ratio:.1f}"
    result = evaluate_predictions(
        y_true=test_df["target_name"].map(label_to_idx),
        y_pred=predictions,
        target_names=target_names,
        run_id=run_name,
        output_dir=output_dir,
        images_dir=images_dir,
    )

    # Log to MLflow
    log_to_mlflow(run_name, model_name, ratio, categories, result)


@hydra.main(version_base=None, config_path="../configs", config_name="experiment")
def main(cfg: DictConfig) -> None:
    """Hydra entry point.

    Args:
        cfg: Hydra config loaded from experiment.yaml.

    Returns:
        None
    """
    run_experiment(cfg)


if __name__ == "__main__":
    main()
