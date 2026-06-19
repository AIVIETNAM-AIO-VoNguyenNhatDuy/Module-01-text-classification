from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from configs.config import IMAGES_DIR
from data.preprocess import load_20newsgroups_processed


def evaluate_predictions(
    *,
    y_true,
    y_pred,
    target_names: list[str],
    run_id: str,
    output_dir: Path,
) -> dict[str, float | str]:
    """Compute metrics, save reports and confusion matrix, and return a result dict.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        target_names: Ordered list of class names corresponding to label indices.
        run_id: Identifier used to name output files.
        output_dir: Directory where the classification report and matrix CSV are saved.

    Returns:
        A dict with keys: run_id, accuracy, macro_f1, weighted_f1,
        report_path, confusion_matrix_path, confusion_matrix_image.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred)

    report_path = output_dir / f"{run_id}_classification_report.json"
    matrix_path = output_dir / f"{run_id}_confusion_matrix.csv"
    image_path = IMAGES_DIR / f"{run_id}_confusion_matrix.png"

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(matrix, index=target_names, columns=target_names).to_csv(matrix_path)

    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=target_names)
    axes = display.plot(xticks_rotation=45).ax_
    axes.figure.tight_layout()
    axes.figure.savefig(image_path, dpi=160)
    axes.figure.clf()

    return {
        "run_id": run_id,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "report_path": str(report_path),
        "confusion_matrix_path": str(matrix_path),
        "confusion_matrix_image": str(image_path),
    }


def evaluate_saved_model(
    *,
    categories: list[str],
    model_path: Path,
    output_dir: Path,
) -> dict[str, float | str]:
    """Load a saved pipeline and evaluate it on the processed test set.

    Args:
        categories: Category names used to filter the test set.
        model_path: Path to the saved sklearn pipeline (.pkl).
        output_dir: Directory where evaluation artifacts are written.

    Returns:
        A dict with keys: run_id, accuracy, macro_f1, weighted_f1,
        report_path, confusion_matrix_path, confusion_matrix_image.

    Raises:
        FileNotFoundError: If model_path does not exist.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    _, test_df = load_20newsgroups_processed()
    if categories:
        test_df = test_df[test_df["target_name"].isin(categories)].reset_index(drop=True)
    target_names = sorted(test_df["target_name"].unique().tolist())
    label_to_idx = {name: i for i, name in enumerate(target_names)}

    pipeline = joblib.load(model_path)
    predictions = pipeline.predict(test_df["clean_text"])

    result = evaluate_predictions(
        y_true=test_df["target_name"].map(label_to_idx),
        y_pred=predictions,
        target_names=target_names,
        run_id="best_model_evaluation",
        output_dir=output_dir,
    )
    print(pd.DataFrame([result]).to_string(index=False))
    return result
