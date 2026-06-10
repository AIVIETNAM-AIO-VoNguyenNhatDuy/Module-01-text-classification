from pathlib import Path

from evaluation.metrics import evaluate_predictions


def test_evaluate_predictions_writes_outputs(tmp_path: Path) -> None:
    result = evaluate_predictions(
        y_true=[0, 1, 1, 0],
        y_pred=[0, 1, 0, 0],
        target_names=["class_a", "class_b"],
        run_id="unit_test",
        output_dir=tmp_path,
    )

    assert 0 <= result["accuracy"] <= 1
    assert Path(result["report_path"]).exists()
    assert Path(result["confusion_matrix_path"]).exists()
