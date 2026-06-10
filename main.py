from __future__ import annotations

import argparse
from pathlib import Path

from configs.config import DEFAULT_CATEGORIES, MODEL_PATH, RESULTS_DIR
from evaluation.metrics import evaluate_saved_model
from pipelines.pipeline import train_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stopword removal text classification pipeline.")
    parser.add_argument("--train", action="store_true", help="Train models and save the best pipeline.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the saved pipeline.")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help="20 Newsgroups categories to use.",
    )
    parser.add_argument(
        "--stopword-ratios",
        nargs="+",
        type=float,
        default=[0.0, 0.2, 0.5, 0.8, 1.0],
        help="Stopword removal ratios to benchmark.",
    )
    parser.add_argument("--model-path", default=str(MODEL_PATH), help="Path to save/load model.")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR), help="Directory for result files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)

    if not args.train and not args.evaluate:
        raise SystemExit("Use --train, --evaluate, or both.")

    if args.train:
        train_pipeline(
            categories=args.categories,
            stopword_ratios=args.stopword_ratios,
            model_path=model_path,
            output_dir=output_dir,
        )

    if args.evaluate:
        evaluate_saved_model(
            categories=args.categories,
            model_path=model_path,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
