from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.evaluation.metrics import evaluate_saved_model
from src.pipelines.pipeline import train_pipeline


@hydra.main(version_base=None, config_path="src/configs", config_name="experiment")
def main(cfg: DictConfig) -> None:
    categories = list(cfg.categories)
    model_path = Path(cfg.paths.model_path)
    output_dir = Path(cfg.paths.results_dir)
    data_dir = Path(cfg.paths.data_dir)
    images_dir = Path(cfg.paths.images_dir)
    model_names = list(cfg.model_names)
    stopword_ratios = list(cfg.stopword_ratios)
    mode = cfg.mode

    if mode not in ("train", "evaluate", "all"):
        raise SystemExit(f"Invalid mode '{mode}'. Choose train, evaluate, or all.")

    if mode in ("train", "all"):
        train_pipeline(
            categories=categories,
            stopword_ratios=stopword_ratios,
            model_names=model_names,
            model_path=model_path,
            output_dir=output_dir,
            images_dir=images_dir,
        )

    if mode in ("evaluate", "all"):
        evaluate_saved_model(
            categories=categories,
            model_path=model_path,
            output_dir=output_dir,
            data_dir=data_dir,
            images_dir=images_dir,
        )


if __name__ == "__main__":
    main()
