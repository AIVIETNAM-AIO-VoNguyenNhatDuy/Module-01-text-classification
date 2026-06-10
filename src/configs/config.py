from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
IMAGES_DIR = ROOT_DIR / "images"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
RESULTS_DIR = REPORTS_DIR / "results"

MODEL_PATH = MODELS_DIR / "best_text_classifier.pkl"

DEFAULT_CATEGORIES = [
    "sci.space",
    "rec.sport.hockey",
    "comp.graphics",
    "talk.politics.misc",
]

MODEL_NAMES = [
    "naive_bayes",
    "logistic_regression",
    "random_forest",
]

STOPWORD_RATIOS = [0.0, 0.2, 0.5, 0.8, 1.0]
