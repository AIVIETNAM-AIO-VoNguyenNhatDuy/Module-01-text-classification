from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB

DEFAULT_RANDOM_STATE = 42
DEFAULT_CV_SPLITS = 3
DEFAULT_SCORING = "f1_macro"

MODEL_PARAM_GRIDS: dict[str, dict[str, list[float | int | str | None]]] = {
    "naive_bayes": {
        "alpha": [0.1, 0.5, 1.0],
    },
    "logistic_regression": {
        "C": [0.1, 1.0, 10.0],
        "class_weight": [None, "balanced"],
    },
    "random_forest": {
        "max_depth": [None, 30],
        "min_samples_leaf": [1, 2],
    },
}


def build_base_model(name: str) -> BaseEstimator:
    """Instantiate a classifier without hyperparameter tuning.

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
        return LogisticRegression(max_iter=1000, random_state=DEFAULT_RANDOM_STATE)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=DEFAULT_RANDOM_STATE,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model: {name}")


def build_model(
    name: str,
    *,
    tune: bool = True,
    cv_splits: int = DEFAULT_CV_SPLITS,
    scoring: str = DEFAULT_SCORING,
    n_jobs: int = -1,
) -> BaseEstimator:
    """Instantiate a classifier, with GridSearchCV enabled by default.

    Args:
        name: One of "naive_bayes", "logistic_regression", or "random_forest".
        tune: Whether to wrap the classifier with GridSearchCV.
        cv_splits: Number of stratified folds used when tune is True.
        scoring: Scoring metric optimized by GridSearchCV.
        n_jobs: Number of parallel jobs used by GridSearchCV.

    Returns:
        An unfitted sklearn estimator. By default this is GridSearchCV, so any
        existing pipeline that calls build_model() automatically performs
        cross-validation and hyperparameter tuning during fit().
    """
    estimator = build_base_model(name)
    if not tune:
        return estimator
    return build_grid_search(
        estimator=estimator,
        model_name=name,
        cv_splits=cv_splits,
        scoring=scoring,
        n_jobs=n_jobs,
    )


def get_param_grid(name: str) -> dict[str, list[float | int | str | None]]:
    """Return the hyperparameter grid for a supported model.

    Args:
        name: One of "naive_bayes", "logistic_regression", or "random_forest".

    Returns:
        A copy of the model's GridSearchCV parameter grid.

    Raises:
        ValueError: If name is not a supported model.
    """
    if name not in MODEL_PARAM_GRIDS:
        raise ValueError(f"Unsupported model: {name}")
    return {parameter: values.copy() for parameter, values in MODEL_PARAM_GRIDS[name].items()}


def build_cv(cv_splits: int = DEFAULT_CV_SPLITS) -> StratifiedKFold:
    """Build the cross-validation splitter used for model selection.

    Args:
        cv_splits: Number of stratified folds.

    Returns:
        A shuffled StratifiedKFold instance.
    """
    return StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=DEFAULT_RANDOM_STATE,
    )


def build_grid_search(
    *,
    estimator: BaseEstimator,
    model_name: str,
    cv_splits: int = DEFAULT_CV_SPLITS,
    scoring: str = DEFAULT_SCORING,
    n_jobs: int = -1,
) -> GridSearchCV:
    """Wrap an estimator with cross-validation and hyperparameter tuning.

    Args:
        estimator: Unfitted sklearn classifier.
        model_name: Name used to select the hyperparameter grid.
        cv_splits: Number of stratified folds for cross-validation.
        scoring: Scoring metric optimized by GridSearchCV.
        n_jobs: Number of parallel jobs used by GridSearchCV.

    Returns:
        An unfitted GridSearchCV object.
    """
    return GridSearchCV(
        estimator=estimator,
        param_grid=get_param_grid(model_name),
        scoring=scoring,
        cv=build_cv(cv_splits),
        n_jobs=n_jobs,
        refit=True,
        return_train_score=True,
    )