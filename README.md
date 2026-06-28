# Impact of Stopword Removal on Text Classification

This repository contains a research project on how stopword removal affects text
classification performance. The project uses the 20 Newsgroups dataset, TF-IDF
features, and classical machine learning models.

The main goal is not only to find the best model, but also to understand whether
removing stopwords helps, how much it helps, and whether the effect is consistent
across models and classes.

## Research Questions

- Does stopword removal improve text classification performance?
- Which stopword removal ratio works best: `0.0`, `0.2`, `0.5`, `0.8`, or `1.0`?
- Which model works best with TF-IDF features?
- Does stopword removal help all classes equally?
- Can aggressive stopword removal hurt performance?

## Dataset

The project uses `sklearn.datasets.fetch_20newsgroups` with the official
train/test split. Headers, footers, and quoted replies are removed when loading
the raw data to reduce metadata leakage.

Current category subset:

- `comp.graphics`
- `rec.sport.hockey`
- `sci.space`
- `talk.politics.misc`

Processed data is stored in:

```text
data/processed/20newsgroups_train.csv
data/processed/20newsgroups_test.csv
```

Current processed dataset size:

| Split | Documents |
| --- | ---: |
| Train | 2,160 |
| Test | 1,447 |

## Methodology

```text
20 Newsgroups
  |
  v
Remove headers, footers, and quotes
  |
  v
Text preprocessing
  |- remove email-like patterns
  |- remove simple signature lines
  |- lowercase text
  |- keep alphabetic characters and whitespace
  |- normalize whitespace
  |- drop empty and very short documents
  |
  v
Stopword removal ratio
  |- 0.0, 0.2, 0.5, 0.8, 1.0
  |
  v
TF-IDF vectorization
  |- min_df=2
  |
  v
Model training and tuning
  |- Multinomial Naive Bayes
  |- Logistic Regression
  |- Random Forest
  |
  v
Evaluation
  |- Accuracy
  |- Macro F1
  |- Weighted F1
  |- Per-class report
  |- Confusion matrix
```

## Experiment Setup

Experiment configuration is stored in:

```text
src/configs/experiment.yaml
```

Default settings:

| Setting | Value |
| --- | --- |
| Vectorizer | TF-IDF |
| Stopword ratios | `0.0`, `0.2`, `0.5`, `0.8`, `1.0` |
| Models | `naive_bayes`, `logistic_regression`, `random_forest` |
| Model selection | `GridSearchCV` |
| CV splitter | 3-fold shuffled `StratifiedKFold` |
| CV scoring | `f1_macro` |
| Random state | `42` |

Model search spaces:

| Model | Hyperparameters |
| --- | --- |
| Naive Bayes | `alpha`: `0.1`, `0.5`, `1.0` |
| Logistic Regression | `C`: `0.1`, `1.0`, `10.0`; `class_weight`: `None`, `balanced` |
| Random Forest | `max_depth`: `None`, `30`; `min_samples_leaf`: `1`, `2` |

## Results

Main metrics are saved in:

```text
reports/results/metrics.csv
```

Best run from the current experiment:

| Model | Stopword Ratio | Accuracy | Macro F1 | Weighted F1 | Best Params |
| --- | ---: | ---: | ---: | ---: | --- |
| Naive Bayes | `0.5` | `0.9143` | `0.9117` | `0.9144` | `alpha=0.1` |

Best result per model:

| Model | Best Ratio | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: | ---: |
| Naive Bayes | `0.5` | `0.9143` | `0.9117` | `0.9144` |
| Logistic Regression | `1.0` | `0.8922` | `0.8904` | `0.8927` |
| Random Forest | `0.8` | `0.8535` | `0.8521` | `0.8550` |

Macro F1 by stopword ratio:

| Model | 0.0 | 0.2 | 0.5 | 0.8 | 1.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Naive Bayes | `0.9108` | `0.9110` | `0.9117` | `0.9110` | `0.9090` |
| Logistic Regression | `0.8775` | `0.8824` | `0.8811` | `0.8843` | `0.8904` |
| Random Forest | `0.8372` | `0.8458` | `0.8497` | `0.8521` | `0.8493` |

Key observations:

- Stopword removal helps, but the improvement is model-dependent.
- Naive Bayes is the best overall model in the current setup.
- Naive Bayes is already strong without stopword removal; ratio `0.5` gives the
  best score, but the gain over ratio `0.0` is small.
- Logistic Regression improves as more stopwords are removed, with its best
  result at ratio `1.0`.
- Random Forest improves up to ratio `0.8`, but it remains weaker than the other
  two models on sparse TF-IDF features.
- Full stopword removal is not automatically the best choice. It slightly hurts
  Naive Bayes compared with partial removal.

Confusion matrix images are generated in:

```text
images/
```

## Project Structure

```text
module-01-text-classification/
|-- data/
|   |-- raw/                         # Local raw data, ignored by git
|   |-- interim/                     # Temporary intermediate files
|   `-- processed/                   # Processed train/test CSV files
|-- images/                          # Generated confusion matrix images
|-- models/                          # Saved model artifacts
|-- notebooks/
|   |-- eda.ipynb
|   |-- preprocessing.ipynb
|   |-- baseline_no_stopwords.ipynb
|   |-- baseline_stopwords.ipynb
|   |-- model_machine_learning.ipynb
|   `-- analysis.ipynb
|-- reports/
|   |-- figures/
|   `-- results/                     # Metrics, reports, confusion matrices
|-- src/
|   |-- configs/experiment.yaml      # Hydra experiment configuration
|   |-- data/preprocess.py           # Data loading and text cleaning
|   |-- data/eda.py                  # EDA helper functions
|   |-- features/build_features.py   # TF-IDF/Count vectorizers and stopword ratios
|   |-- models/model.py              # Model factory and GridSearchCV setup
|   |-- pipelines/pipeline.py        # Full training loop
|   |-- scripts/experiment.py        # Single-run Hydra + MLflow experiment script
|   `-- evaluation/metrics.py        # Metrics and confusion matrix outputs
|-- tests/
|   |-- conftest.py
|   |-- test_preprocess.py
|   |-- test_build_features.py
|   `-- test_metrics.py
|-- main.py
|-- pyproject.toml
|-- requirements.txt
`-- README.md
```

## Installation

### Option 1: `uv` (recommended)

Install dependencies:

```bash
uv sync
```

Install development dependencies as well:

```bash
uv sync --group dev
```

### Option 2: `pip`

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Experiments

### Run the main training pipeline

The intended main entry point is `main.py` with Hydra config values.

```bash
PYTHONPATH=src uv run python main.py mode=train
```

Evaluate a saved best model:

```bash
PYTHONPATH=src uv run python main.py mode=evaluate
```

Train and evaluate:

```bash
PYTHONPATH=src uv run python main.py mode=all
```

On Windows PowerShell, use:

```powershell
$env:PYTHONPATH = "src"
uv run python main.py mode=train
```

### Run one Hydra experiment with MLflow logging

```bash
PYTHONPATH=src uv run python src/scripts/experiment.py model=naive_bayes stopword_ratio=0.5
```

Run all model/ratio combinations with Hydra multirun:

```bash
PYTHONPATH=src uv run python src/scripts/experiment.py --multirun \
  model=naive_bayes,logistic_regression,random_forest \
  stopword_ratio=0.0,0.2,0.5,0.8,1.0
```

Open MLflow UI:

```bash
uv run mlflow ui
```

Then open:

```text
http://localhost:5000
```

## Notebooks

Open notebooks with:

```bash
uv run jupyter lab
```

Notebook roles:

| Notebook | Purpose |
| --- | --- |
| `eda.ipynb` | Explore class distribution, document length, and top words |
| `preprocessing.ipynb` | Build and validate processed train/test CSV files |
| `baseline_no_stopwords.ipynb` | Baseline experiments without stopword removal |
| `baseline_stopwords.ipynb` | Stopword-ratio baseline experiments |
| `model_machine_learning.ipynb` | Model comparison and experiment runs |
| `analysis.ipynb` | Final result analysis and interpretation |

## Testing

Run tests:

```bash
uv run pytest
```

or:

```bash
pytest
```

The tests focus on small utility functions and do not require running the full
training pipeline.

## Generated Artifacts

Generated outputs are intentionally ignored by git:

- `reports/results/*`
- `reports/figures/*`
- `images/*`
- `models/*`
- `mlruns/`
- `mlartifacts/`
- `outputs/`
- `multirun/`

The repository keeps `.gitkeep` files so the expected directories still exist
after cloning.

## Team

| Name | Role |
| --- | --- |
| Võ Nguyễn Nhật Duy | Team Leader |
| Trần Sơn Phát | AI Engineer (Pipeline) |
| Huỳnh Ngọc Minh | AI Engineer (Data) |
| Trần Hải Đăng | AI Engineer (Model) |

