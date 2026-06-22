# Module 01 Project - Impact of Stopword Removal on Text Classification

Research project for the Module 1 text classification topic.

This project studies the impact of stopword removal on text classification performance using the 20 Newsgroups dataset. The team benchmarks classical machine learning models under multiple stopword settings and compares overall metrics, per-class changes, and error cases.

## Research Questions

- Does stopword removal improve text classification performance?
- How does stopword removal affect accuracy and F1-score?
- Which model is most stable when stopwords are removed?
- Does stopword removal help all classes equally, or help some classes while hurting others?
- What happens when removing stopwords at different levels: `0%`, `20%`, `50%`, `80%`, and `100%`?

## Results at a Glance

Fill this section after experiments are completed.

| Model | Stopword Ratio | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: | ---: |
| Naive Bayes | TBD | TBD | TBD | TBD |
| Logistic Regression | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD |

## Evaluation Charts

Generated charts should be saved in `images/`.

- Confusion matrix
- Per-class F1 comparison
- Stopword ratio vs F1-score

## Dataset

Primary dataset: `sklearn.datasets.fetch_20newsgroups`.

Recommended subset for faster iteration:

- `sci.space`
- `rec.sport.hockey`
- `comp.graphics`
- `talk.politics.misc`
- 
## Methodology

```text
20 Newsgroups Dataset
  |
  v
Remove headers, footers, and quotes
  |
  v
Basic text preprocessing
  |- lowercase
  |- remove punctuation
  |- remove special characters
  |- optionally remove numbers
  |
  v
Create experiment branches
  |- Baseline A: TF-IDF without stopword removal
  |- Baseline B: TF-IDF with stopword removal
  |- Optional: CountVectorizer comparison
  |
  v
Train models
  |- Naive Bayes
  |- Logistic Regression
  |- Random Forest
  |
  v
Evaluate
  |- Accuracy
  |- Precision / Recall / F1-score
  |- Confusion matrix
  |- Per-class performance deltas
```

## Project Structure

```text
module-01-text-classification/
|-- data/
|   |-- raw/                         # Local raw data, ignored by git
|   |-- interim/                     # Temporary processed files
|   `-- processed/                   # Final processed files
|-- docs/
|   |-- project-brief.md
|   |-- experiment-plan.md
|   |-- references.md
|   `-- team-workflow.md
|-- images/
|   `-- .gitkeep                     # Generated charts go here
|-- models/
|   `-- .gitkeep                     # Saved model files go here
|-- notebooks/
|   |-- eda.ipynb
|   |-- preprocessing.ipynb
|   |-- baseline_a_no_stopwords.ipynb
|   |-- baseline_b_stopwords.ipynb
|   |-- model_machine_learning.ipynb
|   `-- analysis.ipynb
|-- reports/
|   |-- figures/
|   `-- results/
|-- src/
|   |-- configs/config.py            # Paths, categories, model names
|   |-- data/preprocess.py           # Load and clean text data
|   |-- features/build_features.py   # Vectorizers and stopword ratios
|   |-- models/model.py              # Model factory
|   |-- pipelines/pipeline.py        # Training loop
|   `-- evaluation/metrics.py        # Reports and confusion matrices
|-- tests/
|   |-- conftest.py
|   |-- test_preprocess.py
|   |-- test_build_features.py
|   `-- test_metrics.py
|-- main.py                          # CLI entry point
|-- pyproject.toml
`-- requirements.txt
```

## Getting Started

### uv (recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package and project manager. Install it once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then set up the project:

```bash
uv sync                  # create .venv and install all dependencies
uv sync --group dev      # also install dev dependencies (jupyter, ruff)
```

Run any command inside the environment:

```bash
uv run python <script>
uv run pytest
uv run jupyter lab
```

### pip (alternative)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running Experiments

Run a single experiment with the default config (`src/configs/experiment.yaml`):

```bash
PYTHONPATH=src uv run python src/scripts/experiment.py
```

Override model or stopword ratio from the CLI:

```bash
PYTHONPATH=src uv run python src/scripts/experiment.py model=logistic_regression stopword_ratio=0.5
```

Sweep all models and ratios in one command (15 runs total):

```bash
PYTHONPATH=src uv run python src/scripts/experiment.py --multirun \
  model=naive_bayes,logistic_regression,random_forest \
  stopword_ratio=0.0,0.2,0.5,0.8,1.0
```

View results in the MLflow UI:

```bash
uv run mlflow ui
```

Then open `http://localhost:5000` in your browser.

## Usage

Train all configured TF-IDF experiments and save the best pipeline to `models/`:

```bash
python main.py --train
```

Evaluate the saved model:

```bash
python main.py --evaluate
```

Train then immediately evaluate:

```bash
python main.py --train --evaluate
```

Run with explicit categories:

```bash
python main.py --train --evaluate --categories sci.space rec.sport.hockey comp.graphics talk.politics.misc
```

## Notebooks

Open notebooks inside the project environment:

```bash
uv run jupyter lab
```

Suggested ownership:

| Notebook | Owner |
| --- | --- |
| `eda.ipynb` | Data / EDA |
| `preprocessing.ipynb` | Data preprocessing |
| `baseline_a_no_stopwords.ipynb` | Baseline A |
| `baseline_b_stopwords.ipynb` | Baseline B |
| `model_machine_learning.ipynb` | Model comparison |
| `analysis.ipynb` | Final analysis |

## Testing

```bash
uv run pytest
```

The tests use small synthetic inputs where possible. Running full training may require downloading the 20 Newsgroups dataset through scikit-learn.

## Team

| Name | Role |
| --- | --- |
| TBD | Team Leader & QA |
| TBD | EDA & Preprocessing |
| TBD | Baseline A |
| TBD | Baseline B |
| TBD | Analysis & Report |
