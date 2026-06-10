# Module 01 - Text Classification

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

The original project brief is stored at `docs/Research-Module-1.pdf`.

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

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Usage

Train all configured TF-IDF experiments and save the best pipeline to `models/`:

```powershell
python main.py --train
```

Evaluate the saved model:

```powershell
python main.py --evaluate
```

Train then immediately evaluate:

```powershell
python main.py --train --evaluate
```

Run with explicit categories:

```powershell
python main.py --train --evaluate --categories sci.space rec.sport.hockey comp.graphics talk.politics.misc
```

## Notebooks

Open notebooks inside the project environment:

```powershell
jupyter notebook
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

```powershell
python -m pytest
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
