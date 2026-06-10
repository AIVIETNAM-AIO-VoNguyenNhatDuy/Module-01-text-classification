# Experiment Plan

## Baselines

| ID | Vectorizer | Stopwords | Models |
| --- | --- | --- | --- |
| A | TF-IDF | Keep stopwords | Naive Bayes, Logistic Regression, Random Forest |
| B | TF-IDF | Remove stopwords | Naive Bayes, Logistic Regression, Random Forest |
| C | CountVectorizer | Optional | Naive Bayes, Logistic Regression |

## Stopword Ratios

Use the same train/test split and categories for each setting:

- `0.0`: keep stopwords
- `0.2`: remove 20% of stopword list
- `0.5`: remove 50% of stopword list
- `0.8`: remove 80% of stopword list
- `1.0`: remove all configured stopwords

## Required Outputs

Each experiment should produce:

- `metrics.csv`
- `classification_report.json`
- confusion matrix image
- short notes about observed changes

## Analysis Focus

- Compare macro F1 and weighted F1, not only accuracy.
- Check per-class deltas between no-stopword and stopword-removal runs.
- Highlight whether stopword removal causes uneven effects across classes.
- Inspect misclassified examples containing negation words such as `not`, `no`, and `very`.

