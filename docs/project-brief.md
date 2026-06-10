# Project Brief

## Topic

Impact of stopword removal on text classification.

## Background

The project uses the 20 Newsgroups dataset, a classic text classification dataset with about 20,000 posts from 20 Usenet discussion groups. Each document has one topic label.

For faster experiments and clearer analysis, use a subset of 2 to 4 classes first, such as:

- `sci.space`
- `rec.sport.hockey`
- `comp.graphics`
- `talk.politics.misc`

## Research Questions

- Does stopword removal improve text classification performance?
- How does stopword removal affect accuracy and F1-score?
- Is there a meaningful difference between keeping and removing stopwords when using TF-IDF with Logistic Regression or Naive Bayes?
- Which model is most stable when stopwords are changed?
- How do different stopword removal levels, such as `0%`, `20%`, `50%`, and `80%`, affect model performance?
- Does stopword removal help all classes equally, or improve some classes while hurting others?

## Minimum Pipeline

1. Load 20 Newsgroups data.
2. Clean text:
   - lowercase
   - remove punctuation
   - remove special characters
   - optionally remove numbers
   - tokenize
   - optionally compare stemming
3. Build experiment branches:
   - Baseline A: TF-IDF without stopword removal
   - Baseline B: TF-IDF with stopword removal
   - Optional Baseline C: CountVectorizer
4. Train models:
   - Naive Bayes
   - Logistic Regression
   - Random Forest
5. Evaluate:
   - accuracy
   - precision
   - recall
   - F1-score
   - confusion matrix
6. Perform error analysis:
   - cases where `not`, `no`, or `very` affect meaning
   - classes that are most often confused
   - per-class performance gains and losses

