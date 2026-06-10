# Team Workflow

## Suggested Ownership

| Area | Responsibility |
| --- | --- |
| Data | Load dataset, verify categories, document preprocessing |
| Baseline A | TF-IDF without stopword removal |
| Baseline B | TF-IDF with stopword removal and ratios |
| Analysis | Compare metrics, confusion matrices, per-class effects |

## Working Rules

- Work on feature branches, not directly on `main`.
- Keep notebooks for exploration; move reusable code into `src/`.
- Do not commit downloaded raw datasets or generated model artifacts.
- Every experiment result should include the command/config used to create it.
- Use pull requests for review before merging.

## Recommended PR Size

Keep PRs focused:

- one preprocessing change
- one experiment setup
- one analysis/report update
- one bug fix

