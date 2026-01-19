# Audit Artifact Schema

EIAF produces a structured explainability bundle intended to be stored alongside
model runs for traceability and audit review.

## Artifact Structure (v1)

```json
{
  "schema_version": "1.0",
  "model": {
    "name": "model-name",
    "type": "RandomForestClassifier",
    "task": "classification",
    "framework": "scikit-learn"
  },
  "data_summary": {
    "n_rows": 10000,
    "n_cols": 25,
    "missing_by_feature": {}
  },
  "global_explanations": {
    "permutation_importance": {}
  },
  "local_explanations": {
    "sample_index": 0,
    "what_if_contributions": {}
  },
  "reason_codes": [],
  "stability": {
    "psi": {}
  }
}
