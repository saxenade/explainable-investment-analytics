# Methodology

EIAF follows a layered explainability approach:

## 1. Global Explainability
Global feature importance is computed using permutation importance to identify
features that most influence model performance at a population level.

## 2. Local Explainability
For individual records, EIAF uses a model-agnostic "what-if" approach by replacing
each feature with a baseline value and measuring the change in model output.

This provides directional insight without relying on model internals.

## 3. Reason Codes
Local feature contributions are translated into human-readable reason codes to
support business, risk, and governance interpretation.

## 4. Stability Monitoring
Optional Population Stability Index (PSI) metrics quantify distributional drift
between training and scoring datasets.

## 5. Model Cards
Each run can generate a Markdown model card summarizing:
- intended use
- data characteristics
- top drivers
- limitations
