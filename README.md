# Explainable Investment Analytics Framework (EIAF)

EIAF is a lightweight Python framework for generating **audit-ready explainability artifacts** for **tabular investment / credit-risk models** built with scikit-learn.

It produces:
- **Global explanations** (permutation importance)
- **Local “what-if” explanations** for individual records
- **Reason codes** (human-readable drivers)
- **Stability metrics** (PSI drift checks)
- A **Model Card** (Markdown) + a structured **artifact bundle** (JSON)

## Why this exists
In regulated finance, teams often need consistent, repeatable explainability outputs for:
- model risk management (MRM)
- governance reviews
- audit traceability
- stakeholder communication

EIAF focuses on **standardized artifacts**, not just plots.

## Install (editable)
```bash
pip install -e ".[dev]"


## How to cite this framework

If you use EIAF in research, documentation, or production workflows, please cite:

**Explainable Investment Analytics Framework (EIAF)** — Deepak Saxena, v0.1.0, 2026.  
GitHub repository: `https://github.com/saxenade/explainable-investment-analytics`

### BibTeX
```bibtex
@software{saxena_eiaf_2026,
  author  = {Saxena, Deepak},
  title   = {Explainable Investment Analytics Framework (EIAF)},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/saxenade/explainable-investment-analytics}
}
