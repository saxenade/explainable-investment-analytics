# Explainable Investment Analytics Framework (EIAF) — Overview

EIAF is a lightweight framework designed to standardize explainability outputs for
tabular machine learning models used in investment and credit-risk analytics.

Unlike visualization-focused explainability tools, EIAF emphasizes **audit-ready,
machine-readable artifacts** that can be stored, reviewed, and compared across
model runs.

## Design Goals
- Model-agnostic (scikit-learn first)
- Repeatable and deterministic outputs
- Human-readable reason codes
- Governance- and audit-friendly artifacts

## Target Use Cases
- Credit risk models
- Investment analytics
- Portfolio monitoring
- Model risk management (MRM)
- Regulatory and internal audits


┌────────────────────────┐
│   Training Dataset     │
│  (Tabular Finance Data)│
└──────────┬─────────────┘
           │
           ▼
┌────────────────────────┐
│  ML Model (scikit-learn)│
│  • Classifier/Regressor│
└──────────┬─────────────┘
           │
           ▼
┌──────────────────────────────┐
│ Explainable Pipeline (EIAF)  │
│                              │
│ 1. Global Explainability     │
│    • Permutation Importance │
│                              │
│ 2. Local Explainability      │
│    • What-if Contributions  │
│                              │
│ 3. Reason Code Engine        │
│    • Human-readable Drivers │
│                              │
│ 4. Stability Monitoring      │
│    • PSI Drift Metrics       │
│                              │
│ 5. Model Card Generator      │
│    • Markdown Summary        │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│ Audit-Ready Artifact Bundle      │
│                                  │
│ • explainability_bundle.json     │
│ • model_card.md                  │
│                                  │
│ Stored per model run for:        │
│ • Governance review              │
│ • Audit traceability             │
│ • Model comparison               │
└──────────────────────────────────┘
