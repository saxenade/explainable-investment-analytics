import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from eiaf.pipeline import ExplainablePipeline
from eiaf.config import ExplainableConfig

def main():
    X, y = make_classification(
        n_samples=1200, n_features=12, n_informative=6, n_redundant=2, random_state=42
    )
    feature_names = [f"f{i:02d}" for i in range(X.shape[1])]

    X_train, X_score, y_train, y_score = train_test_split(X, y, test_size=0.25, random_state=42)

    model = RandomForestClassifier(n_estimators=250, random_state=42)
    model.fit(X_train, y_train)

    config = ExplainableConfig(
        top_k=6,
        n_repeats=5,
        random_state=42,
        task="classification",
        model_name="demo-credit-risk-model",
    )

    # Optional templates for nicer reason codes
    templates = {
        "f00": ("Higher f00 increases risk score.", "Lower f00 decreases risk score."),
        "f01": ("Higher f01 increases risk score.", "Lower f01 decreases risk score."),
    }

    pipe = ExplainablePipeline(model=model, feature_names=feature_names, config=config, reason_templates=templates)
    bundle = pipe.explain_batch(X_train, y_true=y_train, score_X_for_stability=X_score, sample_index=0)

    os.makedirs("artifacts", exist_ok=True)
    bundle.save_json("artifacts/explainability_bundle.json")
    bundle.save_model_card("artifacts/model_card.md")

    print("Wrote:")
    print(" - artifacts/explainability_bundle.json")
    print(" - artifacts/model_card.md")

if __name__ == "__main__":
    main()
