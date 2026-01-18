from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, r2_score

from .base import GlobalExplanation, LocalExplanation

@dataclass(frozen=True)
class PermutationExplainer:
    """
    Model-agnostic baseline explainer.

    Global: permutation importance using a simple scoring function.
    Local: "what-if" impact proxy by replacing each feature with baseline value
           and measuring prediction change.

    Notes:
    - Local explanation here is an approximation; it's still useful for reason codes
      and audit artifacts without relying on SHAP.
    """

    n_repeats: int = 5
    random_state: int = 42
    task: str = "classification"  # "classification" or "regression"

    def _score(self, model, X, y) -> float:
        if self.task == "regression":
            preds = model.predict(X)
            return float(r2_score(y, preds))
        # classification
        preds = model.predict(X)
        return float(accuracy_score(y, preds))

    def explain_global(self, model, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> GlobalExplanation:
        result = permutation_importance(
            model,
            X,
            y,
            scoring=None,  # use estimator default; we keep our own local scoring for metadata
            n_repeats=self.n_repeats,
            random_state=self.random_state,
        )
        importances = result.importances_mean
        fi = {feature_names[i]: float(importances[i]) for i in range(len(feature_names))}
        # Sort desc
        fi = dict(sorted(fi.items(), key=lambda kv: kv[1], reverse=True))
        return GlobalExplanation(feature_importance=fi)

    def explain_local(
        self,
        model,
        X_row: np.ndarray,
        feature_names: List[str],
        baseline: np.ndarray,
    ) -> LocalExplanation:
        # baseline must be 1D array of same shape as X_row
        x = X_row.reshape(1, -1)
        base_pred = self._predict_scalar(model, x)

        contributions = {}
        for j, name in enumerate(feature_names):
            x_perturbed = x.copy()
            x_perturbed[0, j] = baseline[j]
            pred_perturbed = self._predict_scalar(model, x_perturbed)
            # signed delta: positive means this feature pushes prediction up (relative to baseline)
            contributions[name] = float(base_pred - pred_perturbed)

        # Sort by absolute magnitude desc
        contributions = dict(sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True))
        return LocalExplanation(feature_contributions=contributions)

    def _predict_scalar(self, model, X: np.ndarray) -> float:
        # For classification, prefer probability of positive class if available
        if hasattr(model, "predict_proba") and self.task == "classification":
            proba = model.predict_proba(X)
            # binary: take positive class prob
            if proba.shape[1] >= 2:
                return float(proba[0, 1])
            return float(proba[0, 0])
        pred = model.predict(X)
        return float(pred[0])
