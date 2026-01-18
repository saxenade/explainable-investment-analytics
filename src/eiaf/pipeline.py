from __future__ import annotations
from dataclasses import asdict
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from .config import ExplainableConfig
from .explain.permutation import PermutationExplainer
from .reason_codes.generator import ReasonCodeGenerator, ReasonCodeConfig
from .metrics.stability import compute_psi
from .audit.artifacts import ExplanationBundle
from .audit.model_card import ModelCard

class ExplainablePipeline:
    """
    Orchestrates:
      - global explanations
      - local explanations for a sample row
      - reason codes
      - optional stability (PSI)
      - audit-ready artifact bundle (JSON) + model card (markdown)
    """

    def __init__(
        self,
        model,
        feature_names: List[str],
        config: ExplainableConfig = ExplainableConfig(),
        reason_templates: Optional[Dict[str, tuple]] = None,
    ):
        self.model = model
        self.feature_names = feature_names
        self.config = config
        self.explainer = PermutationExplainer(
            n_repeats=config.n_repeats,
            random_state=config.random_state,
            task=config.task,
        )
        self.reason_gen = ReasonCodeGenerator(
            templates=reason_templates or {},
            config=ReasonCodeConfig(top_k=min(config.top_k, 8)),
        )

    def explain_batch(
        self,
        X: Any,
        y_true: Optional[Any] = None,
        score_X_for_stability: Optional[Any] = None,
        sample_index: int = 0,
    ) -> ExplanationBundle:
        X_np = self._to_numpy(X)
        y_np = None if y_true is None else np.asarray(y_true)

        if y_np is None:
            # For global permutation importance we need y; otherwise skip global
            global_fi = {name: 0.0 for name in self.feature_names}
        else:
            global_exp = self.explainer.explain_global(self.model, X_np, y_np, self.feature_names)
            global_fi = global_exp.feature_importance

        # Baseline for local explanations: median feature values
        baseline = np.median(X_np, axis=0)

        sample_index = int(np.clip(sample_index, 0, max(X_np.shape[0] - 1, 0)))
        local = self.explainer.explain_local(
            self.model,
            X_np[sample_index],
            self.feature_names,
            baseline=baseline,
        )
        local_contribs = local.feature_contributions

        reason_codes = self.reason_gen.generate(local_contribs)

        data_summary = self._data_summary(X_np)
        model_meta = self._model_meta()

        stability = None
        if score_X_for_stability is not None:
            score_np = self._to_numpy(score_X_for_stability)
            psi = compute_psi(X_np, score_np, self.feature_names, bins=self.config.psi_bins)
            stability = {"psi": psi, "bins": self.config.psi_bins}

        # Build model card markdown
        top_features = list(global_fi.keys())[: self.config.top_k]
        mc = ModelCard(
            title=f"{self.config.model_name or 'Model'} — Explainability Model Card",
            model_type=type(self.model).__name__,
            intended_use="Explain tabular investment/credit-risk models with audit-ready artifacts for governance and review.",
            limitations="Permutation and what-if local explanations are approximations; validate with domain review and stability monitoring.",
            data_summary=f"Rows: {data_summary['n_rows']}, Columns: {data_summary['n_cols']}. Missing values are summarized in artifacts.",
            top_features=top_features,
            evaluation="Provide evaluation metrics (AUC/F1/RMSE) from your training pipeline. This framework records explainability artifacts; it does not enforce a single metric.",
            notes="This bundle is designed to be stored with model runs to support traceability.",
        )

        bundle = ExplanationBundle(
            schema_version=self.config.artifact_version,
            model=model_meta,
            data_summary=data_summary,
            global_explanations={"permutation_importance": dict(list(global_fi.items())[: self.config.top_k])},
            local_explanations={
                "sample_index": sample_index,
                "what_if_contributions": dict(list(local_contribs.items())[: self.config.top_k]),
            },
            reason_codes=[asdict(rc) for rc in reason_codes],
            stability=stability,
            model_card_markdown=mc.to_markdown(),
        )
        return bundle

    def _to_numpy(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.values.astype(float)
        X_np = np.asarray(X)
        if X_np.ndim != 2:
            raise ValueError("X must be 2D array-like")
        return X_np.astype(float)

    def _data_summary(self, X_np: np.ndarray) -> Dict[str, Any]:
        # Missing inferred by NaN
        missing = np.isnan(X_np).sum(axis=0).tolist()
        missing_by_feature = {self.feature_names[i]: int(missing[i]) for i in range(len(self.feature_names))}
        return {
            "n_rows": int(X_np.shape[0]),
            "n_cols": int(X_np.shape[1]),
            "missing_by_feature": missing_by_feature,
        }

    def _model_meta(self) -> Dict[str, Any]:
        return {
            "name": self.config.model_name or "unnamed-model",
            "type": type(self.model).__name__,
            "task": self.config.task,
            "framework": "scikit-learn",
            "artifact_schema_version": self.config.artifact_version,
        }
