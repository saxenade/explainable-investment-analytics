from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

@dataclass(frozen=True)
class PSIFeatureResult:
    feature: str
    psi: float

def _psi_for_feature(train: np.ndarray, score: np.ndarray, bins: int = 10) -> float:
    # robust bin edges using quantiles from train
    eps = 1e-8
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.quantile(train, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf

    train_counts, _ = np.histogram(train, bins=edges)
    score_counts, _ = np.histogram(score, bins=edges)

    train_pct = np.clip(train_counts / max(train_counts.sum(), 1), eps, None)
    score_pct = np.clip(score_counts / max(score_counts.sum(), 1), eps, None)

    psi = float(np.sum((score_pct - train_pct) * np.log(score_pct / train_pct)))
    return psi

def compute_psi(train_X: np.ndarray, score_X: np.ndarray, feature_names: List[str], bins: int = 10) -> Dict[str, float]:
    if train_X.shape[1] != score_X.shape[1]:
        raise ValueError("train_X and score_X must have same number of columns")
    if len(feature_names) != train_X.shape[1]:
        raise ValueError("feature_names length must match X columns")

    out: Dict[str, float] = {}
    for j, name in enumerate(feature_names):
        out[name] = _psi_for_feature(train_X[:, j], score_X[:, j], bins=bins)
    return out
