from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Protocol
import numpy as np

@dataclass(frozen=True)
class GlobalExplanation:
    feature_importance: Dict[str, float]  # name -> importance score

@dataclass(frozen=True)
class LocalExplanation:
    # For a single row: feature -> signed impact proxy (model-agnostic)
    feature_contributions: Dict[str, float]

class Explainer(Protocol):
    def explain_global(self, model, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> GlobalExplanation:
        ...

    def explain_local(self, model, X_row: np.ndarray, feature_names: List[str], baseline: np.ndarray) -> LocalExplanation:
        ...
