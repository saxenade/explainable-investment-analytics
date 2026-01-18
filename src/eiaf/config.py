from dataclasses import dataclass
from typing import Optional, Literal

TaskType = Literal["classification", "regression"]

@dataclass(frozen=True)
class ExplainableConfig:
    """
    Configuration for explanation generation and audit artifacts.

    top_k: number of top features to keep for global & local explanations
    n_repeats: permutation repeats (higher -> more stable, slower)
    random_state: reproducibility
    task: classification or regression (affects some metadata only in v1)
    """
    top_k: int = 8
    n_repeats: int = 5
    random_state: int = 42
    task: TaskType = "classification"
    psi_bins: int = 10
    artifact_version: str = "1.0"
    model_name: Optional[str] = None
