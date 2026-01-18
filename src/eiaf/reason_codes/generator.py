from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass(frozen=True)
class ReasonCode:
    code: str
    message: str
    feature: str
    direction: str  # "up" or "down"
    strength: float

@dataclass(frozen=True)
class ReasonCodeConfig:
    top_k: int = 5
    positive_threshold: float = 0.0  # for signed contributions
    prefix: str = "RC"

class ReasonCodeGenerator:
    """
    Turns local feature contributions into human-readable reason codes.

    contributions: feature -> signed impact proxy
      - positive means feature pushes prediction UP (relative to baseline)
      - negative means pushes DOWN
    """

    def __init__(self, templates: Optional[Dict[str, Tuple[str, str]]] = None, config: ReasonCodeConfig = ReasonCodeConfig()):
        # templates[feature] = (message_when_up, message_when_down)
        self.templates = templates or {}
        self.config = config

    def generate(self, contributions: Dict[str, float]) -> List[ReasonCode]:
        items = list(contributions.items())
        # already sorted by abs in our pipeline; but ensure
        items.sort(key=lambda kv: abs(kv[1]), reverse=True)

        out: List[ReasonCode] = []
        k = 0
        for feature, val in items:
            if k >= self.config.top_k:
                break
            direction = "up" if val >= self.config.positive_threshold else "down"
            msg = self._format_message(feature, direction, val)
            out.append(
                ReasonCode(
                    code=f"{self.config.prefix}-{k+1:02d}",
                    message=msg,
                    feature=feature,
                    direction=direction,
                    strength=float(abs(val)),
                )
            )
            k += 1
        return out

    def _format_message(self, feature: str, direction: str, val: float) -> str:
        if feature in self.templates:
            up_msg, down_msg = self.templates[feature]
            return up_msg if direction == "up" else down_msg

        # Generic fallback
        if direction == "up":
            return f"{feature} is a key driver increasing the model score (relative to baseline)."
        return f"{feature} is a key driver decreasing the model score (relative to baseline)."
