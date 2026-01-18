from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from .model_card import ModelCard
from ..utils.serialization import to_jsonable, write_json

@dataclass
class ExplanationBundle:
    schema_version: str
    model: Dict[str, Any]
    data_summary: Dict[str, Any]
    global_explanations: Dict[str, Any]
    local_explanations: Dict[str, Any]
    reason_codes: List[Dict[str, Any]]
    stability: Optional[Dict[str, Any]] = None
    model_card_markdown: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)

    def save_json(self, path: str) -> None:
        write_json(self.to_dict(), path)

    def save_model_card(self, path: str) -> None:
        if not self.model_card_markdown:
            raise ValueError("model_card_markdown is empty. Generate it via pipeline with a model card config.")
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_card_markdown)
