from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass(frozen=True)
class ModelCard:
    title: str
    model_type: str
    intended_use: str
    limitations: str
    data_summary: str
    top_features: List[str]
    evaluation: str
    notes: Optional[str] = None

    def to_markdown(self) -> str:
        tf = "\n".join([f"- {x}" for x in self.top_features])
        md = f"""# {self.title}

## Model
- **Type:** {self.model_type}

## Intended Use
{self.intended_use}

## Data Summary
{self.data_summary}

## Evaluation
{self.evaluation}

## Top Features (Global)
{tf}

## Limitations
{self.limitations}
"""
        if self.notes:
            md += f"\n## Notes\n{self.notes}\n"
        return md
