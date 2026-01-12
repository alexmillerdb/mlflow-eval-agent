"""Verified working examples generated via /skill-test."""

import yaml
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class GroundTruthExample:
    id: str
    prompt: str
    response: str
    execution_success: bool
    created_at: datetime
    tags: list[str] = field(default_factory=list)

    def to_eval_record(self) -> dict:
        """Convert to MLflow evaluation dataset format."""
        return {
            "inputs": {"prompt": self.prompt},
            "outputs": {"response": self.response},
            "expectations": {"code_executes": True},
        }


def load_ground_truth() -> list[GroundTruthExample]:
    """Load ground truth examples from YAML file."""
    yaml_path = Path(__file__).parent / "ground_truth.yaml"
    if not yaml_path.exists():
        return []

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    examples = []
    for ex in data.get("examples", []):
        # Parse datetime string if present
        if isinstance(ex.get("created_at"), str):
            ex["created_at"] = datetime.fromisoformat(ex["created_at"])
        examples.append(GroundTruthExample(**ex))

    return examples


# Load examples at import time
GROUND_TRUTH: list[GroundTruthExample] = load_ground_truth()
