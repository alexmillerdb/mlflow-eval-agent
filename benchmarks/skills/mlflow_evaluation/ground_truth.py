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
        """Convert to MLflow evaluation dataset format for YAML/list data.

        For offline evaluation with pre-computed outputs.
        outputs can be string or dict - scorers handle both via get_response_text().
        """
        return {
            "inputs": {"prompt": self.prompt},
            "outputs": {"response": self.response},  # Dict format works with our scorers
            "expectations": {"code_executes": True},
        }

    def to_uc_record(self) -> dict:
        """
        Convert to MLflow UC dataset format.

        NOTE: Databricks UC datasets do NOT store 'outputs' column!
        For offline eval with pre-computed outputs, use YAML source instead.
        UC datasets are for online eval with predict_fn.

        We store outputs in expectations for UC datasets so they can be
        retrieved during evaluation if needed.
        """
        return {
            "inputs": {"prompt": self.prompt},
            # outputs stored in expectations since UC doesn't support outputs column
            "expectations": {
                "code_executes": True,
                "response": self.response,  # Store response here for UC
                "id": self.id,
                "tags": self.tags,
                "created_at": self.created_at.isoformat() if self.created_at else None,
                "execution_success": self.execution_success,
            }
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
