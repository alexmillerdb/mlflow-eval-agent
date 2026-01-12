# Ground Truth Format

## File Locations

```
benchmarks/skills/<skill-name>/
├── ground_truth.yaml   # Data storage (append new examples here)
└── ground_truth.py     # Dataclass + YAML loader
```

## YAML Schema

```yaml
# ground_truth.yaml
examples:
  - id: gt-001
    prompt: "The test prompt"
    response: |
      Full response with code...

      ```python
      import mlflow.genai
      # ... code ...
      ```
    execution_success: true
    created_at: "2025-01-10T13:19:00"
    tags:
      - evaluation
      - rag
```

## Python Dataclass

```python
# ground_truth.py
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class GroundTruthExample:
    id: str                              # e.g., "gt-001"
    prompt: str                          # The test prompt
    response: str                        # Full response with code
    execution_success: bool              # True if all code executed
    created_at: datetime                 # When created
    tags: list[str] = field(default_factory=list)  # e.g., ["evaluation", "rag"]

    def to_eval_record(self) -> dict:
        """Convert to MLflow evaluation dataset format."""
        return {
            "inputs": {"prompt": self.prompt},
            "outputs": {"response": self.response},
            "expectations": {"code_executes": True},
        }
```

## Adding a New Example

Append to the `examples` list in `ground_truth.yaml`:

```yaml
examples:
  - id: gt-001
    # ... existing example ...

  - id: gt-002
    prompt: "Create a custom scorer that checks for PII"
    response: |
      Here's a custom scorer...

      ```python
      from mlflow.genai.scorers import scorer
      # ... code ...
      ```
    execution_success: true
    created_at: "2025-01-10T14:30:00"
    tags:
      - scorer
      - custom
```

## ID Convention

- Format: `gt-XXX` (e.g., gt-001, gt-002)
- Auto-increment based on existing IDs in the file
- Parse existing YAML to find the next available number

## Creating New Ground Truth Files

If `benchmarks/skills/<skill-name>/` doesn't exist:

1. Create the directory structure
2. Create `ground_truth.yaml`:
   ```yaml
   # Verified working examples generated via /skill-test
   examples: []
   ```

3. Create `ground_truth.py`:
   ```python
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
           return {
               "inputs": {"prompt": self.prompt},
               "outputs": {"response": self.response},
               "expectations": {"code_executes": True},
           }


   def load_ground_truth() -> list[GroundTruthExample]:
       yaml_path = Path(__file__).parent / "ground_truth.yaml"
       if not yaml_path.exists():
           return []

       with open(yaml_path) as f:
           data = yaml.safe_load(f)

       examples = []
       for ex in data.get("examples", []):
           if isinstance(ex.get("created_at"), str):
               ex["created_at"] = datetime.fromisoformat(ex["created_at"])
           examples.append(GroundTruthExample(**ex))

       return examples


   GROUND_TRUTH: list[GroundTruthExample] = load_ground_truth()
   ```

4. Create `__init__.py`:
   ```python
   """Benchmark for <skill-name> skill."""
   from .ground_truth import GROUND_TRUTH, GroundTruthExample
   __all__ = ["GROUND_TRUTH", "GroundTruthExample"]
   ```
