# mlflow-evaluation Skill Benchmark

Benchmark suite for evaluating the `mlflow-evaluation` skill's code generation capabilities.

## Quick Start

### 1. Generate Ground Truth Examples

Use the `/skill-test` command in Claude Code to generate verified working examples:

```bash
# Start Claude Code and invoke the skill-test command
claude

> /skill-test mlflow-evaluation
```

The workflow:
1. Claude loads the mlflow-evaluation skill context
2. You provide a test prompt (e.g., "Generate an evaluation script for a RAG agent")
3. Claude generates code using the skill's knowledge
4. Code blocks are extracted and executed
5. If successful, you're prompted to save as ground truth

### 2. Example Test Prompts

Use these prompts to build a diverse ground truth dataset:

| Category | Example Prompt |
|----------|---------------|
| **Evaluation Scripts** | "Generate an MLflow evaluation script for a RAG agent" |
| **Custom Scorers** | "Create a custom scorer that checks for PII in responses" |
| **Guidelines** | "Write a Guidelines scorer for code quality evaluation" |
| **Dataset Building** | "Show how to build an evaluation dataset from MLflow traces" |
| **Built-in Scorers** | "Demonstrate using Safety, Correctness, and RelevanceToQuery scorers" |
| **Custom Judges** | "Create a custom judge using make_judge() for API correctness" |

### 3. Ground Truth Storage

Examples are stored in YAML format:

```
benchmarks/skills/mlflow-evaluation/
├── ground_truth.yaml   # Verified examples (data)
├── ground_truth.py     # Dataclass + loader (code)
├── __init__.py
└── README.md
```

**YAML format:**
```yaml
examples:
  - id: gt-001
    prompt: "Generate an evaluation script for a RAG agent"
    response: |
      Here's a complete MLflow evaluation script...
      ```python
      import mlflow.genai
      # ...
      ```
    execution_success: true
    created_at: "2025-01-10T13:19:00"
    tags: [evaluation, rag, scorers]
```

### 4. Load Ground Truth in Python

```python
from benchmarks.skills.mlflow_evaluation import GROUND_TRUTH, GroundTruthExample

# List all examples
for ex in GROUND_TRUTH:
    print(f"{ex.id}: {ex.prompt[:50]}...")

# Convert to MLflow evaluation format
eval_data = [ex.to_eval_record() for ex in GROUND_TRUTH]
```

## Current Ground Truth

| ID | Prompt | Tags |
|----|--------|------|
| gt-001 | Generate an evaluation script for a RAG agent | evaluation, rag, scorers |

**Goal:** Build 10-20 verified examples covering different use cases.

## Running Evaluations (Phase 3-5)

*Coming soon* - The benchmark framework will support:

```bash
# Run evaluation on ground truth examples
python -m scripts.skill_benchmark run --skill mlflow-evaluation

# Compare skill versions
python -m scripts.skill_benchmark compare --version-a 1.0 --version-b 2.0

# Check for regressions
python -m scripts.skill_benchmark regression --skill mlflow-evaluation
```

## Directory Structure

```
benchmarks/
├── BENCHMARK_PLAN.md              # Full framework design
└── skills/
    └── mlflow-evaluation/
        ├── README.md              # This file
        ├── __init__.py
        ├── ground_truth.yaml      # Verified examples
        ├── ground_truth.py        # Dataclass + loader
        ├── config.py              # (Phase 3-5) Quality gates
        ├── test_cases.py          # (Phase 3-5) Test definitions
        └── scorers.py             # (Phase 3-5) Custom scorers
```

## Related Files

- **Skill definition:** `.claude/skills/mlflow-evaluation/SKILL.md`
- **Skill-test command:** `.claude/skills/skill-test/SKILL.md`
- **Full plan:** `benchmarks/BENCHMARK_PLAN.md`
