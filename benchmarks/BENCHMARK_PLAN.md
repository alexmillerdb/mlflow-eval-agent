# SKILL Evaluation Framework with MLflow

## Overview

A framework for systematically measuring and evaluating Claude Code SKILLS using **MLflow 3 GenAI evaluation** logged to **Databricks experiments**. Enables benchmarking, version comparison, and regression detection for skills.

### MLflow 3 Requirements (from mlflow-evaluation skill)

```python
# CRITICAL: Use MLflow 3 GenAI APIs - NOT deprecated mlflow.evaluate()
from mlflow.genai.scorers import scorer, Guidelines, Safety, Correctness
from mlflow.entities import Feedback, Trace, SpanType
import mlflow.genai

# Configure Databricks tracking
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Shared/skill-benchmarks/<skill-name>")

# Run evaluation with MLflow 3 API
results = mlflow.genai.evaluate(
    data=eval_dataset,      # Must have nested {"inputs": {...}} structure
    scorers=[...],
    model_id="skill-benchmark"
)
```

**Key gotchas avoided:**
- Uses `mlflow.genai.evaluate()` NOT `mlflow.evaluate()` (deprecated)
- Data uses nested `{"inputs": {...}}` structure
- Scorers return `Feedback` objects or primitives
- Results logged to Databricks experiment automatically

## Core Components

### 1. Benchmark Structure (separate from skills)

```
benchmarks/
└── skills/
    └── <skill-name>/         # Mirrors .claude/skills/<skill-name>
        ├── __init__.py
        ├── config.py         # Benchmark config: quality gates, settings
        ├── test_cases.py     # Test cases as Python dicts/dataclasses
        ├── scorers.py        # Custom scorers and judges
        └── baselines/        # Saved metrics for regression detection
            └── current.json

# Skills remain clean:
.claude/skills/<skill-name>/
├── SKILL.md              # Skill definition (unchanged)
└── references/           # Reference docs (unchanged)
```

All Python - no YAML parsing, scorers and test cases in same language.

### 2. Test Case Schema (Python)

```python
# benchmarks/skills/mlflow-evaluation/test_cases.py
from dataclasses import dataclass

@dataclass
class TestCase:
    id: str
    name: str
    category: str
    prompt: str
    expectations: dict
    tags: list[str] = None

TEST_CASES = [
    TestCase(
        id="eval-001",
        name="Basic evaluation script",
        category="code-generation",
        prompt="Generate an MLflow evaluation script for a RAG agent...",
        expectations={
            "expected_patterns": ["mlflow.genai.evaluate", "@scorer"],
            "expected_tools": ["Read"],
            "expected_facts": ["Uses mlflow.genai.evaluate() function"],
            # For Guidelines scorer
            "guidelines": [
                "Code should be complete and runnable",
                "Should use MLflow 3 GenAI APIs",
            ],
        },
        tags=["evaluation", "rag"],
    ),
]
```

### 3. Trace Collection (Manual Workflow)

**Step 1:** Configure Databricks and enable tracing for Claude Code:
```bash
# Databricks connection
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="your-token"
export MLFLOW_TRACKING_URI="databricks"

# Create/set experiment for skill traces
export MLFLOW_EXPERIMENT_NAME="/Shared/skill-traces/mlflow-evaluation"

# Enable Claude Code tracing
mlflow autolog claude ~/my-project
```

**Step 2:** Run test cases manually in Claude Code, tagging each session:
```bash
# Before each test, set tags via environment or session start
claude --skill mlflow-evaluation
# Then paste the test case prompt
```

**Step 3:** After collection, tag traces with metadata:
```bash
mlflow traces set-tag --trace-id <id> --key skill.name --value mlflow-evaluation
mlflow traces set-tag --trace-id <id> --key skill.version --value 1.0.0
mlflow traces set-tag --trace-id <id> --key skill.test_case_id --value eval-001
```

This manual approach gives full control over test execution while keeping the MVP simple.

### 4. Scorers, Guidelines & Custom Judges

#### Built-in Scorers
| Scorer | Purpose |
|--------|---------|
| `Safety` | No harmful content (always include) |
| `RelevanceToQuery` | Response addresses task |
| `Correctness` | Matches expected_facts/expected_response |
| `ExpectationsGuidelines` | Per-row guideline evaluation |

#### Guidelines Scorers (LLM-as-judge with natural language criteria)
```python
from mlflow.genai.scorers import Guidelines

# Skill-specific quality guidelines
code_quality_guidelines = Guidelines(
    name="code_quality",
    guidelines="""
    Evaluate the generated code:
    1. Code is complete and runnable (no placeholders)
    2. All imports are included
    3. Follows MLflow 3 GenAI API patterns
    4. Includes error handling where appropriate
    """
)

skill_adherence_guidelines = Guidelines(
    name="skill_adherence",
    guidelines="""
    Evaluate if the response properly uses the skill:
    1. References skill documentation when relevant
    2. Avoids common gotchas documented in the skill
    3. Follows patterns from skill references
    """
)
```

#### Custom Judges (via make_judge)
```python
from mlflow.genai.judges import make_judge

# Custom judge for API correctness
api_correctness_judge = make_judge(
    name="api_correctness",
    judge_prompt="""
    You are evaluating if code uses the correct MLflow 3 GenAI APIs.

    Check for these mistakes:
    - Using mlflow.evaluate() instead of mlflow.genai.evaluate()
    - Flat data structure instead of nested {"inputs": {...}}
    - Missing @scorer decorator on custom scorers
    - Guidelines without 'name' parameter

    Input: {{ inputs.prompt }}
    Output: {{ outputs.response }}

    Return 'yes' if APIs are used correctly, 'no' if mistakes found.
    """,
    output_type="yes_or_no"
)
```

#### Custom Scorers (Python logic)
| Scorer | Purpose |
|--------|---------|
| `code_syntax_valid` | AST parsing - generated code is valid Python |
| `uses_correct_api_patterns` | Regex/pattern matching against GOTCHAS |
| `skill_references_used` | Trace-based - did skill read its references? |
| `tool_selection_accuracy` | Trace-based - correct tools were called |
| `skill_execution_latency` | Performance tracking with aggregations |

#### Three-Tier Scoring Strategy

For skill benchmarks, use a tiered approach that matches scorer types to evaluation needs:

| Tier | Scorer Type | Use For | Example |
|------|-------------|---------|---------|
| 1 | Pattern-matching (`@scorer`) | Deterministic checks | `code_syntax_valid`, `uses_genai_evaluate` |
| 2 | `make_judge` with curated refs | API correctness grounded in skill docs | `api_correctness_judge` with KEY_GOTCHAS |
| 3 | `Guidelines` | Subjective quality assessment | `code_quality`, `response_completeness` |

**Why this approach:**
- **Tier 1** catches objective errors deterministically (no LLM variability)
- **Tier 2** uses LLM judgment but grounded in actual skill documentation
- **Tier 3** handles subjective quality where LLM judgment is appropriate

**Curated References for make_judge:**

The key insight is that `make_judge` can be grounded in skill documentation by extracting
key points from reference files (e.g., GOTCHAS.md) and embedding them directly in the
judge instructions:

```python
# Curated from .claude/skills/mlflow-evaluation/references/GOTCHAS.md
KEY_GOTCHAS = """
- Use mlflow.genai.evaluate() NOT mlflow.evaluate() (deprecated)
- Data requires nested structure: {"inputs": {"query": "..."}}
- predict_fn receives unpacked kwargs: predict_fn(**inputs)
- Valid aggregations: min, max, mean, median, variance, p90 (NOT p50, p99, sum)
"""

api_correctness_judge = make_judge(
    name="api_correctness",
    instructions=f"""
    Evaluate if the generated code follows MLflow 3 GenAI best practices.

    Reference (from skill documentation):
    {KEY_GOTCHAS}

    Code to evaluate: {{{{ outputs }}}}

    Return 'yes' if correct, 'no' if any gotchas are violated.
    """
)
```

This gives the LLM judge the "ground truth" without sending entire reference files,
providing flexibility while staying grounded in authoritative documentation.

### 5. Evaluation Pipeline (MLflow 3 + Databricks)

```
Load benchmark → Collect traces → Build dataset → Run scorers → Log to Databricks
```

Key class: `SkillEvaluationPipeline`
- `collect_traces(version)` - Get traces tagged with skill version via `mlflow.search_traces()`
- `build_eval_dataset(traces)` - Convert to MLflow 3 format: `{"inputs": {...}, "outputs": {...}, "expectations": {...}}`
- `run_evaluation(version)` - Run `mlflow.genai.evaluate()` with scorers, logs to Databricks experiment
- `compare_versions(a, b)` - Detect regressions/improvements using MLflow run comparison
- `check_quality_gates()` - CI/CD gate validation against metric thresholds

**Databricks Integration:**
```python
def run_evaluation(self, version: str):
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(f"/Shared/skill-benchmarks/{self.skill_name}")

    with mlflow.start_run(run_name=f"{self.skill_name}_v{version}"):
        mlflow.set_tags({
            "skill.name": self.skill_name,
            "skill.version": version,
            "evaluation.type": "benchmark"
        })

        # MLflow 3 GenAI evaluation
        results = mlflow.genai.evaluate(
            data=self.eval_dataset,
            scorers=self.get_scorers()
        )

    return results
```

### 6. CLI Commands

```bash
# Run benchmark
python -m scripts.skill_benchmark run --skill mlflow-evaluation --version 1.0.0

# Compare versions (A/B testing)
python -m scripts.skill_benchmark compare --skill mlflow-evaluation --version-a 1.0.0 --version-b 2.0.0

# Check regression against baseline
python -m scripts.skill_benchmark regression --skill mlflow-evaluation

# Initialize benchmark for new skill
python -m scripts.skill_benchmark init --skill my-new-skill
```

---

## Phase 2: Hybrid Execution-Based Evaluation

### The Problem
LLM judges can say code "looks correct" but miss:
- Import errors (module doesn't exist or wrong path)
- Runtime errors (type mismatches, missing args)
- API mismatches (method signature changed)
- Logic errors (code runs but produces wrong results)

### Hybrid Approach

```
[LLM Evaluation]                    [Execution Evaluation]
     │                                      │
     ▼                                      ▼
Guidelines scorer ──────┐    ┌───── Execute extracted code
Custom judge ───────────┤    ├───── Capture success/failure/errors
                        │    │
                        ▼    ▼
                   Combined Score
```

### Execution Scorer Pattern

```python
from mlflow.genai.scorers import scorer
from mlflow.entities import Feedback
import subprocess
import tempfile
import ast
import re

@scorer
def code_executes_successfully(inputs, outputs) -> list[Feedback]:
    """Extract code from response and actually run it."""
    response = outputs.get("response", "")

    # Extract Python code blocks
    code_blocks = re.findall(r'```python\n(.*?)```', response, re.DOTALL)

    if not code_blocks:
        return Feedback(
            name="code_executes",
            value="skip",
            rationale="No Python code blocks found"
        )

    results = []
    for i, code in enumerate(code_blocks):
        # 1. First check syntax (fast fail)
        try:
            ast.parse(code)
        except SyntaxError as e:
            results.append(Feedback(
                name=f"code_block_{i}_syntax",
                value="no",
                rationale=f"Syntax error: {e.msg} at line {e.lineno}"
            ))
            continue

        # 2. Actually execute (with timeout and isolation)
        exec_result = execute_code_safely(code, timeout=30)

        results.append(Feedback(
            name=f"code_block_{i}_executes",
            value="yes" if exec_result.success else "no",
            rationale=exec_result.error or "Executed successfully"
        ))

    return results


def execute_code_safely(code: str, timeout: int = 30) -> ExecutionResult:
    """Execute code in isolated subprocess with timeout."""

    # Add test harness wrapper
    wrapped_code = f'''
import sys
try:
    # Mock MLflow tracking to avoid actual API calls
    import unittest.mock as mock
    with mock.patch('mlflow.set_tracking_uri'), \\
         mock.patch('mlflow.set_experiment'), \\
         mock.patch('mlflow.start_run'), \\
         mock.patch('mlflow.genai.evaluate') as mock_eval:

        # Make evaluate return a mock result
        mock_eval.return_value = type('Result', (), {{'metrics': {{}}, 'run_id': 'test'}})()

        # Execute the generated code
{textwrap.indent(code, "        ")}

    print("__EXECUTION_SUCCESS__")
except Exception as e:
    print(f"__EXECUTION_ERROR__: {{type(e).__name__}}: {{e}}", file=sys.stderr)
    sys.exit(1)
'''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(wrapped_code)
        temp_path = f.name

    try:
        result = subprocess.run(
            ['python', temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
        )

        if "__EXECUTION_SUCCESS__" in result.stdout:
            return ExecutionResult(success=True)
        else:
            error = result.stderr or result.stdout
            return ExecutionResult(success=False, error=error)

    except subprocess.TimeoutExpired:
        return ExecutionResult(success=False, error=f"Timeout after {timeout}s")
    finally:
        os.unlink(temp_path)
```

### Execution Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Syntax only** | AST parse check | Fast, always safe |
| **Import check** | Try importing modules | Catches missing deps |
| **Dry run** | Execute with mocked externals | Validates logic without side effects |
| **Full execution** | Run against test fixtures | End-to-end validation |

### Test Fixtures for Full Execution

```python
# benchmarks/skills/mlflow-evaluation/fixtures.py
"""Test fixtures for execution-based evaluation."""

import mlflow

def setup_test_experiment():
    """Create isolated test experiment for code execution."""
    mlflow.set_tracking_uri("databricks")
    exp = mlflow.set_experiment("/Shared/skill-benchmark-sandbox/test")
    return exp.experiment_id

# Sample data that generated code should work with
SAMPLE_EVAL_DATA = [
    {
        "inputs": {"query": "What is MLflow?"},
        "outputs": {"response": "MLflow is an open source platform..."},
        "expectations": {"expected_facts": ["open source", "ML lifecycle"]}
    }
]

# Sample traces for trace-based code
SAMPLE_TRACE_IDS = ["tr-test-001", "tr-test-002"]
```

### Combined Scorer Strategy

```python
# benchmarks/skills/mlflow-evaluation/scorers.py

def get_scorers(execution_mode: str = "dry_run"):
    """Get all scorers including execution-based ones."""

    scorers = [
        # LLM-based (fast, catches intent issues)
        Safety(),
        code_quality_guidelines,
        api_correctness_judge,

        # Syntax-based (fast, catches obvious errors)
        code_syntax_valid,
    ]

    # Add execution scorer based on mode
    if execution_mode == "syntax_only":
        pass  # code_syntax_valid already added
    elif execution_mode == "dry_run":
        scorers.append(code_executes_with_mocks)
    elif execution_mode == "full":
        scorers.append(code_executes_with_fixtures)

    return scorers
```

### CLI Integration

```bash
# Default: LLM + syntax only (fast)
python -m scripts.skill_benchmark run --skill mlflow-evaluation

# With dry-run execution (mocked externals)
python -m scripts.skill_benchmark run --skill mlflow-evaluation --exec-mode dry_run

# Full execution against test fixtures (slowest, most thorough)
python -m scripts.skill_benchmark run --skill mlflow-evaluation --exec-mode full
```

### Phase 2 Implementation Tasks
- [ ] Implement `execute_code_safely()` with subprocess isolation
- [ ] Create mock harness for MLflow APIs
- [ ] Build test fixtures for full execution mode
- [ ] Add `--exec-mode` flag to CLI
- [ ] Create execution result aggregation

---

## Ground Truth Generation: `/skill-test` Command

### Approach: User-Invocable Skill

A skill that orchestrates the test → execute → save workflow using Claude's existing tools (Bash, Read, Write). No new infrastructure needed.

```
.claude/skills/skill-test/
└── SKILL.md    # Workflow instructions for Claude to follow
```

### Workflow

```
/skill-test mlflow-evaluation
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ 1. Load skill context (.claude/skills/mlflow-evaluation/)    │
│ 2. Accept test prompt from user                              │
│ 3. Generate response (Claude uses skill knowledge)           │
│ 4. Extract Python code blocks from response                  │
│ 5. Execute each block via Bash tool                          │
│ 6. On success → offer to save to ground_truth.py             │
│ 7. On failure → show error, allow retry                      │
└──────────────────────────────────────────────────────────────┘
```

### Skill Definition

```markdown
# .claude/skills/skill-test/SKILL.md

# /skill-test - Interactive Skill Testing & Ground Truth Generation

Test a skill interactively and build verified benchmark datasets.

## Usage
/skill-test <skill-name>

## Workflow

When invoked, follow these steps:

### Step 1: Load Target Skill
Read the skill's SKILL.md and key references to understand its domain.

### Step 2: Get Test Prompt
Ask the user for a test prompt, or let them paste one from test_cases.py.

### Step 3: Generate Response
Respond to the prompt using the loaded skill's knowledge. Include complete,
runnable Python code.

### Step 4: Execute Code
Extract all Python code blocks from your response. For each block:
1. Save to a temp file
2. Run via: `python /tmp/skill_test_<n>.py`
3. Capture stdout/stderr and exit code

### Step 5: Report Results
Show execution results for each code block:
- ✓ Block N: Executed successfully (X.Xs)
- ✗ Block N: Failed - <error message>

### Step 6: Save Ground Truth (on success)
If ALL code blocks execute successfully, ask the user:
"Add this as ground truth? [Y/n]"

If yes, append to `benchmarks/skills/<skill-name>/ground_truth.py`:

```python
GroundTruthExample(
    id="gt-XXX",  # Auto-increment
    prompt='''<the test prompt>''',
    response='''<your full response>''',
    execution_success=True,
    created_at=datetime.now(),
    tags=["<inferred-tags>"],
)
```

### Step 7: On Failure
If code fails, show the error and ask:
"Would you like to [R]etry, [E]dit the code, or [S]kip?"
```

### Ground Truth Storage (YAML + Python)

Data is stored in YAML for readability and easy editing, with a Python loader for integration.

**YAML Data File:**
```yaml
# benchmarks/skills/mlflow-evaluation/ground_truth.yaml
examples:
  - id: gt-001
    prompt: "Generate an evaluation script for a RAG agent"
    response: |
      Here's a complete MLflow evaluation script...

      ```python
      import mlflow.genai
      from mlflow.genai.scorers import Safety, RetrievalGroundedness
      # ... code ...
      ```
    execution_success: true
    created_at: "2025-01-10T13:19:00"
    tags: [evaluation, rag, scorers]
```

**Python Loader:**
```python
# benchmarks/skills/mlflow-evaluation/ground_truth.py
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
```

**Benefits of YAML Storage:**
- Human-readable and easy to edit directly
- Clean separation of data (YAML) from code (Python)
- Multi-line strings (responses with code) are natural in YAML
- Easy to diff/review in version control
- Can be processed by other tools if needed

### Integration with Benchmark Framework

The evaluation pipeline loads ground truth alongside traced data:

```python
def build_eval_dataset(self, include_ground_truth: bool = True) -> list[dict]:
    """Build dataset from traces + verified ground truth."""
    dataset = []

    # Traced examples (from manual Claude Code sessions)
    dataset.extend(self._traces_to_records())

    # Verified ground truth (from /skill-test)
    if include_ground_truth:
        from benchmarks.skills.mlflow_evaluation.ground_truth import GROUND_TRUTH
        dataset.extend([ex.to_eval_record() for ex in GROUND_TRUTH])

    return dataset
```

### Why This Works for MVP

1. **No new tooling** - Uses Claude's existing Bash, Read, Write tools
2. **Simple skill** - Just workflow instructions, no code to maintain
3. **Extensible** - Can add helper scripts later if needed
4. **Interactive** - User stays in control of what gets saved
5. **Verifiable** - Code must actually run to be saved as ground truth

### Phase 3 Implementation Tasks
- [ ] Create `/skill-test` skill at `.claude/skills/skill-test/SKILL.md`
- [ ] Define `GroundTruthExample` dataclass in benchmark template
- [ ] Create empty `ground_truth.py` in each skill's benchmark folder
- [ ] Update pipeline to load ground truth examples
- [ ] Document workflow in BENCHMARK_PLAN.md

---

## Implementation Plan

### Phase 1: `/skill-test` Command (Build First) ✅ COMPLETE
Build the ground truth generation tool first so we can use it to create evaluation data.

- [x] Create `/skill-test` skill at `.claude/skills/skill-test/SKILL.md`
- [x] Define workflow: load skill → get prompt → generate → execute → save
- [x] Create `GroundTruthExample` dataclass template
- [x] Test with mlflow-evaluation skill to generate initial ground truth

### Phase 2: Generate Ground Truth Dataset ✅ COMPLETE
Use `/skill-test` to build verified examples for mlflow-evaluation skill.

- [x] Create `benchmarks/skills/mlflow-evaluation/` structure
- [x] Run `/skill-test mlflow-evaluation` with various prompts:
  - Basic evaluation script generation
  - Custom scorer creation
  - Dataset building from traces
  - Guidelines scorer usage
- [x] Save passing examples to `ground_truth.yaml` (12 verified examples)
- [x] Aim for 10-20 verified working examples

### Phase 3: Benchmark Infrastructure
Build the evaluation framework that consumes ground truth.

- [ ] Define Python module templates (`config.py`, `test_cases.py`, `scorers.py`)
- [ ] Implement `init` command to scaffold benchmark for any skill
- [ ] Create reusable Guidelines scorers (`code_quality`, `skill_adherence`)
- [ ] Create custom judge with `make_judge()` for API correctness
- [ ] Implement `code_syntax_valid` scorer (AST parsing)

### Phase 4: Evaluation Pipeline
- [ ] Implement `SkillEvaluationPipeline` class
- [ ] Load ground truth + traced examples into dataset
- [ ] Run `mlflow.genai.evaluate()` with scorers
- [ ] Log results to Databricks experiment

### Phase 5: CLI and Baseline Management
- [ ] CLI entry points (`run`, `compare`, `regression`, `init`)
- [ ] Baseline save/load functionality
- [ ] Regression detection logic
- [ ] Run first evaluation on mlflow-evaluation skill

---

## Files to Create/Modify

### Phase 1: `/skill-test` Command
| File | Purpose |
|------|---------|
| `.claude/skills/skill-test/SKILL.md` | Interactive skill testing workflow |

### Phase 2: Ground Truth Dataset
| File | Purpose |
|------|---------|
| `benchmarks/skills/mlflow-evaluation/__init__.py` | Package init |
| `benchmarks/skills/mlflow-evaluation/ground_truth.yaml` | Verified working examples data (appended via /skill-test) |
| `benchmarks/skills/mlflow-evaluation/ground_truth.py` | Dataclass definition and YAML loader |

### Phase 3-5: Benchmark Infrastructure
| File | Purpose |
|------|---------|
| `benchmarks/skills/mlflow-evaluation/config.py` | Quality gates, settings |
| `benchmarks/skills/mlflow-evaluation/test_cases.py` | Test cases as Python dataclasses |
| `benchmarks/skills/mlflow-evaluation/scorers.py` | Skill-specific scorers/judges |
| `evaluation/skill_scorers.py` | Reusable scorers (code_syntax_valid, trace-based) |
| `evaluation/skill_judges.py` | Reusable Guidelines and custom judges |
| `scripts/skill_benchmark.py` | CLI entry point |
| `scripts/skill_eval_pipeline.py` | Pipeline implementation |

### Later: Execution-Based Evaluation
| File | Purpose |
|------|---------|
| `benchmarks/skills/mlflow-evaluation/fixtures.py` | Test data for execution mode |
| `evaluation/code_execution.py` | Safe code execution harness |

## Verification

### Phase 1: `/skill-test` Works
1. Run `/skill-test mlflow-evaluation` in Claude Code
2. Provide a test prompt (e.g., "Generate an evaluation script for a RAG agent")
3. Verify Claude generates code and executes it
4. Confirm prompt to save to ground_truth.py appears on success

### Phase 2: Ground Truth Generated
5. Run `/skill-test` with 10+ different prompts covering:
   - Evaluation script generation
   - Custom scorer creation
   - Dataset building
   - Guidelines usage
6. Verify `benchmarks/skills/mlflow-evaluation/ground_truth.py` contains verified examples

### Phase 3-5: Benchmark Framework Works
7. Run `python -m scripts.skill_benchmark init --skill mlflow-evaluation` - creates structure
8. Run `python -m scripts.skill_benchmark run --skill mlflow-evaluation` - produces metrics from ground truth
9. Metrics logged to Databricks experiment
10. Run `compare` between skill versions - detects differences

### Later: Execution-Based Evaluation
11. Run with `--exec-mode dry_run` - code blocks executed with mocks
12. Run with `--exec-mode full` - code runs against test fixtures
