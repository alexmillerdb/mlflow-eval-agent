---
description: Test worker session in isolation
argument-hint: "<experiment-id> --session-dir <path> [--task-type dataset|scorer|script|validate] [--mock] [--background]"
---

Run one worker session on existing or mock tasks.

## Usage

```bash
# Test worker with existing session (streams progress by default)
uv run python -m src.cli test worker -e <experiment-id> --session-dir <path>

# Test worker with mock tasks
uv run python -m src.cli test worker -e 123456789 --session-dir /tmp/test-session --mock

# Test specific task type
uv run python -m src.cli test worker -e 123456789 --session-dir <path> --task-type dataset

# Run in background (returns immediately, writes to log file)
uv run python -m src.cli test worker -e <experiment-id> --session-dir <path> --background
```

## Prerequisites

Either:
1. Run `/test-initializer` first to create tasks, OR
2. Use `--mock` to create sample tasks automatically

## What This Tests

1. **Task Selection** - Does the worker pick the first pending task?
2. **Task Execution** - Can it complete the task successfully?
3. **Status Update** - Does it mark the task as completed?
4. **Artifact Creation** - Does it create expected files in `evaluation/`?

## Task Types

| Type | Creates | Purpose |
|------|---------|---------|
| `dataset` | `evaluation/eval_dataset.py` | Evaluation dataset |
| `scorer` | `evaluation/scorers.py` | Custom scorers |
| `script` | `evaluation/run_eval.py` | Eval runner script |
| `validate` | `state/validation_results.json` | Validation results |

## Follow-up Analysis

```bash
# Analyze the worker's performance
/analyze-trace <trace-id>

# Analyze token usage
/analyze-tokens <trace-id>
```

## Example

```bash
# Test worker on existing session
uv run python -m src.cli test worker -e 1234567890123456789 \
    --session-dir /tmp/mlflow-eval-test-abc123

# Streaming output:
# Testing worker session for experiment 1234567890123456789...
#   -> Read (eval_tasks.json)
#   -> Skill (mlflow-evaluation)
#   -> Write (eval_dataset.py)
#   -> Edit (eval_tasks.json)
#   ✓ Complete (45s)
#
# ==================================================
# Test Result: PASS
# ==================================================
# Trace ID: tr-xyz789
#   Use: /analyze-trace tr-xyz789
# Duration: 45000ms
# Cost: $0.85
```

## Background Mode

```bash
# Run in background for long-running tests
uv run python -m src.cli test worker -e 1234567890123456789 \
    --session-dir /tmp/mlflow-eval-test-abc123 --background

# Output:
# Running in background (PID: 12345)
# Output: /tmp/mlflow-test-a1b2c3d4.log
# Check status: tail -f /tmp/mlflow-test-a1b2c3d4.log
```
