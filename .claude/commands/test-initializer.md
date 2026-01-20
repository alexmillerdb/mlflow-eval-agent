---
description: Test initializer session in isolation
argument-hint: "<experiment-id> [--mock] [--session-dir <path>] [--background]"
---

Run only the initializer session in isolation and capture trace ID for analysis.

## Usage

```bash
# Test initializer with real MLflow (streams progress by default)
uv run python -m src.cli test initializer -e $ARGUMENTS

# Test initializer with mock (no MLflow connection needed)
uv run python -m src.cli test initializer -e 123456789 --mock

# Run in background (returns immediately, writes to log file)
uv run python -m src.cli test initializer -e $ARGUMENTS --background
```

## What This Tests

1. **Trace Analysis** - Can the initializer analyze traces in the experiment?
2. **Task Planning** - Does it create a valid `eval_tasks.json`?
3. **Analysis Output** - Does it create `state/analysis.json` with required fields?

## Expected Outputs

After running, verify these files exist:
- `<session_dir>/eval_tasks.json` - Task list for worker sessions
- `<session_dir>/state/analysis.json` - Initial trace analysis

## Follow-up Analysis

```bash
# Analyze the initializer's performance
/analyze-trace <trace-id>

# Analyze token usage and cache efficiency
/analyze-tokens <trace-id>
```

## Example

```bash
# Test with a real experiment
uv run python -m src.cli test initializer -e 1234567890123456789

# Streaming output:
# Testing initializer session for experiment 1234567890123456789...
#   -> Skill (mlflow-evaluation)
#   -> Read (GOTCHAS.md)
#   -> mlflow_query (search)
#   -> mlflow_query (get)
#   -> save_findings (analysis)
#   -> Write (eval_tasks.json)
#   ✓ Complete (114s)
#
# ==================================================
# Test Result: PASS
# ==================================================
# Trace ID: tr-abc123def456
#   Use: /analyze-trace tr-abc123def456
# Session ID: session_20240115_123456
# Duration: 114000ms
# Cost: $1.15
```

## Background Mode

```bash
# Run in background for long-running tests
uv run python -m src.cli test initializer -e 1234567890123456789 --background

# Output:
# Running in background (PID: 12345)
# Output: /tmp/mlflow-test-a1b2c3d4.log
# Check status: tail -f /tmp/mlflow-test-a1b2c3d4.log
```
