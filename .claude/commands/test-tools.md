---
description: Test MCP tools directly without full agent
argument-hint: "<tool-name> [--operation <op>] [-e <experiment-id>] [--trace-id <id>]"
---

Test individual MCP tools without running the full agent.

## Usage

```bash
# Test mlflow_query search
uv run python -m src.cli test tools mlflow_query --operation search -e <experiment-id>

# Test mlflow_query get
uv run python -m src.cli test tools mlflow_query --operation get --trace-id <trace-id>

# Test mlflow_annotate tag
uv run python -m src.cli test tools mlflow_annotate --operation tag --trace-id <trace-id>

# Test save_findings
uv run python -m src.cli test tools save_findings
```

## Available Tools

| Tool | Operations | Required Args |
|------|------------|---------------|
| `mlflow_query` | search, get, assessment | search: `-e`, get/assessment: `--trace-id` |
| `mlflow_annotate` | tag, feedback, expectation | `--trace-id` |
| `save_findings` | (none) | (none) |

## Operations

### mlflow_query

```bash
# Search traces
uv run python -m src.cli test tools mlflow_query --operation search -e 123456789

# Get trace details
uv run python -m src.cli test tools mlflow_query --operation get --trace-id tr-abc123

# Get assessment
uv run python -m src.cli test tools mlflow_query --operation assessment --trace-id tr-abc123
```

### mlflow_annotate

```bash
# Set tag (requires real trace)
uv run python -m src.cli test tools mlflow_annotate --operation tag --trace-id tr-abc123

# Log feedback
uv run python -m src.cli test tools mlflow_annotate --operation feedback --trace-id tr-abc123

# Log expectation
uv run python -m src.cli test tools mlflow_annotate --operation expectation --trace-id tr-abc123
```

### save_findings

```bash
# Test state persistence
uv run python -m src.cli test tools save_findings
```

## Example Output

```bash
$ uv run python -m src.cli test tools mlflow_query --operation search -e 123456789

Testing tool mlflow_query with operation search...

==================================================
Test Result: PASS
==================================================
Duration: 250ms

Outputs:
{
  "text": "| Trace ID | Status | Duration (ms) |\n|----------|--------|---------------|\n| tr-001 | OK | 2500 |\n..."
}
```

## Debugging Tips

1. **Connection issues**: Check `MLFLOW_TRACKING_URI` and credentials
2. **Not found errors**: Verify experiment ID and trace IDs exist
3. **Permission errors**: Ensure you have access to the experiment
