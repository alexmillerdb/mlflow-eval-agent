# MLflow Evaluation Agent

Autonomous agent for analyzing MLflow traces and building evaluation suites. Built with Claude Agent SDK using a single agent with initializer/worker loop pattern.

## CRITICAL: Read Skills Before Writing Code

**Always read the relevant skill files before generating evaluation code.** Skills contain correct API patterns, common gotchas, and working examples.

### Available Skills

| Skill | Location | Use When |
|-------|----------|----------|
| **mlflow-evaluation** | `.claude/skills/mlflow-evaluation/` | Generating evaluation code, creating scorers, building datasets, analyzing traces, profiling latency, debugging failures, optimizing prompts/context/token budgets |

### Key Gotchas (from mlflow-evaluation skill)

1. Use `mlflow.genai.evaluate()` - NOT `mlflow.evaluate()` (deprecated for GenAI)
2. `predict_fn` receives unpacked kwargs: `predict_fn(**inputs)` not `predict_fn(inputs)`
3. Data requires nested structure: `{"inputs": {"query": "..."}}` not `{"query": "..."}`

**Read `.claude/skills/mlflow-evaluation/references/GOTCHAS.md` for the complete list.**

## Architecture

Single agent with autonomous loop:

1. **Initializer session** (`prompts/initializer.md`) - Analyzes traces, creates task plan
2. **Worker sessions** (`prompts/worker.md`) - Executes one task per session, updates status

State is persisted via JSON files in session directory. Each session runs with fresh context.

## MCP Tools

Three simplified tools served via in-process MCP server:

| Tool | Purpose |
|------|---------|
| `mlflow_query` | Search traces, get trace details, retrieve assessments |
| `mlflow_annotate` | Set tags, log feedback, log expectations on traces |
| `save_findings` | Persist analysis to session state files |

## Configuration

Set via environment variables or `.env`:

```bash
DATABRICKS_HOST=https://your-workspace.databricks.com
DATABRICKS_TOKEN=your-token
MLFLOW_EXPERIMENT_ID=123
```

## Running

### Databricks (Recommended)

```bash
# Deploy bundle
databricks bundle deploy -t dev

# Run notebook job
databricks bundle run eval_agent_notebook -t dev
```

### Local

```bash
# Autonomous mode
uv run python -m src.cli -a -e <experiment_id>

# Interactive mode
uv run python -m src.cli -i
```

## Testing Framework

Use these tests during development to validate changes at each layer:

| Test | Command | When to Use |
|------|---------|-------------|
| `/test-initializer` | `test initializer -e <id>` | After modifying trace analysis or task planning |
| `/test-worker` | `test worker -e <id> --session-dir <path>` | After modifying task execution logic |
| `/test-tools` | `test tools <tool> --operation <op>` | Debugging tool failures, testing MLflow connectivity |
| Integration | `test integration -e <id>` | Before deploying, after major refactoring |

### Development Workflow

```bash
# 1. Test initializer (trace analysis + task planning)
uv run python -m src.cli test initializer -e <experiment_id>

# 2. Test worker on the session from step 1
uv run python -m src.cli test worker -e <experiment_id> --session-dir <path>

# 3. Debug specific tool if issues arise
uv run python -m src.cli test tools mlflow_query --operation search -e <experiment_id>

# 4. Full integration test
uv run python -m src.cli test integration -e <experiment_id> --max-iterations 2

# 5. Analyze performance
/analyze-trace <trace-id>
/analyze-tokens <trace-id>
```

### Key Flags

- `--mock` - Test without MLflow connection (uses mock data)
- `--background` - Run in background, returns immediately
- `--task-type <type>` - Filter worker to specific task type (dataset, scorer, script, validate)
- `--verbose` - Show detailed output

### What Each Test Validates

| Test | Success Criteria |
|------|------------------|
| Initializer | `eval_tasks.json` and `state/analysis.json` exist with required fields |
| Worker | At least one task marked completed, artifacts created in `evaluation/` |
| Tools | Tool executes without error, returns valid output |
| Integration | At least one task completed across all iterations |

### Quick Mock Testing

```bash
# Test initializer without MLflow
uv run python -m src.cli test initializer -e 123 --mock

# Test worker with mock tasks
uv run python -m src.cli test worker -e 123 --session-dir /tmp/test --mock
```

**Key metrics to watch:**
- Cache efficiency (>50% = good)
- Token growth across sessions (should be stable)
- Cost per session (<$1 per initializer, <$0.50 per worker)
