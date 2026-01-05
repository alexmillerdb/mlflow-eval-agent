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
