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

### Testing Ladder

```
Level 1: pytest              → Fast, no external deps
Level 2: Import verification → Catches circular imports
Level 3: Mock (--mock)       → Logic without MLflow
Level 4: Local integration   → Real MLflow locally
Level 5: Autonomous run      → Full agent loop
Level 6: Databricks deploy   → Production validation
```

### Slash Commands

| Command | Purpose |
|---------|---------|
| `/test-initializer` | Test trace analysis + task planning |
| `/test-worker` | Test task execution |
| `/test-tools` | Debug tool failures |
| `/test-phase <n>` | Run testing ladder for phase n |
| `/test-full-cycle <exp-id>` | Complete validation (all 7 steps) |
| `/validate-deployment` | Databricks bundle + app validation |

### Development Workflow

```bash
# 1. Test initializer
uv run python -m src.cli test initializer -e <experiment_id>

# 2. Test worker (use session_dir from step 1)
uv run python -m src.cli test worker -e <experiment_id> --session-dir <path>

# 3. Full integration
uv run python -m src.cli test integration -e <experiment_id> --max-iterations 2

# 4. Analyze performance
/analyze-trace <trace-id>
/analyze-tokens <trace-id>
```

### Error Debugging Workflow

When tests fail:
1. **Capture trace ID** from output
2. **Analyze**: `/analyze-trace <trace-id>` (errors, bottlenecks)
3. **Check tokens**: `/analyze-tokens <trace-id>` (cache efficiency)
4. **Fix** and re-run

### Key Flags

- `--mock` - No MLflow connection (uses mock data)
- `--background` - Returns immediately
- `--task-type <type>` - Filter to dataset|scorer|script|validate

### Key Metrics

| Metric | Good | Bad |
|--------|------|-----|
| Cache efficiency | >50% | <30% |
| Initializer cost | <$1.50 | >$2.50 |
| Worker cost | <$0.75 | >$1.25 |

## Implementation

See `IMPLEMENTATION_SPEC.md` for phased implementation plan with per-phase testing gates.

Completed phases are archived in `docs/archive/` to keep the main spec focused.
