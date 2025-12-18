# MLflow Evaluation Agent

Evaluation-driven agent for analyzing and optimizing GenAI agents on Databricks. Built with Claude Agent SDK using a coordinator + sub-agent architecture with inter-agent communication via shared workspace.

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

### Sub-Agents

| Agent | Role | Writes to Workspace |
|-------|------|---------------------|
| `trace_analyst` | Deep trace analysis using MCP tools | `trace_analysis_summary`, `error_patterns`, `performance_metrics` |
| `context_engineer` | Context optimization recommendations | `context_recommendations` |
| `agent_architect` | Architecture analysis and improvements | Architecture tags on traces |

### Workflow Order

1. `trace_analyst` - Analyze production data first
2. `context_engineer` - Reads trace findings, generates optimizations
3. `agent_architect` - Reads all findings, provides structural recommendations
4. Coordinator synthesizes and generates action plan

## MCP Tools

### In-Process MCP Server (mcp__mlflow-eval__)

All tools are served from a single in-process MCP server:

**MLflow Trace Tools:**
- `search_traces` - Find traces with filters
- `get_trace` - Get detailed trace by ID (use `output_mode='aggressive'` for batch operations)
- `set_trace_tag` / `delete_trace_tag` - Tag traces for dataset building
- `log_feedback` - Store analysis findings on traces
- `log_expectation` - Store ground truth for evaluation
- `get_assessment` / `update_assessment` - Retrieve/modify assessments

**Workspace Tools:**
- `write_to_workspace` / `read_from_workspace` - Inter-agent communication
- `check_workspace_dependencies` - Validate required data exists

## Configuration

Set via environment variables or `.env`:

```bash
DATABRICKS_HOST=https://your-workspace.databricks.com
DATABRICKS_TOKEN=your-token
MLFLOW_EXPERIMENT_ID=123
```

## Running

```bash
# Interactive mode
python src/agent.py -i

# Single query
python src/agent.py "Analyze traces from experiment 123"

# Analyze with filter
python src/agent.py --analyze "attributes.status = 'ERROR'"
```
