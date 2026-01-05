# MLflow Evaluation Agent

An evaluation-driven agent for analyzing and optimizing GenAI agents deployed on Databricks. Built with the Claude Agent SDK following [Anthropic's autonomous coding pattern](https://github.com/anthropics/claude-quickstarts/tree/main/autonomous-coding).

## Overview

This agent helps you:
- **Analyze production traces** to find errors, performance issues, and patterns
- **Generate evaluation suites** with custom scorers based on real failures
- **Optimize agent context** (prompts, RAG, state management, token budgets)
- **Run autonomous evaluation loops** that build complete eval suites

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     AUTONOMOUS MODE (-a)                         │
│    Fresh context per session • File-based state persistence      │
└─────────────────────────────────────────────────────────────────┘
                                │
           ┌────────────────────┴────────────────────┐
           ▼                                         ▼
┌───────────────────────┐                 ┌───────────────────────┐
│   INITIALIZER (1st)   │                 │    WORKER (2nd+)      │
│   prompts/            │                 │    prompts/           │
│   initializer.md      │                 │    worker.md          │
│                       │                 │                       │
│   • Analyze traces    │    creates      │   • Read eval_tasks   │
│   • Create strategy   │ ──────────────► │   • Complete 1 task   │
│   • Write task plan   │  eval_tasks.json│   • Update status     │
└───────────────────────┘                 └───────────────────────┘
           │                                         │
           └────────────────────┬────────────────────┘
                                ▼
           ┌───────────────────────────────────────────┐
           │           3 MCP Tools + Built-ins         │
           ├───────────────────────────────────────────┤
           │ mlflow_query     │ Read, Bash, Glob      │
           │ mlflow_annotate  │ Grep, Skill           │
           │ save_findings    │                       │
           └───────────────────────────────────────────┘
```

### Autonomous Workflow

**Session Architecture:**
- Each session runs in a **fresh context window** (no memory of previous sessions)
- State is persisted via files: `eval_tasks.json` in session directory
- Auto-continues until all tasks complete or user interrupts

**Initializer Session** (`prompts/initializer.md`):
1. Analyzes traces to understand the agent being evaluated
2. Determines evaluation strategy (scorers, dataset approach)
3. Creates `eval_tasks.json` with ordered task plan
4. Saves analysis to session directory

**Worker Sessions** (`prompts/worker.md`):
1. Reads `eval_tasks.json` to find next pending task
2. Completes ONE task (dataset, scorer, script, or validate)
3. Updates task status to "completed"
4. Exits — next session picks up next task

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Databricks workspace with MLflow tracing enabled

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/mlflow-eval-agent.git
cd mlflow-eval-agent

# Install dependencies
uv sync
```

### Environment Variables

Create a `.env` file in the project root:

```bash
# Required
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com

# Authentication (choose one)
DATABRICKS_TOKEN=dapi...                    # Personal access token
DATABRICKS_CONFIG_PROFILE=my-profile        # OR use ~/.databrickscfg profile

# MLflow
MLFLOW_EXPERIMENT_ID=123456789              # Numeric experiment ID
MLFLOW_TRACKING_URI=databricks              # Default: databricks

# Use Databricks FM APIs
ANTHROPIC_BASE_URL=https://your-workspace.cloud.databricks.com/serving-endpoints/anthropic
ANTHROPIC_AUTH_TOKEN=dapi...
ANTHROPIC_API_KEY=""
MODEL=databricks-claude-opus-4-5
CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1


# Databricks Compute (optional)
DATABRICKS_SERVERLESS_COMPUTE_ID=auto       # For serverless SQL
DATABRICKS_CLUSTER_ID=                      # For classic compute
```

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABRICKS_HOST` | Yes | Databricks workspace URL |
| `DATABRICKS_TOKEN` | Yes* | Personal access token |
| `DATABRICKS_CONFIG_PROFILE` | Yes* | Alternative to token - uses ~/.databrickscfg |
| `MLFLOW_EXPERIMENT_ID` | Recommended | Target experiment to analyze |
| `MLFLOW_AGENT_EXPERIMENT_ID` | No | Experiment for agent's own traces (debugging) |
| `MLFLOW_TRACKING_URI` | No | MLflow server (default: `databricks`) |
| `MODEL` | No | Model to use (default: `sonnet`) |

*One of `DATABRICKS_TOKEN` or `DATABRICKS_CONFIG_PROFILE` is required.

### Running the Agent

```bash
# Autonomous mode - builds complete evaluation suite
uv run python -m src.cli -a -e <experiment_id>

# Interactive mode - free-form queries
uv run python -m src.cli -i

# Single query
uv run python -m src.cli "Analyze traces from experiment 123"
```

## Project Structure

```
mlflow-eval-agent/
├── src/
│   ├── agent.py              # Main agent + autonomous loop
│   ├── cli.py                # CLI with -i/-a modes
│   ├── config.py             # Simplified configuration
│   ├── tools.py              # 3 MCP tools
│   ├── mlflow_ops.py         # MLflow operations
│   └── legacy/               # Archived code (old sub-agents, tests)
│
├── prompts/                  # External prompt files
│   ├── initializer.md        # First session: analyze + plan
│   └── worker.md             # Subsequent sessions: execute tasks
│
├── sessions/                 # Session output directories (gitignored)
│
└── .claude/
    ├── commands/             # Slash commands
    └── skills/
        └── mlflow-evaluation/
```

## Tools

The agent uses 3 simplified MCP tools:

| Tool | Purpose |
|------|---------|
| `mlflow_query` | Search traces, get trace details, retrieve assessments |
| `mlflow_annotate` | Set tags, log feedback, log expectations on traces |
| `save_findings` | Persist state to `.claude/state/` for cross-session continuity |

## Skills

Skills provide domain knowledge loaded via the `Skill` tool:

| Skill | Use When |
|-------|----------|
| `mlflow-evaluation` | Generating evaluation code, creating scorers, building datasets |

## License

MIT
