# MLflow Evaluation Agent

An evaluation-driven agent for analyzing and optimizing GenAI agents deployed on Databricks. Built with the Claude Agent SDK using a coordinator + sub-agent architecture.

## Overview

This agent helps you:
- **Analyze production traces** to find errors, performance issues, and patterns
- **Generate evaluation suites** with custom scorers based on real failures
- **Optimize agent context** (prompts, RAG, state management, token budgets)
- **Improve architecture** with multi-agent patterns and tool orchestration

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      COORDINATOR AGENT                          │
│  Orchestrates sub-agents • Synthesizes findings • Generates eval│
└─────────────────────────────────────────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ trace_analyst │───▶│context_engineer│───▶│agent_architect│
│               │    │               │    │               │
│ Search traces │    │ Prompt opt    │    │ Map arch      │
│ Find patterns │    │ Context rot   │    │ Find issues   │
│ Extract evals │    │ Token mgmt    │    │ Recommend     │
└───────────────┘    └───────────────┘    └───────────────┘
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SHARED WORKSPACE                           │
│  trace_analysis_summary • error_patterns • performance_metrics  │
│  context_recommendations • extracted_eval_cases                 │
└─────────────────────────────────────────────────────────────────┘
```

**Workflow**: `trace_analyst` → `context_engineer` → `agent_architect`

Each sub-agent writes findings to a shared workspace. Downstream agents read and build on previous findings.

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
DABS_MODEL=databricks-claude-opus-4-5
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
| `MLFLOW_EXPERIMENT_ID` | Recommended | Target experiment for trace analysis |
| `MLFLOW_TRACKING_URI` | No | MLflow server (default: `databricks`) |
| `DABS_MODEL` | No | Model for coordinator (default: `opus`) |

*One of `DATABRICKS_TOKEN` or `DATABRICKS_CONFIG_PROFILE` is required.

### MCP Server Setup

Add to `.claude/settings.local.json`:

```json
{
  "mcpServers": {
    "mlflow-mcp": {
      "command": "uv",
      "args": ["run", "--with", "mlflow[mcp]>=3.5.1", "mlflow", "mcp", "run"],
      "env": {
        "MLFLOW_TRACKING_URI": "databricks"
      }
    }
  }
}
```

### Running the Agent

```bash
# Interactive mode
uv run python -m src.agent -i

# Single query
uv run python -m src.agent "Analyze traces from experiment 123"

# With filter
uv run python -m src.agent --analyze "attributes.status = 'ERROR'"
```

## Project Structure

```
mlflow-eval-agent/
├── src/
│   ├── agent.py              # Main agent entry point
│   ├── config.py             # Configuration management
│   ├── workspace.py          # Shared workspace for inter-agent communication
│   ├── tools.py              # Tool definitions
│   └── subagents/
│       ├── coordinator.py    # Coordinator prompt generation
│       ├── registry.py       # Agent registry and config
│       ├── trace_analyst.py  # Trace analysis sub-agent
│       ├── context_engineer.py
│       ├── agent_architect.py
│       └── eval_runner.py
│
├── .claude/
│   ├── commands/             # Slash commands
│   └── skills/               # Knowledge bases for agents
│       ├── mlflow-evaluation/
│       ├── trace-analysis/
│       └── context-engineering/
│
└── tests/
```

## Sub-Agents

| Agent | Triggers | Outputs |
|-------|----------|---------|
| **trace_analyst** | traces, errors, performance, debugging | `trace_analysis_summary`, `error_patterns`, `performance_metrics` |
| **context_engineer** | optimize, prompt, context, tokens, RAG | `context_recommendations` |
| **agent_architect** | architecture, multi-agent, design, tools | Tags on traces |
| **eval_runner** | (internal) Executes generated eval code | `eval_results` |

## Skills

Skills provide domain knowledge to agents:

| Skill | Use When |
|-------|----------|
| `mlflow-evaluation` | Generating evaluation code, creating scorers, building datasets |
| `trace-analysis` | Analyzing traces, profiling latency, debugging failures |
| `context-engineering` | Optimizing prompts, RAG context, state management |

## Running Tests

```bash
uv run pytest tests/ -v
```

## License

MIT
