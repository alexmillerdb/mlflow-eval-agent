# MLflow Evaluation Agent

Full-stack application for analyzing MLflow traces and building evaluation suites. Three-tier architecture: core infrastructure, autonomous agent (Claude Agent SDK), and Streamlit chat UI.

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

## MANDATORY: Verify Before Claiming Done

**Do not say "done" until you run these checks.** Pick the tier(s) you changed.

### After ANY code change

```bash
# Import check — catches circular imports and missing modules
uv run python -c "from src.agent.agent import MLflowAgent, AgentResult; from src.core.config import Config"
uv run python -c "from src.core import Config, detect_runtime, RuntimeContext"
uv run python -c "from src.agent.agent import MLflowAgent, AgentResult"
uv run python -c "from src.agent.autonomous import run_autonomous"
uv run python -c "from src.agent.tools import create_tools, MCPTools, BuiltinTools"
uv run python -c "from src.app import async_to_sync_generator, require_auth"

# Unit tests (fast, no external deps)
uv run pytest tests/ -x -q -m "not e2e"
```

### After changing `src/core/`

```bash
uv run pytest tests/test_core/ -x -q
```

### After changing `src/agent/`

```bash
uv run pytest tests/test_tools_unit.py tests/test_sessions.py tests/test_context_optimization.py -x -q
```

### After changing `src/app/`

```bash
uv run pytest tests/test_app/ -x -q
timeout 5 uv run streamlit run src/app/main.py --server.headless=true 2>&1 || true
```

### After changing integration points

```bash
uv run pytest tests/integration/ -x -q
```

## Architecture

Three-tier architecture with unidirectional dependencies:

```
src/core/     →  src/agent/     →  src/app/
(config,         (Claude Agent      (Streamlit UI,
 runtime,         SDK, MCP tools,    streaming,
 auth, files)     prompts, ops)      auth middleware)
```

**Dependency rule:** `core/` ← `agent/` ← `app/`. Never import upstream.

| Tier | Directory | Entry Point | Purpose |
|------|-----------|-------------|---------|
| **Core** | `src/core/` | `from src.core import Config` | Config, runtime detection, auth, Files SDK |
| **Agent** | `src/agent/` | `from src.agent.agent import MLflowAgent` | Autonomous agent loop, MCP tools, MLflow ops |
| **App** | `src/app/` | `uv run streamlit run src/app/main.py` | Streamlit chat UI, async streaming, user auth |

### Agent Loop

1. **Initializer session** (`prompts/initializer.md`) — Analyzes traces, creates task plan
2. **Worker sessions** (`prompts/worker.md`) — Executes one task per session, updates status

State is persisted via JSON files in session directory. Each session runs with fresh context.

## MCP Tools

Nine tools served via in-process MCP server (`src/agent/tools.py`):

| Tool | Purpose |
|------|---------|
| `mlflow_query` | Search traces/runs, get trace details, retrieve assessments |
| `mlflow_annotate` | Set tags, log feedback, log expectations on traces |
| `save_findings` | Persist analysis to session state JSON files |
| `uc_volume_read` | Read file from Unity Catalog Volume |
| `uc_volume_write` | Write file to Unity Catalog Volume |
| `uc_volume_list` | List Unity Catalog Volume directory contents |
| `workspace_read` | Read file from Databricks Workspace |
| `workspace_write` | Write file to Databricks Workspace |
| `workspace_list` | List Databricks Workspace directory contents |

## Configuration

Set via environment variables or `.env`. See `.env.example` for complete list with documentation.

**Required:**

```bash
DATABRICKS_HOST=https://your-workspace.databricks.com
```

**MLflow:**

```bash
MLFLOW_TRACKING_URI=databricks
MLFLOW_EXPERIMENT_ID=123              # Target experiment to analyze
MLFLOW_AGENT_EXPERIMENT_ID=456        # Agent's own traces (optional)
```

**Model provider** (choose one):

```bash
# Option A: Databricks Foundation Model API (recommended)
ANTHROPIC_BASE_URL=https://your-workspace.databricks.com/serving-endpoints/anthropic
ANTHROPIC_AUTH_TOKEN=dapi...
ANTHROPIC_API_KEY=""

# Option B: Direct Anthropic API
ANTHROPIC_API_KEY=sk-ant-api03-...
```

## Running

### Streamlit App

```bash
uv run streamlit run src/app/main.py
```

### CLI

```bash
# Autonomous mode
uv run python -m src.cli -a -e <experiment_id>

# Interactive mode
uv run python -m src.cli -i
```

### Databricks

```bash
databricks bundle deploy -t dev
databricks bundle run eval_agent_notebook -t dev
```

## Testing

### Test directory structure

```
tests/
├── test_core/              # Core tier unit tests (config, runtime, files, auth)
├── test_app/               # App tier unit tests (streaming, auth middleware)
├── test_tools_unit.py      # Agent MCP tools unit tests
├── test_sessions.py        # Agent session management tests
├── test_context_optimization.py  # Context/prompt optimization tests
├── integration/            # Cross-tier integration tests
├── e2e/                    # Playwright browser tests (Streamlit UI)
├── conftest.py             # Shared fixtures
└── harness.py              # Test harness utilities
```

### Running tests

```bash
# All unit tests (excludes e2e browser tests)
uv run pytest tests/ -x -q -m "not e2e"

# By tier
uv run pytest tests/test_core/ -x -q
uv run pytest tests/test_app/ -x -q
uv run pytest tests/test_tools_unit.py tests/test_sessions.py tests/test_context_optimization.py -x -q

# Integration
uv run pytest tests/integration/ -x -q

# E2E (requires running Streamlit app)
uv run pytest tests/e2e/ -x -q
```

### Lazy import verification

Streamlit and CLI entry points use **lazy imports** inside function bodies (e.g., `main.py` imports `stream_autonomous` inside `_run_autonomous_streaming()`). These resolve at request time, not at module load — so standard import checks and unit tests pass even when the target function is missing or renamed.

`TestMainLazyImports` in `tests/test_app/test_streaming.py` mirrors every lazy import in `main.py`. **When adding or changing a lazy import in an entry point, add a corresponding test.**

Pattern:
```python
def test_my_new_lazy_import(self):
    """Imports used by my_function()."""
    from src.some.module import my_function
    assert callable(my_function)
```

Why this matters: the `stream_autonomous` ImportError that crashed the Autonomous tab in the running app was invisible to `uv run python -c "from src.agent.autonomous import ..."` because the old module was cached. The lazy import test catches this in pytest before it reaches users.

### Agent-specific testing (CLI)

```bash
# Test initializer session
uv run python -m src.cli test initializer -e <experiment_id>

# Test worker session (use session_dir from initializer)
uv run python -m src.cli test worker -e <experiment_id> --session-dir <path>

# Full integration
uv run python -m src.cli test integration -e <experiment_id> --max-iterations 2
```

### Key flags

- `--mock` — No MLflow connection (uses mock data)
- `--background` — Returns immediately
- `--task-type <type>` — Filter to dataset|scorer|script|validate

### Optional: Browser validation (Claude in Chrome)

When the Chrome plugin is available, use it for final visual validation after UI changes. This is exploratory — not a replacement for pytest.

**Smoke test after app changes:**

1. Start the app: `uv run streamlit run src/app/main.py`
2. Use `tabs_context_mcp` to get browser context, then `navigate` to the app URL
3. Use `read_page` to verify the accessibility tree (catches missing labels, broken ARIA)
4. Use `find` to locate elements by purpose (e.g., `find("experiment id input")`) — more resilient than coordinates
5. Walk the user journey: sidebar config, tab switching, chat submit, autonomous mode
6. Use `read_console_messages` with `pattern: "error|Error"` to catch silent JS failures
7. Use `gif_creator` to record a walkthrough for PR review

**When to use Chrome plugin vs. Playwright e2e:**

| Scenario | Use |
|----------|-----|
| Repeatable regression test for CI | Playwright (`tests/e2e/`) |
| "Does this look right after my change?" | Chrome plugin |
| Accessibility / screen reader audit | Chrome plugin `read_page` |
| Recording a demo GIF for a PR | Chrome plugin `gif_creator` |
| Debugging Streamlit rerun bugs | Chrome plugin + `read_console_messages` |

## Implementation

See `IMPLEMENTATION_SPEC.md` for phased implementation plan with per-phase testing gates.

Completed phases are archived in `docs/archive/` to keep the main spec focused.
