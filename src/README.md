# Source Code Structure

Core implementation of the MLflow Evaluation Agent. Single agent with initializer/worker loop pattern.

## Module Overview

| Module | Purpose |
|--------|---------|
| `agent.py` | Core agent and autonomous evaluation loop |
| `cli.py` | CLI entry point (interactive, autonomous, test modes) |
| `config.py` | Configuration management from environment variables |
| `tools.py` | MCP tools (`mlflow_query`, `mlflow_annotate`, `save_findings`) |
| `mlflow_ops.py` | Direct MLflow operations (trace/run queries, annotations, state) |
| `test_harness.py` | Testing framework for component isolation |
| `databricks_auth.py` | Auth helper for subprocess execution |
| `runtime.py` | Runtime detection (local vs Databricks) |
| `skills.py` | Skill loading and recommendation engine |

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI (cli.py)                             │
│  -i (interactive) │ -a (autonomous) │ test <component>          │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Agent (agent.py)                             │
│  MLflowAgent.query() → ClaudeSDKClient → stream results         │
│  run_autonomous() → initializer/worker loop                     │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   MCP Tools (tools.py)                          │
│  mlflow_query │ mlflow_annotate │ save_findings                 │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              MLflow Operations (mlflow_ops.py)                  │
│  search_traces() │ get_trace() │ set_tag() │ save_state()       │
└─────────────────────────────────────────────────────────────────┘
```

## Session Directory Structure

Each autonomous run creates a session directory:

```
session_dir/
├── eval_tasks.json          # Task list with status tracking
├── state/
│   ├── analysis.json        # Trace analysis from initializer
│   └── validation_results.json  # Eval script validation results
└── evaluation/
    ├── eval_dataset.py      # Dataset creation code
    ├── scorers.py           # Custom scorer definitions
    └── run_eval.py          # Evaluation execution script
```

## State File Schemas

### eval_tasks.json

```json
{
  "tasks": [
    {
      "id": 1,
      "name": "Build evaluation dataset",
      "type": "dataset",
      "status": "pending",
      "details": "Extract representative samples from traces",
      "attempts": 0
    }
  ]
}
```

**Task types:** `dataset`, `scorer`, `script`, `validate`

**Task statuses:** `pending`, `completed`, `failed`

### state/analysis.json

```json
{
  "experiment_id": "123456789",
  "agent_type": "RAG agent for customer support",
  "dataset_strategy": "traces",
  "has_predict_fn": false,
  "trace_summary": {
    "total_analyzed": 50,
    "success_count": 45,
    "error_count": 5,
    "avg_latency_ms": 2500
  },
  "sample_trace_ids": ["tr-001", "tr-002", "tr-003"],
  "recommended_scorers": [
    {"name": "Safety", "type": "builtin", "rationale": "Required for all agents"},
    {"name": "RetrievalGroundedness", "type": "builtin", "rationale": "RAG agent"}
  ],
  "error_patterns": []
}
```

## Key Functions

### agent.py

- `setup_mlflow()` - Initialize MLflow tracking (call after env vars set)
- `load_prompt(name)` - Load prompt from `prompts/{name}.md`
- `MLflowAgent.query(prompt, session_id)` - Send query, stream results
- `run_autonomous(experiment_id, max_iterations)` - Autonomous evaluation loop

### tools.py

- `create_tools()` - Create 3 MCP tools for agent
- `MCPTools` - Tool name constants (`MLFLOW_QUERY`, `MLFLOW_ANNOTATE`, `SAVE_FINDINGS`)
- `BuiltinTools` - Claude SDK built-in tool names

### mlflow_ops.py

**Trace operations:**
- `search_traces(experiment_id, filter_string, max_results)` - Search traces
- `get_trace(trace_id, detail_level)` - Get trace with detail level: `summary`, `analysis`, `full`

**Run operations:**
- `search_runs(experiment_id, filter_string, max_results)` - Search MLflow Runs
- `get_run(run_id)` - Get run details

**Annotations:**
- `set_tag(trace_id, key, value)` - Set trace tag
- `log_feedback(trace_id, name, value, rationale)` - Log assessment
- `log_expectation(trace_id, name, value)` - Log ground truth

**State management:**
- `set_session_dir(path)` - Set current session directory
- `save_state(key, data)` - Save to `state/{key}.json`
- `load_state(key)` - Load from `state/{key}.json`
- `get_tasks_file()` - Path to `eval_tasks.json`
- `get_task_status()` - Current task completion status
- `all_tasks_complete()` - Check if all tasks done

**Task tracking:**
- `get_task_attempts(task_id)` - Get attempt count
- `increment_task_attempts(task_id)` - Increment attempts, mark failed if exceeded

**Context monitoring:**
- `ContextMetrics` - Tracks context growth within session
- `start_context_monitoring(session_id, prompt)` - Initialize monitoring
- `get_context_metrics()` - Get current metrics
- `record_tool_call(tool_name, input_size, output_size)` - Record tool usage

### config.py

- `Config.from_env(validate=True)` - Load config from environment
- `Config.validate()` - Validate required settings

### runtime.py

- `detect_runtime()` - Detect local vs Databricks environment
- `get_sessions_base_path()` - Get session storage path
- `RuntimeContext` - Enum: `LOCAL`, `DATABRICKS_JOB`

### databricks_auth.py

- `configure_env()` - Configure subprocess env vars for auth
- `get_databricks_host()` - Detect workspace URL
- `get_subprocess_env_vars()` - Get auth env vars for subprocess

### skills.py

- `parse_skill_index(skills_dir)` - Parse skill metadata from `.claude/skills/`
- `recommend_skills(query)` - Recommend skills based on keywords
- `get_skill_gotchas(skill_name)` - Get GOTCHAS.md content
- `get_skill_references(skill_name)` - Get reference file paths

## Constants

| Constant | Value | Location |
|----------|-------|----------|
| `MAX_TASK_ATTEMPTS` | 5 | `mlflow_ops.py` |
| `MAX_OUTPUT_CHARS` | 1500 | `mlflow_ops.py` |
| `CONTEXT_WARNING_KB` | 40 | `mlflow_ops.py` |
| `CONTEXT_CRITICAL_KB` | 80 | `mlflow_ops.py` |
| `AUTO_CONTINUE_DELAY_SECONDS` | 3 | `agent.py` |
