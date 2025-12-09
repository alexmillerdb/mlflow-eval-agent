# MLflow Evaluation Agent

A modular, evaluation-driven agent built with the Claude Agent SDK for analyzing,
evaluating, and optimizing GenAI agents deployed on Databricks.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MLFLOW EVALUATION AGENT                                │
│                                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                         COORDINATOR AGENT                                  │  │
│  │  • Orchestrates sub-agents           • Synthesizes findings               │  │
│  │  • Generates dynamic eval scripts    • Manages shared workspace           │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                             │
│         ┌──────────────────────────┼──────────────────────────┐                 │
│         ▼                          ▼                          ▼                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐             │
│  │ TRACE ANALYST   │    │ CONTEXT         │    │ AGENT           │             │
│  │                 │    │ ENGINEER        │    │ ARCHITECT       │             │
│  │ • Search traces │───▶│ • Prompt opt    │───▶│ • Map arch      │             │
│  │ • Find patterns │    │ • Context rot   │    │ • Find issues   │             │
│  │ • Extract evals │    │ • Token mgmt    │    │ • Recommend     │             │
│  │                 │    │ • State mgmt    │    │                 │             │
│  │ WRITES findings │    │ READS trace     │    │ READS all       │             │
│  │ to workspace    │    │ findings        │    │ findings        │             │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘             │
│         │                          │                          │                 │
│         └──────────────────────────┼──────────────────────────┘                 │
│                                    ▼                                            │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                        SHARED WORKSPACE                                    │  │
│  │  • trace_analysis_summary    • error_patterns    • performance_metrics    │  │
│  │  • context_recommendations   • extracted_eval_cases                       │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                           SKILLS LAYER                                     │  │
│  │  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐           │  │
│  │  │ mlflow-evaluation│ │ trace-analysis   │ │ context-engineer │           │  │
│  │  │ SKILL.md         │ │ SKILL.md         │ │ SKILL.md         │           │  │
│  │  │                  │ │                  │ │                  │           │  │
│  │  │ • Scorer patterns│ │ • Query syntax   │ │ • Prompt arch    │           │  │
│  │  │ • Dataset format │ │ • Span analysis  │ │ • Context rot    │           │  │
│  │  │ • Eval templates │ │ • Error classify │ │ • State mgmt     │           │  │
│  │  │ • Code gen       │ │ • Performance    │ │ • Token budget   │           │  │
│  │  └──────────────────┘ └──────────────────┘ └──────────────────┘           │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                           MCP TOOLS                                        │  │
│  │  search_traces │ get_failures │ compare_runs │ generate_eval_script       │  │
│  │  write_to_workspace │ read_from_workspace                                  │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Inter-Agent Communication via Shared Workspace
Sub-agents don't work in isolation - they share findings through a workspace:

```python
# trace_analyst writes findings
workspace.write("trace_analysis_summary", {
    "error_rate": 0.15,
    "top_errors": ["timeout", "rate_limit"],
    "slow_traces": 23,
}, agent="trace_analyst")

# context_engineer reads and builds on them
trace_findings = workspace.read("trace_analysis_summary")
# Uses trace_findings to target optimizations
```

### 2. Skills-Driven Code Generation
Skills contain complete code patterns that agents use:

```
.claude/skills/
├── mlflow-evaluation/
│   └── SKILL.md          # Scorer patterns, dataset formats, eval templates
├── trace-analysis/
│   └── SKILL.md          # Query syntax, span analysis, error classification
└── context-engineering/
    └── SKILL.md          # Prompt architecture, context rot, state management
```

### 3. Dynamic Evaluation Script Generation
The agent generates complete, runnable evaluation scripts:

```python
# Agent generates this dynamically based on analysis
#!/usr/bin/env python3
import mlflow
from mlflow.genai.scorers import Guidelines, Safety, scorer
from mlflow.entities import Feedback

# Custom scorer based on trace analysis findings
@scorer
def response_quality_scorer(inputs, outputs):
    # Logic derived from failure patterns...
    pass

SCORERS = [
    Safety(),
    Guidelines(name="helpful", guidelines="..."),
    response_quality_scorer,
]

# Dataset extracted from production failures
DATASET = [...]

# Thresholds for CI/CD
THRESHOLDS = {
    "safety/score/mean": 0.95,
    "helpful/score/mean": 0.80,
}
```

## Installation

```bash
pip install claude-agent-sdk mlflow[databricks]
```

## MCP Server Setup

This project requires the official MLflow MCP Server for trace analysis capabilities.

### Claude Code Configuration

Add the following to your Claude Code settings (`.claude/settings.local.json` or project settings):

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

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MLFLOW_TRACKING_URI` | MLflow tracking server URI | `databricks` |
| `DATABRICKS_HOST` | Databricks workspace URL | Required for Databricks |
| `DATABRICKS_TOKEN` | Databricks access token | Required for Databricks |

### Available MCP Tools

Once configured, the following tools become available:

| Tool | Description |
|------|-------------|
| `mcp__mlflow-mcp__search_traces` | Search traces with filters |
| `mcp__mlflow-mcp__get_trace` | Get detailed trace by ID |
| `mcp__mlflow-mcp__search_experiments` | List MLflow experiments |

### Query Syntax Reference

```python
# Status filter
"attributes.status = 'OK'"

# Trace name (note backticks for dots)
"attributes.`mlflow.traceName` = 'my_agent'"

# Time range (milliseconds)
"timestamp_ms > 1699000000000"

# Combined filters
"attributes.status = 'ERROR' AND timestamp_ms > 1699000000000"
```

## Project Structure

```
mlflow-eval-agent/
├── README.md
├── pyproject.toml
│
├── src/
│   └── agent.py                  # Main agent implementation
│
├── .claude/
│   ├── commands/                 # Slash commands for Claude Code
│   │   ├── analyze-trace.md     # Single trace analysis
│   │   ├── eval-analyze-traces.md  # Multi-trace analysis
│   │   ├── optimize-context.md  # Context optimization loop
│   │   └── test-agent.md        # Run agent tests
│   │
│   └── skills/                   # Skills loaded by all agents
│       ├── mlflow-evaluation/
│       │   └── SKILL.md         # Evaluation patterns, scorers
│       ├── trace-analysis/
│       │   └── SKILL.md         # Trace queries, analysis
│       └── context-engineering/
│           └── SKILL.md         # Context optimization
│
├── deployment/
│   ├── databricks/
│   │   ├── job_config.yaml      # Databricks Job
│   │   ├── serving_config.yaml  # Model Serving
│   │   └── app_config.yaml      # Databricks App
│   └── docker/
│       └── Dockerfile
│
└── tests/
    └── test_agent.py
```

## Usage

### Interactive Mode
```bash
python -m agent --interactive
```

```
MLflow Evaluation Agent (Interactive Mode)
Type 'quit' to exit
--------------------------------------------------

You: Analyze my RAG agent's traces from the last hour and create an evaluation suite

Agent: I'll perform a comprehensive analysis using our specialized sub-agents.

[Invoking trace_analyst...]
Found 234 traces in the last hour:
- Success rate: 87%
- Avg latency: 2.3s
- Top errors: context_overflow (15), retrieval_empty (8)

[Writing findings to workspace...]

[Invoking context_engineer...]
Reading trace findings from workspace...
Identified issues:
1. Context overflow in 15 traces - RAG context exceeds token budget
2. Poor grounding in 23 traces - responses not citing sources

[Generating evaluation script...]

Here's your evaluation script:

```python
#!/usr/bin/env python3
# ... complete runnable script ...
```

[Cost: $0.0342]
```

### Programmatic Usage
```python
from agent import MLflowEvalAgent

agent = MLflowEvalAgent()

# Full analysis workflow
script = await agent.analyze_and_generate(
    filter_string="attributes.timestamp_ms > 1699000000000",
    agent_name="my_rag_agent"
)

# Save and run
with open("eval_my_rag_agent.py", "w") as f:
    f.write(script)

# Run evaluation
!python eval_my_rag_agent.py --check-thresholds
```

### Single Query
```bash
python -m agent "Create a safety scorer for customer support responses"
```

## Sub-Agent Details

### trace_analyst
**Triggers**: traces, logs, production, debugging, errors, performance

**Capabilities**:
- Search traces with complex filters
- Classify errors by type
- Profile latency (P50/P95/P99)
- Extract failing cases for evaluation
- Analyze tool call patterns
- Detect context rot signals

**Writes to Workspace**:
- `trace_analysis_summary`
- `error_patterns`
- `performance_metrics`
- `extracted_eval_cases`

### context_engineer
**Triggers**: optimize, prompt, context, tokens, state, memory, RAG

**Capabilities**:
- System prompt optimization
- Few-shot example design
- RAG context formatting
- State management analysis
- Token budget optimization
- Context rot detection & prevention

**Reads from Workspace**: trace_analyst findings
**Writes to Workspace**: `context_recommendations`

### agent_architect
**Triggers**: architecture, multi-agent, design, performance, tool orchestration

**Capabilities**:
- Map agent architecture
- Identify bottlenecks
- Recommend structural changes
- Optimize tool usage
- Suggest parallelization

**Reads from Workspace**: All previous findings

## Deployment Options

### 1. Databricks Job (Scheduled/Batch)
```yaml
# deployment/databricks/job_config.yaml
name: mlflow-eval-agent-job
tasks:
  - task_key: run_analysis
    python_wheel_task:
      package_name: mlflow_eval_agent
      entry_point: analyze
      parameters:
        - "--filter"
        - "attributes.timestamp_ms > {{start_time}}"
    
schedule:
  quartz_cron_expression: "0 0 * * * ?"
```

### 2. Model Serving Endpoint (Interactive API)
```yaml
# deployment/databricks/serving_config.yaml
name: mlflow-eval-agent-endpoint
config:
  served_entities:
    - entity_name: mlflow_eval_agent
      workload_type: CPU
      scale_to_zero_enabled: true
```

### 3. Docker Container
```bash
docker build -t mlflow-eval-agent .
docker run -e DATABRICKS_HOST=... -e DATABRICKS_TOKEN=... mlflow-eval-agent
```

## Comparison: Claude Code vs Agent SDK

| Feature | Claude Code | Agent SDK (This) |
|---------|-------------|------------------|
| File editing | ✅ Direct | ❌ Via tools |
| Multi-user | ❌ Single | ✅ Per-instance |
| Deployment | ❌ Local | ✅ Cloud/container |
| API access | ❌ CLI only | ✅ Python SDK |
| Batch jobs | ❌ Interactive | ✅ Scheduled |
| Cost tracking | ❌ Manual | ✅ Built-in |
| Skills | ✅ Same | ✅ Same |

**Recommendation**: Use Claude Code for development, Agent SDK for deployment.

## Limitations

1. **No Direct File Editing**: Agent suggests changes but doesn't edit files directly
2. **Context Window**: Large trace results must be paginated
3. **State Per-Session**: Workspace resets between sessions
4. **Claude Code CLI Required**: Need Node.js + CLI installed

## Example Workflows

### Workflow 1: Debug Production Issues
```
You: My agent has been failing more in the last hour. What's wrong?

Agent: 
[trace_analyst] Analyzing last hour...
Found spike in errors at 14:30 UTC:
- 45 failures (was 5/hour baseline)
- All failures: context_overflow errors
- Root cause: Large document uploaded at 14:28

Recommendation: Implement chunking for documents > 10KB
```

### Workflow 2: Create Evaluation Suite
```
You: Create an evaluation suite for my customer support chatbot

Agent:
[Analyzing agent code...]
[Generating scorers based on capabilities...]

Generated 5 scorers:
1. safety - Built-in safety check
2. helpful - Does response address the issue?
3. professional_tone - Appropriate language?
4. resolution_offered - Actionable solution provided?
5. no_pii_leaked - No customer data exposed?

Generated 20 test cases covering:
- Common questions (10)
- Edge cases (5)
- Adversarial inputs (5)

[Generating evaluation script...]
```

### Workflow 3: Optimize Context
```
You: My agent's latency has been increasing over conversations

Agent:
[trace_analyst] Analyzing conversation traces...
Context rot detected:
- Token usage: +150% over 10 turns
- Latency: +80% over 10 turns

[context_engineer] Reading findings...
Issues:
1. Full conversation history retained (no pruning)
2. RAG context duplicated across turns
3. No summarization of old context

Recommendations:
1. Implement sliding window (keep last 5 turns)
2. Add conversation summarization
3. Deduplicate RAG context

Generated context manager code:
```python
# ... optimized context management ...
```
```

## Contributing

1. Add new skills in `skills/` directory
2. Add new tools in `create_tools()` function
3. Add new sub-agents in `create_subagents()` function

## License

MIT
