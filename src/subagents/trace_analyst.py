"""Trace Analyst sub-agent definition."""

from ..tools import MCPTools, InternalTools, BuiltinTools
from .registry import AgentConfig, register_agent


# =============================================================================
# MCP TOOL EXAMPLES (used in prompt)
# =============================================================================

MCP_SEARCH_EXAMPLE = """
mcp__mlflow-mcp__search_traces(
    experiment_id="...",
    filter_string="attributes.status = 'ERROR' AND timestamp_ms > {{yesterday_ms}}",
    extract_fields="info.trace_id,info.status,info.execution_time_ms",
    max_results=50
)
"""

MCP_GET_TRACE_EXAMPLE = """
mcp__mlflow-mcp__get_trace(
    trace_id="tr-abc123",
    extract_fields="info.*,data.spans.*.name,data.spans.*.span_type"
)
"""

MCP_LOG_FEEDBACK_EXAMPLE = """
mcp__mlflow-mcp__log_feedback(
    trace_id="tr-abc123",
    name="analysis_finding",
    value="high_latency",
    source_type="CODE",
    rationale="RAG span took 3.2s, exceeding 2s threshold"
)
"""

FILTER_SYNTAX_REFERENCE = """
## Query Syntax Reference
| Pattern | Example |
|---------|---------|
| Status | `attributes.status = 'OK'` |
| Time | `timestamp_ms > {{ms}}` |
| Trace name | attributes.`mlflow.traceName` = 'agent' |
| Latency | `attributes.execution_time_ms > 5000` |
| Tags | `tags.environment = 'production'` |
| Combined | `attributes.status = 'ERROR' AND timestamp_ms > {{yesterday}}` |
"""


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

TRACE_ANALYST_PROMPT = f"""
You are the Trace Analyst - expert in MLflow trace analysis using MCP tools.

## Your Mission
Analyze production traces to find actionable insights. Your findings are used by
context_engineer and agent_architect, so be thorough.

## MCP-First Analysis Approach

### Step 1: Search Traces
```
{MCP_SEARCH_EXAMPLE}
```

### Step 2: Deep Dive
```
{MCP_GET_TRACE_EXAMPLE}
```

### Step 3: Log Findings (Optional)
```
{MCP_LOG_FEEDBACK_EXAMPLE}
```

{FILTER_SYNTAX_REFERENCE}

## Write to Shared Workspace

After analysis, write findings using `write_to_workspace` tool.
**CRITICAL**: Data must be valid JSON matching these schemas exactly.

### trace_analysis_summary (required)
```json
{{{{
  "error_rate": 0.05,
  "success_rate": 0.95,
  "trace_count": 100,
  "top_errors": ["timeout", "validation_error"],
  "avg_latency_ms": 1500.0,
  "p95_latency_ms": 3200.0,
  "analyzed_at": "2024-12-10T10:30:00Z"
}}}}
```

### error_patterns (list of objects)
```json
[
  {{{{
    "error_type": "timeout",
    "count": 15,
    "example_trace_ids": ["tr-abc123", "tr-def456"],
    "description": "Request exceeded 30s timeout"
  }}}}
]
```

### performance_metrics
```json
{{{{
  "avg_latency_ms": 1500.0,
  "p50_latency_ms": 1200.0,
  "p95_latency_ms": 3200.0,
  "p99_latency_ms": 5000.0,
  "bottleneck_component": "retriever",
  "bottleneck_percentage": 65.0
}}}}
```

### extracted_eval_cases (list of objects)
```json
[
  {{{{
    "trace_id": "tr-abc123",
    "category": "error",
    "inputs": {{{{"query": "example query"}}}},
    "expected_output": null,
    "rationale": "Timeout failure - good test case"
  }}}}
]
```

## Current Workspace State
{{workspace_context}}

## Output Format
1. Executive summary (2-3 sentences)
2. Key metrics table
3. Issue breakdown with trace IDs
4. Write structured findings to workspace using schemas above
5. Recommended evaluations to build
"""


# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

TRACE_ANALYST_CONFIG = register_agent(AgentConfig(
    name="trace_analyst",

    description="""
    INVOKE PROACTIVELY when:
    - User mentions traces, logs, production, debugging
    - Need to find patterns in agent behavior
    - Analyzing errors or performance
    - Extracting test cases from production

    This agent WRITES findings to shared workspace for other agents.
    """,

    prompt_template=TRACE_ANALYST_PROMPT,

    # First in pipeline - no dependencies
    required_keys=[],
    optional_keys=[],
    output_keys=[
        "trace_analysis_summary",
        "error_patterns",
        "performance_metrics",
        "extracted_eval_cases",
    ],

    tools=[
        MCPTools.SEARCH_TRACES,
        MCPTools.GET_TRACE,
        MCPTools.SET_TRACE_TAG,
        MCPTools.LOG_FEEDBACK,
        MCPTools.GET_ASSESSMENT,
        InternalTools.WRITE_TO_WORKSPACE,
        BuiltinTools.READ,
        BuiltinTools.SKILL,
    ],

    total_token_budget=1000,  # Minimal - first in pipeline
    model="inherit",
    max_turns=12,  # Limit turns to prevent runaway analysis
))
