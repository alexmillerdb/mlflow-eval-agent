---
description: Analyze traces from an experiment or list of trace IDs
argument-hint: "[experiment-id | trace-ids (comma-separated)]"
model: claude-sonnet-4-5-20250929
---

Analyze traces: **$ARGUMENTS**

## Objective

Analyze multiple traces to identify patterns, bottlenecks, and quality issues across an agent's executions. This command is architecture-agnostic and works with DSPy multi-agent, LangGraph, RAG, or simple tool-calling agents.

## Input Handling

The argument can be:
1. **Experiment ID**: A numeric ID like `3542653191523636`
2. **Comma-separated trace IDs**: Like `tr-abc123,tr-def456,tr-ghi789`
3. **Experiment name**: Like `/Users/user@domain.com/my-experiment`

## Analysis Steps

### Step 1: Fetch Traces

**If argument looks like experiment ID (numeric) or path:**
```
mcp__mlflow-mcp__search_traces(
  experiment_id="$ARGUMENTS",
  max_results=5,
  output="json"
)
```

**If argument contains comma (trace IDs):**
For each trace ID, fetch:
```
mcp__mlflow-mcp__get_trace(
  trace_id="<trace_id>",
  extract_fields="info.trace_id,info.status,info.execution_time_ms,data.spans.*.name,data.spans.*.span_type"
)
```

### Step 2: Architecture Detection

Detect agent architecture by analyzing span patterns:

| Architecture | Indicators |
|-------------|------------|
| DSPy Multi-Agent | Spans named: classifier, rewriter, gatherer, executor |
| LangGraph | Spans named: graph, node, state, langgraph |
| RAG | RETRIEVER span type present |
| Tool-Calling | TOOL span type present |
| Simple Chat | Only LLM/CHAT_MODEL spans |

### Step 3: Latency Analysis

Calculate:
- Total execution time (avg, p50, p95, max)
- Latency by span type (LLM, TOOL, RETRIEVER, etc.)
- Latency by component/stage name
- Identify bottlenecks (spans taking >50% of total time)

### Step 4: Error Analysis

Identify:
- Failed traces (status != OK)
- Failed spans within traces
- Common error patterns
- Retry patterns

### Step 5: Tool/LLM Usage Patterns

Analyze:
- Tool call frequency and distribution
- LLM call count per trace
- Token usage (if available in attributes)
- Tool success/failure rates

## Report Format

```markdown
# Trace Analysis Report

## Summary
- **Traces Analyzed**: N
- **Time Range**: [start] to [end]
- **Architecture Detected**: [architecture type]
- **Success Rate**: X%

## Latency Analysis

| Metric | Value |
|--------|-------|
| Avg Total | X.XXs |
| P50 | X.XXs |
| P95 | X.XXs |
| Max | X.XXs |

### Latency by Component
| Component | Avg (ms) | % of Total |
|-----------|----------|------------|
| [component1] | XXX | XX% |
| [component2] | XXX | XX% |
| ... | ... | ... |

### Bottleneck
**[Component name]** accounts for XX% of total latency.

## Tool Usage
- Total tool calls: N
- Unique tools: N
- Most used: [tool name] (X calls)
- Tool success rate: X%

## LLM Usage
- Total LLM calls: N avg per trace
- Estimated tokens: X input / Y output per trace

## Error Patterns
[If errors exist, list common patterns]

## Recommendations

1. **[Category]**: [Specific recommendation]
2. **[Category]**: [Specific recommendation]
...

## Next Steps
- [ ] [Action item 1]
- [ ] [Action item 2]
```

## Example Usage

```
/eval-analyze-traces 3542653191523636
/eval-analyze-traces tr-abc123,tr-def456
/eval-analyze-traces /Users/<your-username>/my-experiment
```

## Integration with Other Commands

After running this analysis:
- Use `/analyze-trace [trace-id]` for deep dive on specific traces
- Use `/optimize-context` for prompt optimization based on findings
- Run evaluation with the same experiment to track improvements

## MCP Server Setup

Required MCP configuration in `.claude/settings.local.json` or project settings:
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
