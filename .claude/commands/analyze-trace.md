---
description: Analyze MLflow trace using Python script with MLflow 3 SDK
argument-hint: "[trace-id]"
model: claude-sonnet-4-5-20250929
---

Analyze trace: **$ARGUMENTS**

## Analysis Strategy

Run the Python analysis script to get comprehensive trace analysis:

```bash
# Summary analysis (default)
python scripts/analyze_trace.py $ARGUMENTS

# Include full span data with inputs/outputs for detailed review
python scripts/analyze_trace.py $ARGUMENTS --full

# Raw trace data for direct analysis
python scripts/analyze_trace.py $ARGUMENTS --raw
```

**Use `--full` when you need to see actual LLM messages, tool inputs/outputs, and span details.**
**Use `--raw` for complete trace data extraction for custom analysis.**

The script outputs JSON with:
- Summary (trace_id, status, duration, span count)
- Architecture detection (DSPy, LangGraph, RAG, Tool-Calling)
- Latency breakdown by span type
- Top 5 bottlenecks
- Error analysis (failed spans, exceptions)
- Tool call statistics
- LLM call statistics (including tokens)
- Recommendations

With `--full`, also includes:
- Full span details with inputs/outputs
- LLM messages and responses
- Tool call inputs/outputs

With `--raw`:
- Complete trace JSON for custom analysis

## Report Format

Parse the JSON output and format as markdown:

```markdown
# Trace Analysis: [trace_id]

## Summary
| Field | Value |
|-------|-------|
| Status | [OK/ERROR] |
| Execution Time | [X.XX]ms |
| Total Spans | [N] |
| Architecture | [detected architecture] |

## Latency by Span Type

| Span Type | Count | Total (ms) | Avg (ms) |
|-----------|-------|------------|----------|
| [type] | [n] | [total] | [avg] |

## Bottlenecks

Top 5 slowest components:

| Rank | Component | Type | Duration (ms) |
|------|-----------|------|---------------|
| 1 | [name] | [type] | [ms] |

## LLM Usage
- Total LLM calls: [N]
- Total latency: [X]ms
- Input tokens: [N]
- Output tokens: [N]

## Tool Usage
- Total tool calls: [N]
- Unique tools: [N]

## Errors
[If any failed spans, list them with error messages]

## Recommendations
[List recommendations from the report]

## Next Steps
- [ ] [Action based on recommendations]
```

## Fallback: MCP Server

If the Python script fails, fall back to MCP tools:

```
mcp__mlflow-eval__get_trace(
  trace_id="$ARGUMENTS",
  extract_fields="info.trace_id,info.status,info.execution_time_ms,data.spans.*.name,data.spans.*.span_type"
)
```

## Important Notes

- Focus on actionable insights, not raw data dumps
- Identify patterns (e.g., repeated tool calls, context growth)
- Relate findings back to agent design principles
- Suggest specific code or config changes if relevant
