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

## Analysis Strategy

Run the Python batch analysis script:

```bash
python scripts/analyze_traces_batch.py $ARGUMENTS
```

The script outputs JSON with:
- Summary (trace count, success rate, architecture distribution)
- Latency statistics (avg, min, max, p50, p95)
- Span type aggregation (counts and latencies)
- LLM usage (total calls, tokens)
- Tool usage (total calls, avg per trace)
- Error patterns (failed traces, common failed spans)
- Common bottlenecks across traces
- Recommendations

## Report Format

Parse the JSON output and format as markdown:

```markdown
# Trace Analysis Report

## Summary
| Field | Value |
|-------|-------|
| Traces Analyzed | [N] |
| Success Rate | [X]% |
| Architectures | [distribution] |

## Latency Analysis

| Metric | Value |
|--------|-------|
| Average | [X.XX]ms |
| P50 | [X.XX]ms |
| P95 | [X.XX]ms |
| Min | [X.XX]ms |
| Max | [X.XX]ms |

## Span Types

| Type | Count | Total Latency (ms) |
|------|-------|-------------------|
| [type] | [n] | [ms] |

## LLM Usage
- Total LLM calls: [N]
- Avg calls per trace: [X]
- Total input tokens: [N]
- Total output tokens: [N]

## Tool Usage
- Total tool calls: [N]
- Avg calls per trace: [X]

## Common Bottlenecks

| Component | Occurrences | Avg Duration (ms) |
|-----------|-------------|-------------------|
| [name] | [n] | [ms] |

## Error Patterns
- Failed traces: [N]
- Common failed spans: [list]

## Recommendations
[List recommendations from the report]

## Individual Traces

| Trace ID | Status | Duration (ms) | Architecture |
|----------|--------|---------------|--------------|
| [id] | [status] | [ms] | [arch] |

## Next Steps
- [ ] Use `/analyze-trace [trace-id]` for deep dive on specific traces
- [ ] Use `/optimize-context` for prompt optimization based on findings
```

## Fallback: MCP Server

If the Python script fails, fall back to MCP tools:

**For experiment ID:**
```
mcp__mlflow-eval__search_traces(
  experiment_id="$ARGUMENTS",
  max_results=10,
  output="json"
)
```

**For trace IDs:**
For each trace ID, fetch:
```
mcp__mlflow-eval__get_trace(
  trace_id="<trace_id>",
  extract_fields="info.trace_id,info.status,info.execution_time_ms,data.spans.*.name,data.spans.*.span_type"
)
```

## Example Usage

```
/eval-analyze-traces 3542653191523636
/eval-analyze-traces tr-abc123,tr-def456,tr-ghi789
```

## Integration with Other Commands

After running this analysis:
- Use `/analyze-trace [trace-id]` for deep dive on specific traces
- Use `/optimize-context` for prompt optimization based on findings
- Run evaluation with the same experiment to track improvements
