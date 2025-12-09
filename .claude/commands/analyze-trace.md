---
description: Analyze MLflow trace using MCP server and/or Python script
argument-hint: "[trace-id]"
model: claude-sonnet-4-5-20250929
---

Analyze trace: **$ARGUMENTS**

## Configuration

This command requires:
- MLflow MCP server configured (see MCP Server Setup below)
- Optional: Custom trace analysis script in your project

## Analysis Strategy

Use MLflow MCP server for comprehensive trace analysis:

### Phase 1: Quick Overview (MLflow MCP)

1. Use MCP tool to fetch trace data:
   - `mcp__mlflow-mcp__get_trace` with trace_id: $ARGUMENTS

2. Extract high-level insights:
   - Trace status and execution time
   - Number of spans
   - Key attributes and metadata

### Phase 2: Deep Dive (Span Analysis)

3. Analyze span hierarchy and timing:
   - Use `mcp__mlflow-mcp__get_trace` with detailed span extraction
   - Or run a project-specific analysis script if available:
     ```bash
     # Example: python scripts/analyze_trace.py $ARGUMENTS
     ```

4. Look for:
   - Individual span timing and duration
   - Input/output data at each step
   - Attribute patterns
   - Context compression effectiveness (if compression spans present)

## Report Format

Provide a structured markdown report:

```markdown
# Trace Analysis: $ARGUMENTS

## Summary
- Status: [OK/ERROR]
- Execution Time: [X.XX seconds]
- Total Spans: [N]

## Key Findings
1. [Finding 1]
2. [Finding 2]
...

## Performance Insights
- Token usage efficiency
- Tool call effectiveness
- Context compression impact (if applicable)

## Optimization Opportunities
- [Recommendation 1]
- [Recommendation 2]

## Next Steps
- [Suggested action 1]
- [Suggested action 2]
```

## Important Notes

- Focus on actionable insights, not raw data dumps
- Identify patterns (e.g., repeated tool calls, context growth)
- Relate findings back to agent design principles
- Suggest specific code or config changes if relevant

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
