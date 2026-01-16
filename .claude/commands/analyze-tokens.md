---
description: Analyze token usage and cost across agent sessions
argument-hint: "[trace-id]"
model: claude-sonnet-4-5-20250929
---

Analyze token usage and cost for an agent trace.

**Argument**: $ARGUMENTS

## Mode Detection

- If argument starts with `tr-`: Analyze the specified trace ID
- If no argument or empty: Run a quick test first, then analyze the result

## Instructions

### Mode 1: Analyze Existing Trace (trace ID provided)

Run the analysis script with `--tokens` flag:

```bash
uv run python scripts/analyze_trace.py $ARGUMENTS --tokens
```

### Mode 2: Run Test + Analyze (no trace ID)

1. Run a quick agent test to generate a trace:

```bash
uv run python -c "
import asyncio
import mlflow
from src.agent import MLflowAgent, setup_mlflow

setup_mlflow()

async def test():
    agent = MLflowAgent()
    async for result in agent.query('Say hello in 5 words'):
        if result.event_type == 'result':
            print(f'Response: {result.response}')
            print(f'Cost: \${result.cost_usd:.4f}' if result.cost_usd else '')

    trace_id = mlflow.get_last_active_trace_id()
    print(f'Trace ID: {trace_id}')

asyncio.run(test())
"
```

2. Extract the trace ID from output

3. Run the token analysis:
```bash
uv run python scripts/analyze_trace.py <trace-id> --tokens
```

## Report Format

Present the analysis as a table:

| Session | Phase | Input | Output | Cache Read | Cache Create | Cost |
|---------|-------|-------|--------|------------|--------------|------|
| session_1 | initializer | 15,234 | 892 | 0 | 12,456 | $0.0234 |
| session_2 | worker | 18,102 | 1,234 | 11,234 | 1,234 | $0.0189 |
| **Total** | - | **33,336** | **2,126** | **11,234** | **13,690** | **$0.0423** |

## Key Metrics to Highlight

1. **Cache Efficiency** = Cache Read / (Input + Cache Read + Cache Create)
   - > 50% = Good caching
   - < 20% = Poor caching, prompts changing too much

2. **Token Growth** - Is input growing each session?
   - Stable = Good context management
   - Growing = Context accumulation issue

3. **Cost per Session** - Track trends
   - Decreasing = Cache warming up
   - Increasing = More complex tasks

## Recommendations

Based on the analysis, provide actionable recommendations:

- If cache efficiency < 20%: "Consider stabilizing prompts to improve cache hit rate"
- If tokens growing: "Review context management - consider summarization"
- If high cache creation: "First session is expensive but subsequent should improve"
- If high output tokens: "Agent is verbose - consider prompt tuning"
