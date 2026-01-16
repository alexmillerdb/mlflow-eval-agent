---
description: Run agent test and capture trace ID for analysis
argument-hint: "[prompt]"
model: claude-sonnet-4-5-20250929
---

Run a quick agent test and capture the trace ID.

**Prompt**: $ARGUMENTS (default: "Say hello in 5 words")

## Instructions

1. Run the test using the simplified MLflowAgent:

```bash
uv run python -c "
import asyncio
import mlflow
from src.agent import MLflowAgent, setup_mlflow

setup_mlflow()

async def test():
    agent = MLflowAgent()
    prompt = '''$ARGUMENTS''' if '''$ARGUMENTS'''.strip() else 'Say hello in 5 words'

    async for result in agent.query(prompt):
        if result.event_type == 'result':
            print(f'Response: {result.response}')
            print(f'Cost: \${result.cost_usd:.4f}' if result.cost_usd else '')
            if result.usage_data:
                usage = result.usage_data
                print(f'Tokens - Input: {usage.get(\"input_tokens\", 0)}, Output: {usage.get(\"output_tokens\", 0)}, Cache Read: {usage.get(\"cache_read_input_tokens\", 0)}')

    trace_id = mlflow.get_last_active_trace_id()
    print(f'\\nTrace ID: {trace_id}')
    return trace_id

asyncio.run(test())
"
```

2. Extract and report:
   - Response preview
   - Cost and token usage
   - **Trace ID** (critical for follow-up analysis)

3. Suggest next step:
   - "Use `/analyze-tokens <trace-id>` to analyze token usage in detail"
   - "Use `/analyze-trace <trace-id>` for latency and bottleneck analysis"
