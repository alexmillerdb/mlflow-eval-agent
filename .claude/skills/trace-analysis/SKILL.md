---
name: trace-analysis
description: Deep analysis of MLflow traces to understand GenAI agent behavior, architecture, and performance. Use when analyzing traces to understand agent architecture (DSPy/LangGraph/RAG/tool-calling), profiling latency and token usage, debugging failures, analyzing RAG retrieval quality, examining tool call patterns, or extracting test cases. This skill is designed for sub-agent use due to high context requirements.
---

# MLflow Trace Analysis

Comprehensive trace analysis for understanding and optimizing GenAI agents.

## Sub-Agent Design

This skill is designed for use by a dedicated trace analysis sub-agent that:
1. Receives trace IDs or experiment filters from the coordinator
2. Performs deep analysis using the patterns below
3. Writes structured findings to shared workspace
4. Returns concise summary to coordinator

## Analysis Framework

### Phase 1: Architecture Detection

Analyze span patterns to determine agent type:

```python
def detect_architecture(trace):
    """Identify agent architecture from trace structure."""
    spans = trace.data.spans
    names = [s.name.lower() for s in spans]
    types = [str(s.span_type) for s in spans]

    # Check for framework indicators
    indicators = {
        "dspy_multi_agent": any(p in " ".join(names)
            for p in ["classifier", "rewriter", "gatherer", "executor", "signature"]),
        "langgraph": any(p in " ".join(names)
            for p in ["langgraph", "graph", "node", "stategraph"]),
        "langchain": any(p in " ".join(names)
            for p in ["langchain", "chain", "lcel"]),
        "rag": "RETRIEVER" in types,
        "tool_calling": "TOOL" in types,
        "multi_llm": types.count("LLM") + types.count("CHAT_MODEL") > 3,
    }

    # Determine primary architecture
    if indicators["dspy_multi_agent"]:
        arch = "dspy_multi_agent"
        components = extract_dspy_stages(spans)
    elif indicators["langgraph"]:
        arch = "langgraph"
        components = extract_graph_nodes(spans)
    elif indicators["rag"] and indicators["tool_calling"]:
        arch = "rag_with_tools"
        components = {"retriever": True, "tools": True}
    elif indicators["rag"]:
        arch = "rag"
        components = extract_rag_components(spans)
    elif indicators["tool_calling"]:
        arch = "tool_calling"
        components = extract_tool_calls(spans)
    else:
        arch = "simple_chat"
        components = {}

    return {
        "architecture": arch,
        "indicators": indicators,
        "components": components,
        "span_count": len(spans),
        "span_type_distribution": {t: types.count(t) for t in set(types)}
    }
```

### Phase 2: Latency Profiling

Analyze timing across all dimensions:

```python
def profile_latency(trace):
    """Comprehensive latency breakdown."""
    spans = trace.data.spans

    # Get total execution time
    root = next((s for s in spans if s.parent_id is None), None)
    total_ms = (root.end_time_ns - root.start_time_ns) / 1e6 if root else 0

    # By span type
    by_type = {}
    for span in spans:
        t = str(span.span_type) or "UNKNOWN"
        dur = (span.end_time_ns - span.start_time_ns) / 1e6
        if t not in by_type:
            by_type[t] = {"count": 0, "total_ms": 0, "durations": []}
        by_type[t]["count"] += 1
        by_type[t]["total_ms"] += dur
        by_type[t]["durations"].append(dur)

    # Calculate percentiles
    for t in by_type:
        durs = sorted(by_type[t]["durations"])
        by_type[t]["p50_ms"] = durs[len(durs)//2] if durs else 0
        by_type[t]["p95_ms"] = durs[int(len(durs)*0.95)] if durs else 0
        by_type[t]["pct_of_total"] = by_type[t]["total_ms"] / total_ms * 100 if total_ms else 0
        del by_type[t]["durations"]  # Remove raw data

    # Find bottlenecks (>20% of total time)
    bottlenecks = [
        {"span_type": t, "pct": v["pct_of_total"]}
        for t, v in by_type.items()
        if v["pct_of_total"] > 20
    ]

    return {
        "total_ms": round(total_ms, 2),
        "by_span_type": by_type,
        "bottlenecks": bottlenecks,
        "llm_time_pct": by_type.get("LLM", {}).get("pct_of_total", 0) +
                        by_type.get("CHAT_MODEL", {}).get("pct_of_total", 0)
    }
```

### Phase 3: Token Analysis

Analyze token consumption across LLM calls:

```python
def analyze_tokens(trace):
    """Extract token usage from LLM spans."""
    llm_spans = [s for s in trace.data.spans
                 if s.span_type in ["LLM", "CHAT_MODEL"]]

    calls = []
    for span in llm_spans:
        attrs = span.attributes or {}
        calls.append({
            "name": span.name,
            "model": attrs.get("mlflow.chat_model.model") or
                     attrs.get("llm.model_name"),
            "input_tokens": attrs.get("mlflow.chat_model.input_tokens"),
            "output_tokens": attrs.get("mlflow.chat_model.output_tokens"),
            "latency_ms": (span.end_time_ns - span.start_time_ns) / 1e6
        })

    total_input = sum(c["input_tokens"] or 0 for c in calls)
    total_output = sum(c["output_tokens"] or 0 for c in calls)

    return {
        "total_llm_calls": len(calls),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "calls": calls,
        "tokens_per_call_avg": (total_input + total_output) / len(calls) if calls else 0
    }
```

### Phase 4: Context Analysis

Analyze context window utilization for RAG and prompts:

```python
def analyze_context(trace, context_limit=128000):
    """Analyze context window usage and potential issues."""
    issues = []

    # Token usage per LLM call
    token_analysis = analyze_tokens(trace)
    for call in token_analysis["calls"]:
        if call["input_tokens"] and call["input_tokens"] > context_limit * 0.8:
            issues.append({
                "type": "context_overflow_risk",
                "span": call["name"],
                "input_tokens": call["input_tokens"],
                "pct_of_limit": call["input_tokens"] / context_limit * 100
            })

    # RAG context analysis
    retriever_spans = [s for s in trace.data.spans if s.span_type == "RETRIEVER"]
    for span in retriever_spans:
        outputs = span.outputs
        if isinstance(outputs, list) and len(outputs) > 10:
            issues.append({
                "type": "rag_over_retrieval",
                "span": span.name,
                "chunks_retrieved": len(outputs),
                "recommendation": "Consider limiting to top 5-7 chunks"
            })

        # Check for empty retrievals
        if not outputs or outputs == []:
            issues.append({
                "type": "rag_empty_retrieval",
                "span": span.name,
                "recommendation": "Check retrieval query and index"
            })

    # Context growth detection (multi-turn)
    llm_calls_tokens = [c["input_tokens"] for c in token_analysis["calls"]
                        if c["input_tokens"]]
    if len(llm_calls_tokens) > 2:
        growth = (llm_calls_tokens[-1] - llm_calls_tokens[0]) / llm_calls_tokens[0] * 100 \
                 if llm_calls_tokens[0] else 0
        if growth > 50:
            issues.append({
                "type": "context_rot",
                "growth_pct": round(growth, 1),
                "recommendation": "Implement context compression or sliding window"
            })

    return {
        "total_context_tokens": token_analysis["total_input_tokens"],
        "issues": issues,
        "context_utilization_pct": token_analysis["total_input_tokens"] / context_limit * 100
                                   if token_analysis["total_input_tokens"] else 0
    }
```

### Phase 5: RAG Quality Analysis

For RAG agents, analyze retrieval effectiveness:

```python
def analyze_rag(trace):
    """Analyze RAG retrieval quality indicators."""
    retriever_spans = [s for s in trace.data.spans if s.span_type == "RETRIEVER"]

    if not retriever_spans:
        return {"has_rag": False}

    retrieval_metrics = []
    for span in retriever_spans:
        dur = (span.end_time_ns - span.start_time_ns) / 1e6
        outputs = span.outputs if span.outputs else []
        inputs = span.inputs if span.inputs else {}

        # Extract query
        query = inputs.get("query") or inputs.get("question") or str(inputs)[:100]

        retrieval_metrics.append({
            "query": query,
            "chunks_returned": len(outputs) if isinstance(outputs, list) else 1,
            "latency_ms": dur,
            "has_results": bool(outputs)
        })

    return {
        "has_rag": True,
        "retrieval_count": len(retriever_spans),
        "avg_chunks": sum(r["chunks_returned"] for r in retrieval_metrics) / len(retrieval_metrics),
        "empty_retrievals": sum(1 for r in retrieval_metrics if not r["has_results"]),
        "retrievals": retrieval_metrics
    }
```

### Phase 6: Tool Call Analysis

For tool-calling agents, analyze tool usage patterns:

```python
from mlflow.entities import SpanStatusCode

def analyze_tools(trace):
    """Analyze tool call patterns and effectiveness."""
    tool_spans = [s for s in trace.data.spans if s.span_type == "TOOL"]

    if not tool_spans:
        return {"has_tools": False}

    tools = {}
    for span in tool_spans:
        # Normalize tool name (handle fully qualified)
        name = span.name.split(".")[-1] if "." in span.name else span.name
        dur = (span.end_time_ns - span.start_time_ns) / 1e6
        success = not (span.status and span.status.status_code == SpanStatusCode.ERROR)

        if name not in tools:
            tools[name] = {"count": 0, "successes": 0, "total_ms": 0, "failures": []}
        tools[name]["count"] += 1
        tools[name]["total_ms"] += dur
        if success:
            tools[name]["successes"] += 1
        else:
            tools[name]["failures"].append(span.status.description if span.status else "Unknown")

    # Calculate success rates
    for name in tools:
        tools[name]["success_rate"] = tools[name]["successes"] / tools[name]["count"]
        tools[name]["avg_ms"] = tools[name]["total_ms"] / tools[name]["count"]

    return {
        "has_tools": True,
        "unique_tools": len(tools),
        "total_calls": sum(t["count"] for t in tools.values()),
        "overall_success_rate": sum(t["successes"] for t in tools.values()) /
                                sum(t["count"] for t in tools.values()),
        "tools": tools
    }
```

### Phase 7: Error Analysis

Comprehensive error detection and classification:

```python
def analyze_errors(trace):
    """Detect and classify errors in trace."""
    from mlflow.entities import SpanStatusCode

    errors = {
        "failed_spans": [],
        "exceptions": [],
        "empty_outputs": [],
        "rate_limits": [],
        "timeouts": []
    }

    for span in trace.data.spans:
        # Check span status
        if span.status and span.status.status_code == SpanStatusCode.ERROR:
            error_msg = span.status.description or "Unknown error"

            # Classify error type
            if "rate" in error_msg.lower() or "429" in error_msg:
                errors["rate_limits"].append({"span": span.name, "error": error_msg})
            elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                errors["timeouts"].append({"span": span.name, "error": error_msg})
            else:
                errors["failed_spans"].append({"span": span.name, "error": error_msg})

        # Check for exceptions in events
        if span.events:
            for event in span.events:
                if "exception" in event.name.lower():
                    errors["exceptions"].append({
                        "span": span.name,
                        "exception": event.attributes.get("exception.message", "")
                    })

        # Check for empty outputs (potential silent failure)
        if span.outputs is None or span.outputs == {} or span.outputs == []:
            if span.span_type not in ["CHAIN"]:  # Exclude wrapper spans
                errors["empty_outputs"].append(span.name)

    return {
        "has_errors": any(errors[k] for k in errors),
        "error_count": sum(len(v) for v in errors.values()),
        "errors": errors
    }
```

## Producing Analysis Output

Write findings to workspace in this structure:

```python
workspace.write("trace_analysis_summary", {
    "trace_id": trace.info.trace_id,
    "architecture": arch_result["architecture"],
    "total_latency_ms": latency_result["total_ms"],
    "llm_time_pct": latency_result["llm_time_pct"],
    "total_tokens": token_result["total_tokens"],
    "error_count": error_result["error_count"],
    "bottlenecks": latency_result["bottlenecks"],
    "analyzed_at": datetime.now().isoformat()
}, agent="trace_analyst")

workspace.write("performance_metrics", {
    "latency_by_type": latency_result["by_span_type"],
    "token_usage": token_result,
    "bottleneck_component": latency_result["bottlenecks"][0] if latency_result["bottlenecks"] else None
}, agent="trace_analyst")

workspace.write("context_recommendations", context_result["issues"], agent="trace_analyst")
workspace.write("error_patterns", error_result["errors"], agent="trace_analyst")
```

## MCP Server Tools (for Agent Use)

Use MLflow MCP Server tools for interactive trace exploration and analysis.

### Available Tools

| Tool | Purpose |
|------|---------|
| `search_traces` | Find traces with filters and field extraction |
| `get_trace` | Get detailed trace by ID |
| `set_trace_tag` | Tag traces for dataset building |
| `delete_trace_tag` | Remove tags |
| `log_feedback` | Store analysis findings |
| `log_expectation` | Store ground truth |
| `get_assessment` | Retrieve assessments |
| `update_assessment` | Modify assessments |

### Search Traces with Field Extraction

```
mcp__mlflow-mcp__search_traces(
    experiment_id="123",
    filter_string="attributes.status = 'OK' AND timestamp_ms > {cutoff_ms}",
    extract_fields="info.trace_id,info.status,info.execution_time_ms,data.spans.*.name,data.spans.*.span_type",
    max_results=50
)
```

**extract_fields Options:**
- `info.*` - All trace info
- `info.trace_id,info.status` - Specific info fields
- `data.spans.*.name` - All span names
- `data.spans.*.span_type` - All span types
- `data.spans.*.attributes.mlflow.*` - MLflow span attributes
- `data.request,data.response` - Trace input/output

### Get Trace Details

```
mcp__mlflow-mcp__get_trace(
    trace_id="tr-abc123",
    extract_fields="info.*,data.spans.*"
)
```

### Tag Traces for Dataset Building

```
# Mark trace for inclusion in eval dataset
mcp__mlflow-mcp__set_trace_tag(
    trace_id="tr-abc123",
    key="eval_category",
    value="retrieval_error"
)

# Later search by tag
mcp__mlflow-mcp__search_traces(
    filter_string="tags.eval_category = 'retrieval_error'"
)
```

### Log Analysis Findings

```
# Store analysis as feedback (persists in MLflow)
mcp__mlflow-mcp__log_feedback(
    trace_id="tr-abc123",
    name="latency_analysis",
    value="high",
    source_type="CODE",
    rationale="LLM spans account for 65% of total latency"
)
```

### Log Ground Truth / Expectations

```
# When you determine what the correct output should be
mcp__mlflow-mcp__log_expectation(
    trace_id="tr-abc123",
    name="correct_response",
    value="The meeting is scheduled for 3pm on Tuesday."
)
```

### Retrieve Assessments

```
mcp__mlflow-mcp__get_assessment(
    trace_id="tr-abc123",
    assessment_id="latency_analysis"
)
```

## Query Syntax Reference

| Pattern | Example |
|---------|---------|
| Status filter | `attributes.status = 'OK'` |
| Trace name | `attributes.\`mlflow.traceName\` = 'agent'` |
| Time range | `timestamp_ms > 1699000000000` |
| Latency | `attributes.execution_time_ms > 5000` |
| Tags | `tags.environment = 'production'` |
| Combined | `attributes.status = 'ERROR' AND timestamp_ms > {yesterday_ms}` |

## When to Use MCP vs Python SDK

| Use Case | Approach |
|----------|----------|
| Interactive exploration | MCP tools |
| Agent analysis sessions | MCP tools |
| Evaluation scripts | Python SDK |
| Dataset building | Python SDK |
| CI/CD integration | Python SDK |

## Detailed Reference

For complete pattern implementations, see:
- [patterns-trace-analysis.md](../mlflow-evaluation/references/patterns-trace-analysis.md)
