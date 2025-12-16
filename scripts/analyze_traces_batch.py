#!/usr/bin/env python
"""Batch trace analysis using MLflow 3 SDK.

Usage:
    # Analyze traces from an experiment
    python scripts/analyze_traces_batch.py <experiment-id>

    # Analyze specific traces by ID
    python scripts/analyze_traces_batch.py tr-abc123,tr-def456,tr-ghi789

Environment variables (loaded from .env):
    DATABRICKS_HOST: Databricks workspace URL
    DATABRICKS_TOKEN: Databricks personal access token
    - OR -
    DATABRICKS_CONFIG_PROFILE: Named profile from ~/.databrickscfg

Outputs JSON report with:
- Aggregate statistics (avg, p50, p95, max latency)
- Success rate
- Architecture detection
- Common bottlenecks
- Error patterns
- Recommendations
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.entities import SpanStatusCode, SpanType, Trace


def setup_databricks_auth():
    """Configure Databricks authentication from .env or environment variables."""
    # Load .env file if it exists
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        from dotenv import load_dotenv

        load_dotenv(env_path)

    # Set MLflow tracking URI to Databricks
    mlflow.set_tracking_uri("databricks")

    # Check for authentication
    profile = os.environ.get("DATABRICKS_CONFIG_PROFILE")
    token = os.environ.get("DATABRICKS_TOKEN")
    host = os.environ.get("DATABRICKS_HOST")

    if profile:
        # Use named profile from ~/.databrickscfg
        os.environ["DATABRICKS_CONFIG_PROFILE"] = profile
    elif token and host:
        # Use direct token authentication
        os.environ["DATABRICKS_HOST"] = host
        os.environ["DATABRICKS_TOKEN"] = token
    else:
        # Will use default authentication (databricks-cli config, etc.)
        pass


def get_trace(client: MlflowClient, trace_id: str) -> Trace:
    """Fetch trace by ID."""
    return client.get_trace(trace_id)


def search_traces_by_experiment(client: MlflowClient, experiment_id: str, max_results: int = 10) -> list[Trace]:
    """Search traces from an experiment."""
    return client.search_traces(experiment_ids=[experiment_id], max_results=max_results)


def get_traces_by_ids(client: MlflowClient, trace_ids: list[str]) -> list[Trace]:
    """Fetch multiple traces by ID."""
    traces = []
    errors = []
    for tid in trace_ids:
        try:
            traces.append(client.get_trace(tid.strip()))
        except Exception as e:
            errors.append({"trace_id": tid, "error": str(e)})
    return traces, errors


def analyze_single_trace(trace: Trace) -> dict[str, Any]:
    """Analyze a single trace for batch aggregation."""
    spans = trace.data.spans if hasattr(trace, "data") else []

    root_spans = [s for s in spans if s.parent_id is None]
    total_ms = 0
    if root_spans:
        root = root_spans[0]
        total_ms = (root.end_time_ns - root.start_time_ns) / 1e6 if root.end_time_ns else 0

    span_types = [s.span_type for s in spans]
    span_names = [s.name.lower() for s in spans]

    # Architecture detection
    if any(p in " ".join(span_names) for p in ["classifier", "rewriter", "gatherer", "executor"]):
        arch_type = "dspy_multi_agent"
    elif any(p in " ".join(span_names) for p in ["langgraph", "graph", "node", "state"]):
        arch_type = "langgraph"
    elif SpanType.RETRIEVER in span_types and SpanType.TOOL in span_types:
        arch_type = "rag_with_tools"
    elif SpanType.RETRIEVER in span_types:
        arch_type = "rag"
    elif SpanType.TOOL in span_types:
        arch_type = "tool_calling"
    else:
        arch_type = "simple_chat"

    # Count span types
    type_counts = defaultdict(int)
    type_latencies = defaultdict(float)
    for span in spans:
        st = str(span.span_type) if span.span_type else "UNKNOWN"
        type_counts[st] += 1
        duration_ms = (span.end_time_ns - span.start_time_ns) / 1e6 if span.end_time_ns else 0
        type_latencies[st] += duration_ms

    # Error detection
    failed_spans = []
    for span in spans:
        if span.status and span.status.status_code == SpanStatusCode.ERROR:
            failed_spans.append(span.name)

    # LLM stats
    llm_spans = [s for s in spans if s.span_type in [SpanType.LLM, SpanType.CHAT_MODEL]]
    total_input_tokens = 0
    total_output_tokens = 0
    for span in llm_spans:
        attributes = span.attributes or {}
        total_input_tokens += attributes.get("mlflow.chat_model.input_tokens") or 0
        total_output_tokens += attributes.get("mlflow.chat_model.output_tokens") or 0

    # Tool stats
    tool_spans = [s for s in spans if s.span_type == SpanType.TOOL]

    # Bottleneck detection (find slowest non-wrapper span)
    bottleneck = None
    exclude_patterns = ["forward", "predict", "root", "__init__"]
    for span in sorted(spans, key=lambda s: -(s.end_time_ns - s.start_time_ns) if s.end_time_ns else 0):
        if not any(p in span.name.lower() for p in exclude_patterns):
            duration_ms = (span.end_time_ns - span.start_time_ns) / 1e6 if span.end_time_ns else 0
            bottleneck = {"name": span.name, "duration_ms": duration_ms}
            break

    return {
        "trace_id": trace.info.trace_id,
        "status": str(trace.info.status),
        "total_ms": round(total_ms, 2),
        "span_count": len(spans),
        "architecture": arch_type,
        "type_counts": dict(type_counts),
        "type_latencies": {k: round(v, 2) for k, v in type_latencies.items()},
        "failed_spans": failed_spans,
        "llm_calls": len(llm_spans),
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "tool_calls": len(tool_spans),
        "bottleneck": bottleneck,
    }


def aggregate_statistics(trace_analyses: list[dict]) -> dict[str, Any]:
    """Aggregate statistics across multiple traces."""
    if not trace_analyses:
        return {"error": "No traces to analyze"}

    latencies = [t["total_ms"] for t in trace_analyses]
    statuses = [t["status"] for t in trace_analyses]

    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)

    # Architecture distribution
    arch_counts = defaultdict(int)
    for t in trace_analyses:
        arch_counts[t["architecture"]] += 1

    # Type aggregation
    type_total_counts = defaultdict(int)
    type_total_latencies = defaultdict(float)
    for t in trace_analyses:
        for st, count in t["type_counts"].items():
            type_total_counts[st] += count
        for st, latency in t["type_latencies"].items():
            type_total_latencies[st] += latency

    # Error aggregation
    all_failed_spans = []
    for t in trace_analyses:
        all_failed_spans.extend(t["failed_spans"])
    failed_span_counts = defaultdict(int)
    for name in all_failed_spans:
        failed_span_counts[name] += 1

    # Token aggregation
    total_input = sum(t["input_tokens"] for t in trace_analyses)
    total_output = sum(t["output_tokens"] for t in trace_analyses)

    # Common bottlenecks
    bottleneck_counts = defaultdict(int)
    bottleneck_latencies = defaultdict(float)
    for t in trace_analyses:
        if t["bottleneck"]:
            name = t["bottleneck"]["name"]
            bottleneck_counts[name] += 1
            bottleneck_latencies[name] += t["bottleneck"]["duration_ms"]

    common_bottlenecks = sorted(
        [
            {"name": name, "count": count, "avg_ms": round(bottleneck_latencies[name] / count, 2)}
            for name, count in bottleneck_counts.items()
        ],
        key=lambda x: -x["count"],
    )[:5]

    return {
        "summary": {
            "trace_count": n,
            "success_rate": round(sum(1 for s in statuses if "OK" in s) / n, 3) if n > 0 else 0,
            "architectures": dict(arch_counts),
        },
        "latency": {
            "avg_ms": round(sum(latencies) / n, 2) if n > 0 else 0,
            "min_ms": round(min(latencies), 2) if latencies else 0,
            "max_ms": round(max(latencies), 2) if latencies else 0,
            "p50_ms": round(sorted_latencies[n // 2], 2) if n > 0 else 0,
            "p95_ms": round(sorted_latencies[int(n * 0.95)], 2) if n > 0 else 0,
        },
        "span_types": {
            "counts": dict(type_total_counts),
            "latencies_ms": {k: round(v, 2) for k, v in type_total_latencies.items()},
        },
        "llm_usage": {
            "total_calls": sum(t["llm_calls"] for t in trace_analyses),
            "avg_calls_per_trace": round(sum(t["llm_calls"] for t in trace_analyses) / n, 2) if n > 0 else 0,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
        },
        "tool_usage": {
            "total_calls": sum(t["tool_calls"] for t in trace_analyses),
            "avg_calls_per_trace": round(sum(t["tool_calls"] for t in trace_analyses) / n, 2) if n > 0 else 0,
        },
        "errors": {
            "failed_traces": sum(1 for s in statuses if "ERROR" in s),
            "common_failed_spans": dict(sorted(failed_span_counts.items(), key=lambda x: -x[1])[:5]),
        },
        "bottlenecks": common_bottlenecks,
        "traces": trace_analyses,
    }


def generate_recommendations(stats: dict) -> list[str]:
    """Generate recommendations from aggregate statistics."""
    recommendations = []

    if stats["summary"]["success_rate"] < 0.9:
        recommendations.append(
            f"LOW SUCCESS RATE: Only {stats['summary']['success_rate']:.0%} of traces succeeded. "
            f"Review common failed spans: {list(stats['errors']['common_failed_spans'].keys())[:3]}"
        )

    if stats["latency"]["p95_ms"] > stats["latency"]["avg_ms"] * 3:
        recommendations.append(
            f"HIGH LATENCY VARIANCE: P95 ({stats['latency']['p95_ms']}ms) is 3x+ higher than avg "
            f"({stats['latency']['avg_ms']}ms). Investigate outliers."
        )

    if stats["llm_usage"]["avg_calls_per_trace"] > 5:
        recommendations.append(
            f"HIGH LLM CALLS: {stats['llm_usage']['avg_calls_per_trace']} LLM calls per trace on average. "
            "Consider reducing calls or batching."
        )

    if stats["bottlenecks"]:
        top_bottleneck = stats["bottlenecks"][0]
        recommendations.append(
            f"COMMON BOTTLENECK: '{top_bottleneck['name']}' appears in {top_bottleneck['count']} traces "
            f"with avg {top_bottleneck['avg_ms']}ms. Consider optimizing."
        )

    if not recommendations:
        recommendations.append("No major issues detected across traces. Overall health looks good.")

    return recommendations


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_traces_batch.py <experiment-id | trace-ids>", file=sys.stderr)
        print("  experiment-id: Numeric experiment ID", file=sys.stderr)
        print("  trace-ids: Comma-separated trace IDs (e.g., tr-abc123,tr-def456)", file=sys.stderr)
        sys.exit(1)

    arg = sys.argv[1].strip()

    try:
        # Setup Databricks authentication
        setup_databricks_auth()

        client = MlflowClient()
        # Determine if argument is experiment ID or trace IDs
        if "," in arg or arg.startswith("tr-"):
            # Trace IDs
            raw_ids = arg.replace("trace-ids", "").strip()
            trace_ids = [tid.strip() for tid in raw_ids.split(",") if tid.strip()]
            traces, fetch_errors = get_traces_by_ids(client, trace_ids)
        else:
            # Experiment ID
            traces = search_traces_by_experiment(client, arg)
            fetch_errors = []

        if not traces:
            report = {"error": "No traces found", "argument": arg, "fetch_errors": fetch_errors}
            print(json.dumps(report, indent=2))
            sys.exit(1)

        # Analyze each trace
        trace_analyses = [analyze_single_trace(trace) for trace in traces]

        # Aggregate statistics
        stats = aggregate_statistics(trace_analyses)
        stats["recommendations"] = generate_recommendations(stats)

        if fetch_errors:
            stats["fetch_errors"] = fetch_errors

        print(json.dumps(stats, indent=2))

    except Exception as e:
        error_report = {"error": str(e), "argument": arg}
        print(json.dumps(error_report, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
