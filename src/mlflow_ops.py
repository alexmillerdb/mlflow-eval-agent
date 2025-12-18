"""Direct MLflow operations for the evaluation agent.

Simplified from mlflow_client.py - single truncation strategy, direct functions.
Following KISS principle from production-grade agentic AI research.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import mlflow
from mlflow.client import MlflowClient

logger = logging.getLogger(__name__)

# Single truncation limit (vs 4 modes previously)
MAX_OUTPUT_CHARS = 3000

# State directory for file-based persistence
STATE_DIR = Path(".claude/state")


@lru_cache(maxsize=1)
def get_client() -> MlflowClient:
    """Get authenticated MLflow client.

    Authentication via Databricks SDK (config profile or env vars).
    """
    mlflow.set_tracking_uri("databricks")
    return MlflowClient()


def clear_client_cache():
    """Clear cached client (for testing)."""
    get_client.cache_clear()


# =============================================================================
# TRACE OPERATIONS (Direct functions, not tool wrappers)
# =============================================================================

def search_traces(
    experiment_id: str,
    filter_string: Optional[str] = None,
    max_results: int = 20,
) -> list[dict]:
    """Search traces in an experiment.

    Args:
        experiment_id: MLflow experiment ID (numeric string)
        filter_string: Filter like "status = 'OK'"
        max_results: Maximum traces to return

    Returns:
        List of trace summary dicts
    """
    client = get_client()
    traces = client.search_traces(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        max_results=max_results
    )

    return [{
        "trace_id": t.info.trace_id,
        "status": str(t.info.status),
        "execution_time_ms": t.info.execution_time_ms,
        "timestamp_ms": t.info.timestamp_ms,
    } for t in traces]


def get_trace(trace_id: str) -> dict:
    """Get detailed trace with spans.

    Args:
        trace_id: The trace ID to fetch

    Returns:
        Dict with trace info, assessments, and spans
    """
    client = get_client()
    trace = client.get_trace(trace_id=trace_id)

    info = {
        "trace_id": trace.info.trace_id,
        "status": str(trace.info.status),
        "execution_time_ms": trace.info.execution_time_ms,
    }

    # Assessments
    assessments = []
    if hasattr(trace.info, 'assessments') and trace.info.assessments:
        for a in trace.info.assessments:
            source_type = None
            if hasattr(a, 'source') and a.source:
                source_type = str(a.source.source_type) if hasattr(a.source, 'source_type') else None
            assessments.append({
                "name": a.name,
                "value": a.value,
                "source_type": source_type,
                "rationale": a.rationale,
            })

    # Spans with simple truncation
    spans = []
    for span in trace.data.spans:
        spans.append(_format_span(span))

    return {"info": info, "assessments": assessments, "spans": spans}


def _format_span(span) -> dict:
    """Format span with simple truncation."""
    duration_ms = 0.0
    if span.end_time_ns and span.start_time_ns:
        duration_ms = (span.end_time_ns - span.start_time_ns) / 1e6

    attrs = span.attributes or {}
    span_type_str = str(span.span_type) if span.span_type else "UNKNOWN"

    span_data = {
        "span_id": span.span_id,
        "name": span.name,
        "span_type": span_type_str,
        "duration_ms": round(duration_ms, 2),
        "parent_id": span.parent_id,
    }

    # Token usage for LLM spans
    if "LLM" in span_type_str or "CHAT_MODEL" in span_type_str:
        span_data["tokens"] = {
            "input": attrs.get("mlflow.chat_model.input_tokens"),
            "output": attrs.get("mlflow.chat_model.output_tokens"),
        }
        span_data["model"] = attrs.get("mlflow.chat_model.model") or attrs.get("llm.model_name")

    # Error info
    if span.status and hasattr(span.status, 'status_code'):
        if "ERROR" in str(span.status.status_code):
            span_data["error"] = span.status.description or "Unknown error"

    # Inputs/outputs with simple truncation
    if span.inputs:
        span_data["inputs"] = _truncate(str(span.inputs))
    if span.outputs:
        span_data["outputs"] = _truncate(str(span.outputs))

    return span_data


def _truncate(s: str, max_len: int = MAX_OUTPUT_CHARS) -> str:
    """Single truncation strategy."""
    if len(s) <= max_len:
        return s
    return s[:max_len - 20] + f"... [truncated]"


# =============================================================================
# ANNOTATION OPERATIONS
# =============================================================================

def set_tag(trace_id: str, key: str, value: str) -> None:
    """Set a tag on a trace."""
    mlflow.set_trace_tag(trace_id, key, value)


def log_feedback(
    trace_id: str,
    name: str,
    value: Any,
    rationale: Optional[str] = None,
    source_type: str = "CODE"
) -> None:
    """Log feedback assessment to a trace."""
    from mlflow.entities.assessment_source import AssessmentSource

    source = AssessmentSource(source_type=source_type)
    mlflow.log_feedback(
        trace_id=trace_id,
        name=name,
        value=value,
        source=source,
        rationale=rationale
    )


def log_expectation(trace_id: str, name: str, value: Any) -> None:
    """Log expected output to a trace."""
    mlflow.log_expectation(trace_id=trace_id, name=name, value=value)


# =============================================================================
# STATE MANAGEMENT (File-based, replaces 787-line workspace.py)
# =============================================================================

def save_state(key: str, data: Any) -> Path:
    """Save analysis state to JSON file.

    Args:
        key: State key (e.g., "analysis", "recommendations")
        data: Data to save (will be JSON serialized)

    Returns:
        Path to saved file
    """
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = STATE_DIR / f"{key}.json"
    path.write_text(json.dumps(data, indent=2, default=str))
    logger.info(f"State saved: {path}")
    return path


def load_state(key: str) -> Optional[dict]:
    """Load analysis state from JSON file.

    Args:
        key: State key to load

    Returns:
        Loaded data or None if not found
    """
    path = STATE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def clear_state() -> None:
    """Clear all state files."""
    if STATE_DIR.exists():
        for f in STATE_DIR.glob("*.json"):
            f.unlink()
        logger.info("State cleared")


def list_state_keys() -> list[str]:
    """List available state keys."""
    if not STATE_DIR.exists():
        return []
    return [f.stem for f in STATE_DIR.glob("*.json")]


# =============================================================================
# TASK PROGRESS TRACKING (For autonomous mode)
# =============================================================================

TASKS_FILE = Path("eval_tasks.json")


def get_task_status() -> dict:
    """Get current task status for progress tracking."""
    if not TASKS_FILE.exists():
        return {"total": 0, "completed": 0, "pending": 0, "failed": 0, "tasks": []}

    tasks = json.loads(TASKS_FILE.read_text())
    return {
        "total": len(tasks),
        "completed": sum(1 for t in tasks if t.get("status") == "completed"),
        "pending": sum(1 for t in tasks if t.get("status") == "pending"),
        "failed": sum(1 for t in tasks if t.get("status") == "failed"),
        "tasks": tasks,
    }


def all_tasks_complete() -> bool:
    """Check if all tasks are complete."""
    if not TASKS_FILE.exists():
        return False
    tasks = json.loads(TASKS_FILE.read_text())
    return len(tasks) > 0 and all(t.get("status") == "completed" for t in tasks)


def print_progress_summary() -> None:
    """Print current progress with validation status."""
    status = get_task_status()

    if status["total"] == 0:
        print("\nProgress: No tasks created yet (initializer session)")
        return

    pct = (status["completed"] / status["total"]) * 100
    print(f"\nProgress: {status['completed']}/{status['total']} tasks ({pct:.0f}%)")

    # Show task list
    for task in status["tasks"]:
        icon = {"completed": "[x]", "pending": "[ ]", "failed": "[!]"}.get(
            task.get("status", "pending"), "[ ]"
        )
        print(f"  {icon} {task.get('name', 'Unknown task')}")

    if status["failed"] > 0:
        print(f"\n  Warning: {status['failed']} task(s) need fixes")

    # Show validation status if available
    validation_file = STATE_DIR / "validation_results.json"
    if validation_file.exists():
        results = json.loads(validation_file.read_text())
        if results.get("script_success"):
            print("  [x] Eval script runs successfully")
        else:
            print("  [ ] Eval script has errors")

        if results.get("scorers_valid"):
            print("  [x] All scorers returning valid results")
        else:
            print("  [ ] Some scorers have errors/NaN values")


def print_final_summary() -> None:
    """Print final summary when all tasks complete."""
    print("\n" + "=" * 50)
    print("  EVALUATION SETUP COMPLETE")
    print("=" * 50)

    print("\nGenerated Files:")
    eval_dir = Path("evaluation")
    for filename in ["eval_dataset.py", "scorers.py", "run_eval.py"]:
        path = eval_dir / filename
        if path.exists():
            print(f"  [x] {path}")
        else:
            print(f"  [ ] {path} (not found)")

    # Show final validation status
    validation_file = STATE_DIR / "validation_results.json"
    if validation_file.exists():
        results = json.loads(validation_file.read_text())
        print("\nValidation Status:")
        print(f"  Script runs: {'Yes' if results.get('script_success') else 'No'}")
        print(f"  Scorers valid: {'Yes' if results.get('scorers_valid') else 'No'}")

    print("\nTo run evaluation:")
    print("  python evaluation/run_eval.py")


# =============================================================================
# FORMATTING HELPERS
# =============================================================================

def format_traces_table(traces: list[dict]) -> str:
    """Format traces as markdown table."""
    if not traces:
        return "No traces found."

    lines = [
        "| Trace ID | Status | Duration (ms) |",
        "|----------|--------|---------------|"
    ]
    for t in traces[:50]:
        lines.append(f"| {t['trace_id']} | {t['status']} | {t['execution_time_ms']} |")

    if len(traces) > 50:
        lines.append(f"\n... and {len(traces) - 50} more traces")

    return "\n".join(lines)


def text_result(msg: str) -> dict:
    """Create MCP tool result format."""
    return {"content": [{"type": "text", "text": msg}]}
