"""Direct MLflow operations for the evaluation agent.

Simplified from mlflow_client.py - single truncation strategy, direct functions.
Following KISS principle from production-grade agentic AI research.
"""

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import mlflow
from mlflow.client import MlflowClient

logger = logging.getLogger(__name__)

# Single truncation limit (vs 4 modes previously)
MAX_OUTPUT_CHARS = 1500

# Maximum retry attempts for a single task before marking as failed
MAX_TASK_ATTEMPTS = 5

# Context thresholds for warnings
CONTEXT_WARNING_KB = 40
CONTEXT_CRITICAL_KB = 80


# =============================================================================
# CONTEXT MONITORING
# =============================================================================

@dataclass
class ContextMetrics:
    """Tracks context growth within a session for optimization analysis.

    Estimates context size based on tool calls and message patterns.
    """

    session_id: str
    tool_calls: int = 0
    estimated_messages: int = 2  # Initial user + system
    estimated_context_kb: float = 0.0

    def record_tool_call(self, tool_name: str, input_size: int, output_size: int) -> None:
        """Record a tool call and update context estimates.

        Each tool call adds ~2 messages (request + response) to context.
        """
        self.tool_calls += 1
        self.estimated_messages += 2

        # Estimate context growth (input + output in KB)
        self.estimated_context_kb += (input_size + output_size) / 1024

        logger.debug(
            f"Context: {self.tool_calls} tools, "
            f"~{self.estimated_messages} msgs, "
            f"~{self.estimated_context_kb:.1f}KB"
        )

    def to_dict(self) -> dict:
        """Convert metrics to dictionary for logging/saving."""
        return {
            "session_id": self.session_id,
            "tool_calls": self.tool_calls,
            "estimated_messages": self.estimated_messages,
            "estimated_context_kb": round(self.estimated_context_kb, 2),
        }


# Global context metrics for current session
_context_metrics: Optional[ContextMetrics] = None


def start_context_monitoring(session_id: str, initial_prompt: str) -> ContextMetrics:
    """Initialize context monitoring for a new session.

    Args:
        session_id: Session identifier
        initial_prompt: The initial prompt (used to estimate starting size)

    Returns:
        The initialized ContextMetrics instance
    """
    global _context_metrics
    _context_metrics = ContextMetrics(session_id=session_id)
    _context_metrics.estimated_context_kb = len(initial_prompt) / 1024
    return _context_metrics


def get_context_metrics() -> Optional[ContextMetrics]:
    """Get current session's context metrics."""
    return _context_metrics


def record_tool_call(tool_name: str, input_size: int, output_size: int) -> None:
    """Record a tool call to the current session's metrics.

    Safe to call even if monitoring not started (no-op).
    """
    if _context_metrics:
        _context_metrics.record_tool_call(tool_name, input_size, output_size)

        # Warn when context grows large
        if _context_metrics.estimated_context_kb > CONTEXT_CRITICAL_KB:
            logger.warning(
                f"CONTEXT CRITICAL: {_context_metrics.estimated_context_kb:.1f}KB "
                f"exceeds {CONTEXT_CRITICAL_KB}KB threshold"
            )
        elif _context_metrics.estimated_context_kb > CONTEXT_WARNING_KB:
            logger.info(
                f"CONTEXT WARNING: {_context_metrics.estimated_context_kb:.1f}KB "
                f"approaching limit"
            )


def _reset_context_metrics() -> None:
    """Reset context metrics (for testing)."""
    global _context_metrics
    _context_metrics = None

# =============================================================================
# SESSION DIRECTORY MANAGEMENT
# =============================================================================

# Session directory (set by agent on startup)
_session_dir: Path = Path(".")


def set_session_dir(session_dir: Path) -> None:
    """Set the session directory for all paths."""
    global _session_dir
    _session_dir = session_dir

    # For UC volume paths, ensure volume exists before creating subdirectories
    path_str = str(session_dir)
    if path_str.startswith("/Volumes/"):
        _ensure_volume_exists(path_str)

    _session_dir.mkdir(parents=True, exist_ok=True)


def _ensure_volume_exists(path: str) -> None:
    """Create Unity Catalog volume if it doesn't exist.

    UC volumes cannot be created with mkdir - they must be created via SQL.
    Path format: /Volumes/<catalog>/<schema>/<volume>/...
    """
    parts = path.split("/")
    if len(parts) < 5:
        return  # Not a valid UC volume path

    catalog, schema, volume = parts[2], parts[3], parts[4]
    volume_path = Path(f"/Volumes/{catalog}/{schema}/{volume}")

    if volume_path.exists():
        logger.debug(f"UC volume exists: {volume_path}")
        return

    # Create volume using Spark SQL
    logger.info(f"Creating Unity Catalog volume: {catalog}.{schema}.{volume}")
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        spark.sql(f"CREATE VOLUME IF NOT EXISTS `{catalog}`.`{schema}`.`{volume}`")
        logger.info(f"Created Unity Catalog volume: {catalog}.{schema}.{volume}")
    except Exception as e:
        raise ValueError(
            f"Unity Catalog Volume does not exist and could not be created: {volume_path}\n"
            f"Error: {e}\n"
            f"Create manually with: CREATE VOLUME {catalog}.{schema}.{volume}"
        )


def get_session_dir() -> Path:
    """Get current session directory."""
    return _session_dir


def get_state_dir() -> Path:
    """Get state directory for current session."""
    return _session_dir / "state"


def get_tasks_file() -> Path:
    """Get tasks file for current session."""
    return _session_dir / "eval_tasks.json"


def get_evaluation_dir() -> Path:
    """Get evaluation output directory for current session."""
    return _session_dir / "evaluation"


@lru_cache(maxsize=1)
def get_client() -> MlflowClient:
    """Get authenticated MLflow client.

    Respects MLFLOW_TRACKING_URI for local vs Databricks backend.
    - Local MLflow: http://localhost:5000
    - Databricks (local dev): databricks (requires config profile or host/token)
    - Databricks (in cluster): databricks (automatic auth)
    """
    import os
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")
    mlflow.set_tracking_uri(tracking_uri)
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
        locations=[experiment_id],
        filter_string=filter_string,
        max_results=max_results
    )

    return [{
        "trace_id": t.info.trace_id,
        "status": str(t.info.status),
        "execution_time_ms": t.info.execution_time_ms,
        "timestamp_ms": t.info.timestamp_ms,
    } for t in traces]


def search_runs(
    experiment_id: str,
    filter_string: Optional[str] = None,
    max_results: int = 100,
) -> list[dict]:
    """Search MLflow Runs (evaluations, training runs, etc).

    Args:
        experiment_id: MLflow experiment ID (numeric string)
        filter_string: Filter for runs. Valid filter columns include:
            - attributes.run_name: Filter by run name
            - attributes.status: Filter by run status (RUNNING, FINISHED, FAILED, etc.)
            - metrics.<key>: Filter by metric value (e.g., "metrics.accuracy > 0.9")
            - params.<key>: Filter by parameter value
            - tags.<key>: Filter by tag value
        max_results: Maximum runs to return (default 100)

    Returns:
        List of run summary dicts with run_id, run_name, status, metrics, etc.

    Example filters:
        - "attributes.run_name = 'my_eval_20250120'"
        - "attributes.status = 'FINISHED'"
        - "metrics.safety > 0.8"
        - "tags.mlflow.runName LIKE '%eval%'"
    """
    client = get_client()
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        max_results=max_results,
        order_by=["start_time DESC"],
    )

    return [{
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
        "status": run.info.status,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "metrics": dict(run.data.metrics) if run.data.metrics else {},
        "params": dict(run.data.params) if run.data.params else {},
        "artifact_uri": run.info.artifact_uri,
    } for run in runs]


def get_run(run_id: str) -> dict:
    """Get a specific MLflow Run by ID.

    Args:
        run_id: The MLflow run ID

    Returns:
        Dict with run details including metrics, params, and tags
    """
    client = get_client()
    run = client.get_run(run_id)

    return {
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
        "status": run.info.status,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "metrics": dict(run.data.metrics) if run.data.metrics else {},
        "params": dict(run.data.params) if run.data.params else {},
        "tags": dict(run.data.tags) if run.data.tags else {},
        "artifact_uri": run.info.artifact_uri,
    }


@lru_cache(maxsize=5)
def _get_trace_cached(trace_id: str):
    """Cache raw trace to avoid re-fetching within a session."""
    client = get_client()
    return client.get_trace(trace_id=trace_id)


def clear_trace_cache():
    """Clear trace cache between sessions."""
    _get_trace_cached.cache_clear()


def get_trace(trace_id: str, detail_level: str = "summary") -> dict:
    """Get trace with appropriate detail level.

    Args:
        trace_id: The trace ID to fetch
        detail_level: "summary" (~2KB), "analysis" (~10KB), or "full" (~50KB)
            - summary: info + span names/types/durations only
            - analysis: + bottlenecks, errors, token stats
            - full: + all inputs/outputs (current behavior)

    Returns:
        Dict with trace data at requested detail level
    """
    trace = _get_trace_cached(trace_id)

    info = {
        "trace_id": trace.info.trace_id,
        "status": str(trace.info.status),
        "execution_time_ms": trace.info.execution_time_ms,
    }

    if detail_level == "summary":
        # Minimal data: just structure and timing (~2KB)
        spans = []
        for span in trace.data.spans:
            duration_ms = 0.0
            if span.end_time_ns and span.start_time_ns:
                duration_ms = (span.end_time_ns - span.start_time_ns) / 1e6
            spans.append({
                "name": span.name,
                "span_type": str(span.span_type) if span.span_type else "UNKNOWN",
                "duration_ms": round(duration_ms, 2),
            })
        return {"info": info, "spans": spans}

    elif detail_level == "analysis":
        # Add bottlenecks, errors, token data (~10KB)
        spans = []
        total_tokens = {"input": 0, "output": 0}
        errors = []
        bottlenecks = []

        for span in trace.data.spans:
            duration_ms = 0.0
            if span.end_time_ns and span.start_time_ns:
                duration_ms = (span.end_time_ns - span.start_time_ns) / 1e6

            span_type_str = str(span.span_type) if span.span_type else "UNKNOWN"
            attrs = span.attributes or {}

            span_data = {
                "name": span.name,
                "span_type": span_type_str,
                "duration_ms": round(duration_ms, 2),
            }

            # Token usage for LLM spans
            if "LLM" in span_type_str or "CHAT_MODEL" in span_type_str:
                input_tokens = attrs.get("mlflow.chat_model.input_tokens", 0) or 0
                output_tokens = attrs.get("mlflow.chat_model.output_tokens", 0) or 0
                total_tokens["input"] += input_tokens
                total_tokens["output"] += output_tokens
                span_data["tokens"] = {"input": input_tokens, "output": output_tokens}
                span_data["model"] = attrs.get("mlflow.chat_model.model") or attrs.get("llm.model_name")

            # Error info
            if span.status and hasattr(span.status, 'status_code'):
                if "ERROR" in str(span.status.status_code):
                    error_info = {"span": span.name, "error": span.status.description or "Unknown error"}
                    span_data["error"] = error_info["error"]
                    errors.append(error_info)

            # Track bottlenecks (spans > 1 second)
            if duration_ms > 1000:
                bottlenecks.append({"span": span.name, "duration_ms": round(duration_ms, 2)})

            spans.append(span_data)

        # Sort bottlenecks by duration
        bottlenecks.sort(key=lambda x: x["duration_ms"], reverse=True)

        return {
            "info": info,
            "analysis": {
                "total_tokens": total_tokens,
                "error_count": len(errors),
                "errors": errors[:5],  # Top 5 errors
                "bottlenecks": bottlenecks[:5],  # Top 5 slow spans
            },
            "spans": spans,
        }

    else:  # "full" - current behavior
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

        # Spans with full data
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
    state_dir = get_state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / f"{key}.json"
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
    path = get_state_dir() / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def clear_state() -> None:
    """Clear all state files."""
    state_dir = get_state_dir()
    if state_dir.exists():
        for f in state_dir.glob("*.json"):
            f.unlink()
        logger.info("State cleared")


def list_state_keys() -> list[str]:
    """List available state keys."""
    state_dir = get_state_dir()
    if not state_dir.exists():
        return []
    return [f.stem for f in state_dir.glob("*.json")]


# =============================================================================
# TASK PROGRESS TRACKING (For autonomous mode)
# =============================================================================


def get_task_status() -> dict:
    """Get current task status for progress tracking."""
    tasks_file = get_tasks_file()
    if not tasks_file.exists():
        return {"total": 0, "completed": 0, "pending": 0, "failed": 0, "tasks": []}

    data = json.loads(tasks_file.read_text())

    # Handle both formats: list or dict with "tasks" key
    if isinstance(data, dict):
        tasks = data.get("tasks", [])
    else:
        tasks = data

    return {
        "total": len(tasks),
        "completed": sum(1 for t in tasks if t.get("status") == "completed"),
        "pending": sum(1 for t in tasks if t.get("status") == "pending"),
        "failed": sum(1 for t in tasks if t.get("status") == "failed"),
        "tasks": tasks,
    }


def all_tasks_complete() -> bool:
    """Check if all tasks are complete."""
    tasks_file = get_tasks_file()
    if not tasks_file.exists():
        return False

    data = json.loads(tasks_file.read_text())

    # Handle both formats: list or dict with "tasks" key
    if isinstance(data, dict):
        tasks = data.get("tasks", [])
    else:
        tasks = data

    return len(tasks) > 0 and all(t.get("status") == "completed" for t in tasks)


def get_task_attempts(task_id: int) -> int:
    """Get the number of attempts for a task.

    Args:
        task_id: The task ID to check

    Returns:
        Number of attempts (0 if not found or no attempts field)
    """
    tasks_file = get_tasks_file()
    if not tasks_file.exists():
        return 0

    data = json.loads(tasks_file.read_text())

    # Handle both formats: list or dict with "tasks" key
    if isinstance(data, dict):
        tasks = data.get("tasks", [])
    else:
        tasks = data

    for task in tasks:
        if task.get("id") == task_id:
            return task.get("attempts", 0)

    return 0


def increment_task_attempts(task_id: int) -> bool:
    """Increment the attempt counter for a task.

    Args:
        task_id: The task ID to increment

    Returns:
        True if task can continue (under limit), False if exceeded max attempts
    """
    tasks_file = get_tasks_file()
    if not tasks_file.exists():
        return False

    data = json.loads(tasks_file.read_text())

    # Handle both formats: list or dict with "tasks" key
    is_dict_format = isinstance(data, dict)
    if is_dict_format:
        tasks = data.get("tasks", [])
    else:
        tasks = data

    # Find and update the task
    task_found = False
    for task in tasks:
        if task.get("id") == task_id:
            task_found = True
            current_attempts = task.get("attempts", 0)
            new_attempts = current_attempts + 1
            task["attempts"] = new_attempts

            # Check if exceeded limit
            if new_attempts > MAX_TASK_ATTEMPTS:
                task["status"] = "failed"
                task["failure_reason"] = f"Exceeded max attempts ({MAX_TASK_ATTEMPTS})"
                logger.warning(f"Task {task_id} failed: exceeded max attempts ({MAX_TASK_ATTEMPTS})")

                # Save and return False
                if is_dict_format:
                    tasks_file.write_text(json.dumps(data, indent=2))
                else:
                    tasks_file.write_text(json.dumps(tasks, indent=2))
                return False

            break

    if not task_found:
        return False

    # Save updated tasks
    if is_dict_format:
        tasks_file.write_text(json.dumps(data, indent=2))
    else:
        tasks_file.write_text(json.dumps(tasks, indent=2))

    return True


def print_progress_summary() -> None:
    """Log current progress with validation status."""
    status = get_task_status()

    if status["total"] == 0:
        logger.info("Progress: No tasks created yet (initializer session)")
        return

    pct = (status["completed"] / status["total"]) * 100
    logger.info(f"Progress: {status['completed']}/{status['total']} tasks ({pct:.0f}%)")

    # Show task list
    for task in status["tasks"]:
        icon = {"completed": "[x]", "pending": "[ ]", "failed": "[!]"}.get(
            task.get("status", "pending"), "[ ]"
        )
        logger.info(f"  {icon} {task.get('name', 'Unknown task')}")

    if status["failed"] > 0:
        logger.warning(f"  Warning: {status['failed']} task(s) need fixes")

    # Show validation status if available
    validation_file = get_state_dir() / "validation_results.json"
    if validation_file.exists():
        results = json.loads(validation_file.read_text())
        if isinstance(results, dict):
            if results.get("script_success"):
                logger.info("  [x] Eval script runs successfully")
            else:
                logger.info("  [ ] Eval script has errors")

            if results.get("scorers_valid"):
                logger.info("  [x] All scorers returning valid results")
            else:
                logger.info("  [ ] Some scorers have errors/NaN values")


def print_final_summary() -> None:
    """Log final summary when all tasks complete."""
    logger.info("=" * 50)
    logger.info("  EVALUATION SETUP COMPLETE")
    logger.info("=" * 50)

    logger.info("Generated Files:")
    eval_dir = get_evaluation_dir()
    for filename in ["eval_dataset.py", "scorers.py", "run_eval.py"]:
        path = eval_dir / filename
        if path.exists():
            logger.info(f"  [x] {path}")
        else:
            logger.info(f"  [ ] {path} (not found)")

    # Show final validation status
    validation_file = get_state_dir() / "validation_results.json"
    if validation_file.exists():
        results = json.loads(validation_file.read_text())
        if isinstance(results, dict):
            logger.info("Validation Status:")
            logger.info(f"  Script runs: {'Yes' if results.get('script_success') else 'No'}")
            logger.info(f"  Scorers valid: {'Yes' if results.get('scorers_valid') else 'No'}")

    logger.info("To run evaluation:")
    logger.info(f"  python {get_evaluation_dir()}/run_eval.py")


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


def format_runs_table(runs: list[dict]) -> str:
    """Format MLflow runs as markdown table."""
    if not runs:
        return "No runs found."

    lines = [
        "| Run ID | Run Name | Status | Metrics |",
        "|--------|----------|--------|---------|"
    ]
    for run in runs[:50]:
        run_id_short = run['run_id'][:8] + "..."
        run_name = run.get('run_name') or 'N/A'
        # Truncate run name if too long
        if len(run_name) > 30:
            run_name = run_name[:27] + "..."
        status = run.get('status', 'UNKNOWN')
        metrics = run.get('metrics', {})
        # Show first few metrics or count
        if metrics:
            metric_summary = ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                                       for k, v in list(metrics.items())[:3])
            if len(metrics) > 3:
                metric_summary += f" (+{len(metrics)-3} more)"
        else:
            metric_summary = "none"
        lines.append(f"| {run_id_short} | {run_name} | {status} | {metric_summary} |")

    if len(runs) > 50:
        lines.append(f"\n... and {len(runs) - 50} more runs")

    return "\n".join(lines)


def text_result(msg: str) -> dict:
    """Create MCP tool result format."""
    return {"content": [{"type": "text", "text": msg}]}
