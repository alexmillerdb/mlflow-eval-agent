"""Test harness for isolated component testing.

Provides functions for testing:
- Initializer session in isolation
- Worker session in isolation
- MCP tools directly
- Full autonomous loop with limits

Each test captures trace IDs for the existing feedback loop: /analyze-trace, /analyze-tokens
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import mlflow

from .config import Config
from .mlflow_ops import (
    set_session_dir,
    get_tasks_file,
    get_state_dir,
    clear_client_cache,
    clear_trace_cache,
)

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result from a component test."""

    success: bool
    trace_id: Optional[str] = None
    session_id: Optional[str] = None
    error: Optional[str] = None
    outputs: Optional[dict] = None
    duration_ms: Optional[int] = None
    cost_usd: Optional[float] = None


# =============================================================================
# SESSION TESTING
# =============================================================================


async def run_initializer_session(
    experiment_id: str,
    session_dir: Optional[Path] = None,
    mock: bool = False,
    on_progress: Optional[Callable[[str, dict], None]] = None,
) -> TestResult:
    """Run only the initializer session, return trace_id and outputs.

    Args:
        experiment_id: MLflow experiment ID to analyze
        session_dir: Optional session directory (uses temp dir if not provided)
        mock: If True, use mock MLflow client
        on_progress: Optional callback for progress updates, called with (event_type, data)

    Returns:
        TestResult with trace_id, outputs, and success status
    """
    import tempfile
    import time
    from datetime import datetime

    from .agent import MLflowAgent, load_prompt, setup_mlflow

    # Set up session directory
    if session_dir is None:
        session_dir = Path(tempfile.mkdtemp(prefix="mlflow-eval-test-"))

    set_session_dir(session_dir)

    # Set up MLflow tracing (if not mock)
    if not mock:
        setup_mlflow()

    # Create config
    config = Config.from_env(validate=not mock)
    config.experiment_id = experiment_id

    # Load initializer prompt
    prompt = load_prompt("initializer")
    prompt = prompt.replace("{experiment_id}", experiment_id)
    prompt = prompt.replace("{session_dir}", str(session_dir))

    # Create agent
    agent = MLflowAgent(config)

    # Run session with tracing
    start_time = time.time()
    trace_id = None
    result_data = None

    try:
        with mlflow.start_span(name="test_initializer_session") as span:
            trace_id = mlflow.get_current_active_span().request_id if mlflow.get_current_active_span() else None

            async for result in agent.query(prompt):
                result_data = result
                # Call progress callback if provided
                if on_progress:
                    on_progress(result.event_type, {
                        "tool_name": result.tool_name,
                        "response_len": len(result.response) if result.response else 0,
                    })

            span.set_attribute("test_type", "initializer")
            span.set_attribute("experiment_id", experiment_id)

    except Exception as e:
        logger.exception("Initializer session failed")
        return TestResult(
            success=False,
            error=str(e),
            session_id=agent.session_id,
        )

    duration_ms = int((time.time() - start_time) * 1000)

    # Verify outputs
    outputs = verify_initializer_outputs(session_dir)

    return TestResult(
        success=outputs.get("valid", False),
        trace_id=trace_id,
        session_id=agent.session_id,
        outputs=outputs,
        duration_ms=duration_ms,
        cost_usd=result_data.cost_usd if result_data else None,
    )


async def run_worker_session(
    experiment_id: str,
    session_dir: Path,
    task_type: Optional[str] = None,
    mock: bool = False,
    on_progress: Optional[Callable[[str, dict], None]] = None,
) -> TestResult:
    """Run one worker session, return trace_id and task status.

    Args:
        experiment_id: MLflow experiment ID
        session_dir: Session directory (must have eval_tasks.json)
        task_type: Optional filter for specific task type
        mock: If True, use mock MLflow client
        on_progress: Optional callback for progress updates, called with (event_type, data)

    Returns:
        TestResult with trace_id, task status, and success status
    """
    import time

    from .agent import MLflowAgent, load_prompt, setup_mlflow

    # Verify tasks file exists
    tasks_file = get_tasks_file()
    if not tasks_file.exists():
        return TestResult(
            success=False,
            error=f"Tasks file not found: {tasks_file}. Run initializer first or use --mock.",
        )

    set_session_dir(session_dir)

    # Set up MLflow tracing (if not mock)
    if not mock:
        setup_mlflow()

    # Create config
    config = Config.from_env(validate=not mock)
    config.experiment_id = experiment_id

    # Load worker prompt
    prompt = load_prompt("worker")
    prompt = prompt.replace("{experiment_id}", experiment_id)
    prompt = prompt.replace("{session_dir}", str(session_dir))

    # Create agent
    agent = MLflowAgent(config)

    # Run session with tracing
    start_time = time.time()
    trace_id = None
    result_data = None

    try:
        with mlflow.start_span(name="test_worker_session") as span:
            trace_id = mlflow.get_current_active_span().request_id if mlflow.get_current_active_span() else None

            async for result in agent.query(prompt):
                result_data = result
                # Call progress callback if provided
                if on_progress:
                    on_progress(result.event_type, {
                        "tool_name": result.tool_name,
                        "response_len": len(result.response) if result.response else 0,
                    })

            span.set_attribute("test_type", "worker")
            span.set_attribute("experiment_id", experiment_id)
            if task_type:
                span.set_attribute("task_type_filter", task_type)

    except Exception as e:
        logger.exception("Worker session failed")
        return TestResult(
            success=False,
            error=str(e),
            session_id=agent.session_id,
        )

    duration_ms = int((time.time() - start_time) * 1000)

    # Get task status after worker session
    tasks = json.loads(tasks_file.read_text())
    if isinstance(tasks, dict):
        tasks = tasks.get("tasks", [])

    # Check if any task was completed
    completed_tasks = [t for t in tasks if t.get("status") == "completed"]

    return TestResult(
        success=len(completed_tasks) > 0,
        trace_id=trace_id,
        session_id=agent.session_id,
        outputs={"tasks": tasks, "completed_count": len(completed_tasks)},
        duration_ms=duration_ms,
        cost_usd=result_data.cost_usd if result_data else None,
    )


# =============================================================================
# TOOL TESTING
# =============================================================================


async def test_tool_direct(
    tool_name: str,
    operation: str,
    experiment_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    mock: bool = False,
    **kwargs,
) -> TestResult:
    """Call MCP tool directly without agent, return result.

    Args:
        tool_name: Tool name (mlflow_query, mlflow_annotate, save_findings)
        operation: Operation to perform (search, get, tag, feedback, etc.)
        experiment_id: Required for search operations
        trace_id: Required for get/annotate operations
        mock: If True, use mock MLflow client
        **kwargs: Additional arguments for the tool

    Returns:
        TestResult with tool output
    """
    import time
    import tempfile

    from .tools import create_tools
    from .mlflow_ops import text_result

    # Set up session directory for save_findings
    if tool_name == "save_findings":
        session_dir = Path(tempfile.mkdtemp(prefix="mlflow-eval-test-"))
        set_session_dir(session_dir)

    # Create tools - returns [mlflow_query, mlflow_annotate, save_findings] in order
    tools = create_tools()

    # Map by expected order since @tool decorator changes __name__
    tool_map = {
        "mlflow_query": tools[0] if len(tools) > 0 else None,
        "mlflow_annotate": tools[1] if len(tools) > 1 else None,
        "save_findings": tools[2] if len(tools) > 2 else None,
    }

    # Map tool name to function
    tool_fn = tool_map.get(tool_name)
    if tool_fn is None:
        return TestResult(
            success=False,
            error=f"Unknown tool: {tool_name}. Use: mlflow_query, mlflow_annotate, save_findings",
        )

    # Build args
    args = {"operation": operation, **kwargs}
    if experiment_id:
        args["experiment_id"] = experiment_id
    if trace_id:
        args["trace_id"] = trace_id

    # Call tool
    start_time = time.time()
    try:
        result = await tool_fn(args)
        duration_ms = int((time.time() - start_time) * 1000)

        # Extract text from result
        output_text = ""
        if isinstance(result, dict) and "content" in result:
            for item in result["content"]:
                if isinstance(item, dict) and item.get("type") == "text":
                    output_text = item.get("text", "")
                    break

        return TestResult(
            success="Error" not in output_text,
            outputs={"raw": result, "text": output_text},
            duration_ms=duration_ms,
        )

    except Exception as e:
        logger.exception(f"Tool {tool_name} failed")
        return TestResult(
            success=False,
            error=str(e),
        )


# =============================================================================
# MOCK DATA CREATION
# =============================================================================


def create_mock_tasks(session_dir: Path, task_types: Optional[list[str]] = None) -> Path:
    """Create mock eval_tasks.json for worker testing.

    Args:
        session_dir: Session directory to create tasks file in
        task_types: List of task types to create (defaults to all)

    Returns:
        Path to created tasks file
    """
    set_session_dir(session_dir)

    if task_types is None:
        task_types = ["dataset", "scorer", "script", "validate"]

    tasks = []
    for i, task_type in enumerate(task_types, 1):
        task = {
            "id": i,
            "name": f"Mock {task_type} task",
            "type": task_type,
            "status": "pending",
            "details": f"Mock details for {task_type} task",
        }
        tasks.append(task)

    tasks_file = get_tasks_file()
    tasks_file.parent.mkdir(parents=True, exist_ok=True)
    tasks_file.write_text(json.dumps({"tasks": tasks}, indent=2))

    logger.info(f"Created mock tasks file: {tasks_file}")
    return tasks_file


def create_mock_analysis(session_dir: Path, experiment_id: str) -> Path:
    """Create mock state/analysis.json for worker testing.

    Args:
        session_dir: Session directory
        experiment_id: Experiment ID for the analysis

    Returns:
        Path to created analysis file
    """
    set_session_dir(session_dir)

    analysis = {
        "experiment_id": experiment_id,
        "agent_type": "Mock test agent",
        "dataset_strategy": "traces",
        "has_predict_fn": False,
        "trace_summary": {
            "total_analyzed": 10,
            "success_count": 8,
            "error_count": 2,
            "avg_latency_ms": 2500,
        },
        "sample_trace_ids": ["tr-mock-001", "tr-mock-002", "tr-mock-003"],
        "recommended_scorers": [
            {"name": "Safety", "type": "builtin", "rationale": "Required for all agents"},
            {"name": "RelevanceToQuery", "type": "builtin", "rationale": "Mock agent responds to queries"},
        ],
        "error_patterns": [],
    }

    state_dir = get_state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)
    analysis_file = state_dir / "analysis.json"
    analysis_file.write_text(json.dumps(analysis, indent=2))

    logger.info(f"Created mock analysis file: {analysis_file}")
    return analysis_file


# =============================================================================
# OUTPUT VERIFICATION
# =============================================================================


def verify_initializer_outputs(session_dir: Path) -> dict:
    """Check eval_tasks.json and analysis.json exist and are valid.

    Args:
        session_dir: Session directory to check

    Returns:
        Dict with validation results
    """
    set_session_dir(session_dir)

    results = {
        "valid": True,
        "tasks_file_exists": False,
        "tasks_valid": False,
        "analysis_file_exists": False,
        "analysis_valid": False,
        "errors": [],
    }

    # Check tasks file
    tasks_file = get_tasks_file()
    if tasks_file.exists():
        results["tasks_file_exists"] = True
        try:
            data = json.loads(tasks_file.read_text())
            tasks = data.get("tasks", []) if isinstance(data, dict) else data
            if isinstance(tasks, list) and len(tasks) > 0:
                # Check required fields
                required = {"id", "name", "type", "status"}
                for task in tasks:
                    if not required.issubset(task.keys()):
                        results["errors"].append(f"Task missing required fields: {required - task.keys()}")
                        results["valid"] = False
                        break
                else:
                    results["tasks_valid"] = True
            else:
                results["errors"].append("Tasks file is empty or invalid format")
                results["valid"] = False
        except json.JSONDecodeError as e:
            results["errors"].append(f"Tasks file JSON error: {e}")
            results["valid"] = False
    else:
        results["errors"].append(f"Tasks file not found: {tasks_file}")
        results["valid"] = False

    # Check analysis file
    analysis_file = get_state_dir() / "analysis.json"
    if analysis_file.exists():
        results["analysis_file_exists"] = True
        try:
            data = json.loads(analysis_file.read_text())
            # Check for key fields
            required = {"experiment_id", "dataset_strategy", "recommended_scorers"}
            if required.issubset(data.keys()):
                results["analysis_valid"] = True
            else:
                results["errors"].append(f"Analysis missing required fields: {required - data.keys()}")
                results["valid"] = False
        except json.JSONDecodeError as e:
            results["errors"].append(f"Analysis file JSON error: {e}")
            results["valid"] = False
    else:
        results["errors"].append(f"Analysis file not found: {analysis_file}")
        results["valid"] = False

    return results


def verify_worker_outputs(session_dir: Path, task_id: int) -> dict:
    """Check task status updated and artifacts created.

    Args:
        session_dir: Session directory to check
        task_id: Task ID to verify

    Returns:
        Dict with validation results
    """
    set_session_dir(session_dir)

    results = {
        "valid": True,
        "task_found": False,
        "task_completed": False,
        "artifacts_created": [],
        "errors": [],
    }

    # Check task status
    tasks_file = get_tasks_file()
    if not tasks_file.exists():
        results["errors"].append("Tasks file not found")
        results["valid"] = False
        return results

    try:
        data = json.loads(tasks_file.read_text())
        tasks = data.get("tasks", []) if isinstance(data, dict) else data

        for task in tasks:
            if task.get("id") == task_id:
                results["task_found"] = True
                results["task_status"] = task.get("status")
                if task.get("status") == "completed":
                    results["task_completed"] = True
                break

        if not results["task_found"]:
            results["errors"].append(f"Task {task_id} not found")
            results["valid"] = False

    except json.JSONDecodeError as e:
        results["errors"].append(f"Tasks file JSON error: {e}")
        results["valid"] = False

    # Check for artifacts in evaluation directory
    eval_dir = session_dir / "evaluation"
    if eval_dir.exists():
        for artifact in ["eval_dataset.py", "scorers.py", "run_eval.py"]:
            artifact_path = eval_dir / artifact
            if artifact_path.exists():
                results["artifacts_created"].append(artifact)

    return results


# =============================================================================
# INTEGRATION TEST
# =============================================================================


async def run_integration_test(
    experiment_id: str,
    max_iterations: int = 2,
    mock: bool = False,
    on_progress: Optional[Callable[[str, dict], None]] = None,
) -> TestResult:
    """Run full autonomous loop with limited iterations.

    Args:
        experiment_id: MLflow experiment ID to analyze
        max_iterations: Maximum number of iterations
        mock: If True, use mock MLflow client
        on_progress: Optional callback for progress updates (not used for integration tests)

    Returns:
        TestResult with overall success status
    """
    import tempfile
    import time

    from .agent import run_autonomous, setup_mlflow
    from .mlflow_ops import get_task_status

    # Set up session directory
    session_dir = Path(tempfile.mkdtemp(prefix="mlflow-eval-integration-test-"))
    set_session_dir(session_dir)

    # Set up MLflow tracing (if not mock)
    if not mock:
        setup_mlflow()

    start_time = time.time()
    trace_id = None

    try:
        with mlflow.start_span(name="integration_test") as span:
            trace_id = mlflow.get_current_active_span().request_id if mlflow.get_current_active_span() else None

            # Run autonomous loop
            await run_autonomous(experiment_id, max_iterations=max_iterations)

            span.set_attribute("test_type", "integration")
            span.set_attribute("experiment_id", experiment_id)
            span.set_attribute("max_iterations", max_iterations)

    except Exception as e:
        logger.exception("Integration test failed")
        return TestResult(
            success=False,
            error=str(e),
            trace_id=trace_id,
        )

    duration_ms = int((time.time() - start_time) * 1000)

    # Get final task status
    task_status = get_task_status()

    return TestResult(
        success=task_status.get("completed", 0) > 0,
        trace_id=trace_id,
        outputs=task_status,
        duration_ms=duration_ms,
    )


# =============================================================================
# CLI HELPERS
# =============================================================================


def print_test_result(result: TestResult, verbose: bool = False) -> None:
    """Print test result to console.

    Args:
        result: TestResult to print
        verbose: If True, print full output details
    """
    status = "PASS" if result.success else "FAIL"
    print(f"\n{'=' * 50}")
    print(f"Test Result: {status}")
    print(f"{'=' * 50}")

    if result.trace_id:
        print(f"Trace ID: {result.trace_id}")
        print(f"  Use: /analyze-trace {result.trace_id}")

    if result.session_id:
        print(f"Session ID: {result.session_id}")

    if result.duration_ms:
        print(f"Duration: {result.duration_ms}ms")

    if result.cost_usd:
        print(f"Cost: ${result.cost_usd:.4f}")

    if result.error:
        print(f"\nError: {result.error}")

    if verbose and result.outputs:
        print(f"\nOutputs:")
        print(json.dumps(result.outputs, indent=2, default=str))
    elif result.outputs and not result.success:
        # Always show errors on failure
        if "errors" in result.outputs:
            print(f"\nErrors: {result.outputs['errors']}")

    print()
