"""State file reader for displaying progress and generated code.

Reads state files from session directory:
- eval_tasks.json: Task list and progress
- state/analysis.json: Analysis summary
- evaluation/: Generated code files
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TaskStatus:
    """Task progress status."""
    total: int
    completed: int
    pending: int
    failed: int
    tasks: list[dict]


@dataclass
class AnalysisSummary:
    """Analysis summary from initializer."""
    experiment_id: str
    trace_count: int
    status_breakdown: dict
    key_findings: list[str]
    recommendations: list[str]


def get_task_status(session_dir: Path) -> TaskStatus:
    """Read task status from session directory.

    Args:
        session_dir: Path to session directory.

    Returns:
        TaskStatus with progress information.
    """
    tasks_file = session_dir / "eval_tasks.json"

    if not tasks_file.exists():
        return TaskStatus(
            total=0,
            completed=0,
            pending=0,
            failed=0,
            tasks=[],
        )

    try:
        data = json.loads(tasks_file.read_text())

        # Handle both formats: list or dict with "tasks" key
        if isinstance(data, dict):
            tasks = data.get("tasks", [])
        else:
            tasks = data

        return TaskStatus(
            total=len(tasks),
            completed=sum(1 for t in tasks if t.get("status") == "completed"),
            pending=sum(1 for t in tasks if t.get("status") == "pending"),
            failed=sum(1 for t in tasks if t.get("status") == "failed"),
            tasks=tasks,
        )
    except Exception as e:
        logger.warning(f"Could not read task status: {e}")
        return TaskStatus(total=0, completed=0, pending=0, failed=0, tasks=[])


def get_analysis_summary(session_dir: Path) -> Optional[AnalysisSummary]:
    """Read analysis summary from session directory.

    Args:
        session_dir: Path to session directory.

    Returns:
        AnalysisSummary or None if not found.
    """
    analysis_file = session_dir / "state" / "analysis.json"

    if not analysis_file.exists():
        return None

    try:
        data = json.loads(analysis_file.read_text())

        return AnalysisSummary(
            experiment_id=data.get("experiment_id", ""),
            trace_count=data.get("trace_count", 0),
            status_breakdown=data.get("status_breakdown", {}),
            key_findings=data.get("key_findings", []),
            recommendations=data.get("recommendations", []),
        )
    except Exception as e:
        logger.warning(f"Could not read analysis summary: {e}")
        return None


def list_generated_files(session_dir: Path) -> list[dict]:
    """List generated evaluation files.

    Args:
        session_dir: Path to session directory.

    Returns:
        List of file info dicts with name, path, and content.
    """
    eval_dir = session_dir / "evaluation"

    if not eval_dir.exists():
        return []

    files = []
    expected_files = ["eval_dataset.py", "scorers.py", "run_eval.py"]

    for filename in expected_files:
        filepath = eval_dir / filename
        if filepath.exists():
            try:
                content = filepath.read_text()
                files.append({
                    "name": filename,
                    "path": str(filepath),
                    "content": content,
                    "size": len(content),
                })
            except Exception as e:
                logger.warning(f"Could not read {filename}: {e}")

    # Also include any other Python files
    for filepath in eval_dir.glob("*.py"):
        if filepath.name not in expected_files:
            try:
                content = filepath.read_text()
                files.append({
                    "name": filepath.name,
                    "path": str(filepath),
                    "content": content,
                    "size": len(content),
                })
            except Exception as e:
                logger.warning(f"Could not read {filepath.name}: {e}")

    return files


def get_validation_results(session_dir: Path) -> Optional[dict]:
    """Read validation results if available.

    Args:
        session_dir: Path to session directory.

    Returns:
        Validation results dict or None.
    """
    validation_file = session_dir / "state" / "validation_results.json"

    if not validation_file.exists():
        return None

    try:
        return json.loads(validation_file.read_text())
    except Exception as e:
        logger.warning(f"Could not read validation results: {e}")
        return None


def get_session_logs(session_dir: Path, limit: int = 100) -> list[str]:
    """Read session log entries.

    Args:
        session_dir: Path to session directory.
        limit: Maximum number of lines to return.

    Returns:
        List of log lines (newest first).
    """
    log_file = session_dir / "session.log"

    if not log_file.exists():
        return []

    try:
        lines = log_file.read_text().strip().split("\n")
        return lines[-limit:]  # Return last N lines
    except Exception as e:
        logger.warning(f"Could not read session logs: {e}")
        return []
