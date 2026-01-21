"""Task progress visualization component.

Displays:
- Progress bar with completion percentage
- Task list with status badges
- Analysis summary from initializer
- Validation results
"""

from pathlib import Path
from typing import Optional

import streamlit as st

from ..services.state_reader import (
    get_task_status,
    get_analysis_summary,
    get_validation_results,
    TaskStatus,
    AnalysisSummary,
)


def render_progress_tracker(session_dir: Optional[Path]) -> None:
    """Render task progress visualization.

    Args:
        session_dir: Session directory path (or None if no session).
    """
    if not session_dir:
        st.info("No active session. Start the agent to see progress.")
        return

    session_path = Path(session_dir) if isinstance(session_dir, str) else session_dir

    # Get task status
    task_status = get_task_status(session_path)

    # Header with progress
    st.markdown("### Task Progress")

    if task_status.total == 0:
        st.info("Waiting for initializer to create task plan...")
        _render_analysis_summary(session_path)
        return

    # Progress bar
    progress = task_status.completed / task_status.total if task_status.total > 0 else 0
    st.progress(progress, text=f"{task_status.completed}/{task_status.total} tasks completed")

    # Status metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total", task_status.total)
    with col2:
        st.metric("Completed", task_status.completed)
    with col3:
        st.metric("Pending", task_status.pending)
    with col4:
        st.metric("Failed", task_status.failed)

    # Task list
    st.markdown("### Tasks")
    _render_task_list(task_status)

    # Analysis summary
    st.markdown("### Analysis Summary")
    _render_analysis_summary(session_path)

    # Validation results
    validation = get_validation_results(session_path)
    if validation:
        st.markdown("### Validation")
        _render_validation_results(validation)


def _render_task_list(task_status: TaskStatus) -> None:
    """Render task list with status badges.

    Args:
        task_status: Task status information.
    """
    for task in task_status.tasks:
        status = task.get("status", "pending")
        name = task.get("name", "Unknown task")
        task_type = task.get("type", "")

        # Status badge
        status_badges = {
            "completed": ":white_check_mark:",
            "pending": ":hourglass_flowing_sand:",
            "failed": ":x:",
            "in_progress": ":arrow_forward:",
        }
        badge = status_badges.get(status, ":question:")

        # Task row
        with st.expander(f"{badge} {name}", expanded=status == "failed"):
            st.markdown(f"**Type:** {task_type}")
            st.markdown(f"**Status:** {status}")

            if task.get("description"):
                st.markdown(f"**Description:** {task['description']}")

            if task.get("artifacts"):
                st.markdown("**Artifacts:**")
                for artifact in task["artifacts"]:
                    st.code(artifact)

            if task.get("failure_reason"):
                st.error(f"**Failure:** {task['failure_reason']}")

            if task.get("attempts"):
                st.caption(f"Attempts: {task['attempts']}")


def _render_analysis_summary(session_path: Path) -> None:
    """Render analysis summary from initializer.

    Args:
        session_path: Session directory path.
    """
    summary = get_analysis_summary(session_path)

    if not summary:
        st.info("Analysis not yet available.")
        return

    # Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Traces Analyzed", summary.trace_count)
    with col2:
        st.metric("Experiment ID", summary.experiment_id)

    # Status breakdown
    if summary.status_breakdown:
        st.markdown("**Status Breakdown:**")
        for status, count in summary.status_breakdown.items():
            st.markdown(f"- {status}: {count}")

    # Key findings
    if summary.key_findings:
        st.markdown("**Key Findings:**")
        for finding in summary.key_findings[:5]:
            st.markdown(f"- {finding}")

    # Recommendations
    if summary.recommendations:
        st.markdown("**Recommendations:**")
        for rec in summary.recommendations[:5]:
            st.markdown(f"- {rec}")


def _render_validation_results(validation: dict) -> None:
    """Render validation results.

    Args:
        validation: Validation results dictionary.
    """
    col1, col2 = st.columns(2)

    with col1:
        script_ok = validation.get("script_success", False)
        if script_ok:
            st.success("Script runs successfully")
        else:
            st.error("Script has errors")

    with col2:
        scorers_ok = validation.get("scorers_valid", False)
        if scorers_ok:
            st.success("Scorers returning valid results")
        else:
            st.warning("Some scorers have issues")

    # Error details
    if validation.get("errors"):
        with st.expander("Error Details", expanded=True):
            for error in validation["errors"]:
                st.code(error)

    # Scorer results
    if validation.get("scorer_results"):
        with st.expander("Scorer Results"):
            for scorer, result in validation["scorer_results"].items():
                st.markdown(f"**{scorer}:** {result}")
