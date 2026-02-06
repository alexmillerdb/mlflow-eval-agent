"""Task progress display components."""
import streamlit as st

STATUS_ICONS = {
    "pending": "⏳",
    "in_progress": "🔄",
    "completed": "✅",
    "failed": "❌",
}


def render_task_progress():
    """Display compact progress bar with metrics."""
    from src.agent.mlflow_ops import get_task_status

    status = get_task_status()
    total = status["total"]

    if total == 0:
        st.info("No tasks yet. Start an autonomous run to create tasks.")
        return

    completed = status["completed"]
    progress = completed / total if total > 0 else 0

    st.progress(progress, text=f"{completed}/{total} tasks completed")

    col1, col2, col3 = st.columns(3)
    col1.metric("Completed", status["completed"])
    col2.metric("Pending", status["pending"])
    col3.metric("Failed", status["failed"])


def render_task_list():
    """Display expandable task list with details."""
    from src.agent.mlflow_ops import get_task_status

    status = get_task_status()
    tasks = status.get("tasks", [])

    if not tasks:
        return

    for task in tasks:
        icon = STATUS_ICONS.get(task.get("status", "pending"), "❓")
        task_type = task.get("type", "unknown")
        task_status = task.get("status", "pending")

        with st.expander(f"{icon} {task_type} - {task_status}"):
            st.json(task)
