"""Sidebar configuration panel for the MLflow Eval Agent app.

Provides:
- Experiment ID input
- Volume path configuration
- Model selection
- Mode selection (Autonomous / Interactive)
- Session management
- Start/Stop controls
"""

from pathlib import Path
from typing import Optional

import streamlit as st

from ..auth import UserIdentity
from ..services.session_manager import list_user_sessions, SessionInfo
from .output_stream import clear_output


def render_sidebar(user: UserIdentity) -> dict:
    """Render sidebar configuration panel.

    Args:
        user: Current user identity.

    Returns:
        Configuration dictionary with user selections.
    """
    st.sidebar.title("MLflow Eval Agent")

    # User info
    st.sidebar.markdown(f"**User:** {user.display_name}")
    if not user.is_authenticated:
        st.sidebar.caption("(Local development mode)")

    st.sidebar.divider()

    # Configuration section
    st.sidebar.subheader("Configuration")

    # Experiment ID (required)
    experiment_id = st.sidebar.text_input(
        "Experiment ID",
        value=st.session_state.get("experiment_id", ""),
        placeholder="e.g., 2181280362153689",
        help="MLflow experiment ID to analyze",
    )
    st.session_state["experiment_id"] = experiment_id

    # Volume path - context-aware default
    if user.is_authenticated:
        default_volume = "/Volumes/users/alex_miller/mlflow-eval"
    else:
        # Use project root (where sessions/ already exists) for local development
        default_volume = str(Path(__file__).parent.parent.parent)

    volume_path = st.sidebar.text_input(
        "Session Storage Path",
        value=st.session_state.get("volume_path", default_volume),
        help="UC Volume path on Databricks, or local directory for development",
    )
    st.session_state["volume_path"] = volume_path

    # Model selection
    model_options = {
        "Claude Opus 4.5 (Best)": "databricks-claude-opus-4-5",
        "Claude Sonnet 4 (Faster)": "databricks-claude-sonnet-4",
    }
    model_label = st.sidebar.selectbox(
        "Model",
        options=list(model_options.keys()),
        index=0,
        help="Model to use for evaluation",
    )
    model = model_options[model_label]
    st.session_state["model"] = model

    st.sidebar.divider()

    # Mode selection
    st.sidebar.subheader("Mode")
    mode = st.sidebar.radio(
        "Select mode",
        options=["Autonomous", "Interactive"],
        horizontal=True,
        help="Autonomous: Auto-continue loop until complete. Interactive: Chat-based queries.",
        label_visibility="collapsed",
    )
    st.session_state["mode"] = mode.lower()

    # Mode-specific settings
    if mode == "Autonomous":
        max_iterations = st.sidebar.number_input(
            "Max Iterations",
            min_value=1,
            max_value=100,
            value=st.session_state.get("max_iterations", 10),
            help="Maximum number of iterations before stopping",
        )
        st.session_state["max_iterations"] = max_iterations

    st.sidebar.divider()

    # Session management
    st.sidebar.subheader("Session")

    # Show current session
    current_session = st.session_state.get("session_id")
    if current_session:
        st.sidebar.markdown(f"**Current:** `{current_session}`")

    # New session button
    if st.sidebar.button("New Session", use_container_width=True):
        # Clear session identifiers
        st.session_state.pop("session_id", None)
        st.session_state.pop("session_dir", None)
        st.session_state.pop("agent_session_id", None)
        # Clear all output state including iteration history
        clear_output(preserve_history=False)
        st.rerun()

    # Past sessions
    if volume_path and user.session_prefix:
        past_sessions = list_user_sessions(user.session_prefix, volume_path, limit=5)

        if past_sessions:
            st.sidebar.caption("Recent sessions:")
            for session in past_sessions:
                _render_session_item(session)

    st.sidebar.divider()

    # Controls
    is_running = st.session_state.get("running", False)

    col1, col2 = st.sidebar.columns(2)

    with col1:
        start_disabled = is_running or not experiment_id
        if st.button(
            "Start",
            type="primary",
            disabled=start_disabled,
            use_container_width=True,
        ):
            st.session_state["start_requested"] = True
            st.rerun()

    with col2:
        stop_disabled = not is_running
        if st.button(
            "Stop",
            disabled=stop_disabled,
            use_container_width=True,
        ):
            st.session_state["stop_requested"] = True
            st.rerun()

    # Status indicator
    if is_running:
        st.sidebar.markdown(":green_circle: **Running**")
    else:
        st.sidebar.markdown(":white_circle: **Stopped**")

    # Return configuration
    return {
        "experiment_id": experiment_id,
        "volume_path": volume_path,
        "model": model,
        "mode": mode.lower(),
        "max_iterations": st.session_state.get("max_iterations", 10),
    }


def _render_session_item(session: SessionInfo) -> None:
    """Render a past session item in sidebar.

    Args:
        session: Session information.
    """
    # Format timestamp
    time_str = session.created_at.strftime("%Y-%m-%d %H:%M")

    # Status icon
    status_icons = {
        "completed": ":white_check_mark:",
        "running": ":hourglass_flowing_sand:",
        "error": ":x:",
    }
    icon = status_icons.get(session.status, ":question:")

    # Progress
    if session.task_count > 0:
        progress = f"{session.completed_count}/{session.task_count}"
    else:
        progress = "No tasks"

    # Session button
    label = f"{icon} {time_str} ({progress})"

    if st.sidebar.button(
        label,
        key=f"session_{session.session_id}",
        use_container_width=True,
    ):
        st.session_state["session_id"] = session.session_id
        st.session_state["session_dir"] = str(session.session_dir)
        st.session_state["experiment_id"] = session.experiment_id
        st.rerun()


def get_config_validation_errors(config: dict, is_authenticated: bool = True) -> list[str]:
    """Validate configuration and return errors.

    Args:
        config: Configuration dictionary.
        is_authenticated: Whether user is authenticated (Databricks vs local).

    Returns:
        List of error messages (empty if valid).
    """
    errors = []

    if not config.get("experiment_id"):
        errors.append("Experiment ID is required")

    if not config.get("volume_path"):
        errors.append("Session storage path is required")

    volume_path = config.get("volume_path", "")
    if volume_path:
        if is_authenticated:
            # On Databricks, require UC Volume path
            if not volume_path.startswith("/Volumes/"):
                errors.append("Volume path must start with /Volumes/")
        else:
            # Locally, just require an absolute path
            if not volume_path.startswith("/"):
                errors.append("Volume path must be an absolute path")

    return errors
