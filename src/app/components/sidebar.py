"""Sidebar configuration component."""
import os
import streamlit as st
from src.core.runtime import detect_runtime
from src.app.auth import get_current_user


def render_sidebar():
    """Render sidebar with config, user info, and session controls."""
    with st.sidebar:
        # User info (if authenticated)
        user = get_current_user()
        if user:
            st.caption(f"👤 {user['display_name']}")

        st.header("Configuration")

        # Experiment ID
        experiment_id = st.text_input(
            "Experiment ID",
            value=os.getenv("MLFLOW_EXPERIMENT_ID", ""),
            help="MLflow experiment ID to analyze",
        )
        if experiment_id:
            os.environ["MLFLOW_EXPERIMENT_ID"] = experiment_id

        # Mode selector
        def _on_mode_change():
            st.session_state.mode = st.session_state.mode_selector.lower()

        st.selectbox(
            "Mode",
            ["Interactive", "Autonomous"],
            key="mode_selector",
            on_change=_on_mode_change,
            disabled=st.session_state.get("auto_running", False),
        )

        # Output path
        output_path = st.text_input(
            "Output Path",
            value=os.getenv("MLFLOW_AGENT_VOLUME_PATH", ""),
            help="UC Volume path for session storage. Leave empty for local.",
        )
        if output_path:
            os.environ["MLFLOW_AGENT_VOLUME_PATH"] = output_path

        # Runtime info
        runtime = detect_runtime()
        st.caption(f"Runtime: {runtime.context.value}")

        # Autonomous controls
        if st.session_state.get("mode", "interactive") == "autonomous":
            st.divider()
            st.subheader("Autonomous Controls")
            st.number_input(
                "Max Iterations",
                min_value=1,
                max_value=50,
                value=10,
                key="auto_max_iterations",
            )
            col_start, col_stop = st.columns(2)
            with col_start:
                if st.button(
                    "Start",
                    type="primary",
                    use_container_width=True,
                    disabled=st.session_state.get("auto_running", False),
                ):
                    experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID", "")
                    if not experiment_id:
                        st.error("Set Experiment ID first")
                    else:
                        st.session_state.auto_start_requested = True
            with col_stop:
                if st.button(
                    "Stop",
                    use_container_width=True,
                    disabled=not st.session_state.get("auto_running", False),
                ):
                    st.session_state.abort_requested = True
                    if hasattr(st.session_state, "_abort_event"):
                        st.session_state._abort_event.set()

        # Show file panel toggle when in autonomous mode or when session files exist from previous run
        if st.session_state.get("mode", "interactive") == "autonomous" or st.session_state.get("auto_session_dir"):
            st.checkbox("Show File Panel", key="show_file_panel")

        st.divider()

        # Session controls
        st.subheader("Session")
        if st.session_state.get("session_id"):
            st.caption(f"ID: {st.session_state.session_id[:8]}...")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("New", type="secondary", use_container_width=True):
                st.session_state.messages = []
                st.session_state.session_id = None
                st.session_state.auto_session_dir = None
                st.session_state.auto_running = False
                st.rerun()
        with col2:
            if st.button("Clear", type="secondary", use_container_width=True):
                st.session_state.messages = []
                st.session_state.auto_session_dir = None
                st.session_state.auto_running = False
                st.rerun()
