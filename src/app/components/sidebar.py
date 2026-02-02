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

        # Runtime info
        runtime = detect_runtime()
        st.caption(f"Runtime: {runtime.context.value}")

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
                st.rerun()
        with col2:
            if st.button("Clear", type="secondary", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
