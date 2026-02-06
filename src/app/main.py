"""Streamlit chat interface for MLflow Evaluation Agent.

A chat interface that:
- Streams agent responses with tool call expanders via ChatRenderer
- Maintains structured conversation history in session state
- Provides sidebar for experiment configuration
- Streams autonomous mode output with task headers and file viewer
"""

import os
import sys
import threading
from pathlib import Path

# Add parent directory to path for streamlit run compatibility
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st

from src.app.streaming import async_to_sync_generator
from src.app.components import (
    render_sidebar,
    render_task_progress,
    render_task_list,
    ChatRenderer,
    render_file_viewer,
)


# Page configuration
st.set_page_config(
    page_title="MLflow Eval Agent",
    page_icon=":microscope:",
    layout="wide",
)


def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "auto_running" not in st.session_state:
        st.session_state.auto_running = False
    if "auto_messages" not in st.session_state:
        st.session_state.auto_messages = []
    if "auto_session_dir" not in st.session_state:
        st.session_state.auto_session_dir = None
    if "abort_requested" not in st.session_state:
        st.session_state.abort_requested = False


def stream_agent_response_full(prompt: str):
    """Stream full AgentResult objects from the agent.

    Args:
        prompt: User's query

    Yields:
        AgentResult objects with all event types.
    """
    from src.agent.agent import MLflowAgent, setup_mlflow
    from src.core.config import Config
    from src.app.auth import get_obo_token

    # Initialize MLflow once
    if not st.session_state.initialized:
        setup_mlflow()
        st.session_state.initialized = True

    # Create agent with current config + OBO token for per-user file access
    config = Config.from_env(validate=False)
    config.user_token = get_obo_token()
    agent = MLflowAgent(config)

    # Capture session_id before creating closure (can't access st.session_state from worker thread)
    current_session_id = st.session_state.session_id
    # Capture a list to pass session_id back (mutable container for thread-safe handoff)
    session_id_out = [None]

    def create_query():
        return agent.query(prompt, session_id=current_session_id)

    for result in async_to_sync_generator(create_query):
        if result.event_type == "result" and result.session_id:
            session_id_out[0] = result.session_id
        yield result

    # Store session ID for continuity after generator completes
    if session_id_out[0]:
        st.session_state.session_id = session_id_out[0]


def display_chat_history():
    """Display existing chat messages with full tool expanders."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "parts" in message:
                # Structured message format with tool expanders
                ChatRenderer.render_history_message(message["parts"])
            else:
                # Legacy format (plain text content)
                st.markdown(message.get("content", ""))


def handle_user_input(prompt: str):
    """Handle new user input from chat.

    Args:
        prompt: User's message text.
    """
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream assistant response with ChatRenderer
    with st.chat_message("assistant"):
        renderer = ChatRenderer()
        parts = renderer.render(stream_agent_response_full(prompt))

    # Store structured assistant response in history
    st.session_state.messages.append({"role": "assistant", "parts": parts})


def render_autonomous_tab():
    """Render autonomous evaluation mode controls with streaming output."""
    st.subheader("Autonomous Evaluation")

    col1, col2 = st.columns([2, 1])
    with col1:
        auto_exp_id = st.text_input(
            "Experiment ID",
            value=os.getenv("MLFLOW_EXPERIMENT_ID", ""),
            key="auto_exp_id",
        )
    with col2:
        max_iterations = st.number_input(
            "Max Iterations",
            min_value=1,
            max_value=50,
            value=10,
            key="max_iterations",
        )

    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        start_clicked = st.button("Start Autonomous Run", type="primary")
    with btn_col2:
        if st.session_state.auto_running:
            if st.button("Stop"):
                st.session_state.abort_requested = True
                if hasattr(st.session_state, '_abort_event'):
                    st.session_state._abort_event.set()

    if start_clicked:
        if not auto_exp_id:
            st.error("Please enter an Experiment ID")
            return

        os.environ["MLFLOW_EXPERIMENT_ID"] = auto_exp_id
        st.session_state.auto_running = True
        st.session_state.abort_requested = False
        st.session_state.auto_messages = []
        _run_autonomous_streaming(auto_exp_id, max_iterations)

    st.divider()

    # Split layout: chat output left, file viewer right
    col_chat, col_files = st.columns([2, 1])

    with col_chat:
        _display_autonomous_history()

    with col_files:
        session_dir = st.session_state.auto_session_dir
        if session_dir:
            render_file_viewer(Path(session_dir))
        else:
            st.info("Files will appear here during an autonomous run.")

    # Progress display at bottom
    render_task_progress()
    with st.expander("Task Details"):
        render_task_list()


def _run_autonomous_streaming(experiment_id: str, max_iterations: int):
    """Run autonomous evaluation with streaming output to the UI."""
    from src.agent.autonomous import stream_autonomous

    abort_event = threading.Event()
    st.session_state._abort_event = abort_event

    def abort_check():
        return abort_event.is_set()

    def create_stream():
        return stream_autonomous(experiment_id, max_iterations, abort_check=abort_check)

    current_iteration = 0
    last_event = None

    events = iter(async_to_sync_generator(create_stream))
    for event in events:
        if event.event_type == "task_header":
            # Track session directory from first event
            if event.iteration == 1 and st.session_state.auto_session_dir is None:
                _try_detect_session_dir()

            # Render task header
            phase_label = event.phase or "worker"
            st.markdown(f"---\n### Session {event.iteration} ({phase_label})")
            current_iteration = event.iteration

            # Render all agent results for this iteration with a single ChatRenderer
            renderer = ChatRenderer()
            parts = renderer.render(_agent_results_until_boundary(events))
            last_event = _agent_results_until_boundary.last_event

            # Store iteration
            if parts:
                st.session_state.auto_messages.append({
                    "iteration": current_iteration,
                    "parts": parts,
                })

            # Handle the boundary event that ended the iteration
            if last_event:
                if last_event.event_type == "progress":
                    _try_detect_session_dir()
                elif last_event.event_type == "error":
                    st.error(f"Error in session {last_event.iteration}: {last_event.error_message}")
                elif last_event.event_type == "complete":
                    break

        elif event.event_type == "progress":
            _try_detect_session_dir()

        elif event.event_type == "error":
            st.error(f"Error in session {event.iteration}: {event.error_message}")

        elif event.event_type == "complete":
            break

    st.session_state.auto_running = False
    st.session_state.abort_requested = False
    st.success("Autonomous run complete!")


def _agent_results_until_boundary(events_iter):
    """Yield AgentResult objects from events until a non-agent_result event is hit.

    Stores the boundary event in the function attribute `last_event`.
    """
    _agent_results_until_boundary.last_event = None
    for event in events_iter:
        if event.event_type == "agent_result" and event.agent_result:
            yield event.agent_result
        else:
            _agent_results_until_boundary.last_event = event
            return


def _try_detect_session_dir():
    """Try to detect and store the session directory from mlflow_ops."""
    try:
        from src.agent.mlflow_ops import get_session_dir
        session_dir = get_session_dir()
        if session_dir and session_dir.exists():
            st.session_state.auto_session_dir = str(session_dir)
    except Exception:
        pass


def _display_autonomous_history():
    """Display previously streamed autonomous messages."""
    for msg in st.session_state.auto_messages:
        iteration = msg.get("iteration", "?")
        parts = msg.get("parts", [])
        st.markdown(f"---\n### Session {iteration}")
        ChatRenderer.render_history_message(parts)


def main():
    """Main application entry point."""
    st.title("MLflow Eval Agent")

    initialize_session_state()
    render_sidebar()

    tab1, tab2 = st.tabs(["Interactive", "Autonomous"])

    with tab1:
        display_chat_history()

    with tab2:
        render_autonomous_tab()

    # Chat input at page level - docks to bottom of page
    prompt = st.chat_input("Ask about your MLflow traces...")
    if prompt:
        with tab1:
            handle_user_input(prompt)


if __name__ == "__main__":
    main()
