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
import streamlit.components.v1 as components

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


_CHAT_CONTAINER_HEIGHT = 600


def _inject_chat_scroll_css():
    """Inject CSS to make the chat container fill available viewport height."""
    st.markdown(
        """
        <style>
        .st-key-chat-scroll [data-testid="stVerticalBlockBorderWrapper"] > div {
            max-height: calc(100vh - 180px) !important;
            height: calc(100vh - 180px) !important;
        }
        .st-key-chat-scroll [data-testid="stVerticalBlockBorderWrapper"] {
            border: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _inject_scroll_to_bottom():
    """Inject JS to scroll the chat container to the bottom."""
    components.html(
        """
        <script>
        (function() {
            const container = parent.document.querySelector(
                '.st-key-chat-scroll [data-testid="stVerticalBlockBorderWrapper"] > div'
            );
            if (container) {
                container.scrollTop = container.scrollHeight;
            }
        })();
        </script>
        """,
        height=0,
        width=0,
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
    if "auto_session_dir" not in st.session_state:
        st.session_state.auto_session_dir = None
    if "abort_requested" not in st.session_state:
        st.session_state.abort_requested = False
    if "mode" not in st.session_state:
        st.session_state.mode = "interactive"
    if "auto_start_requested" not in st.session_state:
        st.session_state.auto_start_requested = False
    if "show_file_panel" not in st.session_state:
        st.session_state.show_file_panel = True


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

    # Pass session directory from autonomous run for file continuity
    auto_session_dir = st.session_state.get("auto_session_dir")
    if auto_session_dir:
        config.session_dir = Path(auto_session_dir)

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
    """Display existing chat messages with collapsible autonomous sessions."""
    messages = st.session_state.messages

    # Pre-compute the last autonomous message index for expand logic
    last_autonomous_idx = max(
        (i for i, m in enumerate(messages) if m.get("role") == "autonomous"),
        default=-1,
    )

    for idx, message in enumerate(messages):
        role = message.get("role", "user")

        if role == "autonomous":
            # Autonomous iteration — collapsible expander, latest expanded
            iteration = message.get("iteration", "?")
            phase = message.get("phase", "worker")
            is_latest = (idx == last_autonomous_idx)
            with st.expander(
                f"Session {iteration} — {phase.title()}",
                expanded=is_latest,
            ):
                if "parts" in message:
                    ChatRenderer.render_history_message(message["parts"])

        elif role == "auto_status":
            # Status message from autonomous run
            status = message.get("status", "")
            content = message.get("content", "")
            if status == "error":
                st.error(content)
            else:
                st.success(content)

        elif role in ("user", "assistant"):
            with st.chat_message(role):
                if "parts" in message:
                    ChatRenderer.render_history_message(message["parts"])
                else:
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

    # Refresh panel if visible (files may have been modified by assistant)
    panel_placeholder = st.session_state.get("_panel_placeholder")
    if panel_placeholder and st.session_state.get("auto_session_dir"):
        _refresh_panel(panel_placeholder)



def _run_autonomous_streaming(experiment_id: str, max_iterations: int, panel_placeholder=None):
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
                _refresh_panel(panel_placeholder)

            # Render task header as collapsible expander
            phase_label = event.phase or "worker"
            current_iteration = event.iteration
            session_expander = st.expander(
                f"Session {event.iteration} — {phase_label.title()}",
                expanded=True,
            )

            # Render all agent results inside the expander
            with session_expander:
                renderer = ChatRenderer()
                parts = renderer.render(_agent_results_until_boundary(events))
            last_event = _agent_results_until_boundary.last_event

            # Store iteration in unified message list
            if parts:
                st.session_state.messages.append({
                    "role": "autonomous",
                    "iteration": current_iteration,
                    "phase": phase_label,
                    "parts": parts,
                })
                _refresh_panel(panel_placeholder)

            # Handle the boundary event that ended the iteration
            if last_event:
                if last_event.event_type == "progress":
                    _try_detect_session_dir()
                    _refresh_panel(panel_placeholder)
                elif last_event.event_type == "error":
                    error_msg = f"Error in session {last_event.iteration}: {last_event.error_message}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "auto_status",
                        "status": "error",
                        "content": error_msg,
                    })
                elif last_event.event_type == "complete":
                    _refresh_panel(panel_placeholder)
                    break

        elif event.event_type == "progress":
            _try_detect_session_dir()

        elif event.event_type == "error":
            error_msg = f"Error in session {event.iteration}: {event.error_message}"
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "auto_status",
                "status": "error",
                "content": error_msg,
            })

        elif event.event_type == "complete":
            break

    st.session_state.auto_running = False
    st.session_state.abort_requested = False
    st.success("Autonomous run complete!")
    _refresh_panel(panel_placeholder)
    st.session_state.messages.append({
        "role": "auto_status",
        "status": "complete",
        "content": "Autonomous run complete!",
    })


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


def _render_side_panel():
    """Render file viewer and task progress in the autonomous side panel."""
    session_dir = st.session_state.auto_session_dir
    if session_dir:
        render_file_viewer(Path(session_dir))
        st.divider()
        render_task_progress()
        with st.expander("Task Details"):
            render_task_list()
    else:
        st.caption("Waiting for session to start...")


def _refresh_panel(panel_placeholder):
    """Re-render the side panel in its placeholder for live updates."""
    if panel_placeholder is None:
        return
    with panel_placeholder.container():
        _render_side_panel()


def main():
    """Main application entry point."""
    st.title("MLflow Eval Agent")

    initialize_session_state()
    render_sidebar()

    mode = st.session_state.get("mode", "interactive")
    is_auto = mode == "autonomous"
    has_session = bool(st.session_state.get("auto_session_dir"))
    show_panel = st.session_state.get("show_file_panel", True) and (is_auto or has_session)

    # Inject CSS for viewport-relative chat height
    _inject_chat_scroll_css()

    if show_panel:
        col_chat, col_panel = st.columns([2, 1])
    else:
        col_chat = st.container()
        col_panel = None

    # Create scrollable chat container inside the chat column
    with col_chat:
        chat_scroll = st.container(
            height=_CHAT_CONTAINER_HEIGHT, key="chat-scroll", border=False
        )

    # Render chat history inside the scrollable container
    with chat_scroll:
        display_chat_history()

    panel_placeholder = None
    if col_panel is not None:
        with col_panel:
            panel_placeholder = st.empty()
            with panel_placeholder.container():
                _render_side_panel()
    st.session_state._panel_placeholder = panel_placeholder

    # Handle autonomous start (flag set by sidebar button)
    if st.session_state.get("auto_start_requested"):
        st.session_state.auto_start_requested = False
        st.session_state.auto_running = True
        experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID", "")
        max_iterations = st.session_state.get("auto_max_iterations", 10)
        with chat_scroll:
            _run_autonomous_streaming(experiment_id, max_iterations, panel_placeholder=panel_placeholder)

    # Chat input at page level - docks to bottom of page
    prompt = st.chat_input(
        "Ask about your MLflow traces...",
        disabled=st.session_state.get("auto_running", False),
    )
    if prompt:
        with chat_scroll:
            handle_user_input(prompt)
            _inject_scroll_to_bottom()


if __name__ == "__main__":
    main()
