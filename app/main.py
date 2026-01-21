"""MLflow Evaluation Agent Streamlit App.

Entry point for the Databricks App. Provides a web UI for:
- Autonomous evaluation mode (auto-continue loop)
- Interactive mode (chat-based queries)
- Real-time streaming output
- Task progress tracking
- Generated code viewing

Usage:
    streamlit run app/main.py
"""

import logging
import sys
import time
from pathlib import Path
from queue import Queue

import streamlit as st

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.auth import get_user_identity
from app.components import (
    render_sidebar,
    get_config_validation_errors,
    render_output_stream,
    process_stream_event,
    clear_output,
    render_error_banner,
    render_completion_banner,
    render_progress_tracker,
    render_code_viewer,
    render_run_instructions,
    render_message_history,
    render_chat_input,
    add_assistant_message,
    render_example_prompts,
)
from app.services import (
    get_user_session_dir,
    save_session_metadata,
    AgentRunner,
    poll_events,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for Streamlit app."""
    # Page configuration
    st.set_page_config(
        page_title="MLflow Eval Agent",
        page_icon=":robot_face:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Get user identity
    user = get_user_identity()

    # Initialize session state
    _init_session_state(user)

    # Render sidebar and get config
    config = render_sidebar(user)

    # Validate configuration
    errors = get_config_validation_errors(config, is_authenticated=user.is_authenticated)
    if errors and st.session_state.get("start_requested"):
        for error in errors:
            st.error(error)
        st.session_state["start_requested"] = False
        return

    # Handle start/stop requests
    _handle_control_requests(config, user)

    # Poll for events if running
    _poll_agent_events()

    # Main content area
    _render_main_content(config)


def _init_session_state(user):
    """Initialize session state with defaults.

    Args:
        user: Current user identity.
    """
    if "initialized" not in st.session_state:
        st.session_state["initialized"] = True
        st.session_state["user_prefix"] = user.session_prefix
        st.session_state["running"] = False
        st.session_state["queue"] = Queue()
        st.session_state["runner"] = None


def _handle_control_requests(config: dict, user):
    """Handle start/stop button requests.

    Args:
        config: Configuration dictionary.
        user: Current user identity.
    """
    # Handle start request
    if st.session_state.get("start_requested"):
        st.session_state["start_requested"] = False

        # Create or get session
        session_id, session_dir = get_user_session_dir(
            user_prefix=user.session_prefix,
            volume_path=config["volume_path"],
            session_id=st.session_state.get("session_id"),
        )

        st.session_state["session_id"] = session_id
        st.session_state["session_dir"] = str(session_dir)

        # Save session metadata
        save_session_metadata(
            session_dir=session_dir,
            experiment_id=config["experiment_id"],
            mode=config["mode"],
            model=config["model"],
        )

        # Clear previous output
        clear_output()

        # Create agent runner
        queue = st.session_state["queue"]
        runner = AgentRunner(
            queue=queue,
            experiment_id=config["experiment_id"],
            session_dir=session_dir,
            volume_path=config["volume_path"],
            model=config["model"],
        )
        st.session_state["runner"] = runner

        # Start agent based on mode
        if config["mode"] == "autonomous":
            runner.start_autonomous(max_iterations=config["max_iterations"])
        else:
            # Interactive mode - get prompt from chat interface
            prompt = st.session_state.get("pending_prompt")
            if prompt:
                runner.start_interactive(
                    prompt=prompt,
                    session_id=st.session_state.get("agent_session_id"),
                )
                st.session_state["pending_prompt"] = None

        st.session_state["running"] = True
        st.rerun()

    # Handle stop request
    if st.session_state.get("stop_requested"):
        st.session_state["stop_requested"] = False

        runner = st.session_state.get("runner")
        if runner:
            runner.stop()

        st.session_state["running"] = False
        st.rerun()


def _poll_agent_events():
    """Poll for agent events and update UI.

    Note: Interactive mode handles its own polling in _stream_interactive_response()
    to avoid st.rerun() flickering during streaming.
    """
    if not st.session_state.get("running"):
        return

    # Interactive mode handles its own polling loop with placeholders
    # This avoids st.rerun() flickering during streaming
    if st.session_state.get("mode") == "interactive":
        return

    queue = st.session_state.get("queue")
    if not queue:
        return

    # Poll for events (50ms timeout for responsive streaming)
    events = poll_events(queue, timeout=0.05)

    for event in events:
        process_stream_event(event)

        # Check for completion
        if event.event_type == "status":
            status = event.data.get("status")
            if status in ("stopped", "completed", "max_iterations"):
                st.session_state["running"] = False

    # Auto-rerun while running to continue polling (autonomous mode only)
    if st.session_state.get("running"):
        time.sleep(0.05)  # Reduced from 0.5s for more responsive streaming
        st.rerun()


def _render_main_content(config: dict):
    """Render main content area.

    Args:
        config: Configuration dictionary.
    """
    # Error/completion banners
    render_error_banner()
    render_completion_banner()

    # Mode-specific content
    if config["mode"] == "autonomous":
        _render_autonomous_mode()
    else:
        _render_interactive_mode()


def _render_autonomous_mode():
    """Render autonomous mode UI with tabs."""
    tab1, tab2, tab3 = st.tabs(["Output", "Progress", "Generated Code"])

    with tab1:
        render_output_stream()

    with tab2:
        session_dir = st.session_state.get("session_dir")
        render_progress_tracker(Path(session_dir) if session_dir else None)

    with tab3:
        session_dir = st.session_state.get("session_dir")
        render_code_viewer(Path(session_dir) if session_dir else None)
        render_run_instructions(Path(session_dir) if session_dir else None)


def _stream_interactive_response():
    """Stream agent response in interactive mode using placeholder updates.

    Uses a polling loop with st.empty() placeholders for smooth streaming
    without st.rerun() during execution.
    """
    queue = st.session_state.get("queue")
    if not queue:
        return

    # Create streaming UI inside a chat message
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        response_placeholder = st.empty()

        status_placeholder.markdown(":hourglass_flowing_sand: **Thinking...**")

        accumulated_text = ""
        current_tool = None

        # Poll until completion (no st.rerun during this loop)
        while st.session_state.get("running"):
            events = poll_events(queue, timeout=0.05)

            for event in events:
                process_stream_event(event)  # Update session state

                if event.event_type == "text":
                    accumulated_text = event.data.get("response", "")
                    response_placeholder.markdown(accumulated_text)

                elif event.event_type == "tool_use":
                    current_tool = event.data.get("tool_name", "")
                    status_placeholder.markdown(
                        f":hourglass_flowing_sand: Using `{current_tool}`..."
                    )

                elif event.event_type == "tool_result":
                    status_placeholder.markdown(":hourglass_flowing_sand: **Thinking...**")

                elif event.event_type == "status":
                    status = event.data.get("status")
                    if status in ("stopped", "completed", "max_iterations"):
                        st.session_state["running"] = False

            time.sleep(0.02)  # Small sleep to prevent CPU spin

        # Clear status, keep final response
        status_placeholder.empty()

        # Add to chat history for persistence
        if accumulated_text:
            add_assistant_message(accumulated_text)

    # Final rerun to show clean state
    st.rerun()


def _render_interactive_mode():
    """Render interactive mode UI."""
    col1, col2 = st.columns([2, 1])

    with col1:
        # Render message history first
        render_message_history()

        # If agent is running, show streaming response INLINE with chat
        # This appears right after the user's message, before the input area
        if st.session_state.get("running"):
            _stream_interactive_response()

        # Input area (after streaming response)
        prompt = render_chat_input()

        # Handle submitted prompt
        if prompt:
            st.session_state["pending_prompt"] = prompt
            st.session_state["start_requested"] = True
            st.rerun()

        # Example prompts
        example = render_example_prompts()
        if example:
            st.session_state["pending_prompt"] = example
            st.session_state["start_requested"] = True
            st.rerun()

    with col2:
        # Progress and code in sidebar column
        st.markdown("### Session Info")
        session_dir = st.session_state.get("session_dir")

        if session_dir:
            render_progress_tracker(Path(session_dir))

            st.markdown("---")
            st.markdown("#### Generated Code")
            render_code_viewer(Path(session_dir))
        else:
            st.info("Start a conversation to create a session.")


if __name__ == "__main__":
    main()
