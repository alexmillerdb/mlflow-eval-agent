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
    render_iteration_history,
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
from app.services.state_reader import (
    get_task_status,
    get_analysis_summary,
    list_generated_files,
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

    Note: Both interactive and autonomous modes handle their own polling loops
    with st.empty() placeholders to avoid st.rerun() flickering during streaming.
    This function is now a no-op - kept for potential future use.
    """
    # Both modes handle their own polling loops with placeholders
    # Interactive: _stream_interactive_response()
    # Autonomous: _stream_autonomous_response()
    return


def _update_output_placeholders(
    cost_ph,
    input_ph,
    output_ph,
    tools_ph,
    status_ph,
    text_ph,
    tool_calls_container,
    thinking_container,
):
    """Update Output tab placeholders from session state.

    Args:
        cost_ph: Placeholder for cost metric.
        input_ph: Placeholder for input tokens metric.
        output_ph: Placeholder for output tokens metric.
        tools_ph: Placeholder for tool calls metric.
        status_ph: Placeholder for status indicator.
        text_ph: Placeholder for text output.
        tool_calls_container: Container for tool calls.
        thinking_container: Container for thinking blocks.
    """
    import json

    # Update metrics
    cost_ph.metric("Cost", f"${st.session_state.get('total_cost', 0):.4f}")
    input_ph.metric("Input Tokens", f"{st.session_state.get('total_input_tokens', 0):,}")
    output_ph.metric("Output Tokens", f"{st.session_state.get('total_output_tokens', 0):,}")
    tools_ph.metric("Tool Calls", len(st.session_state.get("tool_calls", [])))

    # Update status
    if st.session_state.get("running"):
        phase = st.session_state.get("current_phase", "")
        iteration = st.session_state.get("current_iteration", 0)
        if phase and iteration:
            status_ph.markdown(f":hourglass_flowing_sand: **{phase.title()} (iteration {iteration})...**")
        else:
            status_ph.markdown(":hourglass_flowing_sand: **Agent is running...**")
    else:
        status_ph.empty()

    # Update text output
    text = st.session_state.get("accumulated_text", "")
    if text:
        text_ph.markdown(text)
    else:
        text_ph.info("No output yet. Agent is starting...")

    # Update tool calls
    tool_calls = st.session_state.get("tool_calls", [])
    if tool_calls:
        with tool_calls_container.container():
            with st.expander(f"Tool Calls ({len(tool_calls)})", expanded=False):
                for i, tc in enumerate(tool_calls):
                    st.markdown(f"**{i+1}. {tc.get('tool_name', 'Unknown')}**")
                    if tc.get("tool_input"):
                        st.code(json.dumps(tc["tool_input"], indent=2), language="json")
                    if tc.get("tool_result"):
                        st.text(tc["tool_result"][:500])
                    st.divider()

    # Update thinking blocks
    thinking_blocks = st.session_state.get("thinking_blocks", [])
    if thinking_blocks:
        with thinking_container.container():
            with st.expander(f"Thinking ({len(thinking_blocks)})", expanded=False):
                for i, tb in enumerate(thinking_blocks):
                    st.markdown(f"**Thought {i+1}:**")
                    st.text(tb[:1000])
                    st.divider()


def _update_progress_placeholder(progress_ph, session_path: Path):
    """Update Progress tab by re-reading task files.

    Args:
        progress_ph: Placeholder for progress content.
        session_path: Path to session directory.
    """
    if not session_path:
        progress_ph.info("No active session.")
        return

    task_status = get_task_status(session_path)
    analysis = get_analysis_summary(session_path)

    with progress_ph.container():
        # Progress bar
        if task_status.total > 0:
            progress = task_status.completed / task_status.total
            st.progress(progress, text=f"{task_status.completed}/{task_status.total} tasks completed")

            # Task list
            for task in task_status.tasks:
                status = task.get("status", "pending")
                name = task.get("name", "Unknown task")

                status_badges = {
                    "completed": ":white_check_mark:",
                    "pending": ":hourglass_flowing_sand:",
                    "failed": ":x:",
                    "in_progress": ":arrow_forward:",
                }
                badge = status_badges.get(status, ":question:")
                st.markdown(f"{badge} {name}")
        else:
            st.info("Waiting for initializer to create task plan...")

        # Analysis summary
        if analysis:
            st.markdown("---")
            st.markdown(f"**Traces analyzed:** {analysis.trace_count}")
            if analysis.key_findings:
                st.markdown("**Key findings:**")
                for finding in analysis.key_findings[:3]:
                    st.markdown(f"- {finding}")


def _update_code_placeholder(code_ph, session_path: Path):
    """Update Generated Code tab by re-reading files.

    Args:
        code_ph: Placeholder for code content.
        session_path: Path to session directory.
    """
    if not session_path:
        code_ph.info("No active session.")
        return

    files = list_generated_files(session_path)

    with code_ph.container():
        if not files:
            st.info("No generated files yet. The agent will create evaluation files during task execution.")
            return

        # Expected files checklist
        expected = {"eval_dataset.py", "scorers.py", "run_eval.py"}
        found = {f["name"] for f in files}

        for name in expected:
            if name in found:
                st.markdown(f":white_check_mark: `{name}`")
            else:
                st.markdown(f":hourglass_flowing_sand: `{name}` (pending)")

        # Extra files
        extra = found - expected
        for name in extra:
            st.markdown(f":page_facing_up: `{name}`")


def _stream_autonomous_response():
    """Stream autonomous mode using placeholders (no st.rerun during streaming).

    Uses a polling loop with st.empty() placeholders for smooth streaming
    without st.rerun() during execution. Only calls st.rerun() once at the end.
    """
    queue = st.session_state.get("queue")
    session_dir = st.session_state.get("session_dir")
    session_path = Path(session_dir) if session_dir else None

    # Create tabs with placeholders
    tab1, tab2, tab3 = st.tabs(["Output", "Progress", "Generated Code"])

    with tab1:
        # Iteration history placeholder - updated dynamically on session_end events
        history_ph = st.empty()

        # Helper function to render history into placeholder
        def _render_history_section():
            with history_ph.container():
                render_iteration_history()
                if st.session_state.get("iteration_history"):
                    st.divider()
                st.markdown("### Current Iteration")

        # Initial render
        _render_history_section()

        # Metrics placeholders
        cols = st.columns(4)
        cost_ph = cols[0].empty()
        input_ph = cols[1].empty()
        output_ph = cols[2].empty()
        tools_ph = cols[3].empty()

        # Status placeholder
        status_ph = st.empty()

        # Output area
        st.markdown("### Output")
        text_ph = st.empty()
        tool_calls_container = st.empty()
        thinking_container = st.empty()

    with tab2:
        st.markdown("### Task Progress")
        progress_ph = st.empty()

    with tab3:
        st.markdown("### Generated Evaluation Code")
        code_ph = st.empty()

    # Initial render of all placeholders
    _update_output_placeholders(
        cost_ph, input_ph, output_ph, tools_ph,
        status_ph, text_ph, tool_calls_container, thinking_container
    )
    _update_progress_placeholder(progress_ph, session_path)
    _update_code_placeholder(code_ph, session_path)

    # Polling loop - NO st.rerun() here
    last_file_check = time.time()
    while st.session_state.get("running"):
        events = poll_events(queue, timeout=0.05)

        for event in events:
            process_stream_event(event)

            # Check for completion or session end
            if event.event_type == "status":
                status = event.data.get("status")
                if status in ("stopped", "completed", "max_iterations"):
                    st.session_state["running"] = False
                elif status == "session_end":
                    # Update history when an iteration completes
                    _render_history_section()

        # Update Output placeholders on every iteration (fast)
        _update_output_placeholders(
            cost_ph, input_ph, output_ph, tools_ph,
            status_ph, text_ph, tool_calls_container, thinking_container
        )

        # Check for immediate refresh triggers (file writes detected)
        current_time = time.time()
        if st.session_state.get("task_refresh_ready"):
            _update_progress_placeholder(progress_ph, session_path)
            st.session_state["task_refresh_ready"] = False
            last_file_check = current_time

        if st.session_state.get("code_refresh_ready"):
            _update_code_placeholder(code_ph, session_path)
            st.session_state["code_refresh_ready"] = False
            last_file_check = current_time

        # Fallback: Update Progress and Code tabs every 2 seconds (file reads are slower)
        if current_time - last_file_check > 2.0:
            _update_progress_placeholder(progress_ph, session_path)
            _update_code_placeholder(code_ph, session_path)
            last_file_check = current_time

        time.sleep(0.02)  # Small sleep to prevent CPU spin

    # Final update before rerun
    _update_output_placeholders(
        cost_ph, input_ph, output_ph, tools_ph,
        status_ph, text_ph, tool_calls_container, thinking_container
    )
    _update_progress_placeholder(progress_ph, session_path)
    _update_code_placeholder(code_ph, session_path)

    # Final rerun for clean completed state
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
    """Render autonomous mode UI with tabs.

    Uses placeholder-based streaming when running to avoid st.rerun() flickering.
    Falls back to static rendering when not running.
    """
    # If running, use streaming function with placeholders (no flicker)
    if st.session_state.get("running"):
        _stream_autonomous_response()
        return  # Streaming function handles everything, including final rerun

    # Static rendering when NOT running
    tab1, tab2, tab3 = st.tabs(["Output", "Progress", "Generated Code"])

    with tab1:
        # Iteration history (completed iterations above current output)
        render_iteration_history()

        # Divider between history and current/final output
        if st.session_state.get("iteration_history"):
            st.divider()
            st.markdown("### Final Output")

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
