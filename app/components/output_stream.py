"""Streaming output display for agent responses.

Renders:
- Text output with markdown formatting
- Collapsible thinking blocks
- Tool call details
- Cost/token metrics
- Iteration history for autonomous mode
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import streamlit as st

from ..services.agent_runner import StreamEvent


# Constants for file pattern matching
TASK_FILE_PATTERNS = ["eval_tasks.json"]
CODE_FILE_PATTERNS = ["evaluation/"]


def _detect_file_write(tool_name: str, tool_input: dict) -> tuple[bool, bool]:
    """Detect if a tool call writes to task or code files.

    Args:
        tool_name: Name of the tool being called.
        tool_input: Input parameters for the tool.

    Returns:
        Tuple of (refresh_tasks, refresh_code) booleans.
    """
    if tool_name not in ("Write", "Edit"):
        return False, False

    file_path = tool_input.get("file_path", "") or tool_input.get("path", "")
    refresh_tasks = any(p in file_path for p in TASK_FILE_PATTERNS)
    refresh_code = any(p in file_path for p in CODE_FILE_PATTERNS)

    return refresh_tasks, refresh_code


@dataclass
class IterationRecord:
    """Record of a completed iteration (initializer or worker session).

    Captures all output and metrics from a single agent session for display
    in iteration history.
    """

    iteration: int
    phase: str  # "initializer" or "worker"
    started_at: datetime
    ended_at: Optional[datetime] = None
    text: str = ""
    tool_calls: list = field(default_factory=list)
    thinking_blocks: list = field(default_factory=list)
    cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None


def render_output_stream() -> None:
    """Render the streaming output area.

    Reads from st.session_state and displays accumulated output.
    Uses st.empty() placeholders for smoother real-time streaming.
    """
    # Initialize state
    if "accumulated_text" not in st.session_state:
        st.session_state["accumulated_text"] = ""
    if "tool_calls" not in st.session_state:
        st.session_state["tool_calls"] = []
    if "thinking_blocks" not in st.session_state:
        st.session_state["thinking_blocks"] = []
    if "total_cost" not in st.session_state:
        st.session_state["total_cost"] = 0.0
    if "total_input_tokens" not in st.session_state:
        st.session_state["total_input_tokens"] = 0
    if "total_output_tokens" not in st.session_state:
        st.session_state["total_output_tokens"] = 0
    # Iteration history for autonomous mode
    if "iteration_history" not in st.session_state:
        st.session_state["iteration_history"] = []
    if "current_iteration_record" not in st.session_state:
        st.session_state["current_iteration_record"] = None
    # File change detection flags for immediate refresh
    if "pending_task_refresh" not in st.session_state:
        st.session_state["pending_task_refresh"] = False
    if "pending_code_refresh" not in st.session_state:
        st.session_state["pending_code_refresh"] = False
    if "task_refresh_ready" not in st.session_state:
        st.session_state["task_refresh_ready"] = False
    if "code_refresh_ready" not in st.session_state:
        st.session_state["code_refresh_ready"] = False

    # Metrics row with placeholders for real-time updates
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        cost_placeholder = st.empty()
        cost_placeholder.metric("Cost", f"${st.session_state['total_cost']:.4f}")
    with col2:
        input_placeholder = st.empty()
        input_placeholder.metric("Input Tokens", f"{st.session_state['total_input_tokens']:,}")
    with col3:
        output_placeholder = st.empty()
        output_placeholder.metric("Output Tokens", f"{st.session_state['total_output_tokens']:,}")
    with col4:
        tools_placeholder = st.empty()
        tools_placeholder.metric("Tool Calls", len(st.session_state["tool_calls"]))

    # Status indicator with placeholder
    status_placeholder = st.empty()
    if st.session_state.get("running"):
        status_placeholder.markdown(":hourglass_flowing_sand: **Agent is running...**")

    # Main output area
    st.markdown("### Output")

    # Use st.empty() placeholder for streaming text output
    output_placeholder = st.empty()
    text = st.session_state["accumulated_text"]
    if text:
        output_placeholder.markdown(text)
    else:
        output_placeholder.info("No output yet. Start the agent to see results.")

    # Tool calls (collapsible)
    tool_calls = st.session_state["tool_calls"]
    if tool_calls:
        with st.expander(f"Tool Calls ({len(tool_calls)})", expanded=False):
            for i, tc in enumerate(tool_calls):
                st.markdown(f"**{i+1}. {tc.get('tool_name', 'Unknown')}**")
                if tc.get("tool_input"):
                    st.code(json.dumps(tc["tool_input"], indent=2), language="json")
                if tc.get("tool_result"):
                    st.text(tc["tool_result"][:500])
                st.divider()

    # Thinking blocks (collapsible)
    thinking_blocks = st.session_state["thinking_blocks"]
    if thinking_blocks:
        with st.expander(f"Thinking ({len(thinking_blocks)})", expanded=False):
            for i, tb in enumerate(thinking_blocks):
                st.markdown(f"**Thought {i+1}:**")
                st.text(tb[:1000])
                st.divider()


def process_stream_event(event: StreamEvent) -> None:
    """Process a stream event and update session state.

    Args:
        event: StreamEvent from agent runner.
    """
    if event.event_type == "text":
        # Update accumulated text
        text = event.data.get("response", "")
        st.session_state["accumulated_text"] = text

    elif event.event_type == "thinking":
        # Add thinking block
        thinking = event.data.get("thinking", "")
        if thinking:
            st.session_state["thinking_blocks"].append(thinking)

    elif event.event_type == "tool_use":
        # Add tool call
        tool_name = event.data.get("tool_name")
        tool_input = event.data.get("tool_input", {})
        tool_call = {
            "tool_name": tool_name,
            "tool_input": tool_input,
        }
        st.session_state["tool_calls"].append(tool_call)

        # Detect file writes and set pending refresh flags
        refresh_tasks, refresh_code = _detect_file_write(tool_name, tool_input)
        if refresh_tasks:
            st.session_state["pending_task_refresh"] = True
        if refresh_code:
            st.session_state["pending_code_refresh"] = True

    elif event.event_type == "tool_result":
        # Update last tool call with result
        if st.session_state["tool_calls"]:
            st.session_state["tool_calls"][-1]["tool_result"] = event.data.get("tool_result")

        # Convert pending refresh flags to ready flags (file write completed)
        if st.session_state.get("pending_task_refresh"):
            st.session_state["task_refresh_ready"] = True
            st.session_state["pending_task_refresh"] = False
        if st.session_state.get("pending_code_refresh"):
            st.session_state["code_refresh_ready"] = True
            st.session_state["pending_code_refresh"] = False

    elif event.event_type == "result":
        # Final result with cost/usage
        if event.data.get("cost_usd"):
            st.session_state["total_cost"] += event.data["cost_usd"]

        usage = event.data.get("usage_data", {})
        if usage:
            st.session_state["total_input_tokens"] += usage.get("input_tokens", 0)
            st.session_state["total_output_tokens"] += usage.get("output_tokens", 0)

        # Store session ID for interactive mode continuity
        if event.data.get("session_id"):
            st.session_state["agent_session_id"] = event.data["session_id"]

    elif event.event_type == "status":
        status = event.data.get("status")
        if status == "started":
            st.session_state["running"] = True
        elif status == "stopped":
            # Archive current iteration before stopping
            _archive_current_iteration()
            st.session_state["running"] = False
        elif status == "completed":
            # Archive current iteration before completing
            _archive_current_iteration()
            st.session_state["running"] = False
            st.session_state["completed"] = True
        elif status == "max_iterations":
            # Archive current iteration when max iterations reached
            _archive_current_iteration()
            st.session_state["running"] = False
        elif status == "session_start":
            # New session starting - archive previous and start fresh
            _archive_current_iteration()

            iteration = event.data.get("iteration", 0)
            phase = event.data.get("phase", "unknown")
            st.session_state["current_iteration"] = iteration
            st.session_state["current_phase"] = phase

            # Create new iteration record
            st.session_state["current_iteration_record"] = IterationRecord(
                iteration=iteration,
                phase=phase,
                started_at=datetime.now(),
            )

            # Clear current output but preserve history
            clear_output(preserve_history=True)
        elif status == "session_end":
            # Explicitly archive iteration (emitted by agent_runner)
            _archive_current_iteration()

    elif event.event_type == "error":
        error = event.data.get("error", "Unknown error")
        st.session_state["last_error"] = error

    elif event.event_type == "session_id":
        # Store session ID for interactive mode
        st.session_state["agent_session_id"] = event.data.get("session_id")


def _archive_current_iteration() -> None:
    """Archive current iteration state into iteration_history.

    Captures all current output state into an IterationRecord and appends
    to iteration_history. Called when an iteration ends (session_end event,
    stopped, completed, or max_iterations).
    """
    record = st.session_state.get("current_iteration_record")
    if record is None:
        return

    # Update record with final state
    record.ended_at = datetime.now()
    record.text = st.session_state.get("accumulated_text", "")
    record.tool_calls = list(st.session_state.get("tool_calls", []))
    record.thinking_blocks = list(st.session_state.get("thinking_blocks", []))
    record.cost = st.session_state.get("total_cost", 0.0)
    record.input_tokens = st.session_state.get("total_input_tokens", 0)
    record.output_tokens = st.session_state.get("total_output_tokens", 0)
    record.error = st.session_state.get("last_error")

    # Append to history
    if "iteration_history" not in st.session_state:
        st.session_state["iteration_history"] = []
    st.session_state["iteration_history"].append(record)

    # Clear current record
    st.session_state["current_iteration_record"] = None


def clear_output(preserve_history: bool = False) -> None:
    """Clear output state for current iteration.

    Args:
        preserve_history: If True, keeps iteration_history intact.
            Set False when starting a completely new session.
    """
    st.session_state["accumulated_text"] = ""
    st.session_state["tool_calls"] = []
    st.session_state["thinking_blocks"] = []
    st.session_state["total_cost"] = 0.0
    st.session_state["total_input_tokens"] = 0
    st.session_state["total_output_tokens"] = 0
    st.session_state.pop("last_error", None)
    st.session_state.pop("completed", None)

    # Reset file change detection flags
    st.session_state["pending_task_refresh"] = False
    st.session_state["pending_code_refresh"] = False
    st.session_state["task_refresh_ready"] = False
    st.session_state["code_refresh_ready"] = False

    if not preserve_history:
        st.session_state["iteration_history"] = []
        st.session_state["current_iteration_record"] = None


def render_error_banner() -> None:
    """Render error banner if there's an error."""
    error = st.session_state.get("last_error")
    if error:
        st.error(f"Error: {error}")


def render_completion_banner() -> None:
    """Render completion banner if agent finished."""
    if st.session_state.get("completed"):
        st.success("Evaluation setup complete! Check the Generated Code tab.")


def render_iteration_history() -> None:
    """Render completed iterations as collapsible expanders.

    Displays each completed iteration (initializer/worker) with:
    - Phase label and status icon
    - Metrics (cost, tokens, duration)
    - Output text
    - Tool calls (nested expander)
    - Thinking blocks (nested expander)
    """
    history = st.session_state.get("iteration_history", [])
    if not history:
        return

    st.markdown("### Completed Iterations")

    for record in history:
        # Build expander label with status and metrics
        if record.phase == "initializer":
            phase_label = "Initializer"
            icon = ":white_check_mark:" if not record.error else ":x:"
        else:
            phase_label = f"Worker {record.iteration - 1}"  # Worker iterations start at 2
            icon = ":white_check_mark:" if not record.error else ":x:"

        # Calculate duration
        duration_str = ""
        if record.started_at and record.ended_at:
            duration = (record.ended_at - record.started_at).total_seconds()
            if duration < 60:
                duration_str = f"{duration:.0f}s"
            else:
                duration_str = f"{duration / 60:.1f}m"

        # Metrics summary for label
        metrics_parts = []
        if record.cost > 0:
            metrics_parts.append(f"${record.cost:.3f}")
        if record.input_tokens > 0 or record.output_tokens > 0:
            metrics_parts.append(f"{record.input_tokens + record.output_tokens:,} tokens")
        if duration_str:
            metrics_parts.append(duration_str)

        metrics_summary = " | ".join(metrics_parts) if metrics_parts else ""

        expander_label = f"{icon} {phase_label}"
        if metrics_summary:
            expander_label += f"  ({metrics_summary})"

        with st.expander(expander_label, expanded=False):
            # Show error if present
            if record.error:
                st.error(f"Error: {record.error}")

            # Output text
            if record.text:
                st.markdown(record.text)
            else:
                st.info("No output captured.")

            # Tool calls - inline display (no nested expander)
            if record.tool_calls:
                st.markdown(f"**Tool Calls ({len(record.tool_calls)})**")
                for i, tc in enumerate(record.tool_calls):
                    st.markdown(f"_{i+1}. {tc.get('tool_name', 'Unknown')}_")
                    if tc.get("tool_input"):
                        st.code(json.dumps(tc["tool_input"], indent=2), language="json")
                    if tc.get("tool_result"):
                        st.text(tc["tool_result"][:500])

            # Thinking blocks - inline display (no nested expander)
            if record.thinking_blocks:
                st.markdown(f"**Thinking ({len(record.thinking_blocks)})**")
                for i, tb in enumerate(record.thinking_blocks):
                    st.caption(f"Thought {i+1}:")
                    st.text(tb[:500])
