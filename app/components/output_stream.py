"""Streaming output display for agent responses.

Renders:
- Text output with markdown formatting
- Collapsible thinking blocks
- Tool call details
- Cost/token metrics
"""

import json
from typing import Optional

import streamlit as st

from ..services.agent_runner import StreamEvent


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
        tool_call = {
            "tool_name": event.data.get("tool_name"),
            "tool_input": event.data.get("tool_input"),
        }
        st.session_state["tool_calls"].append(tool_call)

    elif event.event_type == "tool_result":
        # Update last tool call with result
        if st.session_state["tool_calls"]:
            st.session_state["tool_calls"][-1]["tool_result"] = event.data.get("tool_result")

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
            st.session_state["running"] = False
        elif status == "completed":
            st.session_state["running"] = False
            st.session_state["completed"] = True
        elif status == "session_start":
            # New session starting
            iteration = event.data.get("iteration", 0)
            phase = event.data.get("phase", "unknown")
            st.session_state["current_iteration"] = iteration
            st.session_state["current_phase"] = phase

    elif event.event_type == "error":
        error = event.data.get("error", "Unknown error")
        st.session_state["last_error"] = error

    elif event.event_type == "session_id":
        # Store session ID for interactive mode
        st.session_state["agent_session_id"] = event.data.get("session_id")


def clear_output() -> None:
    """Clear all output state."""
    st.session_state["accumulated_text"] = ""
    st.session_state["tool_calls"] = []
    st.session_state["thinking_blocks"] = []
    st.session_state["total_cost"] = 0.0
    st.session_state["total_input_tokens"] = 0
    st.session_state["total_output_tokens"] = 0
    st.session_state.pop("last_error", None)
    st.session_state.pop("completed", None)


def render_error_banner() -> None:
    """Render error banner if there's an error."""
    error = st.session_state.get("last_error")
    if error:
        st.error(f"Error: {error}")


def render_completion_banner() -> None:
    """Render completion banner if agent finished."""
    if st.session_state.get("completed"):
        st.success("Evaluation setup complete! Check the Generated Code tab.")
