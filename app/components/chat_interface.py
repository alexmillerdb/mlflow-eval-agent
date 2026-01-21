"""Interactive mode chat interface.

Provides:
- Text input for user prompts
- Message history display
- Send button
- Session continuity via session_id
"""

from typing import Optional

import streamlit as st


def render_chat_interface() -> Optional[str]:
    """Render chat interface for interactive mode.

    Returns:
        User prompt if submitted, None otherwise.
    """
    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    st.markdown("### Interactive Mode")
    st.caption(
        "Ask questions about the experiment, request specific analyses, "
        "or guide the evaluation process."
    )

    # Display message history
    _render_message_history()

    # Input area
    prompt = _render_input_area()

    return prompt


def render_message_history() -> None:
    """Render chat message history (public interface).

    Use this when you need to control where the input area appears
    relative to other content (e.g., streaming responses).
    """
    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    st.markdown("### Interactive Mode")
    st.caption(
        "Ask questions about the experiment, request specific analyses, "
        "or guide the evaluation process."
    )

    _render_message_history()


def render_chat_input() -> Optional[str]:
    """Render just the chat input area (public interface).

    Returns:
        User prompt if submitted, None otherwise.
    """
    return _render_input_area()


def _render_message_history() -> None:
    """Render chat message history."""
    messages = st.session_state.get("messages", [])

    if not messages:
        st.info("No messages yet. Send a prompt to start the conversation.")
        return

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        else:
            with st.chat_message("assistant"):
                st.markdown(content)


def _render_input_area() -> Optional[str]:
    """Render input area with text box and send button.

    Returns:
        User prompt if submitted, None otherwise.
    """
    # Check if agent is running
    is_running = st.session_state.get("running", False)

    # Use form to handle submit
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])

        with col1:
            prompt = st.text_area(
                "Your message",
                placeholder="e.g., Analyze the error patterns in my traces",
                disabled=is_running,
                height=100,
                label_visibility="collapsed",
            )

        with col2:
            submitted = st.form_submit_button(
                "Send",
                disabled=is_running or not st.session_state.get("experiment_id"),
                use_container_width=True,
            )

    if submitted and prompt:
        # Add to message history
        st.session_state["messages"].append({
            "role": "user",
            "content": prompt,
        })
        return prompt

    return None


def add_assistant_message(content: str) -> None:
    """Add assistant response to message history.

    Args:
        content: Response content.
    """
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    st.session_state["messages"].append({
        "role": "assistant",
        "content": content,
    })


def clear_messages() -> None:
    """Clear message history."""
    st.session_state["messages"] = []


def render_example_prompts() -> Optional[str]:
    """Render example prompt buttons.

    Returns:
        Selected example prompt, or None.
    """
    st.markdown("#### Example Prompts")

    examples = [
        "Analyze the error patterns in my traces and suggest scorers",
        "Create a safety scorer that checks for PII in responses",
        "Build an evaluation dataset from the 10 most recent traces",
        "What are the main bottlenecks in my agent's performance?",
    ]

    selected = None
    cols = st.columns(2)

    for i, example in enumerate(examples):
        col = cols[i % 2]
        with col:
            if st.button(
                example[:50] + "..." if len(example) > 50 else example,
                key=f"example_{i}",
                use_container_width=True,
                disabled=st.session_state.get("running", False),
            ):
                selected = example

    return selected


def render_session_info() -> None:
    """Render session continuity information."""
    agent_session = st.session_state.get("agent_session_id")

    if agent_session:
        st.caption(f"Session ID: `{agent_session[:20]}...`")
        st.caption("Conversation context is preserved.")
    else:
        st.caption("New conversation (no prior context)")
