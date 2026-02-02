"""Streamlit chat interface for MLflow Evaluation Agent.

A minimal chat interface that:
- Streams agent responses using st.write_stream()
- Maintains conversation history in session state
- Provides sidebar for experiment configuration
- Shows tool usage indicators during execution
"""

import os
import sys

# Add parent directory to path for streamlit run compatibility
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st

from src.app.streaming import async_to_sync_generator


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


def setup_sidebar():
    """Configure sidebar with experiment settings and session controls."""
    with st.sidebar:
        st.header("Configuration")

        # Experiment ID input
        experiment_id = st.text_input(
            "Experiment ID",
            value=os.getenv("MLFLOW_EXPERIMENT_ID", ""),
            help="MLflow experiment ID to analyze",
        )

        # Store in environment for agent to pick up
        if experiment_id:
            os.environ["MLFLOW_EXPERIMENT_ID"] = experiment_id

        st.divider()

        # Session controls
        st.subheader("Session")

        if st.session_state.session_id:
            st.caption(f"ID: {st.session_state.session_id[:16]}...")

        if st.button("New Session", type="secondary"):
            st.session_state.messages = []
            st.session_state.session_id = None
            st.rerun()


def display_chat_history():
    """Display existing chat messages."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def stream_agent_response(prompt: str):
    """Stream agent response and yield text chunks for st.write_stream().

    Args:
        prompt: User's query

    Yields:
        Text chunks as the agent generates them
    """
    # Import here to avoid circular imports and allow lazy loading
    from src.agent.agent import MLflowAgent, setup_mlflow, AgentResult
    from src.core.config import Config

    # Initialize MLflow once
    if not st.session_state.initialized:
        setup_mlflow()
        st.session_state.initialized = True

    # Create agent with current config
    config = Config.from_env(validate=False)
    agent = MLflowAgent(config)

    # Track last text position for delta extraction
    last_text_len = 0

    # Capture session_id before creating closure (can't access st.session_state from worker thread)
    current_session_id = st.session_state.session_id

    # Create async generator factory
    def create_query():
        return agent.query(prompt, session_id=current_session_id)

    # Use bridge to convert async to sync
    for result in async_to_sync_generator(create_query):
        if result.event_type == "text":
            # Extract only the new text (delta)
            new_text = result.response[last_text_len:]
            last_text_len = len(result.response)
            if new_text:
                yield new_text

        elif result.event_type == "tool_use":
            # Show tool indicator
            yield f"\n\n:gear: *Using tool: {result.tool_name}*\n\n"

        elif result.event_type == "result":
            # Store session ID for continuity
            if result.session_id:
                st.session_state.session_id = result.session_id


def handle_user_input():
    """Handle new user input from chat."""
    if prompt := st.chat_input("Ask about your MLflow traces..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Stream assistant response
        with st.chat_message("assistant"):
            response = st.write_stream(stream_agent_response(prompt))

        # Store assistant response in history
        st.session_state.messages.append({"role": "assistant", "content": response})


def main():
    """Main application entry point."""
    st.title("MLflow Eval Agent")

    initialize_session_state()
    setup_sidebar()
    display_chat_history()
    handle_user_input()


if __name__ == "__main__":
    main()
