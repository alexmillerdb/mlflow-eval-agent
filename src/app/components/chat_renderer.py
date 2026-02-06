"""Chat renderer for streaming AgentResult objects in Streamlit.

Renders interleaved text, tool usage expanders, and thinking blocks
from the agent's streaming output. Supports both live rendering and
replaying stored message history.
"""

import json
from typing import Iterable, List, Optional

import streamlit as st


class ChatRenderer:
    """Renders streaming AgentResult objects with interleaved text and tool expanders."""

    def render(self, results_iter: Iterable) -> List[dict]:
        """Render a stream of AgentResult objects into Streamlit components.

        Args:
            results_iter: Iterable of AgentResult objects from agent.query().

        Returns:
            List of parts dicts for storing in session state.
        """
        parts: List[dict] = []
        accumulated_text = ""
        last_text_len = 0
        text_placeholder: Optional[st.delta_generator.DeltaGenerator] = None
        tool_container: Optional[st.delta_generator.DeltaGenerator] = None

        for result in results_iter:
            if result.event_type == "text":
                # Create placeholder on first text event or after a tool
                if text_placeholder is None:
                    text_placeholder = st.empty()

                # Delta extraction: only render newly appended text
                new_text = result.response[last_text_len:]
                last_text_len = len(result.response)
                if new_text:
                    accumulated_text += new_text
                    text_placeholder.markdown(accumulated_text)

            elif result.event_type == "tool_use":
                # Finalize current text block
                if accumulated_text:
                    parts.append({"type": "text", "content": accumulated_text})
                    accumulated_text = ""
                    # Do NOT reset last_text_len - result.response is cumulative
                    text_placeholder = None

                # Create tool expander
                tool_name = result.tool_name or "tool"
                tool_input = result.tool_input or {}
                expander = st.expander(f"🔧 {tool_name}", expanded=False)
                with expander:
                    st.code(json.dumps(tool_input, indent=2, default=str), language="json")
                tool_container = expander

                # Start tracking this tool in parts
                parts.append({
                    "type": "tool",
                    "name": tool_name,
                    "input": tool_input,
                    "result": None,
                })

            elif result.event_type == "tool_result":
                # Append result into current tool expander
                tool_result_text = result.tool_result or result.response or ""
                if tool_container is not None:
                    with tool_container:
                        st.code(tool_result_text, language="text")

                # Update the last tool part with result
                for part in reversed(parts):
                    if part["type"] == "tool" and part["result"] is None:
                        part["result"] = tool_result_text
                        break

                tool_container = None
                # Reset so next text gets a new placeholder
                text_placeholder = None

            elif result.event_type == "thinking":
                # Finalize current text block
                if accumulated_text:
                    parts.append({"type": "text", "content": accumulated_text})
                    accumulated_text = ""
                    # Do NOT reset last_text_len - result.response is cumulative
                    text_placeholder = None

                thinking_content = result.thinking_content or result.response or ""
                with st.expander("💭 Thinking", expanded=False):
                    st.markdown(thinking_content)

                parts.append({"type": "thinking", "content": thinking_content})

            elif result.event_type == "result":
                # Finalize everything
                if accumulated_text:
                    parts.append({"type": "text", "content": accumulated_text})
                    accumulated_text = ""

                # Show cost if available
                if result.cost_usd is not None:
                    st.caption(f"Cost: ${result.cost_usd:.4f}")

        # Flush any remaining text
        if accumulated_text:
            parts.append({"type": "text", "content": accumulated_text})

        return parts

    @staticmethod
    def render_history_message(parts: List[dict]) -> None:
        """Re-render a stored message from its parts list.

        Used by display_chat_history to replay previously rendered messages.

        Args:
            parts: List of part dicts as returned by render().
        """
        for part in parts:
            part_type = part.get("type")

            if part_type == "text":
                st.markdown(part["content"])

            elif part_type == "tool":
                tool_name = part.get("name", "tool")
                with st.expander(f"🔧 {tool_name}", expanded=False):
                    st.code(
                        json.dumps(part.get("input", {}), indent=2, default=str),
                        language="json",
                    )
                    if part.get("result"):
                        st.code(part["result"], language="text")

            elif part_type == "thinking":
                with st.expander("💭 Thinking", expanded=False):
                    st.markdown(part["content"])
