"""Tests for the async-to-sync streaming bridge and streaming response pipeline.

Tests are organized in three tiers:
1. async_to_sync_generator unit tests (pure bridge logic)
2. Lazy import verification (catches the exact class of bug where main.py
   references a function that doesn't exist in the imported module)
3. Streaming pipeline integration (AgentResult → async bridge → ChatRenderer,
   consumed lazily like the real app does)
"""

import asyncio
import inspect
from unittest.mock import MagicMock, patch

import pytest

from src.app.streaming import async_to_sync_generator


# =============================================================================
# 1. async_to_sync_generator unit tests
# =============================================================================


def test_async_to_sync_basic():
    """Test basic async to sync conversion."""
    async def async_gen():
        for i in range(5):
            yield i
            await asyncio.sleep(0.01)

    results = list(async_to_sync_generator(async_gen))
    assert results == [0, 1, 2, 3, 4]


def test_async_to_sync_empty():
    """Test empty async generator."""
    async def async_gen():
        return
        yield  # Make it a generator

    results = list(async_to_sync_generator(async_gen))
    assert results == []


def test_async_to_sync_single_item():
    """Test single item generator."""
    async def async_gen():
        yield "only"

    results = list(async_to_sync_generator(async_gen))
    assert results == ["only"]


def test_async_to_sync_error_propagation():
    """Test that errors are propagated from async to sync."""
    async def async_gen():
        yield 1
        yield 2
        raise ValueError("Test error")

    with pytest.raises(ValueError, match="Test error"):
        list(async_to_sync_generator(async_gen))


def test_async_to_sync_error_at_start():
    """Test error at the start of iteration."""
    async def async_gen():
        raise RuntimeError("Immediate error")
        yield  # Never reached

    with pytest.raises(RuntimeError, match="Immediate error"):
        list(async_to_sync_generator(async_gen))


def test_async_to_sync_mixed_types():
    """Test generator yielding different types."""
    async def async_gen():
        yield "text"
        yield 42
        yield {"key": "value"}
        yield [1, 2, 3]

    results = list(async_to_sync_generator(async_gen))
    assert results == ["text", 42, {"key": "value"}, [1, 2, 3]]


def test_async_to_sync_large_stream():
    """Test streaming a larger number of items."""
    count = 100

    async def async_gen():
        for i in range(count):
            yield i

    results = list(async_to_sync_generator(async_gen))
    assert len(results) == count
    assert results == list(range(count))


# =============================================================================
# 2. Lazy import verification
# =============================================================================
# main.py uses lazy imports inside function bodies. These resolve at request
# time, not at module load. A missing function (like the stream_autonomous
# ImportError) passes regular unit tests but crashes the running app.


class TestMainLazyImports:
    """Verify every lazy import inside main.py resolves."""

    def test_stream_agent_response_imports(self):
        """Imports used by stream_agent_response_full()."""
        from src.agent.agent import MLflowAgent, setup_mlflow
        from src.core.config import Config
        from src.app.auth import get_obo_token

        assert callable(setup_mlflow)
        assert callable(get_obo_token)

    def test_autonomous_streaming_imports(self):
        """Imports used by _run_autonomous_streaming()."""
        from src.agent.autonomous import stream_autonomous

        assert inspect.isasyncgenfunction(stream_autonomous)

    def test_session_dir_imports(self):
        """Imports used by _try_detect_session_dir()."""
        from src.agent.mlflow_ops import get_session_dir

        assert callable(get_session_dir)

    def test_top_level_imports(self):
        """Top-level imports in main.py."""
        from src.app.streaming import async_to_sync_generator
        from src.app.components import (
            render_sidebar,
            render_task_progress,
            render_task_list,
            ChatRenderer,
            render_file_viewer,
        )

        assert callable(async_to_sync_generator)
        assert callable(render_sidebar)
        assert callable(render_task_progress)
        assert callable(render_task_list)
        assert callable(render_file_viewer)


# =============================================================================
# 3. Streaming pipeline integration
# =============================================================================
# Tests the full chain: async AgentResult generator → async_to_sync_generator
# → ChatRenderer.render() consuming lazily (not collected into a list first).


def _make_result(event_type="text", response="", **kwargs):
    """Create an AgentResult for testing."""
    from src.agent.agent import AgentResult
    return AgentResult(success=True, response=response, event_type=event_type, **kwargs)


def _make_mock_st():
    """Create a mock st module with placeholder and expander support."""
    mock_st = MagicMock()
    mock_st.empty.return_value = MagicMock()
    mock_expander = MagicMock()
    mock_expander.__enter__ = MagicMock(return_value=mock_expander)
    mock_expander.__exit__ = MagicMock(return_value=False)
    mock_st.expander.return_value = mock_expander
    return mock_st


class TestStreamingResponsePipeline:
    """Test AgentResult objects flowing through the full pipeline lazily."""

    @patch("src.app.components.chat_renderer.st")
    def test_text_only_response(self, mock_st):
        """Text-only stream consumed lazily produces a single text part."""
        mock_st.empty.return_value = MagicMock()
        from src.app.components.chat_renderer import ChatRenderer

        async def fake_agent():
            yield _make_result("text", "Hello ")
            yield _make_result("text", "Hello world")
            yield _make_result("result", "Hello world", cost_usd=0.001, session_id="s1")

        # Consume lazily — ChatRenderer pulls from the sync generator directly,
        # exactly like handle_user_input() does in main.py
        renderer = ChatRenderer()
        parts = renderer.render(async_to_sync_generator(fake_agent))

        assert len(parts) == 1
        assert parts[0] == {"type": "text", "content": "Hello world"}

    @patch("src.app.components.chat_renderer.st")
    def test_text_with_tool_call(self, mock_st):
        """Stream with tool use produces text + tool + text parts."""
        mock_st.empty.return_value = MagicMock()
        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=mock_expander)
        mock_expander.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value = mock_expander
        from src.app.components.chat_renderer import ChatRenderer

        async def fake_agent():
            yield _make_result("text", "Let me search")
            yield _make_result("tool_use", "Let me search",
                               tool_name="mlflow_query",
                               tool_input={"operation": "search", "experiment_id": "123"})
            yield _make_result("tool_result", "Let me search",
                               tool_result="Found 5 traces")
            yield _make_result("text", "Let me search\n\nI found 5 traces.")
            yield _make_result("result", "Let me search\n\nI found 5 traces.")

        renderer = ChatRenderer()
        parts = renderer.render(async_to_sync_generator(fake_agent))

        assert len(parts) == 3
        assert parts[0] == {"type": "text", "content": "Let me search"}
        assert parts[1]["type"] == "tool"
        assert parts[1]["name"] == "mlflow_query"
        assert parts[1]["input"] == {"operation": "search", "experiment_id": "123"}
        assert parts[1]["result"] == "Found 5 traces"
        assert parts[2] == {"type": "text", "content": "\n\nI found 5 traces."}

    @patch("src.app.components.chat_renderer.st")
    def test_thinking_block(self, mock_st):
        """Thinking events produce thinking parts."""
        mock_st.empty.return_value = MagicMock()
        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=mock_expander)
        mock_expander.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value = mock_expander
        from src.app.components.chat_renderer import ChatRenderer

        async def fake_agent():
            yield _make_result("thinking", "", thinking_content="Analyzing the query...")
            yield _make_result("text", "Here are the results.")
            yield _make_result("result", "Here are the results.")

        renderer = ChatRenderer()
        parts = renderer.render(async_to_sync_generator(fake_agent))

        assert len(parts) == 2
        assert parts[0] == {"type": "thinking", "content": "Analyzing the query..."}
        assert parts[1] == {"type": "text", "content": "Here are the results."}

    @patch("src.app.components.chat_renderer.st")
    def test_multiple_tool_calls(self, mock_st):
        """Multiple sequential tool calls each get their own part."""
        mock_st.empty.return_value = MagicMock()
        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=mock_expander)
        mock_expander.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value = mock_expander
        from src.app.components.chat_renderer import ChatRenderer

        async def fake_agent():
            yield _make_result("tool_use", "", tool_name="mlflow_query",
                               tool_input={"operation": "search"})
            yield _make_result("tool_result", "", tool_result="3 traces")
            yield _make_result("tool_use", "", tool_name="mlflow_query",
                               tool_input={"operation": "get", "trace_id": "tr-1"})
            yield _make_result("tool_result", "", tool_result="trace details")
            yield _make_result("text", "Done analyzing.")
            yield _make_result("result", "Done analyzing.")

        renderer = ChatRenderer()
        parts = renderer.render(async_to_sync_generator(fake_agent))

        tool_parts = [p for p in parts if p["type"] == "tool"]
        assert len(tool_parts) == 2
        assert tool_parts[0]["name"] == "mlflow_query"
        assert tool_parts[0]["result"] == "3 traces"
        assert tool_parts[1]["name"] == "mlflow_query"
        assert tool_parts[1]["result"] == "trace details"

    @patch("src.app.components.chat_renderer.st")
    def test_cost_displayed_on_result(self, mock_st):
        """Final result event with cost_usd triggers cost caption."""
        mock_st.empty.return_value = MagicMock()
        from src.app.components.chat_renderer import ChatRenderer

        async def fake_agent():
            yield _make_result("text", "Answer")
            yield _make_result("result", "Answer", cost_usd=0.0042)

        renderer = ChatRenderer()
        parts = renderer.render(async_to_sync_generator(fake_agent))

        mock_st.caption.assert_called_once_with("Cost: $0.0042")

    @patch("src.app.components.chat_renderer.st")
    def test_empty_stream_produces_no_parts(self, mock_st):
        """An empty agent stream (just result) produces no text parts."""
        from src.app.components.chat_renderer import ChatRenderer

        async def fake_agent():
            yield _make_result("result", "")

        renderer = ChatRenderer()
        parts = renderer.render(async_to_sync_generator(fake_agent))

        assert parts == []

    def test_error_during_stream_propagates(self):
        """Errors mid-stream propagate through the bridge."""
        async def fake_agent():
            yield _make_result("text", "Starting...")
            raise ConnectionError("Agent backend unavailable")

        with pytest.raises(ConnectionError, match="Agent backend unavailable"):
            # Consume lazily — error surfaces when generator is iterated
            for _ in async_to_sync_generator(fake_agent):
                pass


# =============================================================================
# 4. Unified message history display
# =============================================================================


class TestUnifiedMessageHistory:
    """Test display_chat_history() handles all role types correctly."""

    @patch("src.app.main.st")
    @patch("src.app.main.ChatRenderer")
    def test_display_user_message(self, mock_renderer_cls, mock_st):
        """User message renders with st.chat_message('user')."""
        from src.app.main import display_chat_history

        mock_st.session_state.messages = [{"role": "user", "content": "Hello"}]
        mock_ctx = MagicMock()
        mock_st.chat_message.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_st.chat_message.return_value.__exit__ = MagicMock(return_value=False)

        display_chat_history()

        mock_st.chat_message.assert_called_with("user")
        mock_st.markdown.assert_called_with("Hello")

    @patch("src.app.main.st")
    @patch("src.app.main.ChatRenderer")
    def test_display_assistant_message(self, mock_renderer_cls, mock_st):
        """Assistant message with parts renders via ChatRenderer."""
        from src.app.main import display_chat_history

        parts = [{"type": "text", "content": "Response"}]
        mock_st.session_state.messages = [{"role": "assistant", "parts": parts}]
        mock_ctx = MagicMock()
        mock_st.chat_message.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_st.chat_message.return_value.__exit__ = MagicMock(return_value=False)

        display_chat_history()

        mock_st.chat_message.assert_called_with("assistant")
        mock_renderer_cls.render_history_message.assert_called_once_with(parts)

    @patch("src.app.main.st")
    @patch("src.app.main.ChatRenderer")
    def test_display_autonomous_message(self, mock_renderer_cls, mock_st):
        """Autonomous message renders with gear avatar and iteration header."""
        from src.app.main import display_chat_history

        parts = [{"type": "text", "content": "Analysis"}]
        mock_st.session_state.messages = [{
            "role": "autonomous",
            "iteration": 1,
            "phase": "initializer",
            "parts": parts,
        }]
        mock_ctx = MagicMock()
        mock_st.chat_message.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_st.chat_message.return_value.__exit__ = MagicMock(return_value=False)

        display_chat_history()

        mock_st.chat_message.assert_called_with("assistant", avatar="\U0001f527")
        mock_st.markdown.assert_called_with("**Session 1** (initializer)")
        mock_renderer_cls.render_history_message.assert_called_once_with(parts)

    @patch("src.app.main.st")
    @patch("src.app.main.ChatRenderer")
    def test_display_auto_status_complete(self, mock_renderer_cls, mock_st):
        """auto_status with status='complete' renders as st.success()."""
        from src.app.main import display_chat_history

        mock_st.session_state.messages = [{
            "role": "auto_status",
            "status": "complete",
            "content": "Autonomous run complete!",
        }]

        display_chat_history()

        mock_st.success.assert_called_once_with("Autonomous run complete!")

    @patch("src.app.main.st")
    @patch("src.app.main.ChatRenderer")
    def test_display_auto_status_error(self, mock_renderer_cls, mock_st):
        """auto_status with status='error' renders as st.error()."""
        from src.app.main import display_chat_history

        mock_st.session_state.messages = [{
            "role": "auto_status",
            "status": "error",
            "content": "Error in session 2: timeout",
        }]

        display_chat_history()

        mock_st.error.assert_called_once_with("Error in session 2: timeout")
