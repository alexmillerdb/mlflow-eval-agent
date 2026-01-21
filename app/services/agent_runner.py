"""Async agent execution wrapper for Streamlit.

Bridges the async MLflowAgent to synchronous Streamlit via a queue-based pattern:

1. Background thread runs async agent loop with new event loop
2. StreamEvent objects are pushed to a queue
3. Streamlit main thread polls queue and updates UI

This pattern allows real-time streaming updates in Streamlit's sync model.
"""

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue, Empty
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class StreamEvent:
    """Event passed from agent thread to Streamlit.

    Attributes:
        event_type: Type of event (text, thinking, tool_use, tool_result, result, error, status)
        data: Event-specific data dictionary
        timestamp: Event timestamp (auto-populated)
    """
    event_type: str  # text | thinking | tool_use | tool_result | result | error | status
    data: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class AgentRunner:
    """Manages async agent execution in a background thread.

    Usage:
        runner = AgentRunner(queue, experiment_id, session_dir, volume_path)
        runner.start_autonomous(max_iterations=10)

        # In Streamlit main thread:
        while runner.is_running:
            try:
                event = queue.get_nowait()
                handle_event(event)
            except Empty:
                time.sleep(0.1)
    """

    def __init__(
        self,
        queue: Queue,
        experiment_id: str,
        session_dir: Path,
        volume_path: str,
        model: str = "databricks-claude-opus-4-5",
    ):
        """Initialize the agent runner.

        Args:
            queue: Queue to push StreamEvents to.
            experiment_id: MLflow experiment ID to analyze.
            session_dir: Session directory for state files.
            volume_path: UC Volume path for storage.
            model: Model to use (default: databricks-claude-opus-4-5).
        """
        self.queue = queue
        self.experiment_id = experiment_id
        self.session_dir = session_dir
        self.volume_path = volume_path
        self.model = model

        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._running = False
        self._error: Optional[str] = None

    @property
    def is_running(self) -> bool:
        """Check if agent is currently running."""
        return self._running and self._thread is not None and self._thread.is_alive()

    @property
    def error(self) -> Optional[str]:
        """Get error message if agent failed."""
        return self._error

    def start_autonomous(self, max_iterations: int = 10) -> None:
        """Start autonomous evaluation mode in background thread.

        Args:
            max_iterations: Maximum iterations before stopping.
        """
        if self.is_running:
            logger.warning("Agent already running, ignoring start request")
            return

        self._stop_flag.clear()
        self._running = True
        self._error = None

        self._thread = threading.Thread(
            target=self._run_autonomous_loop,
            args=(max_iterations,),
            daemon=True,
        )
        self._thread.start()

        self._push_event("status", {"status": "started", "mode": "autonomous"})

    def start_interactive(self, prompt: str, session_id: Optional[str] = None) -> None:
        """Start interactive query in background thread.

        Args:
            prompt: User query to send to agent.
            session_id: Optional session ID for conversation continuity.
        """
        if self.is_running:
            logger.warning("Agent already running, ignoring start request")
            return

        self._stop_flag.clear()
        self._running = True
        self._error = None

        self._thread = threading.Thread(
            target=self._run_interactive_query,
            args=(prompt, session_id),
            daemon=True,
        )
        self._thread.start()

        self._push_event("status", {"status": "started", "mode": "interactive"})

    def stop(self) -> None:
        """Request agent to stop (graceful shutdown)."""
        self._stop_flag.set()
        self._push_event("status", {"status": "stopping"})

    def _push_event(self, event_type: str, data: dict) -> None:
        """Push event to queue for Streamlit to consume."""
        self.queue.put(StreamEvent(event_type=event_type, data=data))

    def _setup_environment(self) -> None:
        """Set up environment variables for agent execution."""
        os.environ["MLFLOW_EXPERIMENT_ID"] = self.experiment_id
        os.environ["MLFLOW_AGENT_VOLUME_PATH"] = self.volume_path
        os.environ["MODEL"] = self.model

    def _run_autonomous_loop(self, max_iterations: int) -> None:
        """Run autonomous evaluation loop (runs in background thread)."""
        try:
            self._setup_environment()

            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                loop.run_until_complete(
                    self._async_autonomous_loop(max_iterations)
                )
            finally:
                loop.close()

        except Exception as e:
            logger.exception(f"Agent error: {e}")
            self._error = str(e)
            self._push_event("error", {"error": str(e)})
        finally:
            self._running = False
            self._push_event("status", {"status": "stopped"})

    def _run_interactive_query(self, prompt: str, session_id: Optional[str]) -> None:
        """Run interactive query (runs in background thread)."""
        try:
            self._setup_environment()

            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                loop.run_until_complete(
                    self._async_interactive_query(prompt, session_id)
                )
            finally:
                loop.close()

        except Exception as e:
            logger.exception(f"Agent error: {e}")
            self._error = str(e)
            self._push_event("error", {"error": str(e)})
        finally:
            self._running = False
            self._push_event("status", {"status": "stopped"})

    async def _async_autonomous_loop(self, max_iterations: int) -> None:
        """Async autonomous evaluation loop."""
        # Import here to avoid circular imports and ensure env vars are set
        from src.agent import MLflowAgent, setup_mlflow, load_prompt
        from src.mlflow_ops import (
            set_session_dir,
            get_tasks_file,
            all_tasks_complete,
        )
        from src.config import Config

        setup_mlflow()
        set_session_dir(self.session_dir)

        config = Config.from_env()
        config.experiment_id = self.experiment_id
        config.model = self.model

        # Check if first run
        is_first_run = not get_tasks_file().exists()

        iteration = 0
        while not self._stop_flag.is_set():
            iteration += 1

            if max_iterations and iteration > max_iterations:
                self._push_event("status", {
                    "status": "max_iterations",
                    "iteration": iteration - 1,
                })
                break

            if not is_first_run and all_tasks_complete():
                self._push_event("status", {"status": "completed"})
                break

            # Create fresh agent per session
            agent = MLflowAgent(config)

            # Choose prompt
            prompt_name = "initializer" if is_first_run else "worker"
            prompt = load_prompt(prompt_name)
            prompt = prompt.replace("{experiment_id}", self.experiment_id)
            prompt = prompt.replace("{session_dir}", str(self.session_dir))

            self._push_event("status", {
                "status": "session_start",
                "iteration": iteration,
                "phase": prompt_name,
            })

            # Run session and stream events
            try:
                async for result in agent.query(prompt):
                    if self._stop_flag.is_set():
                        break

                    self._push_agent_result(result)

            except Exception as e:
                self._push_event("error", {
                    "error": str(e),
                    "iteration": iteration,
                })
                # Continue to next iteration on error

            # Check if initializer created task file
            if is_first_run and get_tasks_file().exists():
                is_first_run = False

            # Brief pause between iterations
            if not self._stop_flag.is_set():
                await asyncio.sleep(2)

    async def _async_interactive_query(
        self,
        prompt: str,
        session_id: Optional[str],
    ) -> None:
        """Async interactive query."""
        # Import here to avoid circular imports
        from src.agent import MLflowAgent, setup_mlflow
        from src.mlflow_ops import set_session_dir
        from src.config import Config

        setup_mlflow()
        set_session_dir(self.session_dir)

        config = Config.from_env()
        config.experiment_id = self.experiment_id
        config.model = self.model

        agent = MLflowAgent(config)

        async for result in agent.query(prompt, session_id=session_id):
            if self._stop_flag.is_set():
                break

            self._push_agent_result(result)

        # Return session_id for continuity
        if agent.session_id:
            self._push_event("session_id", {"session_id": agent.session_id})

    def _push_agent_result(self, result) -> None:
        """Convert AgentResult to StreamEvent and push to queue."""
        data = {
            "success": result.success,
            "response": result.response,
        }

        if result.event_type == "text":
            data["text"] = result.response

        elif result.event_type == "thinking":
            data["thinking"] = result.thinking_content

        elif result.event_type == "tool_use":
            data["tool_name"] = result.tool_name
            data["tool_input"] = result.tool_input

        elif result.event_type == "tool_result":
            data["tool_result"] = result.tool_result

        elif result.event_type == "result":
            data["cost_usd"] = result.cost_usd
            data["session_id"] = result.session_id
            data["usage_data"] = result.usage_data
            data["duration_ms"] = result.duration_ms

        self._push_event(result.event_type, data)


def poll_events(queue: Queue, timeout: float = 0.1) -> list[StreamEvent]:
    """Poll queue for events (non-blocking).

    Args:
        queue: Queue to poll.
        timeout: Timeout in seconds (default 0.1).

    Returns:
        List of events (may be empty).
    """
    events = []
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            event = queue.get_nowait()
            events.append(event)
        except Empty:
            if events:
                break
            time.sleep(0.01)

    return events
