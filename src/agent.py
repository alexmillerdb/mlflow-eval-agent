"""Simplified MLflow Evaluation Agent.

Following Anthropic best practices:
- Single agent with good prompts (no coordinator + sub-agents)
- Session isolation with fresh context
- External prompts in markdown files
- Minimal tool set (3 tools vs 11)

Supports two modes:
- Interactive (-i): Free-form queries
- Autonomous (-a): Auto-continue loop with task tracking

~250 lines vs original ~350 lines
"""

import asyncio
import importlib.resources
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional

# =============================================================================
# MLFLOW SETUP - Deferred to setup_mlflow() function
# Must be called AFTER CLI args are processed and env vars are set.
# =============================================================================
import mlflow

_mlflow_initialized = False


def setup_mlflow():
    """Initialize MLflow tracking. Must be called after env vars are set.

    This function is called from cli.py after CLI args are mapped to env vars.
    The mlflow.anthropic import triggers tracking initialization, so we must
    set tracking_uri first to avoid defaulting to local SQLite.
    """
    global _mlflow_initialized
    if _mlflow_initialized:
        return

    # Configure Databricks env vars for subprocess auth (single entry point)
    from .databricks_auth import configure_env
    configure_env()

    import mlflow as _mlflow  # Local import to avoid scoping issues
    import mlflow.anthropic as mlflow_anthropic

    from .config import Config
    config = Config.from_env(validate=False)

    _mlflow.set_tracking_uri(config.tracking_uri)
    if config.agent_experiment_id:
        _mlflow.set_experiment(experiment_id=config.agent_experiment_id)
        logging.info(f"MLflow experiment set to: {config.agent_experiment_id}")

    mlflow_anthropic.autolog()

    _mlflow_initialized = True


# =============================================================================

from claude_agent_sdk import (
    AgentDefinition,
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
    ResultMessage,
    create_sdk_mcp_server,
)

from .tools import create_tools, MCPTools, BuiltinTools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Re-export context monitoring from mlflow_ops (avoid circular imports)
from .mlflow_ops import (
    ContextMetrics,
    start_context_monitoring,
    get_context_metrics,
    record_tool_call,
    _reset_context_metrics,
)


# =============================================================================
# PROMPT LOADING
# =============================================================================

def _get_prompts_dir() -> Path:
    """Get prompts directory, works both installed and in dev.

    When installed as a wheel, uses importlib.resources.
    In development, falls back to relative path.
    """
    # Try package resources first (installed wheel)
    try:
        files = importlib.resources.files("prompts")
        # Check if it's a real directory we can use
        with importlib.resources.as_file(files) as prompts_path:
            if prompts_path.is_dir():
                return prompts_path
    except (TypeError, FileNotFoundError, ModuleNotFoundError):
        pass

    # Fallback to relative path (development)
    return Path(__file__).parent.parent / "prompts"


@mlflow.trace
def load_prompt(name: str = "system") -> str:
    """Load external prompt from prompts/ directory.

    Works both when installed as a wheel and in development mode.
    """
    # Try importlib.resources first for installed package
    try:
        files = importlib.resources.files("prompts")
        prompt_file = files.joinpath(f"{name}.md")
        return prompt_file.read_text()
    except (TypeError, FileNotFoundError, ModuleNotFoundError):
        pass

    # Fallback to relative path (development)
    path = Path(__file__).parent.parent / "prompts" / f"{name}.md"
    if path.exists():
        return path.read_text()

    logger.warning(f"Prompt file not found: {name}.md")
    return ""


@dataclass
class AgentResult:
    """Result from agent interaction."""
    success: bool
    response: str
    event_type: str = "text"  # text | thinking | tool_use | tool_result | result

    # Optional fields
    cost_usd: Optional[float] = None
    session_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_result: Optional[str] = None
    thinking_content: Optional[str] = None
    usage_data: Optional[dict] = None
    duration_ms: Optional[int] = None


class MLflowAgent:
    """Simplified MLflow Evaluation Agent.

    Features:
    - Single agent with 3 tools (vs coordinator + 4 sub-agents + 11 tools)
    - External prompts for easy iteration
    - File-based state management
    - Session support for multi-turn conversations
    """

    def __init__(self, config: Optional["Config"] = None):
        from .config import Config
        self.config = config or Config.from_env()
        self._last_session_id: Optional[str] = None

    @property
    def session_id(self) -> Optional[str]:
        """Last session ID for resumption."""
        return self._last_session_id

    @mlflow.trace
    def _build_system_prompt(self) -> str:
        """Build minimal system prompt with just experiment context.

        Note: Detailed tool/workflow info moved to worker/initializer prompts
        and mlflow-evaluation skill to reduce token overhead.
        """
        # Only include experiment context - no system.md (deleted for token savings)
        if self.config.experiment_id:
            return f"## Current Experiment\nExperiment ID: `{self.config.experiment_id}`\n"
        return ""

    @mlflow.trace
    def _build_options(self, session_id: Optional[str] = None) -> ClaudeAgentOptions:
        """Build agent options with simplified tool set."""
        tools = create_tools()

        mcp_server = create_sdk_mcp_server(
            name="mlflow-eval",
            version="2.0.0",  # Simplified version
            tools=tools,
        )

        # Load trace-analyzer prompt for sub-agent
        trace_analyzer_prompt = load_prompt("trace-analyzer")

        return ClaudeAgentOptions(
            system_prompt=self._build_system_prompt(),
            mcp_servers={"mlflow-eval": mcp_server},
            resume=session_id,
            allowed_tools=[
                # Built-in Claude tools
                BuiltinTools.READ,
                BuiltinTools.BASH,
                BuiltinTools.GLOB,
                BuiltinTools.GREP,
                BuiltinTools.SKILL,
                BuiltinTools.TASK,  # Required for subagent invocation
                # Our 3 simplified tools
                MCPTools.MLFLOW_QUERY,
                MCPTools.MLFLOW_ANNOTATE,
                MCPTools.SAVE_FINDINGS,
            ],
            agents={
                "trace-analyzer": AgentDefinition(
                    description="Analyzes MLflow traces in batches and returns structured JSON summaries. Use when you need to analyze multiple traces without filling your context.",
                    prompt=trace_analyzer_prompt,
                    tools=[
                        MCPTools.MLFLOW_QUERY,  # Only needs query access
                    ],
                    model=self.config.model,
                ),
            },
            setting_sources=["project"],
            cwd=str(self.config.working_dir),
            permission_mode="bypassPermissions",
            model=self.config.model,
            max_turns=self.config.max_turns,
        )

    @mlflow.trace(name="agent_query", span_type="AGENT")
    async def query(
        self,
        prompt: str,
        session_id: Optional[str] = None
    ) -> AsyncIterator[AgentResult]:
        """Send query and stream results.

        Args:
            prompt: Query for the agent
            session_id: Optional session ID to resume conversation
        """
        start_time = time.time()
        options = self._build_options(session_id=session_id)

        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            response_text = ""

            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            response_text += block.text
                            yield AgentResult(
                                success=True,
                                response=response_text,
                                event_type="text"
                            )

                        elif isinstance(block, ThinkingBlock):
                            yield AgentResult(
                                success=True,
                                response=response_text,
                                event_type="thinking",
                                thinking_content=block.thinking
                            )

                        elif isinstance(block, ToolUseBlock):
                            yield AgentResult(
                                success=True,
                                response=response_text,
                                event_type="tool_use",
                                tool_name=block.name,
                                tool_input=block.input
                            )

                        elif isinstance(block, ToolResultBlock):
                            yield AgentResult(
                                success=not block.is_error if block.is_error is not None else True,
                                response=response_text,
                                event_type="tool_result",
                                tool_result=str(block.content)[:500] if block.content else None
                            )

                elif isinstance(message, ResultMessage):
                    self._last_session_id = message.session_id
                    duration_ms = int((time.time() - start_time) * 1000)

                    # Set token tracking attributes on the current span
                    span = mlflow.get_current_active_span()
                    if span and message.usage:
                        usage = message.usage
                        span.set_attribute("input_tokens", usage.get("input_tokens", 0))
                        span.set_attribute("output_tokens", usage.get("output_tokens", 0))
                        span.set_attribute("cache_creation_input_tokens", usage.get("cache_creation_input_tokens", 0))
                        span.set_attribute("cache_read_input_tokens", usage.get("cache_read_input_tokens", 0))
                        span.set_attribute("total_tokens",
                            usage.get("input_tokens", 0) + usage.get("output_tokens", 0))
                    if span and message.total_cost_usd:
                        span.set_attribute("cost_usd", message.total_cost_usd)

                    yield AgentResult(
                        success=not message.is_error,
                        response=response_text,
                        event_type="result",
                        cost_usd=message.total_cost_usd,
                        session_id=message.session_id,
                        usage_data=message.usage,
                        duration_ms=duration_ms
                    )

    def clear_state(self):
        """Clear file-based state for fresh analysis."""
        from .mlflow_ops import clear_state
        clear_state()
        logger.info("State cleared")


# =============================================================================
# AUTONOMOUS MODE
# =============================================================================

AUTO_CONTINUE_DELAY_SECONDS = 3


@mlflow.trace(name="autonomous_evaluation", span_type="AGENT")
async def run_autonomous(
    experiment_id: str,
    max_iterations: Optional[int] = None,
) -> None:
    """Run autonomous evaluation loop with task tracking.

    Args:
        experiment_id: MLflow experiment ID to analyze
        max_iterations: Maximum iterations (None = until complete)
    """
    from .mlflow_ops import (
        all_tasks_complete,
        print_progress_summary,
        print_final_summary,
        set_session_dir,
        get_tasks_file,
    )
    from .runtime import get_sessions_base_path
    from .config import Config

    config = Config.from_env()
    config.experiment_id = experiment_id

    # Set up session directory (uses Volume in Databricks, local otherwise)
    sessions_base = get_sessions_base_path()
    session_dir = sessions_base / config.session_id
    set_session_dir(session_dir)

    logger.info(f"Session: {config.session_id}")
    logger.info(f"Output:  {session_dir}")

    # Check if first run (no task file exists in this session)
    is_first_run = not get_tasks_file().exists()

    if is_first_run:
        logger.info("=" * 60)
        logger.info("  INITIALIZER SESSION")
        logger.info("  Analyzing traces and creating task plan...")
        logger.info("=" * 60)

    iteration = 0
    while True:
        iteration += 1

        # Check iteration limit
        if max_iterations and iteration > max_iterations:
            logger.info(f"Reached max iterations ({max_iterations}). Stopping.")
            break

        # Check if all tasks complete
        if not is_first_run and all_tasks_complete():
            print_final_summary()
            break

        # Track each iteration as a sub-span
        with mlflow.start_span(name=f"session_{iteration}") as iter_span:
            iter_span.set_attribute("iteration", iteration)
            iter_span.set_attribute("phase", "initializer" if is_first_run else "worker")

            # Fresh agent per session (Anthropic pattern)
            agent = MLflowAgent(config)

            # Choose prompt based on state
            prompt_name = "initializer" if is_first_run else "worker"
            prompt = load_prompt(prompt_name)
            prompt = prompt.replace("{experiment_id}", experiment_id)
            prompt = prompt.replace("{session_dir}", str(session_dir))

            # Start context monitoring for this session
            context_metrics = start_context_monitoring(
                session_id=f"{config.session_id}_iter{iteration}",
                initial_prompt=prompt
            )

            logger.info(f"--- Session {iteration} ({prompt_name}) ---")

            # Run session and stream output
            try:
                async for result in agent.query(prompt):
                    if result.event_type == "text":
                        # Print incremental text (clear line and reprint for streaming effect)
                        pass  # Text is accumulated in result.response

                # Log final response
                if result and result.response:
                    logger.info(f"Response:\n{result.response}")
                    iter_span.set_attribute("response_length", len(result.response))

                    # Show cost if available
                    if result.cost_usd:
                        logger.info(f"[Cost: ${result.cost_usd:.4f}]")
                        iter_span.set_attribute("cost_usd", result.cost_usd)

                    # Propagate token tracking to session span
                    if result.usage_data:
                        usage = result.usage_data
                        iter_span.set_attribute("input_tokens", usage.get("input_tokens", 0))
                        iter_span.set_attribute("output_tokens", usage.get("output_tokens", 0))
                        iter_span.set_attribute("cache_creation_input_tokens", usage.get("cache_creation_input_tokens", 0))
                        iter_span.set_attribute("cache_read_input_tokens", usage.get("cache_read_input_tokens", 0))
                        iter_span.set_attribute("total_tokens",
                            usage.get("input_tokens", 0) + usage.get("output_tokens", 0))

                # Log context metrics to span
                if context_metrics:
                    iter_span.set_attribute("context_tool_calls", context_metrics.tool_calls)
                    iter_span.set_attribute("context_estimated_messages", context_metrics.estimated_messages)
                    iter_span.set_attribute("context_estimated_kb", context_metrics.estimated_context_kb)
                    logger.info(
                        f"[Context] {context_metrics.tool_calls} tool calls, "
                        f"~{context_metrics.estimated_messages} messages, "
                        f"~{context_metrics.estimated_context_kb:.1f}KB"
                    )

            except KeyboardInterrupt:
                iter_span.set_attribute("status", "interrupted")
                logger.info("Interrupted by user.")
                break
            except Exception as e:
                iter_span.set_attribute("error", str(e))
                logger.error(f"Error in session: {e}")
                logger.exception("Session error")

            # After session completes, transition from initializer to worker
            if is_first_run:
                tasks_file = get_tasks_file()
                if tasks_file.exists():
                    is_first_run = False
                    logger.info("✓ Initializer session complete. Switching to worker mode.")
                else:
                    logger.warning("⚠ Initializer did not create task file. Will retry initializer session.")

        # Progress summary
        print_progress_summary()

        # Auto-continue with interrupt window
        from .runtime import detect_runtime, RuntimeContext
        runtime = detect_runtime()

        if runtime.context == RuntimeContext.LOCAL:
            logger.info(f"Continuing in {AUTO_CONTINUE_DELAY_SECONDS}s (Ctrl+C to stop)...")
        else:
            logger.info(f"Continuing in {AUTO_CONTINUE_DELAY_SECONDS}s...")

        try:
            await asyncio.sleep(AUTO_CONTINUE_DELAY_SECONDS)
        except KeyboardInterrupt:
            logger.info("Stopped by user.")
            break


# CLI entry point
if __name__ == "__main__":
    from .cli import main
    asyncio.run(main())
