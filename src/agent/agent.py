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
import logging
import time
from dataclasses import dataclass
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
    from ..core.auth import configure_env
    configure_env()

    import mlflow as _mlflow  # Local import to avoid scoping issues
    import mlflow.anthropic as mlflow_anthropic

    from ..core.config import Config
    config = Config.from_env(validate=False)

    _mlflow.set_tracking_uri(config.tracking_uri)
    if config.agent_experiment_id:
        _mlflow.set_experiment(experiment_id=config.agent_experiment_id)
        logging.info(f"MLflow experiment set to: {config.agent_experiment_id}")

    mlflow_anthropic.autolog()

    _mlflow_initialized = True


# =============================================================================

from claude_agent_sdk import (
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
        from ..core.config import Config
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
                # MLflow tools
                MCPTools.MLFLOW_QUERY,
                MCPTools.MLFLOW_ANNOTATE,
                MCPTools.SAVE_FINDINGS,
                # File tools - UC Volume
                MCPTools.UC_VOLUME_READ,
                MCPTools.UC_VOLUME_WRITE,
                MCPTools.UC_VOLUME_LIST,
                # File tools - Workspace
                MCPTools.WORKSPACE_READ,
                MCPTools.WORKSPACE_WRITE,
                MCPTools.WORKSPACE_LIST,
            ],
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


# CLI entry point
if __name__ == "__main__":
    from ..cli import main
    asyncio.run(main())
