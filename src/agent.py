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
from pathlib import Path
from typing import AsyncIterator, Optional

# =============================================================================
# MLFLOW SETUP - Must happen BEFORE importing mlflow.anthropic
# The mlflow.anthropic import triggers tracking initialization, so we must
# set tracking_uri first to avoid defaulting to local SQLite.
# =============================================================================
import mlflow
from .config import Config

_config = Config.from_env(validate=False)  # Don't validate - may not have all env vars
mlflow.set_tracking_uri(_config.tracking_uri)
if _config.agent_experiment_id:
    mlflow.set_experiment(experiment_id=_config.agent_experiment_id)

import mlflow.anthropic  # Now imports with correct tracking URI
mlflow.anthropic.autolog()

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

# Paths
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


# =============================================================================
# PROMPT LOADING
# =============================================================================

@mlflow.trace
def load_prompt(name: str = "system") -> str:
    """Load external prompt from prompts/ directory."""
    path = PROMPTS_DIR / f"{name}.md"
    if path.exists():
        return path.read_text()
    logger.warning(f"Prompt file not found: {path}")
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
                # Our 3 simplified tools
                MCPTools.MLFLOW_QUERY,
                MCPTools.MLFLOW_ANNOTATE,
                MCPTools.SAVE_FINDINGS,
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
SESSIONS_DIR = Path("sessions")


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

    config = Config.from_env()
    config.experiment_id = experiment_id

    # Set up session directory
    session_dir = SESSIONS_DIR / config.session_id
    set_session_dir(session_dir)

    print(f"\nSession: {config.session_id}")
    print(f"Output:  {session_dir}")

    # Check if first run (no task file exists in this session)
    is_first_run = not get_tasks_file().exists()

    if is_first_run:
        print("\n" + "=" * 60)
        print("  INITIALIZER SESSION")
        print("  Analyzing traces and creating task plan...")
        print("=" * 60 + "\n")

    iteration = 0
    while True:
        iteration += 1

        # Check iteration limit
        if max_iterations and iteration > max_iterations:
            print(f"\nReached max iterations ({max_iterations}). Stopping.")
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

            # After first run, switch to worker mode
            is_first_run = False

            print(f"\n--- Session {iteration} ({prompt_name}) ---\n")

            # Run session and stream output
            try:
                async for result in agent.query(prompt):
                    if result.event_type == "text":
                        # Print incremental text (clear line and reprint for streaming effect)
                        pass  # Text is accumulated in result.response

                # Print final response
                if result and result.response:
                    print(result.response)
                    iter_span.set_attribute("response_length", len(result.response))

                    # Show cost if available
                    if result.cost_usd:
                        print(f"\n[Cost: ${result.cost_usd:.4f}]")
                        iter_span.set_attribute("cost_usd", result.cost_usd)

            except KeyboardInterrupt:
                iter_span.set_attribute("status", "interrupted")
                print("\n\nInterrupted by user.")
                break
            except Exception as e:
                iter_span.set_attribute("error", str(e))
                print(f"\nError in session: {e}")
                logger.exception("Session error")

        # Progress summary
        print_progress_summary()

        # Auto-continue with interrupt window
        print(f"\nContinuing in {AUTO_CONTINUE_DELAY_SECONDS}s (Ctrl+C to stop)...")
        try:
            await asyncio.sleep(AUTO_CONTINUE_DELAY_SECONDS)
        except KeyboardInterrupt:
            print("\n\nStopped by user.")
            break


# CLI entry point
if __name__ == "__main__":
    from .cli import main
    asyncio.run(main())
