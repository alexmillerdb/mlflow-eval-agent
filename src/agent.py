"""
MLflow Evaluation Agent

A modular, evaluation-driven agent built with Claude Agent SDK.
Features coordinator + sub-agent architecture with inter-agent communication
via shared workspace.
"""

import os
import logging
import time
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import mlflow.anthropic

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    ResultMessage,
    create_sdk_mcp_server,
)

from .config import EvalAgentConfig
from .workspace import SharedWorkspace
from .tools import create_tools, MCPTools, InternalTools, BuiltinTools
from .subagents import create_subagents, get_coordinator_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvalAgentResult:
    """Result from agent interaction."""
    success: bool
    response: str
    run_id: Optional[str] = None
    cost_usd: Optional[float] = None
    session_id: Optional[str] = None
    timing_metrics: Optional[dict] = None

    # Event type
    event_type: str = "text"  # text | tool_use | tool_result | todo_update | subagent | result

    # Tool call fields
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_use_id: Optional[str] = None

    # Tool result fields
    tool_result: Optional[str] = None
    tool_is_error: Optional[bool] = None

    # Todo fields
    todos: Optional[list] = None

    # Subagent fields
    subagent_name: Optional[str] = None


class MLflowEvalAgent:
    """MLflow Evaluation Agent with inter-agent communication.

    Features:
    - Coordinator + 3 specialized sub-agents
    - Instance-scoped workspace for inter-agent data passing
    - Skills loaded from filesystem
    - Dynamic evaluation script generation
    """

    def __init__(self, config: Optional[EvalAgentConfig] = None):
        self.config = config or EvalAgentConfig.from_env()

        os.environ["DATABRICKS_HOST"] = self.config.databricks_host
        os.environ["MLFLOW_TRACKING_URI"] = self.config.tracking_uri
        mlflow.set_tracking_uri(self.config.tracking_uri)
        mlflow.set_experiment(self.config.experiment_id)
        mlflow.anthropic.autolog()

        self.workspace = SharedWorkspace(max_context_chars=self.config.workspace_context_max_chars)
        self._client: Optional[ClaudeSDKClient] = None
        self._start_time: Optional[float] = None
        self._last_session_id: Optional[str] = None

    @property
    def session_id(self) -> Optional[str]:
        """Get the last session ID for resumption."""
        return self._last_session_id

    def _validate_skills(self) -> list[str]:
        """Validate skills exist with graceful fallback."""
        warnings = []
        skills_dir = self.config.skills_dir
        expected_skills = [
            "mlflow-evaluation/SKILL.md",
            "trace-analysis/SKILL.md",
            "context-engineering/SKILL.md",
        ]

        for skill_path in expected_skills:
            if not (skills_dir / skill_path).exists():
                warnings.append(f"Skill not found: {skill_path}")

        if warnings:
            logger.warning(f"Some skills are missing: {warnings}")
        return warnings

    def _build_options(self, session_id: Optional[str] = None) -> ClaudeAgentOptions:
        """Build agent options."""
        skill_warnings = self._validate_skills()
        tools = create_tools(self.workspace)

        mcp_server = create_sdk_mcp_server(
            name="mlflow-eval",
            version="1.0.0",
            tools=tools,
        )

        subagents = create_subagents(self.workspace)
        system_prompt = get_coordinator_prompt(self.workspace)

        if skill_warnings:
            system_prompt += f"\n\nNOTE: Some skill files are missing: {skill_warnings}\n"

        return ClaudeAgentOptions(
            system_prompt=system_prompt,
            agents=subagents,
            mcp_servers={"mlflow-eval": mcp_server},
            resume=session_id,
            allowed_tools=[
                # Built-in
                BuiltinTools.READ, BuiltinTools.BASH, BuiltinTools.GLOB,
                BuiltinTools.GREP, BuiltinTools.SKILL,
                # Internal workspace tools
                InternalTools.WRITE_TO_WORKSPACE,
                InternalTools.READ_FROM_WORKSPACE,
                InternalTools.CHECK_DEPENDENCIES,
                # MLflow MCP Server tools
                MCPTools.SEARCH_TRACES, MCPTools.GET_TRACE,
                MCPTools.GET_EXPERIMENT, MCPTools.SEARCH_EXPERIMENTS,
                MCPTools.SET_TRACE_TAG, MCPTools.DELETE_TRACE_TAG,
                MCPTools.LOG_FEEDBACK, MCPTools.LOG_EXPECTATION,
                MCPTools.GET_ASSESSMENT, MCPTools.UPDATE_ASSESSMENT,
            ],
            setting_sources=["project"],
            cwd=str(self.config.working_dir),
            permission_mode="bypassPermissions",
            env={
                "DATABRICKS_HOST": self.config.databricks_host,
                "DATABRICKS_TOKEN": self.config.databricks_token,
                "MLFLOW_TRACKING_URI": "databricks",
            },
            model=self.config.model,
            max_turns=self.config.max_turns,
        )

    async def query(self, prompt: str, session_id: Optional[str] = None) -> AsyncIterator[EvalAgentResult]:
        """Send query and stream results.

        Args:
            prompt: The query to send to the agent.
            session_id: Optional session ID to resume a previous conversation.
        """
        self._start_time = time.time()
        options = self._build_options(session_id=session_id)

        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            response_text = ""

            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            response_text += block.text
                            yield EvalAgentResult(success=True, response=response_text, event_type="text")

                        elif isinstance(block, ToolUseBlock):
                            if block.name == "Task":
                                yield EvalAgentResult(
                                    success=True, response=response_text, event_type="subagent",
                                    subagent_name=block.input.get("subagent_type", "unknown"),
                                    tool_use_id=block.id,
                                )
                            elif block.name == "TodoWrite":
                                yield EvalAgentResult(
                                    success=True, response=response_text, event_type="todo_update",
                                    todos=block.input.get("todos", []),
                                )
                            else:
                                yield EvalAgentResult(
                                    success=True, response=response_text, event_type="tool_use",
                                    tool_name=block.name, tool_input=block.input, tool_use_id=block.id,
                                )

                        elif isinstance(block, ToolResultBlock):
                            yield EvalAgentResult(
                                success=not block.is_error if block.is_error is not None else True,
                                response=response_text, event_type="tool_result",
                                tool_use_id=block.tool_use_id,
                                tool_result=str(block.content)[:500] if block.content else None,
                                tool_is_error=block.is_error,
                            )

                elif isinstance(message, ResultMessage):
                    timing = self.workspace.get_timing_metrics()
                    timing["total_query_time"] = time.time() - self._start_time

                    # Store session_id for follow-up queries
                    self._last_session_id = message.session_id

                    yield EvalAgentResult(
                        success=not message.is_error, response=response_text, event_type="result",
                        cost_usd=message.total_cost_usd, session_id=message.session_id,
                        timing_metrics=timing,
                    )

    async def analyze_and_generate(
        self,
        filter_string: str = "attributes.status = 'OK'",
        agent_name: str = "my_agent",
    ) -> str:
        """Complete workflow: analyze traces and generate evaluation."""
        prompt = f"""
        Please perform a complete analysis and generate an evaluation suite:

        1. Use trace_analyst to analyze traces with filter: {filter_string}
        2. Use context_engineer to identify quality issues
        3. Generate a comprehensive evaluation script for "{agent_name}"

        The evaluation script should:
        - Include scorers based on the issues found
        - Include test cases extracted from failures
        - Have appropriate thresholds for CI/CD
        - Be completely runnable standalone
        """

        final_response = ""
        async for result in self.query(prompt):
            final_response = result.response

        return final_response

    def clear_workspace(self):
        """Clear the workspace for a fresh analysis."""
        self.workspace.clear()
        logger.info("Workspace cleared")


# CLI entry point for backward compatibility
if __name__ == "__main__":
    from .cli import cli
    cli()
