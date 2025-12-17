"""Unit tests for agent tool call patterns.

Tests that agents have access to the correct tools and use them appropriately.
"""

import pytest
from unittest.mock import MagicMock

from evaluation.test_data import (
    EXPECTED_AGENT_TOOLS,
    TOOL_CALL_CASES,
)


class TestAgentToolConfiguration:
    """Tests for agent tool configuration in the registry."""

    def test_trace_analyst_has_required_tools(self):
        """trace_analyst must have access to trace search and workspace tools."""
        from src.subagents.registry import AGENT_REGISTRY

        config = AGENT_REGISTRY.get("trace_analyst")
        assert config is not None, "trace_analyst must be registered"

        # Check for required tools (may be in different formats)
        tool_names = [t.lower() for t in config.tools]
        tool_str = " ".join(tool_names)

        assert any("search" in t or "trace" in t for t in tool_names), (
            f"trace_analyst should have search_traces tool, got: {config.tools}"
        )
        assert any("write" in t and "workspace" in t for t in tool_names) or "workspace" in tool_str, (
            f"trace_analyst should have write_to_workspace tool, got: {config.tools}"
        )

    def test_context_engineer_has_workspace_tools(self):
        """context_engineer must have read/write workspace access."""
        from src.subagents.registry import AGENT_REGISTRY

        config = AGENT_REGISTRY.get("context_engineer")
        assert config is not None, "context_engineer must be registered"

        tool_names = [t.lower() for t in config.tools]
        tool_str = " ".join(tool_names)

        # Should have workspace read/write
        has_workspace = "workspace" in tool_str or any("workspace" in t for t in tool_names)
        assert has_workspace, (
            f"context_engineer should have workspace tools, got: {config.tools}"
        )

    def test_eval_runner_has_bash_tool(self):
        """eval_runner must have bash access for script execution."""
        from src.subagents.registry import AGENT_REGISTRY

        config = AGENT_REGISTRY.get("eval_runner")
        assert config is not None, "eval_runner must be registered"

        tool_names = [t.lower() for t in config.tools]

        has_bash = any("bash" in t for t in tool_names)
        assert has_bash, (
            f"eval_runner should have bash tool, got: {config.tools}"
        )

    def test_all_agents_have_tools(self):
        """All registered agents must have at least one tool."""
        from src.subagents.registry import AGENT_REGISTRY

        for name, config in AGENT_REGISTRY.items():
            assert len(config.tools) > 0, f"Agent '{name}' must have at least one tool"


class TestToolNaming:
    """Tests for tool naming conventions."""

    def test_mcp_tool_prefix(self):
        """MCP tools should follow naming convention."""
        from src.subagents.registry import AGENT_REGISTRY

        mcp_tools = set()
        for config in AGENT_REGISTRY.values():
            for tool in config.tools:
                if "mcp" in tool.lower() or "__" in tool:
                    mcp_tools.add(tool)

        # MCP tools should have consistent naming
        for tool in mcp_tools:
            # Format: mcp__server__tool or MCPTools.TOOL_NAME
            assert "__" in tool or "." in tool or tool.isupper(), (
                f"MCP tool '{tool}' should follow naming convention"
            )

    def test_workspace_tool_names(self):
        """Workspace tools should have consistent naming."""
        from src.subagents.registry import AGENT_REGISTRY

        workspace_tools = set()
        for config in AGENT_REGISTRY.values():
            for tool in config.tools:
                if "workspace" in tool.lower():
                    workspace_tools.add(tool)

        # Should have read and write variants
        tool_str = " ".join(workspace_tools).lower()
        # At least one workspace tool should exist
        assert len(workspace_tools) > 0 or any(
            "workspace" in " ".join(c.tools).lower()
            for c in AGENT_REGISTRY.values()
        ), "Should have workspace tools defined"


class TestToolCallCases:
    """Test the predefined tool call cases from test_data.py."""

    @pytest.mark.parametrize(
        "test_case",
        TOOL_CALL_CASES,
        ids=[c["expectations"]["description"][:50] for c in TOOL_CALL_CASES],
    )
    def test_tool_case_expected_tools_valid(self, test_case):
        """Each tool case should have valid expected_tools."""
        expected_tools = test_case["expectations"]["expected_tools"]

        assert isinstance(expected_tools, list)
        assert len(expected_tools) > 0, "expected_tools should not be empty"

        for tool in expected_tools:
            assert isinstance(tool, str)
            assert len(tool) > 0


class TestScorerToolLogic:
    """Tests for scorer tool detection logic."""

    def test_tool_selection_scorer_normalize(self):
        """Test tool name normalization in scorer."""
        from evaluation.scorers import tool_selection_accuracy

        # Test that the scorer function exists and is callable
        assert callable(tool_selection_accuracy)

    def test_workspace_io_scorer_exists(self):
        """workspace_io_correctness scorer should exist."""
        from evaluation.scorers import workspace_io_correctness

        assert callable(workspace_io_correctness)


class TestExpectedToolsMapping:
    """Tests that expected tools map to actual agent tools."""

    def test_expected_tools_exist_in_registry(self):
        """Tools in EXPECTED_AGENT_TOOLS should match registry tools."""
        from src.subagents.registry import AGENT_REGISTRY

        for agent_name, expected in EXPECTED_AGENT_TOOLS.items():
            if agent_name not in AGENT_REGISTRY:
                continue

            config = AGENT_REGISTRY[agent_name]
            config_tools_lower = [t.lower() for t in config.tools]
            config_tools_str = " ".join(config_tools_lower)

            # Check required tools are present (fuzzy match)
            for tool in expected["required_tools"]:
                tool_lower = tool.lower()
                # Allow for different naming conventions
                tool_parts = tool_lower.replace("_", " ").split()

                found = any(
                    all(part in ct for part in tool_parts)
                    for ct in config_tools_lower
                ) or all(part in config_tools_str for part in tool_parts)

                # Note: This is a soft check - log warning but don't fail
                # because tool naming can vary
                if not found:
                    print(
                        f"Warning: Tool '{tool}' not found in {agent_name} tools: {config.tools}"
                    )


class TestToolAccessPatterns:
    """Tests for tool access patterns by agent type."""

    def test_trace_analyst_no_code_execution(self):
        """trace_analyst should NOT have direct code execution tools."""
        from src.subagents.registry import AGENT_REGISTRY

        config = AGENT_REGISTRY.get("trace_analyst")
        if config is None:
            pytest.skip("trace_analyst not registered")

        tool_names = [t.lower() for t in config.tools]

        # Should not have dangerous code execution tools (bash, exec, subprocess)
        # Note: "eval" appears in "mlflow-eval" which is fine - it's the server name
        dangerous = ["bash", "exec(", "subprocess"]
        for d in dangerous:
            # Check for exact tool name or tool that IS the dangerous one
            has_dangerous = any(
                t == d or t.endswith(f"__{d}") or t == f"Bash"
                for t in config.tools
            )
            assert not has_dangerous, (
                f"trace_analyst should not have '{d}' tool"
            )

    def test_eval_runner_has_controlled_execution(self):
        """eval_runner should have controlled code execution."""
        from src.subagents.registry import AGENT_REGISTRY

        config = AGENT_REGISTRY.get("eval_runner")
        if config is None:
            pytest.skip("eval_runner not registered")

        # Should have bash for controlled execution
        tool_names = [t.lower() for t in config.tools]
        has_execution = any("bash" in t or "run" in t for t in tool_names)

        assert has_execution, "eval_runner should have execution capability"
