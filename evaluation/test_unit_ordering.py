"""Unit tests for agent sequential ordering.

Tests that agents execute in the correct dependency-based order and that
the workspace properly enforces dependencies.
"""

import pytest
from unittest.mock import MagicMock

from evaluation.test_data import (
    EXPECTED_WORKFLOW_ORDER,
    EXPECTED_AGENT_DEPENDENCIES,
    ORDERING_CASES,
)


class TestWorkflowOrder:
    """Tests for workflow order computation."""

    def test_get_workflow_order_returns_list(self):
        """get_workflow_order should return a list of agent names."""
        from src.subagents.registry import get_workflow_order

        order = get_workflow_order()

        assert isinstance(order, list)
        assert len(order) > 0

    def test_trace_analyst_early_in_order(self):
        """trace_analyst should be early (no dependencies on other agents)."""
        from src.subagents.registry import get_workflow_order

        order = get_workflow_order()

        # trace_analyst should appear before context_engineer and agent_architect
        if "trace_analyst" in order and "context_engineer" in order:
            trace_idx = order.index("trace_analyst")
            context_idx = order.index("context_engineer")
            assert trace_idx < context_idx, (
                f"trace_analyst should come before context_engineer. Order: {order}"
            )

    def test_context_engineer_after_trace_analyst(self):
        """context_engineer depends on trace_analyst outputs."""
        from src.subagents.registry import get_workflow_order

        order = get_workflow_order()

        if "context_engineer" in order and "trace_analyst" in order:
            trace_idx = order.index("trace_analyst")
            context_idx = order.index("context_engineer")
            assert context_idx > trace_idx, (
                "context_engineer must come after trace_analyst"
            )

    def test_eval_runner_registered(self):
        """eval_runner should be in the workflow order."""
        from src.subagents.registry import get_workflow_order

        order = get_workflow_order()

        # eval_runner should be registered
        assert "eval_runner" in order, f"eval_runner should be in workflow. Order: {order}"

    def test_workflow_order_respects_dependencies(self):
        """Workflow order should respect dependencies (trace_analyst before dependents)."""
        from src.subagents.registry import get_workflow_order, AGENT_REGISTRY

        order = get_workflow_order()

        # Core dependency: trace_analyst must come before context_engineer
        # because context_engineer requires trace_analyst's outputs
        if "trace_analyst" in order and "context_engineer" in order:
            trace_idx = order.index("trace_analyst")
            context_idx = order.index("context_engineer")
            assert trace_idx < context_idx, (
                f"trace_analyst must precede context_engineer. Order: {order}"
            )

        # Verify all agents are in the order
        for agent_name in AGENT_REGISTRY:
            assert agent_name in order, f"Agent '{agent_name}' must be in workflow order"


class TestDependencyValidation:
    """Tests for dependency validation before agent execution."""

    def test_validate_agent_can_run_exists(self):
        """validate_agent_can_run function should exist."""
        from src.subagents import validate_agent_can_run

        assert callable(validate_agent_can_run)

    def test_trace_analyst_can_run_with_empty_workspace(self):
        """trace_analyst has no dependencies, should always be able to run."""
        from src.subagents import validate_agent_can_run
        from src.workspace import SharedWorkspace

        workspace = SharedWorkspace()
        can_run, missing, message = validate_agent_can_run("trace_analyst", workspace)

        assert can_run, f"trace_analyst should be able to run: {message}"
        assert len(missing) == 0

    def test_context_engineer_cannot_run_without_trace_data(self):
        """context_engineer requires trace_analysis_summary and error_patterns."""
        from src.subagents import validate_agent_can_run
        from src.workspace import SharedWorkspace

        workspace = SharedWorkspace()
        can_run, missing, message = validate_agent_can_run(
            "context_engineer", workspace, strict=True
        )

        assert not can_run, "context_engineer should NOT run without dependencies"
        assert "trace_analysis_summary" in missing or "error_patterns" in missing

    def test_context_engineer_can_run_with_trace_data(self, mock_workspace_with_trace_data):
        """context_engineer can run when trace data is present."""
        from src.subagents import validate_agent_can_run

        can_run, missing, message = validate_agent_can_run(
            "context_engineer", mock_workspace_with_trace_data
        )

        assert can_run, f"context_engineer should be able to run: {message}"

    def test_eval_runner_cannot_run_without_generated_code(self):
        """eval_runner requires generated_eval_code."""
        from src.subagents import validate_agent_can_run
        from src.workspace import SharedWorkspace

        workspace = SharedWorkspace()
        can_run, missing, message = validate_agent_can_run(
            "eval_runner", workspace, strict=True
        )

        assert not can_run, "eval_runner should NOT run without generated code"
        assert "generated_eval_code" in missing


class TestWorkspaceSchemaValidation:
    """Tests for workspace schema validation."""

    def test_trace_analysis_summary_schema(self):
        """trace_analysis_summary must follow schema."""
        from src.workspace import SharedWorkspace

        workspace = SharedWorkspace()

        # Valid data
        valid_data = {
            "error_rate": 0.1,
            "success_rate": 0.9,
            "trace_count": 100,
            "avg_latency_ms": 250.5,
            "experiment_id": "123",
            "time_range": "last 24 hours",
        }

        success, msg = workspace.write("trace_analysis_summary", valid_data, agent="test")
        assert success, f"Valid data should be accepted: {msg}"

    def test_error_patterns_schema(self):
        """error_patterns must be a list of ErrorPattern objects."""
        from src.workspace import SharedWorkspace

        workspace = SharedWorkspace()

        # Valid data - list of error patterns
        valid_data = [
            {
                "error_type": "TimeoutError",
                "count": 5,
                "example_trace_ids": ["trace_1", "trace_2"],
            },
            {
                "error_type": "ValidationError",
                "count": 3,
                "example_trace_ids": ["trace_3"],
            },
        ]

        success, msg = workspace.write("error_patterns", valid_data, agent="test")
        assert success, f"Valid error_patterns should be accepted: {msg}"


class TestOrderingCases:
    """Test the predefined ordering cases from test_data.py."""

    @pytest.mark.parametrize(
        "test_case",
        ORDERING_CASES,
        ids=[c["expectations"]["description"][:50] for c in ORDERING_CASES],
    )
    def test_ordering_case_dependencies(self, test_case):
        """Each ordering case should respect dependency order."""
        from src.subagents.registry import AGENT_REGISTRY

        expected_order = test_case["expectations"]["expected_order"]

        # Build dependency map
        producers = {}
        for name, config in AGENT_REGISTRY.items():
            for key in config.output_keys:
                producers[key] = name

        # Verify each agent's dependencies are produced by earlier agents
        for i, agent in enumerate(expected_order[1:], start=1):
            if agent not in AGENT_REGISTRY:
                continue

            config = AGENT_REGISTRY[agent]
            earlier_agents = set(expected_order[:i])

            for required_key in config.required_keys:
                producer = producers.get(required_key)
                if producer:
                    assert producer in earlier_agents, (
                        f"Agent '{agent}' requires key '{required_key}' "
                        f"which is produced by '{producer}', but '{producer}' "
                        f"is not in the earlier agents: {earlier_agents}"
                    )
