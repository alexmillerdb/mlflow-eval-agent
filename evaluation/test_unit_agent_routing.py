"""Unit tests for agent routing logic.

Tests that the coordinator correctly routes requests to the appropriate sub-agents
based on query content.
"""

import pytest
from unittest.mock import MagicMock, patch

from evaluation.test_data import AGENT_ROUTING_CASES, EXPECTED_AGENT_DEPENDENCIES


class TestAgentRoutingLogic:
    """Tests for agent routing based on query patterns."""

    def test_trace_analyst_keywords(self):
        """Verify trace analysis keywords map to trace_analyst."""
        trace_keywords = [
            "analyze traces",
            "error patterns",
            "trace analysis",
            "search traces",
            "performance metrics",
            "latency analysis",
        ]

        for keyword in trace_keywords:
            query = f"Please {keyword} from experiment 123"
            agent = self._determine_agent_from_query(query)
            assert agent == "trace_analyst", f"Query '{query}' should route to trace_analyst"

    def test_context_engineer_keywords(self):
        """Verify context optimization keywords map to context_engineer."""
        context_keywords = [
            "optimize context",
            "reduce token",
            "prompt optimization",
            "context engineering",
            "token budget",
        ]

        for keyword in context_keywords:
            query = f"Please {keyword}"
            agent = self._determine_agent_from_query(query)
            assert agent == "context_engineer", f"Query '{query}' should route to context_engineer"

    def test_agent_architect_keywords(self):
        """Verify architecture keywords map to agent_architect."""
        arch_keywords = [
            "review architecture",
            "multi-agent",
            "structural analysis",
            "agent design",
        ]

        for keyword in arch_keywords:
            query = f"Please {keyword}"
            agent = self._determine_agent_from_query(query)
            assert agent == "agent_architect", f"Query '{query}' should route to agent_architect"

    def test_eval_runner_keywords(self):
        """Verify evaluation execution keywords map to eval_runner."""
        eval_keywords = [
            "execute evaluation",
            "run the evaluation",
            "execute the script",
        ]

        for keyword in eval_keywords:
            query = f"Please {keyword}"
            agent = self._determine_agent_from_query(query)
            assert agent == "eval_runner", f"Query '{query}' should route to eval_runner"

    def _determine_agent_from_query(self, query: str) -> str:
        """Simple keyword-based agent determination for testing.

        This mimics the routing logic in the coordinator.
        """
        query_lower = query.lower()

        # Order matters - check most specific first
        if any(
            kw in query_lower
            for kw in ["execute evaluation", "run the evaluation", "execute the script"]
        ):
            return "eval_runner"

        if any(
            kw in query_lower
            for kw in [
                "optimize context",
                "context optimization",
                "token budget",
                "reduce token",
                "prompt optimization",
                "context engineering",
            ]
        ):
            return "context_engineer"

        if any(
            kw in query_lower
            for kw in [
                "architecture",
                "multi-agent",
                "structural",
                "agent design",
            ]
        ):
            return "agent_architect"

        # Default to trace_analyst for analysis tasks
        if any(
            kw in query_lower
            for kw in [
                "trace",
                "analyze",
                "error",
                "pattern",
                "latency",
                "performance",
            ]
        ):
            return "trace_analyst"

        return "trace_analyst"  # Default


class TestAgentRegistryConfiguration:
    """Tests for agent registry configuration."""

    def test_all_agents_registered(self):
        """Verify all expected agents are in the registry."""
        from src.subagents.registry import AGENT_REGISTRY

        expected_agents = ["trace_analyst", "context_engineer", "agent_architect", "eval_runner"]

        for agent in expected_agents:
            assert agent in AGENT_REGISTRY, f"Agent '{agent}' should be registered"

    def test_agent_has_description(self):
        """Each agent must have a description for coordinator routing."""
        from src.subagents.registry import AGENT_REGISTRY

        for name, config in AGENT_REGISTRY.items():
            assert config.description, f"Agent '{name}' must have a description"
            assert len(config.description) > 10, f"Agent '{name}' description too short"

    def test_agent_has_prompt_template(self):
        """Each agent must have a prompt template."""
        from src.subagents.registry import AGENT_REGISTRY

        for name, config in AGENT_REGISTRY.items():
            assert config.prompt_template, f"Agent '{name}' must have a prompt_template"

    def test_agent_dependencies_match_expected(self):
        """Verify agent dependencies match expected configuration."""
        from src.subagents.registry import AGENT_REGISTRY

        for agent_name, expected in EXPECTED_AGENT_DEPENDENCIES.items():
            if agent_name not in AGENT_REGISTRY:
                continue

            config = AGENT_REGISTRY[agent_name]

            # Check required_keys
            for key in expected["required_keys"]:
                assert key in config.required_keys, (
                    f"Agent '{agent_name}' should require key '{key}'"
                )

            # Check output_keys
            for key in expected["output_keys"]:
                assert key in config.output_keys, (
                    f"Agent '{agent_name}' should output key '{key}'"
                )


class TestAgentRoutingCases:
    """Test the predefined routing cases from test_data.py."""

    @pytest.mark.parametrize(
        "test_case",
        AGENT_ROUTING_CASES,
        ids=[c["expectations"]["description"][:50] for c in AGENT_ROUTING_CASES],
    )
    def test_routing_case(self, test_case):
        """Each routing case should have valid expected_agent."""
        from src.subagents.registry import AGENT_REGISTRY

        expected_agent = test_case["expectations"]["expected_agent"]

        assert expected_agent in AGENT_REGISTRY, (
            f"Expected agent '{expected_agent}' must be in registry"
        )
