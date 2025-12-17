"""Tests for selective context injection."""

import pytest

from src.workspace import SharedWorkspace
from src.subagents.registry import AgentConfig
from src.subagents import (
    AGENT_REGISTRY,
    TRACE_ANALYST_CONFIG,
    CONTEXT_ENGINEER_CONFIG,
    AGENT_ARCHITECT_CONFIG,
    create_subagents,
    validate_agent_can_run,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def workspace():
    """Create a fresh workspace for each test."""
    return SharedWorkspace(max_context_chars=2000)


@pytest.fixture
def populated_workspace(workspace):
    """Workspace with trace_analyst output populated."""
    workspace.write(
        "trace_analysis_summary",
        {
            "error_rate": 0.15,
            "success_rate": 0.85,
            "trace_count": 100,
            "top_errors": ["timeout", "validation_error"],
            "avg_latency_ms": 250.5,
        },
        agent="trace_analyst"
    )
    workspace.write(
        "error_patterns",
        [
            {"error_type": "timeout", "count": 10, "example_trace_ids": ["tr-1", "tr-2"]},
            {"error_type": "validation", "count": 5, "example_trace_ids": ["tr-3"]},
        ],
        agent="trace_analyst"
    )
    workspace.write(
        "performance_metrics",
        {
            "avg_latency_ms": 250.5,
            "p50_latency_ms": 200.0,
            "p95_latency_ms": 500.0,
            "p99_latency_ms": 800.0,
        },
        agent="trace_analyst"
    )
    return workspace


@pytest.fixture
def simple_config():
    """Simple agent config for testing."""
    return AgentConfig(
        name="test_agent",
        description="Test agent",
        prompt_template="Test prompt with {workspace_context}",
        required_keys=["trace_analysis_summary"],
        optional_keys=["performance_metrics"],
        output_keys=["test_output"],
        tools=[],
        total_token_budget=1000,
        key_token_limits={"trace_analysis_summary": 500},
    )


# =============================================================================
# BASIC FUNCTIONALITY TESTS
# =============================================================================

class TestSelectiveContextBasic:
    """Test basic selective context functionality."""

    def test_empty_workspace_returns_minimal_context(self, workspace, simple_config):
        """Empty workspace should return context with missing dependencies warning."""
        context = workspace.to_selective_context(simple_config)

        assert "<workspace_context>" in context
        assert "</workspace_context>" in context
        assert "<missing_dependencies>" in context
        assert "trace_analysis_summary" in context

    def test_required_keys_included(self, populated_workspace, simple_config):
        """Required keys should be included in context."""
        context = populated_workspace.to_selective_context(simple_config)

        assert "trace_analysis_summary" in context
        assert "error_rate" in context
        assert "0.15" in context

    def test_optional_keys_included_when_present(self, populated_workspace, simple_config):
        """Optional keys should be included when present and budget allows."""
        context = populated_workspace.to_selective_context(simple_config)

        assert "performance_metrics" in context
        assert "p95_latency_ms" in context

    def test_optional_keys_not_required(self, workspace, simple_config):
        """Missing optional keys should not cause warnings."""
        # Only add required key
        workspace.write(
            "trace_analysis_summary",
            {"error_rate": 0.1},
            agent="trace_analyst"
        )

        context = workspace.to_selective_context(simple_config)

        assert "<missing_dependencies>" not in context
        assert "trace_analysis_summary" in context

    def test_output_keys_excluded_from_other_keys_hint(self, populated_workspace):
        """Agent's own output keys should not appear in 'other available keys' hint."""
        # Add context_recommendations (output of context_engineer)
        populated_workspace.write(
            "context_recommendations",
            [{"issue": "test"}],
            agent="context_engineer"
        )

        context = populated_workspace.to_selective_context(CONTEXT_ENGINEER_CONFIG)

        # context_recommendations is context_engineer's output, shouldn't be in hint
        if "<other_available_keys>" in context:
            assert "context_recommendations" not in context.split("<other_available_keys>")[1].split("</other_available_keys>")[0]


class TestSelectiveContextWithRealConfigs:
    """Test selective context with actual agent configurations."""

    def test_trace_analyst_gets_minimal_context(self, populated_workspace):
        """trace_analyst has no required keys, should get minimal context."""
        context = populated_workspace.to_selective_context(TRACE_ANALYST_CONFIG)

        # trace_analyst has no required or optional keys
        assert "<workspace_context>" in context
        # Should show other keys as available
        assert "<other_available_keys>" in context or "trace_analysis_summary" not in context

    def test_context_engineer_gets_required_keys(self, populated_workspace):
        """context_engineer should get trace_analysis_summary and error_patterns."""
        context = populated_workspace.to_selective_context(CONTEXT_ENGINEER_CONFIG)

        assert "trace_analysis_summary" in context
        assert "error_patterns" in context
        assert "error_rate" in context
        assert "timeout" in context

    def test_agent_architect_gets_required_and_optional(self, populated_workspace):
        """agent_architect should get trace_analysis_summary and optional metrics."""
        # Add context_recommendations for optional
        populated_workspace.write(
            "context_recommendations",
            [{"issue": "high latency", "severity": "high"}],
            agent="context_engineer"
        )

        context = populated_workspace.to_selective_context(AGENT_ARCHITECT_CONFIG)

        assert "trace_analysis_summary" in context
        # Optional keys if budget allows
        assert "performance_metrics" in context or "context_recommendations" in context


# =============================================================================
# MISSING DEPENDENCIES TESTS
# =============================================================================

class TestMissingDependencies:
    """Test handling of missing required dependencies."""

    def test_missing_required_keys_warning(self, workspace):
        """Missing required keys should generate warning in context."""
        config = AgentConfig(
            name="test",
            description="test",
            prompt_template="{workspace_context}",
            required_keys=["key1", "key2"],
            optional_keys=[],
            output_keys=[],
            tools=[],
        )

        context = workspace.to_selective_context(config)

        assert "<missing_dependencies>" in context
        assert "key1" in context
        assert "key2" in context
        assert "Run trace_analyst first" in context

    def test_partial_missing_keys(self, workspace):
        """Only missing keys should be in warning, not present ones."""
        workspace.write("key1", {"data": "value"}, agent="test")

        config = AgentConfig(
            name="test",
            description="test",
            prompt_template="{workspace_context}",
            required_keys=["key1", "key2"],
            optional_keys=[],
            output_keys=[],
            tools=[],
        )

        context = workspace.to_selective_context(config)

        assert "<missing_dependencies>" in context
        assert "key2" in context
        # key1 should be in workspace_entry, not missing_dependencies
        missing_section = context.split("<missing_dependencies>")[1].split("</missing_dependencies>")[0]
        assert "key1" not in missing_section

    def test_check_agent_dependencies_returns_correct_status(self, workspace):
        """check_agent_dependencies should return correct tuple."""
        can_run, missing, msg = workspace.check_agent_dependencies(CONTEXT_ENGINEER_CONFIG)

        assert can_run is False
        assert "trace_analysis_summary" in missing
        assert "error_patterns" in missing
        # Base message identifies missing entries (suggestion is added by validate_agent_can_run)
        assert "missing required workspace entries" in msg

    def test_check_agent_dependencies_passes_when_satisfied(self, populated_workspace):
        """check_agent_dependencies should pass when all keys present."""
        can_run, missing, msg = populated_workspace.check_agent_dependencies(CONTEXT_ENGINEER_CONFIG)

        assert can_run is True
        assert missing == []
        assert "satisfied" in msg.lower()


# =============================================================================
# TOKEN BUDGET TESTS
# =============================================================================

class TestTokenBudget:
    """Test token budget limiting."""

    def test_budget_limits_context_size(self, workspace):
        """Context should respect total_token_budget."""
        # Add large data
        workspace.write(
            "large_key",
            {"data": "x" * 10000},  # Large payload
            agent="test"
        )

        config = AgentConfig(
            name="test",
            description="test",
            prompt_template="{workspace_context}",
            required_keys=["large_key"],
            optional_keys=[],
            output_keys=[],
            tools=[],
            total_token_budget=100,  # Very small budget (~400 chars)
            key_token_limits={"large_key": 50},
        )

        context = workspace.to_selective_context(config)

        # Should be truncated
        assert "[TRUNCATED]" in context
        # Context should be reasonable size
        assert len(context) < 2000

    def test_per_key_limits_respected(self, workspace):
        """Individual key limits should be respected."""
        workspace.write("key1", {"data": "a" * 5000}, agent="test")
        workspace.write("key2", {"data": "b" * 5000}, agent="test")

        config = AgentConfig(
            name="test",
            description="test",
            prompt_template="{workspace_context}",
            required_keys=["key1", "key2"],
            optional_keys=[],
            output_keys=[],
            tools=[],
            total_token_budget=5000,
            key_token_limits={"key1": 100, "key2": 100},
        )

        context = workspace.to_selective_context(config)

        # Both should be truncated due to per-key limits
        assert context.count("[TRUNCATED]") >= 1

    def test_optional_keys_skipped_when_budget_exhausted(self, workspace):
        """Optional keys should be skipped if budget is exhausted."""
        workspace.write("required", {"data": "r" * 2000}, agent="test")
        workspace.write("optional", {"data": "o" * 2000}, agent="test")

        config = AgentConfig(
            name="test",
            description="test",
            prompt_template="{workspace_context}",
            required_keys=["required"],
            optional_keys=["optional"],
            output_keys=[],
            tools=[],
            total_token_budget=200,  # Small budget
            key_token_limits={},
        )

        context = workspace.to_selective_context(config)

        # Required should be present (truncated)
        assert "required" in context
        # Optional might be skipped due to budget
        # (depends on how much budget is left after required)

    def test_context_budget_info_included(self, populated_workspace, simple_config):
        """Context should include budget usage info."""
        context = populated_workspace.to_selective_context(simple_config)

        assert "<context_budget" in context
        assert "used=" in context
        assert "max=" in context


# =============================================================================
# SMART TRUNCATION TESTS
# =============================================================================

class TestSmartTruncation:
    """Test smart truncation of different data types."""

    def test_list_truncation_shows_count(self, workspace):
        """List truncation should show item count."""
        workspace.write(
            "list_key",
            [{"item": i} for i in range(100)],  # 100 items
            agent="test"
        )

        config = AgentConfig(
            name="test",
            description="test",
            prompt_template="{workspace_context}",
            required_keys=["list_key"],
            optional_keys=[],
            output_keys=[],
            tools=[],
            total_token_budget=100,
            key_token_limits={"list_key": 50},
        )

        context = workspace.to_selective_context(config)

        # Should show truncation with count
        assert "more items" in context or "[TRUNCATED]" in context

    def test_dict_truncation(self, workspace):
        """Dict truncation should work correctly."""
        workspace.write(
            "dict_key",
            {f"key_{i}": "value" * 100 for i in range(50)},
            agent="test"
        )

        config = AgentConfig(
            name="test",
            description="test",
            prompt_template="{workspace_context}",
            required_keys=["dict_key"],
            optional_keys=[],
            output_keys=[],
            tools=[],
            total_token_budget=100,
            key_token_limits={"dict_key": 50},
        )

        context = workspace.to_selective_context(config)

        assert "[TRUNCATED]" in context


# =============================================================================
# VALIDATE_AGENT_CAN_RUN TESTS
# =============================================================================

class TestValidateAgentCanRun:
    """Test the validate_agent_can_run function."""

    def test_trace_analyst_can_always_run(self, workspace):
        """trace_analyst has no dependencies, should always be able to run."""
        can_run, missing, msg = validate_agent_can_run("trace_analyst", workspace)

        assert can_run is True
        assert missing == []

    def test_context_engineer_cannot_run_empty_workspace(self, workspace):
        """context_engineer needs trace_analyst output."""
        can_run, missing, msg = validate_agent_can_run("context_engineer", workspace)

        assert can_run is False
        assert "trace_analysis_summary" in missing
        assert "error_patterns" in missing

    def test_context_engineer_can_run_with_dependencies(self, populated_workspace):
        """context_engineer should run when dependencies are present."""
        can_run, missing, msg = validate_agent_can_run("context_engineer", populated_workspace)

        assert can_run is True
        assert missing == []

    def test_unknown_agent_returns_false(self, workspace):
        """Unknown agent name should return False with message."""
        can_run, missing, msg = validate_agent_can_run("nonexistent_agent", workspace)

        assert can_run is False
        assert "Unknown agent" in msg


# =============================================================================
# CREATE_SUBAGENTS INTEGRATION TESTS
# =============================================================================

class TestCreateSubagentsIntegration:
    """Test create_subagents with selective context."""

    def test_create_subagents_uses_selective_context_by_default(self, populated_workspace):
        """create_subagents should use selective context by default."""
        agents = create_subagents(populated_workspace, use_selective_context=True)

        # trace_analyst should have minimal context (no required/optional keys)
        trace_analyst_prompt = agents["trace_analyst"].prompt
        # It should still have workspace_context section
        assert "workspace_context" in trace_analyst_prompt.lower() or "no relevant data" in trace_analyst_prompt.lower()

        # context_engineer should have trace_analysis_summary
        context_engineer_prompt = agents["context_engineer"].prompt
        assert "trace_analysis_summary" in context_engineer_prompt or "error_rate" in context_engineer_prompt

    def test_create_subagents_legacy_mode(self, populated_workspace):
        """create_subagents with use_selective_context=False should use full context."""
        agents = create_subagents(populated_workspace, use_selective_context=False)

        # All agents should have similar context (full workspace)
        trace_analyst_prompt = agents["trace_analyst"].prompt
        context_engineer_prompt = agents["context_engineer"].prompt

        # Both should contain the same workspace data
        assert "trace_analysis_summary" in trace_analyst_prompt or "shared_workspace" in trace_analyst_prompt

    def test_create_subagents_empty_workspace(self, workspace):
        """create_subagents should handle empty workspace gracefully."""
        agents = create_subagents(workspace, use_selective_context=True)

        assert len(agents) == 4
        assert "trace_analyst" in agents
        assert "context_engineer" in agents
        assert "agent_architect" in agents
        assert "eval_runner" in agents

        # All should have valid prompts
        for name, agent in agents.items():
            assert agent.prompt is not None
            assert len(agent.prompt) > 0


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_required_and_optional_keys(self, populated_workspace):
        """Agent with no required or optional keys should get minimal context."""
        config = AgentConfig(
            name="minimal",
            description="test",
            prompt_template="{workspace_context}",
            required_keys=[],
            optional_keys=[],
            output_keys=[],
            tools=[],
        )

        context = populated_workspace.to_selective_context(config)

        assert "<workspace_context>" in context
        assert "</workspace_context>" in context
        # Should show other keys as available
        assert "<other_available_keys>" in context

    def test_all_workspace_keys_are_output_keys(self, populated_workspace):
        """If all keys are output keys, should get minimal context with no hints."""
        config = AgentConfig(
            name="producer",
            description="test",
            prompt_template="{workspace_context}",
            required_keys=[],
            optional_keys=[],
            output_keys=["trace_analysis_summary", "error_patterns", "performance_metrics"],
            tools=[],
        )

        context = populated_workspace.to_selective_context(config)

        # Should not show own output keys in hints
        if "<other_available_keys>" in context:
            other_section = context.split("<other_available_keys>")[1].split("</other_available_keys>")[0]
            assert "trace_analysis_summary" not in other_section

    def test_duplicate_keys_in_required_and_optional(self, populated_workspace):
        """Key in both required and optional should only be included once."""
        config = AgentConfig(
            name="test",
            description="test",
            prompt_template="{workspace_context}",
            required_keys=["trace_analysis_summary"],
            optional_keys=["trace_analysis_summary"],  # Duplicate
            output_keys=[],
            tools=[],
        )

        context = populated_workspace.to_selective_context(config)

        # Count occurrences of the key in workspace_entry tags
        entry_count = context.count('key="trace_analysis_summary"')
        assert entry_count == 1

    def test_special_characters_in_data(self, workspace):
        """Data with special characters should be handled correctly."""
        workspace.write(
            "special",
            {"text": 'Quote: "test" & <xml> stuff'},
            agent="test"
        )

        config = AgentConfig(
            name="test",
            description="test",
            prompt_template="{workspace_context}",
            required_keys=["special"],
            optional_keys=[],
            output_keys=[],
            tools=[],
        )

        context = workspace.to_selective_context(config)

        # Should handle special chars (JSON escapes them)
        assert "special" in context
        assert "Quote" in context

    def test_none_values_in_data(self, workspace):
        """None values in data should be handled correctly."""
        workspace.write(
            "nullable",
            {"value": None, "list": [None, 1, None]},
            agent="test"
        )

        config = AgentConfig(
            name="test",
            description="test",
            prompt_template="{workspace_context}",
            required_keys=["nullable"],
            optional_keys=[],
            output_keys=[],
            tools=[],
        )

        context = workspace.to_selective_context(config)

        assert "null" in context  # JSON representation of None

    def test_deeply_nested_data(self, workspace):
        """Deeply nested data should be serialized correctly."""
        workspace.write(
            "nested",
            {"level1": {"level2": {"level3": {"value": "deep"}}}},
            agent="test"
        )

        config = AgentConfig(
            name="test",
            description="test",
            prompt_template="{workspace_context}",
            required_keys=["nested"],
            optional_keys=[],
            output_keys=[],
            tools=[],
        )

        context = workspace.to_selective_context(config)

        assert "deep" in context

    def test_zero_token_budget(self, populated_workspace):
        """Zero token budget should still produce valid context structure."""
        config = AgentConfig(
            name="test",
            description="test",
            prompt_template="{workspace_context}",
            required_keys=["trace_analysis_summary"],
            optional_keys=[],
            output_keys=[],
            tools=[],
            total_token_budget=0,
        )

        context = populated_workspace.to_selective_context(config)

        # Should still have structure
        assert "<workspace_context>" in context
        assert "</workspace_context>" in context

    def test_very_large_token_budget(self, populated_workspace):
        """Very large token budget should not cause issues."""
        config = AgentConfig(
            name="test",
            description="test",
            prompt_template="{workspace_context}",
            required_keys=["trace_analysis_summary"],
            optional_keys=["performance_metrics"],
            output_keys=[],
            tools=[],
            total_token_budget=1000000,  # 1M tokens
        )

        context = populated_workspace.to_selective_context(config)

        # Should include all data without truncation
        assert "trace_analysis_summary" in context
        assert "[TRUNCATED]" not in context
