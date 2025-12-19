"""Unit tests for SharedWorkspace caching and schema validation."""

import time
import pytest

from src.workspace import (
    SharedWorkspace,
    CacheEntry,
    validate_workspace_entry,
    TraceAnalysisSummary,
    EvalResults,
    GeneratedEvalCode,
    WORKSPACE_SCHEMAS,
)


# =============================================================================
# CACHE ENTRY TESTS
# =============================================================================

class TestCacheEntry:
    """Tests for the CacheEntry dataclass."""

    def test_cache_entry_creation(self):
        """Test basic cache entry creation."""
        entry = CacheEntry(data={"key": "value"}, ttl_seconds=60)
        assert entry.data == {"key": "value"}
        assert entry.ttl_seconds == 60
        assert not entry.is_expired

    def test_cache_entry_expiration(self):
        """Test TTL expiration detection."""
        # Create entry with very short TTL
        entry = CacheEntry(data="test", ttl_seconds=0)
        # Wait a tiny bit for time to pass
        time.sleep(0.01)
        # Should be expired after time passes with 0 TTL
        assert entry.is_expired

    def test_cache_entry_remaining_ttl(self):
        """Test remaining TTL calculation."""
        entry = CacheEntry(data="test", ttl_seconds=300)
        # Remaining should be close to TTL
        assert 299 <= entry.remaining_ttl <= 300

    def test_cache_entry_age(self):
        """Test age calculation."""
        entry = CacheEntry(data="test")
        # Age should be near 0 for newly created entry
        assert entry.age_seconds < 1


# =============================================================================
# WORKSPACE CACHING TESTS
# =============================================================================

class TestWorkspaceCaching:
    """Tests for SharedWorkspace caching functionality."""

    def test_set_and_get_cached(self):
        """Test basic cache set and get."""
        workspace = SharedWorkspace()
        workspace.set_cached("test_key", {"data": 123})

        result = workspace.get_cached("test_key")
        assert result == {"data": 123}

    def test_cache_miss(self):
        """Test cache miss returns None."""
        workspace = SharedWorkspace()
        result = workspace.get_cached("nonexistent")
        assert result is None

    def test_cache_expiration(self):
        """Test expired cache entries return None."""
        workspace = SharedWorkspace()
        # Set with 0 TTL (immediately expires)
        workspace.set_cached("expired_key", "data", ttl_seconds=0)

        # Small delay to ensure expiration
        time.sleep(0.01)

        result = workspace.get_cached("expired_key")
        assert result is None

    def test_get_or_compute_cached(self):
        """Test get_or_compute returns cached value."""
        workspace = SharedWorkspace()
        call_count = 0

        def expensive_fn():
            nonlocal call_count
            call_count += 1
            return "computed_value"

        # First call should compute
        result1 = workspace.get_or_compute("compute_key", expensive_fn)
        assert result1 == "computed_value"
        assert call_count == 1

        # Second call should use cache
        result2 = workspace.get_or_compute("compute_key", expensive_fn)
        assert result2 == "computed_value"
        assert call_count == 1  # Still 1, not recomputed

    def test_get_or_compute_recomputes_after_expiry(self):
        """Test get_or_compute recomputes after TTL expires."""
        workspace = SharedWorkspace()
        call_count = 0

        def expensive_fn():
            nonlocal call_count
            call_count += 1
            return f"value_{call_count}"

        # First call with very short TTL
        result1 = workspace.get_or_compute("key", expensive_fn, ttl_seconds=0)
        assert result1 == "value_1"

        # Small delay to ensure expiration
        time.sleep(0.01)

        # Should recompute
        result2 = workspace.get_or_compute("key", expensive_fn, ttl_seconds=0)
        assert result2 == "value_2"
        assert call_count == 2

    def test_invalidate_specific_key(self):
        """Test invalidating a specific cache key."""
        workspace = SharedWorkspace()
        workspace.set_cached("key1", "value1")
        workspace.set_cached("key2", "value2")

        count = workspace.invalidate_cache("key1")
        assert count == 1
        assert workspace.get_cached("key1") is None
        assert workspace.get_cached("key2") == "value2"

    def test_invalidate_all_cache(self):
        """Test invalidating all cache entries."""
        workspace = SharedWorkspace()
        workspace.set_cached("key1", "value1")
        workspace.set_cached("key2", "value2")
        workspace.set_cached("key3", "value3")

        count = workspace.invalidate_cache()
        assert count == 3
        assert workspace.get_cached("key1") is None
        assert workspace.get_cached("key2") is None
        assert workspace.get_cached("key3") is None

    def test_cache_stats(self):
        """Test cache statistics."""
        workspace = SharedWorkspace()
        workspace.set_cached("valid_key", "value", ttl_seconds=300)
        workspace.set_cached("expired_key", "value", ttl_seconds=0)

        stats = workspace.cache_stats()
        assert stats["total_entries"] == 2
        assert "valid_key" in stats["keys"]

    def test_clear_with_cache(self):
        """Test clear() also clears cache."""
        workspace = SharedWorkspace()
        workspace.write("workspace_key", {"data": 1}, "test")
        workspace.set_cached("cache_key", "cached_data")

        workspace.clear(clear_cache=True)

        assert workspace.read("workspace_key") is None
        assert workspace.get_cached("cache_key") is None

    def test_clear_without_cache(self):
        """Test clear() can preserve cache."""
        workspace = SharedWorkspace()
        workspace.write("workspace_key", {"data": 1}, "test")
        workspace.set_cached("cache_key", "cached_data")

        workspace.clear(clear_cache=False)

        assert workspace.read("workspace_key") is None
        assert workspace.get_cached("cache_key") == "cached_data"


# =============================================================================
# SCHEMA VALIDATION TESTS
# =============================================================================

class TestSchemaValidation:
    """Tests for workspace schema validation."""

    def test_trace_analysis_summary_valid(self):
        """Test valid trace_analysis_summary validation."""
        data = {
            "error_rate": 0.05,
            "success_rate": 0.95,
            "trace_count": 100,
            "top_errors": ["timeout"],
            "avg_latency_ms": 1500.0,
        }
        is_valid, parsed, msg = validate_workspace_entry("trace_analysis_summary", data)
        assert is_valid
        assert parsed["error_rate"] == 0.05

    def test_trace_analysis_summary_extra_fields(self):
        """Test trace_analysis_summary allows extra fields."""
        data = {
            "error_rate": 0.05,
            "custom_field": "allowed",
        }
        is_valid, parsed, msg = validate_workspace_entry("trace_analysis_summary", data)
        assert is_valid
        assert parsed["custom_field"] == "allowed"

    def test_eval_results_valid(self):
        """Test valid eval_results validation."""
        data = {
            "scorer_results": {"correctness": 0.9, "safety": 1.0},
            "pass_rate": 0.85,
            "failed_cases": [{"trace_id": "tr-123", "reason": "wrong answer"}],
            "recommendations": ["Add more examples"],
        }
        is_valid, parsed, msg = validate_workspace_entry("eval_results", data)
        assert is_valid
        assert parsed["pass_rate"] == 0.85

    def test_generated_eval_code_valid(self):
        """Test valid generated_eval_code validation."""
        data = {
            "code": "import mlflow\nresults = mlflow.genai.evaluate(...)",
            "scorers": ["Correctness", "Safety"],
            "dataset_size": 10,
        }
        is_valid, parsed, msg = validate_workspace_entry("generated_eval_code", data)
        assert is_valid
        assert "mlflow" in parsed["code"]

    def test_json_string_parsing(self):
        """Test that JSON strings are parsed correctly."""
        json_str = '{"error_rate": 0.1, "trace_count": 50}'
        is_valid, parsed, msg = validate_workspace_entry("trace_analysis_summary", json_str)
        assert is_valid
        assert parsed["error_rate"] == 0.1

    def test_invalid_json_string(self):
        """Test invalid JSON string handling."""
        invalid_json = "not valid json"
        is_valid, parsed, msg = validate_workspace_entry("trace_analysis_summary", invalid_json)
        assert not is_valid
        assert "Invalid JSON" in msg

    def test_schema_registry_includes_new_schemas(self):
        """Test that new schemas are registered."""
        assert "eval_results" in WORKSPACE_SCHEMAS
        assert "generated_eval_code" in WORKSPACE_SCHEMAS
        assert WORKSPACE_SCHEMAS["eval_results"] == EvalResults
        assert WORKSPACE_SCHEMAS["generated_eval_code"] == GeneratedEvalCode


# =============================================================================
# WORKSPACE INTEGRATION TESTS
# =============================================================================

class TestWorkspaceIntegration:
    """Integration tests for workspace features together."""

    def test_write_and_cache_interaction(self):
        """Test that workspace write and cache are independent."""
        workspace = SharedWorkspace()

        # Write to workspace (validated storage)
        workspace.write("trace_analysis_summary", {"error_rate": 0.1}, "test_agent")

        # Cache is separate
        workspace.set_cached("analysis_cache", {"cached": True})

        assert workspace.read("trace_analysis_summary")["error_rate"] == 0.1
        assert workspace.get_cached("analysis_cache")["cached"] == True

    def test_default_ttl_configuration(self):
        """Test default TTL can be configured."""
        workspace = SharedWorkspace(default_ttl_seconds=60)
        workspace.set_cached("key", "value")

        # Entry should use default TTL
        entry = workspace._cache["key"]
        assert entry.ttl_seconds == 60

    def test_custom_ttl_override(self):
        """Test TTL can be overridden per entry."""
        workspace = SharedWorkspace(default_ttl_seconds=60)
        workspace.set_cached("key", "value", ttl_seconds=120)

        entry = workspace._cache["key"]
        assert entry.ttl_seconds == 120


# =============================================================================
# DYNAMIC BUDGET TESTS
# =============================================================================

class TestDynamicBudget:
    """Tests for dynamic token budget computation."""

    def test_empty_workspace_full_budget(self):
        """Empty workspace should return full budget."""
        from src.subagents.registry import AgentConfig

        workspace = SharedWorkspace()
        config = AgentConfig(
            name="test",
            description="test",
            prompt_template="test",
            total_token_budget=2000,
        )

        budget = workspace.compute_dynamic_budget(config)
        assert budget == 2000

    def test_sparse_workspace_full_budget(self):
        """Small workspace content should return full budget."""
        from src.subagents.registry import AgentConfig

        workspace = SharedWorkspace()
        workspace.write("trace_analysis_summary", {"error_rate": 0.1}, "test")

        config = AgentConfig(
            name="test",
            description="test",
            prompt_template="test",
            total_token_budget=2000,
        )

        budget = workspace.compute_dynamic_budget(config)
        assert budget == 2000  # Full budget for sparse data

    def test_moderate_workspace_reduced_budget(self):
        """Moderate workspace should reduce budget to 85%."""
        from src.subagents.registry import AgentConfig

        workspace = SharedWorkspace()
        # Add ~5000 chars of data
        large_data = {"data": "x" * 5000}
        workspace.write("trace_analysis_summary", large_data, "test")

        config = AgentConfig(
            name="test",
            description="test",
            prompt_template="test",
            total_token_budget=2000,
        )

        budget = workspace.compute_dynamic_budget(config)
        assert budget == 1700  # 85% of 2000

    def test_dense_workspace_aggressive_budget(self):
        """Dense workspace should reduce budget significantly."""
        from src.subagents.registry import AgentConfig

        workspace = SharedWorkspace()
        # Add ~60000 chars of data
        large_data = {"data": "x" * 60000}
        workspace.write("trace_analysis_summary", large_data, "test")

        config = AgentConfig(
            name="test",
            description="test",
            prompt_template="test",
            total_token_budget=2000,
        )

        budget = workspace.compute_dynamic_budget(config)
        assert budget == 1000  # 50% of 2000


# =============================================================================
# SUMMARIZATION TESTS
# =============================================================================

class TestSummarization:
    """Tests for workspace entry summarization."""

    def test_summarize_nonexistent_key(self):
        """Summarizing nonexistent key returns None."""
        workspace = SharedWorkspace()
        result = workspace.summarize_entry("nonexistent")
        assert result is None

    def test_summarize_small_list(self):
        """Small lists are returned as-is."""
        workspace = SharedWorkspace()
        # Use valid ErrorPattern schema data
        workspace.write("error_patterns", [
            {"error_type": "timeout", "count": 5, "example_trace_ids": ["tr-1"]}
        ], "test")

        summary = workspace.summarize_entry("error_patterns")
        assert "timeout" in summary

    def test_summarize_large_list(self):
        """Large lists show count and samples."""
        workspace = SharedWorkspace()
        # Create valid error patterns
        large_list = [
            {"error_type": f"error_{i}", "count": i, "example_trace_ids": [f"tr-{i}"]}
            for i in range(100)
        ]
        workspace.write("error_patterns", large_list, "test")

        summary = workspace.summarize_entry("error_patterns")
        assert "[100 items]" in summary
        assert "First:" in summary
        assert "Last:" in summary

    def test_summarize_dict_priority_keys(self):
        """Dict summarization prioritizes important keys."""
        workspace = SharedWorkspace()
        data = {
            "error_rate": 0.05,
            "success_rate": 0.95,
            "trace_count": 100,
            "some_other_key": "value",
            "another_key": "data",
        }
        workspace.write("trace_analysis_summary", data, "test")

        summary = workspace.summarize_entry("trace_analysis_summary")
        assert "error_rate" in summary
        assert "success_rate" in summary
        assert "trace_count" in summary

    def test_summarize_empty_dict_uses_defaults(self):
        """Empty dict gets filled with schema defaults."""
        workspace = SharedWorkspace()
        # TraceAnalysisSummary has optional fields with defaults
        workspace.write("trace_analysis_summary", {}, "test")

        summary = workspace.summarize_entry("trace_analysis_summary")
        # Should have error_rate in the summary (default value)
        assert "error_rate" in summary

    def test_summarize_empty_list(self):
        """Empty list returns descriptive string."""
        workspace = SharedWorkspace()
        workspace.write("error_patterns", [], "test")

        summary = workspace.summarize_entry("error_patterns")
        assert summary == "Empty list"


# =============================================================================
# SELECTIVE CONTEXT WITH DYNAMIC BUDGET TESTS
# =============================================================================

class TestSelectiveContextDynamicBudget:
    """Tests for selective context with dynamic budget integration."""

    def test_selective_context_uses_dynamic_budget_by_default(self):
        """to_selective_context should use dynamic budget by default."""
        from src.subagents.registry import AgentConfig

        workspace = SharedWorkspace()
        # Add moderate data
        workspace.write("trace_analysis_summary", {"data": "x" * 5000}, "test")

        config = AgentConfig(
            name="test",
            description="test",
            prompt_template="test",
            required_keys=["trace_analysis_summary"],
            total_token_budget=2000,
        )

        # This should use dynamic budget (85% with moderate data)
        context = workspace.to_selective_context(config, use_dynamic_budget=True)
        assert "<workspace_context>" in context

    def test_selective_context_can_disable_dynamic_budget(self):
        """Dynamic budget can be disabled."""
        from src.subagents.registry import AgentConfig

        workspace = SharedWorkspace()
        workspace.write("trace_analysis_summary", {"data": "x" * 5000}, "test")

        config = AgentConfig(
            name="test",
            description="test",
            prompt_template="test",
            required_keys=["trace_analysis_summary"],
            total_token_budget=2000,
        )

        # This should use full static budget
        context = workspace.to_selective_context(config, use_dynamic_budget=False)
        assert "<workspace_context>" in context


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
