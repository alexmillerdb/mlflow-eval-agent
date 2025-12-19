"""Tests for workspace Pydantic schema validation."""

import json
import pytest

from src.workspace import (
    SharedWorkspace,
    TraceAnalysisSummary,
    ErrorPattern,
    PerformanceMetrics,
    ContextRecommendation,
    ExtractedEvalCase,
    validate_workspace_entry,
    get_schema_json,
    WORKSPACE_SCHEMAS,
    LIST_KEYS,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def workspace():
    """Create a fresh workspace for each test."""
    return SharedWorkspace(max_context_chars=2000)


# =============================================================================
# SCHEMA VALIDATION TESTS
# =============================================================================

class TestTraceAnalysisSummarySchema:
    """Test TraceAnalysisSummary schema validation."""

    def test_valid_full_data(self, workspace):
        """Full valid data should pass validation."""
        data = {
            "error_rate": 0.05,
            "success_rate": 0.95,
            "trace_count": 100,
            "top_errors": ["timeout", "validation_error"],
            "avg_latency_ms": 1500.0,
            "p95_latency_ms": 3200.0,
            "analyzed_at": "2024-12-10T10:30:00Z"
        }
        success, msg = workspace.write("trace_analysis_summary", data, "trace_analyst")
        assert success, f"Expected success but got: {msg}"

    def test_valid_minimal_data(self, workspace):
        """Minimal valid data should pass validation."""
        data = {"error_rate": 0.1}
        success, msg = workspace.write("trace_analysis_summary", data, "trace_analyst")
        assert success, f"Expected success but got: {msg}"

    def test_valid_with_extra_fields(self, workspace):
        """Extra fields should be allowed (extra='allow')."""
        data = {
            "error_rate": 0.05,
            "custom_field": "extra_value",
            "another_field": 123
        }
        success, msg = workspace.write("trace_analysis_summary", data, "trace_analyst")
        assert success, f"Expected success but got: {msg}"

    def test_invalid_error_rate_above_1(self, workspace):
        """Error rate above 1 should fail validation."""
        data = {"error_rate": 1.5}
        success, msg = workspace.write("trace_analysis_summary", data, "trace_analyst")
        assert not success
        assert "validation" in msg.lower() or "error" in msg.lower()

    def test_invalid_error_rate_negative(self, workspace):
        """Negative error rate should fail validation."""
        data = {"error_rate": -0.1}
        success, msg = workspace.write("trace_analysis_summary", data, "trace_analyst")
        assert not success

    def test_invalid_trace_count_negative(self, workspace):
        """Negative trace count should fail validation."""
        data = {"trace_count": -5}
        success, msg = workspace.write("trace_analysis_summary", data, "trace_analyst")
        assert not success


class TestErrorPatternSchema:
    """Test ErrorPattern schema validation (list key)."""

    def test_valid_error_patterns_list(self, workspace):
        """Valid list of error patterns should pass."""
        data = [
            {
                "error_type": "timeout",
                "count": 15,
                "example_trace_ids": ["tr-abc123", "tr-def456"],
                "description": "Request exceeded 30s timeout"
            },
            {
                "error_type": "validation_error",
                "count": 8,
                "example_trace_ids": ["tr-xyz789"],
                "description": "Input validation failed"
            }
        ]
        success, msg = workspace.write("error_patterns", data, "trace_analyst")
        assert success, f"Expected success but got: {msg}"

    def test_valid_minimal_error_pattern(self, workspace):
        """Minimal error pattern should pass."""
        data = [{"error_type": "timeout"}]
        success, msg = workspace.write("error_patterns", data, "trace_analyst")
        assert success, f"Expected success but got: {msg}"

    def test_empty_list_valid(self, workspace):
        """Empty list should be valid."""
        data = []
        success, msg = workspace.write("error_patterns", data, "trace_analyst")
        assert success, f"Expected success but got: {msg}"

    def test_invalid_not_a_list(self, workspace):
        """Dict instead of list should fail."""
        data = {"error_type": "timeout", "count": 10}
        success, msg = workspace.write("error_patterns", data, "trace_analyst")
        assert not success
        assert "list" in msg.lower()

    def test_invalid_missing_required_field(self, workspace):
        """Missing error_type should fail."""
        data = [{"count": 10, "description": "Some error"}]
        success, msg = workspace.write("error_patterns", data, "trace_analyst")
        assert not success
        assert "error_type" in msg.lower() or "validation" in msg.lower()

    def test_invalid_negative_count(self, workspace):
        """Negative count should fail."""
        data = [{"error_type": "timeout", "count": -5}]
        success, msg = workspace.write("error_patterns", data, "trace_analyst")
        assert not success


class TestPerformanceMetricsSchema:
    """Test PerformanceMetrics schema validation."""

    def test_valid_full_metrics(self, workspace):
        """Full valid metrics should pass."""
        data = {
            "avg_latency_ms": 1500.0,
            "p50_latency_ms": 1200.0,
            "p95_latency_ms": 3200.0,
            "p99_latency_ms": 5000.0,
            "bottleneck_component": "retriever",
            "bottleneck_percentage": 65.0
        }
        success, msg = workspace.write("performance_metrics", data, "trace_analyst")
        assert success, f"Expected success but got: {msg}"

    def test_valid_minimal_metrics(self, workspace):
        """Minimal metrics should pass."""
        data = {"avg_latency_ms": 100.0}
        success, msg = workspace.write("performance_metrics", data, "trace_analyst")
        assert success, f"Expected success but got: {msg}"

    def test_invalid_bottleneck_percentage_over_100(self, workspace):
        """Bottleneck percentage over 100 should fail."""
        data = {"bottleneck_percentage": 150.0}
        success, msg = workspace.write("performance_metrics", data, "trace_analyst")
        assert not success


class TestContextRecommendationSchema:
    """Test ContextRecommendation schema validation (list key)."""

    def test_valid_recommendations(self, workspace):
        """Valid recommendations list should pass."""
        data = [
            {
                "issue": "System prompt lacks explicit format guidelines",
                "severity": "high",
                "current_state": "System prompt only specifies task",
                "recommended_change": "Add JSON format specification",
                "expected_impact": "Reduce parsing errors by ~40%"
            }
        ]
        success, msg = workspace.write("context_recommendations", data, "context_engineer")
        assert success, f"Expected success but got: {msg}"

    def test_valid_minimal_recommendation(self, workspace):
        """Minimal recommendation should pass."""
        data = [{"issue": "Missing guidelines", "recommended_change": "Add guidelines"}]
        success, msg = workspace.write("context_recommendations", data, "context_engineer")
        assert success, f"Expected success but got: {msg}"

    def test_invalid_missing_issue(self, workspace):
        """Missing issue field should fail."""
        data = [{"recommended_change": "Do something"}]
        success, msg = workspace.write("context_recommendations", data, "context_engineer")
        assert not success
        assert "issue" in msg.lower() or "validation" in msg.lower()

    def test_invalid_missing_recommended_change(self, workspace):
        """Missing recommended_change field should fail."""
        data = [{"issue": "Some issue"}]
        success, msg = workspace.write("context_recommendations", data, "context_engineer")
        assert not success


class TestExtractedEvalCaseSchema:
    """Test ExtractedEvalCase schema validation (list key)."""

    def test_valid_eval_cases(self, workspace):
        """Valid eval cases list should pass."""
        data = [
            {
                "trace_id": "tr-abc123",
                "category": "error",
                "inputs": {"query": "example query"},
                "expected_output": None,
                "rationale": "Timeout failure - good test case"
            }
        ]
        success, msg = workspace.write("extracted_eval_cases", data, "trace_analyst")
        assert success, f"Expected success but got: {msg}"

    def test_valid_minimal_eval_case(self, workspace):
        """Minimal eval case should pass."""
        data = [{"trace_id": "tr-123"}]
        success, msg = workspace.write("extracted_eval_cases", data, "trace_analyst")
        assert success, f"Expected success but got: {msg}"

    def test_invalid_missing_trace_id(self, workspace):
        """Missing trace_id should fail."""
        data = [{"category": "error", "rationale": "Some reason"}]
        success, msg = workspace.write("extracted_eval_cases", data, "trace_analyst")
        assert not success


# =============================================================================
# JSON STRING PARSING TESTS
# =============================================================================

class TestJSONStringParsing:
    """Test JSON string input parsing."""

    def test_json_string_parsed_and_validated(self, workspace):
        """JSON string should be parsed and validated."""
        json_str = '{"error_rate": 0.05, "trace_count": 100}'
        success, msg = workspace.write("trace_analysis_summary", json_str, "trace_analyst")
        assert success, f"Expected success but got: {msg}"

        # Verify data was stored correctly
        stored = workspace.read("trace_analysis_summary")
        assert stored["error_rate"] == 0.05
        assert stored["trace_count"] == 100

    def test_json_string_list_parsed(self, workspace):
        """JSON string list should be parsed and validated."""
        json_str = '[{"error_type": "timeout", "count": 10}]'
        success, msg = workspace.write("error_patterns", json_str, "trace_analyst")
        assert success, f"Expected success but got: {msg}"

        stored = workspace.read("error_patterns")
        assert len(stored) == 1
        assert stored[0]["error_type"] == "timeout"

    def test_invalid_json_string_fails(self, workspace):
        """Invalid JSON string should fail with clear error."""
        invalid_json = '{"error_rate": 0.05, missing_quote: "value"}'
        success, msg = workspace.write("trace_analysis_summary", invalid_json, "trace_analyst")
        assert not success
        assert "JSON" in msg or "json" in msg

    def test_json_string_with_schema_violation_fails(self, workspace):
        """Valid JSON but invalid schema should fail."""
        json_str = '{"error_rate": 1.5}'  # Above 1.0 limit
        success, msg = workspace.write("trace_analysis_summary", json_str, "trace_analyst")
        assert not success

    def test_empty_json_string_object(self, workspace):
        """Empty JSON object should pass for optional-only schemas."""
        json_str = '{}'
        success, msg = workspace.write("trace_analysis_summary", json_str, "trace_analyst")
        assert success, f"Expected success but got: {msg}"

    def test_empty_json_string_array(self, workspace):
        """Empty JSON array should pass for list keys."""
        json_str = '[]'
        success, msg = workspace.write("error_patterns", json_str, "trace_analyst")
        assert success, f"Expected success but got: {msg}"


# =============================================================================
# VALIDATE_WORKSPACE_ENTRY FUNCTION TESTS
# =============================================================================

class TestValidateWorkspaceEntry:
    """Test the validate_workspace_entry function directly."""

    def test_returns_parsed_data_on_success(self):
        """Should return parsed/validated data on success."""
        data = {"error_rate": 0.1, "trace_count": 50}
        is_valid, parsed, msg = validate_workspace_entry("trace_analysis_summary", data)

        assert is_valid
        assert parsed["error_rate"] == 0.1
        assert parsed["trace_count"] == 50

    def test_returns_original_data_on_failure(self):
        """Should return original data on validation failure."""
        data = {"error_rate": 5.0}  # Invalid
        is_valid, parsed, msg = validate_workspace_entry("trace_analysis_summary", data)

        assert not is_valid
        assert parsed == data  # Original data returned

    def test_unknown_key_passes_with_warning(self):
        """Unknown key should pass with warning message."""
        data = {"any": "data"}
        is_valid, parsed, msg = validate_workspace_entry("unknown_key", data)

        assert is_valid
        assert "Warning" in msg or "no schema" in msg.lower()

    def test_list_validation_validates_each_item(self):
        """List validation should validate each item individually."""
        data = [
            {"error_type": "timeout", "count": 10},
            {"error_type": "validation", "count": 5}
        ]
        is_valid, parsed, msg = validate_workspace_entry("error_patterns", data)

        assert is_valid
        assert len(parsed) == 2

    def test_list_validation_fails_on_any_invalid_item(self):
        """List validation should fail if any item is invalid."""
        data = [
            {"error_type": "timeout", "count": 10},
            {"error_type": "validation", "count": -1}  # Invalid count
        ]
        is_valid, parsed, msg = validate_workspace_entry("error_patterns", data)

        assert not is_valid
        assert "Item 1" in msg or "item" in msg.lower()


# =============================================================================
# GET_SCHEMA_JSON FUNCTION TESTS
# =============================================================================

class TestGetSchemaJson:
    """Test the get_schema_json function."""

    def test_returns_schema_for_known_key(self):
        """Should return JSON schema for known keys."""
        schema = get_schema_json("trace_analysis_summary")

        assert schema is not None
        assert "properties" in schema
        assert "error_rate" in schema["properties"]

    def test_returns_array_schema_for_list_keys(self):
        """Should return array schema for list keys."""
        schema = get_schema_json("error_patterns")

        assert schema is not None
        assert schema["type"] == "array"
        assert "items" in schema

    def test_returns_none_for_unknown_key(self):
        """Should return None for unknown keys."""
        schema = get_schema_json("unknown_key")
        assert schema is None


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestWorkspaceIntegration:
    """Integration tests for workspace with Pydantic schemas."""

    def test_write_read_roundtrip(self, workspace):
        """Data should survive write/read roundtrip."""
        original = {
            "error_rate": 0.15,
            "success_rate": 0.85,
            "trace_count": 100,
            "top_errors": ["timeout", "validation"],
            "avg_latency_ms": 250.5
        }

        workspace.write("trace_analysis_summary", original, "trace_analyst")
        retrieved = workspace.read("trace_analysis_summary")

        assert retrieved["error_rate"] == 0.15
        assert retrieved["success_rate"] == 0.85
        assert retrieved["trace_count"] == 100
        assert "timeout" in retrieved["top_errors"]

    def test_json_string_write_read_roundtrip(self, workspace):
        """JSON string data should survive write/read roundtrip."""
        json_str = '{"error_rate": 0.2, "trace_count": 50}'

        workspace.write("trace_analysis_summary", json_str, "trace_analyst")
        retrieved = workspace.read("trace_analysis_summary")

        assert retrieved["error_rate"] == 0.2
        assert retrieved["trace_count"] == 50

    def test_multiple_writes_overwrite(self, workspace):
        """Later writes should overwrite earlier ones."""
        workspace.write("trace_analysis_summary", {"error_rate": 0.1}, "agent1")
        workspace.write("trace_analysis_summary", {"error_rate": 0.2}, "agent2")

        retrieved = workspace.read("trace_analysis_summary")
        assert retrieved["error_rate"] == 0.2

    def test_failed_validation_does_not_overwrite(self, workspace):
        """Failed validation should not overwrite existing data."""
        workspace.write("trace_analysis_summary", {"error_rate": 0.1}, "agent1")

        # Try to write invalid data
        success, _ = workspace.write("trace_analysis_summary", {"error_rate": 5.0}, "agent2")
        assert not success

        # Original data should remain
        retrieved = workspace.read("trace_analysis_summary")
        assert retrieved["error_rate"] == 0.1


# =============================================================================
# SCHEMA REGISTRY TESTS
# =============================================================================

class TestSchemaRegistry:
    """Test schema registry configuration."""

    def test_all_list_keys_have_schemas(self):
        """All LIST_KEYS should have corresponding schemas."""
        for key in LIST_KEYS:
            assert key in WORKSPACE_SCHEMAS, f"Missing schema for list key: {key}"

    def test_known_keys_documented(self):
        """Known workspace keys should be in WORKSPACE_SCHEMAS."""
        expected_keys = [
            "trace_analysis_summary",
            "error_patterns",
            "performance_metrics",
            "context_recommendations",
            "extracted_eval_cases",
        ]
        for key in expected_keys:
            assert key in WORKSPACE_SCHEMAS, f"Missing expected key: {key}"
