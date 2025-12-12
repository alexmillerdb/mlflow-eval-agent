"""Shared workspace for inter-agent communication."""

import json
import logging
import time
from typing import Any, Optional, Union, TYPE_CHECKING

from pydantic import BaseModel, Field, ValidationError

if TYPE_CHECKING:
    from .subagents.registry import AgentConfig

logger = logging.getLogger(__name__)


# =============================================================================
# WORKSPACE SCHEMAS (Pydantic Models)
# =============================================================================

class TraceAnalysisSummary(BaseModel):
    """Schema for trace_analysis_summary workspace entry."""
    error_rate: Optional[float] = Field(None, ge=0, le=1, description="Fraction of traces with errors")
    success_rate: Optional[float] = Field(None, ge=0, le=1, description="Fraction of successful traces")
    trace_count: Optional[int] = Field(None, ge=0, description="Total traces analyzed")
    top_errors: list[str] = Field(default_factory=list, description="Most common error types")
    avg_latency_ms: Optional[float] = Field(None, description="Average latency in ms")
    p95_latency_ms: Optional[float] = Field(None, description="95th percentile latency")
    analyzed_at: Optional[str] = Field(None, description="ISO timestamp of analysis")

    # Allow extra fields for flexibility
    model_config = {"extra": "allow"}


class ErrorPattern(BaseModel):
    """Schema for individual error pattern."""
    error_type: str = Field(..., description="Type/category of error")
    count: int = Field(0, ge=0, description="Number of occurrences")
    example_trace_ids: list[str] = Field(default_factory=list, description="Example trace IDs")
    description: Optional[str] = Field(None, description="Error description")

    model_config = {"extra": "allow"}


class PerformanceMetrics(BaseModel):
    """Schema for performance_metrics workspace entry."""
    avg_latency_ms: Optional[float] = Field(None, description="Average latency")
    p50_latency_ms: Optional[float] = Field(None, description="Median latency")
    p95_latency_ms: Optional[float] = Field(None, description="95th percentile latency")
    p99_latency_ms: Optional[float] = Field(None, description="99th percentile latency")
    bottleneck_component: Optional[str] = Field(None, description="Component causing bottleneck")
    bottleneck_percentage: Optional[float] = Field(None, ge=0, le=100, description="Percentage of time in bottleneck")

    model_config = {"extra": "allow"}


class ContextRecommendation(BaseModel):
    """Schema for individual context recommendation."""
    issue: str = Field(..., description="Issue identified")
    severity: str = Field("medium", description="Severity: high, medium, low")
    current_state: Optional[str] = Field(None, description="Current state description")
    recommended_change: str = Field(..., description="Recommended change")
    expected_impact: Optional[str] = Field(None, description="Expected impact of change")

    model_config = {"extra": "allow"}


class ExtractedEvalCase(BaseModel):
    """Schema for extracted evaluation case."""
    trace_id: str = Field(..., description="Source trace ID")
    category: Optional[str] = Field(None, description="Category: error, success, edge_case")
    inputs: Optional[dict] = Field(None, description="Input data")
    expected_output: Optional[str] = Field(None, description="Expected output")
    rationale: Optional[str] = Field(None, description="Why this was selected")

    model_config = {"extra": "allow"}


# Registry mapping workspace keys to Pydantic models
# None means list validation only, model means validate each item
WORKSPACE_SCHEMAS: dict[str, Optional[type[BaseModel]]] = {
    "trace_analysis_summary": TraceAnalysisSummary,
    "error_patterns": ErrorPattern,  # List of ErrorPattern
    "performance_metrics": PerformanceMetrics,
    "context_recommendations": ContextRecommendation,  # List of ContextRecommendation
    "extracted_eval_cases": ExtractedEvalCase,  # List of ExtractedEvalCase
    "quality_issues": None,  # Untyped list
}

# Keys that expect lists
LIST_KEYS = {"error_patterns", "context_recommendations", "extracted_eval_cases", "quality_issues"}


def validate_workspace_entry(key: str, data: Any) -> tuple[bool, Any, str]:
    """Validate and parse workspace entry against schema.

    Returns (is_valid, parsed_data, message).
    parsed_data is the validated/coerced data if valid, or original data if no schema.
    """
    # Handle JSON string input
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            return False, data, f"Invalid JSON for key '{key}': {e}"

    # No schema defined - accept as-is
    if key not in WORKSPACE_SCHEMAS:
        return True, data, f"Warning: No schema defined for key '{key}'"

    schema = WORKSPACE_SCHEMAS[key]

    # List keys
    if key in LIST_KEYS:
        if not isinstance(data, list):
            return False, data, f"Expected list for key '{key}', got {type(data).__name__}"

        # Validate each item if schema exists
        if schema is not None:
            validated_items = []
            for i, item in enumerate(data):
                try:
                    validated = schema.model_validate(item)
                    validated_items.append(validated.model_dump())
                except ValidationError as e:
                    return False, data, f"Item {i} in '{key}' failed validation: {e}"
            return True, validated_items, "Valid"

        return True, data, "Valid"

    # Single object keys
    if schema is not None:
        try:
            validated = schema.model_validate(data)
            return True, validated.model_dump(), "Valid"
        except ValidationError as e:
            return False, data, f"Validation failed for '{key}': {e}"

    # No schema, just type check
    if not isinstance(data, dict):
        return False, data, f"Expected dict for key '{key}', got {type(data).__name__}"

    return True, data, "Valid"


def get_schema_json(key: str) -> Optional[dict]:
    """Get JSON schema for a workspace key (for prompt injection)."""
    if key not in WORKSPACE_SCHEMAS:
        return None

    schema = WORKSPACE_SCHEMAS[key]
    if schema is None:
        return {"type": "array", "items": {}}

    if key in LIST_KEYS:
        return {
            "type": "array",
            "items": schema.model_json_schema()
        }

    return schema.model_json_schema()


# =============================================================================
# SHARED WORKSPACE
# =============================================================================

class SharedWorkspace:
    """Shared workspace for inter-agent communication.

    Sub-agents write findings here, other agents can read them.
    Instance-scoped (not global singleton) with schema validation.
    """

    def __init__(self, max_context_chars: int = 2000):
        self._data: dict[str, Any] = {}
        self._timestamps: dict[str, float] = {}
        self._max_context_chars = max_context_chars
        self._write_history: list[dict] = []

    def write(self, key: str, data: Any, agent: str = "unknown") -> tuple[bool, str]:
        """Write data to workspace with validation. Returns (success, message).

        Data can be a dict/list or a JSON string. JSON strings will be parsed
        and validated against the schema for the key.
        """
        is_valid, parsed_data, msg = validate_workspace_entry(key, data)
        if not is_valid:
            logger.warning(f"Workspace validation failed: {msg}")
            return False, msg

        # Store the validated/parsed data
        self._data[key] = {
            "data": parsed_data,
            "written_by": agent,
            "timestamp": time.time(),
        }
        self._timestamps[key] = time.time()
        self._write_history.append({"key": key, "agent": agent, "timestamp": time.time()})

        logger.info(f"Workspace: {agent} wrote '{key}'")
        return True, f"Successfully wrote to workspace: {key}"

    def read(self, key: str) -> Optional[Any]:
        """Read data from shared workspace."""
        entry = self._data.get(key)
        return entry["data"] if entry else None

    def read_if_fresh(self, key: str, max_age_seconds: int = 300) -> Optional[Any]:
        """Read data only if recent enough."""
        entry = self._data.get(key)
        if not entry:
            return None
        if time.time() - entry["timestamp"] > max_age_seconds:
            return None
        return entry["data"]

    def list_keys(self) -> list[str]:
        """List all keys in workspace."""
        return list(self._data.keys())

    def has_required_dependencies(self, required_keys: list[str]) -> tuple[bool, list[str]]:
        """Check if required dependencies exist. Returns (all_present, missing_keys)."""
        missing = [k for k in required_keys if k not in self._data]
        return len(missing) == 0, missing

    def get_timing_metrics(self) -> dict:
        """Get timing metrics for inter-agent coordination."""
        if not self._write_history:
            return {"total_writes": 0}

        return {
            "total_writes": len(self._write_history),
            "agents_involved": list(set(w["agent"] for w in self._write_history)),
            "first_write": self._write_history[0]["timestamp"],
            "last_write": self._write_history[-1]["timestamp"],
            "duration_seconds": self._write_history[-1]["timestamp"] - self._write_history[0]["timestamp"],
        }

    def to_context_string(self) -> str:
        """Convert workspace to context string for prompt injection."""
        if not self._data:
            return ""

        parts = ["<shared_workspace>"]
        truncated_keys = []

        for key, entry in self._data.items():
            serialized = json.dumps(entry['data'], indent=2, default=str)
            if len(serialized) > self._max_context_chars:
                serialized = serialized[:self._max_context_chars] + "... [TRUNCATED]"
                truncated_keys.append(key)

            parts.append(f"""
<workspace_entry key="{key}" written_by="{entry['written_by']}">
{serialized}
</workspace_entry>
""")

        if truncated_keys:
            parts.append(f"""
<truncation_warning>
The following entries were truncated due to size: {', '.join(truncated_keys)}
Use read_from_workspace tool to get full content.
</truncation_warning>
""")

        parts.append("</shared_workspace>")
        return "\n".join(parts)

    def clear(self):
        """Clear all workspace data."""
        self._data.clear()
        self._timestamps.clear()
        self._write_history.clear()

    # =========================================================================
    # SELECTIVE CONTEXT INJECTION
    # =========================================================================

    def to_selective_context(self, config: "AgentConfig") -> str:
        """Generate context with only relevant keys for a specific agent.

        Uses the agent's configuration to determine which workspace keys to include:
        - required_keys: Must be present (warns if missing)
        - optional_keys: Included if available and budget allows
        - output_keys: Excluded (agent's own outputs shouldn't be in its context)

        Args:
            config: AgentConfig with required_keys, optional_keys, output_keys,
                   total_token_budget, and key_token_limits

        Returns:
            XML-formatted context string with only relevant entries
        """
        parts = ["<workspace_context>"]
        chars_used = 0
        max_chars = config.total_token_budget * 4  # ~4 chars per token estimate

        missing_required = []
        included_keys = []

        # Process required keys first (must be present)
        for key in config.required_keys:
            entry = self._data.get(key)
            if not entry:
                missing_required.append(key)
                continue

            key_limit = config.key_token_limits.get(key, 500) * 4
            serialized = self._serialize_entry(key, entry, max_chars=key_limit)
            chars_used += len(serialized)
            parts.append(serialized)
            included_keys.append(key)

        # Add missing dependencies warning
        if missing_required:
            parts.append(f"""
<missing_dependencies>
Required workspace entries are missing: {missing_required}
Run trace_analyst first to populate these entries.
</missing_dependencies>
""")

        # Process optional keys if budget allows
        for key in config.optional_keys:
            if chars_used >= max_chars:
                break
            if key not in self._data:
                continue
            if key in included_keys:
                continue  # Already included as required

            entry = self._data[key]
            remaining_budget = max_chars - chars_used
            key_limit = min(
                config.key_token_limits.get(key, 500) * 4,
                remaining_budget
            )

            if key_limit < 100:  # Skip if budget too small to be useful
                continue

            serialized = self._serialize_entry(key, entry, max_chars=key_limit)
            chars_used += len(serialized)
            parts.append(serialized)
            included_keys.append(key)

        # Show hint about other available data (excluding agent's own outputs)
        other_keys = [
            k for k in self._data.keys()
            if k not in included_keys
            and k not in config.output_keys
        ]
        if other_keys:
            parts.append(f"""
<other_available_keys>
Additional workspace entries exist: {other_keys}
Use read_from_workspace tool if needed.
</other_available_keys>
""")

        # Budget usage info for debugging
        if chars_used > 0:
            parts.append(f"""
<context_budget used="{chars_used}" max="{max_chars}" keys="{included_keys}" />
""")

        parts.append("</workspace_context>")
        return "\n".join(parts)

    def _serialize_entry(
        self,
        key: str,
        entry: dict,
        max_chars: int = 2000
    ) -> str:
        """Serialize a single workspace entry with smart truncation.

        Args:
            key: The workspace key name
            entry: The entry dict containing 'data' and 'written_by'
            max_chars: Maximum characters for the serialized data

        Returns:
            XML-formatted entry string
        """
        data = entry["data"]
        agent = entry.get("written_by", "unknown")

        serialized = json.dumps(data, indent=2, default=str)

        if len(serialized) > max_chars:
            # Smart truncation based on data type
            if isinstance(data, list) and len(data) > 2:
                # For lists, show first few items with count
                truncated_data = data[:2]
                serialized = json.dumps(truncated_data, indent=2, default=str)
                serialized += f"\n... and {len(data) - 2} more items [TRUNCATED]"
            elif isinstance(data, dict):
                # For dicts, truncate the JSON string
                serialized = serialized[:max_chars - 50] + "\n... [TRUNCATED]"
            else:
                serialized = serialized[:max_chars - 50] + "\n... [TRUNCATED]"

        return f"""
<workspace_entry key="{key}" written_by="{agent}">
{serialized}
</workspace_entry>
"""

    def check_agent_dependencies(
        self,
        config: "AgentConfig"
    ) -> tuple[bool, list[str], str]:
        """Check if all required dependencies exist for an agent.

        Args:
            config: AgentConfig with required_keys

        Returns:
            Tuple of (can_run, missing_keys, message)
        """
        missing = [k for k in config.required_keys if k not in self._data]

        if not missing:
            return True, [], f"All dependencies satisfied for {config.name}"

        # Determine which agent produces the missing keys
        # This is a hint - in practice you'd look this up from the registry
        message = f"Cannot run {config.name}: missing required workspace entries {missing}."

        if "trace_analysis_summary" in missing or "error_patterns" in missing:
            message += " Run trace_analyst first."

        return False, missing, message
