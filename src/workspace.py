"""Shared workspace for inter-agent communication."""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union, TYPE_CHECKING

from pydantic import BaseModel, Field, ValidationError, model_validator

if TYPE_CHECKING:
    from .subagents.registry import AgentConfig

logger = logging.getLogger(__name__)


# =============================================================================
# CACHE INFRASTRUCTURE
# =============================================================================

@dataclass
class CacheEntry:
    """A cached entry with TTL tracking."""
    data: Any
    created_at: float = field(default_factory=time.time)
    ttl_seconds: int = 300  # 5 minutes default

    @property
    def is_expired(self) -> bool:
        """Check if entry has exceeded its TTL."""
        return time.time() - self.created_at > self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return time.time() - self.created_at

    @property
    def remaining_ttl(self) -> float:
        """Get remaining TTL in seconds (0 if expired)."""
        remaining = self.ttl_seconds - self.age_seconds
        return max(0, remaining)


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


class EvalResults(BaseModel):
    """Schema for eval_results workspace entry (from eval_runner sub-agent)."""
    scorer_results: dict[str, float] = Field(default_factory=dict, description="Scorer name -> score")
    pass_rate: float = Field(0.0, ge=0, le=1, description="Fraction of test cases passed")
    failed_cases: list[dict] = Field(default_factory=list, description="Details of failed test cases")
    recommendations: list[str] = Field(default_factory=list, description="Improvement recommendations")
    eval_run_id: Optional[str] = Field(None, description="MLflow run ID for the evaluation")
    executed_at: Optional[str] = Field(None, description="ISO timestamp of execution")

    model_config = {"extra": "allow"}


class GeneratedEvalCode(BaseModel):
    """Schema for generated_eval_code workspace entry.

    Either 'code' (inline Python code) or 'code_path' (path to file) must be provided.
    This allows flexibility for both inline code generation and file-based patterns.
    """
    code: Optional[str] = Field(None, description="Python code for evaluation (inline)")
    code_path: Optional[str] = Field(None, description="Path where code was written")
    scorers: list[str] = Field(default_factory=list, description="Scorers used in evaluation")
    dataset_size: Optional[int] = Field(None, description="Number of test cases")
    description: Optional[str] = Field(None, description="What this evaluation tests")
    # Keep file_path as alias for backward compatibility
    file_path: Optional[str] = Field(None, description="Alias for code_path (deprecated)")

    model_config = {"extra": "allow"}

    @model_validator(mode='after')
    def validate_code_or_path(self) -> 'GeneratedEvalCode':
        """Ensure either code or code_path is provided."""
        # Support file_path as alias for code_path
        effective_path = self.code_path or self.file_path
        if not self.code and not effective_path:
            raise ValueError(
                "Must provide either 'code' (inline Python) or 'code_path' (path to file). "
                "Got neither."
            )
        return self


# Registry mapping workspace keys to Pydantic models
# None means list validation only, model means validate each item
WORKSPACE_SCHEMAS: dict[str, Optional[type[BaseModel]]] = {
    "trace_analysis_summary": TraceAnalysisSummary,
    "error_patterns": ErrorPattern,  # List of ErrorPattern
    "performance_metrics": PerformanceMetrics,
    "context_recommendations": ContextRecommendation,  # List of ContextRecommendation
    "extracted_eval_cases": ExtractedEvalCase,  # List of ExtractedEvalCase
    "quality_issues": None,  # Untyped list
    # Eval loop schemas
    "eval_results": EvalResults,
    "generated_eval_code": GeneratedEvalCode,
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


def get_schema_hint(key: str) -> str:
    """Get a human-readable schema hint for error messages.

    Returns a formatted string showing required and optional fields
    to help users fix validation errors.
    """
    schema = WORKSPACE_SCHEMAS.get(key)
    if schema is None:
        if key in LIST_KEYS:
            return f"Expected: list of objects for '{key}'"
        return f"Expected: dict for '{key}'"

    # Get model fields info
    fields_info = []
    for field_name, field_info in schema.model_fields.items():
        is_required = field_info.is_required()
        field_type = str(field_info.annotation).replace("typing.", "")
        marker = "REQUIRED" if is_required else "optional"
        desc = field_info.description or ""
        fields_info.append(f"  - {field_name}: {field_type} ({marker}) {desc}")

    hint_lines = [f"Schema for '{key}':"]
    if key in LIST_KEYS:
        hint_lines.append("Expected: list of objects, each with:")
    else:
        hint_lines.append("Expected: object with:")
    hint_lines.extend(fields_info)

    return "\n".join(hint_lines)


# =============================================================================
# SHARED WORKSPACE
# =============================================================================

class SharedWorkspace:
    """Shared workspace for inter-agent communication.

    Sub-agents write findings here, other agents can read them.
    Instance-scoped (not global singleton) with schema validation and caching.

    Features:
    - Pydantic schema validation for structured data
    - TTL-based caching to avoid redundant sub-agent work
    - Selective context injection per agent
    - Write history tracking for debugging
    """

    def __init__(
        self,
        max_context_chars: int = 2000,
        default_ttl_seconds: int = 300,
    ):
        self._data: dict[str, Any] = {}
        self._timestamps: dict[str, float] = {}
        self._max_context_chars = max_context_chars
        self._default_ttl = default_ttl_seconds
        self._write_history: list[dict] = []
        # Cache for expensive computations (e.g., trace analysis results)
        self._cache: dict[str, CacheEntry] = {}

    # =========================================================================
    # CACHING METHODS
    # =========================================================================

    def get_cached(self, key: str) -> Optional[Any]:
        """Get a cached value if it exists and hasn't expired.

        Returns None if key not in cache or TTL expired.
        """
        entry = self._cache.get(key)
        if entry is None:
            return None
        if entry.is_expired:
            del self._cache[key]
            logger.debug(f"Cache expired for key: {key}")
            return None
        logger.debug(f"Cache hit for key: {key} (age: {entry.age_seconds:.1f}s)")
        return entry.data

    def set_cached(
        self,
        key: str,
        data: Any,
        ttl_seconds: Optional[int] = None
    ) -> None:
        """Set a cached value with TTL.

        Args:
            key: Cache key
            data: Data to cache
            ttl_seconds: TTL in seconds (uses default if not specified)
        """
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        self._cache[key] = CacheEntry(data=data, ttl_seconds=ttl)
        logger.debug(f"Cached key: {key} (ttl: {ttl}s)")

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], Any],
        ttl_seconds: Optional[int] = None
    ) -> Any:
        """Get cached value or compute and cache it.

        This is the primary caching interface for expensive operations.

        Args:
            key: Cache key
            compute_fn: Function to compute the value if not cached
            ttl_seconds: TTL in seconds (uses default if not specified)

        Returns:
            Cached or computed value
        """
        cached = self.get_cached(key)
        if cached is not None:
            return cached

        logger.info(f"Cache miss for key: {key}, computing...")
        result = compute_fn()
        self.set_cached(key, result, ttl_seconds)
        return result

    def invalidate_cache(self, key: Optional[str] = None) -> int:
        """Invalidate cached entries.

        Args:
            key: Specific key to invalidate, or None to clear all cache

        Returns:
            Number of entries invalidated
        """
        if key is not None:
            if key in self._cache:
                del self._cache[key]
                logger.debug(f"Invalidated cache key: {key}")
                return 1
            return 0

        count = len(self._cache)
        self._cache.clear()
        logger.debug(f"Invalidated all {count} cache entries")
        return count

    def cache_stats(self) -> dict:
        """Get cache statistics."""
        valid_entries = {k: v for k, v in self._cache.items() if not v.is_expired}
        expired_entries = {k: v for k, v in self._cache.items() if v.is_expired}
        return {
            "total_entries": len(self._cache),
            "valid_entries": len(valid_entries),
            "expired_entries": len(expired_entries),
            "keys": list(valid_entries.keys()),
        }

    # =========================================================================
    # WORKSPACE METHODS
    # =========================================================================

    def write(self, key: str, data: Any, agent: str = "unknown") -> tuple[bool, str]:
        """Write data to workspace with validation. Returns (success, message).

        Data can be a dict/list or a JSON string. JSON strings will be parsed
        and validated against the schema for the key.

        On validation failure, returns helpful error message with schema hint.
        """
        is_valid, parsed_data, msg = validate_workspace_entry(key, data)
        if not is_valid:
            # Include schema hint to help fix validation errors
            schema_hint = get_schema_hint(key)
            enhanced_msg = f"[Workspace] Validation error: {msg}\n\n{schema_hint}"
            logger.warning(f"Workspace validation failed for '{key}': {msg}")
            return False, enhanced_msg

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

    def clear(self, clear_cache: bool = True):
        """Clear all workspace data.

        Args:
            clear_cache: Also clear the computation cache (default True)
        """
        self._data.clear()
        self._timestamps.clear()
        self._write_history.clear()
        if clear_cache:
            self._cache.clear()

    # =========================================================================
    # SELECTIVE CONTEXT INJECTION
    # =========================================================================

    def compute_dynamic_budget(self, config: "AgentConfig") -> int:
        """Compute dynamic token budget based on workspace content size.

        Adjusts the agent's token budget based on how much data is in the workspace.
        When workspace is sparse, use full budget. When dense, compress to essentials.

        Args:
            config: AgentConfig with total_token_budget

        Returns:
            Adjusted token budget
        """
        if not self._data:
            return config.total_token_budget

        # Calculate total workspace content size
        total_chars = sum(
            len(json.dumps(entry["data"], default=str))
            for entry in self._data.values()
        )

        base_budget = config.total_token_budget

        # Scale budget based on content density
        if total_chars < 2000:
            # Sparse workspace - use full budget
            return base_budget
        elif total_chars < 10000:
            # Moderate workspace - 85% budget
            return int(base_budget * 0.85)
        elif total_chars < 50000:
            # Dense workspace - 70% budget, focus on essentials
            return int(base_budget * 0.70)
        else:
            # Very dense workspace - 50% budget, aggressive compression
            return int(base_budget * 0.50)

    def summarize_entry(self, key: str, max_chars: int = 500) -> Optional[str]:
        """Generate an intelligent summary of a workspace entry.

        Instead of truncating data, this creates a meaningful summary that
        preserves key information while reducing token usage.

        Args:
            key: Workspace entry key
            max_chars: Maximum characters for the summary

        Returns:
            Summary string or None if key doesn't exist
        """
        entry = self._data.get(key)
        if not entry:
            return None

        data = entry["data"]

        # Handle different data types
        if isinstance(data, list):
            return self._summarize_list(data, max_chars)
        elif isinstance(data, dict):
            return self._summarize_dict(data, max_chars)
        else:
            return str(data)[:max_chars]

    def _summarize_list(self, data: list, max_chars: int) -> str:
        """Summarize a list with item counts and samples."""
        if not data:
            return "Empty list"

        count = len(data)
        if count <= 2:
            return json.dumps(data, indent=2, default=str)[:max_chars]

        # Show first item, last item, and count
        first = json.dumps(data[0], default=str)
        last = json.dumps(data[-1], default=str)

        # Truncate individual items if needed
        item_budget = (max_chars - 100) // 2
        if len(first) > item_budget:
            first = first[:item_budget] + "..."
        if len(last) > item_budget:
            last = last[:item_budget] + "..."

        return f"[{count} items]\nFirst: {first}\nLast: {last}"

    def _summarize_dict(self, data: dict, max_chars: int) -> str:
        """Summarize a dict with key counts and important values."""
        if not data:
            return "{}"

        keys = list(data.keys())

        # Priority keys to always show (common in our schemas)
        priority_keys = [
            "error_rate", "success_rate", "pass_rate", "avg_latency_ms",
            "trace_count", "top_errors", "recommendations", "scorer_results"
        ]

        summary_parts = []
        chars_used = 0

        # Show priority keys first
        for key in priority_keys:
            if key in data and chars_used < max_chars:
                value = data[key]
                value_str = json.dumps(value, default=str)
                if len(value_str) > 100:
                    value_str = value_str[:97] + "..."
                line = f"{key}: {value_str}"
                summary_parts.append(line)
                chars_used += len(line) + 1

        # Add remaining keys summary
        remaining = [k for k in keys if k not in priority_keys]
        if remaining and chars_used < max_chars:
            summary_parts.append(f"... +{len(remaining)} more keys: {remaining[:5]}")

        return "\n".join(summary_parts)

    def to_selective_context(self, config: "AgentConfig", use_dynamic_budget: bool = True) -> str:
        """Generate context with only relevant keys for a specific agent.

        Uses the agent's configuration to determine which workspace keys to include:
        - required_keys: Must be present (warns if missing)
        - optional_keys: Included if available and budget allows
        - output_keys: Excluded (agent's own outputs shouldn't be in its context)

        Args:
            config: AgentConfig with required_keys, optional_keys, output_keys,
                   total_token_budget, and key_token_limits
            use_dynamic_budget: If True, adjust budget based on workspace size

        Returns:
            XML-formatted context string with only relevant entries
        """
        parts = ["<workspace_context>"]
        chars_used = 0

        # Use dynamic budget if enabled
        if use_dynamic_budget:
            budget = self.compute_dynamic_budget(config)
        else:
            budget = config.total_token_budget

        max_chars = budget * 4  # ~4 chars per token estimate

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

        # Return just the base message - the caller (validate_agent_can_run in __init__.py)
        # will add suggestions about which agent to run first using _get_dependency_producers
        message = f"Cannot run {config.name}: missing required workspace entries {missing}."

        return False, missing, message
