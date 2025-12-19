"""Tool result compression for coordinator context management.

Intercepts large tool results before they enter message history.
Writes full data to files, returns summaries with file path references.
Agent can use Read tool to access full data when needed.
"""

import json
import logging
import os
import time
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class OutputMode(Enum):
    """Output mode for trace data truncation.

    PREVIEW: Minimal output (500 chars) - fixed small limit
    SUMMARY: Adaptive truncation by span type (good for 1-3 traces)
    AGGRESSIVE: More aggressive truncation (1000 chars max, for 5+ traces)
    FULL: No truncation - complete data
    """
    PREVIEW = "preview"
    SUMMARY = "summary"
    AGGRESSIVE = "aggressive"
    FULL = "full"


def get_output_mode() -> OutputMode:
    """Get output mode from environment variable or default.

    Set TRACE_OUTPUT_MODE to: preview, summary, aggressive, or full
    Default: aggressive (good balance for batch operations)
    """
    mode_str = os.environ.get("TRACE_OUTPUT_MODE", "aggressive").lower()
    try:
        return OutputMode(mode_str)
    except ValueError:
        logger.warning(f"Invalid TRACE_OUTPUT_MODE '{mode_str}', using 'aggressive'")
        return OutputMode.AGGRESSIVE

# Cache directory for tool results (relative to working directory)
CACHE_DIR = Path(".claude/cache/tool_results")

# Threshold for storing in file vs returning inline
LARGE_RESULT_THRESHOLD = 2000  # chars

# Claude Agent SDK has 1MB JSON buffer limit - stay safely under
MAX_INLINE_SIZE = 800_000  # bytes (leave headroom for JSON encoding)


def compress_tool_result(
    tool_name: str,
    result: dict[str, Any],
    trace_id: Optional[str] = None,
    max_inline_chars: int = 500,
    mode: Optional[OutputMode] = None,
) -> dict[str, Any]:
    """Compress large tool results by writing to file and returning summary.

    Small results (< LARGE_RESULT_THRESHOLD) are returned as-is.
    Large results are written to a file and a summary with the file path is returned.

    Args:
        tool_name: Name of the tool that produced the result
        result: Original tool result dict with "content" key
        trace_id: Optional trace ID for file naming
        max_inline_chars: Maximum chars for inline summary (used in PREVIEW mode)
        mode: Output mode - PREVIEW (500 chars), SUMMARY (adaptive), FULL (no limit)
              If None, uses TRACE_OUTPUT_MODE env var or defaults to SUMMARY

    Returns:
        Original result if small, or summary with file path if large (PREVIEW/SUMMARY)
        Full result for FULL mode (still cached for reference)
    """
    from .mlflow_client import text_result

    # Determine output mode
    if mode is None:
        mode = get_output_mode()

    # Extract text content from MCP result format
    try:
        if isinstance(result, dict) and "content" in result:
            content = result["content"]
            if isinstance(content, list) and len(content) > 0:
                text_content = content[0].get("text", "")
            else:
                text_content = str(content)
        else:
            text_content = str(result)
    except (KeyError, IndexError, TypeError):
        text_content = str(result)

    # FULL mode: return complete result IF it fits, otherwise compress
    if mode == OutputMode.FULL:
        content_size = len(text_content.encode('utf-8'))
        if content_size > MAX_INLINE_SIZE:
            # Exceeds SDK limit - must compress regardless of mode
            file_path = _write_to_cache(tool_name, text_content, trace_id)
            logger.warning(
                f"FULL mode requested but result ({content_size:,} bytes) exceeds "
                f"SDK limit ({MAX_INLINE_SIZE:,} bytes). Returning file reference."
            )
            # Use AGGRESSIVE limits for summary since we're forced to compress
            adaptive_limit = _get_adaptive_limit(tool_name, text_content, OutputMode.AGGRESSIVE)
            summary = _summarize_tool_result(tool_name, text_content, adaptive_limit)
            return text_result(
                f"{summary}\n\n"
                f"[Full data ({content_size:,} bytes) saved to: {file_path}]\n"
                f"Use the Read tool to access full trace data."
            )
        # Small enough - return complete result
        if len(text_content) > LARGE_RESULT_THRESHOLD:
            file_path = _write_to_cache(tool_name, text_content, trace_id)
            logger.info(f"FULL mode: Returning complete result ({len(text_content):,} chars, cached to {file_path})")
        return result

    # Check if result is small enough to return inline
    if len(text_content) <= LARGE_RESULT_THRESHOLD:
        return result

    # Write full result to file
    file_path = _write_to_cache(tool_name, text_content, trace_id)

    # Generate summary based on tool type and mode
    if mode == OutputMode.PREVIEW:
        # Legacy behavior: fixed 500 char limit
        summary = _summarize_tool_result(tool_name, text_content, max_inline_chars)
    else:
        # SUMMARY/AGGRESSIVE mode: adaptive limits
        adaptive_limit = _get_adaptive_limit(tool_name, text_content, mode)
        summary = _summarize_tool_result(tool_name, text_content, adaptive_limit)

    # Return summary with file reference
    return text_result(
        f"{summary}\n\n"
        f"[Full data saved to: {file_path}]\n"
        f"Use the Read tool to access full trace data if needed."
    )


def _get_adaptive_limit(tool_name: str, content: str, mode: OutputMode) -> int:
    """Get adaptive character limit based on tool type and output mode.

    Args:
        tool_name: Name of the tool
        content: Raw content string
        mode: Output mode (determines aggressiveness of truncation)

    Returns:
        Character limit for summarization
    """
    # AGGRESSIVE mode: strict limits for batch operations
    if mode == OutputMode.AGGRESSIVE:
        if "get_trace" in tool_name:
            return 1000  # Reduced from 4000 for batch operations
        elif "search_traces" in tool_name:
            return 800   # Reduced from 2000
        else:
            return 500   # Reduced from 1000

    # SUMMARY mode: more generous limits for trace tools
    if "get_trace" in tool_name:
        return 4000  # Traces need more context for LLM analysis
    elif "search_traces" in tool_name:
        return 2000  # Search results can be more compact
    else:
        return 1000  # Default for other tools


def _write_to_cache(
    tool_name: str,
    content: str,
    trace_id: Optional[str] = None,
) -> Path:
    """Write content to cache file.

    Args:
        tool_name: Tool name for file naming
        content: Content to write
        trace_id: Optional trace ID for file naming

    Returns:
        Path to the written file
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Generate filename
    if trace_id:
        filename = f"{trace_id}.json"
    else:
        # Use hash + timestamp for unique filename
        content_hash = abs(hash(content)) % 100000
        timestamp = int(time.time())
        filename = f"{tool_name}_{timestamp}_{content_hash}.json"

    file_path = CACHE_DIR / filename
    file_path.write_text(content, encoding="utf-8")

    logger.info(f"Cached tool result to {file_path} ({len(content):,} chars)")
    return file_path


def _summarize_tool_result(
    tool_name: str,
    content: str,
    max_chars: int,
) -> str:
    """Generate summary based on tool type.

    Args:
        tool_name: Name of the tool
        content: Raw content string (usually JSON)
        max_chars: Maximum characters for summary

    Returns:
        Human-readable summary string
    """
    if "get_trace" in tool_name:
        return _summarize_trace_result(content, max_chars)
    elif "search_traces" in tool_name:
        return _summarize_search_result(content, max_chars)
    else:
        return _generic_summary(content, max_chars)


def _summarize_trace_result(content: str, max_chars: int) -> str:
    """Summarize get_trace output focusing on actionable info.

    Extracts:
    - trace_id, status, duration
    - span count and error count
    - LLM calls and total tokens
    - Top 3 bottlenecks (slowest spans)
    """
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return _generic_summary(content, max_chars)

    info = data.get("info", {})
    spans = data.get("spans", [])
    assessments = data.get("assessments", [])

    # Basic info
    summary_parts = [
        f"**Trace Summary**",
        f"- ID: {info.get('trace_id', 'unknown')}",
        f"- Status: {info.get('status', 'unknown')}",
        f"- Duration: {info.get('execution_time_ms', 0):,.0f}ms",
        f"- Spans: {len(spans)} total",
    ]

    # Count errors
    error_spans = [s for s in spans if s.get("error")]
    if error_spans:
        summary_parts.append(f"- Errors: {len(error_spans)} spans with errors")
        # Show first error
        first_error = error_spans[0]
        error_msg = first_error.get("error", {}).get("message", "unknown")[:100]
        summary_parts.append(f"  First error ({first_error.get('name')}): {error_msg}")

    # LLM usage
    llm_spans = [s for s in spans if s.get("tokens")]
    if llm_spans:
        total_input = sum(s.get("tokens", {}).get("input") or 0 for s in llm_spans)
        total_output = sum(s.get("tokens", {}).get("output") or 0 for s in llm_spans)
        summary_parts.append(f"- LLM calls: {len(llm_spans)}")
        if total_input or total_output:
            summary_parts.append(f"- Tokens: {total_input:,} in / {total_output:,} out")

    # Top 3 bottlenecks (slowest spans, excluding wrappers)
    exclude_patterns = ["forward", "predict", "root", "__init__"]
    filterable_spans = [
        s for s in spans
        if not any(p in s.get("name", "").lower() for p in exclude_patterns)
    ]
    sorted_spans = sorted(filterable_spans, key=lambda s: -(s.get("duration_ms") or 0))
    top_spans = sorted_spans[:3]

    if top_spans:
        summary_parts.append(f"- Top bottlenecks:")
        for s in top_spans:
            duration = s.get("duration_ms", 0)
            summary_parts.append(f"  - {s.get('name')}: {duration:,.0f}ms")

    # Assessments
    if assessments:
        summary_parts.append(f"- Assessments: {len(assessments)} attached")

    return "\n".join(summary_parts)


def _summarize_search_result(content: str, max_chars: int) -> str:
    """Summarize search_traces output.

    Shows header, count, and first few rows.
    """
    lines = content.strip().split("\n")

    if len(lines) <= 10:
        return content[:max_chars]

    # Count data rows (exclude header and separator)
    data_rows = [l for l in lines if l.startswith("|") and "Trace ID" not in l and "---" not in l]
    count = len(data_rows)

    # Return header + first 5 rows + count
    header = lines[:2] if len(lines) >= 2 else lines
    first_rows = lines[2:7] if len(lines) > 2 else []

    summary = "\n".join(header + first_rows)
    if count > 5:
        summary += f"\n\n... and {count - 5} more traces"

    return summary[:max_chars]


def _generic_summary(content: str, max_chars: int) -> str:
    """Generic summarization for unknown tool types."""
    if len(content) <= max_chars:
        return content

    return content[:max_chars - 50] + "\n... [truncated]"


# =============================================================================
# CACHE MANAGEMENT
# =============================================================================


def cleanup_cache(max_age_hours: int = 24) -> int:
    """Remove cached tool result files older than max_age_hours.

    Args:
        max_age_hours: Maximum age in hours before file is deleted

    Returns:
        Number of files removed
    """
    if not CACHE_DIR.exists():
        return 0

    removed = 0
    max_age_seconds = max_age_hours * 3600
    now = time.time()

    for file_path in CACHE_DIR.glob("*.json"):
        try:
            file_age = now - file_path.stat().st_mtime
            if file_age > max_age_seconds:
                file_path.unlink()
                removed += 1
                logger.debug(f"Removed old cache file: {file_path}")
        except (OSError, IOError) as e:
            logger.warning(f"Failed to remove cache file {file_path}: {e}")

    if removed > 0:
        logger.info(f"Cleaned up {removed} old cache files")

    return removed


def get_cache_stats() -> dict[str, Any]:
    """Get statistics about the tool result cache.

    Returns:
        Dictionary with cache statistics
    """
    if not CACHE_DIR.exists():
        return {
            "exists": False,
            "file_count": 0,
            "total_size_bytes": 0,
            "total_size_mb": 0,
        }

    files = list(CACHE_DIR.glob("*.json"))
    total_size = sum(f.stat().st_size for f in files)

    # Get age distribution
    now = time.time()
    ages_hours = [(now - f.stat().st_mtime) / 3600 for f in files]

    return {
        "exists": True,
        "file_count": len(files),
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "oldest_hours": round(max(ages_hours), 1) if ages_hours else 0,
        "newest_hours": round(min(ages_hours), 1) if ages_hours else 0,
        "cache_dir": str(CACHE_DIR),
    }
