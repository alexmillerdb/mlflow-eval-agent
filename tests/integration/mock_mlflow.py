"""Mock MLflow client for offline testing.

Enables testing without Databricks connection by providing a mock implementation
of the MLflow client with pre-populated sample data.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MockSpanStatus:
    """Mock span status."""
    status_code: str = "OK"
    description: Optional[str] = None


@dataclass
class MockSpan:
    """Mock MLflow trace span."""
    span_id: str
    name: str
    span_type: str = "UNKNOWN"
    start_time_ns: int = 1700000000000000000
    end_time_ns: int = 1700000002500000000  # 2.5 seconds later
    parent_id: Optional[str] = None
    inputs: Optional[dict] = None
    outputs: Optional[dict] = None
    attributes: dict = field(default_factory=dict)
    status: Optional[MockSpanStatus] = None

    def __post_init__(self):
        if self.status is None:
            self.status = MockSpanStatus()


@dataclass
class MockTraceInfo:
    """Mock trace info."""
    trace_id: str
    status: str = "OK"
    execution_time_ms: int = 2500
    timestamp_ms: int = 1700000000000
    assessments: list = field(default_factory=list)


@dataclass
class MockTraceData:
    """Mock trace data."""
    spans: list = field(default_factory=list)


@dataclass
class MockTrace:
    """Mock MLflow trace object."""
    info: MockTraceInfo
    data: MockTraceData

    @classmethod
    def from_dict(cls, data: dict) -> "MockTrace":
        """Create mock trace from dictionary."""
        info_data = data.get("info", {})
        info = MockTraceInfo(
            trace_id=info_data.get("trace_id", "tr-unknown"),
            status=info_data.get("status", "OK"),
            execution_time_ms=info_data.get("execution_time_ms", 0),
            timestamp_ms=info_data.get("timestamp_ms", 0),
            assessments=info_data.get("assessments", []),
        )

        spans_data = data.get("spans", [])
        spans = []
        for span_data in spans_data:
            span = MockSpan(
                span_id=span_data.get("span_id", "span-unknown"),
                name=span_data.get("name", "unknown"),
                span_type=span_data.get("span_type", "UNKNOWN"),
                parent_id=span_data.get("parent_id"),
                inputs=span_data.get("inputs"),
                outputs=span_data.get("outputs"),
                attributes=span_data.get("attributes", {}),
            )

            # Set timing
            if "duration_ms" in span_data:
                duration_ns = span_data["duration_ms"] * 1_000_000
                span.end_time_ns = span.start_time_ns + duration_ns

            spans.append(span)

        trace_data = MockTraceData(spans=spans)
        return cls(info=info, data=trace_data)


class MockMLflowClient:
    """Mock MLflow client for offline testing.

    Usage:
        client = MockMLflowClient()
        client.add_trace("tr-001", {"status": "OK", ...})
        traces = client.search_traces(["123"], max_results=10)
    """

    def __init__(self):
        self._traces: dict[str, dict] = {}
        self._full_traces: dict[str, dict] = {}
        self._tags: dict[str, dict[str, str]] = {}

    def add_trace(self, trace_id: str, data: dict) -> None:
        """Add a trace to the mock client.

        Args:
            trace_id: Trace ID
            data: Trace summary data (trace_id, status, execution_time_ms, timestamp_ms)
        """
        self._traces[trace_id] = data

    def set_full_trace(self, trace_id: str, data: dict) -> None:
        """Set full trace data for a trace ID.

        Args:
            trace_id: Trace ID
            data: Full trace data including spans and assessments
        """
        self._full_traces[trace_id] = data

    def search_traces(
        self,
        locations: list[str],
        filter_string: Optional[str] = None,
        max_results: int = 100,
    ) -> list[MockTrace]:
        """Search for traces.

        Args:
            locations: Experiment IDs (ignored in mock)
            filter_string: Optional filter (basic parsing supported)
            max_results: Maximum results to return

        Returns:
            List of mock traces
        """
        traces = []

        for trace_id, data in self._traces.items():
            # Basic filter support
            if filter_string:
                filter_lower = filter_string.lower()
                if "status" in filter_lower:
                    if "error" in filter_lower and data.get("status") != "ERROR":
                        continue
                    if "ok" in filter_lower and data.get("status") != "OK":
                        continue

            # Create mock trace info
            info = MockTraceInfo(
                trace_id=trace_id,
                status=data.get("status", "OK"),
                execution_time_ms=data.get("execution_time_ms", 0),
                timestamp_ms=data.get("timestamp_ms", 0),
            )

            # Use full trace data if available
            full_data = self._full_traces.get(trace_id)
            if full_data:
                trace = MockTrace.from_dict(full_data)
                trace.info = info
            else:
                trace = MockTrace(info=info, data=MockTraceData())

            traces.append(trace)

            if len(traces) >= max_results:
                break

        return traces

    def get_trace(self, trace_id: str) -> MockTrace:
        """Get a specific trace.

        Args:
            trace_id: Trace ID to fetch

        Returns:
            Mock trace object

        Raises:
            KeyError: If trace not found
        """
        # Check full traces first
        if trace_id in self._full_traces:
            return MockTrace.from_dict(self._full_traces[trace_id])

        # Fall back to summary data
        if trace_id in self._traces:
            data = self._traces[trace_id]
            info = MockTraceInfo(
                trace_id=trace_id,
                status=data.get("status", "OK"),
                execution_time_ms=data.get("execution_time_ms", 0),
                timestamp_ms=data.get("timestamp_ms", 0),
            )
            return MockTrace(info=info, data=MockTraceData())

        raise KeyError(f"Trace not found: {trace_id}")

    def set_trace_tag(self, trace_id: str, key: str, value: str) -> None:
        """Set a tag on a trace.

        Args:
            trace_id: Trace ID
            key: Tag key
            value: Tag value
        """
        if trace_id not in self._tags:
            self._tags[trace_id] = {}
        self._tags[trace_id][key] = value

    def get_trace_tags(self, trace_id: str) -> dict[str, str]:
        """Get tags for a trace.

        Args:
            trace_id: Trace ID

        Returns:
            Dictionary of tags
        """
        return self._tags.get(trace_id, {})


def create_sample_traces(count: int = 10) -> MockMLflowClient:
    """Create a mock client with sample traces.

    Args:
        count: Number of traces to create

    Returns:
        MockMLflowClient with pre-populated traces
    """
    client = MockMLflowClient()

    for i in range(count):
        trace_id = f"tr-sample-{i:03d}"
        status = "ERROR" if i % 5 == 0 else "OK"  # 20% error rate

        # Summary data
        client.add_trace(trace_id, {
            "trace_id": trace_id,
            "status": status,
            "execution_time_ms": 1000 + (i * 200),
            "timestamp_ms": 1700000000000 + (i * 1000),
        })

        # Full trace data
        spans = [
            {
                "span_id": f"span-{i}-001",
                "name": "agent_query",
                "span_type": "AGENT",
                "duration_ms": 900 + (i * 180),
                "inputs": {"query": f"Test query {i}"},
                "outputs": {"response": f"Test response {i}"},
            },
            {
                "span_id": f"span-{i}-002",
                "name": "llm_call",
                "span_type": "LLM",
                "duration_ms": 600 + (i * 100),
                "parent_id": f"span-{i}-001",
                "attributes": {
                    "mlflow.chat_model.input_tokens": 400 + (i * 50),
                    "mlflow.chat_model.output_tokens": 150 + (i * 20),
                    "mlflow.chat_model.model": "claude-3-sonnet",
                },
            },
        ]

        client.set_full_trace(trace_id, {
            "info": {
                "trace_id": trace_id,
                "status": status,
                "execution_time_ms": 1000 + (i * 200),
            },
            "spans": spans,
            "assessments": [],
        })

    return client
