"""Custom scorers for evaluating the mlflow-eval-agent framework.

Scorers:
- agent_routing_accuracy: Check if correct agent was invoked
- execution_order_scorer: Verify agents executed in dependency order
- tool_selection_accuracy: Check if correct tools were called
"""

from typing import Any

from mlflow.entities import Feedback, Trace, SpanType
from mlflow.genai.scorers import scorer


@scorer
def agent_routing_accuracy(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    expectations: dict[str, Any],
    trace: Trace,
) -> Feedback:
    """Check if the correct agent was invoked based on trace spans.

    Looks for agent spans in the trace and compares to expected_agent
    in the expectations.

    Args:
        inputs: The input data (query, etc.)
        outputs: The agent outputs
        expectations: Must contain 'expected_agent' key
        trace: MLflow Trace object

    Returns:
        Feedback with yes/no value and rationale
    """
    expected_agent = expectations.get("expected_agent")

    if expected_agent is None:
        return Feedback(
            name="agent_routing_accuracy",
            value="skip",
            rationale="No expected_agent in expectations",
        )

    # Search for agent spans in the trace
    # Agents are typically represented as spans with the agent name
    all_spans = trace.search_spans() if trace else []

    # Known agent names from the registry
    known_agents = ["trace_analyst", "context_engineer", "agent_architect", "eval_runner"]

    # Find which agents were invoked
    invoked_agents = []
    for span in all_spans:
        span_name_lower = span.name.lower() if hasattr(span, "name") else ""
        for agent in known_agents:
            if agent in span_name_lower:
                invoked_agents.append(agent)
                break

    # Check if expected agent was invoked
    expected_invoked = expected_agent in invoked_agents

    if expected_invoked:
        return Feedback(
            name="agent_routing_accuracy",
            value="yes",
            rationale=f"Expected agent '{expected_agent}' was correctly invoked. "
            f"All invoked agents: {invoked_agents}",
        )
    else:
        return Feedback(
            name="agent_routing_accuracy",
            value="no",
            rationale=f"Expected agent '{expected_agent}' was NOT invoked. "
            f"Invoked agents: {invoked_agents}",
        )


@scorer
def execution_order_scorer(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    expectations: dict[str, Any],
    trace: Trace,
) -> Feedback:
    """Verify agents executed in dependency order via span timestamps.

    Expected order from registry dependencies:
    trace_analyst -> context_engineer -> agent_architect -> eval_runner

    Args:
        inputs: The input data
        outputs: The agent outputs
        expectations: Must contain 'expected_order' key (list of agent names)
        trace: MLflow Trace object

    Returns:
        Feedback with yes/no value and rationale
    """
    expected_order = expectations.get("expected_order", [])

    if not expected_order:
        return Feedback(
            name="execution_order",
            value="skip",
            rationale="No expected_order in expectations",
        )

    # Get all spans and extract agent invocations with timestamps
    all_spans = trace.search_spans() if trace else []

    # Known agent names
    known_agents = ["trace_analyst", "context_engineer", "agent_architect", "eval_runner"]

    # Build list of (agent_name, start_time) tuples
    agent_invocations = []
    for span in all_spans:
        span_name_lower = span.name.lower() if hasattr(span, "name") else ""
        for agent in known_agents:
            if agent in span_name_lower:
                start_time = getattr(span, "start_time_ns", 0)
                agent_invocations.append((agent, start_time))
                break

    # Sort by start time to get actual execution order
    agent_invocations.sort(key=lambda x: x[1])
    actual_order = [agent for agent, _ in agent_invocations]

    # Remove duplicates while preserving order
    seen = set()
    actual_order_unique = []
    for agent in actual_order:
        if agent not in seen:
            seen.add(agent)
            actual_order_unique.append(agent)

    # Check if expected order is a subsequence of actual order
    expected_idx = 0
    for agent in actual_order_unique:
        if expected_idx < len(expected_order) and agent == expected_order[expected_idx]:
            expected_idx += 1

    order_correct = expected_idx == len(expected_order)

    if order_correct:
        return Feedback(
            name="execution_order",
            value="yes",
            rationale=f"Agents executed in correct order. "
            f"Expected: {expected_order}, Actual: {actual_order_unique}",
        )
    else:
        return Feedback(
            name="execution_order",
            value="no",
            rationale=f"Agents did NOT execute in expected order. "
            f"Expected: {expected_order}, Actual: {actual_order_unique}",
        )


@scorer
def tool_selection_accuracy(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    expectations: dict[str, Any],
    trace: Trace,
) -> Feedback:
    """Check if the correct tools were called.

    Based on Pattern 14 from patterns-scorers.md.

    Args:
        inputs: The input data
        outputs: The agent outputs
        expectations: Must contain 'expected_tools' key (list of tool names)
        trace: MLflow Trace object

    Returns:
        Feedback with yes/no value and rationale
    """
    expected_tools = expectations.get("expected_tools", [])

    if not expected_tools:
        return Feedback(
            name="tool_selection_accuracy",
            value="skip",
            rationale="No expected_tools in expectations",
        )

    # Get actual tool calls from TOOL spans
    tool_spans = trace.search_spans(span_type=SpanType.TOOL) if trace else []
    actual_tools = {span.name for span in tool_spans if hasattr(span, "name")}

    # Normalize names (handle fully qualified names like "mcp__mlflow__search_traces")
    def normalize(name: str) -> str:
        # Extract the core tool name from various formats
        if "__" in name:
            parts = name.split("__")
            return parts[-1]
        if "." in name:
            return name.split(".")[-1]
        return name

    expected_normalized = {normalize(t) for t in expected_tools}
    actual_normalized = {normalize(t) for t in actual_tools}

    # Check if all expected tools were called
    missing = expected_normalized - actual_normalized
    extra = actual_normalized - expected_normalized

    all_expected_called = len(missing) == 0

    rationale_parts = [
        f"Expected: {sorted(expected_normalized)}",
        f"Actual: {sorted(actual_normalized)}",
    ]
    if missing:
        rationale_parts.append(f"Missing: {sorted(missing)}")
    if extra:
        rationale_parts.append(f"Extra: {sorted(extra)}")

    return Feedback(
        name="tool_selection_accuracy",
        value="yes" if all_expected_called else "no",
        rationale=" | ".join(rationale_parts),
    )


@scorer
def workspace_io_correctness(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    expectations: dict[str, Any],
    trace: Trace,
) -> Feedback:
    """Verify workspace read/write patterns match agent role.

    Checks that agents read from required keys and write to expected output keys.

    Args:
        inputs: The input data
        outputs: The agent outputs
        expectations: Must contain 'expected_reads' and/or 'expected_writes' keys
        trace: MLflow Trace object

    Returns:
        Feedback with yes/no value and rationale
    """
    expected_reads = expectations.get("expected_reads", [])
    expected_writes = expectations.get("expected_writes", [])

    if not expected_reads and not expected_writes:
        return Feedback(
            name="workspace_io_correctness",
            value="skip",
            rationale="No expected_reads or expected_writes in expectations",
        )

    # Get tool spans and extract workspace operations
    tool_spans = trace.search_spans(span_type=SpanType.TOOL) if trace else []

    actual_reads = set()
    actual_writes = set()

    for span in tool_spans:
        span_name = span.name.lower() if hasattr(span, "name") else ""
        span_inputs = getattr(span, "inputs", {}) or {}

        if "read_from_workspace" in span_name or "read" in span_name:
            if isinstance(span_inputs, dict) and "key" in span_inputs:
                actual_reads.add(span_inputs["key"])

        if "write_to_workspace" in span_name or "write" in span_name:
            if isinstance(span_inputs, dict) and "key" in span_inputs:
                actual_writes.add(span_inputs["key"])

    # Check reads
    missing_reads = set(expected_reads) - actual_reads
    missing_writes = set(expected_writes) - actual_writes

    all_correct = len(missing_reads) == 0 and len(missing_writes) == 0

    rationale_parts = []
    if expected_reads:
        rationale_parts.append(f"Expected reads: {expected_reads}, Actual: {sorted(actual_reads)}")
    if expected_writes:
        rationale_parts.append(f"Expected writes: {expected_writes}, Actual: {sorted(actual_writes)}")
    if missing_reads:
        rationale_parts.append(f"Missing reads: {sorted(missing_reads)}")
    if missing_writes:
        rationale_parts.append(f"Missing writes: {sorted(missing_writes)}")

    return Feedback(
        name="workspace_io_correctness",
        value="yes" if all_correct else "no",
        rationale=" | ".join(rationale_parts),
    )
