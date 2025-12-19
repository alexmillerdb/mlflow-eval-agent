"""
Scorers for Multi-Genie Orchestrator agent evaluation.

Includes:
- Built-in scorers: Safety, RelevanceToQuery
- Custom Guidelines scorers: correct_routing, efficient_routing, data_presentation, summary_quality
- Trace-based scorers: genie_routing_efficiency, latency_analysis

Note: Correctness and RetrievalGroundedness are NOT applicable:
- Correctness: Would require ground truth expected_facts for each query
- RetrievalGroundedness: Agent has no RETRIEVER spans (routing agent, not RAG)
"""

from mlflow.genai.scorers import Guidelines, Safety, RelevanceToQuery, scorer
from mlflow.entities import Feedback, Trace


# =============================================================================
# Built-in Scorers
# =============================================================================

# Safety scorer - ensures no harmful content in responses
safety_scorer = Safety()

# Relevance scorer - ensures response addresses the user's query
relevance_scorer = RelevanceToQuery()


# =============================================================================
# Guidelines-Based Scorers (LLM judges)
# =============================================================================

# Scorer 1: Correct routing to appropriate Genies
correct_routing_scorer = Guidelines(
    name="correct_routing",
    guidelines="""The agent should route questions to the appropriate Genie based on the question topic:

1. Sales pipeline questions (pipeline status, deals, revenue, segments, regions, Americas sales data)
   should be routed to the customer_sales Genie.

2. Supply chain questions (raw materials, shortages, forecasts, inventory, suppliers, optimization)
   should be routed to the supply_chain Genie.

3. Questions about the agent system itself (what does the agent do, capabilities, how it works)
   should be answered directly by the supervisor WITHOUT calling any Genie.

4. Questions that span both domains may need both Genies, but only if both are truly required.

The routing decision should be clear and appropriate for the question asked."""
)

# Scorer 2: Efficient routing (minimize unnecessary Genie calls)
efficient_routing_scorer = Guidelines(
    name="efficient_routing",
    guidelines="""The agent should minimize Genie calls for efficiency:

1. For pure sales questions (pipeline, deals, revenue by region/segment), ONLY the customer_sales
   Genie should be called. The supply_chain Genie should NOT be called for sales-only queries.

2. For pure supply chain questions (material shortages, inventory, forecasts), ONLY the supply_chain
   Genie should be called. The customer_sales Genie should NOT be called.

3. Calling both Genies when only one is needed wastes time and resources.

4. The workflow summary should not show unnecessary Genie calls.

5. If a Genie returns "no data" or irrelevant response, that was likely an unnecessary call.

Review the workflow execution and assess if all Genie calls were necessary."""
)

# Scorer 3: Data presentation quality
data_presentation_scorer = Guidelines(
    name="data_presentation",
    guidelines="""When presenting tabular data from Genies, the response should:

1. Include a clear headline that summarizes what was requested
   (e.g., "Sales Pipeline for the Americas by Segment:")

2. Present the data table clearly, preserving formatting and readability

3. NOT over-explain or add unnecessary analysis to simple data requests

4. Be concise and actionable - users want the data, not lengthy explanations

5. If multiple data sources were used, clearly attribute data to the correct source

The response should let the data speak for itself while being professionally presented."""
)

# Scorer 4: Workflow summary quality
summary_quality_scorer = Guidelines(
    name="summary_quality",
    guidelines="""The workflow summary should be accurate and helpful:

1. Correctly describe which agents were used and why they were selected

2. NOT claim agents were used when they weren't (e.g., don't mention supply_chain
   if it wasn't actually called)

3. Correctly attribute data and findings to the right agent

4. Provide a meaningful summary of the execution, not just repeat the response

5. Help users understand the multi-agent workflow that was executed

6. Be factually accurate about what happened during execution

The summary should build trust by accurately describing the system's behavior."""
)


# =============================================================================
# Trace-Based Custom Scorers
# =============================================================================

@scorer
def genie_call_count(outputs) -> Feedback:
    """Count how many Genies were called based on agents_used in outputs.

    This scorer works with pre-computed outputs that include agents_used metadata.
    """
    agents_used = outputs.get("agents_used", [])

    if agents_used is None:
        agents_used = []

    count = len(agents_used)

    return Feedback(
        name="genie_call_count",
        value=count,
        rationale=f"Genies called: {agents_used if agents_used else 'none'}"
    )


@scorer
def has_workflow_summary(outputs) -> Feedback:
    """Check if the response includes a workflow summary."""
    summary = outputs.get("workflow_summary", "")

    has_summary = bool(summary and len(summary) > 50)

    return Feedback(
        name="has_workflow_summary",
        value=has_summary,
        rationale=f"Workflow summary {'present' if has_summary else 'missing or too short'} ({len(summary) if summary else 0} chars)"
    )


@scorer
def response_has_data_table(outputs) -> Feedback:
    """Check if the response contains a markdown table."""
    response = outputs.get("response", "")

    if response is None:
        return Feedback(
            name="has_data_table",
            value=False,
            rationale="No response available"
        )

    # Check for markdown table indicators
    has_table = "|" in response and "---" in response.replace("|", "").replace(":", "").replace("-", "---")
    has_table = has_table or ("|" in response and response.count("|") >= 4)

    return Feedback(
        name="has_data_table",
        value=has_table,
        rationale=f"Response {'contains' if has_table else 'does not contain'} a data table"
    )


@scorer(aggregations=["mean", "min", "max", "median", "p90"])
def response_length_words(outputs) -> int:
    """Count words in the response for length analysis."""
    response = outputs.get("response", "")

    if response is None:
        return 0

    return len(response.split())


# =============================================================================
# Routing Quality Scorers (for metadata-enhanced evaluation)
# =============================================================================

@scorer
def routing_matches_expected(inputs, outputs) -> Feedback:
    """
    Check if actual routing matches expected routing.

    This scorer requires metadata with 'expected_routing' and 'actual_routing' fields.
    When using with mlflow.genai.evaluate(), include these in the outputs dict.
    """
    # This will be called on outputs that include metadata
    expected = outputs.get("metadata", {}).get("expected_routing")
    actual = outputs.get("metadata", {}).get("actual_routing")

    if expected is None or actual is None:
        return Feedback(
            name="routing_matches_expected",
            value="skip",
            rationale="Missing routing metadata (expected_routing or actual_routing)"
        )

    # Normalize to lists for comparison
    expected_list = [expected] if isinstance(expected, str) else list(expected)
    actual_list = [actual] if isinstance(actual, str) else list(actual)

    matches = set(expected_list) == set(actual_list)

    return Feedback(
        name="routing_matches_expected",
        value="yes" if matches else "no",
        rationale=f"Expected: {expected_list}, Actual: {actual_list}"
    )


# =============================================================================
# Scorer Collections
# =============================================================================

# All scorers for comprehensive evaluation
ALL_SCORERS = [
    # Built-in
    safety_scorer,
    relevance_scorer,
    # Guidelines
    correct_routing_scorer,
    efficient_routing_scorer,
    data_presentation_scorer,
    summary_quality_scorer,
    # Custom
    genie_call_count,
    has_workflow_summary,
    response_has_data_table,
    response_length_words,
]

# Core scorers for quick evaluation
CORE_SCORERS = [
    safety_scorer,
    relevance_scorer,
    correct_routing_scorer,
    efficient_routing_scorer,
]

# Guidelines-only scorers (for LLM judge evaluation)
GUIDELINES_SCORERS = [
    correct_routing_scorer,
    efficient_routing_scorer,
    data_presentation_scorer,
    summary_quality_scorer,
]

# Metric scorers (custom code-based)
METRIC_SCORERS = [
    genie_call_count,
    has_workflow_summary,
    response_has_data_table,
    response_length_words,
]


def get_scorers(preset: str = "all"):
    """
    Get a preset collection of scorers.

    Args:
        preset: One of "all", "core", "guidelines", "metrics"

    Returns:
        List of scorer objects
    """
    presets = {
        "all": ALL_SCORERS,
        "core": CORE_SCORERS,
        "guidelines": GUIDELINES_SCORERS,
        "metrics": METRIC_SCORERS,
    }

    return presets.get(preset, ALL_SCORERS)


if __name__ == "__main__":
    print("Available scorer presets:")
    print(f"  all: {len(ALL_SCORERS)} scorers")
    print(f"  core: {len(CORE_SCORERS)} scorers")
    print(f"  guidelines: {len(GUIDELINES_SCORERS)} scorers")
    print(f"  metrics: {len(METRIC_SCORERS)} scorers")

    print("\nAll scorers:")
    for s in ALL_SCORERS:
        name = getattr(s, "name", s.__name__ if hasattr(s, "__name__") else str(s))
        print(f"  - {name}")
