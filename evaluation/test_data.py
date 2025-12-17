"""Test data definitions for evaluation tests.

Each test case follows the MLflow GenAI data schema:
{
    "inputs": {...},       # Passed to predict_fn
    "expectations": {...}  # Used by scorers for validation
}
"""

# -----------------------------------------------------------------------------
# Agent Routing Test Cases (5 cases)
# -----------------------------------------------------------------------------

AGENT_ROUTING_CASES = [
    {
        "inputs": {"query": "Analyze the recent traces and identify patterns"},
        "expectations": {
            "expected_agent": "trace_analyst",
            "description": "Trace analysis request should route to trace_analyst",
        },
    },
    {
        "inputs": {"query": "What are the error patterns in the last 24 hours?"},
        "expectations": {
            "expected_agent": "trace_analyst",
            "description": "Error pattern analysis routes to trace_analyst",
        },
    },
    {
        "inputs": {"query": "Optimize the prompt context to reduce token usage"},
        "expectations": {
            "expected_agent": "context_engineer",
            "description": "Context optimization routes to context_engineer",
        },
    },
    {
        "inputs": {"query": "Review the multi-agent architecture and suggest improvements"},
        "expectations": {
            "expected_agent": "agent_architect",
            "description": "Architecture review routes to agent_architect",
        },
    },
    {
        "inputs": {"query": "Execute the generated evaluation script"},
        "expectations": {
            "expected_agent": "eval_runner",
            "description": "Evaluation execution routes to eval_runner",
        },
    },
]


# -----------------------------------------------------------------------------
# Sequential Ordering Test Cases (3 cases)
# -----------------------------------------------------------------------------

ORDERING_CASES = [
    {
        "inputs": {"query": "Run full analysis pipeline on recent traces"},
        "expectations": {
            "expected_order": ["trace_analyst", "context_engineer", "agent_architect"],
            "description": "Full pipeline runs in dependency order",
        },
    },
    {
        "inputs": {"query": "Analyze traces and optimize context"},
        "expectations": {
            "expected_order": ["trace_analyst", "context_engineer"],
            "description": "Context-only pipeline runs trace_analyst first",
        },
    },
    {
        "inputs": {"query": "Analyze traces and review architecture"},
        "expectations": {
            "expected_order": ["trace_analyst", "agent_architect"],
            "description": "Architecture review requires trace_analyst first",
        },
    },
]


# -----------------------------------------------------------------------------
# Tool Call Test Cases (4 cases)
# -----------------------------------------------------------------------------

TOOL_CALL_CASES = [
    {
        "inputs": {"query": "Analyze the recent traces and summarize findings"},
        "expectations": {
            "expected_agent": "trace_analyst",
            "expected_tools": ["search_traces", "get_trace", "write_to_workspace"],
            "expected_writes": [
                "trace_analysis_summary",
                "error_patterns",
                "performance_metrics",
            ],
            "description": "trace_analyst must use trace tools and write findings",
        },
    },
    {
        "inputs": {"query": "Optimize the context based on trace analysis"},
        "expectations": {
            "expected_agent": "context_engineer",
            "expected_tools": ["read_from_workspace", "write_to_workspace"],
            "expected_reads": ["trace_analysis_summary", "error_patterns"],
            "expected_writes": ["context_recommendations"],
            "description": "context_engineer reads trace data, writes recommendations",
        },
    },
    {
        "inputs": {"query": "Review the agent architecture"},
        "expectations": {
            "expected_agent": "agent_architect",
            "expected_tools": ["read_from_workspace", "get_trace"],
            "expected_reads": ["trace_analysis_summary"],
            "description": "agent_architect reads findings, may inspect traces",
        },
    },
    {
        "inputs": {"query": "Execute the generated evaluation"},
        "expectations": {
            "expected_agent": "eval_runner",
            "expected_tools": ["read_from_workspace", "bash"],
            "expected_reads": ["generated_eval_code"],
            "expected_writes": ["eval_results"],
            "description": "eval_runner reads code, executes via bash, writes results",
        },
    },
]


# -----------------------------------------------------------------------------
# Integration Test Data (combines multiple aspects)
# -----------------------------------------------------------------------------

INTEGRATION_EVAL_DATA = [
    {
        "inputs": {"query": "Analyze the error patterns from traces"},
        "expectations": {
            "expected_agent": "trace_analyst",
            "expected_tools": ["search_traces", "get_trace", "write_to_workspace"],
            "expected_order": ["trace_analyst"],
        },
    },
    {
        "inputs": {"query": "Based on the trace analysis, optimize the prompt context"},
        "expectations": {
            "expected_agent": "context_engineer",
            "expected_tools": ["read_from_workspace", "write_to_workspace"],
            "expected_order": ["trace_analyst", "context_engineer"],
        },
    },
]


# -----------------------------------------------------------------------------
# Agent Configuration Expected Values (for unit tests)
# -----------------------------------------------------------------------------

EXPECTED_AGENT_TOOLS = {
    "trace_analyst": {
        "required_tools": ["search_traces", "get_trace", "write_to_workspace"],
        "optional_tools": ["set_trace_tag", "log_feedback"],
    },
    "context_engineer": {
        "required_tools": ["read_from_workspace", "write_to_workspace"],
        "optional_tools": ["search_traces", "get_trace", "log_feedback"],
    },
    "agent_architect": {
        "required_tools": ["read_from_workspace"],
        "optional_tools": ["search_traces", "get_trace", "set_trace_tag"],
    },
    "eval_runner": {
        "required_tools": ["read_from_workspace", "write_to_workspace"],
        "optional_tools": ["bash", "read"],
    },
}

EXPECTED_AGENT_DEPENDENCIES = {
    "trace_analyst": {
        "required_keys": [],  # No dependencies (runs first)
        "output_keys": ["trace_analysis_summary", "error_patterns", "performance_metrics"],
    },
    "context_engineer": {
        "required_keys": ["trace_analysis_summary", "error_patterns"],
        "output_keys": ["context_recommendations"],
    },
    "agent_architect": {
        "required_keys": ["trace_analysis_summary"],
        "output_keys": [],  # Tags traces directly
    },
    "eval_runner": {
        "required_keys": ["generated_eval_code"],
        "output_keys": ["eval_results"],
    },
}

EXPECTED_WORKFLOW_ORDER = [
    "trace_analyst",
    "context_engineer",
    "agent_architect",
    "eval_runner",
]
