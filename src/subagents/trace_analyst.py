"""Trace Analyst sub-agent definition."""

from claude_agent_sdk import AgentDefinition

from ..tools import MCPTools, InternalTools, BuiltinTools
from .registry import AgentConfig, register_agent
from .prompts import MCP_SEARCH_EXAMPLE, MCP_GET_TRACE_EXAMPLE, MCP_LOG_FEEDBACK_EXAMPLE, FILTER_SYNTAX_REFERENCE


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

TRACE_ANALYST_PROMPT = f"""
You are the Trace Analyst - expert in MLflow trace analysis using MCP tools.

## Your Mission
Analyze production traces to find actionable insights. Your findings are used by
context_engineer and agent_architect, so be thorough.

## MCP-First Analysis Approach

### Step 1: Search Traces
```
{MCP_SEARCH_EXAMPLE}
```

### Step 2: Deep Dive
```
{MCP_GET_TRACE_EXAMPLE}
```

### Step 3: Log Findings (Optional)
```
{MCP_LOG_FEEDBACK_EXAMPLE}
```

{FILTER_SYNTAX_REFERENCE}

## Write to Shared Workspace
After analysis, write findings for other agents:
- "trace_analysis_summary" - High-level findings
- "error_patterns" - Classified errors with example trace IDs
- "performance_metrics" - Latency percentiles, bottlenecks
- "extracted_eval_cases" - Trace IDs suitable for eval datasets

## Current Workspace State
{{workspace_context}}

## Output Format
1. Executive summary (2-3 sentences)
2. Key metrics table
3. Issue breakdown with trace IDs
4. Recommended evaluations to build
"""


# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

TRACE_ANALYST_CONFIG = register_agent(AgentConfig(
    name="trace_analyst",

    description="""
    INVOKE PROACTIVELY when:
    - User mentions traces, logs, production, debugging
    - Need to find patterns in agent behavior
    - Analyzing errors or performance
    - Extracting test cases from production

    This agent WRITES findings to shared workspace for other agents.
    """,

    prompt_template=TRACE_ANALYST_PROMPT,

    # First in pipeline - no dependencies
    required_keys=[],
    optional_keys=[],
    output_keys=[
        "trace_analysis_summary",
        "error_patterns",
        "performance_metrics",
        "extracted_eval_cases",
    ],

    tools=[
        MCPTools.SEARCH_TRACES,
        MCPTools.GET_TRACE,
        MCPTools.SET_TRACE_TAG,
        MCPTools.LOG_FEEDBACK,
        MCPTools.GET_ASSESSMENT,
        InternalTools.WRITE_TO_WORKSPACE,
        BuiltinTools.READ,
        BuiltinTools.SKILL,
    ],

    total_token_budget=1000,  # Minimal - first in pipeline
    model="sonnet",
))


# =============================================================================
# AGENT FACTORY (for backwards compatibility)
# =============================================================================

def create_trace_analyst(workspace_context: str) -> AgentDefinition:
    """Create the trace analyst sub-agent.

    This function is kept for backwards compatibility.
    The preferred approach is to use TRACE_ANALYST_CONFIG with the registry.
    """
    return AgentDefinition(
        description=TRACE_ANALYST_CONFIG.description,
        prompt=TRACE_ANALYST_CONFIG.prompt_template.format(workspace_context=workspace_context),
        tools=TRACE_ANALYST_CONFIG.tools,
        model=TRACE_ANALYST_CONFIG.model,
    )
