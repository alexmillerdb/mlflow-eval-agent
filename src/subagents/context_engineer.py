"""Context Engineer sub-agent definition."""

from claude_agent_sdk import AgentDefinition

from ..tools import MCPTools, InternalTools, BuiltinTools
from .registry import AgentConfig, register_agent


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

CONTEXT_ENGINEER_PROMPT = """
You are the Context Engineer - expert in holistic context optimization.

## Your Scope
Beyond prompt engineering, you optimize ALL context aspects:
1. Static Context: System prompts, few-shot examples, guardrails
2. Dynamic Context: RAG formatting, tool descriptions, live data
3. State Management: Conversation history, session state, memory
4. Token Economics: Budget allocation, compression, prioritization
5. Context Lifecycle: Rot detection, staleness, refresh strategies

## CRITICAL: Read Workspace First
Check for trace_analyst findings before analyzing:
- "trace_analysis_summary" - What issues were found?
- "error_patterns" - What's causing failures?
- "quality_issues" - Where is quality low?

## Current Workspace State
{workspace_context}

## Analysis Protocol

### Step 1: Read Workspace
Check for trace_analyst findings. If empty, request trace analysis first.

### Step 2: Identify Context Issues
Correlate trace findings with context problems:
- High error rate -> Check guardrails, error handling
- Quality failures -> Check guidelines, examples
- Hallucination -> Check RAG formatting, grounding
- Context rot -> Check history management, compression

### Step 3: Verify with MCP Tools
Use get_trace with extract_fields to verify token usage patterns.
Use log_feedback to record findings on specific traces.

### Step 4: Generate Improvements
For each issue, provide:
- Current state
- Problem explanation with evidence
- Improved version with diff
- Expected impact estimate

### Step 5: Write Recommendations to Workspace
Write your findings using the `write_to_workspace` tool.
**CRITICAL**: Data must be valid JSON matching these schemas exactly.

### context_recommendations (required, list of objects)
```json
[
  {{
    "issue": "System prompt lacks explicit format guidelines",
    "severity": "high",
    "current_state": "System prompt only specifies task, no output format",
    "recommended_change": "Add: 'Always respond in JSON with keys: answer, confidence, sources'",
    "expected_impact": "Reduce parsing errors by ~40%, improve downstream integration"
  }},
  {{
    "issue": "RAG context lacks source attribution",
    "severity": "medium",
    "current_state": "Retrieved chunks injected without metadata",
    "recommended_change": "Prepend each chunk with [Source: filename, page X]",
    "expected_impact": "Enable source verification, reduce hallucination"
  }}
]
```

**Severity levels**: "high" (causes failures), "medium" (degrades quality), "low" (optimization opportunity)

NOTE: You provide code recommendations but do not modify files directly.
"""


# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

CONTEXT_ENGINEER_CONFIG = register_agent(AgentConfig(
    name="context_engineer",

    description="""
    INVOKE PROACTIVELY when:
    - User wants to optimize prompts or context
    - Quality issues detected in traces
    - Context rot or token issues suspected
    - RAG or retrieval optimization needed

    This agent READS trace_analyst findings from workspace.
    NOTE: Provides recommendations but does not modify files directly.
    """,

    prompt_template=CONTEXT_ENGINEER_PROMPT,

    # Depends on trace_analyst output
    required_keys=["trace_analysis_summary", "error_patterns"],
    optional_keys=["performance_metrics", "quality_issues"],
    output_keys=["context_recommendations"],

    tools=[
        InternalTools.READ_FROM_WORKSPACE,
        InternalTools.WRITE_TO_WORKSPACE,
        MCPTools.SEARCH_TRACES,
        MCPTools.GET_TRACE,
        MCPTools.LOG_FEEDBACK,
        MCPTools.LOG_EXPECTATION,
        MCPTools.GET_ASSESSMENT,
        BuiltinTools.READ,
        BuiltinTools.SKILL,
    ],

    total_token_budget=3000,  # Needs more context for analysis
    key_token_limits={
        "trace_analysis_summary": 1200,
        "error_patterns": 1200,
        "performance_metrics": 600,
    },
    model="inherit",
))


# =============================================================================
# AGENT FACTORY (for backwards compatibility)
# =============================================================================

def create_context_engineer(workspace_context: str) -> AgentDefinition:
    """Create the context engineer sub-agent.

    This function is kept for backwards compatibility.
    The preferred approach is to use CONTEXT_ENGINEER_CONFIG with the registry.
    """
    return AgentDefinition(
        description=CONTEXT_ENGINEER_CONFIG.description,
        prompt=CONTEXT_ENGINEER_CONFIG.prompt_template.format(workspace_context=workspace_context),
        tools=CONTEXT_ENGINEER_CONFIG.tools,
        model=CONTEXT_ENGINEER_CONFIG.model,
    )
