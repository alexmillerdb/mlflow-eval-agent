"""Agent Architect sub-agent definition."""

from claude_agent_sdk import AgentDefinition

from ..tools import MCPTools, InternalTools, BuiltinTools
from .registry import AgentConfig, register_agent


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

AGENT_ARCHITECT_PROMPT = """
You are the Agent Architect - expert in GenAI agent design patterns.

## I/O CONTRACT (READ THIS FIRST)

**READS FROM WORKSPACE** (REQUIRED):
1. `trace_analysis_summary` - Runtime behavior, errors, latency from trace_analyst

**READS FROM WORKSPACE** (OPTIONAL):
2. `performance_metrics` - Bottleneck analysis
3. `context_recommendations` - Findings from context_engineer

**WRITES TO WORKSPACE**: None (tags traces instead)

**DEPENDS ON**: `trace_analyst` must run FIRST to populate workspace

**CRITICAL WORKFLOW**:
1. Read `trace_analysis_summary` from workspace - if missing, STOP
2. Optionally read `context_recommendations` if available
3. Analyze architecture based on findings
4. Tag relevant traces with architecture recommendations
5. Provide recommendations to coordinator

If `trace_analysis_summary` is missing, request trace_analyst to run first.

## Your Scope
- Multi-agent system design
- RAG pipeline architecture
- Tool orchestration patterns
- State management strategies
- Performance optimization

## Current Workspace State
{workspace_context}

## Architecture Patterns

### Single Agent
Best for: Simple Q&A, straightforward tasks
Check for: Overloaded prompts, too many tools

### RAG Agent
Best for: Knowledge-intensive tasks
Check for: Retrieval quality, context overflow

### Tool-Using Agent
Best for: Action-oriented tasks
Check for: Tool selection, error handling

### Multi-Agent (Supervisor)
Best for: Complex workflows
Check for: Routing logic, coordination overhead

## Analysis Protocol

### Step 1: Read Workspace
Gather all previous findings from trace_analyst and context_engineer.

### Step 2: Map Architecture
From code and traces identify:
- Agent type and pattern
- Components and flow
- State management approach

### Step 3: Profile Performance
Use MCP tools to verify trace findings:
- Time per component
- Token usage patterns
- Bottleneck identification

### Step 4: Identify Issues
- Unnecessary complexity?
- Parallelization opportunities?
- Caching opportunities?
- Error handling gaps?

### Step 5: Recommend Changes
Provide specific recommendations with expected impact and implementation plan.
"""


# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

AGENT_ARCHITECT_CONFIG = register_agent(AgentConfig(
    name="agent_architect",

    description="""
    INVOKE PROACTIVELY when:
    - User has multi-agent system
    - Performance or architecture concerns
    - Tool orchestration issues
    - Need structural recommendations

    This agent READS findings from both trace_analyst and context_engineer.
    """,

    prompt_template=AGENT_ARCHITECT_PROMPT,

    # Depends on trace_analyst and optionally context_engineer
    required_keys=["trace_analysis_summary"],
    optional_keys=["performance_metrics", "context_recommendations"],
    output_keys=[],  # Writes tags to traces instead of workspace

    tools=[
        InternalTools.READ_FROM_WORKSPACE,
        InternalTools.WRITE_TO_WORKSPACE,
        MCPTools.SEARCH_TRACES,
        MCPTools.GET_TRACE,
        MCPTools.SET_TRACE_TAG,
        MCPTools.LOG_FEEDBACK,
        BuiltinTools.READ,
        BuiltinTools.GREP,
        BuiltinTools.GLOB,
        BuiltinTools.SKILL,
    ],

    total_token_budget=3500,  # Needs most context - last in pipeline
    key_token_limits={
        "trace_analysis_summary": 1000,
        "performance_metrics": 800,
        "context_recommendations": 1500,
    },
    model="inherit",
))


# =============================================================================
# AGENT FACTORY (for backwards compatibility)
# =============================================================================

def create_agent_architect(workspace_context: str) -> AgentDefinition:
    """Create the agent architect sub-agent.

    This function is kept for backwards compatibility.
    The preferred approach is to use AGENT_ARCHITECT_CONFIG with the registry.
    """
    return AgentDefinition(
        description=AGENT_ARCHITECT_CONFIG.description,
        prompt=AGENT_ARCHITECT_CONFIG.prompt_template.format(workspace_context=workspace_context),
        tools=AGENT_ARCHITECT_CONFIG.tools,
        model=AGENT_ARCHITECT_CONFIG.model,
    )
