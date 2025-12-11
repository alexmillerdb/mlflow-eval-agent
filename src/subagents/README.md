# Sub-Agents

This module contains the specialized sub-agents for the MLflow Evaluation Agent.

## Architecture

The agent system uses a **registry pattern** for centralized configuration:

```
registry.py          # AgentConfig dataclass + AGENT_REGISTRY
trace_analyst.py     # Trace analysis agent
context_engineer.py  # Context optimization agent
agent_architect.py   # Architecture analysis agent
prompts.py           # Shared prompt templates
__init__.py          # Public API + agent registration
```

## Adding a New Agent

### Step 1: Create the Agent File

Create `src/subagents/my_new_agent.py`:

```python
"""My New Agent sub-agent definition."""

from claude_agent_sdk import AgentDefinition

from ..tools import MCPTools, InternalTools, BuiltinTools
from .registry import AgentConfig, register_agent


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

MY_NEW_AGENT_PROMPT = """
You are the My New Agent - expert in [your specialty].

## Your Mission
[Describe what this agent does]

## Current Workspace State
{workspace_context}

## Instructions
[Agent-specific instructions]
"""


# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

MY_NEW_AGENT_CONFIG = register_agent(AgentConfig(
    name="my_new_agent",

    description="""
    INVOKE PROACTIVELY when:
    - [Condition 1]
    - [Condition 2]

    This agent [reads/writes] [what] from/to workspace.
    """,

    prompt_template=MY_NEW_AGENT_PROMPT,

    # Context dependencies
    required_keys=["trace_analysis_summary"],  # Must exist before running
    optional_keys=["performance_metrics"],      # Include if available
    output_keys=["my_new_findings"],            # Keys this agent writes

    # Tools this agent can use
    tools=[
        InternalTools.READ_FROM_WORKSPACE,
        InternalTools.WRITE_TO_WORKSPACE,
        MCPTools.SEARCH_TRACES,
        MCPTools.GET_TRACE,
        BuiltinTools.READ,
    ],

    # Token budget for context injection (Phase 2)
    total_token_budget=2000,
    key_token_limits={
        "trace_analysis_summary": 1000,
    },

    model="sonnet",
))


# =============================================================================
# AGENT FACTORY (for backwards compatibility)
# =============================================================================

def create_my_new_agent(workspace_context: str) -> AgentDefinition:
    """Create the my new agent sub-agent."""
    return AgentDefinition(
        description=MY_NEW_AGENT_CONFIG.description,
        prompt=MY_NEW_AGENT_CONFIG.prompt_template.format(workspace_context=workspace_context),
        tools=MY_NEW_AGENT_CONFIG.tools,
        model=MY_NEW_AGENT_CONFIG.model,
    )
```

### Step 2: Register the Agent

Add the import to `src/subagents/__init__.py`:

```python
# Import agent modules to trigger registration
from . import trace_analyst
from . import context_engineer
from . import agent_architect
from . import my_new_agent  # <-- Add this line
```

That's it! The agent is now:
- Registered in `AGENT_REGISTRY`
- Included in `create_subagents()` output
- Listed in the coordinator prompt automatically

## Agent Configuration Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Unique identifier (e.g., "trace_analyst") |
| `description` | `str` | When to invoke this agent (shown in coordinator prompt) |
| `prompt_template` | `str` | System prompt with `{workspace_context}` placeholder |
| `required_keys` | `list[str]` | Workspace keys that MUST exist before running |
| `optional_keys` | `list[str]` | Workspace keys to include if available |
| `output_keys` | `list[str]` | Workspace keys this agent writes |
| `tools` | `list[str]` | Tool names this agent can use |
| `model` | `str` | Model to use (default: "sonnet") |
| `total_token_budget` | `int` | Max tokens for context injection |
| `key_token_limits` | `dict[str, int]` | Per-key token limits |

## Current Agents

| Agent | Dependencies | Outputs |
|-------|--------------|---------|
| `trace_analyst` | (none) | trace_analysis_summary, error_patterns, performance_metrics, extracted_eval_cases |
| `context_engineer` | trace_analysis_summary, error_patterns | context_recommendations |
| `agent_architect` | trace_analysis_summary | (tags traces directly) |

## Workflow Order

The registry automatically computes workflow order based on dependencies:

```python
from src.subagents import get_workflow_order
print(get_workflow_order())  # ['trace_analyst', 'context_engineer', 'agent_architect']
```

## Accessing Agent Configs

```python
from src.subagents import (
    AGENT_REGISTRY,
    TRACE_ANALYST_CONFIG,
    CONTEXT_ENGINEER_CONFIG,
    AGENT_ARCHITECT_CONFIG,
    list_agents,
    get_workflow_order,
)

# List all agents
print(list_agents())  # ['trace_analyst', 'context_engineer', 'agent_architect']

# Get specific config
config = TRACE_ANALYST_CONFIG
print(config.output_keys)  # ['trace_analysis_summary', ...]

# Iterate registry
for name, config in AGENT_REGISTRY.items():
    print(f"{name}: requires {config.required_keys}")
```

## Available Tools

Import tool constants from `src/tools.py`:

```python
from ..tools import MCPTools, InternalTools, BuiltinTools

# MLflow MCP Server tools
MCPTools.SEARCH_TRACES
MCPTools.GET_TRACE
MCPTools.SET_TRACE_TAG
MCPTools.LOG_FEEDBACK
MCPTools.LOG_EXPECTATION
MCPTools.GET_ASSESSMENT

# Internal workspace tools
InternalTools.READ_FROM_WORKSPACE
InternalTools.WRITE_TO_WORKSPACE
InternalTools.CHECK_DEPENDENCIES

# Claude SDK built-in tools
BuiltinTools.READ
BuiltinTools.BASH
BuiltinTools.GLOB
BuiltinTools.GREP
BuiltinTools.SKILL
```
