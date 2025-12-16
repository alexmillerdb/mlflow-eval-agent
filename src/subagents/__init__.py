"""Sub-agent definitions for MLflow Evaluation Agent."""

from claude_agent_sdk import AgentDefinition

from ..workspace import SharedWorkspace
from ..tools import MCPTools, InternalTools, BuiltinTools

# Import registry
from .registry import AGENT_REGISTRY, AgentConfig, register_agent, get_workflow_order, list_agents

# Import agent modules to trigger registration (order matters for registration)
from . import trace_analyst
from . import context_engineer
from . import agent_architect
from . import eval_runner

# Import coordinator prompt generator and config
from .coordinator import get_coordinator_system_prompt, COORDINATOR_CONFIG

# Re-export individual configs for direct access
from .trace_analyst import TRACE_ANALYST_CONFIG
from .context_engineer import CONTEXT_ENGINEER_CONFIG
from .agent_architect import AGENT_ARCHITECT_CONFIG
from .eval_runner import EVAL_RUNNER_CONFIG


def create_subagents(
    workspace: SharedWorkspace,
    use_selective_context: bool = True
) -> dict[str, AgentDefinition]:
    """Create all sub-agent definitions from registry with workspace awareness.

    Uses AGENT_REGISTRY to create AgentDefinition instances for each
    registered sub-agent. Each agent receives only the workspace context
    relevant to its dependencies (selective context injection).

    Args:
        workspace: SharedWorkspace instance for context injection
        use_selective_context: If True (default), use selective context based on
            agent's required_keys/optional_keys. If False, use full context.

    Returns:
        Dictionary mapping agent names to AgentDefinition instances
    """
    agents = {}

    for name, config in AGENT_REGISTRY.items():
        if use_selective_context:
            # Selective context: only keys relevant to this agent
            workspace_context = workspace.to_selective_context(config)
            if not workspace_context.strip() or workspace_context == "<workspace_context>\n</workspace_context>":
                workspace_context = "No relevant data in workspace yet."
        else:
            # Full context: all keys (legacy behavior)
            workspace_context = workspace.to_context_string() or "No data in workspace yet."

        # Note: max_turns is stored in config for coordinator reference
        # The Claude SDK doesn't support max_turns in AgentDefinition directly
        agents[name] = AgentDefinition(
            description=config.description,
            prompt=config.prompt_template.format(workspace_context=workspace_context),
            tools=config.tools,
            model=config.model,
        )

    return agents


def validate_agent_can_run(
    agent_name: str,
    workspace: SharedWorkspace
) -> tuple[bool, list[str], str]:
    """Check if an agent has all required dependencies to run.

    Use this before invoking a sub-agent to fail fast with a clear message
    if required workspace entries are missing.

    Args:
        agent_name: Name of the agent to validate
        workspace: SharedWorkspace instance to check

    Returns:
        Tuple of (can_run, missing_keys, message)

    Example:
        can_run, missing, msg = validate_agent_can_run("context_engineer", workspace)
        if not can_run:
            print(f"Cannot run: {msg}")
    """
    config = AGENT_REGISTRY.get(agent_name)
    if not config:
        return False, [], f"Unknown agent: {agent_name}"

    return workspace.check_agent_dependencies(config)


def get_coordinator_prompt(workspace: SharedWorkspace, experiment_id: str = "") -> str:
    """Get coordinator system prompt with current workspace state.

    Generates the coordinator prompt dynamically from the registry,
    ensuring it stays in sync with registered agents.

    Uses selective context injection based on COORDINATOR_CONFIG to
    prevent context bloat - only includes relevant workspace keys with
    token limits applied.

    Args:
        workspace: SharedWorkspace instance for context injection
        experiment_id: MLflow experiment ID (numeric string)

    Returns:
        Complete coordinator system prompt
    """
    # Use selective context based on COORDINATOR_CONFIG for efficient context management
    workspace_context = workspace.to_selective_context(COORDINATOR_CONFIG)
    if not workspace_context.strip() or workspace_context == "<workspace_context>\n</workspace_context>":
        workspace_context = "Empty - no analysis run yet."
    return get_coordinator_system_prompt(workspace_context, experiment_id=experiment_id)


__all__ = [
    # Main functions
    "create_subagents",
    "get_coordinator_prompt",
    "validate_agent_can_run",
    # Registry
    "AGENT_REGISTRY",
    "AgentConfig",
    "register_agent",
    "get_workflow_order",
    "list_agents",
    # Individual configs
    "COORDINATOR_CONFIG",
    "TRACE_ANALYST_CONFIG",
    "CONTEXT_ENGINEER_CONFIG",
    "AGENT_ARCHITECT_CONFIG",
    "EVAL_RUNNER_CONFIG",
    # Tool constants
    "MCPTools",
    "InternalTools",
    "BuiltinTools",
]
