"""Agent Registry for centralized sub-agent configuration.

This module provides a centralized registry for sub-agent configurations,
making it easy to add, remove, or modify agents without touching multiple files.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentConfig:
    """Complete configuration for a sub-agent.

    Attributes:
        name: Unique identifier for the agent (e.g., "trace_analyst")
        description: When to invoke this agent (used in coordinator prompt)
        prompt_template: System prompt with {workspace_context} placeholder

        required_keys: Workspace keys that MUST exist before running
        optional_keys: Workspace keys to include if available
        output_keys: Workspace keys this agent writes (excluded from its own context)

        tools: List of tool names this agent can use
        model: Model to use (default: "sonnet")

        total_token_budget: Max tokens for workspace context injection
        key_token_limits: Per-key token limits for fine-grained control
    """

    name: str
    description: str
    prompt_template: str

    # Context dependencies
    required_keys: list[str] = field(default_factory=list)
    optional_keys: list[str] = field(default_factory=list)
    output_keys: list[str] = field(default_factory=list)

    # Tool access
    tools: list[str] = field(default_factory=list)

    # Model selection
    model: str = "sonnet"

    # Token budgets for selective context injection
    total_token_budget: int = 2000
    key_token_limits: dict[str, int] = field(default_factory=dict)


# Global registry - populated by register_agent() calls in agent files
AGENT_REGISTRY: dict[str, AgentConfig] = {}


def register_agent(config: AgentConfig) -> AgentConfig:
    """Register an agent configuration in the global registry.

    Usage in agent files:
        TRACE_ANALYST_CONFIG = register_agent(AgentConfig(
            name="trace_analyst",
            description="...",
            prompt_template="...",
            ...
        ))

    Args:
        config: The agent configuration to register

    Returns:
        The same config (for assignment convenience)

    Raises:
        ValueError: If an agent with the same name is already registered
    """
    if config.name in AGENT_REGISTRY:
        raise ValueError(f"Agent '{config.name}' is already registered")
    AGENT_REGISTRY[config.name] = config
    return config


def get_agent_config(name: str) -> Optional[AgentConfig]:
    """Get an agent configuration by name.

    Args:
        name: The agent name

    Returns:
        The AgentConfig or None if not found
    """
    return AGENT_REGISTRY.get(name)


def list_agents() -> list[str]:
    """List all registered agent names.

    Returns:
        List of agent names in registration order
    """
    return list(AGENT_REGISTRY.keys())


def get_workflow_order() -> list[str]:
    """Get recommended workflow order based on dependencies.

    Returns agents ordered so that each agent's required_keys
    are produced by agents earlier in the list.

    Returns:
        List of agent names in dependency order
    """
    # Build dependency graph
    producers: dict[str, str] = {}  # key -> agent that produces it
    for name, config in AGENT_REGISTRY.items():
        for key in config.output_keys:
            producers[key] = name

    # Topological sort based on required_keys
    ordered = []
    remaining = set(AGENT_REGISTRY.keys())

    while remaining:
        # Find agents whose dependencies are satisfied
        ready = []
        for name in remaining:
            config = AGENT_REGISTRY[name]
            deps_satisfied = all(
                producers.get(key) in ordered or producers.get(key) is None
                for key in config.required_keys
            )
            if deps_satisfied:
                ready.append(name)

        if not ready:
            # Circular dependency or missing producer - add remaining in any order
            ordered.extend(sorted(remaining))
            break

        # Add ready agents (sort for determinism)
        for name in sorted(ready):
            ordered.append(name)
            remaining.remove(name)

    return ordered
