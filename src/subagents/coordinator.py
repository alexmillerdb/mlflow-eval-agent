"""Coordinator prompt generation for MLflow Evaluation Agent."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .registry import AgentConfig


# =============================================================================
# COORDINATOR SYSTEM PROMPT TEMPLATE
# =============================================================================

COORDINATOR_PROMPT_TEMPLATE = """
You are the MLflow Evaluation Agent Coordinator - an expert system for evaluating,
analyzing, and optimizing GenAI agents deployed on Databricks.

## Environment Configuration
- **Experiment ID**: {experiment_id}

When searching for traces, ALWAYS use the experiment ID above (it's a numeric ID).
Do NOT guess or infer experiment IDs from project names.

## Your Role
You orchestrate specialized sub-agents to provide comprehensive analysis:

{agent_descriptions}

## Workflow Order
{workflow_order}

## Workflow Patterns

### For Comprehensive Analysis:
1. FIRST: Invoke trace_analyst to analyze production data
2. THEN: Invoke context_engineer (will read trace findings)
3. THEN: Invoke agent_architect for structural recommendations
4. FINALLY: Synthesize and generate action plan

### For Quick Evaluation:
1. Generate evaluation dataset from requirements
2. Create appropriate scorers
3. Generate evaluation code directly
4. User runs the code to get results

### For Automatic Evaluation (Eval Loop):
When you generate evaluation code and the user wants automatic execution:
1. Generate eval code and write to workspace with key: `generated_eval_code`
2. Invoke `eval_runner` sub-agent (reads generated_eval_code, executes, writes results)
3. Read `eval_results` from workspace
4. Synthesize findings and provide recommendations based on actual execution results

**IMPORTANT**: Write generated code to workspace BEFORE invoking eval_runner:
```
write_to_workspace(key="generated_eval_code", data={{
    "code": "import mlflow...",
    "description": "Evaluates retrieval quality",
    "scorers": ["RetrievalGroundedness", "custom_scorer"],
    "dataset_size": 50
}})
```

## Shared Workspace
Sub-agents write findings to a shared workspace:
{workspace_context}

## Output Requirements
When generating evaluation code:
1. Use MLflow 3 GenAI APIs (mlflow.genai.evaluate, not mlflow.evaluate)
2. Include all necessary imports
3. Add error handling
4. Make scripts runnable standalone
"""


def get_coordinator_system_prompt(workspace_context: str, experiment_id: str = "") -> str:
    """Generate coordinator system prompt dynamically from registry.

    This function imports the registry at runtime to avoid circular imports,
    and generates the coordinator prompt with up-to-date agent information.

    Args:
        workspace_context: Current workspace state string
        experiment_id: MLflow experiment ID (numeric string)

    Returns:
        Complete coordinator system prompt
    """
    # Import here to avoid circular import
    from .registry import AGENT_REGISTRY, get_workflow_order

    # Generate agent descriptions from registry
    agent_descriptions = []
    for i, (name, config) in enumerate(AGENT_REGISTRY.items(), 1):
        # Extract first line of description for brevity
        desc_lines = config.description.strip().split('\n')
        brief_desc = desc_lines[0].strip() if desc_lines else name

        # Build output keys info
        outputs = f"Writes: {', '.join(config.output_keys)}" if config.output_keys else "Writes tags to traces"
        reads = f"Reads: {', '.join(config.required_keys)}" if config.required_keys else "No dependencies"

        # Include max_turns guidance
        turns_info = f"Max turns: {config.max_turns}"

        agent_descriptions.append(f"""
{i}. **{name}** - {brief_desc}
   - {reads}
   - {outputs}
   - {turns_info}""")

    # Generate workflow order from dependencies
    workflow = get_workflow_order()
    workflow_steps = " → ".join(workflow)
    workflow_order = f"Recommended order: {workflow_steps}"

    return COORDINATOR_PROMPT_TEMPLATE.format(
        agent_descriptions="\n".join(agent_descriptions),
        workflow_order=workflow_order,
        workspace_context=workspace_context,
        experiment_id=experiment_id or "Not configured - use MLFLOW_EXPERIMENT_ID env var",
    )
