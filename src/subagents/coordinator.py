"""Coordinator prompt generation for MLflow Evaluation Agent."""

from .registry import AgentConfig


# =============================================================================
# COORDINATOR CONFIG
# =============================================================================

COORDINATOR_CONFIG = AgentConfig(
    name="coordinator",
    description="Main orchestration agent that coordinates sub-agents and synthesizes findings.",
    prompt_template="{workspace_context}",  # Coordinator uses its own prompt template below

    # Coordinator doesn't require anything initially, reads from sub-agent outputs
    required_keys=[],
    optional_keys=[
        "trace_analysis_summary",
        "error_patterns",
        "performance_metrics",
        "context_recommendations",
        "eval_results",
    ],
    output_keys=[],  # Coordinator doesn't write to workspace

    # Token budgets for selective context injection
    total_token_budget=4000,  # ~16K chars for workspace context
    key_token_limits={
        "trace_analysis_summary": 1000,
        "error_patterns": 500,
        "performance_metrics": 500,
        "context_recommendations": 1000,
        "eval_results": 1000,
    },
)


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

## CRITICAL: Delegation Rules

**ALWAYS delegate trace analysis to sub-agents:**
- ❌ DON'T: Call get_trace multiple times yourself
- ✅ DO: Invoke trace_analyst for batch analysis
- ❌ DON'T: Accumulate full trace data in your context
- ✅ DO: Read compressed findings from workspace
- ❌ NEVER use output_mode='full' when analyzing multiple traces
- ✅ ALWAYS use output_mode='aggressive' or 'summary'

**When user asks to "analyze N traces":**
1. Invoke trace_analyst with the query
2. Wait for workspace to populate (trace_analyst will write findings)
3. Read trace_analysis_summary from workspace
4. Synthesize findings (DO NOT re-fetch traces)

**When sub-agent fails or reports issues:**
1. ❌ DO NOT fall back to fetching full traces yourself
2. ❌ DO NOT compensate by doing the sub-agent's job with full data
3. ✅ Report the sub-agent error to the user
4. ✅ Ask user to check sub-agent configuration or retry

**Context Budget Awareness:**
- Each full trace can be 10K+ tokens
- Analyzing 5+ traces requires sub-agent delegation
- Sub-agents compress findings → workspace → you read summaries
- NEVER fetch traces with output_mode='full' unless analyzing a single trace

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

## Available Skills

**IMPORTANT**: Before generating code, invoke the relevant skill to load correct API patterns.

| Skill | Invoke With | Use When |
|-------|-------------|----------|
| mlflow-evaluation | `Skill(skill="mlflow-evaluation")` | Generating evaluation code, creating scorers, building datasets |
| trace-analysis | `Skill(skill="trace-analysis")` | Deep trace analysis, profiling latency, debugging failures |
| context-engineering | `Skill(skill="context-engineering")` | Context optimization, prompt engineering, token budgets |

Skills contain working examples and common gotchas. Always invoke `mlflow-evaluation` before writing evaluation code.

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
