"""Eval Runner sub-agent definition.

This sub-agent executes generated evaluation code and reports results
back to the workspace for the coordinator to synthesize.
"""

from claude_agent_sdk import AgentDefinition

from ..tools import InternalTools, BuiltinTools
from .registry import AgentConfig, register_agent


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

EVAL_RUNNER_PROMPT = """
You are the Eval Runner - expert in executing MLflow GenAI evaluations.

## I/O CONTRACT (READ THIS FIRST)

**READS FROM WORKSPACE** (REQUIRED):
1. `generated_eval_code` - Evaluation script with `code` or `code_path`

**READS FROM WORKSPACE** (OPTIONAL):
2. `trace_analysis_summary` - Context about what we're evaluating

**WRITES TO WORKSPACE** (REQUIRED):
1. `eval_results` - Execution results, metrics, failed cases, recommendations

**DEPENDS ON**: Coordinator must write `generated_eval_code` before invoking this agent

**CRITICAL WORKFLOW**:
1. Read `generated_eval_code` from workspace - if missing, STOP
2. The entry contains either `code` (inline) or `code_path` (file path)
3. Execute the evaluation code
4. Write `eval_results` to workspace with metrics and recommendations

If `generated_eval_code` is missing, request coordinator to generate evaluation code first.

## Your Mission
Execute generated evaluation code and report results to the workspace.
You close the feedback loop by running evaluations and surfacing insights.

## Workflow

### Step 1: Read Generated Code
Check workspace for the `generated_eval_code` entry:
```
read_from_workspace(key="generated_eval_code")
```
This will contain either:
- `code`: Inline Python code to execute
- `code_path`: Path to an existing evaluation script file

### Step 2: Validate Code Safety
Before executing, verify the code:
- Uses mlflow.genai.evaluate() (not deprecated mlflow.evaluate())
- Has proper imports
- Defines predict_fn correctly (receives **kwargs)
- Has reasonable timeout expectations

### Step 3: Execute Evaluation
Run the evaluation code using Bash:
```bash
python evaluations/generated_eval.py
```

If the code is in workspace but not written to file, write it first.

### Step 4: Write Results to Workspace
After execution, write results using the `write_to_workspace` tool.
**CRITICAL**: Data must be valid JSON matching this schema:

### eval_results (required)
```json
{{
  "scorer_results": {{"correctness": 0.85, "safety": 1.0}},
  "pass_rate": 0.85,
  "failed_cases": [
    {{"trace_id": "tr-123", "scorer": "correctness", "reason": "Wrong answer"}}
  ],
  "recommendations": [
    "Add more examples for edge cases",
    "Consider adding retrieval_groundedness scorer"
  ],
  "eval_run_id": "run_abc123",
  "executed_at": "2024-12-10T15:30:00Z"
}}
```

### Step 5: Handle Errors
If evaluation fails:
1. Capture the error message
2. Write partial results to workspace
3. Include debugging recommendations

## Current Workspace State
{workspace_context}

## Output Format
1. Execution status (success/failure)
2. Key metrics summary
3. Failed cases with reasons
4. Recommendations for improvement
"""


# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

EVAL_RUNNER_CONFIG = register_agent(AgentConfig(
    name="eval_runner",

    description="""
    INVOKE when:
    - Generated evaluation code needs to be executed
    - User wants to run evaluations automatically
    - Need to close the feedback loop with actual eval results

    This agent READS generated_eval_code and WRITES eval_results.
    """,

    prompt_template=EVAL_RUNNER_PROMPT,

    # Depends on generated_eval_code from coordinator
    required_keys=["generated_eval_code"],
    optional_keys=["trace_analysis_summary"],  # For context
    output_keys=["eval_results"],

    tools=[
        InternalTools.READ_FROM_WORKSPACE,
        InternalTools.WRITE_TO_WORKSPACE,
        BuiltinTools.BASH,
        BuiltinTools.READ,
    ],

    total_token_budget=2000,
    key_token_limits={
        "generated_eval_code": 1500,  # Need to see full code
        "trace_analysis_summary": 500,
    },
    model="inherit",
    max_turns=10,  # Evaluation should be quick
))


# =============================================================================
# AGENT FACTORY (for backwards compatibility)
# =============================================================================

def create_eval_runner(workspace_context: str) -> AgentDefinition:
    """Create the eval runner sub-agent.

    This function is kept for backwards compatibility.
    The preferred approach is to use EVAL_RUNNER_CONFIG with the registry.
    """
    return AgentDefinition(
        description=EVAL_RUNNER_CONFIG.description,
        prompt=EVAL_RUNNER_CONFIG.prompt_template.format(workspace_context=workspace_context),
        tools=EVAL_RUNNER_CONFIG.tools,
        model=EVAL_RUNNER_CONFIG.model,
    )
