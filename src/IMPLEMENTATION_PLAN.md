# MLflow Eval Agent - Robustness Improvements

## Overview

This plan addresses issues identified in trace `tr-e53ef53d13b761affeeefb9487bb0a64`:

1. **Inter-Agent Communication Failure**: `context_engineer` didn't read `trace_analyst`'s outputs
2. **Validation Error**: `generated_eval_code` missing required `code` field
3. **Suboptimal Pattern**: Generated eval uses mock predict_fn instead of existing trace outputs

## Scope

| Priority | Description | Status |
|----------|-------------|--------|
| P1 | Critical Fixes - Validation & dependencies | In Progress |
| P2 | Prompt Improvements - I/O contracts | In Progress |
| P3 | Skill Integration | Deferred |
| P4 | State Machine | Deferred |

## Implementation Steps

### Step 1: Fix Workspace Schema
**File**: `src/workspace.py:119-126`

```python
class GeneratedEvalCode(BaseModel):
    """Schema for generated_eval_code workspace entry."""
    code: Optional[str] = Field(None, description="Python code for evaluation")
    code_path: Optional[str] = Field(None, description="Path where code was written")
    scorers: list[str] = Field(default_factory=list)
    dataset_size: Optional[int] = Field(None)
    description: Optional[str] = Field(None)

    model_config = {"extra": "allow"}

    @model_validator(mode='after')
    def validate_code_or_path(self) -> 'GeneratedEvalCode':
        if not self.code and not self.code_path:
            raise ValueError("Must provide either 'code' or 'code_path'")
        return self
```

### Step 2: Add Workspace Error Handling
**File**: `src/workspace.py:346-367`

- Update `write()` to return `tuple[bool, str]`
- Add `_get_schema_hint()` helper
- Enhance error messages with schema hints

### Step 3: Enforce Workspace Dependencies
**File**: `src/subagents/__init__.py`

```python
def validate_agent_can_run(
    agent_name: str,
    workspace: SharedWorkspace,
    strict: bool = True
) -> tuple[bool, list[str], str]:
    """Check if agent has required dependencies.

    Args:
        strict: If True, block execution when dependencies missing.
    """
    # ... implementation
```

Add `_get_dependency_producers()` to suggest which agent to run first.

### Step 4-7: Add I/O Contracts to Subagent Prompts

Each subagent gets an explicit I/O contract:

```
## I/O CONTRACT (READ THIS FIRST)

**READS FROM WORKSPACE**: [list with purpose]
**WRITES TO WORKSPACE**: [list - REQUIRED]
**DEPENDS ON**: [which agents must run first]
**CRITICAL WORKFLOW**: [step-by-step instructions]
```

**Files**:
- `src/subagents/trace_analyst.py` - First in pipeline, writes 4 keys
- `src/subagents/context_engineer.py` - Requires trace_analyst outputs
- `src/subagents/agent_architect.py` - Requires trace_analysis_summary
- `src/subagents/eval_runner.py` - Requires generated_eval_code

### Step 8: Update Coordinator Prompt
**File**: `src/subagents/coordinator.py:61-86`

Add dependency-aware invocation rules with examples:

```
**CORRECT sequence**:
1. Check workspace → empty
2. Invoke trace_analyst (no deps)
3. Wait for workspace writes
4. Check workspace → trace_analysis_summary exists
5. Invoke context_engineer (deps satisfied)

**WRONG sequence**:
1. Invoke context_engineer immediately
   → Fails: missing trace_analysis_summary
```

## Success Criteria

1. No invalid workspace states
2. Clear error messages when dependencies missing
3. Inter-agent communication works
4. `generated_eval_code` accepts both `code` and `code_path`
5. All subagents have explicit I/O contracts

## Files Modified

| File | Changes |
|------|---------|
| `src/workspace.py` | Schema fix, error handling |
| `src/subagents/__init__.py` | Strict dependency validation |
| `src/subagents/trace_analyst.py` | I/O contract |
| `src/subagents/context_engineer.py` | I/O contract |
| `src/subagents/agent_architect.py` | I/O contract |
| `src/subagents/eval_runner.py` | I/O contract |
| `src/subagents/coordinator.py` | Dependency checking |
