---
description: Run evaluation-driven optimization loop for agent prompts and context
argument-hint: "[--quick | --full] [--agent dspy|langgraph]"
model: claude-sonnet-4-5-20250929
---

Run context optimization loop: **$ARGUMENTS**

## Objective

Execute an evaluation-driven optimization cycle that:
1. Runs evaluation to establish current metrics
2. Analyzes traces and signatures for issues
3. Generates specific fix recommendations
4. Tracks optimization history for comparison

## Arguments

- `--quick`: Run quick evaluation (limited test cases, faster iteration)
- `--full`: Run full evaluation (all test cases, comprehensive)
- `--agent dspy`: Optimize DSPy multi-agent (default)
- `--agent langgraph`: Optimize LangGraph agent

Default: `--quick --agent dspy`

## Configuration

Customize thresholds by setting environment variables or editing this command:

| Metric | Default Target | Environment Variable |
|--------|---------------|---------------------|
| classifier_accuracy | 95% | `EVAL_CLASSIFIER_TARGET` |
| tool_selection_accuracy | 90% | `EVAL_TOOL_SELECTION_TARGET` |
| follows_instructions | 80% | `EVAL_INSTRUCTIONS_TARGET` |
| stage_latency | 30s | `EVAL_LATENCY_TARGET` |

## Workflow

### Step 1: Run Evaluation

Run your project's evaluation script. Examples:

**For DSPy agent:**
```bash
# Customize path to your evaluation script
python path/to/run_eval.py 2>&1
```

**For LangGraph agent:**
```bash
# Customize path to your evaluation script
python path/to/evaluate_app.py 2>&1
```

Capture the output and extract:
- Metric scores (classifier_accuracy, tool_selection_accuracy, follows_instructions)
- Trace IDs from the evaluation run
- Error messages and failures

### Step 2: Extract Metrics

Parse evaluation output for key metrics:

| Metric | Description | Target |
|--------|-------------|--------|
| classifier_accuracy | Query type correctly classified | 95%+ |
| tool_selection_accuracy | Correct tools selected | 90%+ |
| follows_instructions | Output format matches spec | 80%+ |
| stage_latency | Executor latency | <30s |

Report current values vs targets.

### Step 3: Analyze Root Causes

**For underperforming metrics, analyze:**

**If follows_instructions < 80%:**
- Run signature analysis: `python -m evaluation.analyze_signatures`
- Check ExecutorSignature.answer field description
- Look for format specification issues

**If tool_selection_accuracy < 90%:**
- Check ClassifierSignature.required_tools examples
- Verify tool name normalization in stage_scorers.py
- Analyze trace tool spans for mismatches

**If classifier_accuracy < 95%:**
- Check ClassifierSignature docstring clarity
- Review query_type mapping descriptions
- Look for ambiguous classification cases

**If latency > 30s:**
- Analyze trace spans for bottlenecks
- Check signature verbosity (total description chars)
- Consider context compression opportunities

### Step 4: Generate Fix Suggestions

For each identified issue, generate:

```markdown
## Fix Suggestion

**Issue**: [Description of the problem]
**Metric Impact**: [Which metric this affects]
**File**: [path/to/file.py]
**Line**: [line number if known]

**Current**:
```
[current code/text]
```

**Suggested**:
```
[suggested change]
```

**Rationale**: [Why this change should help]
```

### Step 5: Track Optimization History

Log this iteration to `optimization_history.json`:

```json
{
  "iterations": [
    {
      "timestamp": "2024-01-15T10:30:00Z",
      "metrics": {
        "classifier_accuracy": 1.0,
        "tool_selection_accuracy": 0.714,
        "follows_instructions": 0.375
      },
      "issues_found": 5,
      "fixes_applied": [],
      "trace_ids": ["tr-abc123"]
    }
  ]
}
```

## Output Format

```markdown
# Optimization Report - Iteration N

## Current Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| classifier_accuracy | 100% | 95%+ | ✅ |
| tool_selection_accuracy | 71.4% | 90%+ | ⚠️ |
| follows_instructions | 37.5% | 80%+ | 🔴 |
| executor_latency | 42s | <30s | ⚠️ |

## Root Cause Analysis

### follows_instructions (37.5% - Critical)

**Finding**: ExecutorSignature.answer has 15-line format spec
**Root Cause**: Complex markdown template in OutputField description
**Impact**: LLM struggles to match exact format

### tool_selection_accuracy (71.4% - Needs Work)

**Finding**: No examples in required_tools field
**Root Cause**: Tool selection relies on mapping text without concrete examples
**Impact**: Ambiguous tool selection for edge cases

## Recommended Fixes

### Fix 1: Simplify ExecutorSignature.answer format

**File**: `signatures/executor_signatures.py`

**Action**: Reduce answer description to essential format requirements

### Fix 2: Add examples to required_tools

**File**: `signatures/classifier_signatures.py`

**Action**: Add 2-3 concrete examples of query → tool mapping

## Next Steps

- [ ] Apply Fix 1 and re-run evaluation
- [ ] If follows_instructions improves, apply Fix 2
- [ ] Target: 70%+ follows_instructions, 85%+ tool_selection
```

## Integration

This command integrates with:
- `/eval:analyze-traces [experiment-id]` - Deep trace analysis
- `python -m evaluation.analyze_signatures` - Signature analysis
- `run_quick_eval.py` - Evaluation execution

## Example Usage

```
/optimize-context --quick
/optimize-context --full --agent dspy
/optimize-context --quick --agent langgraph
```

## MCP Server Setup

Required MCP configuration in `.claude/settings.local.json` or project settings:
```json
{
  "mcpServers": {
    "mlflow-mcp": {
      "command": "uv",
      "args": ["run", "--with", "mlflow[mcp]>=3.5.1", "mlflow", "mcp", "run"],
      "env": {
        "MLFLOW_TRACKING_URI": "databricks"
      }
    }
  }
}
```
