---
description: Run complete test cycle from mock to Databricks
argument-hint: "<experiment-id>"
---

Run the full testing ladder for production readiness validation. This command executes all test levels sequentially, stopping on first failure.

## Usage

```bash
# Full cycle with real experiment
/test-full-cycle 1234567890123456789

# The experiment ID is used for Levels 3-5 (real MLflow integration)
```

## Workflow

### Step 1: Unit Tests
```bash
uv run pytest tests/ -v
```
**Success**: All tests pass

### Step 2: Import Verification
```bash
uv run python -c "
from src.core.config import Config
from src.core.runtime import detect_runtime, RuntimeContext
from src.agent.agent import MLflowAgent, setup_mlflow
from src.agent.autonomous import run_autonomous
from src.agent.tools import create_tools
print('All imports OK')
print(f'Tools: {len(create_tools())}')
"
```
**Success**: No import errors, correct tool count

### Step 3: Mock Integration
```bash
uv run python -m src.cli test tools mlflow_query --operation search -e 123 --mock
uv run python -m src.cli test initializer -e 123 --mock
```
**Success**: Tools execute without error, mock initializer completes

### Step 4: Initializer Test
```bash
uv run python -m src.cli test initializer -e $ARGUMENTS
```
**Success**: Creates `eval_tasks.json` and `state/analysis.json`

### Step 5: Worker Test
```bash
uv run python -m src.cli test worker -e $ARGUMENTS --session-dir <path-from-step-4>
```
**Success**: Completes at least 1 task, creates artifacts in `evaluation/`

### Step 6: Integration Test
```bash
uv run python -m src.cli test integration -e $ARGUMENTS --max-iterations 3
```
**Success**: Completes with >0 tasks done

### Step 7: Performance Analysis
```bash
/analyze-trace <trace-id>
/analyze-tokens <trace-id>
```
**Success**:
- Cache efficiency >50%
- No runaway context growth
- Cost per session reasonable (<$1.50 for initializer, <$0.75 for worker)

## Success Criteria Summary

| Step | Pass Criteria |
|------|---------------|
| 1 | All pytest tests pass |
| 2 | All imports resolve (no circular imports) |
| 3 | Mock tools and initializer complete |
| 4 | `eval_tasks.json` + `state/analysis.json` exist with required fields |
| 5 | At least one task marked completed, artifacts created |
| 6 | At least one task completed across iterations |
| 7 | Cache efficiency >50%, stable token growth |

## Error Debugging

When a step fails:

1. **Capture trace ID** from test output
2. **Analyze trace**: `/analyze-trace <trace-id>`
   - Check span errors
   - Identify bottlenecks
   - Review tool call sequence
3. **Analyze tokens**: `/analyze-tokens <trace-id>`
   - Check cache efficiency (>50% = good)
   - Identify context growth issues
4. **Fix issue** based on analysis
5. **Re-run failing step** to verify fix
6. **Continue** from where you left off

## Example Run

```bash
/test-full-cycle 1234567890123456789

# Output:
# ==========================================
# Full Test Cycle - Experiment: 1234567890123456789
# ==========================================
#
# [1/7] Unit Tests...
#   ✓ 42 tests passed
#
# [2/7] Import Verification...
#   ✓ All imports OK (9 tools)
#
# [3/7] Mock Integration...
#   ✓ Mock tools working
#   ✓ Mock initializer completed
#
# [4/7] Initializer Test...
#   ✓ eval_tasks.json created (4 tasks)
#   ✓ state/analysis.json created
#   Trace: tr-init-abc123
#
# [5/7] Worker Test...
#   ✓ Task "dataset" completed
#   ✓ evaluation/eval_dataset.py created
#   Trace: tr-worker-def456
#
# [6/7] Integration Test...
#   ✓ 2/4 tasks completed in 3 iterations
#   Trace: tr-integration-ghi789
#
# [7/7] Performance Analysis...
#   Cache efficiency: 67%
#   Total cost: $3.45
#   Context growth: stable
#
# ==========================================
# Full Cycle: PASS
# ==========================================
```

## Key Metrics to Watch

| Metric | Good | Warning | Bad |
|--------|------|---------|-----|
| Cache efficiency | >50% | 30-50% | <30% |
| Initializer cost | <$1.50 | $1.50-$2.50 | >$2.50 |
| Worker cost | <$0.75 | $0.75-$1.25 | >$1.25 |
| Context growth | Stable | Slow growth | Runaway |
| Task completion | >50% | 25-50% | <25% |
