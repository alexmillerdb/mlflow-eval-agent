---
name: mlflow-evaluation
description: MLflow 3 GenAI evaluation patterns for building datasets, creating scorers, and running evaluations. Use when (1) generating evaluation code with mlflow.genai.evaluate(), (2) creating custom @scorer functions, (3) building evaluation datasets from traces, (4) using built-in scorers (Guidelines, Correctness, Safety, RetrievalGroundedness), (5) debugging evaluation failures or API errors, (6) setting up production monitoring with registered scorers, (7) analyzing traces for performance or errors. Always reference ./references/ for correct API usage.
---

# MLflow 3 GenAI Evaluation

## Workflow

1. **Before writing code**: Read `GOTCHAS.md` to avoid common mistakes
2. **For API signatures**: Consult `CRITICAL-interfaces.md`
3. **For patterns**: Choose the relevant reference file below

## Reference Files

| Task | Reference | When to Read |
|------|-----------|--------------|
| API signatures & schemas | `CRITICAL-interfaces.md` | Before writing any evaluation code |
| Common mistakes | `GOTCHAS.md` | Before writing any code (critical) |
| Running evaluations | `patterns-evaluation.md` | Setting up mlflow.genai.evaluate() |
| Custom scorers | `patterns-scorers.md` | Creating @scorer functions or class-based scorers |
| Building datasets | `patterns-datasets.md` | Creating eval data from traces or scratch |
| Trace analysis | `patterns-trace-analysis.md` | Debugging, profiling, or analyzing agent behavior |
| Context optimization | `patterns-context-optimization.md` | Optimizing token usage in agents |
| User workflows | `user-journeys.md` | Step-by-step guides for common scenarios |

## Quick Start

```python
import mlflow
from mlflow.genai.scorers import Guidelines, Safety

mlflow.set_experiment("/Shared/my-experiment")

results = mlflow.genai.evaluate(
    data=[{"inputs": {"query": "What is MLflow?"}}],
    predict_fn=my_app,  # receives **unpacked inputs as kwargs
    scorers=[Safety(), Guidelines(name="helpful", guidelines="Be helpful")]
)
```

See `CRITICAL-interfaces.md` for full API details and data schema.

## Critical Gotchas (Top 3)

1. **Use `mlflow.genai.evaluate()`** - NOT `mlflow.evaluate()` (deprecated for GenAI)
2. **`predict_fn` receives unpacked kwargs** - `predict_fn(**inputs)` not `predict_fn(inputs)`
3. **Data requires nested structure** - `{"inputs": {"query": "..."}}` not `{"query": "..."}`

See `GOTCHAS.md` for the complete list of 15+ common mistakes.
