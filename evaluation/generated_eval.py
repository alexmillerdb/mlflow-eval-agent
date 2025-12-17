#!/usr/bin/env python3
"""MLflow Evaluation Script for Agent Quality Assessment

Evaluates agent responses for clarity, helpfulness, and safety.
"""

import mlflow
from mlflow.genai.scorers import Guidelines, Safety, scorer
from mlflow.entities import Feedback

# Set experiment
mlflow.set_tracking_uri("databricks")
experiment = mlflow.set_experiment(experiment_id="2945663160531572")
print(f"Using experiment: {experiment.name} (ID: {experiment.experiment_id})")

# Define custom scorers
@scorer
def response_clarity(outputs):
    """Check if response is clear and well-structured."""
    if not outputs:
        return Feedback(name="response_clarity", value=False, rationale="No output provided")
    
    response = outputs.get("response", str(outputs))
    
    # Check for repetitive content (indicator of quality issues)
    words = response.lower().split()
    if len(words) > 20:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            return Feedback(
                name="response_clarity", 
                value=False, 
                rationale=f"Response appears repetitive (unique word ratio: {unique_ratio:.2f})"
            )
    
    # Check for common error patterns
    error_patterns = ["API Error", "REQUEST_LIMIT_EXCEEDED", "429"]
    for pattern in error_patterns:
        if pattern in response:
            return Feedback(
                name="response_clarity",
                value=False,
                rationale=f"Response contains error: {pattern}"
            )
    
    return Feedback(name="response_clarity", value=True, rationale="Response is clear")


@scorer
def response_completeness(outputs):
    """Check if response provides complete information."""
    if not outputs:
        return Feedback(name="response_completeness", value=False, rationale="No output")
    
    response = outputs.get("response", str(outputs))
    
    # Check minimum length for a substantive response
    if len(response) < 50:
        return Feedback(
            name="response_completeness",
            value=False,
            rationale="Response too short to be complete"
        )
    
    # Check if response was truncated or interrupted
    truncation_indicators = ["...", "[truncated]", "exceeded", "limit"]
    for indicator in truncation_indicators:
        if indicator.lower() in response.lower()[-100:]:
            return Feedback(
                name="response_completeness",
                value=False,
                rationale=f"Response appears incomplete: found '{indicator}'"
            )
    
    return Feedback(name="response_completeness", value=True, rationale="Response appears complete")


# Build evaluation dataset from recent traces
print("Searching for traces...")
traces_df = mlflow.search_traces(
    experiment_ids=["2945663160531572"],
    max_results=10,
    order_by=["attributes.timestamp_ms DESC"]
)

print(f"Found {len(traces_df)} traces")

# Convert traces to evaluation format
eval_data = []
for idx, row in traces_df.iterrows():
    trace = row.get("trace")
    if trace and hasattr(trace, 'data'):
        # Get root span inputs/outputs
        root_span = trace.data.spans[0] if trace.data.spans else None
        if root_span:
            inputs = root_span.inputs or {}
            outputs = root_span.outputs or {}
            
            # Handle prompt extraction
            prompt = inputs.get("prompt", inputs.get("query", str(inputs)[:200]))
            
            eval_data.append({
                "inputs": {"query": prompt[:500] if isinstance(prompt, str) else str(prompt)[:500]},
                "outputs": outputs
            })

if not eval_data:
    print("No valid trace data found. Creating sample evaluation data.")
    eval_data = [
        {
            "inputs": {"query": "Analyze traces and optimize context"},
            "outputs": {"response": "I'll analyze the traces from your experiment."}
        }
    ]

print(f"Evaluating {len(eval_data)} records...")

# Define scorers
scorers = [
    response_clarity,
    response_completeness,
    Safety(),
    Guidelines(
        name="helpful",
        guidelines="The response must be helpful and informative, addressing the user's request with actionable information."
    ),
    Guidelines(
        name="professional",
        guidelines="The response must be professional and well-organized, avoiding unnecessary repetition."
    )
]

# Run evaluation
print("Running evaluation...")
results = mlflow.genai.evaluate(
    data=eval_data,
    scorers=scorers
)

print(f"\n=== Evaluation Complete ===")
print(f"Run ID: {results.run_id}")
print(f"\nMetrics:")
for metric, value in results.metrics.items():
    print(f"  {metric}: {value}")

print("\nView detailed results in MLflow UI")
