#!/usr/bin/env python
"""Run MLflow integration evaluations for mlflow-eval-agent framework.

This script runs a single comprehensive evaluation using the real agent and logs results to MLflow.
Experiment: /Users/alex.miller@databricks.com/mlflow-eval-agent-evaluations

Usage:
    # Run evaluation (uses MLFLOW_EXPERIMENT_ID from environment)
    uv run python evaluation/run_evaluation.py

    # Run with specific experiment ID
    uv run python evaluation/run_evaluation.py --experiment-id 123456789

    # Or via environment variable
    EVAL_EXPERIMENT_ID=123456789 uv run python evaluation/run_evaluation.py

    # Dry run (print data, don't execute)
    uv run python evaluation/run_evaluation.py --dry-run
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()


def setup_mlflow():
    """Configure MLflow tracking."""
    import mlflow

    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment("/Users/alex.miller@databricks.com/mlflow-eval-agent-evaluations")

    return mlflow


def build_combined_dataset():
    """Build a combined dataset with metadata about test case types.

    Returns:
        List of evaluation records with 'inputs', 'expectations', and metadata.
    """
    from evaluation.test_data import (
        AGENT_ROUTING_CASES,
        INTEGRATION_EVAL_DATA,
        ORDERING_CASES,
        TOOL_CALL_CASES,
    )

    combined = []

    # Add routing cases with metadata
    for case in AGENT_ROUTING_CASES:
        combined.append({
            "inputs": case["inputs"],
            "expectations": {
                **case["expectations"],
                "test_type": "routing",
                "test_source": "AGENT_ROUTING_CASES",
            },
        })

    # Add ordering cases with metadata
    for case in ORDERING_CASES:
        combined.append({
            "inputs": case["inputs"],
            "expectations": {
                **case["expectations"],
                "test_type": "ordering",
                "test_source": "ORDERING_CASES",
            },
        })

    # Add tool call cases with metadata
    for case in TOOL_CALL_CASES:
        combined.append({
            "inputs": case["inputs"],
            "expectations": {
                **case["expectations"],
                "test_type": "tools",
                "test_source": "TOOL_CALL_CASES",
            },
        })

    # Add integration cases with metadata
    for case in INTEGRATION_EVAL_DATA:
        combined.append({
            "inputs": case["inputs"],
            "expectations": {
                **case["expectations"],
                "test_type": "integration",
                "test_source": "INTEGRATION_EVAL_DATA",
            },
        })

    return combined


def create_predict_fn(experiment_id: str = None):
    """Create predict function for evaluation.

    CRITICAL: predict_fn receives **unpacked inputs as kwargs (per GOTCHAS.md)

    Args:
        experiment_id: Optional experiment ID to use for trace searches.
                       If provided, overrides MLFLOW_EXPERIMENT_ID in environment.
    """
    # Set experiment ID in environment if provided
    if experiment_id:
        os.environ["MLFLOW_EXPERIMENT_ID"] = experiment_id

    def predict_fn(query: str, context: str = None):
        """Predict function that runs the full agent.

        Args:
            query: The user query (unpacked from inputs)
            context: Optional context (unpacked from inputs)

        Returns:
            dict with response and metadata
        """
        from src.agent import MLflowEvalAgent
        from src.config import EvalAgentConfig

        try:
            config = EvalAgentConfig.from_env(validate=False)
            agent = MLflowEvalAgent(config)

            async def run_query():
                results = []
                async for result in agent.query(query):
                    results.append(result)
                return results

            results = asyncio.run(run_query())

            response_text = ""
            for r in results:
                if hasattr(r, "response") and r.response:
                    response_text += r.response

            return {
                "response": response_text or "No response generated",
                "result_count": len(results),
            }

        except Exception as e:
            return {
                "response": f"Error: {str(e)}",
                "error": str(e),
            }

    return predict_fn


def run_evaluation(mlflow, predict_fn, dry_run=False):
    """Run comprehensive evaluation with all test cases and scorers."""
    from mlflow.genai.scorers import Guidelines, Safety

    from evaluation.scorers import (
        agent_routing_accuracy,
        execution_order_scorer,
        tool_selection_accuracy,
        workspace_io_correctness,
    )

    # Build combined dataset
    eval_data = build_combined_dataset()

    print("\n=== MLflow Eval Agent - Comprehensive Evaluation ===")
    print(f"Total test cases: {len(eval_data)}")

    # Show breakdown by type
    type_counts = {}
    for case in eval_data:
        test_type = case["expectations"].get("test_type", "unknown")
        type_counts[test_type] = type_counts.get(test_type, 0) + 1

    print("\nTest case breakdown:")
    for test_type, count in sorted(type_counts.items()):
        print(f"  - {test_type}: {count}")

    if dry_run:
        print("\nTest cases:")
        for i, case in enumerate(eval_data):
            test_type = case["expectations"].get("test_type", "unknown")
            query = case["inputs"]["query"][:50]
            print(f"  [{i+1}] ({test_type}) {query}...")
        return None

    print("\nRunning evaluation...")

    with mlflow.start_run(run_name="comprehensive_eval"):
        results = mlflow.genai.evaluate(
            data=eval_data,
            predict_fn=predict_fn,
            scorers=[
                # Custom scorers for agent behavior
                agent_routing_accuracy,
                execution_order_scorer,
                tool_selection_accuracy,
                workspace_io_correctness,
                # Built-in scorers for response quality
                Safety(),
                Guidelines(
                    name="helpful",
                    guidelines="The response must be helpful and informative",
                ),
                Guidelines(
                    name="clear",
                    guidelines="The response must be clear and well-structured",
                ),
            ],
        )

    print(f"\n{'='*50}")
    print(f"Run ID: {results.run_id}")
    print(f"\nMetrics Summary:")
    for metric, value in sorted(results.metrics.items()):
        print(f"  {metric}: {value}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run MLflow integration evaluations for mlflow-eval-agent"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print test data without running evaluation",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=os.environ.get("EVAL_EXPERIMENT_ID"),
        help="Experiment ID for the agent to search traces from. "
        "Can also be set via EVAL_EXPERIMENT_ID environment variable.",
    )

    args = parser.parse_args()

    # Resolve experiment ID: CLI arg > EVAL_EXPERIMENT_ID > MLFLOW_EXPERIMENT_ID
    experiment_id = args.experiment_id or os.environ.get("MLFLOW_EXPERIMENT_ID")

    print("MLflow Evaluation Agent - Integration Evaluation Runner")
    print("=" * 50)
    print(f"Evaluation Experiment: /Users/alex.miller@databricks.com/mlflow-eval-agent-evaluations")
    print(f"Agent Experiment ID: {experiment_id or '(not set - agent will use default)'}")
    print(f"Dry run: {args.dry_run}")

    mlflow = setup_mlflow()
    predict_fn = create_predict_fn(experiment_id=experiment_id)

    result = run_evaluation(mlflow, predict_fn, args.dry_run)

    if not args.dry_run and result:
        print("\n" + "=" * 50)
        print("Evaluation Complete!")
        print(f"\nRun ID: {result.run_id}")
        print(
            "\nView results in MLflow UI: "
            "/Users/alex.miller@databricks.com/mlflow-eval-agent-evaluations"
        )


if __name__ == "__main__":
    main()
