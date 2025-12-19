"""
Evaluation script for Multi-Genie Orchestrator agent.

This script runs MLflow GenAI evaluation on pre-computed outputs from production traces.
No predict_fn is needed since we're evaluating existing responses.

Usage:
    python run_eval.py                  # Run full evaluation
    python run_eval.py --preset core    # Run only core scorers
    python run_eval.py --dry-run        # Print config without running

Scorers:
    - Safety: Ensures no harmful content
    - RelevanceToQuery: Ensures response addresses the query
    - correct_routing: Verifies appropriate Genie routing
    - efficient_routing: Checks for unnecessary Genie calls
    - data_presentation: Evaluates tabular data presentation quality
    - summary_quality: Checks workflow summary accuracy
    - genie_call_count: Counts Genies called
    - has_workflow_summary: Checks for workflow summary presence
    - response_has_data_table: Checks for markdown tables
    - response_length_words: Word count analysis
"""

import argparse
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
import mlflow.genai

from evaluation.eval_dataset import get_eval_data_for_mlflow, EVAL_DATASET_WITH_OUTPUTS
from evaluation.scorers import get_scorers, ALL_SCORERS


def setup_mlflow(experiment_id: str = None):
    """Configure MLflow experiment for evaluation results."""
    if experiment_id:
        mlflow.set_experiment(experiment_id=experiment_id)
    else:
        # Use environment variable or default
        exp_id = os.environ.get("MLFLOW_EXPERIMENT_ID", "2181280362153689")
        mlflow.set_experiment(experiment_id=exp_id)

    print(f"MLflow experiment: {mlflow.get_experiment(mlflow.active_run().info.experiment_id).name if mlflow.active_run() else 'Not set'}")


def run_evaluation(
    preset: str = "all",
    dry_run: bool = False,
    verbose: bool = True
):
    """
    Run evaluation on the Multi-Genie Orchestrator agent.

    Args:
        preset: Scorer preset ("all", "core", "guidelines", "metrics")
        dry_run: If True, print config without running evaluation
        verbose: If True, print detailed progress

    Returns:
        EvaluationResult object with metrics and per-row results
    """
    # Get evaluation data
    eval_data = get_eval_data_for_mlflow()

    if verbose:
        print(f"\n{'='*60}")
        print("Multi-Genie Orchestrator Evaluation")
        print(f"{'='*60}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Dataset size: {len(eval_data)} cases")
        print(f"Scorer preset: {preset}")

    # Get scorers
    scorers = get_scorers(preset)

    if verbose:
        print(f"\nScorers ({len(scorers)}):")
        for s in scorers:
            name = getattr(s, "name", getattr(s, "__name__", str(s)))
            print(f"  - {name}")

    # Preview data
    if verbose:
        print(f"\nSample data (first case):")
        sample = eval_data[0] if eval_data else {}
        print(f"  Input: {str(sample.get('inputs', {}))[:100]}...")
        print(f"  Output keys: {list(sample.get('outputs', {}).keys())}")

    if dry_run:
        print("\n[DRY RUN] Would run evaluation with above config")
        return None

    # Run evaluation
    print(f"\nRunning evaluation...")

    try:
        results = mlflow.genai.evaluate(
            data=eval_data,
            scorers=scorers
            # No predict_fn - using pre-computed outputs
        )

        print(f"\nEvaluation complete!")
        print(f"Run ID: {results.run_id}")

        # Print metrics summary
        if hasattr(results, 'metrics') and results.metrics:
            print(f"\n{'='*40}")
            print("Metrics Summary")
            print(f"{'='*40}")
            for metric_name, metric_value in sorted(results.metrics.items()):
                if isinstance(metric_value, float):
                    print(f"  {metric_name}: {metric_value:.4f}")
                else:
                    print(f"  {metric_name}: {metric_value}")

        return results

    except Exception as e:
        print(f"\nError running evaluation: {e}")
        raise


def print_detailed_results(results):
    """Print detailed per-row results."""
    if not hasattr(results, 'tables') or not results.tables:
        print("No detailed results available")
        return

    print(f"\n{'='*60}")
    print("Detailed Results")
    print(f"{'='*60}")

    # Get the main results table
    if 'eval_results' in results.tables:
        df = results.tables['eval_results']
        print(f"\nResults table shape: {df.shape}")
        print(f"\nColumns: {list(df.columns)}")

        # Show first few rows
        print(f"\nFirst {min(3, len(df))} rows:")
        print(df.head(3).to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Run MLflow GenAI evaluation on Multi-Genie Orchestrator"
    )
    parser.add_argument(
        "--preset",
        choices=["all", "core", "guidelines", "metrics"],
        default="all",
        help="Scorer preset to use (default: all)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config without running evaluation"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Print detailed per-row results"
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="MLflow experiment ID (default: from env or 2181280362153689)"
    )

    args = parser.parse_args()

    # Setup MLflow
    if not args.dry_run:
        setup_mlflow(args.experiment_id)

    # Run evaluation
    results = run_evaluation(
        preset=args.preset,
        dry_run=args.dry_run,
        verbose=not args.quiet
    )

    # Print detailed results if requested
    if args.detailed and results:
        print_detailed_results(results)

    return results


if __name__ == "__main__":
    main()
