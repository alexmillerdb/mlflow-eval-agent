"""
Skill Evaluation Pipeline for benchmark evaluation.

This module provides the SkillEvaluationPipeline class that:
- Loads ground truth examples from YAML
- Builds evaluation datasets in MLflow format
- Runs mlflow.genai.evaluate() with configurable scorers
- Logs results to MLflow experiments
- Checks quality gates
- Saves baselines for regression detection
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import os

import mlflow
import mlflow.genai

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.skills.mlflow_evaluation.ground_truth import load_ground_truth
from benchmarks.skills.mlflow_evaluation.config import BenchmarkConfig, get_config
from benchmarks.skills.mlflow_evaluation.scorers import get_scorers


def setup_databricks_auth(require_llm: bool = False) -> bool:
    """
    Set up and validate Databricks authentication for LLM-based scorers.

    In Databricks runtime, authentication is handled automatically via OAuth.
    Locally, DATABRICKS_HOST and either DATABRICKS_TOKEN or DATABRICKS_CONFIG_PROFILE
    must be set.

    Args:
        require_llm: If True, raise error when auth is missing. If False, just warn.

    Returns:
        True if auth is configured, False otherwise

    Raises:
        ValueError: If require_llm=True and auth is not configured
    """
    # Check if running in Databricks runtime (auth handled automatically)
    if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        print("Running in Databricks environment - using automatic OAuth auth")
        return True

    # Local environment: check for required auth config
    host = os.environ.get("DATABRICKS_HOST")
    token = os.environ.get("DATABRICKS_TOKEN")
    profile = os.environ.get("DATABRICKS_CONFIG_PROFILE")

    if host and (token or profile):
        auth_method = "token" if token else f"profile '{profile}'"
        print(f"Using Databricks auth: {host} ({auth_method})")
        return True

    # Auth not configured
    if require_llm:
        raise ValueError(
            "Databricks authentication required for LLM-based scorers (Tier 2/3).\n"
            "Set environment variables:\n"
            "  DATABRICKS_HOST=https://your-workspace.cloud.databricks.com\n"
            "  DATABRICKS_TOKEN=dapi...  (or DATABRICKS_CONFIG_PROFILE=default)\n"
            "\n"
            "Or use --preset quick to skip LLM scorers."
        )

    print("Warning: Databricks auth not configured. LLM-based scorers may fail.")
    return False


class SkillEvaluationPipeline:
    """Pipeline for evaluating skill benchmark examples."""

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """
        Initialize the evaluation pipeline.

        Args:
            config: Benchmark configuration. If None, uses default config.
        """
        self.config = config or get_config()
        self.ground_truth = []
        self.eval_dataset = []
        self.results = None

    def load_ground_truth(self) -> list:
        """Load verified ground truth examples from YAML file."""
        self.ground_truth = load_ground_truth()
        print(f"Loaded {len(self.ground_truth)} ground truth examples")
        return self.ground_truth

    def build_eval_dataset(self) -> list[dict]:
        """
        Build evaluation dataset from ground truth examples.

        Converts GroundTruthExample objects to MLflow evaluation format:
        {
            "inputs": {"prompt": "..."},
            "outputs": {"response": "..."},
        }
        """
        if not self.ground_truth:
            self.load_ground_truth()

        self.eval_dataset = []
        for example in self.ground_truth:
            record = example.to_eval_record()
            self.eval_dataset.append(record)

        # Apply max_examples limit if configured
        if self.config.max_examples is not None:
            self.eval_dataset = self.eval_dataset[:self.config.max_examples]

        print(f"Built evaluation dataset with {len(self.eval_dataset)} examples")
        return self.eval_dataset

    def setup_mlflow(self):
        """Configure MLflow tracking."""
        mlflow.set_tracking_uri(self.config.tracking_uri)
        mlflow.set_experiment(self.config.experiment_name)
        print(f"MLflow tracking: {self.config.tracking_uri}")
        print(f"Experiment: {self.config.experiment_name}")

    def run_evaluation(self, run_name: Optional[str] = None) -> "mlflow.genai.EvaluationResult":
        """
        Run MLflow GenAI evaluation on the ground truth dataset.

        Args:
            run_name: Optional name for the MLflow run

        Returns:
            EvaluationResult with metrics and per-row results
        """
        if not self.eval_dataset:
            self.build_eval_dataset()

        self.setup_mlflow()

        # Check if preset requires LLM and validate auth
        requires_llm = self.config.scorer_preset in ("full", "tier2", "tier3", "all")
        if requires_llm:
            setup_databricks_auth(require_llm=True)

        # Get scorers for the configured preset with model
        scorers = get_scorers(self.config.scorer_preset, model=self.config.judge_model)
        print(f"Using {len(scorers)} scorers (preset: {self.config.scorer_preset})")
        if requires_llm:
            print(f"Judge model: {self.config.judge_model}")

        # Generate run name if not provided
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{self.config.skill_name}_v{self.config.skill_version}_{timestamp}"

        print(f"\nRunning evaluation: {run_name}")
        print(f"Dataset size: {len(self.eval_dataset)} examples")
        print("-" * 50)

        with mlflow.start_run(run_name=run_name):
            # Log configuration tags
            mlflow.set_tags(self.config.tags)

            # Run evaluation
            self.results = mlflow.genai.evaluate(
                data=self.eval_dataset,
                scorers=scorers,
                # No predict_fn - using pre-computed outputs from ground truth
            )

            # Log additional metadata
            mlflow.log_param("num_examples", len(self.eval_dataset))
            mlflow.log_param("scorer_preset", self.config.scorer_preset)
            mlflow.log_param("num_scorers", len(scorers))

        print(f"\nEvaluation complete!")
        print(f"Run ID: {self.results.run_id}")

        return self.results

    def print_metrics(self):
        """Print metrics summary from the evaluation results."""
        if self.results is None:
            print("No results available. Run evaluation first.")
            return

        print("\n" + "=" * 50)
        print("Metrics Summary")
        print("=" * 50)

        for metric_name, metric_value in sorted(self.results.metrics.items()):
            if isinstance(metric_value, float):
                print(f"  {metric_name}: {metric_value:.4f}")
            else:
                print(f"  {metric_name}: {metric_value}")

    def check_quality_gates(self) -> tuple[bool, list[str]]:
        """
        Check if evaluation results pass all quality gates.

        Returns:
            (passed, failures): Boolean and list of failed gate descriptions
        """
        if self.results is None:
            raise ValueError("No results available. Run evaluation first.")

        passed, failures = self.config.check_quality_gates(self.results.metrics)

        if passed:
            print("\n✓ All quality gates passed!")
        else:
            print("\n✗ Quality gates failed:")
            for failure in failures:
                print(f"  - {failure}")

        return passed, failures

    def save_baseline(self, baseline_path: Optional[str] = None):
        """
        Save current metrics as baseline for future regression detection.

        Args:
            baseline_path: Path to save baseline JSON. Defaults to baselines/current.json
        """
        if self.results is None:
            raise ValueError("No results available. Run evaluation first.")

        if baseline_path is None:
            baseline_dir = Path(__file__).parent.parent / "benchmarks" / "skills" / "mlflow-evaluation" / "baselines"
            baseline_dir.mkdir(exist_ok=True)
            baseline_path = baseline_dir / "current.json"

        baseline = {
            "version": self.config.skill_version,
            "timestamp": datetime.now().isoformat(),
            "run_id": self.results.run_id,
            "metrics": self.results.metrics,
            "scorer_preset": self.config.scorer_preset,
            "num_examples": len(self.eval_dataset),
        }

        with open(baseline_path, "w") as f:
            json.dump(baseline, f, indent=2, default=str)

        print(f"\nBaseline saved to: {baseline_path}")

    def load_baseline(self, baseline_path: Optional[str] = None) -> Optional[dict]:
        """
        Load baseline metrics for comparison.

        Args:
            baseline_path: Path to baseline JSON. Defaults to baselines/current.json

        Returns:
            Baseline dict or None if not found
        """
        if baseline_path is None:
            baseline_path = Path(__file__).parent.parent / "benchmarks" / "skills" / "mlflow-evaluation" / "baselines" / "current.json"

        if not Path(baseline_path).exists():
            print(f"No baseline found at: {baseline_path}")
            return None

        with open(baseline_path) as f:
            baseline = json.load(f)

        print(f"Loaded baseline from: {baseline_path}")
        print(f"  Version: {baseline.get('version')}")
        print(f"  Timestamp: {baseline.get('timestamp')}")

        return baseline

    def compare_to_baseline(self, baseline: dict) -> dict:
        """
        Compare current results to baseline metrics.

        Args:
            baseline: Baseline dict from load_baseline()

        Returns:
            Dictionary of metric comparisons
        """
        if self.results is None:
            raise ValueError("No results available. Run evaluation first.")

        baseline_metrics = baseline.get("metrics", {})
        current_metrics = self.results.metrics

        comparisons = {}
        for metric in set(baseline_metrics.keys()) | set(current_metrics.keys()):
            baseline_val = baseline_metrics.get(metric)
            current_val = current_metrics.get(metric)

            if baseline_val is not None and current_val is not None:
                if isinstance(baseline_val, (int, float)) and isinstance(current_val, (int, float)):
                    diff = current_val - baseline_val
                    pct = (diff / baseline_val * 100) if baseline_val != 0 else 0
                    comparisons[metric] = {
                        "baseline": baseline_val,
                        "current": current_val,
                        "diff": diff,
                        "pct_change": pct,
                    }

        return comparisons

    def print_comparison(self, comparisons: dict):
        """Print comparison results in a formatted table."""
        print("\n" + "=" * 70)
        print("Baseline Comparison")
        print("=" * 70)
        print(f"{'Metric':<40} {'Baseline':>10} {'Current':>10} {'Change':>10}")
        print("-" * 70)

        for metric, comp in sorted(comparisons.items()):
            baseline = comp["baseline"]
            current = comp["current"]
            diff = comp["diff"]
            pct = comp["pct_change"]

            # Color indicator
            if diff > 0.01:
                indicator = "↑"
            elif diff < -0.01:
                indicator = "↓"
            else:
                indicator = "="

            print(f"{metric:<40} {baseline:>10.4f} {current:>10.4f} {diff:>+.4f} {indicator}")


def run_pipeline(
    scorer_preset: str = "full",
    version: str = "1.0.0",
    max_examples: Optional[int] = None,
    judge_model: Optional[str] = None,
    check_gates: bool = True,
    save_baseline: bool = False,
    compare_baseline: bool = False,
) -> SkillEvaluationPipeline:
    """
    Convenience function to run the full evaluation pipeline.

    Args:
        scorer_preset: Which scorer preset to use
        version: Version string for the skill
        max_examples: Limit number of examples
        judge_model: LLM model for Tier 2/3 scorers (None = use env or default)
        check_gates: Whether to check quality gates
        save_baseline: Whether to save results as new baseline
        compare_baseline: Whether to compare to existing baseline

    Returns:
        SkillEvaluationPipeline instance with results
    """
    config = get_config(
        scorer_preset=scorer_preset,
        version=version,
        max_examples=max_examples,
        judge_model=judge_model,
    )

    pipeline = SkillEvaluationPipeline(config)
    pipeline.load_ground_truth()
    pipeline.build_eval_dataset()
    pipeline.run_evaluation()
    pipeline.print_metrics()

    if check_gates:
        pipeline.check_quality_gates()

    if compare_baseline:
        baseline = pipeline.load_baseline()
        if baseline:
            comparisons = pipeline.compare_to_baseline(baseline)
            pipeline.print_comparison(comparisons)

    if save_baseline:
        pipeline.save_baseline()

    return pipeline


if __name__ == "__main__":
    # Example usage
    print("Skill Evaluation Pipeline")
    print("=" * 50)

    # Run with quick preset for testing
    pipeline = run_pipeline(
        scorer_preset="quick",
        version="1.0.0",
        max_examples=3,  # Limit for quick test
        check_gates=True,
        save_baseline=False,
    )
