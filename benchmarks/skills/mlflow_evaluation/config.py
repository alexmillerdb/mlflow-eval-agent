"""
Benchmark configuration for mlflow-evaluation skill.

Defines quality gates, scorer presets, and evaluation settings.
"""

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class QualityGate:
    """A quality threshold that must be met for the benchmark to pass."""
    metric: str
    threshold: float
    comparison: str = ">="  # ">=", ">", "<=", "<", "=="

    def check(self, value: float) -> bool:
        """Check if the value passes this quality gate."""
        if self.comparison == ">=":
            return value >= self.threshold
        elif self.comparison == ">":
            return value > self.threshold
        elif self.comparison == "<=":
            return value <= self.threshold
        elif self.comparison == "<":
            return value < self.threshold
        elif self.comparison == "==":
            return value == self.threshold
        return False

    def __str__(self):
        return f"{self.metric} {self.comparison} {self.threshold}"


@dataclass
class BenchmarkConfig:
    """Configuration for skill benchmark evaluation."""

    # Skill identification
    skill_name: str = "mlflow-evaluation"
    skill_version: str = "1.0.0"

    # MLflow experiment settings
    experiment_name: Optional[str] = None  # Will be set from env or default
    tracking_uri: Optional[str] = None  # Will be set from env or default

    # Quality gates (thresholds for pass/fail)
    quality_gates: list[QualityGate] = field(default_factory=list)

    # Scorer preset to use
    scorer_preset: str = "full"  # "full", "quick", "tier1", "tier2", "tier3"

    # LLM judge model for Tier 2/3 scorers
    judge_model: Optional[str] = None  # Will be set from env or default

    # Evaluation settings
    max_examples: Optional[int] = None  # None = use all examples
    tags: dict = field(default_factory=dict)  # Additional tags to log

    # Dataset source settings
    data_source: str = "yaml"  # "yaml", "uc", "auto"
    uc_table_name: Optional[str] = None  # UC table for MLflow-managed dataset
    dataset_tags: dict = field(default_factory=lambda: {
        "skill": "mlflow-evaluation",
        "version": "1.0",
    })

    def __post_init__(self):
        # Set tracking URI from environment or default to local SQLite
        if self.tracking_uri is None:
            self.tracking_uri = os.environ.get(
                "MLFLOW_TRACKING_URI",
                "sqlite:///mlflow.db"
            )

        # Set experiment name from environment or default
        # Use simple name for local, nested path for Databricks
        if self.experiment_name is None:
            self.experiment_name = os.environ.get("SKILL_BENCHMARK_MLFLOW_EXPERIMENT_NAME")
            if self.experiment_name is None:
                # Use simple name that works in both local and Databricks
                self.experiment_name = f"skill-benchmarks-{self.skill_name}"

        # Set judge model from environment or default
        if self.judge_model is None:
            self.judge_model = os.environ.get(
                "BENCHMARK_JUDGE_MODEL",
                "databricks:/databricks-gpt-5-2"
            )

        # Set UC table name from environment if not provided
        if self.uc_table_name is None:
            self.uc_table_name = os.environ.get("BENCHMARK_UC_TABLE")

        # Set default quality gates if none provided
        if not self.quality_gates:
            self.quality_gates = DEFAULT_QUALITY_GATES.copy()

    def check_quality_gates(self, metrics: dict) -> tuple[bool, list[str]]:
        """
        Check if all quality gates pass.

        Args:
            metrics: Dictionary of metric names to values

        Returns:
            (passed, failures): Boolean and list of failed gate descriptions
        """
        failures = []
        for gate in self.quality_gates:
            if gate.metric in metrics:
                value = metrics[gate.metric]
                if not gate.check(value):
                    failures.append(f"{gate} (actual: {value:.4f})")

        return len(failures) == 0, failures


# Default quality gates for mlflow-evaluation skill
DEFAULT_QUALITY_GATES = [
    # Safety must always pass
    QualityGate(metric="safety/score/mean", threshold=1.0, comparison=">="),

    # Syntax and structure (100% required)
    QualityGate(metric="code_syntax_valid/score/mean", threshold=1.0, comparison=">="),

    # API correctness - strict requirements (95%+ pass rate)
    QualityGate(metric="uses_genai_evaluate/score/mean", threshold=0.95, comparison=">="),
    QualityGate(metric="has_nested_inputs/score/mean", threshold=0.95, comparison=">="),
    QualityGate(metric="has_scorer_decorator/score/mean", threshold=0.95, comparison=">="),
    QualityGate(metric="uses_valid_aggregations/score/mean", threshold=0.95, comparison=">="),

    # Tier 3 subjective quality (yes/no = 1.0/0.0, threshold 0.85 = 85% pass rate)
    QualityGate(metric="code_quality/score/mean", threshold=0.85, comparison=">="),
    QualityGate(metric="response_completeness/score/mean", threshold=0.85, comparison=">="),
    QualityGate(metric="explanation_clarity/score/mean", threshold=0.85, comparison=">="),
]


# Scorer preset configurations
SCORER_PRESETS = {
    "quick": {
        "description": "Fast evaluation with deterministic scorers only (Tier 1)",
        "scorers": ["tier1"],
        "estimated_time": "~30 seconds",
    },
    "full": {
        "description": "Complete evaluation: Tier 1 deterministic + Tier 3 quality",
        "scorers": ["tier1", "tier3"],
        "estimated_time": "~1-2 minutes (Tier 3 LLM calls)",
    },
    "tier1": {
        "description": "Pattern-matching scorers only (API correctness + metrics)",
        "scorers": ["tier1"],
        "estimated_time": "~30 seconds",
    },
    "tier3": {
        "description": "Guidelines scorers only (requires LLM)",
        "scorers": ["tier3"],
        "estimated_time": "~1-2 minutes",
    },
}


def get_config(
    scorer_preset: str = "full",
    version: str = "1.0.0",
    max_examples: Optional[int] = None,
    judge_model: Optional[str] = None,
    data_source: str = "yaml",
    uc_table_name: Optional[str] = None,
) -> BenchmarkConfig:
    """
    Get benchmark configuration with optional overrides.

    Args:
        scorer_preset: Which scorer preset to use
        version: Version string for the skill being evaluated
        max_examples: Limit number of examples (None = all)
        judge_model: LLM model for Tier 2/3 scorers (None = use env or default)
        data_source: Dataset source ("yaml", "uc", "auto")
                     "yaml" recommended for benchmarks with pre-computed outputs
        uc_table_name: Unity Catalog table name (None = use env)

    Returns:
        BenchmarkConfig instance
    """
    return BenchmarkConfig(
        skill_version=version,
        scorer_preset=scorer_preset,
        max_examples=max_examples,
        judge_model=judge_model,
        data_source=data_source,
        uc_table_name=uc_table_name,
        tags={
            "skill.name": "mlflow-evaluation",
            "skill.version": version,
            "evaluation.type": "benchmark",
            "scorer.preset": scorer_preset,
            "data.source": data_source,
        }
    )


if __name__ == "__main__":
    print("MLflow Evaluation Skill - Benchmark Config")
    print("=" * 50)

    config = get_config()
    print(f"\nSkill: {config.skill_name} v{config.skill_version}")
    print(f"Tracking URI: {config.tracking_uri}")
    print(f"Experiment: {config.experiment_name}")
    print(f"Judge Model: {config.judge_model}")

    print(f"\nQuality Gates ({len(config.quality_gates)}):")
    for gate in config.quality_gates:
        print(f"  - {gate}")

    print(f"\nScorer Presets:")
    for name, info in SCORER_PRESETS.items():
        print(f"  {name}: {info['description']}")
