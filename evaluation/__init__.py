"""
Evaluation package for Multi-Genie Orchestrator agent.

Components:
- eval_dataset: Pre-computed evaluation dataset from production traces
- scorers: Custom and built-in scorers for multi-agent evaluation
- run_eval: Main evaluation script

Usage:
    from evaluation.eval_dataset import EVAL_DATASET, get_eval_data_for_mlflow
    from evaluation.scorers import get_scorers, ALL_SCORERS

    # Run evaluation
    import mlflow.genai
    results = mlflow.genai.evaluate(
        data=get_eval_data_for_mlflow(),
        scorers=get_scorers("core")
    )
"""

from evaluation.eval_dataset import (
    EVAL_DATASET,
    EVAL_DATASET_WITH_OUTPUTS,
    get_eval_data_for_mlflow,
    get_eval_data_with_metadata,
)

from evaluation.scorers import (
    get_scorers,
    ALL_SCORERS,
    CORE_SCORERS,
    GUIDELINES_SCORERS,
    METRIC_SCORERS,
    safety_scorer,
    relevance_scorer,
    correct_routing_scorer,
    efficient_routing_scorer,
    data_presentation_scorer,
    summary_quality_scorer,
)

__all__ = [
    # Dataset
    "EVAL_DATASET",
    "EVAL_DATASET_WITH_OUTPUTS",
    "get_eval_data_for_mlflow",
    "get_eval_data_with_metadata",
    # Scorers
    "get_scorers",
    "ALL_SCORERS",
    "CORE_SCORERS",
    "GUIDELINES_SCORERS",
    "METRIC_SCORERS",
    "safety_scorer",
    "relevance_scorer",
    "correct_routing_scorer",
    "efficient_routing_scorer",
    "data_presentation_scorer",
    "summary_quality_scorer",
]
