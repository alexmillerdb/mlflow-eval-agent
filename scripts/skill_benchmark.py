#!/usr/bin/env python
"""
CLI for running skill benchmark evaluations.

Usage:
    # Run benchmark evaluation
    python -m scripts.skill_benchmark run --skill mlflow-evaluation

    # Run with specific preset
    python -m scripts.skill_benchmark run --skill mlflow-evaluation --preset quick

    # Dry run (show config without executing)
    python -m scripts.skill_benchmark run --skill mlflow-evaluation --dry-run

    # Compare to baseline
    python -m scripts.skill_benchmark run --skill mlflow-evaluation --compare-baseline

    # Save new baseline
    python -m scripts.skill_benchmark run --skill mlflow-evaluation --save-baseline

    # Check regression against baseline
    python -m scripts.skill_benchmark regression --skill mlflow-evaluation
"""

import argparse
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def cmd_run(args):
    """Run benchmark evaluation for a skill."""
    from scripts.skill_eval_pipeline import run_pipeline, SkillEvaluationPipeline
    from benchmarks.skills.mlflow_evaluation.config import get_config, SCORER_PRESETS
    from benchmarks.skills.mlflow_evaluation.scorers import get_scorers
    from benchmarks.skills.mlflow_evaluation.dataset import DatasetSource

    print(f"\n{'='*60}")
    print(f"Skill Benchmark: {args.skill}")
    print(f"{'='*60}")

    # Validate skill
    if args.skill != "mlflow-evaluation":
        print(f"Error: Unknown skill '{args.skill}'")
        print("Available skills: mlflow-evaluation")
        return 1

    # Get configuration
    config = get_config(
        scorer_preset=args.preset,
        version=args.version,
        max_examples=args.max_examples,
        judge_model=args.model,
        data_source=args.data_source,
        uc_table_name=args.uc_table,
    )

    print(f"\nConfiguration:")
    print(f"  Skill: {config.skill_name} v{config.skill_version}")
    print(f"  Preset: {args.preset}")
    if args.preset in SCORER_PRESETS:
        print(f"    {SCORER_PRESETS[args.preset]['description']}")
    print(f"  Tracking: {config.tracking_uri}")
    print(f"  Experiment: {config.experiment_name}")
    print(f"  Data Source: {config.data_source}")
    if config.uc_table_name:
        print(f"  UC Table: {config.uc_table_name}")

    # Show model for LLM presets
    requires_llm = args.preset in ("full", "tier3", "all")
    if requires_llm:
        print(f"  Judge Model: {config.judge_model}")

    # Get and show scorers (pass model for LLM scorers)
    scorers = get_scorers(args.preset, model=config.judge_model)
    print(f"\nScorers ({len(scorers)}):")
    for s in scorers[:5]:  # Show first 5
        name = getattr(s, "name", getattr(s, "__name__", str(s)))
        print(f"  - {name}")
    if len(scorers) > 5:
        print(f"  ... and {len(scorers) - 5} more")

    # Show quality gates
    print(f"\nQuality Gates ({len(config.quality_gates)}):")
    for gate in config.quality_gates:
        print(f"  - {gate}")

    if args.dry_run:
        print("\n[DRY RUN] Would run evaluation with above config")
        return 0

    # Run the pipeline
    print("\n" + "-" * 60)
    pipeline = run_pipeline(
        scorer_preset=args.preset,
        version=args.version,
        max_examples=args.max_examples,
        judge_model=args.model,
        check_gates=not args.no_gates,
        save_baseline=args.save_baseline,
        compare_baseline=args.compare_baseline,
        data_source=args.data_source,
        uc_table_name=args.uc_table,
    )

    # Return exit code based on quality gates
    if not args.no_gates:
        passed, _ = pipeline.check_quality_gates()
        return 0 if passed else 1

    return 0


def cmd_regression(args):
    """Check for regression against baseline."""
    from scripts.skill_eval_pipeline import SkillEvaluationPipeline
    from benchmarks.skills.mlflow_evaluation.config import get_config

    print(f"\n{'='*60}")
    print(f"Regression Check: {args.skill}")
    print(f"{'='*60}")

    if args.skill != "mlflow-evaluation":
        print(f"Error: Unknown skill '{args.skill}'")
        return 1

    config = get_config(scorer_preset="quick", version=args.version)
    pipeline = SkillEvaluationPipeline(config)

    # Load baseline
    baseline = pipeline.load_baseline()
    if baseline is None:
        print("\nNo baseline found. Run with --save-baseline first.")
        return 1

    # Run quick evaluation
    print("\nRunning evaluation...")
    pipeline.load_ground_truth()
    pipeline.build_eval_dataset()
    pipeline.run_evaluation()
    pipeline.print_metrics()

    # Compare to baseline
    comparisons = pipeline.compare_to_baseline(baseline)
    pipeline.print_comparison(comparisons)

    # Check for regressions (any metric decreased by more than threshold)
    threshold = args.threshold
    regressions = []
    for metric, comp in comparisons.items():
        if comp["diff"] < -threshold:
            regressions.append(f"{metric}: {comp['baseline']:.4f} -> {comp['current']:.4f} ({comp['diff']:+.4f})")

    if regressions:
        print(f"\n✗ Regressions detected (threshold: {threshold}):")
        for reg in regressions:
            print(f"  - {reg}")
        return 1
    else:
        print(f"\n✓ No regressions detected (threshold: {threshold})")
        return 0


def cmd_init(args):
    """Initialize benchmark structure for a new skill."""
    print(f"\n{'='*60}")
    print(f"Initialize Benchmark: {args.skill}")
    print(f"{'='*60}")

    skill_dir = Path("benchmarks/skills") / args.skill
    if skill_dir.exists():
        print(f"Error: Benchmark already exists at {skill_dir}")
        return 1

    print(f"\nCreating benchmark structure at: {skill_dir}")

    # Create directories
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "baselines").mkdir(exist_ok=True)

    # Create __init__.py
    (skill_dir / "__init__.py").write_text(f'"""Benchmark for {args.skill} skill."""\n')

    # Create placeholder files
    (skill_dir / "ground_truth.yaml").write_text(f"""# Ground truth examples for {args.skill} skill
# Generated via /skill-test {args.skill}

examples: []
""")

    (skill_dir / "config.py").write_text(f'''"""Benchmark configuration for {args.skill} skill."""

from dataclasses import dataclass, field
from typing import Optional


# TODO: Define quality gates and configuration
# See benchmarks/skills/mlflow-evaluation/config.py for example
''')

    (skill_dir / "scorers.py").write_text(f'''"""Scorers for {args.skill} skill benchmark."""

from mlflow.genai.scorers import scorer, Guidelines, Safety
from mlflow.entities import Feedback


# TODO: Implement scorers using three-tier strategy
# See benchmarks/skills/mlflow-evaluation/scorers.py for example
''')

    print(f"\nCreated:")
    print(f"  - {skill_dir}/__init__.py")
    print(f"  - {skill_dir}/ground_truth.yaml")
    print(f"  - {skill_dir}/config.py")
    print(f"  - {skill_dir}/scorers.py")
    print(f"  - {skill_dir}/baselines/")

    print(f"\nNext steps:")
    print(f"  1. Run /skill-test {args.skill} to generate ground truth examples")
    print(f"  2. Define scorers in scorers.py")
    print(f"  3. Configure quality gates in config.py")
    print(f"  4. Run: python -m scripts.skill_benchmark run --skill {args.skill}")

    return 0


def cmd_sync(args):
    """Sync ground truth examples to Unity Catalog dataset."""
    from benchmarks.skills.mlflow_evaluation.dataset import DatasetSource

    print(f"\n{'='*60}")
    print(f"Dataset Sync: {args.skill}")
    print(f"{'='*60}")

    if args.skill != "mlflow-evaluation":
        print(f"Error: Unknown skill '{args.skill}'")
        print("Available skills: mlflow-evaluation")
        return 1

    if args.to_uc:
        # Sync YAML to Unity Catalog
        uc_table = args.uc_table
        if not uc_table:
            import os
            uc_table = os.environ.get("BENCHMARK_UC_TABLE")

        if not uc_table:
            print("Error: UC table name required.")
            print("  Set BENCHMARK_UC_TABLE environment variable or use --uc-table")
            return 1

        print(f"\nSyncing YAML → UC dataset: {uc_table}")

        try:
            count = DatasetSource.sync_yaml_to_uc(
                uc_table_name=uc_table,
                dry_run=args.dry_run
            )
            if args.dry_run:
                print(f"\n[DRY RUN] Would sync {count} records")
            else:
                print(f"\n✓ Successfully synced {count} records to UC")
            return 0
        except Exception as e:
            print(f"\n✗ Sync failed: {e}")
            return 1

    elif args.info:
        # Show dataset info
        info = DatasetSource.get_dataset_info()
        print("\nDataset Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        return 0

    else:
        print("Error: Specify --to-uc to sync YAML to UC, or --info to show dataset info")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="CLI for skill benchmark evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run benchmark evaluation")
    run_parser.add_argument(
        "--skill", "-s",
        required=True,
        help="Skill name (e.g., mlflow-evaluation)"
    )
    run_parser.add_argument(
        "--preset", "-p",
        default="full",
        choices=["full", "quick", "tier1", "tier3"],
        help="Scorer preset (default: full)"
    )
    run_parser.add_argument(
        "--version", "-v",
        default="1.0.0",
        help="Skill version (default: 1.0.0)"
    )
    run_parser.add_argument(
        "--max-examples", "-n",
        type=int,
        default=None,
        help="Limit number of examples (default: all)"
    )
    run_parser.add_argument(
        "--model", "-m",
        default=None,
        help="LLM model for Tier 2/3 scorers (default: from env or databricks:/databricks-gpt-5-2)"
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show config without running evaluation"
    )
    run_parser.add_argument(
        "--no-gates",
        action="store_true",
        help="Skip quality gate checks"
    )
    run_parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save results as new baseline"
    )
    run_parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare results to existing baseline"
    )
    run_parser.add_argument(
        "--data-source",
        default="auto",
        choices=["auto", "yaml", "uc"],
        help="Dataset source: auto (detect), yaml (local), uc (Unity Catalog)"
    )
    run_parser.add_argument(
        "--uc-table",
        default=None,
        help="Unity Catalog table name (default: from BENCHMARK_UC_TABLE env)"
    )
    run_parser.set_defaults(func=cmd_run)

    # Regression command
    reg_parser = subparsers.add_parser("regression", help="Check for regression against baseline")
    reg_parser.add_argument(
        "--skill", "-s",
        required=True,
        help="Skill name"
    )
    reg_parser.add_argument(
        "--version", "-v",
        default="1.0.0",
        help="Skill version"
    )
    reg_parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.05,
        help="Regression threshold (default: 0.05)"
    )
    reg_parser.set_defaults(func=cmd_regression)

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize benchmark for new skill")
    init_parser.add_argument(
        "--skill", "-s",
        required=True,
        help="Skill name to initialize"
    )
    init_parser.set_defaults(func=cmd_init)

    # Sync command
    sync_parser = subparsers.add_parser("sync", help="Sync dataset between YAML and Unity Catalog")
    sync_parser.add_argument(
        "--skill", "-s",
        required=True,
        help="Skill name"
    )
    sync_parser.add_argument(
        "--to-uc",
        action="store_true",
        help="Sync YAML examples to Unity Catalog dataset"
    )
    sync_parser.add_argument(
        "--info",
        action="store_true",
        help="Show dataset info"
    )
    sync_parser.add_argument(
        "--uc-table",
        default=None,
        help="Unity Catalog table name (default: from BENCHMARK_UC_TABLE env)"
    )
    sync_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be synced without making changes"
    )
    sync_parser.set_defaults(func=cmd_sync)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
