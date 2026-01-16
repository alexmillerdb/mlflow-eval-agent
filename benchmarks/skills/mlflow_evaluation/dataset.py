"""
Dataset abstraction layer for MLflow evaluation.

Provides transparent access to evaluation datasets from:
- Local YAML file (for development)
- Unity Catalog MLflow-managed dataset (for Databricks production)

Auto-detection: Uses UC dataset in Databricks runtime, YAML otherwise.
"""

import os
from pathlib import Path
from typing import Union, Optional

from .ground_truth import load_ground_truth, GroundTruthExample


def is_databricks_runtime() -> bool:
    """Check if running in Databricks runtime environment."""
    return os.environ.get("DATABRICKS_RUNTIME_VERSION") is not None


def get_uc_table_name() -> Optional[str]:
    """Get Unity Catalog table name from environment."""
    return os.environ.get("BENCHMARK_UC_TABLE")


class DatasetSource:
    """
    Abstraction for evaluation dataset sources.

    Supports both local YAML files and Unity Catalog MLflow-managed datasets.
    Auto-detection based on environment.
    """

    @staticmethod
    def from_yaml() -> list[dict]:
        """
        Load evaluation dataset from local YAML file.

        Returns:
            List of evaluation records in MLflow format:
            [{"inputs": {...}, "outputs": {...}, "expectations": {...}}, ...]
        """
        examples = load_ground_truth()
        return [ex.to_eval_record() for ex in examples]

    @staticmethod
    def from_uc(
        table_name: str,
        auto_populate: bool = True,
        tags: Optional[dict] = None,
    ) -> "mlflow.genai.datasets.Dataset":
        """
        Load evaluation dataset from Unity Catalog.

        Requires Databricks runtime or configured Databricks Connect.

        Args:
            table_name: Fully qualified UC table name (catalog.schema.table)
            auto_populate: If True, automatically populate new datasets from YAML
            tags: Optional tags to add when creating new dataset

        Returns:
            MLflow Dataset object

        Raises:
            ImportError: If mlflow.genai.datasets not available
            RuntimeError: If Spark session cannot be established
        """
        import mlflow
        import mlflow.genai.datasets

        # Initialize Spark session for UC access
        try:
            from databricks.connect import DatabricksSession
            spark = DatabricksSession.builder.remote(serverless=True).getOrCreate()
        except ImportError:
            # Fallback for Databricks runtime where Spark is pre-configured
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.getOrCreate()

        # Get experiment ID from environment for dataset association
        experiment_name = os.environ.get("SKILL_BENCHMARK_MLFLOW_EXPERIMENT_NAME")
        experiment_id = None

        if experiment_name:
            mlflow.set_experiment(experiment_name)
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment:
                experiment_id = experiment.experiment_id
                print(f"Dataset will be linked to experiment: {experiment_name} (ID: {experiment_id})")

        # Default tags for dataset
        default_tags = {"skill": "mlflow-evaluation", "version": "1.0"}
        dataset_tags = {**default_tags, **(tags or {})}

        # Get or create the dataset
        is_new = False
        try:
            dataset = mlflow.genai.datasets.get_dataset(table_name)
            print(f"Loaded existing UC dataset: {table_name}")
        except Exception:
            # Dataset doesn't exist yet - create with experiment_id and tags
            dataset = mlflow.genai.datasets.create_dataset(
                name=table_name,
                experiment_id=[experiment_id] if experiment_id else None,
                tags=dataset_tags,
            )
            print(f"Created new UC dataset: {table_name}")
            if experiment_id:
                print(f"  Linked to experiment ID: {experiment_id}")
            print(f"  Tags: {dataset_tags}")
            is_new = True

        # Auto-populate new datasets from YAML
        if is_new and auto_populate:
            print("Auto-populating new dataset from YAML ground truth...")
            examples = load_ground_truth()
            records = [ex.to_eval_record() for ex in examples]
            dataset.merge_records(records)
            print(f"Added {len(records)} records from YAML")

        return dataset

    @staticmethod
    def auto(
        source: str = "auto",
        uc_table_name: Optional[str] = None,
        tags: Optional[dict] = None,
    ) -> Union[list[dict], "mlflow.genai.datasets.Dataset"]:
        """
        Auto-detect and load the appropriate dataset source.

        Args:
            source: Data source to use:
                - "auto": UC in Databricks, YAML otherwise
                - "yaml": Force local YAML
                - "uc": Force Unity Catalog
            uc_table_name: UC table name (required if source="uc", otherwise from env)
            tags: Optional tags for UC dataset creation

        Returns:
            Either a list of eval records (YAML) or MLflow Dataset object (UC)
        """
        if source == "yaml":
            print("Using YAML dataset source")
            return DatasetSource.from_yaml()

        if source == "uc":
            table_name = uc_table_name or get_uc_table_name()
            if not table_name:
                raise ValueError(
                    "Unity Catalog table name required. "
                    "Set BENCHMARK_UC_TABLE environment variable or pass uc_table_name."
                )
            print(f"Using UC dataset source: {table_name}")
            return DatasetSource.from_uc(table_name, tags=tags)

        # Auto-detect
        if is_databricks_runtime():
            table_name = uc_table_name or get_uc_table_name()
            if table_name:
                print(f"Auto-detected Databricks runtime, using UC: {table_name}")
                return DatasetSource.from_uc(table_name, tags=tags)
            else:
                print("Databricks runtime detected but no UC table configured, using YAML")
                return DatasetSource.from_yaml()
        else:
            print("Local environment detected, using YAML dataset")
            return DatasetSource.from_yaml()

    @staticmethod
    def sync_yaml_to_uc(
        uc_table_name: Optional[str] = None,
        dry_run: bool = False
    ) -> int:
        """
        Sync local YAML examples to Unity Catalog dataset.

        Args:
            uc_table_name: Target UC table name (from env if not provided)
            dry_run: If True, only show what would be synced

        Returns:
            Number of records synced

        Raises:
            ValueError: If UC table name not provided
        """
        table_name = uc_table_name or get_uc_table_name()
        if not table_name:
            raise ValueError(
                "Unity Catalog table name required. "
                "Set BENCHMARK_UC_TABLE environment variable or pass uc_table_name."
            )

        # Load YAML examples
        examples = load_ground_truth()
        records = [ex.to_eval_record() for ex in examples]

        if dry_run:
            print(f"[DRY RUN] Would sync {len(records)} records to {table_name}")
            for i, record in enumerate(records[:3]):
                prompt_preview = str(record.get("inputs", {}).get("prompt", ""))[:50]
                print(f"  {i+1}. {prompt_preview}...")
            if len(records) > 3:
                print(f"  ... and {len(records) - 3} more")
            return len(records)

        # Get or create UC dataset (disable auto-populate since we're explicitly syncing)
        dataset = DatasetSource.from_uc(table_name, auto_populate=False)

        # Merge records into dataset
        dataset.merge_records(records)
        print(f"Synced {len(records)} records to {table_name}")

        return len(records)

    @staticmethod
    def get_dataset_info(source: str = "auto", uc_table_name: Optional[str] = None) -> dict:
        """
        Get information about the dataset source.

        Args:
            source: Data source ("auto", "yaml", "uc")
            uc_table_name: UC table name if using UC source

        Returns:
            Dict with source info (type, count, etc.)
        """
        info = {
            "source": source,
            "is_databricks": is_databricks_runtime(),
            "uc_table_configured": get_uc_table_name() is not None,
        }

        if source == "yaml" or (source == "auto" and not is_databricks_runtime()):
            examples = load_ground_truth()
            info["type"] = "yaml"
            info["count"] = len(examples)
            info["path"] = str(Path(__file__).parent / "ground_truth.yaml")
        elif source == "uc" or (source == "auto" and is_databricks_runtime()):
            table_name = uc_table_name or get_uc_table_name()
            info["type"] = "uc"
            info["table_name"] = table_name
            if table_name:
                try:
                    dataset = DatasetSource.from_uc(table_name)
                    info["count"] = len(dataset.to_df())
                except Exception as e:
                    info["error"] = str(e)

        return info


if __name__ == "__main__":
    print("Dataset Source Info")
    print("=" * 50)

    info = DatasetSource.get_dataset_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("\nYAML Dataset Preview:")
    records = DatasetSource.from_yaml()
    print(f"  Total records: {len(records)}")
    for i, record in enumerate(records[:3]):
        prompt = str(record.get("inputs", {}).get("prompt", ""))[:60]
        print(f"  {i+1}. {prompt}...")
