"""Simplified configuration for MLflow Evaluation Agent.

Minimal config following KISS principle - only essential settings.
~50 lines vs original ~80 lines
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """Minimal configuration for the evaluation agent."""

    # Databricks auth (prefer config profile)
    databricks_host: str = ""
    databricks_token: str = ""
    databricks_config_profile: str = ""

    # MLflow
    experiment_id: str = ""
    tracking_uri: str = "databricks"

    # Model
    model: str = "sonnet"

    # Paths
    working_dir: Path = field(default_factory=Path.cwd)

    # Limits
    max_turns: int = 50

    @classmethod
    def from_env(cls, validate: bool = True) -> "Config":
        """Load configuration from environment variables."""
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass  # dotenv optional

        config = cls(
            databricks_host=os.getenv("DATABRICKS_HOST", ""),
            databricks_token=os.getenv("DATABRICKS_TOKEN", ""),
            databricks_config_profile=os.getenv("DATABRICKS_CONFIG_PROFILE", ""),
            experiment_id=os.getenv("MLFLOW_EXPERIMENT_ID", ""),
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "databricks"),
            model=os.getenv("DABS_MODEL", "sonnet"),
        )

        if validate:
            config.validate()

        return config

    def validate(self) -> None:
        """Validate required configuration."""
        # Either config profile or host+token required
        if not self.databricks_config_profile and not self.databricks_host:
            raise ValueError(
                "Configuration error: Set DATABRICKS_CONFIG_PROFILE or DATABRICKS_HOST"
            )
