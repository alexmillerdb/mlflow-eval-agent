"""Simplified configuration for MLflow Evaluation Agent.

Minimal config following KISS principle - only essential settings.
~50 lines vs original ~80 lines
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
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
    experiment_id: str = ""  # Target experiment to analyze
    agent_experiment_id: str = ""  # Experiment for agent's own traces
    tracking_uri: str = "databricks"

    # Model
    model: str = "sonnet"

    # Paths
    working_dir: Path = field(default_factory=Path.cwd)

    # Limits
    max_turns: int = 30

    # Session (auto-generated if not provided)
    session_id: str = ""

    @classmethod
    def from_env(cls, validate: bool = True) -> "Config":
        """Load configuration from environment variables."""
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass  # dotenv optional

        # Generate session ID if not provided
        # Uses job context for Databricks Jobs, timestamp otherwise
        from .runtime import detect_runtime

        session_id = os.getenv("SESSION_ID", "")
        if not session_id:
            runtime = detect_runtime()
            if runtime.session_prefix:
                # Databricks Job: use job-{id}-run-{id} format
                session_id = runtime.session_prefix
            else:
                # Local: use timestamp format
                session_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")

        config = cls(
            databricks_host=os.getenv("DATABRICKS_HOST", ""),
            databricks_token=os.getenv("DATABRICKS_TOKEN", ""),
            databricks_config_profile=os.getenv("DATABRICKS_CONFIG_PROFILE", ""),
            experiment_id=os.getenv("MLFLOW_EXPERIMENT_ID", ""),
            agent_experiment_id=os.getenv("MLFLOW_AGENT_EXPERIMENT_ID", ""),
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "databricks"),
            model=os.getenv("MODEL", "databricks-claude-opus-4-5"),
            session_id=session_id,
        )

        if validate:
            config.validate()

        return config

    def validate(self) -> None:
        """Validate required configuration."""
        from .runtime import detect_runtime

        runtime = detect_runtime()

        # Skip Databricks auth check when running inside Databricks (auth is automatic)
        if runtime.is_databricks:
            return

        # Local mode requires explicit credentials
        if not self.databricks_config_profile and not self.databricks_host:
            raise ValueError(
                "Configuration error: Set DATABRICKS_CONFIG_PROFILE or DATABRICKS_HOST"
            )
