"""Configuration for MLflow Evaluation Agent."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EvalAgentConfig:
    """Configuration for the evaluation agent.

    All fields can be loaded from environment variables via from_env().
    """
    # Databricks
    databricks_host: str = ""
    databricks_token: str = ""
    databricks_config_profile: str = ""
    serverless_compute_id: str = "auto"
    cluster_id: str = ""

    # MLflow
    experiment_id: str = ""
    experiment_path: str = "/Shared/mlflow-eval-agent"
    tracking_uri: str = "databricks"

    # Model
    model: str = "sonnet"
    sub_agents_model: str = ""
    anthropic_base_url: str = ""
    anthropic_auth_token: str = ""

    # Project
    project_name: str = "mlflow-eval-agent"
    environment: str = "development"

    # Paths
    working_dir: Path = field(default_factory=Path.cwd)
    skills_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / ".claude" / "skills")

    # Limits
    max_turns: int = 50
    max_trace_results: int = 1000
    default_page_size: int = 50
    workspace_context_max_chars: int = 2000

    def __post_init__(self):
        if not self.sub_agents_model:
            self.sub_agents_model = self.model

    def validate(self) -> None:
        """Validate required fields. Raises ValueError if missing."""
        if not self.databricks_host:
            raise ValueError("Configuration error: DATABRICKS_HOST is required")

    @classmethod
    def from_env(cls, env_file: Optional[str] = None, validate: bool = True) -> "EvalAgentConfig":
        """Load configuration from environment variables."""
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_file)

        config = cls(
            databricks_host=os.getenv("DATABRICKS_HOST", ""),
            databricks_token=os.getenv("DATABRICKS_TOKEN", ""),
            databricks_config_profile=os.getenv("DATABRICKS_CONFIG_PROFILE", ""),
            serverless_compute_id=os.getenv("DATABRICKS_SERVERLESS_COMPUTE_ID", "auto"),
            cluster_id=os.getenv("DATABRICKS_CLUSTER_ID", ""),
            experiment_id=os.getenv("MLFLOW_EXPERIMENT_ID", ""),
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "databricks"),
            model=os.getenv("DABS_MODEL", "sonnet"),
            sub_agents_model=os.getenv("SUB_AGENTS_MODEL", ""),
            anthropic_base_url=os.getenv("ANTHROPIC_BASE_URL", ""),
            anthropic_auth_token=os.getenv("ANTHROPIC_AUTH_TOKEN", ""),
            project_name=os.getenv("PROJECT_NAME", "mlflow-eval-agent"),
            environment=os.getenv("ENVIRONMENT", "development"),
        )

        if validate:
            config.validate()

        return config
