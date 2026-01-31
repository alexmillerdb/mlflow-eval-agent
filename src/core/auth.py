"""Databricks authentication helpers for subprocess execution.

Single entry point: call configure_env() once at startup.
All subprocesses will inherit the configured environment.
"""

import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)

_configured = False


def configure_env() -> dict[str, str]:
    """Configure environment variables for Databricks/MLflow auth.

    Call this ONCE at startup. Sets env vars that subprocess will inherit:
    - DATABRICKS_HOST: Workspace URL (required for MLflow tracking)
    - DATABRICKS_TOKEN: Auth token (if available from ANTHROPIC_AUTH_TOKEN)
    - MLFLOW_TRACKING_URI: Set to "databricks" if in Databricks environment

    Returns dict of env vars that were set (for logging).
    """
    global _configured
    if _configured:
        return {}  # Idempotent

    configured = {}

    # 1. Set DATABRICKS_HOST if not already set
    if not os.getenv("DATABRICKS_HOST"):
        if host := _detect_databricks_host():
            os.environ["DATABRICKS_HOST"] = host
            configured["DATABRICKS_HOST"] = host

    # 2. Set DATABRICKS_TOKEN from ANTHROPIC_AUTH_TOKEN if available
    # (In Databricks, the same PAT is used for both Anthropic endpoint and MLflow)
    if not os.getenv("DATABRICKS_TOKEN"):
        if token := os.getenv("ANTHROPIC_AUTH_TOKEN"):
            os.environ["DATABRICKS_TOKEN"] = token
            configured["DATABRICKS_TOKEN"] = "***"  # Don't log actual token

    # 3. Set MLFLOW_TRACKING_URI if in Databricks environment
    if os.getenv("DATABRICKS_HOST") and not os.getenv("MLFLOW_TRACKING_URI"):
        os.environ["MLFLOW_TRACKING_URI"] = "databricks"
        configured["MLFLOW_TRACKING_URI"] = "databricks"

    if configured:
        logger.info(f"Configured subprocess env vars: {list(configured.keys())}")

    _configured = True
    return configured


@lru_cache
def _detect_databricks_host() -> str | None:
    """Detect Databricks workspace host URL."""
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        return w.config.host.rstrip("/") if w.config.host else None
    except Exception:
        return None


# Keep these for backward compatibility but they're no longer needed
def get_databricks_host() -> str | None:
    """Get Databricks workspace host URL."""
    return os.getenv("DATABRICKS_HOST") or _detect_databricks_host()


def get_subprocess_env_vars() -> dict[str, str]:
    """Get environment variables needed for subprocess auth.

    Note: After calling configure_env(), these are already in os.environ
    and subprocess will inherit them automatically.
    """
    env_vars = {}
    for var in ["DATABRICKS_HOST", "DATABRICKS_TOKEN", "MLFLOW_TRACKING_URI"]:
        if val := os.getenv(var):
            env_vars[var] = val
    return env_vars
