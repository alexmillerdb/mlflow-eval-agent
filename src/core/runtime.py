"""Runtime environment detection for Databricks Jobs.

Detects execution context and provides appropriate paths for:
- Local development
- Databricks Jobs (with Unity Catalog Volume storage)
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class RuntimeContext(Enum):
    """Runtime execution context."""
    LOCAL = "local"
    DATABRICKS_JOB = "databricks_job"


@dataclass
class RuntimeInfo:
    """Runtime environment information."""
    context: RuntimeContext
    job_id: str | None = None
    job_run_id: str | None = None
    volume_path: str | None = None

    @property
    def is_databricks(self) -> bool:
        """Check if running in Databricks."""
        return self.context == RuntimeContext.DATABRICKS_JOB

    @property
    def session_prefix(self) -> str:
        """Generate session ID prefix based on context.

        Returns 'job-{id}-run-{id}' for Databricks Jobs, empty string otherwise.
        """
        if self.context == RuntimeContext.DATABRICKS_JOB and self.job_id and self.job_run_id:
            return f"job-{self.job_id}-run-{self.job_run_id}"
        return ""


def detect_runtime() -> RuntimeInfo:
    """Detect the current runtime environment.

    Environment variable detection:
    - DB_IS_JOB: Set to 'TRUE' by Databricks in job context
    - DB_JOB_ID / DATABRICKS_JOB_ID: Job identifier
    - DB_JOB_RUN_ID / DATABRICKS_JOB_RUN_ID: Run identifier
    - MLFLOW_AGENT_VOLUME_PATH: Unity Catalog Volume for storage

    Returns:
        RuntimeInfo with detected context and metadata.
    """
    # Check for explicit volume path (implies Databricks intent)
    volume_path = os.getenv("MLFLOW_AGENT_VOLUME_PATH")

    # Check Databricks job context - multiple detection methods
    # python_wheel_task in serverless may not set DB_IS_JOB
    is_job = (
        os.getenv("DB_IS_JOB", "").upper() == "TRUE" or
        os.getenv("DATABRICKS_RUNTIME_VERSION") is not None or
        Path("/databricks").exists()
    )

    if is_job:
        job_id = os.getenv("DB_JOB_ID") or os.getenv("DATABRICKS_JOB_ID")
        job_run_id = os.getenv("DB_JOB_RUN_ID") or os.getenv("DATABRICKS_JOB_RUN_ID")

        logger.info(f"Detected Databricks Job context: job_id={job_id}, run_id={job_run_id}")

        return RuntimeInfo(
            context=RuntimeContext.DATABRICKS_JOB,
            job_id=job_id,
            job_run_id=job_run_id,
            volume_path=volume_path,
        )

    # Local development
    if volume_path:
        logger.info(f"Local mode with Volume path: {volume_path}")
    else:
        logger.debug("Local development mode")

    return RuntimeInfo(
        context=RuntimeContext.LOCAL,
        volume_path=volume_path,  # Can still use Volume via mount in local dev
    )


def get_sessions_base_path() -> Path:
    """Get the base path for session storage.

    Priority:
    1. MLFLOW_AGENT_VOLUME_PATH if set (Volume storage)
    2. Local 'sessions/' directory (development)

    Returns:
        Path to sessions directory.
    """
    volume_path = os.getenv("MLFLOW_AGENT_VOLUME_PATH")

    if volume_path:
        base = Path(volume_path) / "sessions"
        logger.info(f"Using Volume storage: {base}")
        return base

    # Warn if in Databricks without Volume configured
    runtime = detect_runtime()
    if runtime.is_databricks:
        logger.warning(
            "Running in Databricks but MLFLOW_AGENT_VOLUME_PATH not set. "
            "Sessions will be written to local filesystem and may not persist."
        )

    return Path("sessions")
