"""Session sync between local filesystem and UC Volumes.

Provides automatic session persistence to Unity Catalog Volumes in Databricks Apps,
where container restarts would otherwise lose local /tmp files.

Pattern: "Hybrid Local + Sync"
- Agent writes to local /tmp normally (Write tool, save_findings)
- Session files automatically sync to UC Volumes at session boundaries
- On restart, sessions restore from UC if local missing

Files synced:
- eval_tasks.json (task plan from initializer)
- state/*.json (analysis, progress)
- evaluation/*.py (generated scorers, scripts)
"""

import io
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def should_sync(volume_path: Optional[str]) -> bool:
    """Check if UC Volume sync is enabled.

    Args:
        volume_path: MLFLOW_AGENT_VOLUME_PATH value (may be None)

    Returns:
        True if sync should be performed (volume_path is set and valid)
    """
    return bool(volume_path and volume_path.startswith("/Volumes/"))


def _get_workspace_client():
    """Get WorkspaceClient with auto OAuth detection.

    In Databricks Apps runtime, credentials are auto-detected.
    For local development, uses DATABRICKS_HOST/TOKEN env vars or config profile.
    """
    from databricks.sdk import WorkspaceClient
    return WorkspaceClient()


def _get_uc_session_path(volume_path: str, session_id: str) -> str:
    """Build UC Volume path for a session.

    Args:
        volume_path: Base UC Volume path (e.g., /Volumes/catalog/schema/volume)
        session_id: Session identifier

    Returns:
        Full UC path for session (e.g., /Volumes/.../sessions/session_id)
    """
    return f"{volume_path.rstrip('/')}/sessions/{session_id}"


def sync_session_to_uc(
    session_dir: Path,
    session_id: str,
    volume_path: str,
) -> bool:
    """Upload session files to UC Volume.

    Syncs:
    - eval_tasks.json
    - state/*.json
    - evaluation/*.py

    Args:
        session_dir: Local session directory
        session_id: Session identifier
        volume_path: UC Volume base path

    Returns:
        True if sync succeeded, False on error
    """
    if not session_dir.exists():
        logger.warning(f"Session dir does not exist, skipping sync: {session_dir}")
        return False

    try:
        client = _get_workspace_client()
        uc_session_path = _get_uc_session_path(volume_path, session_id)
        uploaded = 0

        # Sync eval_tasks.json
        tasks_file = session_dir / "eval_tasks.json"
        if tasks_file.exists():
            _upload_file(client, tasks_file, f"{uc_session_path}/eval_tasks.json")
            uploaded += 1

        # Sync state/*.json
        state_dir = session_dir / "state"
        if state_dir.exists():
            for json_file in state_dir.glob("*.json"):
                _upload_file(
                    client,
                    json_file,
                    f"{uc_session_path}/state/{json_file.name}"
                )
                uploaded += 1

        # Sync evaluation/*.py
        eval_dir = session_dir / "evaluation"
        if eval_dir.exists():
            for py_file in eval_dir.glob("*.py"):
                _upload_file(
                    client,
                    py_file,
                    f"{uc_session_path}/evaluation/{py_file.name}"
                )
                uploaded += 1

        logger.info(f"Synced {uploaded} files to UC: {uc_session_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to sync session to UC: {e}")
        return False


def sync_session_from_uc(
    session_dir: Path,
    session_id: str,
    volume_path: str,
) -> bool:
    """Restore session files from UC Volume.

    Downloads:
    - eval_tasks.json
    - state/*.json
    - evaluation/*.py

    Args:
        session_dir: Local session directory to restore to
        session_id: Session identifier
        volume_path: UC Volume base path

    Returns:
        True if any files were restored, False if nothing found or error
    """
    try:
        client = _get_workspace_client()
        uc_session_path = _get_uc_session_path(volume_path, session_id)

        # Check if session exists in UC
        try:
            contents = list(client.files.list_directory_contents(uc_session_path))
            if not contents:
                logger.debug(f"No UC session found at {uc_session_path}")
                return False
        except Exception:
            # Directory doesn't exist
            logger.debug(f"UC session path does not exist: {uc_session_path}")
            return False

        # Ensure local directories exist
        session_dir.mkdir(parents=True, exist_ok=True)

        downloaded = 0

        # Download eval_tasks.json
        try:
            downloaded += _download_file(
                client,
                f"{uc_session_path}/eval_tasks.json",
                session_dir / "eval_tasks.json"
            )
        except Exception:
            pass  # File may not exist

        # Download state/*.json
        try:
            state_contents = client.files.list_directory_contents(
                f"{uc_session_path}/state"
            )
            state_dir = session_dir / "state"
            state_dir.mkdir(exist_ok=True)

            for item in state_contents:
                if not item.is_directory and item.name.endswith(".json"):
                    downloaded += _download_file(
                        client,
                        item.path,
                        state_dir / item.name
                    )
        except Exception:
            pass  # state/ may not exist

        # Download evaluation/*.py
        try:
            eval_contents = client.files.list_directory_contents(
                f"{uc_session_path}/evaluation"
            )
            eval_dir = session_dir / "evaluation"
            eval_dir.mkdir(exist_ok=True)

            for item in eval_contents:
                if not item.is_directory and item.name.endswith(".py"):
                    downloaded += _download_file(
                        client,
                        item.path,
                        eval_dir / item.name
                    )
        except Exception:
            pass  # evaluation/ may not exist

        if downloaded > 0:
            logger.info(f"Restored {downloaded} files from UC: {uc_session_path}")
            return True

        return False

    except Exception as e:
        logger.error(f"Failed to restore session from UC: {e}")
        return False


def _upload_file(client, local_path: Path, uc_path: str) -> None:
    """Upload a single file to UC Volume.

    Args:
        client: WorkspaceClient instance
        local_path: Local file path
        uc_path: UC Volume destination path
    """
    content = local_path.read_bytes()
    client.files.upload(uc_path, io.BytesIO(content), overwrite=True)
    logger.debug(f"Uploaded {local_path.name} to {uc_path}")


def _download_file(client, uc_path: str, local_path: Path) -> int:
    """Download a single file from UC Volume.

    Args:
        client: WorkspaceClient instance
        uc_path: UC Volume source path
        local_path: Local destination path

    Returns:
        1 if file downloaded, 0 if not found
    """
    try:
        response = client.files.download(uc_path)
        content = response.contents.read()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(content)
        logger.debug(f"Downloaded {uc_path} to {local_path}")
        return 1
    except Exception:
        return 0
