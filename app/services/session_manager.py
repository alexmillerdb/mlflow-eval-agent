"""Session management for user isolation.

Handles:
- Creating user-specific session directories in UC Volume
- Listing past sessions for a user
- Session resumption
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    """Information about a session."""
    session_id: str
    session_dir: Path
    created_at: datetime
    experiment_id: str
    status: str  # "running", "completed", "error"
    task_count: int
    completed_count: int


def get_user_session_dir(
    user_prefix: str,
    volume_path: str,
    session_id: Optional[str] = None,
) -> tuple[str, Path]:
    """Get or create a session directory for a user.

    Args:
        user_prefix: User's session prefix (from email hash).
        volume_path: Base Unity Catalog Volume path.
        session_id: Optional existing session ID to resume.

    Returns:
        Tuple of (session_id, session_dir Path).
    """
    sessions_base = Path(volume_path) / "sessions"

    if session_id:
        # Resume existing session
        session_dir = sessions_base / session_id
        if session_dir.exists():
            return session_id, session_dir
        else:
            logger.warning(f"Session {session_id} not found, creating new session")

    # Create new session with user prefix and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_session_id = f"{user_prefix}_{timestamp}"
    session_dir = sessions_base / new_session_id

    # Create directory (handle UC Volume creation if needed)
    _ensure_session_dir(session_dir)

    return new_session_id, session_dir


def _ensure_session_dir(session_dir: Path) -> None:
    """Create session directory, handling UC Volume paths.

    Args:
        session_dir: Path to session directory.
    """
    path_str = str(session_dir)

    # For UC volume paths, ensure volume exists first
    if path_str.startswith("/Volumes/"):
        _ensure_volume_exists(path_str)

    session_dir.mkdir(parents=True, exist_ok=True)


def _ensure_volume_exists(path: str) -> None:
    """Create Unity Catalog volume if it doesn't exist.

    UC volumes cannot be created with mkdir - they must be created via SQL.
    Path format: /Volumes/<catalog>/<schema>/<volume>/...
    """
    parts = path.split("/")
    if len(parts) < 5:
        return  # Not a valid UC volume path

    catalog, schema, volume = parts[2], parts[3], parts[4]
    volume_path = Path(f"/Volumes/{catalog}/{schema}/{volume}")

    if volume_path.exists():
        return

    # Try to create volume using Spark SQL
    logger.info(f"Creating Unity Catalog volume: {catalog}.{schema}.{volume}")
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        spark.sql(f"CREATE VOLUME IF NOT EXISTS `{catalog}`.`{schema}`.`{volume}`")
        logger.info(f"Created Unity Catalog volume: {catalog}.{schema}.{volume}")
    except ImportError:
        logger.warning("PySpark not available, assuming volume exists")
    except Exception as e:
        logger.warning(f"Could not create volume (may already exist): {e}")


def list_user_sessions(
    user_prefix: str,
    volume_path: str,
    limit: int = 20,
) -> list[SessionInfo]:
    """List past sessions for a user.

    Args:
        user_prefix: User's session prefix (from email hash).
        volume_path: Base Unity Catalog Volume path.
        limit: Maximum number of sessions to return.

    Returns:
        List of SessionInfo, sorted by creation time (newest first).
    """
    sessions_base = Path(volume_path) / "sessions"

    if not sessions_base.exists():
        return []

    sessions = []

    # Find all session directories for this user
    for session_dir in sessions_base.iterdir():
        if not session_dir.is_dir():
            continue

        session_id = session_dir.name

        # Only include sessions for this user (matching prefix)
        if not session_id.startswith(user_prefix):
            continue

        # Parse session info
        session_info = _parse_session_info(session_id, session_dir)
        if session_info:
            sessions.append(session_info)

    # Sort by creation time (newest first) and limit
    sessions.sort(key=lambda s: s.created_at, reverse=True)
    return sessions[:limit]


def _parse_session_info(session_id: str, session_dir: Path) -> Optional[SessionInfo]:
    """Parse session information from directory.

    Args:
        session_id: Session ID (directory name).
        session_dir: Path to session directory.

    Returns:
        SessionInfo or None if invalid.
    """
    try:
        # Parse timestamp from session_id: {prefix}_{YYYYMMDD}_{HHMMSS}
        parts = session_id.split("_")
        if len(parts) >= 3:
            date_str = parts[1]
            time_str = parts[2]
            created_at = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
        else:
            # Fallback to directory mtime
            created_at = datetime.fromtimestamp(session_dir.stat().st_mtime)

        # Read task status if available
        tasks_file = session_dir / "eval_tasks.json"
        experiment_id = ""
        task_count = 0
        completed_count = 0
        status = "running"

        if tasks_file.exists():
            try:
                data = json.loads(tasks_file.read_text())
                if isinstance(data, dict):
                    tasks = data.get("tasks", [])
                    experiment_id = data.get("experiment_id", "")
                else:
                    tasks = data

                task_count = len(tasks)
                completed_count = sum(1 for t in tasks if t.get("status") == "completed")

                if task_count > 0 and completed_count == task_count:
                    status = "completed"
                elif any(t.get("status") == "failed" for t in tasks):
                    status = "error"
            except Exception:
                pass

        # Read experiment_id from analysis state if not in tasks
        if not experiment_id:
            analysis_file = session_dir / "state" / "analysis.json"
            if analysis_file.exists():
                try:
                    analysis = json.loads(analysis_file.read_text())
                    experiment_id = analysis.get("experiment_id", "")
                except Exception:
                    pass

        return SessionInfo(
            session_id=session_id,
            session_dir=session_dir,
            created_at=created_at,
            experiment_id=experiment_id,
            status=status,
            task_count=task_count,
            completed_count=completed_count,
        )
    except Exception as e:
        logger.warning(f"Could not parse session {session_id}: {e}")
        return None


def save_session_metadata(
    session_dir: Path,
    experiment_id: str,
    mode: str,
    model: str,
) -> None:
    """Save session metadata to a file.

    Args:
        session_dir: Session directory path.
        experiment_id: MLflow experiment ID.
        mode: "autonomous" or "interactive".
        model: Model being used.
    """
    metadata = {
        "experiment_id": experiment_id,
        "mode": mode,
        "model": model,
        "created_at": datetime.now().isoformat(),
    }

    metadata_file = session_dir / "session_metadata.json"
    metadata_file.write_text(json.dumps(metadata, indent=2))
