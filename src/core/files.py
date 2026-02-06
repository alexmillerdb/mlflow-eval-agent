"""Databricks Files SDK wrapper for UC Volumes and Workspace Files.

Provides unified interface for file operations on:
- Unity Catalog Volumes (/Volumes/<catalog>/<schema>/<volume>/...)
- Databricks Workspace Files (/Workspace/...)

Single entry point: use functions directly, WorkspaceClient is lazily imported.
"""

import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class FilesSDKError(Exception):
    """Exception for Files SDK operations."""
    pass


# =============================================================================
# WORKSPACE CLIENT
# =============================================================================


class WorkspaceClientManager:
    """Manages WorkspaceClient instances.

    Service client (no token) is cached as a singleton.
    OBO clients (with user_token) are never cached -- each call creates a fresh client.
    """

    _service_client = None
    _lock = threading.Lock()

    @classmethod
    def get_service_client(cls):
        """Get cached service-principal WorkspaceClient."""
        if cls._service_client is None:
            with cls._lock:
                if cls._service_client is None:
                    try:
                        from databricks.sdk import WorkspaceClient
                        cls._service_client = WorkspaceClient()
                    except ImportError:
                        raise FilesSDKError(
                            "[Files] databricks-sdk not installed. Run: pip install databricks-sdk"
                        )
                    except Exception as e:
                        raise FilesSDKError(f"[Files] Failed to create WorkspaceClient: {e}")
        return cls._service_client

    @classmethod
    def get_user_client(cls, user_token: str):
        """Create a fresh OBO WorkspaceClient (never cached)."""
        host = os.getenv("DATABRICKS_HOST")
        if not host:
            raise FilesSDKError("[Files] DATABRICKS_HOST required for OBO authentication")
        try:
            from databricks.sdk import WorkspaceClient
            return WorkspaceClient(host=host, token=user_token)
        except ImportError:
            raise FilesSDKError(
                "[Files] databricks-sdk not installed. Run: pip install databricks-sdk"
            )
        except Exception as e:
            raise FilesSDKError(f"[Files] Failed to create WorkspaceClient: {e}")

    @classmethod
    def get_client(cls, user_token: Optional[str] = None):
        """Get appropriate client: OBO if token provided, service otherwise."""
        if user_token:
            return cls.get_user_client(user_token)
        return cls.get_service_client()

    @classmethod
    def clear_cache(cls):
        """Clear cached service client (for testing)."""
        with cls._lock:
            cls._service_client = None


def get_workspace_client(user_token: Optional[str] = None):
    """Get WorkspaceClient with optional on-behalf-of token.

    Args:
        user_token: Optional OAuth token for OBO authentication.
                   If not provided, uses default auth (profile or env vars).

    Returns:
        WorkspaceClient instance

    Raises:
        FilesSDKError: If client creation fails
    """
    return WorkspaceClientManager.get_client(user_token)


def clear_workspace_client_cache():
    """Clear cached WorkspaceClient (for testing)."""
    WorkspaceClientManager.clear_cache()


# =============================================================================
# UC VOLUME HELPERS
# =============================================================================


def get_uc_volume_base_path() -> str:
    """Build UC Volume base path from environment variables.

    Reads from:
    - UC_CATALOG_NAME (default: "users")
    - UC_SCHEMA_NAME (required)
    - UC_VOLUME (default: "agent_testing")

    Returns:
        Path like "/Volumes/users/alex_miller/agent_testing"

    Raises:
        FilesSDKError: If required env vars are missing
    """
    catalog = os.getenv("UC_CATALOG_NAME", "users")
    schema = os.getenv("UC_SCHEMA_NAME")
    volume = os.getenv("UC_VOLUME", "agent_testing")

    if not schema:
        raise FilesSDKError(
            "[Files] UC_SCHEMA_NAME environment variable required. "
            "Set UC_CATALOG_NAME, UC_SCHEMA_NAME, UC_VOLUME for UC Volume access."
        )

    return f"/Volumes/{catalog}/{schema}/{volume}"


def _normalize_uc_path(path: str) -> str:
    """Normalize a UC Volume path.

    Handles both absolute paths (/Volumes/...) and relative paths.
    Relative paths are resolved against get_uc_volume_base_path().

    Args:
        path: File path (absolute or relative)

    Returns:
        Absolute UC Volume path
    """
    if path.startswith("/Volumes/"):
        return path
    # Treat as relative to volume base
    base = get_uc_volume_base_path()
    # Remove leading slash from relative path if present
    path = path.lstrip("/")
    return f"{base}/{path}"


# =============================================================================
# UC VOLUME OPERATIONS
# =============================================================================


def uc_volume_read(path: str, user_token: Optional[str] = None) -> str:
    """Read text file from Unity Catalog Volume.

    Args:
        path: File path (absolute /Volumes/... or relative to volume base)
        user_token: Optional OAuth token for OBO authentication

    Returns:
        File contents as string

    Raises:
        FilesSDKError: If read fails
    """
    full_path = _normalize_uc_path(path)
    logger.debug(f"Reading UC Volume file: {full_path}")

    try:
        client = get_workspace_client(user_token)
        response = client.files.download(full_path)
        content = response.contents.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        return content
    except Exception as e:
        raise FilesSDKError(f"[Files] Failed to read {full_path}: {e}")


def uc_volume_write(path: str, content: str, user_token: Optional[str] = None) -> str:
    """Write text file to Unity Catalog Volume.

    Args:
        path: File path (absolute /Volumes/... or relative to volume base)
        content: Text content to write
        user_token: Optional OAuth token for OBO authentication

    Returns:
        Full path of written file

    Raises:
        FilesSDKError: If write fails
    """
    import io

    full_path = _normalize_uc_path(path)
    logger.debug(f"Writing UC Volume file: {full_path}")

    try:
        client = get_workspace_client(user_token)
        # Convert string to bytes and wrap in BytesIO (SDK expects BinaryIO)
        content_bytes = content.encode("utf-8") if isinstance(content, str) else content
        content_stream = io.BytesIO(content_bytes)
        client.files.upload(full_path, content_stream, overwrite=True)
        logger.info(f"Written: {full_path}")
        return full_path
    except Exception as e:
        raise FilesSDKError(f"[Files] Failed to write {full_path}: {e}")


def uc_volume_list(path: str = "", user_token: Optional[str] = None) -> list[dict]:
    """List contents of Unity Catalog Volume directory.

    Args:
        path: Directory path (absolute /Volumes/... or relative to volume base)
              Empty string lists volume root
        user_token: Optional OAuth token for OBO authentication

    Returns:
        List of dicts with 'name', 'path', 'is_directory', 'size' keys

    Raises:
        FilesSDKError: If listing fails
    """
    if path:
        full_path = _normalize_uc_path(path)
    else:
        full_path = get_uc_volume_base_path()

    logger.debug(f"Listing UC Volume directory: {full_path}")

    try:
        client = get_workspace_client(user_token)
        items = client.files.list_directory_contents(full_path)

        result = []
        for item in items:
            result.append({
                "name": item.name,
                "path": item.path,
                "is_directory": item.is_directory,
                "size": getattr(item, "file_size", None),
            })
        return result
    except Exception as e:
        raise FilesSDKError(f"[Files] Failed to list {full_path}: {e}")


def uc_volume_exists(path: str, user_token: Optional[str] = None) -> bool:
    """Check if file or directory exists in Unity Catalog Volume.

    Args:
        path: File path (absolute /Volumes/... or relative to volume base)
        user_token: Optional OAuth token for OBO authentication

    Returns:
        True if path exists, False otherwise
    """
    full_path = _normalize_uc_path(path)
    logger.debug(f"Checking UC Volume path exists: {full_path}")

    try:
        client = get_workspace_client(user_token)
        client.files.get_metadata(full_path)
        return True
    except Exception:
        return False


def uc_volume_delete(path: str, user_token: Optional[str] = None) -> bool:
    """Delete file from Unity Catalog Volume.

    Args:
        path: File path (absolute /Volumes/... or relative to volume base)
        user_token: Optional OAuth token for OBO authentication

    Returns:
        True if deleted successfully

    Raises:
        FilesSDKError: If delete fails
    """
    full_path = _normalize_uc_path(path)
    logger.debug(f"Deleting UC Volume file: {full_path}")

    try:
        client = get_workspace_client(user_token)
        client.files.delete(full_path)
        logger.info(f"Deleted: {full_path}")
        return True
    except Exception as e:
        raise FilesSDKError(f"[Files] Failed to delete {full_path}: {e}")


# =============================================================================
# WORKSPACE FILE OPERATIONS
# =============================================================================


def _normalize_workspace_path(path: str) -> str:
    """Normalize a Workspace path.

    Ensures path starts with /Workspace/.

    Args:
        path: File path

    Returns:
        Path starting with /Workspace/
    """
    # Already has /Workspace/ prefix
    if path.startswith("/Workspace/"):
        return path
    # Exact match for /Workspace
    if path == "/Workspace":
        return path
    # Has leading slash but not /Workspace
    if path.startswith("/"):
        return f"/Workspace{path}"
    # No leading slash
    return f"/Workspace/{path}"


def workspace_read(path: str, user_token: Optional[str] = None) -> str:
    """Read file from Databricks Workspace.

    Args:
        path: File path (will be prefixed with /Workspace/ if needed)
        user_token: Optional OAuth token for OBO authentication

    Returns:
        File contents as string

    Raises:
        FilesSDKError: If read fails
    """
    full_path = _normalize_workspace_path(path)
    logger.debug(f"Reading Workspace file: {full_path}")

    try:
        client = get_workspace_client(user_token)
        # Use workspace API for /Workspace paths
        import base64
        content = client.workspace.export_workspace(full_path)
        # Content is base64 encoded
        if content.content:
            decoded = base64.b64decode(content.content)
            if isinstance(decoded, bytes):
                decoded = decoded.decode("utf-8")
            return decoded
        return ""
    except Exception as e:
        raise FilesSDKError(f"[Workspace] Failed to read {full_path}: {e}")


def workspace_write(
    path: str,
    content: str,
    overwrite: bool = True,
    user_token: Optional[str] = None
) -> str:
    """Write file to Databricks Workspace.

    Args:
        path: File path (will be prefixed with /Workspace/ if needed)
        content: Text content to write
        overwrite: Whether to overwrite existing file (default True)
        user_token: Optional OAuth token for OBO authentication

    Returns:
        Full path of written file

    Raises:
        FilesSDKError: If write fails
    """
    full_path = _normalize_workspace_path(path)
    logger.debug(f"Writing Workspace file: {full_path}")

    try:
        client = get_workspace_client(user_token)
        import base64
        from databricks.sdk.service.workspace import ImportFormat, Language

        # Encode content as base64
        content_bytes = content.encode("utf-8") if isinstance(content, str) else content
        content_b64 = base64.b64encode(content_bytes).decode("utf-8")

        client.workspace.import_workspace(
            path=full_path,
            content=content_b64,
            format=ImportFormat.AUTO,
            overwrite=overwrite,
        )
        logger.info(f"Written: {full_path}")
        return full_path
    except Exception as e:
        raise FilesSDKError(f"[Workspace] Failed to write {full_path}: {e}")


def workspace_list(path: str = "/Workspace", user_token: Optional[str] = None) -> list[dict]:
    """List contents of Databricks Workspace directory.

    Args:
        path: Directory path (default /Workspace)
        user_token: Optional OAuth token for OBO authentication

    Returns:
        List of dicts with 'name', 'path', 'type' keys

    Raises:
        FilesSDKError: If listing fails
    """
    full_path = _normalize_workspace_path(path)
    logger.debug(f"Listing Workspace directory: {full_path}")

    try:
        client = get_workspace_client(user_token)
        items = client.workspace.list(full_path)

        result = []
        for item in items:
            result.append({
                "name": item.path.split("/")[-1] if item.path else "",
                "path": item.path,
                "type": str(item.object_type) if item.object_type else "UNKNOWN",
            })
        return result
    except Exception as e:
        raise FilesSDKError(f"[Workspace] Failed to list {full_path}: {e}")


def workspace_mkdir(path: str, user_token: Optional[str] = None) -> str:
    """Create directory in Databricks Workspace.

    Args:
        path: Directory path (will be prefixed with /Workspace/ if needed)
        user_token: Optional OAuth token for OBO authentication

    Returns:
        Full path of created directory

    Raises:
        FilesSDKError: If creation fails
    """
    full_path = _normalize_workspace_path(path)
    logger.debug(f"Creating Workspace directory: {full_path}")

    try:
        client = get_workspace_client(user_token)
        client.workspace.mkdirs(full_path)
        logger.info(f"Created directory: {full_path}")
        return full_path
    except Exception as e:
        raise FilesSDKError(f"[Workspace] Failed to create directory {full_path}: {e}")
