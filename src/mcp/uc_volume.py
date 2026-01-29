"""MCP tools for Unity Catalog Volume operations.

Provides file sync between local workspace and UC Volumes in Databricks Apps,
where FUSE mounts are unavailable. WorkspaceClient auto-detects OAuth credentials
in Databricks Apps runtime.

Tools:
- sync_from_uc: Download files from UC Volume to local workspace
- sync_to_uc: Upload files from local workspace to UC Volume
- list_uc_directory: List contents of a UC Volume directory
"""

import io
import logging
import os
from pathlib import Path
from typing import Any

from claude_agent_sdk import tool, create_sdk_mcp_server
from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)

MCP_SERVER_NAME = "uc-volume"


def _text_result(msg: str) -> dict:
    """Create MCP tool result format."""
    return {"content": [{"type": "text", "text": msg}]}


class UCVolumeTools:
    """Tool names for the UC Volume MCP server."""
    SYNC_FROM_UC = f"mcp__{MCP_SERVER_NAME}__sync_from_uc"
    SYNC_TO_UC = f"mcp__{MCP_SERVER_NAME}__sync_to_uc"
    LIST_UC_DIRECTORY = f"mcp__{MCP_SERVER_NAME}__list_uc_directory"


def _validate_uc_path(uc_path: str) -> str:
    """Validate that uc_path is a valid UC Volume path.

    Args:
        uc_path: Path to validate

    Returns:
        Validated path

    Raises:
        ValueError: If path doesn't start with /Volumes/
    """
    if not uc_path.startswith("/Volumes/"):
        raise ValueError(f"UC path must start with '/Volumes/', got: {uc_path}")
    return uc_path


def _get_workspace_client() -> WorkspaceClient:
    """Get WorkspaceClient with auto OAuth detection.

    In Databricks Apps runtime, credentials are auto-detected.
    For local development, uses DATABRICKS_HOST/TOKEN env vars or config profile.
    """
    return WorkspaceClient()


def create_uc_volume_server():
    """Create MCP server for UC Volume operations.

    WorkspaceClient auto-detects OAuth in Databricks Apps runtime.
    For local development, set DATABRICKS_HOST and DATABRICKS_TOKEN env vars.
    """

    @tool(
        "sync_from_uc",
        "Download files from UC Volume to local workspace. Use for loading data, configs, or previous session artifacts.",
        {
            "uc_path": str,
            "local_path": str,
            "recursive": bool,
        }
    )
    async def sync_from_uc(args: dict[str, Any]) -> dict[str, Any]:
        """Download files from UC Volume to local workspace.

        Args:
            uc_path: Source path in UC Volume (must start with /Volumes/)
            local_path: Destination path on local filesystem
            recursive: If True, download directory contents recursively (default: False)

        Returns:
            dict with success status and details
        """
        try:
            uc_path = _validate_uc_path(args.get("uc_path", ""))
            local_path = args.get("local_path", "")
            recursive = args.get("recursive", False)

            if not local_path:
                return _text_result("[UC Volume] Error: local_path required")

            client = _get_workspace_client()
            local_path_obj = Path(local_path)

            # Ensure parent directory exists
            local_path_obj.parent.mkdir(parents=True, exist_ok=True)

            if recursive:
                # Download directory recursively
                downloaded = _download_directory(client, uc_path, local_path_obj)
                return _text_result(
                    f"[UC Volume] Downloaded {downloaded} files from {uc_path} to {local_path}"
                )
            else:
                # Download single file
                response = client.files.download(uc_path)
                content = response.contents.read()
                local_path_obj.write_bytes(content)
                return _text_result(
                    f"[UC Volume] Downloaded {uc_path} to {local_path} ({len(content)} bytes)"
                )

        except ValueError as e:
            return _text_result(f"[UC Volume] Validation error: {str(e)}")
        except Exception as e:
            logger.exception("Error in sync_from_uc")
            return _text_result(f"[UC Volume] Error: {str(e)}")

    @tool(
        "sync_to_uc",
        "Upload files from local workspace to UC Volume. Use for persisting results, datasets, or evaluation artifacts.",
        {
            "local_path": str,
            "uc_path": str,
            "recursive": bool,
        }
    )
    async def sync_to_uc(args: dict[str, Any]) -> dict[str, Any]:
        """Upload files from local workspace to UC Volume.

        Args:
            local_path: Source path on local filesystem
            uc_path: Destination path in UC Volume (must start with /Volumes/)
            recursive: If True, upload directory contents recursively (default: False)

        Returns:
            dict with success status and details
        """
        try:
            local_path = args.get("local_path", "")
            uc_path = _validate_uc_path(args.get("uc_path", ""))
            recursive = args.get("recursive", False)

            if not local_path:
                return _text_result("[UC Volume] Error: local_path required")

            local_path_obj = Path(local_path)
            if not local_path_obj.exists():
                return _text_result(f"[UC Volume] Error: local path does not exist: {local_path}")

            client = _get_workspace_client()

            if recursive and local_path_obj.is_dir():
                # Upload directory recursively
                uploaded = _upload_directory(client, local_path_obj, uc_path)
                return _text_result(
                    f"[UC Volume] Uploaded {uploaded} files from {local_path} to {uc_path}"
                )
            else:
                # Upload single file
                content = local_path_obj.read_bytes()
                client.files.upload(uc_path, io.BytesIO(content), overwrite=True)
                return _text_result(
                    f"[UC Volume] Uploaded {local_path} to {uc_path} ({len(content)} bytes)"
                )

        except ValueError as e:
            return _text_result(f"[UC Volume] Validation error: {str(e)}")
        except Exception as e:
            logger.exception("Error in sync_to_uc")
            return _text_result(f"[UC Volume] Error: {str(e)}")

    @tool(
        "list_uc_directory",
        "List contents of a UC Volume directory. Returns files and subdirectories with sizes.",
        {
            "uc_path": str,
        }
    )
    async def list_uc_directory(args: dict[str, Any]) -> dict[str, Any]:
        """List contents of a UC Volume directory.

        Args:
            uc_path: Directory path in UC Volume (must start with /Volumes/)

        Returns:
            dict with directory listing
        """
        try:
            uc_path = _validate_uc_path(args.get("uc_path", ""))

            client = _get_workspace_client()
            contents = client.files.list_directory_contents(uc_path)

            # Format listing
            items = []
            for item in contents:
                entry = {
                    "name": item.name,
                    "path": item.path,
                    "is_directory": item.is_directory,
                }
                if not item.is_directory and hasattr(item, "file_size"):
                    entry["size"] = item.file_size
                if hasattr(item, "modification_time"):
                    entry["modified"] = str(item.modification_time)
                items.append(entry)

            # Format as text table
            lines = [f"Contents of {uc_path}:", ""]
            for item in items:
                item_type = "DIR " if item["is_directory"] else "FILE"
                size_str = f"{item.get('size', 0):>10} bytes" if not item["is_directory"] else "           -"
                lines.append(f"  {item_type}  {size_str}  {item['name']}")

            if not items:
                lines.append("  (empty directory)")

            lines.append("")
            lines.append(f"Total: {len(items)} items")

            return _text_result("\n".join(lines))

        except ValueError as e:
            return _text_result(f"[UC Volume] Validation error: {str(e)}")
        except Exception as e:
            logger.exception("Error in list_uc_directory")
            return _text_result(f"[UC Volume] Error: {str(e)}")

    return create_sdk_mcp_server(
        name=MCP_SERVER_NAME,
        version="1.0.0",
        tools=[sync_from_uc, sync_to_uc, list_uc_directory],
    )


def _download_directory(client: WorkspaceClient, uc_path: str, local_path: Path) -> int:
    """Recursively download a directory from UC Volume.

    Args:
        client: WorkspaceClient instance
        uc_path: Source directory path in UC Volume
        local_path: Destination directory path

    Returns:
        Number of files downloaded
    """
    downloaded = 0
    local_path.mkdir(parents=True, exist_ok=True)

    contents = client.files.list_directory_contents(uc_path)
    for item in contents:
        local_item_path = local_path / item.name

        if item.is_directory:
            # Recurse into subdirectory
            downloaded += _download_directory(client, item.path, local_item_path)
        else:
            # Download file
            response = client.files.download(item.path)
            content = response.contents.read()
            local_item_path.write_bytes(content)
            downloaded += 1

    return downloaded


def _upload_directory(client: WorkspaceClient, local_path: Path, uc_path: str) -> int:
    """Recursively upload a directory to UC Volume.

    Args:
        client: WorkspaceClient instance
        local_path: Source directory path
        uc_path: Destination directory path in UC Volume

    Returns:
        Number of files uploaded
    """
    uploaded = 0

    for item in local_path.iterdir():
        uc_item_path = f"{uc_path}/{item.name}"

        if item.is_dir():
            # Recurse into subdirectory
            uploaded += _upload_directory(client, item, uc_item_path)
        else:
            # Upload file
            content = item.read_bytes()
            client.files.upload(uc_item_path, io.BytesIO(content), overwrite=True)
            uploaded += 1

    return uploaded
