"""Core infrastructure for MLflow Evaluation Agent.

Contains configuration, runtime detection, authentication utilities, and file operations.
"""

from .config import Config
from .runtime import detect_runtime, RuntimeContext, RuntimeInfo, get_sessions_base_path
from .auth import configure_env, get_databricks_host, get_subprocess_env_vars
from .files import (
    uc_volume_read,
    uc_volume_write,
    uc_volume_list,
    uc_volume_exists,
    uc_volume_delete,
    workspace_read,
    workspace_write,
    workspace_list,
    workspace_mkdir,
    get_uc_volume_base_path,
    get_workspace_client,
    FilesSDKError,
)

__all__ = [
    # Config
    "Config",
    # Runtime
    "detect_runtime",
    "RuntimeContext",
    "RuntimeInfo",
    "get_sessions_base_path",
    # Auth
    "configure_env",
    "get_databricks_host",
    "get_subprocess_env_vars",
    # Files - UC Volume
    "uc_volume_read",
    "uc_volume_write",
    "uc_volume_list",
    "uc_volume_exists",
    "uc_volume_delete",
    # Files - Workspace
    "workspace_read",
    "workspace_write",
    "workspace_list",
    "workspace_mkdir",
    # Files - Helpers
    "get_uc_volume_base_path",
    "get_workspace_client",
    "FilesSDKError",
]
