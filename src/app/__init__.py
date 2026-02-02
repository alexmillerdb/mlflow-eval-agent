"""Application layer for MLflow Evaluation Agent.

Provides Streamlit chat interface and async-to-sync streaming utilities.
"""

from .auth import (
    configure_user_context,
    get_current_user,
    get_user_volume_path,
    require_auth,
)
from .streaming import async_to_sync_generator

__all__ = [
    "async_to_sync_generator",
    "configure_user_context",
    "get_current_user",
    "get_user_volume_path",
    "require_auth",
]
