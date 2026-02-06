"""Authentication helpers for Streamlit app."""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_obo_token() -> Optional[str]:
    """Extract on-behalf-of token from Streamlit request headers.

    In Databricks Apps, the proxy injects `x-forwarded-access-token`
    containing the logged-in user's OAuth token.

    Returns:
        OBO token string, or None if not available.
    """
    try:
        import streamlit as st
        return st.context.headers.get("x-forwarded-access-token")
    except Exception:
        return None


def get_current_user() -> Optional[dict]:
    """Get current user info from Databricks.

    Uses the WorkspaceClient to fetch the current authenticated user's info.
    When running in Databricks Apps, uses the OBO token so the identity
    reflects the logged-in user rather than the service principal.

    Returns:
        dict with user_name, display_name, id or None on failure.
    """
    try:
        from src.core.files import get_workspace_client

        user_token = get_obo_token()
        client = get_workspace_client(user_token=user_token)
        me = client.current_user.me()
        return {
            "user_name": me.user_name,
            "display_name": me.display_name or me.user_name,
            "id": me.id,
        }
    except Exception as e:
        logger.warning(f"Could not get current user: {e}")
        return None


def configure_user_context() -> None:
    """Configure auth context and store user info in session state.

    Fetches current user and stores in st.session_state.user.
    Safe to call multiple times - only fetches once per session.
    """
    import streamlit as st

    if "user" not in st.session_state:
        user = get_current_user()
        if user:
            st.session_state.user = user
            logger.info(f"Authenticated as: {user['user_name']}")
        else:
            st.session_state.user = None


def get_user_volume_path(base_volume: str) -> str:
    """Get user-specific volume path for session storage.

    Args:
        base_volume: Base UC Volume path (e.g., "/Volumes/catalog/schema/volume")

    Returns:
        User-specific path like "{base_volume}/users/{user_name}"
        Falls back to "{base_volume}/users/anonymous" if no user.
    """
    import streamlit as st

    user = st.session_state.get("user")
    if user:
        return f"{base_volume}/users/{user['user_name']}"
    return f"{base_volume}/users/anonymous"


def require_auth():
    """Guard function to require authentication.

    Configures user context and stops execution if not authenticated.
    Use at the top of pages that require authentication.
    """
    import streamlit as st

    configure_user_context()
    if not st.session_state.get("user"):
        st.error("Authentication required. Please ensure you're logged into Databricks.")
        st.stop()
