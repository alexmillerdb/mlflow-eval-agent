"""User authentication and identity extraction for Databricks Apps.

Extracts user identity from:
1. Databricks OAuth headers (X-Forwarded-Email) when deployed as Databricks App
2. Local username fallback for development

Generates deterministic session prefixes from user email hash for isolation.
"""

import getpass
import hashlib
from dataclasses import dataclass
from typing import Optional

import streamlit as st


@dataclass
class UserIdentity:
    """User identity information."""
    email: str
    display_name: str
    session_prefix: str
    is_authenticated: bool


def get_user_identity() -> UserIdentity:
    """Extract user identity from Databricks OAuth headers or fallback to local user.

    In Databricks Apps, the OAuth proxy sets:
    - X-Forwarded-Email: user's email address
    - X-Forwarded-User: user's username
    - X-Forwarded-Preferred-Username: preferred username

    Returns:
        UserIdentity with email, display name, and session prefix.
    """
    email: Optional[str] = None
    display_name: Optional[str] = None

    # Try to get from Streamlit headers (Databricks App context)
    try:
        headers = st.context.headers
        if headers:
            # Databricks OAuth proxy headers
            email = headers.get("X-Forwarded-Email")
            display_name = headers.get("X-Forwarded-Preferred-Username") or headers.get("X-Forwarded-User")
    except Exception:
        # st.context.headers may not be available in all Streamlit versions
        pass

    # Fallback to local user for development
    if not email:
        local_user = getpass.getuser()
        email = f"{local_user}@local"
        display_name = local_user

    if not display_name:
        display_name = email.split("@")[0] if email else "unknown"

    # Generate deterministic session prefix from email
    session_prefix = _generate_session_prefix(email)

    return UserIdentity(
        email=email,
        display_name=display_name,
        session_prefix=session_prefix,
        is_authenticated=not email.endswith("@local"),
    )


def _generate_session_prefix(email: str) -> str:
    """Generate a deterministic 8-character prefix from email hash.

    This ensures each user gets isolated session directories while
    maintaining a predictable prefix for debugging and lookup.

    Args:
        email: User's email address.

    Returns:
        8-character hex prefix (e.g., "a1b2c3d4").
    """
    return hashlib.md5(email.lower().encode()).hexdigest()[:8]


def require_authentication() -> Optional[UserIdentity]:
    """Check if user is authenticated (for pages requiring auth).

    Returns:
        UserIdentity if authenticated, None otherwise (displays error).
    """
    user = get_user_identity()

    if not user.is_authenticated:
        st.warning(
            "Running in local development mode. "
            "Deploy as a Databricks App for full authentication."
        )

    return user
