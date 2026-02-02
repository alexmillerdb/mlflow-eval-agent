"""Tests for authentication helpers."""
from unittest.mock import MagicMock, patch

import pytest


class TestGetCurrentUser:
    """Tests for get_current_user function."""

    def test_get_current_user_success(self):
        """Returns user dict on successful API call."""
        from src.app.auth import get_current_user

        mock_user = MagicMock()
        mock_user.user_name = "test@example.com"
        mock_user.display_name = "Test User"
        mock_user.id = "user-123"

        mock_client = MagicMock()
        mock_client.current_user.me.return_value = mock_user

        with patch("src.core.files.get_workspace_client", return_value=mock_client):
            result = get_current_user()

        assert result == {
            "user_name": "test@example.com",
            "display_name": "Test User",
            "id": "user-123",
        }

    def test_get_current_user_no_display_name(self):
        """Falls back to user_name when display_name is None."""
        from src.app.auth import get_current_user

        mock_user = MagicMock()
        mock_user.user_name = "test@example.com"
        mock_user.display_name = None
        mock_user.id = "user-123"

        mock_client = MagicMock()
        mock_client.current_user.me.return_value = mock_user

        with patch("src.core.files.get_workspace_client", return_value=mock_client):
            result = get_current_user()

        assert result["display_name"] == "test@example.com"

    def test_get_current_user_failure(self):
        """Returns None on exception."""
        from src.app.auth import get_current_user

        with patch(
            "src.core.files.get_workspace_client",
            side_effect=Exception("Connection failed"),
        ):
            result = get_current_user()

        assert result is None


class MockSessionState(dict):
    """Mock Streamlit session_state that supports both dict and attribute access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


class TestConfigureUserContext:
    """Tests for configure_user_context function."""

    def test_configure_user_context_stores_user(self):
        """Stores user in session state on success."""
        from src.app.auth import configure_user_context

        mock_session_state = MockSessionState()

        mock_user = {
            "user_name": "test@example.com",
            "display_name": "Test User",
            "id": "user-123",
        }

        with patch("src.app.auth.get_current_user", return_value=mock_user):
            with patch("streamlit.session_state", mock_session_state):
                configure_user_context()

        assert mock_session_state["user"] == mock_user

    def test_configure_user_context_stores_none_on_failure(self):
        """Stores None in session state on failure."""
        from src.app.auth import configure_user_context

        mock_session_state = MockSessionState()

        with patch("src.app.auth.get_current_user", return_value=None):
            with patch("streamlit.session_state", mock_session_state):
                configure_user_context()

        assert mock_session_state["user"] is None

    def test_configure_user_context_skips_if_already_set(self):
        """Does not re-fetch if user already in session state."""
        from src.app.auth import configure_user_context

        existing_user = {"user_name": "existing@example.com"}
        mock_session_state = MockSessionState({"user": existing_user})

        with patch("src.app.auth.get_current_user") as mock_get_user:
            with patch("streamlit.session_state", mock_session_state):
                configure_user_context()

        # Should not call get_current_user since user already exists
        mock_get_user.assert_not_called()
        assert mock_session_state["user"] == existing_user


class TestGetUserVolumePath:
    """Tests for get_user_volume_path function."""

    def test_get_user_volume_path_authenticated(self):
        """Returns user-specific path when authenticated."""
        from src.app.auth import get_user_volume_path

        mock_session_state = {
            "user": {"user_name": "test@example.com"},
        }

        with patch("streamlit.session_state", mock_session_state):
            result = get_user_volume_path("/Volumes/catalog/schema/volume")

        assert result == "/Volumes/catalog/schema/volume/users/test@example.com"

    def test_get_user_volume_path_anonymous(self):
        """Returns anonymous path when not authenticated."""
        from src.app.auth import get_user_volume_path

        mock_session_state = {"user": None}

        with patch("streamlit.session_state", mock_session_state):
            result = get_user_volume_path("/Volumes/catalog/schema/volume")

        assert result == "/Volumes/catalog/schema/volume/users/anonymous"

    def test_get_user_volume_path_no_user_key(self):
        """Returns anonymous path when user key is missing."""
        from src.app.auth import get_user_volume_path

        mock_session_state = {}

        with patch("streamlit.session_state", mock_session_state):
            result = get_user_volume_path("/Volumes/catalog/schema/volume")

        assert result == "/Volumes/catalog/schema/volume/users/anonymous"


class TestRequireAuth:
    """Tests for require_auth function."""

    def test_require_auth_passes_when_authenticated(self):
        """Does not stop when user is authenticated."""
        from src.app.auth import require_auth

        mock_session_state = {
            "user": {"user_name": "test@example.com"},
        }

        with patch("src.app.auth.configure_user_context"):
            with patch("streamlit.session_state", mock_session_state):
                with patch("streamlit.error") as mock_error:
                    with patch("streamlit.stop") as mock_stop:
                        require_auth()

        mock_error.assert_not_called()
        mock_stop.assert_not_called()

    def test_require_auth_stops_when_not_authenticated(self):
        """Calls st.stop when user is not authenticated."""
        from src.app.auth import require_auth

        mock_session_state = {"user": None}

        with patch("src.app.auth.configure_user_context"):
            with patch("streamlit.session_state", mock_session_state):
                with patch("streamlit.error") as mock_error:
                    with patch("streamlit.stop") as mock_stop:
                        require_auth()

        mock_error.assert_called_once()
        mock_stop.assert_called_once()
