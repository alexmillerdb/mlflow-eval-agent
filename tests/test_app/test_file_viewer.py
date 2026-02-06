"""Tests for the expander-based file viewer component."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


class TestFileViewer:
    """Test render_file_viewer with expander-based layout."""

    @patch("src.app.components.file_viewer.st")
    def test_newest_file_expanded(self, mock_st, tmp_path):
        """Most recently modified file's expander is expanded."""
        from src.app.components.file_viewer import render_file_viewer

        # Create json files with different mtimes (_discover_files picks up *.json)
        old_file = tmp_path / "old.json"
        old_file.write_text('{"old": true}')
        time.sleep(0.05)
        new_file = tmp_path / "new.json"
        new_file.write_text('{"new": true}')

        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=mock_expander)
        mock_expander.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value = mock_expander

        render_file_viewer(tmp_path)

        # Check expander calls
        calls = mock_st.expander.call_args_list
        assert len(calls) == 2

        # Find the call for new.json — should be expanded=True
        new_call = [c for c in calls if "new.json" in str(c)]
        old_call = [c for c in calls if "old.json" in str(c)]
        assert len(new_call) == 1
        assert len(old_call) == 1
        assert new_call[0] == call("\U0001f4c4 new.json", expanded=True)
        assert old_call[0] == call("\U0001f4c4 old.json", expanded=False)

    @patch("src.app.components.file_viewer.st")
    def test_empty_session_dir(self, mock_st, tmp_path):
        """Empty session dir shows info message."""
        from src.app.components.file_viewer import render_file_viewer

        render_file_viewer(tmp_path)

        mock_st.info.assert_called_once()
        assert "No files found" in str(mock_st.info.call_args)

    @patch("src.app.components.file_viewer.st")
    def test_missing_session_dir(self, mock_st, tmp_path):
        """Non-existent session dir shows info message."""
        from src.app.components.file_viewer import render_file_viewer

        render_file_viewer(tmp_path / "nonexistent")

        mock_st.info.assert_called_once()
        assert "No session directory" in str(mock_st.info.call_args)

    @patch("src.app.components.file_viewer.st")
    def test_single_file_expanded(self, mock_st, tmp_path):
        """Single file is always expanded."""
        from src.app.components.file_viewer import render_file_viewer

        (tmp_path / "only.json").write_text('{"only": true}')

        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=mock_expander)
        mock_expander.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value = mock_expander

        render_file_viewer(tmp_path)

        calls = mock_st.expander.call_args_list
        assert len(calls) == 1
        assert calls[0] == call("\U0001f4c4 only.json", expanded=True)
