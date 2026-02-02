"""Integration tests for Files SDK operations.

These tests require Databricks authentication and will be skipped if not configured.
Uses real UC Volume/Workspace operations - creates test files with unique names.
"""

import os
import uuid
import pytest
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def is_databricks_configured() -> bool:
    """Check if Databricks auth is configured."""
    # Check for config profile or host+token
    has_profile = bool(os.getenv("DATABRICKS_CONFIG_PROFILE"))
    has_host_token = bool(os.getenv("DATABRICKS_HOST")) and bool(os.getenv("DATABRICKS_TOKEN"))
    return has_profile or has_host_token


def is_uc_configured() -> bool:
    """Check if UC Volume env vars are configured."""
    return bool(os.getenv("UC_SCHEMA_NAME"))


skip_no_databricks = pytest.mark.skipif(
    not is_databricks_configured(),
    reason="Databricks auth not configured (set DATABRICKS_CONFIG_PROFILE or DATABRICKS_HOST+TOKEN)"
)

skip_no_uc = pytest.mark.skipif(
    not is_uc_configured(),
    reason="UC Volume not configured (set UC_SCHEMA_NAME)"
)


@skip_no_databricks
@skip_no_uc
class TestUCVolumeIntegration:
    """Integration tests for UC Volume operations."""

    @pytest.fixture
    def unique_filename(self):
        """Generate unique filename for test isolation."""
        return f"test_{uuid.uuid4().hex[:8]}.txt"

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear WorkspaceClient cache between tests."""
        from src.core.files import clear_workspace_client_cache
        clear_workspace_client_cache()
        yield
        clear_workspace_client_cache()

    def test_write_read_delete_roundtrip(self, unique_filename):
        """Test complete write -> read -> delete cycle."""
        from src.core.files import (
            uc_volume_write,
            uc_volume_read,
            uc_volume_exists,
            uc_volume_delete,
            get_uc_volume_base_path,
        )

        test_content = f"Test content {uuid.uuid4()}"

        # Write file
        full_path = uc_volume_write(unique_filename, test_content)
        assert full_path.startswith("/Volumes/")
        assert unique_filename in full_path

        try:
            # Verify exists
            assert uc_volume_exists(unique_filename) is True

            # Read back
            content = uc_volume_read(unique_filename)
            assert content == test_content

        finally:
            # Cleanup
            uc_volume_delete(unique_filename)

        # Verify deleted
        assert uc_volume_exists(unique_filename) is False

    def test_list_directory(self, unique_filename):
        """Test listing directory after creating file."""
        from src.core.files import (
            uc_volume_write,
            uc_volume_list,
            uc_volume_delete,
        )

        # Create test file
        uc_volume_write(unique_filename, "list test content")

        try:
            # List volume root
            items = uc_volume_list("")

            # Should be a list
            assert isinstance(items, list)

            # Find our file
            file_names = [item["name"] for item in items]
            assert unique_filename in file_names

        finally:
            # Cleanup
            uc_volume_delete(unique_filename)

    def test_overwrite_file(self, unique_filename):
        """Test overwriting existing file."""
        from src.core.files import (
            uc_volume_write,
            uc_volume_read,
            uc_volume_delete,
        )

        # Write initial content
        uc_volume_write(unique_filename, "initial content")

        try:
            # Overwrite with new content
            uc_volume_write(unique_filename, "updated content")

            # Read back
            content = uc_volume_read(unique_filename)
            assert content == "updated content"

        finally:
            uc_volume_delete(unique_filename)

    def test_read_nonexistent_raises(self):
        """Test reading nonexistent file raises error."""
        from src.core.files import uc_volume_read, FilesSDKError

        nonexistent = f"nonexistent_{uuid.uuid4().hex[:8]}.txt"

        with pytest.raises(FilesSDKError) as exc_info:
            uc_volume_read(nonexistent)

        assert "[Files]" in str(exc_info.value)

    def test_unicode_content(self, unique_filename):
        """Test handling of unicode content."""
        from src.core.files import (
            uc_volume_write,
            uc_volume_read,
            uc_volume_delete,
        )

        unicode_content = "Hello 世界! 🚀 Привет мир!"

        uc_volume_write(unique_filename, unicode_content)

        try:
            content = uc_volume_read(unique_filename)
            assert content == unicode_content
        finally:
            uc_volume_delete(unique_filename)


@skip_no_databricks
class TestWorkspaceIntegration:
    """Integration tests for Workspace file operations.

    Note: Workspace tests may require specific permissions.
    These are more limited to avoid permission issues.
    """

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear WorkspaceClient cache between tests."""
        from src.core.files import clear_workspace_client_cache
        clear_workspace_client_cache()
        yield
        clear_workspace_client_cache()

    def test_list_workspace_root(self):
        """Test listing /Workspace root directory."""
        from src.core.files import workspace_list

        items = workspace_list("/Workspace")

        # Should be a list
        assert isinstance(items, list)

        # Should have some items (Users, Repos, etc. are standard)
        # Don't assert specific items as workspace contents vary


@skip_no_databricks
@skip_no_uc
class TestGetUCVolumeBasePathIntegration:
    """Integration test for get_uc_volume_base_path with real env."""

    def test_path_format(self):
        """Test that path is correctly formatted."""
        from src.core.files import get_uc_volume_base_path

        path = get_uc_volume_base_path()

        # Should start with /Volumes/
        assert path.startswith("/Volumes/")

        # Should have exactly 4 path segments: /Volumes/catalog/schema/volume
        parts = path.split("/")
        assert len(parts) == 5  # ['', 'Volumes', 'catalog', 'schema', 'volume']
        assert parts[1] == "Volumes"
