"""Unit tests for src/core/files.py - Files SDK wrapper.

Tests UC Volume and Workspace file operations using mocks.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import io


# =============================================================================
# HELPER FUNCTIONS TESTS
# =============================================================================


class TestGetUCVolumeBasePath:
    """Tests for get_uc_volume_base_path()."""

    def test_returns_path_with_defaults(self, monkeypatch):
        """Test returns path using default catalog and volume."""
        from src.core.files import get_uc_volume_base_path

        monkeypatch.setenv("UC_CATALOG_NAME", "users")
        monkeypatch.setenv("UC_SCHEMA_NAME", "alex_miller")
        monkeypatch.setenv("UC_VOLUME", "agent_testing")

        path = get_uc_volume_base_path()
        assert path == "/Volumes/users/alex_miller/agent_testing"

    def test_returns_path_with_custom_values(self, monkeypatch):
        """Test returns path with custom env values."""
        from src.core.files import get_uc_volume_base_path

        monkeypatch.setenv("UC_CATALOG_NAME", "my_catalog")
        monkeypatch.setenv("UC_SCHEMA_NAME", "my_schema")
        monkeypatch.setenv("UC_VOLUME", "my_volume")

        path = get_uc_volume_base_path()
        assert path == "/Volumes/my_catalog/my_schema/my_volume"

    def test_uses_defaults_for_catalog_and_volume(self, monkeypatch):
        """Test uses default values for catalog and volume."""
        from src.core.files import get_uc_volume_base_path

        monkeypatch.delenv("UC_CATALOG_NAME", raising=False)
        monkeypatch.setenv("UC_SCHEMA_NAME", "test_schema")
        monkeypatch.delenv("UC_VOLUME", raising=False)

        path = get_uc_volume_base_path()
        assert path == "/Volumes/users/test_schema/agent_testing"

    def test_raises_without_schema(self, monkeypatch):
        """Test raises error when schema is missing."""
        from src.core.files import get_uc_volume_base_path, FilesSDKError

        monkeypatch.delenv("UC_SCHEMA_NAME", raising=False)

        with pytest.raises(FilesSDKError) as exc_info:
            get_uc_volume_base_path()

        assert "UC_SCHEMA_NAME" in str(exc_info.value)


class TestGetWorkspaceClient:
    """Tests for get_workspace_client()."""

    def test_creates_client_without_token(self):
        """Test creates WorkspaceClient with default auth."""
        from src.core.files import get_workspace_client, clear_workspace_client_cache

        clear_workspace_client_cache()

        with patch("databricks.sdk.WorkspaceClient") as mock_class:
            mock_client = MagicMock()
            mock_class.return_value = mock_client

            client = get_workspace_client()

            mock_class.assert_called_once_with()
            assert client == mock_client

        clear_workspace_client_cache()

    def test_creates_client_with_token(self, monkeypatch):
        """Test creates WorkspaceClient with OBO token."""
        from src.core.files import get_workspace_client, clear_workspace_client_cache

        clear_workspace_client_cache()
        monkeypatch.setenv("DATABRICKS_HOST", "https://test.cloud.databricks.com")

        with patch("databricks.sdk.WorkspaceClient") as mock_class:
            mock_client = MagicMock()
            mock_class.return_value = mock_client

            client = get_workspace_client("user-token-123")

            mock_class.assert_called_once_with(
                host="https://test.cloud.databricks.com",
                token="user-token-123"
            )

        clear_workspace_client_cache()

    def test_raises_without_host_for_obo(self, monkeypatch):
        """Test raises error when host missing for OBO auth."""
        from src.core.files import get_workspace_client, FilesSDKError, clear_workspace_client_cache

        clear_workspace_client_cache()
        monkeypatch.delenv("DATABRICKS_HOST", raising=False)

        with pytest.raises(FilesSDKError) as exc_info:
            get_workspace_client("user-token-123")

        assert "DATABRICKS_HOST" in str(exc_info.value)

        clear_workspace_client_cache()


# =============================================================================
# UC VOLUME OPERATIONS TESTS
# =============================================================================


class TestUCVolumeOperations:
    """Tests for UC Volume file operations."""

    @pytest.fixture
    def mock_client(self):
        """Create mock WorkspaceClient."""
        client = MagicMock()
        return client

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Set up environment variables."""
        monkeypatch.setenv("UC_CATALOG_NAME", "users")
        monkeypatch.setenv("UC_SCHEMA_NAME", "test_user")
        monkeypatch.setenv("UC_VOLUME", "test_volume")

    def test_uc_volume_read_absolute_path(self, mock_client):
        """Test reading file with absolute path."""
        from src.core.files import uc_volume_read, clear_workspace_client_cache

        clear_workspace_client_cache()

        # Mock file download
        mock_response = MagicMock()
        mock_response.contents.read.return_value = b"file content here"
        mock_client.files.download.return_value = mock_response

        with patch("src.core.files.get_workspace_client", return_value=mock_client):
            content = uc_volume_read("/Volumes/cat/schema/vol/test.txt")

        mock_client.files.download.assert_called_once_with("/Volumes/cat/schema/vol/test.txt")
        assert content == "file content here"

    def test_uc_volume_read_relative_path(self, mock_client):
        """Test reading file with relative path."""
        from src.core.files import uc_volume_read, clear_workspace_client_cache

        clear_workspace_client_cache()

        mock_response = MagicMock()
        mock_response.contents.read.return_value = b"relative file content"
        mock_client.files.download.return_value = mock_response

        with patch("src.core.files.get_workspace_client", return_value=mock_client):
            content = uc_volume_read("subdir/test.txt")

        mock_client.files.download.assert_called_once_with(
            "/Volumes/users/test_user/test_volume/subdir/test.txt"
        )
        assert content == "relative file content"

    def test_uc_volume_write(self, mock_client):
        """Test writing file to UC Volume."""
        from src.core.files import uc_volume_write, clear_workspace_client_cache

        clear_workspace_client_cache()

        with patch("src.core.files.get_workspace_client", return_value=mock_client):
            result = uc_volume_write("output.txt", "test content")

        mock_client.files.upload.assert_called_once()
        call_args = mock_client.files.upload.call_args
        assert call_args[0][0] == "/Volumes/users/test_user/test_volume/output.txt"
        # Second arg is now a BytesIO stream
        stream = call_args[0][1]
        assert stream.read() == b"test content"
        assert call_args[1]["overwrite"] is True
        assert result == "/Volumes/users/test_user/test_volume/output.txt"

    def test_uc_volume_list(self, mock_client):
        """Test listing UC Volume directory."""
        from src.core.files import uc_volume_list, clear_workspace_client_cache

        clear_workspace_client_cache()

        # Mock directory listing
        mock_item1 = MagicMock()
        mock_item1.name = "file1.txt"
        mock_item1.path = "/Volumes/users/test_user/test_volume/file1.txt"
        mock_item1.is_directory = False
        mock_item1.file_size = 1024

        mock_item2 = MagicMock()
        mock_item2.name = "subdir"
        mock_item2.path = "/Volumes/users/test_user/test_volume/subdir"
        mock_item2.is_directory = True

        mock_client.files.list_directory_contents.return_value = [mock_item1, mock_item2]

        with patch("src.core.files.get_workspace_client", return_value=mock_client):
            items = uc_volume_list("")

        assert len(items) == 2
        assert items[0]["name"] == "file1.txt"
        assert items[0]["is_directory"] is False
        assert items[0]["size"] == 1024
        assert items[1]["name"] == "subdir"
        assert items[1]["is_directory"] is True

    def test_uc_volume_exists_true(self, mock_client):
        """Test checking file exists returns True."""
        from src.core.files import uc_volume_exists, clear_workspace_client_cache

        clear_workspace_client_cache()

        mock_client.files.get_metadata.return_value = MagicMock()

        with patch("src.core.files.get_workspace_client", return_value=mock_client):
            exists = uc_volume_exists("test.txt")

        assert exists is True

    def test_uc_volume_exists_false(self, mock_client):
        """Test checking file exists returns False for missing file."""
        from src.core.files import uc_volume_exists, clear_workspace_client_cache

        clear_workspace_client_cache()

        mock_client.files.get_metadata.side_effect = Exception("Not found")

        with patch("src.core.files.get_workspace_client", return_value=mock_client):
            exists = uc_volume_exists("missing.txt")

        assert exists is False

    def test_uc_volume_delete(self, mock_client):
        """Test deleting file from UC Volume."""
        from src.core.files import uc_volume_delete, clear_workspace_client_cache

        clear_workspace_client_cache()

        with patch("src.core.files.get_workspace_client", return_value=mock_client):
            result = uc_volume_delete("test.txt")

        mock_client.files.delete.assert_called_once_with(
            "/Volumes/users/test_user/test_volume/test.txt"
        )
        assert result is True

    def test_uc_volume_read_error(self, mock_client):
        """Test read raises FilesSDKError on failure."""
        from src.core.files import uc_volume_read, FilesSDKError, clear_workspace_client_cache

        clear_workspace_client_cache()

        mock_client.files.download.side_effect = Exception("Network error")

        with patch("src.core.files.get_workspace_client", return_value=mock_client):
            with pytest.raises(FilesSDKError) as exc_info:
                uc_volume_read("test.txt")

        assert "[Files]" in str(exc_info.value)
        assert "Network error" in str(exc_info.value)


# =============================================================================
# WORKSPACE OPERATIONS TESTS
# =============================================================================


class TestWorkspaceOperations:
    """Tests for Workspace file operations."""

    @pytest.fixture
    def mock_client(self):
        """Create mock WorkspaceClient."""
        client = MagicMock()
        return client

    def test_workspace_read(self, mock_client):
        """Test reading file from Workspace."""
        from src.core.files import workspace_read, clear_workspace_client_cache
        import base64

        clear_workspace_client_cache()

        # Mock workspace export
        content_b64 = base64.b64encode(b"workspace file content").decode("utf-8")
        mock_response = MagicMock()
        mock_response.content = content_b64
        mock_client.workspace.export_workspace.return_value = mock_response

        with patch("src.core.files.get_workspace_client", return_value=mock_client):
            content = workspace_read("/Users/test/file.py")

        mock_client.workspace.export_workspace.assert_called_once_with(
            "/Workspace/Users/test/file.py"
        )
        assert content == "workspace file content"

    def test_workspace_read_with_workspace_prefix(self, mock_client):
        """Test reading file already prefixed with /Workspace/."""
        from src.core.files import workspace_read, clear_workspace_client_cache
        import base64

        clear_workspace_client_cache()

        content_b64 = base64.b64encode(b"test content").decode("utf-8")
        mock_response = MagicMock()
        mock_response.content = content_b64
        mock_client.workspace.export_workspace.return_value = mock_response

        with patch("src.core.files.get_workspace_client", return_value=mock_client):
            content = workspace_read("/Workspace/Users/test/file.py")

        mock_client.workspace.export_workspace.assert_called_once_with(
            "/Workspace/Users/test/file.py"
        )

    def test_workspace_write(self, mock_client):
        """Test writing file to Workspace."""
        from src.core.files import workspace_write, clear_workspace_client_cache

        clear_workspace_client_cache()

        with patch("src.core.files.get_workspace_client", return_value=mock_client):
            result = workspace_write("/Users/test/output.py", "print('hello')")

        mock_client.workspace.import_workspace.assert_called_once()
        call_kwargs = mock_client.workspace.import_workspace.call_args[1]
        assert call_kwargs["path"] == "/Workspace/Users/test/output.py"
        assert call_kwargs["overwrite"] is True
        assert result == "/Workspace/Users/test/output.py"

    def test_workspace_list(self, mock_client):
        """Test listing Workspace directory."""
        from src.core.files import workspace_list, clear_workspace_client_cache

        clear_workspace_client_cache()

        # Mock directory listing
        mock_item1 = MagicMock()
        mock_item1.path = "/Workspace/Users/test/file.py"
        mock_item1.object_type = "FILE"

        mock_item2 = MagicMock()
        mock_item2.path = "/Workspace/Users/test/subdir"
        mock_item2.object_type = "DIRECTORY"

        mock_client.workspace.list.return_value = [mock_item1, mock_item2]

        with patch("src.core.files.get_workspace_client", return_value=mock_client):
            items = workspace_list("/Users/test")

        assert len(items) == 2
        assert items[0]["name"] == "file.py"
        assert items[0]["type"] == "FILE"
        assert items[1]["name"] == "subdir"
        assert items[1]["type"] == "DIRECTORY"

    def test_workspace_mkdir(self, mock_client):
        """Test creating Workspace directory."""
        from src.core.files import workspace_mkdir, clear_workspace_client_cache

        clear_workspace_client_cache()

        with patch("src.core.files.get_workspace_client", return_value=mock_client):
            result = workspace_mkdir("/Users/test/newdir")

        mock_client.workspace.mkdirs.assert_called_once_with("/Workspace/Users/test/newdir")
        assert result == "/Workspace/Users/test/newdir"

    def test_workspace_read_error(self, mock_client):
        """Test read raises FilesSDKError on failure."""
        from src.core.files import workspace_read, FilesSDKError, clear_workspace_client_cache

        clear_workspace_client_cache()

        mock_client.workspace.export_workspace.side_effect = Exception("Access denied")

        with patch("src.core.files.get_workspace_client", return_value=mock_client):
            with pytest.raises(FilesSDKError) as exc_info:
                workspace_read("/Users/test/file.py")

        assert "[Workspace]" in str(exc_info.value)
        assert "Access denied" in str(exc_info.value)


# =============================================================================
# TOOL INTEGRATION TESTS
# =============================================================================


class TestFileTools:
    """Tests for file MCP tools."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch, session_dir):
        """Set up environment and session directory."""
        monkeypatch.setenv("UC_CATALOG_NAME", "users")
        monkeypatch.setenv("UC_SCHEMA_NAME", "test_user")
        monkeypatch.setenv("UC_VOLUME", "test_volume")

    @pytest.mark.asyncio
    async def test_uc_volume_read_tool(self, session_dir):
        """Test uc_volume_read MCP tool."""
        from src.agent.tools import create_tools
        from tests.test_tools_unit import get_tool_by_name

        tools = create_tools()
        read_tool = get_tool_by_name(tools, "uc_volume_read")

        with patch("src.core.files.uc_volume_read", return_value="file content"):
            result = await read_tool({"path": "test.txt"})

        assert "content" in result
        assert result["content"][0]["text"] == "file content"

    @pytest.mark.asyncio
    async def test_uc_volume_read_tool_missing_path(self, session_dir):
        """Test uc_volume_read tool returns error without path."""
        from src.agent.tools import create_tools
        from tests.test_tools_unit import get_tool_by_name

        tools = create_tools()
        read_tool = get_tool_by_name(tools, "uc_volume_read")

        result = await read_tool({})

        assert "content" in result
        assert "Error" in result["content"][0]["text"]
        assert "path" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    async def test_uc_volume_write_tool(self, session_dir):
        """Test uc_volume_write MCP tool."""
        from src.agent.tools import create_tools
        from tests.test_tools_unit import get_tool_by_name

        tools = create_tools()
        write_tool = get_tool_by_name(tools, "uc_volume_write")

        with patch("src.core.files.uc_volume_write", return_value="/Volumes/test/file.txt"):
            result = await write_tool({"path": "file.txt", "content": "hello"})

        assert "content" in result
        assert "Written" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_uc_volume_list_tool(self, session_dir):
        """Test uc_volume_list MCP tool."""
        from src.agent.tools import create_tools
        from tests.test_tools_unit import get_tool_by_name

        tools = create_tools()
        list_tool = get_tool_by_name(tools, "uc_volume_list")

        mock_items = [
            {"name": "file1.txt", "is_directory": False, "path": "/Volumes/test/file1.txt", "size": 100},
            {"name": "dir1", "is_directory": True, "path": "/Volumes/test/dir1", "size": None},
        ]

        with patch("src.core.files.uc_volume_list", return_value=mock_items):
            result = await list_tool({"path": ""})

        assert "content" in result
        text = result["content"][0]["text"]
        assert "file1.txt" in text
        assert "dir1" in text

    @pytest.mark.asyncio
    async def test_workspace_read_tool(self, session_dir):
        """Test workspace_read MCP tool."""
        from src.agent.tools import create_tools
        from tests.test_tools_unit import get_tool_by_name

        tools = create_tools()
        read_tool = get_tool_by_name(tools, "workspace_read")

        with patch("src.core.files.workspace_read", return_value="workspace content"):
            result = await read_tool({"path": "/Users/test/file.py"})

        assert "content" in result
        assert result["content"][0]["text"] == "workspace content"

    @pytest.mark.asyncio
    async def test_workspace_write_tool(self, session_dir):
        """Test workspace_write MCP tool."""
        from src.agent.tools import create_tools
        from tests.test_tools_unit import get_tool_by_name

        tools = create_tools()
        write_tool = get_tool_by_name(tools, "workspace_write")

        with patch("src.core.files.workspace_write", return_value="/Workspace/Users/test/file.py"):
            result = await write_tool({"path": "/Users/test/file.py", "content": "print(1)"})

        assert "content" in result
        assert "Written" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_workspace_list_tool(self, session_dir):
        """Test workspace_list MCP tool."""
        from src.agent.tools import create_tools
        from tests.test_tools_unit import get_tool_by_name

        tools = create_tools()
        list_tool = get_tool_by_name(tools, "workspace_list")

        mock_items = [
            {"name": "file.py", "type": "FILE", "path": "/Workspace/Users/test/file.py"},
            {"name": "subdir", "type": "DIRECTORY", "path": "/Workspace/Users/test/subdir"},
        ]

        with patch("src.core.files.workspace_list", return_value=mock_items):
            result = await list_tool({"path": "/Users/test"})

        assert "content" in result
        text = result["content"][0]["text"]
        assert "file.py" in text
        assert "subdir" in text


class TestToolCount:
    """Test that tool count is correct."""

    def test_create_tools_returns_nine_tools(self):
        """create_tools should return exactly 9 tools."""
        from src.agent.tools import create_tools

        tools = create_tools()
        assert len(tools) == 9
