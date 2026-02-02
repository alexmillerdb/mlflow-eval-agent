# Phase 2: Files SDK Integration ✅

**Status**: Complete (2026-02-02)
**Branch**: `fix-env-vars`

## Goal

Add MCP tools for Unity Catalog Volumes and Databricks Workspace Files, enabling the agent to read, write, and list files in Databricks storage.

## Features Completed

### Feature 2.1: Create Files SDK Wrapper ✅

Created `src/core/files.py` with:
- `get_workspace_client(user_token)` - Get WorkspaceClient with optional OBO token
- `get_uc_volume_base_path()` - Build `/Volumes/{catalog}/{schema}/{volume}` from env vars
- `FilesSDKError` exception class for consistent error handling

**UC Volume Operations:**
- `uc_volume_read(path)` - Read text file
- `uc_volume_write(path, content)` - Write text file (uses BytesIO stream)
- `uc_volume_list(path)` - List directory contents
- `uc_volume_exists(path)` - Check if file exists (uses `get_metadata`)
- `uc_volume_delete(path)` - Delete file

**Workspace Operations:**
- `workspace_read(path)` - Read file (base64 decoding)
- `workspace_write(path, content)` - Write file (base64 encoding)
- `workspace_list(path)` - List directory
- `workspace_mkdir(path)` - Create directory

**Key Implementation Details:**
- Lazy imports for `WorkspaceClient` (matching `auth.py` pattern)
- Path normalization helpers for both UC Volumes and Workspace
- `_normalize_workspace_path()` handles `/Workspace` prefix correctly

### Feature 2.2: Add File Tools to Agent ✅

Added 6 new MCP tools in `src/agent/tools.py`:
- `uc_volume_read_tool`
- `uc_volume_write_tool`
- `uc_volume_list_tool`
- `workspace_read_tool`
- `workspace_write_tool`
- `workspace_list_tool`

Tools follow existing patterns:
- `@mlflow.trace` + `@tool` decorators
- Return `mlflow_ops.text_result()`
- Call `record_tool_call()` for context tracking
- Proper error handling with `FilesSDKError`

### Feature 2.3: Update MCPTools Constants ✅

Added tool name constants to `MCPTools` class:
```python
UC_VOLUME_READ = f"mcp__{MCP_SERVER_NAME}__uc_volume_read"
UC_VOLUME_WRITE = f"mcp__{MCP_SERVER_NAME}__uc_volume_write"
UC_VOLUME_LIST = f"mcp__{MCP_SERVER_NAME}__uc_volume_list"
WORKSPACE_READ = f"mcp__{MCP_SERVER_NAME}__workspace_read"
WORKSPACE_WRITE = f"mcp__{MCP_SERVER_NAME}__workspace_write"
WORKSPACE_LIST = f"mcp__{MCP_SERVER_NAME}__workspace_list"
```

### Feature 2.4: Update Agent Allowed Tools ✅

Updated `src/agent/agent.py` `_build_options()` to include all 6 file tools in `allowed_tools` list.

Total tools: 9 MCP tools + 5 built-in = 14 allowed tools

### Feature 2.5: Add Files SDK Tests ✅

Created `tests/test_core/test_files.py` with 29 unit tests:
- `TestGetUCVolumeBasePath` - Environment variable handling
- `TestGetWorkspaceClient` - Client creation with/without OBO token
- `TestUCVolumeOperations` - All UC Volume operations with mocks
- `TestWorkspaceOperations` - All Workspace operations with mocks
- `TestFileTools` - MCP tool wrappers
- `TestToolCount` - Verify 9 tools created

### Feature 2.6: Files SDK Integration Testing ✅

Created `tests/integration/test_files_integration.py` with 7 integration tests:
- `TestUCVolumeIntegration` - Real UC Volume operations
  - Write/read/delete roundtrip
  - List directory
  - Overwrite file
  - Read nonexistent raises error
  - Unicode content handling
- `TestWorkspaceIntegration` - Real Workspace operations
  - List workspace root
- `TestGetUCVolumeBasePathIntegration` - Path format validation

Tests use `load_dotenv()` and skip gracefully if Databricks not configured.

## Environment Variables

```bash
UC_CATALOG_NAME="users"      # Default: "users"
UC_SCHEMA_NAME="alex_miller" # Required
UC_VOLUME="agent_testing"    # Default: "agent_testing"
DATABRICKS_CONFIG_PROFILE=aws-apps  # OAuth (preferred)
DATABRICKS_HOST + DATABRICKS_TOKEN  # Fallback
```

## Files Changed

| File | Action | Purpose |
|------|--------|---------|
| `src/core/files.py` | CREATE | Files SDK wrapper |
| `src/core/__init__.py` | MODIFY | Export files functions |
| `src/agent/tools.py` | MODIFY | Add 6 file tools + MCPTools constants |
| `src/agent/agent.py` | MODIFY | Add file tools to allowed_tools |
| `tests/test_core/__init__.py` | CREATE | Test package init |
| `tests/test_core/test_files.py` | CREATE | Unit tests (29 tests) |
| `tests/integration/test_files_integration.py` | CREATE | Integration tests (7 tests) |
| `tests/test_tools_unit.py` | MODIFY | Update expected tool count to 9 |

## Verification Results

```bash
# Import verification
uv run python -c "from src.core.files import uc_volume_read, workspace_read; print('OK')"
# OK

# Tool count
uv run python -c "from src.agent.tools import create_tools; print(len(create_tools()), 'tools')"
# 9 tools

# Unit tests
uv run pytest tests/test_core/test_files.py -v
# 29 passed

# Integration tests (with Databricks auth)
uv run pytest tests/integration/test_files_integration.py -v
# 7 passed

# Full test suite
uv run pytest tests/ --ignore=tests/e2e -v
# 104 passed
```

## Success Criteria

- [x] Unit tests pass with mocked WorkspaceClient
- [x] Real SDK operations succeed (write/read/list/delete)
- [x] Tool wrappers return correct MCP format
- [x] Unicode content handled correctly
- [x] Path normalization works for both absolute and relative paths

## Fixes During Integration

1. `files.upload()` requires `BytesIO` stream, not raw bytes
2. `_normalize_workspace_path("/Workspace")` was doubling the prefix
3. `uc_volume_exists()` should use `get_metadata()` not `get_status()`
