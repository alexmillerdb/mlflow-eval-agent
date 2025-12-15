# File-System Based State Management for MLflow Eval Agent

Analysis of current workspace patterns and design recommendations for file-system based state management, inspired by Claude Code's architecture.

---

## Current Architecture Assessment

### What's Working Well

**1. Schema-Validated Workspace**
```python
# src/workspace.py - Strong typing prevents garbage-in
WORKSPACE_SCHEMAS = {
    "trace_analysis_summary": TraceAnalysisSummary,
    "error_patterns": list,
    "performance_metrics": PerformanceMetrics,
    ...
}
```
- TypedDict schemas catch invalid data early
- Clear contracts between sub-agents
- Self-documenting expected data shapes

**2. Decoupled Inter-Agent Communication**
```python
# Sub-agents write without knowing who reads
workspace.write("trace_analysis_summary", data, agent="trace_analyst")

# Other agents read what they need
data = workspace.read("trace_analysis_summary")
```
- Loose coupling between agents
- Each agent can evolve independently
- Clear data flow: trace_analyst → context_engineer → agent_architect

**3. Context Injection via `to_context_string()`**
```python
def to_context_string(self) -> str:
    # Generates XML blocks for prompt injection
    parts.append(f'<workspace_entry key="{key}" written_by="{entry["written_by"]}">')
```
- Automatic workspace state in prompts
- XML structure is LLM-friendly
- Truncation warning when data is too large

**4. Timing Metrics for Observability**
```python
def get_timing_metrics(self) -> dict:
    return {
        "total_writes": len(self._write_history),
        "agents_involved": list(set(w["agent"] for w in self._write_history)),
        "duration_seconds": ...,
    }
```
- Track agent coordination patterns
- Debug slow workflows
- Measure inter-agent latency

**5. Freshness Checking**
```python
def read_if_fresh(self, key: str, max_age_seconds: int = 300) -> Optional[Any]:
    if time.time() - entry["timestamp"] > max_age_seconds:
        return None
```
- Prevents stale data usage
- Useful for long-running sessions

---

### What Can Be Improved

**1. Ephemeral State (Lost on Restart)**
```python
# Current: In-memory only
self._data: dict[str, Any] = {}

# Problem: All analysis lost if agent crashes or restarts
# No way to resume a partially completed workflow
```

**2. Context Bloat from `to_context_string()`**
```python
# Current: Dumps ALL workspace data into prompts
for key, entry in self._data.items():
    serialized = json.dumps(entry['data'], indent=2, default=str)
    # Even truncated, 6 keys × 2000 chars = 12K chars in every prompt
```
- **Problem**: All sub-agent prompts get full workspace state
- **Impact**: Wasted tokens, slower responses, higher costs
- **Better**: Selective context injection based on agent needs

**3. Static Workspace Context at Agent Creation**
```python
# src/subagents/__init__.py
def create_subagents(workspace: SharedWorkspace) -> dict[str, AgentDefinition]:
    workspace_context = workspace.to_context_string()  # Snapshot at creation time!
    return {
        "trace_analyst": create_trace_analyst(workspace_context),
        ...
    }
```
- **Problem**: Workspace context is baked into agent prompts at creation
- **Impact**: Later workspace updates not visible to agents without re-creation
- **Better**: Dynamic context retrieval via tools

**4. No Versioning or History**
```python
# Current: Overwrites previous values
self._data[key] = {"data": data, "written_by": agent, "timestamp": time.time()}
```
- **Problem**: Can't see what changed or roll back
- **Impact**: Debugging difficult, no audit trail
- **Better**: Append-only with version tracking

**5. All-or-Nothing Context**
```python
# Current: Sub-agents see everything or nothing
prompt=f"""
## Current Workspace State
{workspace_context}  # ALL keys dumped here
"""
```
- **Problem**: trace_analyst doesn't need context_recommendations
- **Impact**: Irrelevant context dilutes important information
- **Better**: Dependency-based selective injection

**6. No Tool Call Persistence**
- Tool calls and results are in conversation transcript only
- No way to learn from successful tool patterns
- Can't audit what tools were called with what inputs

---

## File-System Based State Management Design

### Inspired by Claude Code Patterns

Claude Code uses a hierarchical file structure for state:
```
~/.claude/
├── projects/<hash>/<session>.jsonl    # Conversation transcripts
├── CLAUDE.md                          # User memory
└── settings.json                      # Configuration

.claude/
├── workspace/                         # Shared state (our addition)
├── agents/                            # Agent definitions
├── commands/                          # Slash commands
├── skills/                            # Skills
└── CLAUDE.md                          # Project memory
```

### Proposed Structure for MLflow Eval Agent

```
.claude/
├── workspace/                         # Persistent shared state
│   ├── _manifest.json                 # Index of all entries + metadata
│   ├── _history/                      # Version history
│   │   └── trace_analysis_summary/
│   │       ├── v1_1702234567.json
│   │       └── v2_1702234890.json
│   ├── trace_analysis_summary.json
│   ├── error_patterns.json
│   ├── performance_metrics.json
│   ├── context_recommendations.json
│   └── extracted_eval_cases.json
│
├── tool_calls/                        # Tool call history
│   ├── _index.json                    # Searchable index
│   └── 2024-12-10/
│       ├── search_traces_001.json
│       ├── get_trace_002.json
│       └── ...
│
├── plans/                             # Analysis plans
│   ├── current_plan.md               # Active plan
│   └── archive/
│       └── 2024-12-10_analysis.md
│
├── sessions/                          # Session state for resumption
│   ├── current.json                   # Active session pointer
│   └── <session-id>/
│       ├── transcript.jsonl          # Conversation history
│       ├── workspace_snapshot.json   # Workspace at session start
│       └── agent_states/
│           ├── trace_analyst.json    # Resumable agent state
│           ├── context_engineer.json
│           └── agent_architect.json
│
└── config/
    ├── agent_tools.json              # Tool access per agent
    └── context_budgets.json          # Token budgets per agent
```

---

## Context Engineering Improvements

### 1. Selective Context Injection

**Current Problem**: All agents get all workspace data.

**Solution**: Dependency declarations per agent.

```python
# src/subagents/context_engineer.py
CONTEXT_ENGINEER_DEPENDENCIES = {
    "required": ["trace_analysis_summary", "error_patterns"],
    "optional": ["performance_metrics"],
    "excluded": ["extracted_eval_cases", "context_recommendations"],  # Don't inject own output
}

def create_context_engineer(workspace: SharedWorkspace) -> AgentDefinition:
    # Only inject relevant context
    context = workspace.to_selective_context(
        keys=CONTEXT_ENGINEER_DEPENDENCIES["required"],
        optional_keys=CONTEXT_ENGINEER_DEPENDENCIES["optional"]
    )
```

**New Workspace Method**:
```python
def to_selective_context(
    self,
    keys: list[str],
    optional_keys: list[str] = None,
    max_chars_per_key: int = 1500
) -> str:
    """Generate context with only specified keys."""
    parts = ["<workspace_context>"]

    for key in keys:
        entry = self._data.get(key)
        if not entry:
            parts.append(f'<missing_dependency key="{key}" />')
            continue
        parts.append(self._format_entry(key, entry, max_chars_per_key))

    for key in (optional_keys or []):
        if key in self._data:
            parts.append(self._format_entry(key, self._data[key], max_chars_per_key))

    parts.append("</workspace_context>")
    return "\n".join(parts)
```

### 2. Dynamic Context via Tools (Not Prompts)

**Current Problem**: Workspace context baked into prompt at agent creation.

**Solution**: Read context dynamically via tools.

```python
@tool("get_workspace_context", "Get current workspace state for specified keys", {"keys": list})
async def get_workspace_context_tool(args: dict) -> dict:
    """Dynamically fetch workspace context."""
    keys = args.get("keys", [])
    result = {}
    for key in keys:
        data = workspace.read(key)
        if data:
            result[key] = data
    return text_result(json.dumps(result, indent=2))
```

**Agent Prompt Update**:
```python
prompt=f"""
## Context Retrieval
Before analysis, use get_workspace_context to load relevant findings:
- For context_engineer: keys=["trace_analysis_summary", "error_patterns"]
- For agent_architect: keys=["trace_analysis_summary", "context_recommendations"]

This ensures you have the LATEST data, not stale snapshots.
"""
```

### 3. Context Budget Management

**Problem**: No control over how much context each agent consumes.

**Solution**: Token budgets per key per agent.

```python
# .claude/config/context_budgets.json
{
    "trace_analyst": {
        "workspace_total_tokens": 2000,
        "per_key_limits": {
            "trace_analysis_summary": 800,
            "error_patterns": 600,
            "performance_metrics": 400
        }
    },
    "context_engineer": {
        "workspace_total_tokens": 3000,
        "per_key_limits": {
            "trace_analysis_summary": 1000,
            "error_patterns": 1200,
            "performance_metrics": 800
        }
    }
}
```

```python
class SharedWorkspace:
    def to_context_string(self, budget: dict = None) -> str:
        """Generate context string respecting token budgets."""
        if not budget:
            return self._to_context_string_unbounded()

        parts = ["<workspace_context>"]
        total_chars = 0
        max_total = budget.get("workspace_total_tokens", 2000) * 4  # ~4 chars/token

        for key, entry in self._data.items():
            if total_chars >= max_total:
                parts.append("<budget_exceeded />")
                break

            key_limit = budget.get("per_key_limits", {}).get(key, 500) * 4
            serialized = json.dumps(entry['data'], indent=2)

            if len(serialized) > key_limit:
                serialized = serialized[:key_limit] + "... [BUDGET_TRUNCATED]"

            parts.append(self._format_entry(key, serialized))
            total_chars += len(serialized)

        parts.append("</workspace_context>")
        return "\n".join(parts)
```

### 4. Workspace Summaries for Context Compression

**Problem**: Full JSON dumps are verbose.

**Solution**: Generate summaries for context injection, full data via tools.

```python
@tool("get_workspace_summary", "Get concise summary of workspace entries", {})
async def get_workspace_summary_tool(args: dict) -> dict:
    """Generate brief summary of workspace state."""
    summary_parts = []
    for key, entry in workspace._data.items():
        data = entry["data"]
        if isinstance(data, list):
            summary_parts.append(f"- {key}: {len(data)} items")
        elif isinstance(data, dict):
            summary_parts.append(f"- {key}: {list(data.keys())}")
    return text_result("\n".join(summary_parts))
```

**Agent Flow**:
1. Agent calls `get_workspace_summary` → sees what's available
2. Agent decides which keys are relevant
3. Agent calls `read_from_workspace` for specific keys
4. Agent processes with focused context

---

## File-System Implementation

### Core Classes

```python
# src/persistence/filesystem.py

from pathlib import Path
import json
from datetime import datetime
from typing import Any, Optional
import hashlib


class FileSystemWorkspace:
    """File-backed workspace with versioning and selective loading."""

    def __init__(self, base_path: Path = None):
        self.base_path = base_path or Path(".claude/workspace")
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Any] = {}  # In-memory cache
        self._manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        """Load workspace manifest (index of all entries)."""
        manifest_path = self.base_path / "_manifest.json"
        if manifest_path.exists():
            return json.loads(manifest_path.read_text())
        return {"entries": {}, "created_at": datetime.now().isoformat()}

    def _save_manifest(self):
        """Persist manifest to disk."""
        manifest_path = self.base_path / "_manifest.json"
        manifest_path.write_text(json.dumps(self._manifest, indent=2))

    def write(self, key: str, data: Any, agent: str = "unknown") -> tuple[bool, str]:
        """Write entry to filesystem with versioning."""
        # Validate schema
        is_valid, msg = validate_workspace_entry(key, data)
        if not is_valid:
            return False, msg

        # Version the old entry if exists
        if key in self._manifest["entries"]:
            self._archive_entry(key)

        # Write new entry
        entry = {
            "data": data,
            "written_by": agent,
            "timestamp": datetime.now().isoformat(),
            "version": self._manifest["entries"].get(key, {}).get("version", 0) + 1,
        }

        file_path = self.base_path / f"{key}.json"
        file_path.write_text(json.dumps(entry, indent=2, default=str))

        # Update manifest
        self._manifest["entries"][key] = {
            "version": entry["version"],
            "written_by": agent,
            "timestamp": entry["timestamp"],
            "size_bytes": file_path.stat().st_size,
        }
        self._save_manifest()

        # Update cache
        self._cache[key] = entry

        return True, f"Wrote {key} v{entry['version']} to workspace"

    def read(self, key: str, use_cache: bool = True) -> Optional[Any]:
        """Read entry, optionally from cache."""
        if use_cache and key in self._cache:
            return self._cache[key]["data"]

        file_path = self.base_path / f"{key}.json"
        if not file_path.exists():
            return None

        entry = json.loads(file_path.read_text())
        self._cache[key] = entry
        return entry["data"]

    def _archive_entry(self, key: str):
        """Archive current entry to history."""
        file_path = self.base_path / f"{key}.json"
        if not file_path.exists():
            return

        history_dir = self.base_path / "_history" / key
        history_dir.mkdir(parents=True, exist_ok=True)

        entry = json.loads(file_path.read_text())
        version = entry.get("version", 1)
        timestamp = int(datetime.now().timestamp())

        archive_path = history_dir / f"v{version}_{timestamp}.json"
        archive_path.write_text(file_path.read_text())

    def get_history(self, key: str, limit: int = 10) -> list[dict]:
        """Get version history for an entry."""
        history_dir = self.base_path / "_history" / key
        if not history_dir.exists():
            return []

        versions = sorted(history_dir.glob("v*.json"), reverse=True)[:limit]
        return [json.loads(v.read_text()) for v in versions]

    def to_selective_context(
        self,
        keys: list[str],
        max_chars_per_key: int = 1500
    ) -> str:
        """Generate context with only specified keys."""
        parts = ["<workspace_context>"]

        for key in keys:
            data = self.read(key)
            if data is None:
                parts.append(f'  <entry key="{key}" status="missing" />')
                continue

            serialized = json.dumps(data, indent=2, default=str)
            if len(serialized) > max_chars_per_key:
                serialized = serialized[:max_chars_per_key] + "\n... [truncated]"

            parts.append(f'  <entry key="{key}">\n{serialized}\n  </entry>')

        parts.append("</workspace_context>")
        return "\n".join(parts)


class ToolCallLogger:
    """Persist tool calls for learning and auditing."""

    def __init__(self, base_path: Path = None):
        self.base_path = base_path or Path(".claude/tool_calls")
        self.base_path.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        tool_name: str,
        tool_input: dict,
        tool_output: Any,
        agent: str = "coordinator",
        execution_time_ms: int = None,
        is_error: bool = False
    ):
        """Log a tool call to filesystem."""
        date_dir = self.base_path / datetime.now().strftime("%Y-%m-%d")
        date_dir.mkdir(exist_ok=True)

        # Find next sequence number
        existing = list(date_dir.glob(f"{tool_name.split('__')[-1]}_*.json"))
        seq = len(existing) + 1

        entry = {
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_output": str(tool_output)[:5000],  # Truncate large outputs
            "agent": agent,
            "timestamp": datetime.now().isoformat(),
            "execution_time_ms": execution_time_ms,
            "is_error": is_error,
        }

        file_path = date_dir / f"{tool_name.split('__')[-1]}_{seq:03d}.json"
        file_path.write_text(json.dumps(entry, indent=2, default=str))

        # Update index
        self._update_index(tool_name, file_path)

    def _update_index(self, tool_name: str, file_path: Path):
        """Update searchable index."""
        index_path = self.base_path / "_index.json"
        index = {}
        if index_path.exists():
            index = json.loads(index_path.read_text())

        if tool_name not in index:
            index[tool_name] = []

        index[tool_name].append({
            "path": str(file_path.relative_to(self.base_path)),
            "timestamp": datetime.now().isoformat()
        })

        # Keep last 100 per tool
        index[tool_name] = index[tool_name][-100:]

        index_path.write_text(json.dumps(index, indent=2))

    def get_recent(self, tool_name: str = None, limit: int = 10) -> list[dict]:
        """Get recent tool calls, optionally filtered by tool name."""
        index_path = self.base_path / "_index.json"
        if not index_path.exists():
            return []

        index = json.loads(index_path.read_text())

        if tool_name:
            paths = index.get(tool_name, [])[-limit:]
        else:
            # All tools, sorted by timestamp
            all_entries = []
            for entries in index.values():
                all_entries.extend(entries)
            paths = sorted(all_entries, key=lambda x: x["timestamp"], reverse=True)[:limit]

        result = []
        for entry in paths:
            path = self.base_path / entry["path"]
            if path.exists():
                result.append(json.loads(path.read_text()))

        return result


class SessionManager:
    """Manage resumable sessions."""

    def __init__(self, base_path: Path = None):
        self.base_path = base_path or Path(".claude/sessions")
        self.base_path.mkdir(parents=True, exist_ok=True)

    def create_session(self, workspace: "FileSystemWorkspace") -> str:
        """Create new session with workspace snapshot."""
        session_id = hashlib.md5(
            datetime.now().isoformat().encode()
        ).hexdigest()[:12]

        session_dir = self.base_path / session_id
        session_dir.mkdir(parents=True)

        # Save workspace snapshot
        snapshot = {
            "workspace_manifest": workspace._manifest,
            "created_at": datetime.now().isoformat(),
        }
        (session_dir / "workspace_snapshot.json").write_text(
            json.dumps(snapshot, indent=2)
        )

        # Update current session pointer
        (self.base_path / "current.json").write_text(
            json.dumps({"session_id": session_id, "started_at": datetime.now().isoformat()})
        )

        return session_id

    def get_current_session(self) -> Optional[str]:
        """Get current session ID."""
        current_path = self.base_path / "current.json"
        if current_path.exists():
            return json.loads(current_path.read_text()).get("session_id")
        return None

    def save_agent_state(
        self,
        session_id: str,
        agent_name: str,
        agent_id: str,
        workspace_keys_written: list[str]
    ):
        """Save agent state for resumption."""
        session_dir = self.base_path / session_id / "agent_states"
        session_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "agent_id": agent_id,
            "workspace_keys_written": workspace_keys_written,
            "saved_at": datetime.now().isoformat(),
        }
        (session_dir / f"{agent_name}.json").write_text(json.dumps(state, indent=2))

    def get_agent_state(self, session_id: str, agent_name: str) -> Optional[dict]:
        """Get saved agent state for resumption."""
        state_path = self.base_path / session_id / "agent_states" / f"{agent_name}.json"
        if state_path.exists():
            return json.loads(state_path.read_text())
        return None
```

---

## Local vs Databricks (UC Volumes) Considerations

### Abstraction Layer

```python
# src/persistence/storage.py

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional


class StorageBackend(ABC):
    """Abstract storage backend for workspace persistence."""

    @abstractmethod
    def write(self, path: str, data: bytes) -> None:
        """Write data to storage."""
        pass

    @abstractmethod
    def read(self, path: str) -> Optional[bytes]:
        """Read data from storage."""
        pass

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if path exists."""
        pass

    @abstractmethod
    def list(self, prefix: str) -> list[str]:
        """List paths under prefix."""
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """Delete path."""
        pass


class LocalFileStorage(StorageBackend):
    """Local filesystem storage."""

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def write(self, path: str, data: bytes) -> None:
        full_path = self.base_path / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(data)

    def read(self, path: str) -> Optional[bytes]:
        full_path = self.base_path / path
        if full_path.exists():
            return full_path.read_bytes()
        return None

    def exists(self, path: str) -> bool:
        return (self.base_path / path).exists()

    def list(self, prefix: str) -> list[str]:
        prefix_path = self.base_path / prefix
        if not prefix_path.exists():
            return []
        return [str(p.relative_to(self.base_path)) for p in prefix_path.rglob("*") if p.is_file()]

    def delete(self, path: str) -> None:
        full_path = self.base_path / path
        if full_path.exists():
            full_path.unlink()


class UCVolumeStorage(StorageBackend):
    """Unity Catalog Volumes storage for Databricks."""

    def __init__(
        self,
        catalog: str,
        schema: str,
        volume: str,
        workspace_client: "WorkspaceClient" = None
    ):
        self.catalog = catalog
        self.schema = schema
        self.volume = volume
        self.base_path = f"/Volumes/{catalog}/{schema}/{volume}"

        # Use Databricks SDK for file operations
        from databricks.sdk import WorkspaceClient
        self.client = workspace_client or WorkspaceClient()

    def _full_path(self, path: str) -> str:
        return f"{self.base_path}/{path}"

    def write(self, path: str, data: bytes) -> None:
        """Write to UC Volume via Files API."""
        full_path = self._full_path(path)

        # Ensure parent directories exist (UC Volumes auto-create)
        self.client.files.upload(full_path, data, overwrite=True)

    def read(self, path: str) -> Optional[bytes]:
        """Read from UC Volume."""
        full_path = self._full_path(path)
        try:
            response = self.client.files.download(full_path)
            return response.contents.read()
        except Exception:
            return None

    def exists(self, path: str) -> bool:
        """Check if file exists in UC Volume."""
        full_path = self._full_path(path)
        try:
            self.client.files.get_status(full_path)
            return True
        except Exception:
            return False

    def list(self, prefix: str) -> list[str]:
        """List files under prefix in UC Volume."""
        full_path = self._full_path(prefix)
        try:
            entries = self.client.files.list_directory_contents(full_path)
            return [e.path.replace(self.base_path + "/", "") for e in entries if not e.is_directory]
        except Exception:
            return []

    def delete(self, path: str) -> None:
        """Delete file from UC Volume."""
        full_path = self._full_path(path)
        try:
            self.client.files.delete(full_path)
        except Exception:
            pass


def get_storage_backend(config: "EvalAgentConfig") -> StorageBackend:
    """Factory function to get appropriate storage backend."""
    if config.storage_backend == "uc_volumes":
        return UCVolumeStorage(
            catalog=config.uc_catalog,
            schema=config.uc_schema,
            volume=config.uc_volume,
        )
    else:
        return LocalFileStorage(Path(config.workspace_path or ".claude/workspace"))
```

### Configuration for Storage Backend

```python
# src/config.py additions

@dataclass
class EvalAgentConfig:
    # ... existing fields ...

    # Storage backend configuration
    storage_backend: str = "local"  # "local" or "uc_volumes"
    workspace_path: str = ".claude/workspace"  # For local storage

    # Unity Catalog Volumes configuration (for Databricks)
    uc_catalog: str = ""
    uc_schema: str = ""
    uc_volume: str = "mlflow_eval_agent"
```

### Environment Variables

```bash
# Local development
STORAGE_BACKEND=local
WORKSPACE_PATH=.claude/workspace

# Databricks deployment
STORAGE_BACKEND=uc_volumes
UC_CATALOG=main
UC_SCHEMA=mlflow_eval_agent
UC_VOLUME=workspace
```

---

## Updated Workspace with Storage Backend

```python
# src/persistence/workspace.py

import json
from datetime import datetime
from typing import Any, Optional

from .storage import StorageBackend, get_storage_backend


class FileSystemWorkspace:
    """Workspace backed by pluggable storage (local or UC Volumes)."""

    def __init__(self, storage: StorageBackend):
        self.storage = storage
        self._cache: dict[str, Any] = {}
        self._manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        data = self.storage.read("_manifest.json")
        if data:
            return json.loads(data.decode())
        return {"entries": {}, "created_at": datetime.now().isoformat()}

    def _save_manifest(self):
        self.storage.write(
            "_manifest.json",
            json.dumps(self._manifest, indent=2).encode()
        )

    def write(self, key: str, data: Any, agent: str = "unknown") -> tuple[bool, str]:
        """Write entry to storage with versioning."""
        # Validate
        is_valid, msg = validate_workspace_entry(key, data)
        if not is_valid:
            return False, msg

        # Archive old version
        if key in self._manifest["entries"]:
            self._archive_entry(key)

        # Write new entry
        entry = {
            "data": data,
            "written_by": agent,
            "timestamp": datetime.now().isoformat(),
            "version": self._manifest["entries"].get(key, {}).get("version", 0) + 1,
        }

        self.storage.write(f"{key}.json", json.dumps(entry, indent=2).encode())

        # Update manifest
        self._manifest["entries"][key] = {
            "version": entry["version"],
            "written_by": agent,
            "timestamp": entry["timestamp"],
        }
        self._save_manifest()

        # Update cache
        self._cache[key] = entry

        return True, f"Wrote {key} v{entry['version']}"

    def read(self, key: str, use_cache: bool = True) -> Optional[Any]:
        """Read entry from storage."""
        if use_cache and key in self._cache:
            return self._cache[key]["data"]

        data = self.storage.read(f"{key}.json")
        if data is None:
            return None

        entry = json.loads(data.decode())
        self._cache[key] = entry
        return entry["data"]

    def _archive_entry(self, key: str):
        """Archive current entry to history."""
        data = self.storage.read(f"{key}.json")
        if data is None:
            return

        entry = json.loads(data.decode())
        version = entry.get("version", 1)
        timestamp = int(datetime.now().timestamp())

        self.storage.write(
            f"_history/{key}/v{version}_{timestamp}.json",
            data
        )
```

---

## Summary: Recommended Improvements

| Area | Current | Proposed |
|------|---------|----------|
| **Persistence** | In-memory only | File-backed with storage abstraction |
| **Context Injection** | All keys to all agents | Selective per-agent dependencies |
| **Context Loading** | Static at creation | Dynamic via tools |
| **Versioning** | None (overwrites) | Append-only with history |
| **Tool Calls** | Not persisted | Logged to filesystem |
| **Session State** | Lost on restart | Resumable with snapshots |
| **Storage Backend** | N/A | Local filesystem or UC Volumes |
| **Token Budget** | Global truncation | Per-agent, per-key limits |

---

## Migration Path

### Phase 1: Storage Abstraction
1. Create `src/persistence/storage.py` with `StorageBackend` interface
2. Implement `LocalFileStorage` and `UCVolumeStorage`
3. Update `EvalAgentConfig` with storage settings
4. Keep existing `SharedWorkspace` working (backwards compatible)

### Phase 2: File-Backed Workspace
1. Create `FileSystemWorkspace` using storage backend
2. Add versioning and history
3. Add manifest for indexing
4. Make workspace swappable (in-memory vs file-backed)

### Phase 3: Context Engineering
1. Add selective context injection
2. Add per-agent dependency declarations
3. Add dynamic context tools
4. Add token budget management

### Phase 4: Tool Call Logging
1. Create `ToolCallLogger` class
2. Wrap existing tools to log calls
3. Add `get_recent_tool_calls` tool for learning

### Phase 5: Session Management
1. Create `SessionManager` for resumable sessions
2. Save workspace snapshots at session boundaries
3. Track agent IDs for sub-agent resumption
