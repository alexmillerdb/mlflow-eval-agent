# Selective Context Injection & Session Management

Deep dive into context engineering and resumable sessions for the MLflow Eval Agent.

---

## Part 1: Selective Context Injection

### The Problem

Current implementation dumps ALL workspace data into EVERY agent's prompt:

```python
# src/subagents/__init__.py - Current behavior
def create_subagents(workspace: SharedWorkspace) -> dict[str, AgentDefinition]:
    workspace_context = workspace.to_context_string()  # ALL keys, ALL agents
    return {
        "trace_analyst": create_trace_analyst(workspace_context),
        "context_engineer": create_context_engineer(workspace_context),
        "agent_architect": create_agent_architect(workspace_context),
    }
```

**Context waste example:**
- `trace_analyst` gets `context_recommendations` (wrote by context_engineer) - irrelevant
- `context_engineer` gets its own output back - wasteful
- Every agent pays token cost for data they don't need

### The Solution: Dependency Declarations

Each agent declares what it needs and what it produces:

```python
# src/subagents/dependencies.py

from dataclasses import dataclass
from typing import Optional


@dataclass
class AgentContextConfig:
    """Context configuration for a sub-agent."""

    # Keys this agent MUST have before running
    required_keys: list[str]

    # Keys this agent CAN use if available
    optional_keys: list[str]

    # Keys this agent WRITES (excluded from its own context)
    output_keys: list[str]

    # Token budget for context injection
    total_token_budget: int = 2000

    # Per-key token limits (chars ≈ tokens * 4)
    key_token_limits: dict[str, int] = None

    def __post_init__(self):
        if self.key_token_limits is None:
            self.key_token_limits = {}


# Agent dependency declarations
AGENT_CONTEXT_CONFIGS = {
    "trace_analyst": AgentContextConfig(
        required_keys=[],  # First in pipeline, no dependencies
        optional_keys=[],
        output_keys=[
            "trace_analysis_summary",
            "error_patterns",
            "performance_metrics",
            "extracted_eval_cases"
        ],
        total_token_budget=1000,  # Minimal context needed
    ),

    "context_engineer": AgentContextConfig(
        required_keys=["trace_analysis_summary", "error_patterns"],
        optional_keys=["performance_metrics"],
        output_keys=["context_recommendations"],
        total_token_budget=3000,  # Needs more context for analysis
        key_token_limits={
            "trace_analysis_summary": 1200,
            "error_patterns": 1200,
            "performance_metrics": 600,
        }
    ),

    "agent_architect": AgentContextConfig(
        required_keys=["trace_analysis_summary"],
        optional_keys=["performance_metrics", "context_recommendations"],
        output_keys=[],  # Writes tags to traces, not workspace
        total_token_budget=3500,
        key_token_limits={
            "trace_analysis_summary": 1000,
            "performance_metrics": 800,
            "context_recommendations": 1500,
        }
    ),
}
```

### Updated Workspace Methods

```python
# src/workspace.py additions

class SharedWorkspace:
    # ... existing code ...

    def to_selective_context(
        self,
        config: "AgentContextConfig",
        include_summary: bool = True
    ) -> str:
        """Generate context string with only relevant keys for an agent.

        Args:
            config: Agent's context configuration
            include_summary: Whether to include brief summary of excluded keys
        """
        parts = ["<workspace_context>"]

        # Track what's missing for dependency errors
        missing_required = []
        chars_used = 0
        max_chars = config.total_token_budget * 4  # Rough token-to-char ratio

        # Process required keys first
        for key in config.required_keys:
            entry = self._data.get(key)
            if not entry:
                missing_required.append(key)
                continue

            key_limit = config.key_token_limits.get(key, 500) * 4
            serialized = self._serialize_entry(key, entry, max_chars=key_limit)
            chars_used += len(serialized)
            parts.append(serialized)

        # Warn about missing required keys
        if missing_required:
            parts.append(f"""
<missing_dependencies>
The following REQUIRED keys are missing: {missing_required}
Run trace_analyst first to populate these entries.
</missing_dependencies>
""")

        # Process optional keys if budget allows
        for key in config.optional_keys:
            if chars_used >= max_chars:
                break

            entry = self._data.get(key)
            if not entry:
                continue

            remaining_budget = max_chars - chars_used
            key_limit = min(
                config.key_token_limits.get(key, 500) * 4,
                remaining_budget
            )
            serialized = self._serialize_entry(key, entry, max_chars=key_limit)
            chars_used += len(serialized)
            parts.append(serialized)

        # Optional: Show summary of what else exists (without full content)
        if include_summary:
            other_keys = [
                k for k in self._data.keys()
                if k not in config.required_keys
                and k not in config.optional_keys
                and k not in config.output_keys
            ]
            if other_keys:
                parts.append(f"""
<other_available_keys>
Additional workspace entries exist: {other_keys}
Use read_from_workspace tool if needed.
</other_available_keys>
""")

        parts.append("</workspace_context>")
        return "\n".join(parts)

    def _serialize_entry(
        self,
        key: str,
        entry: dict,
        max_chars: int = 2000
    ) -> str:
        """Serialize a single entry with truncation."""
        data = entry["data"]
        agent = entry.get("written_by", "unknown")

        serialized = json.dumps(data, indent=2, default=str)

        if len(serialized) > max_chars:
            # Smart truncation: try to preserve structure
            if isinstance(data, list) and len(data) > 3:
                # For lists, show first few items
                truncated_data = data[:3]
                serialized = json.dumps(truncated_data, indent=2, default=str)
                serialized += f"\n... and {len(data) - 3} more items"
            elif isinstance(data, dict) and len(serialized) > max_chars:
                serialized = serialized[:max_chars - 50] + "\n... [TRUNCATED]"
            else:
                serialized = serialized[:max_chars - 50] + "\n... [TRUNCATED]"

        return f"""
<workspace_entry key="{key}" written_by="{agent}">
{serialized}
</workspace_entry>
"""

    def check_agent_dependencies(
        self,
        config: "AgentContextConfig"
    ) -> tuple[bool, list[str]]:
        """Check if all required dependencies exist for an agent."""
        missing = [k for k in config.required_keys if k not in self._data]
        return len(missing) == 0, missing
```

### Updated Sub-Agent Creation

```python
# src/subagents/__init__.py - Updated

from .dependencies import AGENT_CONTEXT_CONFIGS


def create_subagents(workspace: SharedWorkspace) -> dict[str, AgentDefinition]:
    """Create sub-agents with selective context injection."""
    return {
        "trace_analyst": create_trace_analyst(
            workspace.to_selective_context(AGENT_CONTEXT_CONFIGS["trace_analyst"])
        ),
        "context_engineer": create_context_engineer(
            workspace.to_selective_context(AGENT_CONTEXT_CONFIGS["context_engineer"])
        ),
        "agent_architect": create_agent_architect(
            workspace.to_selective_context(AGENT_CONTEXT_CONFIGS["agent_architect"])
        ),
    }


def validate_agent_can_run(
    agent_name: str,
    workspace: SharedWorkspace
) -> tuple[bool, str]:
    """Check if an agent has all required dependencies.

    Use this before invoking a sub-agent to fail fast.
    """
    config = AGENT_CONTEXT_CONFIGS.get(agent_name)
    if not config:
        return True, "No config defined"

    can_run, missing = workspace.check_agent_dependencies(config)
    if can_run:
        return True, "All dependencies present"
    else:
        return False, f"Missing required workspace entries: {missing}"
```

### Coordinator Prompt Update

Add dependency awareness to coordinator:

```python
# src/subagents/prompts.py

AGENT_DEPENDENCY_GUIDE = """
## Agent Dependencies

Each sub-agent has specific workspace requirements:

| Agent | Requires | Produces |
|-------|----------|----------|
| trace_analyst | (none) | trace_analysis_summary, error_patterns, performance_metrics |
| context_engineer | trace_analysis_summary, error_patterns | context_recommendations |
| agent_architect | trace_analysis_summary | (tags traces directly) |

**Workflow Order**: trace_analyst → context_engineer → agent_architect

If you invoke an agent before its dependencies exist, it will fail.
Use check_workspace_dependencies tool to verify before invoking.
"""

COORDINATOR_SYSTEM_PROMPT = f"""
You are the MLflow Evaluation Agent Coordinator...

{AGENT_DEPENDENCY_GUIDE}

## Current Workspace State
{{workspace_context}}
...
"""
```

---

## Part 2: Session Management

### The Problem

Current implementation loses all state on restart:
- Workspace data is in-memory only
- No way to resume a partially completed analysis
- Sub-agent IDs not tracked for resumption
- No checkpoint/rollback capability

### Session Architecture

```
.claude/sessions/
├── current.json                    # Points to active session
├── <session-id>/
│   ├── metadata.json              # Session metadata
│   ├── workspace_snapshot.json    # Workspace state at session start
│   ├── checkpoints/               # Periodic snapshots
│   │   ├── cp_001_<timestamp>.json
│   │   └── cp_002_<timestamp>.json
│   ├── agent_states/              # Sub-agent resumption data
│   │   ├── trace_analyst.json
│   │   ├── context_engineer.json
│   │   └── agent_architect.json
│   └── transcript.jsonl           # Conversation history (optional)
```

### Core Session Manager

```python
# src/persistence/session.py

import json
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from enum import Enum


class SessionStatus(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SessionMetadata:
    """Metadata for a session."""
    session_id: str
    experiment_id: str
    status: SessionStatus
    created_at: str
    updated_at: str
    user_query: str  # Initial query that started session
    checkpoint_count: int = 0
    agents_invoked: list[str] = None

    def __post_init__(self):
        if self.agents_invoked is None:
            self.agents_invoked = []


@dataclass
class AgentState:
    """State for a sub-agent that can be resumed."""
    agent_name: str
    agent_id: str  # Claude SDK agent ID for resumption
    status: str  # "pending", "running", "completed", "failed"
    workspace_keys_written: list[str]
    started_at: str
    completed_at: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class Checkpoint:
    """A point-in-time snapshot of session state."""
    checkpoint_id: str
    created_at: str
    trigger: str  # "auto", "manual", "agent_complete"
    workspace_keys: list[str]
    agents_completed: list[str]


class SessionManager:
    """Manages session lifecycle and state persistence."""

    def __init__(self, base_path: Path = None, auto_checkpoint: bool = True):
        self.base_path = base_path or Path(".claude/sessions")
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.auto_checkpoint = auto_checkpoint
        self._current_session: Optional[SessionMetadata] = None

    # =========================================================================
    # Session Lifecycle
    # =========================================================================

    def create_session(
        self,
        experiment_id: str,
        user_query: str,
        workspace: "SharedWorkspace"
    ) -> str:
        """Create a new session and return its ID."""
        session_id = self._generate_session_id(experiment_id, user_query)
        session_dir = self.base_path / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Create metadata
        now = datetime.now().isoformat()
        metadata = SessionMetadata(
            session_id=session_id,
            experiment_id=experiment_id,
            status=SessionStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            user_query=user_query,
        )

        # Save metadata
        self._save_metadata(session_id, metadata)

        # Save initial workspace snapshot
        self._save_workspace_snapshot(session_id, workspace, "initial")

        # Update current session pointer
        self._set_current_session(session_id)

        self._current_session = metadata
        return session_id

    def resume_session(
        self,
        session_id: str,
        workspace: "SharedWorkspace"
    ) -> SessionMetadata:
        """Resume an existing session, restoring workspace state."""
        metadata = self._load_metadata(session_id)
        if not metadata:
            raise ValueError(f"Session not found: {session_id}")

        # Load latest checkpoint or initial snapshot
        self._restore_workspace(session_id, workspace)

        # Update session status
        metadata.status = SessionStatus.ACTIVE
        metadata.updated_at = datetime.now().isoformat()
        self._save_metadata(session_id, metadata)

        self._set_current_session(session_id)
        self._current_session = metadata
        return metadata

    def pause_session(self, session_id: str, workspace: "SharedWorkspace"):
        """Pause session with checkpoint for later resumption."""
        metadata = self._load_metadata(session_id)
        if not metadata:
            return

        # Create checkpoint
        self.create_checkpoint(session_id, workspace, trigger="pause")

        # Update status
        metadata.status = SessionStatus.PAUSED
        metadata.updated_at = datetime.now().isoformat()
        self._save_metadata(session_id, metadata)

    def complete_session(self, session_id: str, workspace: "SharedWorkspace"):
        """Mark session as completed with final snapshot."""
        metadata = self._load_metadata(session_id)
        if not metadata:
            return

        # Final checkpoint
        self.create_checkpoint(session_id, workspace, trigger="complete")

        # Update status
        metadata.status = SessionStatus.COMPLETED
        metadata.updated_at = datetime.now().isoformat()
        self._save_metadata(session_id, metadata)

    def get_current_session(self) -> Optional[str]:
        """Get the current active session ID."""
        current_path = self.base_path / "current.json"
        if current_path.exists():
            data = json.loads(current_path.read_text())
            return data.get("session_id")
        return None

    def list_sessions(
        self,
        experiment_id: str = None,
        status: SessionStatus = None,
        limit: int = 20
    ) -> list[SessionMetadata]:
        """List sessions with optional filters."""
        sessions = []
        for session_dir in self.base_path.iterdir():
            if not session_dir.is_dir() or session_dir.name.startswith("_"):
                continue

            metadata = self._load_metadata(session_dir.name)
            if not metadata:
                continue

            # Apply filters
            if experiment_id and metadata.experiment_id != experiment_id:
                continue
            if status and metadata.status != status:
                continue

            sessions.append(metadata)

        # Sort by updated_at descending
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions[:limit]

    # =========================================================================
    # Checkpoints
    # =========================================================================

    def create_checkpoint(
        self,
        session_id: str,
        workspace: "SharedWorkspace",
        trigger: str = "auto"
    ) -> str:
        """Create a checkpoint of current state."""
        metadata = self._load_metadata(session_id)
        if not metadata:
            raise ValueError(f"Session not found: {session_id}")

        checkpoint_id = f"cp_{metadata.checkpoint_count + 1:03d}_{int(datetime.now().timestamp())}"

        checkpoint_dir = self.base_path / session_id / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save workspace state
        workspace_data = {
            key: {
                "data": entry["data"],
                "written_by": entry.get("written_by", "unknown"),
                "timestamp": entry.get("timestamp", 0),
            }
            for key, entry in workspace._data.items()
        }

        # Save checkpoint
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            created_at=datetime.now().isoformat(),
            trigger=trigger,
            workspace_keys=list(workspace._data.keys()),
            agents_completed=metadata.agents_invoked.copy(),
        )

        checkpoint_path = checkpoint_dir / f"{checkpoint_id}.json"
        checkpoint_path.write_text(json.dumps({
            "metadata": asdict(checkpoint),
            "workspace": workspace_data,
        }, indent=2, default=str))

        # Update session metadata
        metadata.checkpoint_count += 1
        metadata.updated_at = datetime.now().isoformat()
        self._save_metadata(session_id, metadata)

        return checkpoint_id

    def restore_checkpoint(
        self,
        session_id: str,
        checkpoint_id: str,
        workspace: "SharedWorkspace"
    ):
        """Restore workspace to a specific checkpoint."""
        checkpoint_path = self.base_path / session_id / "checkpoints" / f"{checkpoint_id}.json"
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")

        data = json.loads(checkpoint_path.read_text())
        workspace_data = data.get("workspace", {})

        # Clear and restore workspace
        workspace.clear()
        for key, entry in workspace_data.items():
            workspace._data[key] = entry
            workspace._timestamps[key] = entry.get("timestamp", 0)

    def list_checkpoints(self, session_id: str) -> list[Checkpoint]:
        """List all checkpoints for a session."""
        checkpoint_dir = self.base_path / session_id / "checkpoints"
        if not checkpoint_dir.exists():
            return []

        checkpoints = []
        for cp_file in sorted(checkpoint_dir.glob("cp_*.json")):
            data = json.loads(cp_file.read_text())
            metadata = data.get("metadata", {})
            checkpoints.append(Checkpoint(**metadata))

        return checkpoints

    # =========================================================================
    # Agent State Management
    # =========================================================================

    def save_agent_state(
        self,
        session_id: str,
        agent_name: str,
        agent_id: str,
        status: str,
        workspace_keys_written: list[str] = None
    ):
        """Save agent state for potential resumption."""
        agent_dir = self.base_path / session_id / "agent_states"
        agent_dir.mkdir(parents=True, exist_ok=True)

        # Load existing state if any
        state_path = agent_dir / f"{agent_name}.json"
        existing = None
        if state_path.exists():
            existing = json.loads(state_path.read_text())

        state = AgentState(
            agent_name=agent_name,
            agent_id=agent_id,
            status=status,
            workspace_keys_written=workspace_keys_written or [],
            started_at=existing.get("started_at") if existing else datetime.now().isoformat(),
            completed_at=datetime.now().isoformat() if status in ["completed", "failed"] else None,
        )

        state_path.write_text(json.dumps(asdict(state), indent=2))

        # Update session metadata
        metadata = self._load_metadata(session_id)
        if metadata and agent_name not in metadata.agents_invoked:
            metadata.agents_invoked.append(agent_name)
            self._save_metadata(session_id, metadata)

        # Auto-checkpoint on agent completion
        if self.auto_checkpoint and status == "completed":
            # Need workspace reference - this should be passed in
            pass

    def get_agent_state(
        self,
        session_id: str,
        agent_name: str
    ) -> Optional[AgentState]:
        """Get saved state for an agent."""
        state_path = self.base_path / session_id / "agent_states" / f"{agent_name}.json"
        if state_path.exists():
            data = json.loads(state_path.read_text())
            return AgentState(**data)
        return None

    def get_resumable_agent_id(
        self,
        session_id: str,
        agent_name: str
    ) -> Optional[str]:
        """Get agent ID for resumption if agent was previously invoked."""
        state = self.get_agent_state(session_id, agent_name)
        if state and state.status in ["running", "completed"]:
            return state.agent_id
        return None

    # =========================================================================
    # Private Helpers
    # =========================================================================

    def _generate_session_id(self, experiment_id: str, query: str) -> str:
        """Generate a unique session ID."""
        content = f"{experiment_id}:{query}:{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _set_current_session(self, session_id: str):
        """Update current session pointer."""
        current_path = self.base_path / "current.json"
        current_path.write_text(json.dumps({
            "session_id": session_id,
            "set_at": datetime.now().isoformat()
        }))

    def _save_metadata(self, session_id: str, metadata: SessionMetadata):
        """Save session metadata."""
        metadata_path = self.base_path / session_id / "metadata.json"
        # Convert enum to string for JSON serialization
        data = asdict(metadata)
        data["status"] = metadata.status.value
        metadata_path.write_text(json.dumps(data, indent=2))

    def _load_metadata(self, session_id: str) -> Optional[SessionMetadata]:
        """Load session metadata."""
        metadata_path = self.base_path / session_id / "metadata.json"
        if not metadata_path.exists():
            return None

        data = json.loads(metadata_path.read_text())
        data["status"] = SessionStatus(data["status"])
        return SessionMetadata(**data)

    def _save_workspace_snapshot(
        self,
        session_id: str,
        workspace: "SharedWorkspace",
        name: str = "snapshot"
    ):
        """Save workspace state snapshot."""
        snapshot_path = self.base_path / session_id / f"workspace_{name}.json"
        workspace_data = {
            key: entry for key, entry in workspace._data.items()
        }
        snapshot_path.write_text(json.dumps(workspace_data, indent=2, default=str))

    def _restore_workspace(self, session_id: str, workspace: "SharedWorkspace"):
        """Restore workspace from latest checkpoint or initial snapshot."""
        # Try latest checkpoint first
        checkpoints = self.list_checkpoints(session_id)
        if checkpoints:
            latest = checkpoints[-1]
            self.restore_checkpoint(session_id, latest.checkpoint_id, workspace)
            return

        # Fall back to initial snapshot
        snapshot_path = self.base_path / session_id / "workspace_initial.json"
        if snapshot_path.exists():
            data = json.loads(snapshot_path.read_text())
            workspace.clear()
            for key, entry in data.items():
                workspace._data[key] = entry
                workspace._timestamps[key] = entry.get("timestamp", 0)
```

### Integration with MLflowEvalAgent

```python
# src/agent.py updates

class MLflowEvalAgent:
    def __init__(self, config: Optional[EvalAgentConfig] = None):
        self.config = config or EvalAgentConfig.from_env()
        self.workspace = SharedWorkspace(...)

        # Session management
        self.session_manager = SessionManager(
            base_path=Path(self.config.sessions_path or ".claude/sessions"),
            auto_checkpoint=self.config.auto_checkpoint
        )
        self._current_session_id: Optional[str] = None

    async def query(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        resume: bool = False
    ) -> AsyncIterator[EvalAgentResult]:
        """Query with session management."""

        # Handle session
        if resume and session_id:
            # Resume existing session
            metadata = self.session_manager.resume_session(session_id, self.workspace)
            self._current_session_id = session_id
        elif session_id:
            # Continue existing session without full restore
            self._current_session_id = session_id
        else:
            # Create new session
            self._current_session_id = self.session_manager.create_session(
                experiment_id=self.config.experiment_id,
                user_query=prompt,
                workspace=self.workspace
            )

        # Build options with session awareness
        options = self._build_options(session_id=self._current_session_id)

        async with ClaudeSDKClient(options=options) as client:
            # ... existing query logic ...

            async for message in client.receive_response():
                # Track agent invocations
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, ToolUseBlock) and block.name == "Task":
                            agent_name = block.input.get("subagent_type")
                            self.session_manager.save_agent_state(
                                self._current_session_id,
                                agent_name,
                                agent_id=block.id,  # Will be updated when result comes back
                                status="running"
                            )

                # ... rest of message handling ...

                elif isinstance(message, ResultMessage):
                    # Auto-checkpoint on completion
                    self.session_manager.create_checkpoint(
                        self._current_session_id,
                        self.workspace,
                        trigger="query_complete"
                    )

    async def resume_agent(
        self,
        agent_name: str,
        additional_prompt: str = ""
    ) -> AsyncIterator[EvalAgentResult]:
        """Resume a specific sub-agent from its last state."""
        if not self._current_session_id:
            raise ValueError("No active session")

        agent_id = self.session_manager.get_resumable_agent_id(
            self._current_session_id,
            agent_name
        )

        if not agent_id:
            raise ValueError(f"No resumable state for agent: {agent_name}")

        # Resume agent via Claude SDK
        options = self._build_options(session_id=self._current_session_id)
        options.agents[agent_name].resume = agent_id  # SDK support for agent resumption

        # ... invoke agent ...

    def list_sessions(self, **kwargs) -> list:
        """List available sessions."""
        return self.session_manager.list_sessions(**kwargs)

    def get_session_checkpoints(self, session_id: str) -> list:
        """Get checkpoints for a session."""
        return self.session_manager.list_checkpoints(session_id)

    def restore_to_checkpoint(self, session_id: str, checkpoint_id: str):
        """Restore workspace to a specific checkpoint."""
        self.session_manager.restore_checkpoint(session_id, checkpoint_id, self.workspace)
```

### CLI Commands for Session Management

```python
# src/cli.py additions

@click.group()
def sessions():
    """Session management commands."""
    pass


@sessions.command("list")
@click.option("--experiment-id", "-e", help="Filter by experiment")
@click.option("--status", "-s", type=click.Choice(["active", "paused", "completed"]))
@click.option("--limit", "-n", default=10, help="Max sessions to show")
def list_sessions(experiment_id, status, limit):
    """List available sessions."""
    agent = MLflowEvalAgent()
    status_enum = SessionStatus(status) if status else None
    sessions = agent.list_sessions(
        experiment_id=experiment_id,
        status=status_enum,
        limit=limit
    )

    for s in sessions:
        click.echo(f"{s.session_id} | {s.status.value:10} | {s.updated_at} | {s.user_query[:50]}...")


@sessions.command("resume")
@click.argument("session_id")
def resume_session(session_id):
    """Resume an existing session."""
    agent = MLflowEvalAgent()

    async def run():
        async for result in agent.query("", session_id=session_id, resume=True):
            # Handle resume flow...
            pass

    asyncio.run(run())


@sessions.command("checkpoints")
@click.argument("session_id")
def show_checkpoints(session_id):
    """Show checkpoints for a session."""
    agent = MLflowEvalAgent()
    checkpoints = agent.get_session_checkpoints(session_id)

    for cp in checkpoints:
        click.echo(f"{cp.checkpoint_id} | {cp.trigger:10} | {cp.created_at} | keys: {cp.workspace_keys}")


@sessions.command("restore")
@click.argument("session_id")
@click.argument("checkpoint_id")
def restore_checkpoint(session_id, checkpoint_id):
    """Restore to a specific checkpoint."""
    agent = MLflowEvalAgent()
    agent.restore_to_checkpoint(session_id, checkpoint_id)
    click.echo(f"Restored to checkpoint: {checkpoint_id}")
```

---

## Summary: Implementation Priority

### Phase 1: Selective Context (High Impact, Medium Effort)

1. Create `src/subagents/dependencies.py` with `AgentContextConfig`
2. Add `to_selective_context()` to `SharedWorkspace`
3. Update `create_subagents()` to use selective context
4. Update coordinator prompt with dependency guide
5. **Result**: Immediate token savings, better agent focus

### Phase 2: Session Management (Medium Impact, Higher Effort)

1. Create `src/persistence/session.py` with `SessionManager`
2. Add session integration to `MLflowEvalAgent.query()`
3. Add auto-checkpointing on agent completion
4. Add CLI commands for session management
5. **Result**: Resumable workflows, audit trail

### Phase 3: Agent Resumption (Lower Impact, Requires SDK Support)

1. Track agent IDs in session state
2. Add `resume_agent()` method
3. Test with Claude SDK agent resumption
4. **Result**: Can continue interrupted agent work

---

## Token Savings Estimate

| Scenario | Current | With Selective Context |
|----------|---------|------------------------|
| trace_analyst context | ~4000 chars | ~500 chars |
| context_engineer context | ~4000 chars | ~2500 chars |
| agent_architect context | ~4000 chars | ~3000 chars |
| **Total per workflow** | ~12000 chars | ~6000 chars |
| **Savings** | - | **~50%** |

At ~4 chars/token and $15/M tokens (Sonnet), a workflow saving 6000 chars saves ~1500 tokens = $0.02 per workflow. At scale (1000 workflows/day), that's $20/day or $600/month.
