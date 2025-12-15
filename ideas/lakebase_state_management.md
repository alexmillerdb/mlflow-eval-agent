# Databricks Lakebase State Management for MLflow Eval Agent

## Overview

Add PostgreSQL persistence via Databricks Lakebase to enable:
- **Full persistence**: workspace entries, tool calls, todos, sub-agent results
- **Context engineering**: pull relevant context from past sessions into prompts
- **Session continuity**: resume conversations across restarts

**Framework**: Stay with Claude Agent SDK (no LangGraph migration)

---

## PostgreSQL Schema

### Core Tables

```sql
-- Sessions: conversation threads
CREATE TABLE sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id VARCHAR(255),
    user_id VARCHAR(255),
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ended_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50) DEFAULT 'active',  -- active, completed, abandoned
    metadata JSONB DEFAULT '{}'::jsonb,

    INDEX idx_sessions_experiment (experiment_id),
    INDEX idx_sessions_user (user_id),
    INDEX idx_sessions_started_at (started_at DESC)
);

-- Workspace entries: persisted SharedWorkspace data
CREATE TABLE workspace_entries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(session_id) ON DELETE CASCADE,
    key VARCHAR(255) NOT NULL,
    data JSONB NOT NULL,
    written_by VARCHAR(100) NOT NULL,  -- agent name
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    version INTEGER DEFAULT 1,
    is_current BOOLEAN DEFAULT TRUE,

    INDEX idx_workspace_session (session_id),
    INDEX idx_workspace_key (key),
    INDEX idx_workspace_agent (written_by),
    INDEX idx_workspace_current (session_id, key, is_current) WHERE is_current = TRUE,
    INDEX idx_workspace_data_gin ON workspace_entries USING GIN (data)
);

-- Tool calls: all tool invocations with inputs/outputs
CREATE TABLE tool_calls (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(session_id) ON DELETE CASCADE,
    tool_use_id VARCHAR(255),  -- Claude's tool_use_id
    tool_name VARCHAR(255) NOT NULL,
    tool_input JSONB NOT NULL,
    tool_output JSONB,
    is_error BOOLEAN DEFAULT FALSE,
    error_message TEXT,
    agent_name VARCHAR(100),  -- which agent called it
    execution_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    INDEX idx_tool_calls_session (session_id),
    INDEX idx_tool_calls_tool_name (tool_name),
    INDEX idx_tool_calls_agent (agent_name),
    INDEX idx_tool_calls_created_at (created_at DESC),
    INDEX idx_tool_calls_input_gin ON tool_calls USING GIN (tool_input)
);

-- Todos: persistent task tracking
CREATE TABLE todos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(session_id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    active_form TEXT NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',  -- pending, in_progress, completed
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    sequence_order INTEGER,

    INDEX idx_todos_session (session_id),
    INDEX idx_todos_status (status),
    INDEX idx_todos_session_status (session_id, status)
);

-- Sub-agent results: stores sub-agent invocations and outcomes
CREATE TABLE subagent_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(session_id) ON DELETE CASCADE,
    agent_name VARCHAR(100) NOT NULL,
    task_description TEXT,
    result_summary TEXT,
    result_data JSONB,
    workspace_keys_written TEXT[],  -- which workspace keys were populated
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    execution_time_ms INTEGER,
    success BOOLEAN,
    error_message TEXT,

    INDEX idx_subagent_session (session_id),
    INDEX idx_subagent_agent (agent_name),
    INDEX idx_subagent_created (started_at DESC)
);

-- Context embeddings: for semantic search over past interactions (optional, requires pgvector)
CREATE TABLE context_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(session_id) ON DELETE CASCADE,
    source_type VARCHAR(50) NOT NULL,  -- workspace_entry, tool_call, subagent_result
    source_id UUID NOT NULL,
    content_text TEXT NOT NULL,
    embedding vector(1536),  -- OpenAI/Anthropic embedding dimension
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    INDEX idx_embeddings_session (session_id),
    INDEX idx_embeddings_source (source_type, source_id),
    INDEX idx_embeddings_vector ON context_embeddings USING ivfflat (embedding vector_cosine_ops)
);

-- Query history: tracks user queries for context
CREATE TABLE query_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(session_id) ON DELETE CASCADE,
    query_text TEXT NOT NULL,
    response_summary TEXT,
    cost_usd DECIMAL(10, 6),
    tokens_used INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    INDEX idx_query_session (session_id),
    INDEX idx_query_created (created_at DESC)
);
```

### Views for Context Retrieval

```sql
-- Recent workspace state across sessions
CREATE VIEW recent_workspace_summary AS
SELECT
    we.key,
    we.data,
    we.written_by,
    we.created_at,
    s.experiment_id,
    s.session_id
FROM workspace_entries we
JOIN sessions s ON we.session_id = s.session_id
WHERE we.is_current = TRUE
  AND we.created_at > NOW() - INTERVAL '7 days'
ORDER BY we.created_at DESC;

-- Tool call patterns for learning
CREATE VIEW tool_call_patterns AS
SELECT
    tool_name,
    agent_name,
    COUNT(*) as call_count,
    AVG(execution_time_ms) as avg_execution_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY execution_time_ms) as p95_execution_ms,
    SUM(CASE WHEN is_error THEN 1 ELSE 0 END)::float / COUNT(*) as error_rate
FROM tool_calls
WHERE created_at > NOW() - INTERVAL '30 days'
GROUP BY tool_name, agent_name;
```

---

## New Modules

```
src/persistence/
    __init__.py           # Public exports
    connection.py         # Lakebase OAuth connection pool
    repository.py         # CRUD operations for each table
    context_retriever.py  # Context retrieval strategies
    migrations.py         # Schema migration utilities
```

### `connection.py` - Lakebase OAuth Connection Pool

```python
"""Databricks Lakebase connection with OAuth credential rotation."""

from databricks.sdk import WorkspaceClient
from psycopg_pool import ConnectionPool
import psycopg
import uuid
from typing import Optional


class LakebaseCredentialConnection(psycopg.Connection):
    """Connection class that auto-injects OAuth tokens from Databricks."""

    _workspace_client: Optional[WorkspaceClient] = None
    _instance_name: Optional[str] = None

    @classmethod
    def configure(cls, workspace_client: WorkspaceClient, instance_name: str):
        """Configure the connection class with Databricks credentials."""
        cls._workspace_client = workspace_client
        cls._instance_name = instance_name

    @classmethod
    def connect(cls, conninfo='', **kwargs):
        """Connect with auto-injected OAuth token."""
        if cls._workspace_client is None:
            raise RuntimeError("LakebaseCredentialConnection not configured. Call configure() first.")

        # Generate fresh OAuth token (cached for ~50 minutes by SDK)
        cred = cls._workspace_client.database.generate_database_credential(
            request_id=str(uuid.uuid4()),
            instance_names=[cls._instance_name]
        )
        kwargs['password'] = cred.token
        return super().connect(conninfo, **kwargs)


def create_pool(
    host: str,
    database: str,
    username: str,
    min_size: int = 1,
    max_size: int = 10,
) -> ConnectionPool:
    """Create a connection pool with OAuth credential rotation."""
    conninfo = f"dbname={database} user={username} host={host} sslmode=require"

    return ConnectionPool(
        conninfo=conninfo,
        connection_class=LakebaseCredentialConnection,
        min_size=min_size,
        max_size=max_size,
        kwargs={"autocommit": True, "row_factory": psycopg.rows.dict_row}
    )
```

### `repository.py` - Data Access Layer

```python
"""Repository classes for persistence operations."""

from dataclasses import dataclass
from typing import Any, Optional
from uuid import UUID
from psycopg_pool import ConnectionPool
import json


@dataclass
class SessionRepository:
    pool: ConnectionPool

    async def create(
        self,
        experiment_id: str = None,
        user_id: str = None,
        metadata: dict = None
    ) -> UUID:
        """Create a new session and return its ID."""
        async with self.pool.connection() as conn:
            result = await conn.execute(
                """
                INSERT INTO sessions (experiment_id, user_id, metadata)
                VALUES ($1, $2, $3)
                RETURNING session_id
                """,
                [experiment_id, user_id, json.dumps(metadata or {})]
            )
            row = await result.fetchone()
            return row["session_id"]

    async def get(self, session_id: UUID) -> Optional[dict]:
        """Get session by ID."""
        async with self.pool.connection() as conn:
            result = await conn.execute(
                "SELECT * FROM sessions WHERE session_id = $1",
                [session_id]
            )
            return await result.fetchone()

    async def end(self, session_id: UUID, status: str = "completed"):
        """Mark session as ended."""
        async with self.pool.connection() as conn:
            await conn.execute(
                """
                UPDATE sessions
                SET ended_at = NOW(), status = $2
                WHERE session_id = $1
                """,
                [session_id, status]
            )


@dataclass
class WorkspaceRepository:
    pool: ConnectionPool

    async def write(
        self,
        session_id: UUID,
        key: str,
        data: dict,
        agent: str
    ) -> UUID:
        """Write workspace entry, creating new version if key exists."""
        async with self.pool.connection() as conn:
            # Mark existing entries as not current
            await conn.execute(
                """
                UPDATE workspace_entries
                SET is_current = FALSE, updated_at = NOW()
                WHERE session_id = $1 AND key = $2 AND is_current = TRUE
                """,
                [session_id, key]
            )

            # Get next version number
            result = await conn.execute(
                """
                SELECT COALESCE(MAX(version), 0) + 1 as next_version
                FROM workspace_entries
                WHERE session_id = $1 AND key = $2
                """,
                [session_id, key]
            )
            row = await result.fetchone()
            version = row["next_version"]

            # Insert new version
            result = await conn.execute(
                """
                INSERT INTO workspace_entries
                    (session_id, key, data, written_by, version, is_current)
                VALUES ($1, $2, $3, $4, $5, TRUE)
                RETURNING id
                """,
                [session_id, key, json.dumps(data), agent, version]
            )
            row = await result.fetchone()
            return row["id"]

    async def read(self, session_id: UUID, key: str) -> Optional[dict]:
        """Read current workspace entry by key."""
        async with self.pool.connection() as conn:
            result = await conn.execute(
                """
                SELECT data, written_by, created_at, version
                FROM workspace_entries
                WHERE session_id = $1 AND key = $2 AND is_current = TRUE
                """,
                [session_id, key]
            )
            row = await result.fetchone()
            if row:
                return {
                    "data": row["data"],
                    "written_by": row["written_by"],
                    "timestamp": row["created_at"].timestamp(),
                    "version": row["version"]
                }
            return None

    async def read_current(self, session_id: UUID) -> dict[str, dict]:
        """Read all current workspace entries for a session."""
        async with self.pool.connection() as conn:
            result = await conn.execute(
                """
                SELECT key, data, written_by, created_at, version
                FROM workspace_entries
                WHERE session_id = $1 AND is_current = TRUE
                """,
                [session_id]
            )
            rows = await result.fetchall()
            return {
                row["key"]: {
                    "data": row["data"],
                    "written_by": row["written_by"],
                    "timestamp": row["created_at"].timestamp(),
                    "version": row["version"]
                }
                for row in rows
            }

    async def search_across_sessions(
        self,
        key: str,
        experiment_id: str = None,
        limit: int = 5
    ) -> list[dict]:
        """Search workspace entries across sessions."""
        async with self.pool.connection() as conn:
            query = """
                SELECT we.key, we.data, we.written_by, we.created_at,
                       s.session_id, s.experiment_id
                FROM workspace_entries we
                JOIN sessions s ON we.session_id = s.session_id
                WHERE we.key = $1 AND we.is_current = TRUE
            """
            params = [key]

            if experiment_id:
                query += " AND s.experiment_id = $2"
                params.append(experiment_id)

            query += " ORDER BY we.created_at DESC LIMIT $" + str(len(params) + 1)
            params.append(limit)

            result = await conn.execute(query, params)
            return await result.fetchall()


@dataclass
class ToolCallRepository:
    pool: ConnectionPool

    async def create(
        self,
        session_id: UUID,
        tool_name: str,
        tool_input: dict,
        tool_output: dict = None,
        is_error: bool = False,
        error_message: str = None,
        agent_name: str = None,
        execution_time_ms: int = None,
        tool_use_id: str = None
    ) -> UUID:
        """Record a tool call."""
        async with self.pool.connection() as conn:
            result = await conn.execute(
                """
                INSERT INTO tool_calls
                    (session_id, tool_name, tool_input, tool_output,
                     is_error, error_message, agent_name, execution_time_ms, tool_use_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING id
                """,
                [
                    session_id, tool_name, json.dumps(tool_input),
                    json.dumps(tool_output) if tool_output else None,
                    is_error, error_message, agent_name, execution_time_ms, tool_use_id
                ]
            )
            row = await result.fetchone()
            return row["id"]

    async def get_by_tool_name(
        self,
        tool_name: str,
        success_only: bool = True,
        limit: int = 10
    ) -> list[dict]:
        """Get tool calls by name for learning patterns."""
        async with self.pool.connection() as conn:
            query = """
                SELECT tool_input, tool_output, agent_name, execution_time_ms, created_at
                FROM tool_calls
                WHERE tool_name = $1
            """
            params = [tool_name]

            if success_only:
                query += " AND is_error = FALSE"

            query += " ORDER BY created_at DESC LIMIT $2"
            params.append(limit)

            result = await conn.execute(query, params)
            return await result.fetchall()


@dataclass
class TodoRepository:
    pool: ConnectionPool

    async def upsert_batch(
        self,
        session_id: UUID,
        todos: list[dict]
    ) -> None:
        """Update or insert todos for a session."""
        async with self.pool.connection() as conn:
            # Clear existing todos for this session
            await conn.execute(
                "DELETE FROM todos WHERE session_id = $1",
                [session_id]
            )

            # Insert new todos
            for i, todo in enumerate(todos):
                await conn.execute(
                    """
                    INSERT INTO todos
                        (session_id, content, active_form, status, sequence_order,
                         completed_at)
                    VALUES ($1, $2, $3, $4, $5,
                            CASE WHEN $4 = 'completed' THEN NOW() ELSE NULL END)
                    """,
                    [
                        session_id, todo["content"], todo["activeForm"],
                        todo["status"], i
                    ]
                )

    async def get_all(self, session_id: UUID) -> list[dict]:
        """Get all todos for a session."""
        async with self.pool.connection() as conn:
            result = await conn.execute(
                """
                SELECT content, active_form, status, created_at, completed_at
                FROM todos
                WHERE session_id = $1
                ORDER BY sequence_order
                """,
                [session_id]
            )
            return await result.fetchall()


@dataclass
class SubagentResultRepository:
    pool: ConnectionPool

    async def create(
        self,
        session_id: UUID,
        agent_name: str,
        task_description: str = None,
        result_summary: str = None,
        result_data: dict = None,
        workspace_keys_written: list[str] = None,
        success: bool = True,
        error_message: str = None,
        execution_time_ms: int = None
    ) -> UUID:
        """Record a sub-agent execution."""
        async with self.pool.connection() as conn:
            result = await conn.execute(
                """
                INSERT INTO subagent_results
                    (session_id, agent_name, task_description, result_summary,
                     result_data, workspace_keys_written, completed_at,
                     success, error_message, execution_time_ms)
                VALUES ($1, $2, $3, $4, $5, $6, NOW(), $7, $8, $9)
                RETURNING id
                """,
                [
                    session_id, agent_name, task_description, result_summary,
                    json.dumps(result_data) if result_data else None,
                    workspace_keys_written, success, error_message, execution_time_ms
                ]
            )
            row = await result.fetchone()
            return row["id"]

    async def get_recent(
        self,
        experiment_id: str = None,
        agent_name: str = None,
        limit: int = 10
    ) -> list[dict]:
        """Get recent sub-agent results."""
        async with self.pool.connection() as conn:
            query = """
                SELECT sr.*, s.experiment_id
                FROM subagent_results sr
                JOIN sessions s ON sr.session_id = s.session_id
                WHERE 1=1
            """
            params = []

            if experiment_id:
                params.append(experiment_id)
                query += f" AND s.experiment_id = ${len(params)}"

            if agent_name:
                params.append(agent_name)
                query += f" AND sr.agent_name = ${len(params)}"

            params.append(limit)
            query += f" ORDER BY sr.started_at DESC LIMIT ${len(params)}"

            result = await conn.execute(query, params)
            return await result.fetchall()
```

### `context_retriever.py` - Context Retrieval Strategies

```python
"""Context retrieval strategies for pulling relevant data into prompts."""

from dataclasses import dataclass
from typing import Optional
from uuid import UUID
import json


@dataclass
class ContextRetriever:
    """Retrieves relevant context from persistence for prompt injection."""

    repos: dict  # Contains all repository instances
    config: "EvalAgentConfig"

    async def get_relevant_context(
        self,
        query: str,
        session_id: UUID,
        strategies: list[str] = None,
        max_tokens: int = None
    ) -> str:
        """
        Retrieve relevant context from past sessions.

        Strategies:
        - recency: Recent workspace entries
        - key_match: Entries matching specific keys
        - tool_patterns: Successful tool call examples
        - semantic: Embedding similarity (requires pgvector)
        """
        strategies = strategies or ["recency", "key_match"]
        max_tokens = max_tokens or self.config.context_max_tokens

        context_parts = []

        if "recency" in strategies:
            recent = await self._get_recent_context()
            if recent:
                context_parts.append(f"## Recent Analysis Results\n{recent}")

        if "key_match" in strategies:
            # Get workspace keys likely relevant to the query
            relevant_keys = self._extract_relevant_keys(query)
            for key in relevant_keys:
                entries = await self.repos["workspace"].search_across_sessions(
                    key=key,
                    experiment_id=self.config.experiment_id,
                    limit=3
                )
                if entries:
                    formatted = self._format_workspace_entries(entries)
                    context_parts.append(f"## Past {key} findings\n{formatted}")

        if "tool_patterns" in strategies:
            # Get tool call patterns for commonly used tools
            patterns = await self._get_tool_patterns()
            if patterns:
                context_parts.append(f"## Tool Usage Patterns\n{patterns}")

        combined = "\n\n".join(context_parts)

        # Truncate to max tokens (rough estimate: 4 chars per token)
        max_chars = max_tokens * 4
        if len(combined) > max_chars:
            combined = combined[:max_chars] + "\n... (truncated)"

        return combined

    async def _get_recent_context(self) -> str:
        """Get recent workspace entries across sessions."""
        workspace_repo = self.repos["workspace"]

        recent_keys = [
            "trace_analysis_summary",
            "error_patterns",
            "performance_metrics",
            "context_recommendations"
        ]

        results = []
        for key in recent_keys:
            entries = await workspace_repo.search_across_sessions(
                key=key,
                experiment_id=self.config.experiment_id,
                limit=2
            )
            if entries:
                results.append(self._format_workspace_entries(entries, key))

        return "\n".join(results) if results else ""

    async def _get_tool_patterns(self) -> str:
        """Get patterns from past tool calls."""
        tool_repo = self.repos["tool_call"]

        # Get patterns for commonly used MLflow tools
        tools = [
            "mcp__mlflow-mcp__search_traces",
            "mcp__mlflow-mcp__get_trace",
            "mcp__mlflow-mcp__log_feedback"
        ]

        patterns = []
        for tool in tools:
            calls = await tool_repo.get_by_tool_name(tool, success_only=True, limit=3)
            if calls:
                patterns.append(f"### {tool}\n{self._format_tool_calls(calls)}")

        return "\n".join(patterns) if patterns else ""

    def _extract_relevant_keys(self, query: str) -> list[str]:
        """Extract workspace keys likely relevant to the query."""
        query_lower = query.lower()

        key_triggers = {
            "trace_analysis_summary": ["trace", "analysis", "summary", "overview"],
            "error_patterns": ["error", "failure", "exception", "bug"],
            "performance_metrics": ["performance", "latency", "slow", "speed"],
            "context_recommendations": ["context", "prompt", "optimize", "improve"],
        }

        relevant = []
        for key, triggers in key_triggers.items():
            if any(t in query_lower for t in triggers):
                relevant.append(key)

        return relevant

    def _format_workspace_entries(
        self,
        entries: list[dict],
        key: str = None
    ) -> str:
        """Format workspace entries for context injection."""
        lines = []
        for entry in entries:
            timestamp = entry.get("created_at", "")
            agent = entry.get("written_by", "unknown")
            data = entry.get("data", {})

            # Truncate large data
            data_str = json.dumps(data, indent=2)
            if len(data_str) > 500:
                data_str = data_str[:500] + "\n... (truncated)"

            lines.append(f"- From {agent} at {timestamp}:\n```json\n{data_str}\n```")

        return "\n".join(lines)

    def _format_tool_calls(self, calls: list[dict]) -> str:
        """Format tool calls for context injection."""
        lines = []
        for call in calls[:3]:  # Limit to 3 examples
            input_str = json.dumps(call.get("tool_input", {}), indent=2)
            if len(input_str) > 200:
                input_str = input_str[:200] + "..."
            lines.append(f"Input: {input_str}")

        return "\n".join(lines)

    async def get_recent_workspace_state(
        self,
        experiment_id: str,
        max_age_days: int = None
    ) -> dict[str, dict]:
        """Get recent workspace state across all sessions for an experiment."""
        max_age = max_age_days or self.config.context_max_age_days

        # Query across all workspace keys
        all_entries = {}
        for key in ["trace_analysis_summary", "error_patterns", "performance_metrics",
                    "context_recommendations", "extracted_eval_cases"]:
            entries = await self.repos["workspace"].search_across_sessions(
                key=key,
                experiment_id=experiment_id,
                limit=5
            )
            if entries:
                all_entries[key] = entries

        return all_entries

    async def get_tool_call_examples(
        self,
        tool_name: str,
        success_only: bool = True,
        limit: int = 3
    ) -> list[dict]:
        """Find similar past tool calls for learning."""
        return await self.repos["tool_call"].get_by_tool_name(
            tool_name=tool_name,
            success_only=success_only,
            limit=limit
        )
```

---

## Files to Modify

### `src/config.py`

Add Lakebase configuration fields:

```python
@dataclass
class EvalAgentConfig:
    # ... existing fields ...

    # Lakebase persistence
    lakebase_enabled: bool = False
    lakebase_instance_name: str = ""
    lakebase_database: str = "databricks_postgres"
    lakebase_username: str = ""  # defaults to Databricks user
    lakebase_pool_min_size: int = 1
    lakebase_pool_max_size: int = 10

    # Context retrieval settings
    context_max_age_days: int = 7
    context_max_tokens: int = 2000
    context_embedding_enabled: bool = False  # requires pgvector setup

    @classmethod
    def from_env(cls, env_file: Optional[str] = None, validate: bool = True) -> "EvalAgentConfig":
        # Add to existing from_env:
        config = cls(
            # ... existing fields ...
            lakebase_enabled=os.getenv("LAKEBASE_ENABLED", "false").lower() == "true",
            lakebase_instance_name=os.getenv("LAKEBASE_INSTANCE_NAME", ""),
            lakebase_database=os.getenv("LAKEBASE_DATABASE", "databricks_postgres"),
            lakebase_username=os.getenv("LAKEBASE_USERNAME", ""),
            lakebase_pool_min_size=int(os.getenv("LAKEBASE_POOL_MIN_SIZE", "1")),
            lakebase_pool_max_size=int(os.getenv("LAKEBASE_POOL_MAX_SIZE", "10")),
            context_max_age_days=int(os.getenv("CONTEXT_MAX_AGE_DAYS", "7")),
            context_max_tokens=int(os.getenv("CONTEXT_MAX_TOKENS", "2000")),
            context_embedding_enabled=os.getenv("CONTEXT_EMBEDDING_ENABLED", "false").lower() == "true",
        )
```

### `src/workspace.py`

Add persistence layer to SharedWorkspace:

```python
class SharedWorkspace:
    """Shared workspace with optional Lakebase persistence."""

    def __init__(
        self,
        max_context_chars: int = 2000,
        persistence: Optional["WorkspaceRepository"] = None,
        session_id: Optional[UUID] = None
    ):
        self._data: dict[str, Any] = {}
        self._timestamps: dict[str, float] = {}
        self._max_context_chars = max_context_chars
        self._write_history: list[dict] = []

        # Persistence layer (optional)
        self._persistence = persistence
        self._session_id = session_id

    async def write_async(
        self,
        key: str,
        data: Any,
        agent: str = "unknown"
    ) -> tuple[bool, str]:
        """Async write with persistence."""
        # Validate
        is_valid, msg = validate_workspace_entry(key, data)
        if not is_valid:
            return False, msg

        # In-memory write
        self._data[key] = {
            "data": data,
            "written_by": agent,
            "timestamp": time.time(),
        }
        self._timestamps[key] = time.time()
        self._write_history.append({
            "key": key,
            "agent": agent,
            "timestamp": time.time()
        })

        # Persist if enabled
        if self._persistence and self._session_id:
            await self._persistence.write(self._session_id, key, data, agent)

        return True, f"Successfully wrote to workspace: {key}"

    async def load_from_persistence(self, session_id: UUID) -> None:
        """Load workspace state from persistence."""
        if not self._persistence:
            return

        self._session_id = session_id
        entries = await self._persistence.read_current(session_id)

        for key, entry in entries.items():
            self._data[key] = entry
            self._timestamps[key] = entry.get("timestamp", time.time())
```

### `src/agent.py`

Integrate persistence:

```python
class MLflowEvalAgent:
    def __init__(self, config: Optional[EvalAgentConfig] = None):
        self.config = config or EvalAgentConfig.from_env()

        # ... existing initialization ...

        # Initialize persistence if enabled
        self._pool: Optional[ConnectionPool] = None
        self._repos: Optional[dict] = None
        self._context_retriever: Optional[ContextRetriever] = None

        if self.config.lakebase_enabled:
            self._init_persistence()

    def _init_persistence(self):
        """Initialize Lakebase connection and repositories."""
        from databricks.sdk import WorkspaceClient
        from .persistence.connection import LakebaseCredentialConnection, create_pool
        from .persistence.repository import (
            SessionRepository, WorkspaceRepository,
            ToolCallRepository, TodoRepository, SubagentResultRepository
        )
        from .persistence.context_retriever import ContextRetriever

        w = WorkspaceClient()
        LakebaseCredentialConnection.configure(w, self.config.lakebase_instance_name)

        instance = w.database.get_database_instance(name=self.config.lakebase_instance_name)

        self._pool = create_pool(
            host=instance.read_write_dns,
            database=self.config.lakebase_database,
            username=self.config.lakebase_username or w.current_user.me().user_name,
            min_size=self.config.lakebase_pool_min_size,
            max_size=self.config.lakebase_pool_max_size,
        )

        self._repos = {
            "session": SessionRepository(self._pool),
            "workspace": WorkspaceRepository(self._pool),
            "tool_call": ToolCallRepository(self._pool),
            "todo": TodoRepository(self._pool),
            "subagent": SubagentResultRepository(self._pool),
        }

        self._context_retriever = ContextRetriever(self._repos, self.config)

    async def query(
        self,
        prompt: str,
        session_id: Optional[str] = None
    ) -> AsyncIterator[EvalAgentResult]:
        """Send query with persistence support."""
        self._start_time = time.time()

        # Create or resume session
        db_session_id = None
        if self.config.lakebase_enabled:
            if session_id:
                # Resume existing session
                db_session_id = UUID(session_id)
                await self.workspace.load_from_persistence(db_session_id)
            else:
                # Create new session
                db_session_id = await self._repos["session"].create(
                    experiment_id=self.config.experiment_id,
                    user_id=self._get_current_user()
                )
                self.workspace = SharedWorkspace(
                    max_context_chars=self.config.workspace_context_max_chars,
                    persistence=self._repos["workspace"],
                    session_id=db_session_id
                )

            # Retrieve relevant context from past sessions
            historical_context = await self._context_retriever.get_relevant_context(
                query=prompt,
                session_id=db_session_id,
                max_tokens=self.config.context_max_tokens
            )
            if historical_context:
                prompt = f"{prompt}\n\n<historical_context>\n{historical_context}\n</historical_context>"

        # ... rest of existing query implementation ...
```

### `src/subagents/prompts.py`

Add historical context placeholder:

```python
COORDINATOR_SYSTEM_PROMPT = """
You are the MLflow Evaluation Agent Coordinator...

## Shared Workspace
Sub-agents write findings to a shared workspace:
{workspace_context}

## Historical Context
Relevant findings from previous sessions (if available):
{historical_context}

## Output Requirements
...
"""
```

---

## Environment Variables

```bash
# .env additions

# Lakebase Persistence
LAKEBASE_ENABLED=true
LAKEBASE_INSTANCE_NAME=mlflow-eval-agent-db
LAKEBASE_DATABASE=databricks_postgres
LAKEBASE_USERNAME=  # defaults to Databricks user
LAKEBASE_POOL_MIN_SIZE=1
LAKEBASE_POOL_MAX_SIZE=10

# Context Retrieval
CONTEXT_MAX_AGE_DAYS=7
CONTEXT_MAX_TOKENS=2000
CONTEXT_EMBEDDING_ENABLED=false  # requires pgvector setup
```

---

## Dependencies

Add to `pyproject.toml`:

```toml
dependencies = [
    # ... existing ...
    "psycopg[binary,pool]>=3.1.0",
    "databricks-sdk>=0.56.0",
]

[project.optional-dependencies]
embeddings = [
    "pgvector>=0.2.0",
    "sentence-transformers>=2.2.0",
]
```

---

## Migration Path

### Phase 1: Infrastructure (Non-Breaking)
1. Create `src/persistence/` module with all classes
2. Add config fields with defaults that disable persistence
3. Add new dependencies to `pyproject.toml`
4. Create migration SQL files in `migrations/` directory
5. **Test:** Existing functionality unchanged when `LAKEBASE_ENABLED=false`

### Phase 2: Opt-In Persistence
1. Update `SharedWorkspace` to accept optional persistence backend
2. Update `MLflowEvalAgent.__init__` to initialize persistence when enabled
3. Add `write_async` method to workspace (existing `write` unchanged)
4. Update tools to optionally persist calls
5. **Test:** Persistence works when enabled, existing behavior unchanged when disabled

### Phase 3: Context Retrieval Integration
1. Add `ContextRetriever` class
2. Update `query()` to retrieve and inject historical context
3. Update system prompts to include historical context section
4. Add CLI flag `--no-history` to disable context retrieval
5. **Test:** Context retrieval improves agent responses

### Phase 4: Advanced Features (Optional)
1. Add embedding generation for semantic search
2. Add pgvector extension and similarity search
3. Add todo persistence with sync to TodoWrite tool results
4. Add session management CLI commands

---

## Context Retrieval Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Recency** | Recent workspace entries by key | "What did trace_analyst find recently?" |
| **Key Match** | Entries matching key pattern | "Get all error_patterns across sessions" |
| **Tool Patterns** | Successful tool call examples | Learning from past tool usage |
| **Semantic** | Embedding similarity (requires pgvector) | "Similar questions asked before" |

---

## Key Differences from LangGraph Approach

Since we're staying with Claude Agent SDK (not LangGraph), we can't use `PostgresSaver` directly:

| Aspect | LangGraph | Claude Agent SDK + Custom Persistence |
|--------|-----------|---------------------------------------|
| Checkpointing | Automatic via PostgresSaver | Explicit write_async() calls |
| State Restoration | Built-in time-travel | Manual load_from_persistence() |
| Schema | Auto-generated | Custom schema design |
| Context Injection | Automatic in state | Manual via ContextRetriever |
| Tool Call Logging | Via graph hooks | Custom tool wrappers |

**Benefits of custom approach:**
- More control over what gets persisted
- Can selectively retrieve context (not full checkpoint)
- Schema optimized for our specific use case
- Easier to extend with semantic search

---

## Sources

- [Databricks Stateful Agents](https://docs.databricks.com/aws/en/generative-ai/agent-framework/stateful-agents)
- [Lakebase Notebook Example](https://docs.databricks.com/aws/en/notebooks/source/generative-ai/stateful-agent-lakebase.html)
- [LangGraph PostgresSaver Best Practices](https://sparkco.ai/blog/mastering-langgraph-checkpointing-best-practices-for-2025)
- [Advanced Memory Persistence Strategies](https://sparkco.ai/blog/advanced-memory-persistence-strategies-in-ai-agents)
- [LangChain Persistence Docs](https://docs.langchain.com/oss/python/langgraph/persistence)
